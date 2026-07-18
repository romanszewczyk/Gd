# Author: Roman Szewczyk
# License: MIT License (see LICENSE file)

"""Robustness cross-check of the alpha = 0 assumption.

The analysis fixes the inter-domain coupling alpha = 0 throughout. This script
tests that choice with a profile analysis: for every major loop and for each
value of alpha on a fixed grid spanning the literature range (0 to 3e-3), the
remaining four parameters (a, k, c, Ms) are re-optimised locally, starting from
the committed per-loop optimum (parameters_per_temperature.json). The profile
of the misfit versus alpha shows whether a non-zero inter-domain coupling
improves the description, and the drift of the re-fitted Ms versus alpha
quantifies the bias that the alpha = 0 choice could introduce into the
order-parameter scaling. A power law Ms(T_K) = Ms0 (1 - T_K/Tc)^beta is fitted
to the per-temperature Ms of the alpha = 0 and best-alpha columns and compared.

The script reads committed results and the processed loops; it writes a JSON
summary only.
"""

import os
import sys
import json
import time
import numpy as np
from multiprocessing import Pool
from scipy.optimize import minimize, curve_fit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ja_model import MU0, single_loop, calc_metrics
from paths import PROCESSED_NPZ, results_json

SUBSAMPLE = 500                                    # loop points for the re-fits
ALPHA_GRID = [0.0, 1e-4, 3e-4, 1e-3, 3e-3]         # literature range of alpha
MAXFEV = 500                                       # Nelder-Mead budget per fit
N_WORKERS = 12


def _sse(theta, alpha, H, B, M0):
    a, k, c, Ms = theta
    if a <= 0 or k <= 0 or c < 0 or c > 0.95 or Ms <= 0:
        return 1e30
    try:
        _, Bsim = single_loop(a, k, c, Ms, alpha, H, M0=M0)
    except Exception:
        return 1e30
    if len(Bsim) != len(B) or not np.all(np.isfinite(Bsim)):
        return 1e30
    return float(np.sum((B - Bsim) ** 2))


def _refit(job):
    """Local Nelder-Mead re-fit of (a, k, c, Ms) at fixed alpha."""
    i, alpha, theta0, H, B, M0 = job
    t0 = time.time()
    r = minimize(_sse, theta0, args=(alpha, H, B, M0), method='Nelder-Mead',
                 options={'maxfev': MAXFEV, 'xatol': 1e-6, 'fatol': 1e-12})
    a, k, c, Ms = r.x
    _, Bsim = single_loop(a, k, c, Ms, alpha, H, M0=M0)
    m = calc_metrics(B, Bsim, 4 if alpha == 0.0 else 5)
    return (i, alpha, {'a': float(a), 'k': float(k), 'c': float(c),
                       'Ms': float(Ms), 'alpha': float(alpha),
                       'sse': float(m['sse']), 'r2': float(m['r2']),
                       'rmse': float(m['rmse']), 'seconds': time.time() - t0})


def _powerlaw(T_K, Ms0, beta, Tc):
    return Ms0 * np.clip(1.0 - T_K / Tc, 1e-9, None) ** beta


def fit_powerlaw(T_K, Ms):
    popt, pcov = curve_fit(_powerlaw, T_K, Ms, p0=[2.1e6, 0.30, 291.5],
                           maxfev=20000)
    err = np.sqrt(np.diag(pcov))
    return {'Ms0': float(popt[0]), 'beta': float(popt[1]),
            'Tc_K': float(popt[2]), 'beta_err': float(err[1])}


def main():
    data = np.load(PROCESSED_NPZ, allow_pickle=True)
    T = np.array(data['temperatures'], dtype=float)
    mH, mB = data['major_H'], data['major_B']
    per_T = json.load(open(results_json('parameters_per_temperature.json')))

    jobs = []
    for i in range(len(T)):
        H = np.asarray(mH[i], dtype=float).ravel()
        B = np.asarray(mB[i], dtype=float).ravel()
        M0 = B[0] / MU0 - H[0]
        if len(H) > SUBSAMPLE:
            idx = np.arange(0, len(H), len(H) // SUBSAMPLE)
            if idx[-1] != len(H) - 1:
                idx = np.append(idx, len(H) - 1)
            H, B = H[idx], B[idx]
        p = per_T[i]['params']
        theta0 = np.array([p['a'], p['k'], p['c'], p['Ms']])
        for alpha in ALPHA_GRID:
            jobs.append((i, alpha, theta0, H, B, M0))

    with Pool(N_WORKERS) as pool:
        results = pool.map(_refit, jobs)

    table = {}
    for i, alpha, fit in results:
        table.setdefault(i, {})[alpha] = fit

    rows = []
    print(f"{'T [C]':>6} {'best alpha':>10} {'R2(a=0)':>9} {'R2(best)':>9} "
          f"{'dMs(best) [%]':>13} {'dMs(3e-3) [%]':>13}")
    for i in range(len(T)):
        prof = table[i]
        f0 = prof[0.0]
        best_alpha = min(prof, key=lambda al: prof[al]['sse'])
        fb = prof[best_alpha]
        dms_best = 100.0 * (fb['Ms'] - f0['Ms']) / f0['Ms']
        dms_max = 100.0 * (prof[ALPHA_GRID[-1]]['Ms'] - f0['Ms']) / f0['Ms']
        print(f"{T[i]:>6.0f} {best_alpha:>10.1e} {f0['r2']:>9.5f} "
              f"{fb['r2']:>9.5f} {dms_best:>13.3f} {dms_max:>13.3f}", flush=True)
        rows.append({'temperature': float(T[i]),
                     'profile': {f'{al:g}': prof[al] for al in ALPHA_GRID},
                     'best_alpha': float(best_alpha),
                     'dMs_best_percent': float(dms_best),
                     'dMs_alpha_max_percent': float(dms_max)})

    T_K = T + 273.15
    ms0col = np.array([table[i][0.0]['Ms'] for i in range(len(T))])
    msbcol = np.array([table[i][rows[i]['best_alpha']]['Ms']
                       for i in range(len(T))])
    pl0 = fit_powerlaw(T_K, ms0col)
    plb = fit_powerlaw(T_K, msbcol)
    print("\nPower law, alpha = 0 column:   "
          f"Ms0={pl0['Ms0']/1e6:.3f} MA/m, beta={pl0['beta']:.3f}, "
          f"Tc={pl0['Tc_K']:.2f} K")
    print("Power law, best-alpha column:  "
          f"Ms0={plb['Ms0']/1e6:.3f} MA/m, beta={plb['beta']:.3f}, "
          f"Tc={plb['Tc_K']:.2f} K")

    out = {'description': ('Profile of the per-loop re-fit versus fixed alpha: '
                           '(a, k, c, Ms) re-optimised locally from the '
                           'committed per-loop optimum at each alpha, on '
                           'identical subsampled loops.'),
           'subsample': SUBSAMPLE, 'alpha_grid': ALPHA_GRID,
           'maxfev': MAXFEV, 'loops': rows,
           'powerlaw_alpha0': pl0, 'powerlaw_best_alpha': plb}
    path = results_json('alpha_crosscheck.json')
    json.dump(out, open(path, 'w'), indent=2)
    print(f"\nWrote {path}")


if __name__ == '__main__':
    main()
