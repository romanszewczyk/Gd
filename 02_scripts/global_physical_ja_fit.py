# Author: Roman Szewczyk
# License: MIT License (see LICENSE file)

"""Global physical Jiles-Atherton fit (Stage B).

All twelve major loops are fitted simultaneously with the temperature
dependence of every parameter expressed as a smooth physical law, so the
parameter-versus-temperature trends are smooth by construction rather than
smoothed afterwards:

    Ms(T_K) = Ms0 * (1 - T_K/Tc)**beta          critical power law
    a(T), k(T), c(T)                            quadratics in T [deg C],
                                                each set by its value at three
                                                anchor temperatures

The objective is the mean over temperatures of (1 - R2_i) = SSE_i/SST_i, which
weights every loop equally. A plain sum of squared errors would be dominated
by the high-Ms low-temperature loops and would under-fit the loops near the
Curie point. alpha = 0, scalar k.
"""

import os
import sys
import json
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ja_model import MU0, single_loop, calc_metrics
from paths import PROCESSED_NPZ, results_json

ALPHA = 0.0
ANCHORS_T = np.array([-38.0, -11.5, 15.0])   # anchor temperatures for a, k, c
SUBSAMPLE = 500                               # loop points used during the search

# x = [Ms0, beta, Tc, a_lo, a_mid, a_hi, k_lo, k_mid, k_hi, c_lo, c_mid, c_hi]
BOUNDS = [
    (1.8e6, 2.8e6),                                          # Ms0
    (0.25, 0.50),                                            # beta
    (291.0, 295.0),                                          # Tc [K]
    (2000.0, 6000.0), (3000.0, 8000.0), (4000.0, 11000.0),  # a anchors
    (30.0, 1500.0), (30.0, 1800.0), (30.0, 2500.0),         # k anchors
    (0.00, 0.45), (0.00, 0.40), (0.00, 0.50),               # c anchors
]


def laws(x, T_C, T_K):
    """Per-temperature a, k, c, Ms and the polynomial coefficients from x."""
    Ms0, beta, Tc = x[0], x[1], x[2]
    a_c = np.polyfit(ANCHORS_T, x[3:6], 2)
    k_c = np.polyfit(ANCHORS_T, x[6:9], 2)
    c_c = np.polyfit(ANCHORS_T, x[9:12], 2)
    ms = Ms0 * np.clip(1.0 - T_K / Tc, 1e-9, None) ** beta
    return (np.polyval(a_c, T_C), np.polyval(k_c, T_C),
            np.polyval(c_c, T_C), ms, (a_c, k_c, c_c))


class _Objective:
    """Mean normalised SSE over all loops; picklable for DE workers."""

    def __init__(self, loops, T_C, T_K):
        self.loops, self.T_C, self.T_K = loops, T_C, T_K

    def __call__(self, x):
        a, k, c, ms, _ = laws(x, self.T_C, self.T_K)
        pen = 0.0
        for i in range(len(self.loops)):
            if a[i] <= 0 or k[i] <= 0 or ms[i] <= 0:
                pen += 1.0
            if c[i] < 0:
                pen += 1.0 - c[i]
            if c[i] > 0.95:
                pen += 1.0 + c[i]
        if pen > 0:
            return 10.0 + pen
        total = 0.0
        for i, (H, B, M0, sst) in enumerate(self.loops):
            try:
                _, Bsim = single_loop(a[i], k[i], c[i], ms[i], ALPHA, H, M0=M0)
            except Exception:
                return 1e6
            if len(Bsim) != len(B) or not np.all(np.isfinite(Bsim)):
                return 1e6
            total += np.sum((B - Bsim) ** 2) / sst
        return total / len(self.loops)


def build_loops(mH, mB, n, subsample=None):
    loops = []
    for i in range(n):
        H = np.asarray(mH[i], dtype=float).ravel()
        B = np.asarray(mB[i], dtype=float).ravel()
        M0 = B[0] / MU0 - H[0]
        if subsample and len(H) > subsample:
            idx = np.arange(0, len(H), len(H) // subsample)
            if idx[-1] != len(H) - 1:
                idx = np.append(idx, len(H) - 1)
            H, B = H[idx], B[idx]
        loops.append((H, B, M0, float(np.sum((B - B.mean()) ** 2))))
    return loops


def initial_guess(T_C):
    """Physical x0 seeded from the per-temperature fit when available."""
    Tc0 = 293.4
    f = results_json('parameters_per_temperature.json')
    if os.path.exists(f):
        r = [e for e in json.load(open(f)) if e is not None]
        if len(r) == len(T_C):
            tC = np.array([e['temperature'] for e in r])
            tK = tC + 273.15
            a = np.array([e['params']['a'] for e in r])
            k = np.array([e['params']['k'] for e in r])
            c = np.array([e['params']['c'] for e in r])
            ms = np.array([e['params']['Ms'] for e in r])
            tau = np.clip(1.0 - tK / Tc0, 1e-6, None)
            sol = np.linalg.lstsq(np.vstack([np.ones_like(tau), np.log(tau)]).T,
                                  np.log(ms), rcond=None)[0]
            Ms0, beta = float(np.exp(sol[0])), float(np.clip(sol[1], 0.25, 0.50))
            return np.array([Ms0, beta, Tc0,
                             *np.polyval(np.polyfit(tC, a, 2), ANCHORS_T),
                             *np.polyval(np.polyfit(tC, k, 2), ANCHORS_T),
                             *np.clip(np.polyval(np.polyfit(tC, c, 2), ANCHORS_T),
                                      0.02, 0.4)])
    return np.array([2.14e6, 0.30, Tc0, 3500, 6000, 8600,
                     270, 120, 70, 0.30, 0.13, 0.29])


def main():
    data = np.load(PROCESSED_NPZ, allow_pickle=True)
    T_C = np.array(data['temperatures'], dtype=float)
    T_K = T_C + 273.15
    mH, mB = data['major_H'], data['major_B']
    n = len(T_C)

    loops_sub = build_loops(mH, mB, n, subsample=SUBSAMPLE)
    loops_full = build_loops(mH, mB, n)
    obj_sub = _Objective(loops_sub, T_C, T_K)
    obj_full = _Objective(loops_full, T_C, T_K)

    x0 = np.clip(initial_guess(T_C), [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])
    print(f"initial mean(1-R2) = {obj_full(x0):.5f}", flush=True)

    t0 = time.time()
    de = differential_evolution(obj_sub, BOUNDS, x0=x0, popsize=12, maxiter=50,
                                tol=1e-8, mutation=(0.4, 1.0), recombination=0.7,
                                seed=42, polish=False, workers=-1,
                                updating='deferred')
    best_x, best = de.x, obj_sub(de.x)
    res = minimize(obj_sub, best_x, method='Nelder-Mead', bounds=BOUNDS,
                   options={'maxiter': 250, 'xatol': 1e-6, 'fatol': 1e-9})
    if res.fun < best:
        best_x = res.x
    print(f"fit done, mean(1-R2) = {obj_sub(best_x):.5f}  ({time.time() - t0:.0f}s)",
          flush=True)

    a, k, c, ms, (a_c, k_c, c_c) = laws(best_x, T_C, T_K)
    Ms0, beta, Tc = best_x[0], best_x[1], best_x[2]
    print(f"\nMs(T_K) = {Ms0/1e6:.4f}e6 * (1 - T_K/{Tc:.2f})**{beta:.4f}")
    print(f"a(T) = {a_c[0]:.5f} T^2 + {a_c[1]:.4f} T + {a_c[2]:.2f}")
    print(f"k(T) = {k_c[0]:.5f} T^2 + {k_c[1]:.4f} T + {k_c[2]:.2f}")
    print(f"c(T) = {c_c[0]:.6f} T^2 + {c_c[1]:.5f} T + {c_c[2]:.5f}\n")

    results, r2 = [], []
    print(f"{'T':>5} {'Ms/1e6':>8} {'a':>8} {'k':>7} {'c':>7} {'R2adj':>9} {'RMSE':>8}")
    for i in range(n):
        H, B, M0, _ = loops_full[i]
        _, Bsim = single_loop(a[i], k[i], c[i], ms[i], ALPHA, H, M0=M0)
        m = calc_metrics(B, Bsim, 3)
        r2.append(m['r2_adj'])
        print(f"{T_C[i]:>5.0f} {ms[i]/1e6:>8.3f} {a[i]:>8.0f} {k[i]:>7.0f} "
              f"{c[i]:>7.4f} {m['r2_adj']:>9.5f} {m['rmse']:>8.5f}")
        results.append({'temperature': float(T_C[i]), 'M0': float(M0),
                        'params': {'a': float(a[i]), 'k': float(k[i]),
                                   'c': float(c[i]), 'Ms': float(ms[i]),
                                   'alpha': ALPHA},
                        'metrics': {kk: float(vv) for kk, vv in m.items()},
                        'B_sim': Bsim.tolist()})

    r2 = np.array(r2)
    mid = (T_C >= -15) & (T_C <= 10)
    print(f"\nmean R2adj = {r2.mean():.5f} +/- {r2.std():.5f}  "
          f"(>0.99: {int((r2 > 0.99).sum())}/{n}, -15..+10 mean: {r2[mid].mean():.5f})")

    json.dump(results, open(results_json('parameters_physical.json'), 'w'), indent=2)
    json.dump({'Ms0': float(Ms0), 'Tc': float(Tc), 'beta': float(beta),
               'a_coeffs': [float(v) for v in a_c],
               'k_coeffs': [float(v) for v in k_c],
               'c_coeffs': [float(v) for v in c_c],
               'alpha': ALPHA, 'anchors_T': [float(v) for v in ANCHORS_T]},
              open(results_json('physical_model_coeffs.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
