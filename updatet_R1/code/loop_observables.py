# Author: Roman Szewczyk
# License: MIT License (see LICENSE file)

"""Model-free loop observables of the main hysteresis-loop series of polycrystalline gadolinium.

For every temperature and every measured amplitude (55, 10 and 0.4 kA/m) one full cycle is
isolated from the record (the records start with the initial magnetization curve) and the
coercive field Hc, the remanence Br, the loss per cycle W = |oint B dH|, the closure error
B_start - B_end and the tip magnetization M_tip = B_max/mu0 - H_max are evaluated.

Drift. The 55 kA/m records do not close: the fluxmeter integrator drifts by up to 0.02 T
within one cycle, which adds a spurious area of order (closure) x H_max, comparable to the
loop area itself; the raw and the linearly detrended areas differ by up to a factor of four.
The 10 kA/m records close to better than 0.004 T and give the same area with and without
detrending. The loss per cycle and the quantities derived from it are therefore taken from
the 10 kA/m loops, whose coercive field is the same as that of the 55 kA/m loops (both are
major loops as far as the irreversible process is concerned). The 55 kA/m loops provide the
coercive field, the remanence (crossing quantities, affected by the drift only at the level of
the branch asymmetry) and M_tip. The loss-averaged pinning field k_eff = W / (4 mu0 M_amp),
with M_amp the magnetization at the loop tip of the same loop, is the mean field against
which the magnetization is dragged over a cycle. The same quantities are evaluated for the
global-model 55 kA/m loops (B_sim of global_fit/parameters_physical.json).

Output: results/json/loop_observables.json (all observables, measured and modelled, plus
power laws). The figures are drawn by plot_figures.py.
"""

import os, sys, json
import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from ja_model import MU0

NPZ = os.path.join(ROOT, 'data', 'gd_processed.npz')
GLOBAL = os.path.join(ROOT, 'global_fit', 'parameters_physical.json')
COEFFS = os.path.join(ROOT, 'global_fit', 'physical_model_coeffs.json')
OUT_JSON = os.path.join(ROOT, 'results', 'json', 'loop_observables.json')


def full_cycle(H):
    """Index range [i0, i1] of one full cycle: from the first field maximum to the next."""
    hi = np.where(H >= 0.985 * H.max())[0]
    i0 = hi[0]
    gaps = np.where(np.diff(hi) > 500)[0]
    i1 = hi[gaps[0] + 1] if len(gaps) else len(H) - 1
    return i0, i1


def observables(H, B):
    """Hc (mean |H| at B = 0), Br (mean |B| at H = 0), loss per cycle W = |oint B dH| (J/m3), closure error."""
    hc, br = [], []
    for j in range(1, len(B)):
        if B[j - 1] * B[j] < 0:
            t = abs(B[j - 1]) / (abs(B[j - 1]) + abs(B[j]))
            hc.append(abs(H[j - 1] + t * (H[j] - H[j - 1])))
        if H[j - 1] * H[j] < 0:
            t = abs(H[j - 1]) / (abs(H[j - 1]) + abs(H[j]))
            br.append(abs(B[j - 1] + t * (B[j] - B[j - 1])))
    # enclosed area = path integral of B dH around the cycle (the cycle starts and ends at the
    # same field, so the path is closed in H; the open-polygon shoelace sum is origin-dependent
    # and must not be used)
    W = abs(np.trapezoid(B, H))
    closure = float(B[0] - B[-1])
    Bdet = B + closure * np.arange(len(B)) / (len(B) - 1)     # linear detrend: cycle closes
    W_det = abs(np.trapezoid(Bdet, H))
    return (float(np.mean(hc)) if hc else np.nan, float(np.mean(br)) if br else np.nan,
            float(W), closure, float(W_det))


def power_law(T_K, A, Tc, x):
    return A * np.clip(1.0 - T_K / Tc, 1e-9, None) ** x


def fit_power(T_K, y, Tc_fixed=None):
    if Tc_fixed is None:
        p, c = curve_fit(power_law, T_K, y, p0=[y[0] * 2, 293.0, 0.5], maxfev=20000)
        e = np.sqrt(np.diag(c))
        return {'A': p[0], 'Tc': p[1], 'x': p[2], 'A_err': e[0], 'Tc_err': e[1], 'x_err': e[2]}
    f = lambda t, A, x: power_law(t, A, Tc_fixed, x)
    p, c = curve_fit(f, T_K, y, p0=[y[0] * 2, 0.5], maxfev=20000)
    e = np.sqrt(np.diag(c))
    return {'A': p[0], 'Tc': Tc_fixed, 'x': p[1], 'A_err': e[0], 'x_err': e[1]}


def main():
    d = np.load(NPZ, allow_pickle=True)
    T = np.array(d['temperatures'], dtype=float)
    T_K = T + 273.15
    glob = json.load(open(GLOBAL))
    coeffs = json.load(open(COEFFS))
    res = {'temperature_C': T.tolist()}

    for key in ('major', 'mid', 'minor'):
        rows = []
        for i in range(len(T)):
            H = np.asarray(d[key + '_H'][i], float)
            B = np.asarray(d[key + '_B'][i], float)
            i0, i1 = full_cycle(H)
            Hc, Br, W, closure, W_det = observables(H[i0:i1 + 1], B[i0:i1 + 1])
            Mtip = float(B.max() / MU0 - H.max())
            rows.append({'T_C': float(T[i]), 'Hc': Hc, 'Br': Br, 'W': W, 'W_detrended': W_det,
                         'M_tip': Mtip, 'k_eff': W / (4 * MU0 * Mtip), 'closure_T': closure,
                         'H_max': float(H.max())})
        res[key] = rows

    # global-model loops (major only): same observables from B_sim
    rows = []
    for i in range(len(T)):
        H = np.asarray(d['major_H'][i], float)
        Bs = np.asarray(glob[i]['B_sim'], float)
        i0, i1 = full_cycle(H)
        Hc, Br, W, _, _ = observables(H[i0:i1 + 1], Bs[i0:i1 + 1])
        rows.append({'T_C': float(T[i]), 'Hc': Hc, 'Br': Br, 'W': W,
                     'k_JA': glob[i]['params']['k'], 'c_JA': glob[i]['params']['c']})
    res['global_model_major'] = rows

    maj = res['major']; mid = res['mid']
    Hc = np.array([r['Hc'] for r in maj]); Br = np.array([r['Br'] for r in maj])
    Mtip = np.array([r['M_tip'] for r in maj])
    W10 = np.array([r['W'] for r in mid]); Hc10 = np.array([r['Hc'] for r in mid])
    Br10 = np.array([r['Br'] for r in mid]); M10 = np.array([r['M_tip'] for r in mid])
    keff10 = np.array([r['k_eff'] for r in mid])
    Tc_Ms = coeffs['Tc']
    res['power_laws'] = {
        'W10_free_Tc': fit_power(T_K, W10), 'W10_fixed_Tc': fit_power(T_K, W10, Tc_Ms),
        'Br_fixed_Tc': fit_power(T_K, Br, Tc_Ms), 'Br10_fixed_Tc': fit_power(T_K, Br10, Tc_Ms),
        'M_tip_free_Tc': fit_power(T_K, Mtip), 'M_tip_fixed_293': fit_power(T_K, Mtip, 293.0),
        'M10_free_Tc': fit_power(T_K, M10)}
    # Tc of the 10 kA/m loss by a chi-square profile (log-linear fit for each Tc on a grid)
    grid = np.arange(T_K.max() + 0.5, 300.01, 0.05); chi2 = []; par = []
    for Tc in grid:
        X = np.log(1 - T_K / Tc); Y = np.log(W10)
        A_ = np.vstack([np.ones_like(X), X]).T; sol = np.linalg.lstsq(A_, Y, rcond=None)[0]
        chi2.append(float(np.sum((Y - A_ @ sol) ** 2))); par.append(sol)
    chi2 = np.array(chi2); ib = int(np.argmin(chi2)); s2 = chi2[ib] / (len(T_K) - 3)
    ok = grid[chi2 <= chi2[ib] + s2]
    res['power_laws']['W10_Tc_profile'] = {'Tc': float(grid[ib]), 'Tc_lo': float(ok.min()), 'Tc_hi': float(ok.max()),
                                           'x': float(par[ib][1]), 'method': 'log-linear chi2 profile in Tc'}
    res['ratios_cold_to_warm'] = {'W10': W10[0] / W10[-1], 'Br55': Br[0] / Br[-1], 'Br10': Br10[0] / Br10[-1],
                                  'M_tip55': Mtip[0] / Mtip[-1], 'M10': M10[0] / M10[-1],
                                  'k_eff10': keff10[0] / keff10[-1], 'Hc55': Hc[0] / Hc[-1], 'Hc10': Hc10[0] / Hc10[-1]}
    res['drift_note'] = ('55 kA/m records: closure up to %.3f T, raw vs detrended area differ by up to a factor %.1f; '
                         '10 kA/m records: closure <= %.4f T, raw/detrended within %.0f %%' % (
                         max(abs(r['closure_T']) for r in maj),
                         max(max(r['W'], r['W_detrended']) / max(min(r['W'], r['W_detrended']), 1e-9) for r in maj),
                         max(abs(r['closure_T']) for r in mid),
                         100 * max(abs(r['W'] - r['W_detrended']) / r['W'] for r in mid)))
    json.dump(res, open(OUT_JSON, 'w'), indent=1)

    pw = res['power_laws']
    print(res['drift_note'])
    print(f"W (10 kA/m loops): {W10[0]:.0f} -> {W10[-1]:.0f} J/m3 ({W10[0]/W10[-1]:.2f}x), Tc(W) = "
          f"{pw['W10_free_Tc']['Tc']:.1f} +/- {pw['W10_free_Tc']['Tc_err']:.1f} K, exponent {pw['W10_free_Tc']['x']:.2f} "
          f"+/- {pw['W10_free_Tc']['x_err']:.2f}; with Tc fixed: exponent {pw['W10_fixed_Tc']['x']:.2f}")
    print(f"Br (55 kA/m): {Br[0]:.3f} -> {Br[-1]:.3f} T ({Br[0]/Br[-1]:.2f}x); Br (10 kA/m): {Br10[0]:.3f} -> {Br10[-1]:.3f} T ({Br10[0]/Br10[-1]:.2f}x)")
    print(f"Hc (55 kA/m): {Hc.min():.0f}-{Hc.max():.0f} A/m, Hc(+15)/Hc(-38) = {Hc[-1]/Hc[0]:.2f}; Hc (10 kA/m): {Hc10.min():.0f}-{Hc10.max():.0f} A/m")
    print(f"M(10 kA/m): {M10[0]/1e3:.0f} -> {M10[-1]/1e3:.0f} kA/m ({M10[0]/M10[-1]:.2f}x); k_eff (10 kA/m): {keff10.min():.0f}-{keff10.max():.0f} A/m, "
          f"{keff10[0]:.0f} -> {keff10[-1]:.0f} ({keff10[0]/keff10[-1]:.2f}x)")
    print(f"M_tip (55 kA/m) law: beta = {pw['M_tip_free_Tc']['x']:.3f} +/- {pw['M_tip_free_Tc']['x_err']:.3f}, "
          f"Tc = {pw['M_tip_free_Tc']['Tc']:.1f} +/- {pw['M_tip_free_Tc']['Tc_err']:.1f} K")

    print('written', OUT_JSON)


if __name__ == '__main__':
    main()
