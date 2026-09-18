# Author: Roman Szewczyk
# License: MIT License (see LICENSE file)

"""Supplementary hysteresis-loop series of polycrystalline gadolinium through the Curie point.

Data: data/gd_supplementary_loops_near_Tc.csv - one descending and one ascending
branch (201 points each) of B(H) loops of peak field 6.06 kA/m at -30, -20, -10, 0, 10, 15,
17.5 and 20 C, and one loop of peak field 54.8 kA/m at -30 C measured in the same series as a
consistency check against the main series (the column seria_Hmax_nom_A_per_m is the nominal
setpoint; the actual peak field is taken from the data).

For every loop: coercive field (mean |H| of the two B = 0 crossings, the difference between
them being the field-offset bound), remanence, loss per cycle W = |oint B dH| (the branches
close to < 0.5 mT, no drift correction needed), maximum separation of the two branches, tip
magnetization M_amp = B_max/mu0 - H_max and k_eff = W/(4 mu0 M_amp).

Output: results/json/near_tc_loops.json. The figures are drawn by plot_figures.py.
"""

import os, sys, csv, json, collections
import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from ja_model import MU0

CSV = os.path.join(ROOT, 'data', 'gd_supplementary_loops_near_Tc.csv')
NPZ = os.path.join(ROOT, 'data', 'gd_processed.npz')
COEFFS = os.path.join(ROOT, 'global_fit', 'physical_model_coeffs.json')
OUT_JSON = os.path.join(ROOT, 'results', 'json', 'near_tc_loops.json')


def load_loops():
    loops = collections.OrderedDict()
    for r in csv.DictReader(open(CSV), delimiter=';'):
        key = (float(r['T_C']), float(r['seria_Hmax_nom_A_per_m']))
        loops.setdefault(key, {}).setdefault(r['branch'], []).append(
            (int(r['i']), float(r['H_A_per_m']), float(r['B_T'])))
    out = collections.OrderedDict()
    for key, b in loops.items():
        br = {}
        for name, v in b.items():
            v = sorted(v); br[name] = (np.array([x[1] for x in v]), np.array([x[2] for x in v]))
        out[key] = br
    return out


def crossings(H, B):
    hc = []
    for j in range(1, len(B)):
        if B[j - 1] * B[j] < 0:
            t = abs(B[j - 1]) / (abs(B[j - 1]) + abs(B[j])); hc.append(H[j - 1] + t * (H[j] - H[j - 1]))
    return hc


def at_H0(H, B):
    o = np.argsort(H); return float(np.interp(0.0, H[o], B[o]))


def loop_observables(Hd, Bd, Ha, Ba):
    W = abs(np.trapezoid(Bd, Hd) + np.trapezoid(Ba, Ha))
    hcd, hca = crossings(Hd, Bd), crossings(Ha, Ba)
    Hc_d = hcd[0] if hcd else np.nan; Hc_a = hca[0] if hca else np.nan
    Br_d, Br_a = at_H0(Hd, Bd), at_H0(Ha, Ba)
    sep = float(np.max(np.abs(Bd - np.interp(Hd, Ha, Ba))))
    Bmax = float(max(Bd.max(), Ba.max())); Hmax = float(max(Hd.max(), Ha.max()))
    M_amp = Bmax / MU0 - Hmax
    Hc = float(np.nanmean([abs(Hc_d), abs(Hc_a)])) if sep > 0 else 0.0
    Br = 0.5 * (abs(Br_d) + abs(Br_a)) if sep > 0 else 0.0
    return dict(Hc=Hc, Hc_desc=float(Hc_d), Hc_asc=float(Hc_a), Br=Br, Br_desc=Br_d, Br_asc=Br_a,
                W=float(W), branch_separation_max_T=sep, B_max=Bmax, H_max=Hmax, M_amp=float(M_amp),
                k_eff=float(W / (4 * MU0 * M_amp)) if M_amp > 0 else np.nan,
                closure_T=float(Bd[0] - Ba[-1]))


def power_law(T_K, A, Tc, x):
    return A * np.clip(1.0 - T_K / Tc, 1e-9, None) ** x


def main():
    loops = load_loops()
    coeffs = json.load(open(COEFFS)); Tc_Ms = coeffs['Tc']
    res = {'series_6kA': [], 'check_55kA_minus30': None}
    for (T, nom), br in loops.items():
        ob = loop_observables(*br['desc'], *br['asc']); ob['T_C'] = T; ob['H_nominal'] = nom
        if nom == 10000: res['series_6kA'].append(ob)
        else: res['check_55kA_minus30'] = ob
    res['series_6kA'].sort(key=lambda r: r['T_C'])
    ser = res['series_6kA']
    T = np.array([r['T_C'] for r in ser]); T_K = T + 273.15
    W = np.array([r['W'] for r in ser]); Hc = np.array([r['Hc'] for r in ser])
    Br = np.array([r['Br'] for r in ser]); M = np.array([r['M_amp'] for r in ser])
    keff = np.array([r['k_eff'] for r in ser])
    # power law of W(T) over the hysteretic loops (W > 0), Tc free
    m = W > 1.0      # loops with resolvable hysteresis (the 20 C loop has W ~ 1e-12 J/m3)
    # Tc by a chi-square profile (the three-parameter fit is ill-conditioned this close to Tc):
    # for each Tc on a grid, fit log W = log A + x log(1 - T/Tc) linearly; 1-sigma from delta chi2 = 1
    # with the residual variance of the best fit
    grid = np.arange(T_K[m].max() + 0.1, 300.01, 0.05); chi2 = []; par = []
    for Tc in grid:
        X = np.log(1 - T_K[m] / Tc); Y = np.log(W[m])
        A_ = np.vstack([np.ones_like(X), X]).T; sol, r, _, _ = np.linalg.lstsq(A_, Y, rcond=None)
        chi2.append(float(np.sum((Y - A_ @ sol) ** 2))); par.append(sol)
    chi2 = np.array(chi2); ib = int(np.argmin(chi2)); s2 = chi2[ib] / (m.sum() - 3)
    ok = grid[chi2 <= chi2[ib] + s2]
    res['W_power_law_Tc_free'] = {'A': float(np.exp(par[ib][0])), 'Tc': float(grid[ib]), 'Tc_lo': float(ok.min()),
                                  'Tc_hi': float(ok.max()), 'Tc_err': float(0.5 * (ok.max() - ok.min())),
                                  'x': float(par[ib][1]), 'x_err': float(np.nan), 'method': 'log-linear chi2 profile in Tc'}
    f2 = lambda t, A, x: power_law(t, A, Tc_Ms, x)
    p2, c2 = curve_fit(f2, T_K[m], W[m], p0=[W[0] * 2, 0.5]); e2 = np.sqrt(np.diag(c2))
    res['W_power_law_Tc_Ms'] = {'A': p2[0], 'Tc': Tc_Ms, 'x': p2[1], 'x_err': e2[1]}

    # comparison with the main series at -30 C (55 kA/m loop) and at the common temperatures
    d = np.load(NPZ, allow_pickle=True); Tq = list(d['temperatures'])
    main_obs = json.load(open(os.path.join(ROOT, 'results', 'json', 'loop_observables.json')))
    i = Tq.index(-30.0); Hm = np.asarray(d['major_H'][i], float); Bm = np.asarray(d['major_B'][i], float)
    asc = (np.gradient(Hm) > 0) & (Hm > 0) & (Hm < 20000); o = np.argsort(Hm[asc])
    chk = res['check_55kA_minus30']
    res['series_comparison_minus30_55kA'] = {
        'B_max_main': float(Bm.max()), 'B_max_supplementary': chk['B_max'],
        'B_at_6kA_ascending_main': float(np.interp(chk['H_nominal'] * 0 + 6060, Hm[asc][o], Bm[asc][o])),
        'Hc_main': main_obs['major'][i]['Hc'], 'Hc_supplementary': chk['Hc'],
        'Br_main': main_obs['major'][i]['Br'], 'Br_supplementary': chk['Br'],
        'W_main_10kA': main_obs['mid'][i]['W'], 'W_supplementary_55kA': chk['W']}
    res['common_temperatures'] = []
    for r in ser:
        if r['T_C'] in Tq:
            j = Tq.index(r['T_C']); mm = main_obs['mid'][j]
            res['common_temperatures'].append({'T_C': r['T_C'], 'Hc_main_10kA': mm['Hc'], 'Hc_suppl_6kA': r['Hc'],
                                               'W_main_10kA': mm['W'], 'W_suppl_6kA': r['W'],
                                               'Br_main_10kA': mm['Br'], 'Br_suppl_6kA': r['Br']})
    json.dump(res, open(OUT_JSON, 'w'), indent=1)

    pl = res['W_power_law_Tc_free']
    print('6 kA/m series:  T(C)   Hc    Br     W     M_amp  k_eff  sep(mT)')
    for r in ser:
        print(f"  {r['T_C']:>5}  {r['Hc']:5.0f}  {r['Br']:.4f}  {r['W']:6.1f}  {r['M_amp']/1e3:5.0f}  {r['k_eff']:5.0f}  {r['branch_separation_max_T']*1e3:5.1f}")
    print(f"W power law (Tc profile, hysteretic loops): Tc = {pl['Tc']:.2f} K (1-sigma {pl['Tc_lo']:.2f}-{pl['Tc_hi']:.2f}), x = {pl['x']:.2f}")
    sc = res['series_comparison_minus30_55kA']
    print(f"-30 C, 55 kA/m: B_max main {sc['B_max_main']:.4f} / suppl {sc['B_max_supplementary']:.4f} T; "
          f"Hc main {sc['Hc_main']:.0f} / suppl {sc['Hc_supplementary']:.0f} A/m; Br main {sc['Br_main']:.3f} / suppl {sc['Br_supplementary']:.3f} T")

    print('written', OUT_JSON)


if __name__ == '__main__':
    main()
