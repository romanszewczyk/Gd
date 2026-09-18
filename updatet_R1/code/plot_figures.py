# Author: Roman Szewczyk
# License: MIT License (see LICENSE file)

"""Figures of the model-free loop observables (each quantity plotted once; square panels
without titles):

    pinning_global_fit.jpg          k(T) and c(T) of the global Jiles-Atherton fit, with the
                                    coercive field of the global-model loops
    coercive_field_keff.jpg         measured coercive field (55 and 10 kA/m loops of the main
                                    series, 6 kA/m loops of the supplementary series) and
                                    loss-averaged pinning field k_eff = W/(4 mu0 M_amp)
    loss_per_cycle.jpg              loss per cycle W of the same loops with power laws
                                    A(1 - T/Tc)^x
    remanence.jpg                   remanence Br
    loops_through_Tc.jpg            supplementary 6 kA/m loops at 15, 17.5 and 20 C
    magnetization_branch_sep.jpg    magnetization at the loop tip and maximum branch
                                    separation of the supplementary loops versus temperature

Reads results/json/loop_observables.json, results/json/near_tc_loops.json, the supplementary
CSV and the global-fit laws. Outputs go to results/figures/.
"""

import os, csv, json, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OBS = os.path.join(ROOT, 'results', 'json', 'loop_observables.json')
NTC = os.path.join(ROOT, 'results', 'json', 'near_tc_loops.json')
CSV = os.path.join(ROOT, 'data', 'gd_supplementary_loops_near_Tc.csv')
COEFFS = os.path.join(ROOT, 'global_fit', 'physical_model_coeffs.json')
FIGDIR = os.path.join(ROOT, 'results', 'figures')
SIZE = 4.6          # inches, square sub-figure
DPI = 150

plt.rcParams.update({'font.size': 10, 'legend.fontsize': 8.5, 'axes.labelsize': 11})


def power_law(T_K, A, Tc, x):
    return A * np.clip(1.0 - T_K / Tc, 1e-9, None) ** x


def law_curve(T_K, W, Tc):
    """A and x of the log-linear fit at fixed Tc (as used in the chi-square profile)."""
    X = np.log(1 - T_K / Tc); A_ = np.vstack([np.ones_like(X), X]).T
    sol = np.linalg.lstsq(A_, np.log(W), rcond=None)[0]
    return np.exp(sol[0]), sol[1]


def square():
    fig, ax = plt.subplots(figsize=(SIZE, SIZE))
    ax.grid(alpha=0.3)
    return fig, ax


def save(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.tight_layout(); fig.savefig(path, dpi=DPI); plt.close(fig)
    return path


def main():
    obs = json.load(open(OBS)); ntc = json.load(open(NTC)); coeffs = json.load(open(COEFFS))
    Tc_C = coeffs['Tc'] - 273.15
    maj, mid, gm = obs['major'], obs['mid'], obs['global_model_major']
    T = np.array(obs['temperature_C']); T_K = T + 273.15
    sup = ntc['series_6kA']; Ts = np.array([r['T_C'] for r in sup]); Ts_K = Ts + 273.15
    hyst = np.array([r['W'] > 1.0 for r in sup])
    xlab = r'$T$ [$^\circ$C]'
    paths5, paths6 = [], []

    # ---- 5a: global-fit k(T), c(T) and modelled coercive field
    fig, a = square()
    a.plot(T, [r['k_JA'] for r in gm], 'r-', lw=1.8, label=r'$k(T)$, global fit')
    a.plot(T, [r['Hc'] for r in gm], 'r:', lw=1.4, label=r'$H_c$ of the global-model loops')
    a.set_xlabel(xlab); a.set_ylabel(r'$k$, $H_c$ [A/m]'); a.set_ylim(0, 500); a.set_xlim(-40, 22)
    a2 = a.twinx(); a2.plot(T, [r['c_JA'] for r in gm], 'b--', lw=1.6, label=r'$c(T)$, global fit')
    a2.set_ylabel(r'$c$ [-]'); a2.set_ylim(0, 0.5)
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, loc='upper center')
    paths5.append(save(fig, 'pinning_global_fit.jpg'))

    # ---- 5b: measured coercive field and loss-averaged pinning field
    fig, b = square()
    b.plot(T, [r['Hc'] for r in maj], 'bo-', ms=4, label=r'$H_c$, 55 kA/m (main)')
    b.plot(T, [r['Hc'] for r in mid], 'bs--', ms=4, mfc='none', label=r'$H_c$, 10 kA/m (main)')
    b.plot(Ts, [r['Hc'] for r in sup], 'cD-.', ms=4, mfc='none', label=r'$H_c$, 6 kA/m (supplementary)')
    b.plot(T, [r['k_eff'] for r in mid], 'g^-', ms=4, label=r'$k_{\rm eff}$, 10 kA/m (main)')
    b.plot(Ts, [r['k_eff'] for r in sup], 'gv-.', ms=4, mfc='none', label=r'$k_{\rm eff}$, 6 kA/m (supplementary)')
    b.axvline(Tc_C, color='grey', lw=0.8, ls=':')
    b.set_xlabel(xlab); b.set_ylabel(r'$H_c$, $k_{\rm eff}$ [A/m]'); b.set_ylim(0, 1200); b.set_xlim(-40, 22)
    b.legend(loc='upper left', ncol=1)
    paths5.append(save(fig, 'coercive_field_keff.jpg'))

    # ---- 5c: loss per cycle with power laws
    fig, c = square()
    W10 = np.array([r['W'] for r in mid]); pr = obs['power_laws']['W10_Tc_profile']
    A1, x1 = law_curve(T_K, W10, pr['Tc']); Tg = np.linspace(-38, pr['Tc'] - 273.15, 300)
    c.plot(T, W10, 'ko-', ms=4, label=r'$W$, 10 kA/m (main)')
    c.plot(Tg, power_law(Tg + 273.15, A1, pr['Tc'], x1), 'k--', lw=1, label=r'power law, $T_c$ = %.1f K' % pr['Tc'])
    Ws = np.array([r['W'] for r in sup]); ps = ntc['W_power_law_Tc_free']
    A2, x2 = law_curve(Ts_K[hyst], Ws[hyst], ps['Tc']); Tg2 = np.linspace(-30, ps['Tc'] - 273.15, 300)
    c.plot(Ts, Ws, 'kD-.', ms=4, mfc='none', label=r'$W$, 6 kA/m (supplementary)')
    c.plot(Tg2, power_law(Tg2 + 273.15, A2, ps['Tc'], x2), 'k:', lw=1, label=r'power law, $T_c$ = %.1f K' % ps['Tc'])
    c.axvline(Tc_C, color='grey', lw=0.8, ls=':')
    c.set_xlabel(xlab); c.set_ylabel(r'$W$ [J/m$^3$]'); c.set_ylim(0, 1300); c.set_xlim(-40, 22)
    c.legend(loc='upper right')
    paths5.append(save(fig, 'loss_per_cycle.jpg'))

    # ---- 5d: remanence
    fig, d = square()
    d.plot(T, np.array([r['Br'] for r in maj]) * 1e3, 'mo-', ms=4, label=r'$B_r$, 55 kA/m (main)')
    d.plot(T, np.array([r['Br'] for r in mid]) * 1e3, 'ms--', ms=4, mfc='none', label=r'$B_r$, 10 kA/m (main)')
    d.plot(Ts, np.array([r['Br'] for r in sup]) * 1e3, 'mD-.', ms=4, mfc='none', label=r'$B_r$, 6 kA/m (supplementary)')
    d.axvline(Tc_C, color='grey', lw=0.8, ls=':')
    d.set_xlabel(xlab); d.set_ylabel(r'$B_r$ [mT]'); d.set_ylim(0, 220); d.set_xlim(-40, 22)
    d.legend(loc='upper right')
    paths5.append(save(fig, 'remanence.jpg'))

    # ---- 6a: supplementary loops through Tc
    loops = collections.OrderedDict()
    for r in csv.DictReader(open(CSV), delimiter=';'):
        key = (float(r['T_C']), float(r['seria_Hmax_nom_A_per_m']))
        loops.setdefault(key, {}).setdefault(r['branch'], []).append((int(r['i']), float(r['H_A_per_m']), float(r['B_T'])))
    fig, a = square()
    for Tsel, col in [(15.0, 'b'), (17.5, 'g'), (20.0, 'r')]:
        br = loops[(Tsel, 10000.0)]
        H = np.r_[[x[1] for x in sorted(br['desc'])], [x[1] for x in sorted(br['asc'])]]
        B = np.r_[[x[2] for x in sorted(br['desc'])], [x[2] for x in sorted(br['asc'])]]
        a.plot(H / 1e3, B * 1e3, col + '-', lw=1.1, label=r'%g $^\circ$C' % Tsel)
    a.set_xlabel(r'$H$ [kA/m]'); a.set_ylabel(r'$B$ [mT]'); a.legend(loc='upper left')
    paths6.append(save(fig, 'loops_through_Tc.jpg'))

    # ---- 6b: magnetization at 6 kA/m and branch separation
    fig, b = square()
    b.plot(Ts, np.array([r['M_amp'] for r in sup]) / 1e3, 'rs-', ms=4, label=r'$M_{\rm amp}$ at 6 kA/m')
    b.set_xlabel(xlab); b.set_ylabel(r'$M_{\rm amp}$ [kA/m]'); b.set_ylim(0, 600); b.set_xlim(-32, 22)
    b2 = b.twinx(); b2.plot(Ts, np.array([r['branch_separation_max_T'] for r in sup]) * 1e3, 'k^--', ms=4, label='max. branch separation')
    b2.set_ylabel('branch separation [mT]'); b2.set_ylim(0, 300)
    b.axvline(Tc_C, color='grey', lw=0.8, ls=':')
    h1, l1 = b.get_legend_handles_labels(); h2, l2 = b2.get_legend_handles_labels()
    b.legend(h1 + h2, l1 + l2, loc='lower left')
    paths6.append(save(fig, 'magnetization_branch_sep.jpg'))

    for p in paths5 + paths6: print('written', p)


if __name__ == '__main__':
    main()
