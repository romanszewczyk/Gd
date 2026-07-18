# Author: Roman Szewczyk
# License: MIT License (see LICENSE file)

"""Combine the two uncertainty components and write the summary tables.

The combined standard uncertainty of each reported order-parameter quantity is
the quadrature sum of the measurement-system (calibration, Component A) and
identifiability (Component B) contributions,
    u_c = sqrt( u_cal^2 + u_idn^2 ) .

Only the order-parameter scalars (Ms0, Tc, beta) are tabulated as a budget: they
are the quantities for which a marginal standard uncertainty is meaningful. The
temperature laws a(T), k(T), c(T) are determined by the well-conditioned global
fit but are mutually *dependent*; that dependency is reported as a correlation
matrix rather than as misleading marginal uncertainties.

Outputs (in results/):
    json/uncertainty_summary.json      machine-readable summary
    tables/uncertainty_summary.csv     order-parameter quantities, both components
    tables/uncertainty_budget.tex      LaTeX budget table (Ms0, Tc, beta)
    tables/parameter_correlations.tex  LaTeX correlation matrix
"""

import csv
import json
import numpy as np

import uncertainty_model as um
from correlation_analysis import PARAMS


def main():
    cal = json.load(open(um.rj('calibration_uncertainty.json')))
    idn = json.load(open(um.rj('identifiability_uncertainty.json')))
    corr = json.load(open(um.rj('parameter_correlations.json')))
    coeffs = json.load(open(um.GLOBAL_COEFFS))
    cp = cal['parameters']

    rows = []   # (symbol, unit, value, u_cal, u_idn, u_comb)

    def add(sym, unit, value, u_cal, u_idn):
        rows.append((sym, unit, value, u_cal, u_idn, float(np.hypot(u_cal, u_idn))))

    add('Ms0', 'MA/m', coeffs['Ms0'] / 1e6, cp['Ms0']['u_cal'] / 1e6, idn['Ms0']['u'] / 1e6)
    add('Tc', 'K', coeffs['Tc'], cp['Tc']['u_cal'], idn['Tc']['u'])
    add('beta', '-', coeffs['beta'], cp['beta']['u_cal'], idn['beta']['u'])

    print(f"{'quantity':>10} {'value':>10} {'u_cal':>11} {'u_idn':>11} {'u_comb':>11} {'rel':>7}")
    summary = {'method': 'u_c = sqrt(u_cal^2 + u_idn^2)',
               'k_trend_decreasing_fraction': idn['k_trend_decreasing_fraction'],
               'correlation_matrix': corr['correlation_matrix'],
               'correlation_params': list(PARAMS),
               'quantities': []}
    for sym, unit, val, uc, ui, ucomb in rows:
        rel = ucomb / abs(val) if val else 0.0
        print(f"{sym:>10} {val:>10.4g} {uc:>11.3g} {ui:>11.3g} {ucomb:>11.3g} {rel:>6.2%}")
        summary['quantities'].append(
            {'symbol': sym, 'unit': unit, 'value': val, 'u_cal': uc,
             'u_idn': ui, 'u_combined': ucomb, 'rel_combined': rel})
    json.dump(summary, open(um.rj('uncertainty_summary.json'), 'w'), indent=2)

    with open(um.rt('uncertainty_summary.csv'), 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['quantity', 'unit', 'value', 'u_calibration', 'u_identifiability',
                     'u_combined', 'rel_combined'])
        for sym, unit, val, uc, ui, ucomb in rows:
            wr.writerow([sym, unit, f'{val:.5g}', f'{uc:.4g}', f'{ui:.4g}',
                         f'{ucomb:.4g}', f'{(ucomb/abs(val) if val else 0):.4f}'])

    # --- LaTeX: order-parameter uncertainty budget ---
    def fu(v):
        return f'{v:.0f}' if abs(v) >= 10 else f'{v:.2g}'
    label = {'Ms0': r'$M_{s0}$ (MA/m)', 'Tc': r'$T_c$ (K)', 'beta': r'$\beta$'}
    vfmt = {'Ms0': '.2f', 'Tc': '.1f', 'beta': '.3f'}   # value formats as quoted
    lines = [r'\begin{table}[htbp]', r'  \centering',
             r'  \caption{Combined standard uncertainty of the order-parameter '
             r'quantities. $u_{\mathrm{cal}}$ is the measurement-system calibration '
             r'contribution (Type B) and $u_{\mathrm{idn}}$ the identifiability '
             r'contribution (Type A); $u_c$ is their quadrature sum. The measurement '
             r'system is not the limiting factor.}',
             r'  \label{tab:uncertainty}',
             r'  \begin{tabular}{lcccc}', r'    \toprule',
             r'    Quantity & Value & $u_{\mathrm{cal}}$ & $u_{\mathrm{idn}}$ & $u_c$ \\',
             r'    \midrule']
    for sym, unit, val, uc, ui, ucomb in rows:
        lines.append(f'    {label[sym]} & {format(val, vfmt[sym])} & {fu(uc)} & {fu(ui)} & {fu(ucomb)} \\\\')
    lines += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}']
    with open(um.rt('uncertainty_budget.tex'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # --- LaTeX: parameter correlation matrix (the dependency) ---
    Rm = np.array(corr['correlation_matrix'])
    sym = {'Ms': r'$M_s$'}
    head = ' & '.join(sym.get(p, f'${p}$') for p in PARAMS)
    clines = [r'\begin{table}[htbp]', r'  \centering',
              r'  \caption{Empirical correlation matrix of the per-temperature '
              r'Jiles--Atherton parameters (log-residuals of the independent fits '
              r'about the global laws). The parameters are well determined by the '
              r'global fit but mutually dependent: $a$ and $M_s$ are strongly '
              r'correlated, and $a$, $k$, $c$ are mutually correlated.}',
              r'  \label{tab:correlations}',
              r'  \begin{tabular}{l' + 'c' * len(PARAMS) + r'}', r'    \toprule',
              f'     & {head} \\\\', r'    \midrule']
    for i, p in enumerate(PARAMS):
        nm = sym.get(p, f'${p}$')
        clines.append('    ' + nm + ' & '
                      + ' & '.join(f'{Rm[i, j]:.2f}' for j in range(len(PARAMS))) + r' \\')
    clines += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}']
    with open(um.rt('parameter_correlations.tex'), 'w') as f:
        f.write('\n'.join(clines) + '\n')

    print(f"\nk(T) decreasing in {summary['k_trend_decreasing_fraction']:.1%} of trials")
    print('saved uncertainty_summary.json/.csv, uncertainty_budget.tex, parameter_correlations.tex')


if __name__ == '__main__':
    main()
