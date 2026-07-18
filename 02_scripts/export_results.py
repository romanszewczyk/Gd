# Author: Roman Szewczyk
# License: MIT License (see LICENSE file)

"""Export the parameter tables as CSV and a LaTeX table for publication."""

import os
import sys
import csv
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import results_json, table


def write_csv(path, entries):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['T_C', 'a_A_m', 'k_A_m', 'c', 'Ms_A_m', 'alpha',
                    'R2', 'R2_adj', 'RMSE_T'])
        for e in entries:
            p, m = e['params'], e['metrics']
            w.writerow([f"{e['temperature']:.0f}", f"{p['a']:.1f}", f"{p['k']:.1f}",
                        f"{p['c']:.5f}", f"{p['Ms']:.1f}", f"{p['alpha']:.1f}",
                        f"{m['r2']:.5f}", f"{m['r2_adj']:.5f}", f"{m['rmse']:.5f}"])


def main():
    phys = json.load(open(results_json('parameters_physical.json')))
    per_t = json.load(open(results_json('parameters_per_temperature.json')))

    write_csv(table('parameters_physical.csv'), phys)
    write_csv(table('parameters_per_temperature.csv'), per_t)

    with open(table('model_comparison.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['T_C', 'physical_R2adj', 'per_temperature_R2adj', 'delta'])
        for a, b in zip(phys, per_t):
            ra, rb = a['metrics']['r2_adj'], b['metrics']['r2_adj']
            w.writerow([f"{a['temperature']:.0f}", f"{ra:.5f}", f"{rb:.5f}",
                        f"{ra - rb:+.5f}"])

    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{Temperature-dependent Jiles-Atherton parameters for gadolinium.}",
             r"\label{tab:ja_params}", r"\begin{tabular}{cccccc}", r"\hline",
             r"$T$ (\textdegree C) & $M_s$ (MA/m) & $a$ (A/m) & $k$ (A/m) & $c$ & $R^2_{\mathrm{adj}}$ \\",
             r"\hline"]
    for p in phys:
        pp, m = p['params'], p['metrics']
        lines.append(f"{p['temperature']:.0f} & {pp['Ms']/1e6:.3f} & {pp['a']:.0f} & "
                     f"{pp['k']:.0f} & {pp['c']:.4f} & {m['r2_adj']:.4f} \\\\")
    mean_r2 = np.mean([p['metrics']['r2_adj'] for p in phys])
    lines += [r"\hline",
              f"\\multicolumn{{5}}{{r}}{{Mean $R^2_{{\\mathrm{{adj}}}}$}} & {mean_r2:.4f} \\\\",
              r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(table('ja_parameters.tex'), 'w') as f:
        f.write("\n".join(lines) + "\n")

    print('Saved parameters_physical.csv, parameters_per_temperature.csv, '
          'model_comparison.csv, ja_parameters.tex')


if __name__ == '__main__':
    main()
