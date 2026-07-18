# Author: Roman Szewczyk
# License: MIT License (see LICENSE file)

"""Figures for the two fits.

Per-temperature fit (each loop fitted independently) and global physical fit
(all loops, smooth temperature laws) each get their own loop figure and their
own parameter figure, so the two can be compared directly:

    major_loops_per_temperature.jpg   loops, per-temperature fit
    major_loops_global.jpg            loops, global physical model
    parameters_per_temperature.jpg    a, k, c, Ms vs T, per-temperature fit
    parameters_global.jpg             a, k, c, Ms vs T, global physical model
    r2_comparison.jpg                 goodness of fit of the two
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import PROCESSED_NPZ, results_json, figure


def plot_loops(T, mH, mB, entries, color, model_label, suptitle, fname):
    """Measured vs modelled major loops, one panel per temperature."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    for i, ax in enumerate(axes.flat):
        H = np.asarray(mH[i], dtype=float).ravel()
        B = np.asarray(mB[i], dtype=float).ravel()
        ax.plot(H, B, 'b-', lw=1, label='measured')
        ax.plot(H, np.array(entries[i]['B_sim']), color, lw=1, label=model_label)
        ax.set_title(f"T = {T[i]:.0f} C,  R2adj = {entries[i]['metrics']['r2_adj']:.4f}",
                     fontsize=9)
        ax.set_xlabel('H [A/m]', fontsize=8)
        ax.set_ylabel('B [T]', fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=8)
    if suptitle:
        fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    fig.savefig(figure(fname), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_parameters(T, entries, point_label, suptitle, fname, coeffs=None):
    """a, k, c and Ms versus temperature for one fit.

    If coeffs is given, the smooth physical laws are overlaid (global fit).
    """
    color = 'rs-' if coeffs is not None else 'bo-'
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    if coeffs is not None:
        Tf = np.linspace(T.min(), coeffs['Tc'] - 273.15, 200)
        TfK = Tf + 273.15

    def panel(ax, key, ylabel, title, law=None, law_label=None):
        ax.plot(T, [e['params'][key] for e in entries], color, ms=6, label=point_label)
        if law is not None:
            ax.plot(Tf, law, 'g--', lw=2, label=law_label)
        ax.set_xlabel('T [C]'); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ms_law = (coeffs['Ms0'] * (1 - TfK / coeffs['Tc']) ** coeffs['beta'] / 1e6
              if coeffs is not None else None)
    ax = axes[0, 0]
    ax.plot(T, [e['params']['Ms'] / 1e6 for e in entries], color, ms=6, label=point_label)
    if coeffs is not None:
        ax.plot(Tf, ms_law, 'g--', lw=2,
                label=f"Ms0 (1-T/Tc)^beta, beta={coeffs['beta']:.3f}")
    ax.set_xlabel('T [C]'); ax.set_ylabel('Ms [MA/m]')
    ax.set_title('Saturation magnetisation'); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    panel(axes[0, 1], 'a', 'a [A/m]', 'Langevin parameter a(T)',
          np.polyval(coeffs['a_coeffs'], Tf) if coeffs is not None else None, 'physical law')
    panel(axes[1, 0], 'k', 'k [A/m]', 'Pinning parameter k(T)',
          np.polyval(coeffs['k_coeffs'], Tf) if coeffs is not None else None, 'physical law')
    panel(axes[1, 1], 'c', 'c [-]', 'Reversibility parameter c(T)',
          np.polyval(coeffs['c_coeffs'], Tf) if coeffs is not None else None, 'physical law')
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    fig.savefig(figure(fname), dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_r2(T, per_t, phys):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(T))
    ax.bar(x - 0.2, [p['metrics']['r2_adj'] for p in per_t], 0.4,
           label='per-temperature fit', color='steelblue')
    ax.bar(x + 0.2, [p['metrics']['r2_adj'] for p in phys], 0.4,
           label='global physical model', color='indianred')
    ax.axhline(0.99, color='green', ls='--', lw=1, label='R2adj = 0.99')
    ax.set_xticks(x); ax.set_xticklabels([f'{t:.0f}' for t in T])
    ax.set_xlabel('T [C]'); ax.set_ylabel('R2adj')
    ax.set_ylim(0.985, 1.0005); ax.grid(alpha=0.3, axis='y'); ax.legend()
    ax.set_title('Goodness of fit')
    fig.tight_layout()
    fig.savefig(figure('r2_comparison.jpg'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    data = np.load(PROCESSED_NPZ, allow_pickle=True)
    T = np.array(data['temperatures'], dtype=float)
    mH, mB = data['major_H'], data['major_B']

    phys = json.load(open(results_json('parameters_physical.json')))
    per_t = json.load(open(results_json('parameters_per_temperature.json')))
    coeffs = json.load(open(results_json('physical_model_coeffs.json')))

    plot_loops(T, mH, mB, per_t, 'g--', 'per-temperature fit',
               'Major loops: independent per-temperature fit',
               'major_loops_per_temperature.jpg')
    plot_loops(T, mH, mB, phys, 'r--', 'global physical model',
               None,   # manuscript figure: no overall title, the caption carries it
               'major_loops_global.jpg')
    plot_parameters(T, per_t, 'per-temperature fit',
                    'Parameters: independent per-temperature fit',
                    'parameters_per_temperature.jpg', coeffs=None)
    plot_parameters(T, phys, 'global physical model',
                    'Parameters: global physical model with smooth laws',
                    'parameters_global.jpg', coeffs=coeffs)
    plot_r2(T, per_t, phys)

    print('Saved major_loops_per_temperature.jpg, major_loops_global.jpg, '
          'parameters_per_temperature.jpg, parameters_global.jpg, r2_comparison.jpg')


if __name__ == '__main__':
    main()
