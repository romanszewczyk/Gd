# Author: Roman Szewczyk
# License: MIT License (see LICENSE file)

"""Physics-results figures for the manuscript that are not produced elsewhere.

    a_effective_moment.jpg   a(T) and the implied effective moment m = kB T/(mu0 a)
    k_c_global.jpg           pinning k(T) and reversibility c(T) from the global fit

The remaining manuscript figures are produced by ``plot_physical_model.py``
(global major loops) and by the uncertainty module
(``04_uncertainty_assessment/plot_uncertainty.py``: Ms power law with its band).
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ja_model import MU0
from paths import PROCESSED_NPZ, results_json, figure

KB = 1.380649e-23      # Boltzmann constant [J/K]


def fig_effective_moment(T_C, coeffs):
    Tg = np.linspace(T_C.min(), T_C.max(), 200)
    TgK = Tg + 273.15
    a = np.polyval(coeffs['a_coeffs'], Tg)
    m = KB * TgK / (MU0 * a)                      # effective moment [A m^2]
    m_muB = m / 9.2740100783e-24                  # in Bohr magnetons
    ms = coeffs['Ms0'] * (1.0 - TgK / coeffs['Tc']) ** coeffs['beta']

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Tg, a, 'b-', lw=2, label='$a(T)$')
    ax.set_xlabel('T [°C]'); ax.set_ylabel('$a$ [A/m]', color='b')
    ax.tick_params(axis='y', labelcolor='b'); ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    m_k = m_muB / 1e3                              # in 10^3 Bohr magnetons
    ax2.plot(Tg, m_k, 'r--', lw=2, label='$m=k_BT/(\\mu_0 a)$')
    # Ms scaled onto the same right axis range for visual comparison of the trend
    ms_scaled = m_k.min() + (ms - ms.min()) / (ms.max() - ms.min()) * (m_k.max() - m_k.min())
    ax2.plot(Tg, ms_scaled, 'k:', lw=1.5, label='$M_s(T)$ (scaled)')
    ax2.set_ylabel('effective moment $m$ [$10^3\\,\\mu_B$]', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc='upper left', fontsize=9)
    fig.tight_layout(); fig.savefig(figure('a_effective_moment.jpg'), dpi=150)
    plt.close(fig)


def fig_kc_global(T_C, coeffs):
    """Pinning k(T) (left axis) and reversibility c(T) (right axis)."""
    phys = json.load(open(results_json('parameters_physical.json')))
    Tp = np.array([e['temperature'] for e in phys])
    k_pts = np.array([e['params']['k'] for e in phys])
    c_pts = np.array([e['params']['c'] for e in phys])
    Tg = np.linspace(T_C.min(), T_C.max(), 200)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Tp, k_pts, 'bo', ms=6, label='$k$ global fit')
    ax.plot(Tg, np.polyval(coeffs['k_coeffs'], Tg), 'b-', lw=2, label='$k(T)$ law')
    ax.set_xlabel('T [°C]'); ax.set_ylabel('pinning $k$ [A/m]', color='b')
    ax.tick_params(axis='y', labelcolor='b'); ax.grid(alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(Tp, c_pts, 'rs', ms=6, label='$c$ global fit')
    ax2.plot(Tg, np.polyval(coeffs['c_coeffs'], Tg), 'r--', lw=2, label='$c(T)$ law')
    ax2.set_ylabel('reversibility $c$ [-]', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc='upper right', fontsize=9)
    fig.tight_layout(); fig.savefig(figure('k_c_global.jpg'), dpi=150)
    plt.close(fig)


def main():
    data = np.load(PROCESSED_NPZ, allow_pickle=True)
    T_C = np.array(data['temperatures'], dtype=float)
    coeffs = json.load(open(results_json('physical_model_coeffs.json')))

    fig_effective_moment(T_C, coeffs)
    fig_kc_global(T_C, coeffs)
    print('Saved a_effective_moment.jpg, k_c_global.jpg')


if __name__ == '__main__':
    main()
