# Author: Roman Szewczyk
# License: MIT License (see LICENSE file)

"""Figure for the uncertainty assessment.

    Ms_powerlaw_uncertainty.jpg   Ms(T) power law with its 95 % confidence band

The parameter dependency is reported as a correlation matrix (a table, written by
combine_and_report.py); it is not duplicated as a figure. The temperature laws
a(T), k(T), c(T) are determined by the well-conditioned global fit and are not
plotted with marginal "uncertainty bands", which would misrepresent the
mutually dependent parameters as independently uncertain.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import uncertainty_model as um


def main():
    idn = json.load(open(um.rj('identifiability_uncertainty.json')))
    summ = json.load(open(um.rj('uncertainty_summary.json')))
    q = {e['symbol']: e for e in summ['quantities']}
    glob = json.load(open(um.GLOBAL_PARAMS))
    Tg = np.array(idn['temperature_grid_C'])
    b = idn['ms_band']
    T = np.array([e['temperature'] for e in glob])
    ms_pts = np.array([e['params']['Ms'] for e in glob]) / 1e6

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(Tg, np.array(b['lo95']) / 1e6, np.array(b['hi95']) / 1e6,
                    color='indianred', alpha=0.25, label='95 % confidence band')
    ax.plot(Tg, np.array(b['mean']) / 1e6, 'r-', lw=2, label='mean power law')
    ax.plot(T, ms_pts, 'ks', ms=6, label='global-fit $M_s$')
    ax.set_xlabel('T [°C]'); ax.set_ylabel('$M_s$ [MA/m]')
    # Annotate with the reported quantities: the global-fit values and their
    # combined standard uncertainties u_c (as in the text and budget table),
    # not the Monte Carlo means, so figure and text quote identical numbers.
    # In-axes text box, not a title: the journal figure carries no top title.
    ax.text(0.03, 0.05,
            f"$M_{{s0}}={q['Ms0']['value']:.2f}\\pm{q['Ms0']['u_combined']:.2f}$ MA/m\n"
            f"$T_c={q['Tc']['value']:.1f}\\pm{q['Tc']['u_combined']:.1f}$ K\n"
            f"$\\beta={q['beta']['value']:.3f}\\pm{q['beta']['u_combined']:.3f}$",
            transform=ax.transAxes, fontsize=9, va='bottom')
    ax.grid(alpha=0.3); ax.legend(loc='upper right')
    fig.tight_layout(); fig.savefig(um.rf('Ms_powerlaw_uncertainty.jpg'), dpi=150)
    plt.close(fig)
    print('saved Ms_powerlaw_uncertainty.jpg')


if __name__ == '__main__':
    main()
