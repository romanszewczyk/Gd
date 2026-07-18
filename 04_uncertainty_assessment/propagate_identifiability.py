# Author: Roman Szewczyk
# License: MIT License (see LICENSE file)

"""Component B: identifiability (Type A) uncertainty of the order parameter.

The global Jiles--Atherton fit is well conditioned (see ``propagate_calibration.py``:
the condition number of the scaled normal matrix is about 1.1e4) and reproduces
every loop, so it determines all four temperature laws; the four parameters are
not poorly determined but mutually *dependent* (their correlation is quantified
in ``correlation_analysis.py``). Per-loop identification, by contrast, is
degenerate -- a single loop does not fix k and c -- which is exactly why the
global, temperature-constrained fit is used.

For the order-parameter quantities (Ms0, Tc, beta) the per-temperature
saturation magnetization is itself well identified (loop-level scatter ~2 %), so
its scatter is a meaningful Type A input. We propagate it into Ms0, Tc and beta
with a Monte Carlo (JCGM 101): in each trial the twelve per-temperature Ms values
are perturbed by a log-normal factor matched to the observed scatter (drawn
jointly with the other parameters from the empirical covariance, so the
dependency is respected) and the power law is re-fitted. The spread of the
re-fitted Ms0, Tc, beta is the identifiability contribution to those quantities,
and the 95 % band of Ms(T) is reported for the figure. The robustness of the
decreasing k(T) trend is recorded as the fraction of trials preserving it.
"""

import json
import numpy as np
from scipy.optimize import least_squares

import uncertainty_model as um
from correlation_analysis import load_log_residuals, covariance, PARAMS

N_LAW = 100000
SEED = 12345


def fit_powerlaw(T_K, Ms, x_init):
    """Fit Ms0 (1 - T_K/Tc)^beta; bounded so the base stays positive."""
    def resid(z):
        Ms0, beta, Tc = z
        base = 1.0 - T_K / Tc
        if np.any(base <= 0):
            return np.full_like(Ms, 1e6)
        return Ms0 * base ** beta - Ms
    lo = [1.0e6, 0.10, T_K.max() + 0.5]
    hi = [4.0e6, 0.60, 320.0]
    z0 = np.clip(x_init, lo, hi)
    r = least_squares(resid, z0, bounds=(lo, hi), xtol=1e-12, ftol=1e-12, max_nfev=200)
    return r.x


def main():
    rng = np.random.default_rng(SEED)
    T_C, L, vals_ind, vals_law = load_log_residuals()
    Sigma, R, sig_ln = covariance(L)
    T_K = T_C + 273.15
    coeffs = json.load(open(um.GLOBAL_COEFFS))
    idx = {p: i for i, p in enumerate(PARAMS)}
    n_T = len(T_C)
    Tgrid = np.linspace(T_C.min(), T_C.max(), 120); TgridK = Tgrid + 273.15

    keep = {'Ms0': [], 'beta': [], 'Tc': []}
    ms_cv = np.zeros((N_LAW, len(Tgrid)))
    k_decreasing = 0
    D = rng.multivariate_normal(np.zeros(4), Sigma, size=(N_LAW, n_T))   # joint, correlated
    for t in range(N_LAW):
        f = np.exp(D[t])
        ms_t = vals_law['Ms'] * f[:, idx['Ms']]
        k_t = vals_law['k'] * f[:, idx['k']]
        Ms0, beta, Tc = fit_powerlaw(T_K, ms_t, (coeffs['Ms0'], coeffs['beta'], coeffs['Tc']))
        keep['Ms0'].append(Ms0); keep['beta'].append(beta); keep['Tc'].append(Tc)
        ms_cv[t] = Ms0 * np.clip(1.0 - TgridK / Tc, 1e-9, None) ** beta
        kc = np.polyfit(T_C, k_t, 2)
        if np.polyval(kc, Tgrid)[-1] < np.polyval(kc, Tgrid)[0]:
            k_decreasing += 1

    def stats(arr):
        arr = np.asarray(arr, float)
        return {'mean': float(np.mean(arr)), 'u': float(np.std(arr, ddof=1)),
                'ci95_lo': float(np.percentile(arr, 2.5)),
                'ci95_hi': float(np.percentile(arr, 97.5))}

    ms_band = {'mean': np.mean(ms_cv, axis=0).tolist(),
               'lo95': np.percentile(ms_cv, 2.5, axis=0).tolist(),
               'hi95': np.percentile(ms_cv, 97.5, axis=0).tolist()}

    result = {
        'description': 'Component B: identifiability (Type A) uncertainty of the '
                       'order-parameter quantities (Ms0, Tc, beta), propagated from '
                       'the well-identified saturation magnetization with a correlated '
                       'Monte Carlo. The temperature laws are determined by the '
                       'well-conditioned global fit; their mutual dependency is given '
                       'by the correlation matrix (correlation_analysis.py).',
        'n_law': N_LAW, 'seed': SEED,
        'loop_level_relative_scatter': {
            p: float(np.sqrt(np.mean(((vals_ind[p] - vals_law[p]) / vals_law[p]) ** 2)))
            for p in PARAMS},
        'correlation_matrix': R.tolist(),
        'temperature_grid_C': Tgrid.tolist(),
        'Ms0': stats(keep['Ms0']), 'beta': stats(keep['beta']), 'Tc': stats(keep['Tc']),
        'k_trend_decreasing_fraction': k_decreasing / N_LAW,
        'ms_band': ms_band,
    }
    json.dump(result, open(um.rj('identifiability_uncertainty.json'), 'w'), indent=2)

    print(f"Ms0 = {result['Ms0']['mean']/1e6:.4f} +/- {result['Ms0']['u']/1e6:.4f} MA/m "
          f"({result['Ms0']['u']/result['Ms0']['mean']:.1%})")
    print(f"Tc  = {result['Tc']['mean']:.2f} +/- {result['Tc']['u']:.2f} K")
    print(f"beta= {result['beta']['mean']:.4f} +/- {result['beta']['u']:.4f}")
    print(f"k(T) decreasing in {result['k_trend_decreasing_fraction']:.1%} of trials")
    print(f"saved {um.rj('identifiability_uncertainty.json')}")


if __name__ == '__main__':
    main()
