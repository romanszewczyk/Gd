# Author: Roman Szewczyk
# License: MIT License (see LICENSE file)

"""Empirical correlation structure of the Jiles--Atherton parameters.

The four Jiles--Atherton parameters (a, k, c, Ms) are *not* independent. For a
single loop the anhysteretic field scale a, the pinning k and the reversibility
c trade off against one another (the well-known ill-conditioning of JA
identification), and a and Ms are tied through the loop amplitude. The
uncertainty analysis therefore uses the *joint* (correlated) distribution of
the parameters, not the product of their marginals.

This module estimates that joint distribution empirically from the two
independent extractions already in the repository:

  * the independent per-temperature fits (parameters_per_temperature.json), and
  * the smooth global temperature laws    (parameters_physical.json).

For every temperature i and parameter p it forms the log-residual

    L_p(T_i) = ln( p_independent(T_i) / p_law(T_i) ),

i.e. the multiplicative deviation of the freely-fitted value from the smooth
law. The twelve temperatures provide twelve samples of the 4-vector
(L_a, L_k, L_c, L_Ms). Their mean is the systematic offset between the two
extractions; their *covariance* Sigma (centred) is the scatter that the
identifiability uncertainty propagates. Working in log space keeps every
positive parameter positive under perturbation and makes the diagonal of Sigma
the (squared) relative scatter used by the previous, independent-draw analysis,
so the two are directly comparable: setting the off-diagonal of Sigma to zero
recovers the old independent model exactly.

Outputs ``results/json/parameter_correlations.json`` with the relative scatter,
the covariance Sigma and the correlation matrix R, for use by
``propagate_identifiability.py`` and ``plot_uncertainty.py``.
"""

import json
import numpy as np

import uncertainty_model as um

PARAMS = ('a', 'k', 'c', 'Ms')


def load_log_residuals():
    """Return (T_C, L) with L[p] the log-residuals ln(independent/law)."""
    ind = json.load(open(um.PER_TEMP))
    glob = json.load(open(um.GLOBAL_PARAMS))
    T_C = np.array([e['temperature'] for e in glob], dtype=float)
    vals_ind = {p: np.array([e['params'][p] for e in ind], dtype=float) for p in PARAMS}
    vals_law = {p: np.array([e['params'][p] for e in glob], dtype=float) for p in PARAMS}
    L = np.vstack([np.log(vals_ind[p] / vals_law[p]) for p in PARAMS])  # (4, n_T)
    return T_C, L, vals_ind, vals_law


def covariance(L):
    """Centred covariance Sigma (4x4) and correlation matrix R of the log-residuals.

    np.cov centres each row (subtracts the per-parameter mean over temperatures),
    so Sigma is the *scatter* covariance, independent of the systematic offset
    between the two extractions.
    """
    Sigma = np.cov(L, ddof=1)                       # (4, 4)
    d = np.sqrt(np.diag(Sigma))
    R = Sigma / np.outer(d, d)
    return Sigma, R, d


def main():
    T_C, L, vals_ind, vals_law = load_log_residuals()
    Sigma, R, sig_ln = covariance(L)

    # relative (linear) scatter, for comparison with the diagonal log scatter
    rel = {p: float(np.sqrt(np.mean(((vals_ind[p] - vals_law[p]) / vals_law[p]) ** 2)))
           for p in PARAMS}

    print("log-residual scatter (diagonal of Sigma, sigma_ln):")
    for p, s in zip(PARAMS, sig_ln):
        print(f"  {p:>3}: sigma_ln = {s:.4f}   (linear rel. scatter {rel[p]:.4f})")

    print("\ncorrelation matrix R of (a, k, c, Ms) log-residuals:")
    print("        " + "  ".join(f"{p:>6}" for p in PARAMS))
    for p, row in zip(PARAMS, R):
        print(f"  {p:>3}  " + "  ".join(f"{v:>6.2f}" for v in row))

    # off-diagonal strength: largest |correlation| among a, k, c
    akc = R[:3, :3]
    off = akc[np.triu_indices(3, 1)]
    print(f"\nmax |corr| among (a,k,c): {np.max(np.abs(off)):.2f}")

    out = {
        'description': 'Empirical joint (correlated) distribution of the JA '
                       'parameters from log-residuals ln(independent/law) over '
                       'the twelve temperatures.',
        'params': list(PARAMS),
        'n_temperatures': int(L.shape[1]),
        'sigma_ln': dict(zip(PARAMS, sig_ln.tolist())),
        'relative_scatter': rel,
        'covariance_logspace': Sigma.tolist(),
        'correlation_matrix': R.tolist(),
    }
    json.dump(out, open(um.rj('parameter_correlations.json'), 'w'), indent=2)
    print(f"\nsaved {um.rj('parameter_correlations.json')}")


if __name__ == '__main__':
    main()
