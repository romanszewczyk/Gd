# Author: Roman Szewczyk
# License: MIT License (see LICENSE file)

"""Input-uncertainty model and shared utilities for the uncertainty assessment.

The Jiles--Atherton parameters of gadolinium are extracted from measured
B(H) loops (see ../02_scripts). This module collects everything the
uncertainty propagation needs:

  * the input (Type B) standard uncertainties of the measurement system,
  * loading of the measured major loops and the published global model,
  * the global forward model B_sim(x) and its Jacobian G = dB_sim/dx with
    respect to the twelve global parameters,
  * the weighted least-squares estimator operator used for propagation.

Why the *global* model is the right level for propagation
---------------------------------------------------------
Per-loop identification of (a, k, c, M_s) is ill-conditioned: re-fitting a
single loop from slightly different starting points returns (a, k, c) that
differ by several per cent at constant fit quality, because these parameters
trade off against one another (the degeneracy that motivates the global fit in
the first place). Propagating measurement uncertainty through such a degenerate
per-loop fit is meaningless. The *global* fit -- twelve smooth temperature laws
shared by all loops -- is well conditioned, so we linearise and propagate
through it: the twelve global parameters are the measurand.

Parameterisation
----------------
The global model is described, exactly as in the fit
(../02_scripts/global_physical_ja_fit.py), by the twelve-vector

    x = [Ms0, beta, Tc, a(-38), a(-11.5), a(15),
                       k(-38), k(-11.5), k(15),
                       c(-38), c(-11.5), c(15)]

(the three anchor temperatures of ANCHORS_T). ``laws(x)`` turns x into the
per-temperature (a, k, c, M_s) and the polynomial coefficients of a(T), k(T),
c(T). The anchor values have interpretable, comparable magnitudes, which keeps
the Jacobian well scaled.

Measurement-system uncertainty (Type B)
---------------------------------------
The loops were recorded with the calibrated ferrograph/hysteresisgraph system
of Charubin, Nowicki, Marusenkov, Szewczyk et al. (JAMRIS 12(3), 2018), whose
calibration against certified standards gives a relative accuracy of 0.1 % for
the flux density B and 0.01 % for the magnetizing field H. A calibration
accuracy is a *scale* (gain) uncertainty: it multiplies every point of every
loop by the same unknown factor, so it is fully correlated across the points of
a loop and across all twelve temperatures (the same instrument):

    B_meas = (1 + g_B) * B_true,     u(g_B) = REL_U_B
    H_meas = (1 + g_H) * H_true,     u(g_H) = REL_U_H .

The quoted accuracies are taken directly as relative standard uncertainties
(a conservative reading; read as the half-width of a rectangular distribution
they would be smaller by sqrt(3)). Because g_B and g_H rescale the axes without
changing the loop *shape*, they propagate almost entirely into the amplitude
scales and leave the dimensionless exponent beta and the Curie temperature T_c
essentially untouched -- a result the propagation makes quantitative.
"""

import os
import sys
import json
import numpy as np
from scipy.optimize import least_squares

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(os.path.dirname(_HERE), '02_scripts')
sys.path.insert(0, _SCRIPTS)

from ja_model import MU0, single_loop                       # noqa: E402
from global_physical_ja_fit import laws, ANCHORS_T, ALPHA   # noqa: E402

# --- Repository paths -------------------------------------------------------
ROOT = os.path.dirname(_HERE)
PROCESSED_NPZ = os.path.join(ROOT, '01_data', 'processed', 'gd_processed.npz')
GLOBAL_PARAMS = os.path.join(ROOT, '03_results', 'json', 'parameters_physical.json')
GLOBAL_COEFFS = os.path.join(ROOT, '03_results', 'json', 'physical_model_coeffs.json')
PER_TEMP = os.path.join(ROOT, '03_results', 'json', 'parameters_per_temperature.json')

RESULTS_JSON = os.path.join(_HERE, 'results', 'json')
RESULTS_TABLES = os.path.join(_HERE, 'results', 'tables')
RESULTS_FIGURES = os.path.join(_HERE, 'results', 'figures')
for _d in (RESULTS_JSON, RESULTS_TABLES, RESULTS_FIGURES):
    os.makedirs(_d, exist_ok=True)


def rj(name):
    return os.path.join(RESULTS_JSON, name)


def rt(name):
    return os.path.join(RESULTS_TABLES, name)


def rf(name):
    return os.path.join(RESULTS_FIGURES, name)


# --- Input (Type B) standard uncertainties of the measurement system --------
REL_U_B = 1.0e-3    # 0.1 %  : calibration scale uncertainty of flux density B
REL_U_H = 1.0e-4    # 0.01 % : calibration scale uncertainty of field H

# names of the twelve global parameters, in the order of x
X_NAMES = ['Ms0', 'beta', 'Tc',
           'a(-38)', 'a(-11.5)', 'a(15)',
           'k(-38)', 'k(-11.5)', 'k(15)',
           'c(-38)', 'c(-11.5)', 'c(15)']


def load_major_loops(subsample=None):
    """Return (T_C, T_K, loops). Each loop is a dict with H, B, M0, SST, Bmax."""
    data = np.load(PROCESSED_NPZ, allow_pickle=True)
    T_C = np.array(data['temperatures'], dtype=float)
    mH, mB = data['major_H'], data['major_B']
    loops = []
    for i in range(len(T_C)):
        H = np.asarray(mH[i], dtype=float).ravel()
        B = np.asarray(mB[i], dtype=float).ravel()
        if subsample and len(H) > subsample:
            idx = np.linspace(0, len(H) - 1, subsample).astype(int)
            H, B = H[idx], B[idx]
        loops.append({'H': H, 'B': B, 'M0': B[0] / MU0 - H[0],
                      'SST': float(np.sum((B - B.mean()) ** 2)),
                      'Bmax': float(np.max(np.abs(B)))})
    return T_C, T_C + 273.15, loops


def x0_from_coeffs(coeffs=None):
    """Twelve-vector x of the published global model (anchor parameterisation)."""
    if coeffs is None:
        coeffs = json.load(open(GLOBAL_COEFFS))
    a_anch = np.polyval(coeffs['a_coeffs'], ANCHORS_T)
    k_anch = np.polyval(coeffs['k_coeffs'], ANCHORS_T)
    c_anch = np.polyval(coeffs['c_coeffs'], ANCHORS_T)
    return np.array([coeffs['Ms0'], coeffs['beta'], coeffs['Tc'],
                     *a_anch, *k_anch, *c_anch], dtype=float)


def global_simulate(x, loops, T_C, T_K, H_scale=1.0):
    """Concatenated B_sim over all loops for global parameters x.

    ``H_scale`` rescales the field axis (used to obtain the H-gain sensitivity).
    """
    a, k, c, ms, _ = laws(x, T_C, T_K)
    out = []
    for i, lp in enumerate(loops):
        H = lp['H'] * H_scale
        M0 = lp['M0']               # M0 from the measured loop start
        _, B = single_loop(a[i], k[i], c[i], ms[i], ALPHA, H, M0=M0)
        out.append(B)
    return np.concatenate(out)


def weights_vector(loops):
    """Per-point weights of the global objective: w_ij = 1/(N_loops * SST_i)."""
    n = len(loops)
    return np.concatenate([np.full(len(lp['B']), 1.0 / (n * lp['SST']))
                           for lp in loops])


def global_jacobian(x, loops, T_C, T_K, rel_step=1e-3, abs_floor=1e-9):
    """G[m, j] = dB_sim_m/dx_j by central differences (m over all loop points)."""
    x = np.asarray(x, dtype=float)
    base = global_simulate(x, loops, T_C, T_K)
    G = np.zeros((len(base), len(x)))
    for j in range(len(x)):
        dx = abs(x[j]) * rel_step + abs_floor
        xp, xm = x.copy(), x.copy()
        xp[j] += dx
        xm[j] -= dx
        G[:, j] = (global_simulate(xp, loops, T_C, T_K)
                   - global_simulate(xm, loops, T_C, T_K)) / (2.0 * dx)
    return G, base


def estimator_operator(G, w, scale):
    """K = (G^T W G)^{-1} G^T W : maps a data perturbation to the WLS update.

    The normal equations are solved in parameters scaled by ``scale`` (the
    natural magnitude of each x_j) so that the very different units of the
    twelve parameters do not, on their own, make G^T W G numerically singular.
    The returned K is in original (unscaled) units. ``cond`` is the condition
    number of the *scaled* normal matrix -- the meaningful measure of how well
    the global fit determines the parameters.
    """
    scale = np.asarray(scale, dtype=float)
    S = np.diag(scale)
    Gs = G @ S                          # columns now dimensionless-ish
    GsTW = Gs.T * w                     # (p, m)
    GsTWGs = GsTW @ Gs                  # (p, p), well scaled
    cond = float(np.linalg.cond(GsTWGs))
    K = S @ (np.linalg.solve(GsTWGs, GsTW))
    return K, GsTWGs, cond


# --- Optional per-loop re-fit (used only for cross-checks, NOT for the main
#     propagation, because per-loop identification is degenerate) ------------
def refit_loop(H, B, M0, theta_init, bounds_frac=(0.3, 3.0)):
    """Local least-squares re-fit of one loop for (a, k, c, Ms)."""
    theta_init = np.asarray(theta_init, dtype=float)

    def resid(z):
        a, k, c, Ms = z * theta_init
        try:
            _, Bsim = single_loop(a, k, c, Ms, ALPHA, H, M0=M0)
        except Exception:
            return np.full(len(B), 1e3)
        if len(Bsim) != len(B) or not np.all(np.isfinite(Bsim)):
            return np.full(len(B), 1e3)
        return Bsim - B

    # diff_step must exceed the ~uT jitter of the adaptive ODE solver.
    res = least_squares(resid, np.ones(4),
                        bounds=(bounds_frac[0], bounds_frac[1]),
                        method='trf', diff_step=1e-3,
                        xtol=1e-10, ftol=1e-12, max_nfev=300)
    return res.x * theta_init, bool(res.success)
