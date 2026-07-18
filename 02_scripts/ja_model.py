# Author: Roman Szewczyk
# License: MIT License (see LICENSE file)

"""Isotropic Jiles-Atherton model of magnetic hysteresis.

Standard formulation with a Langevin anhysteretic magnetisation, scalar
pinning k, reversibility c and inter-domain coupling alpha. The governing
ODE is integrated per monotonic field segment with an adaptive solver and
interpolated back onto the measurement grid.

Reference:
    D.C. Jiles, D.L. Atherton, "Theory of ferromagnetic hysteresis",
    J. Magn. Magn. Mater. 61 (1986) 48.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

MU0 = 4.0e-7 * np.pi
_XSMALL = 1e-3   # below |He/a| < _XSMALL the Langevin function and its
                 # derivative are evaluated from their Taylor series, because
                 # the closed forms lose precision to cancellation there


def man(a, Ms, He):
    """Langevin anhysteretic magnetisation Ms*L(He/a), L(x) = coth(x) - 1/x."""
    He = np.asarray(He, dtype=float)
    scalar = He.ndim == 0
    He = np.atleast_1d(He)
    if Ms == 0 or a == 0:
        out = np.zeros_like(He)
        return float(out[0]) if scalar else out
    x = He / a
    out = np.empty_like(x)
    small = np.abs(x) < _XSMALL
    xs = x[small]
    # Series L(x) = x/3 - x^3/45 + 2 x^5/945 (exact at x = 0, no cancellation).
    out[small] = Ms * (xs / 3.0 - xs**3 / 45.0 + 2.0 * xs**5 / 945.0)
    xl = x[~small]
    out[~small] = Ms * (1.0 / np.tanh(xl) - 1.0 / xl)
    return float(out[0]) if scalar else out


def dman(a, Ms, He):
    """dMan/dHe: analytic derivative (Ms/a)*(1/x^2 - csch(x)^2), x = He/a."""
    He = np.asarray(He, dtype=float)
    scalar = He.ndim == 0
    He = np.atleast_1d(He)
    if Ms == 0 or a == 0:
        out = np.zeros_like(He)
        return float(out[0]) if scalar else out
    x = He / a
    out = np.empty_like(x)
    small = np.abs(x) < _XSMALL
    xs = x[small]
    # Series L'(x) = 1/3 - x^2/15 + 2 x^4/189.
    out[small] = Ms / a * (1.0 / 3.0 - xs**2 / 15.0 + 2.0 * xs**4 / 189.0)
    xl = x[~small]
    # csch(x)^2 underflows to 0 for |x| > ~355; clip before sinh to avoid the
    # overflow warning (the clipped value already gives csch^2 = 0 exactly).
    out[~small] = Ms / a * (1.0 / xl**2 - 1.0 / np.sinh(np.clip(xl, -350.0, 350.0)) ** 2)
    return float(out[0]) if scalar else out


def dMdH(a, k, c, Ms, alpha, M, H, Hstart, Hend):
    """Right-hand side of the JA ODE dM/dH for one monotonic segment."""
    He = H + alpha * M
    Man = man(a, Ms, He)
    dM1 = Man - M
    delta = 1.0 if Hend >= Hstart else -1.0
    # Keep the irreversible change consistent with the sweep direction.
    if Hend > Hstart:
        dM1 = max(dM1, 0.0)
    elif Hend < Hstart:
        dM1 = min(dM1, 0.0)
    denom = (1.0 + c) * (delta * k - alpha * (Man - M))
    if abs(denom) < 1e-70:
        denom = 1e-70
    return dM1 / denom + c / (1.0 + c) * dman(a, Ms, He)


def _solve_segment(a, k, c, Ms, alpha, Hstart, Hend, M0):
    """Integrate the ODE from Hstart to Hend with an adaptive RK45 solver."""
    if Hend == Hstart:
        return np.array([Hstart, Hend]), np.array([M0, M0])
    span = abs(Hend - Hstart)
    sol = solve_ivp(
        lambda H, M: dMdH(a, k, c, Ms, alpha, M[0], H, Hstart, Hend),
        (Hstart, Hend), [M0], method='RK45',
        rtol=1e-4, atol=1e-6, max_step=span / 10.0, first_step=span / 10.0)
    return sol.t, sol.y[0]


def _interp(Hi, Mi, H_query):
    """Cubic interpolation of solver output onto the measurement grid."""
    Hi = np.asarray(Hi, dtype=float)
    Mi = np.asarray(Mi, dtype=float)
    order = np.argsort(Hi)
    Hi, Mi = Hi[order], Mi[order]
    keep = np.diff(Hi, prepend=-np.inf) > 0
    Hi, Mi = Hi[keep], Mi[keep]
    if len(Hi) > 2:
        return CubicSpline(Hi, Mi)(H_query)
    return np.full_like(H_query, Mi[-1] if len(Mi) else 0.0)


def single_loop(a, k, c, Ms, alpha, H, M0=0.0):
    """Simulate one B(H) hysteresis loop.

    Splits the field sweep H into monotonic segments, integrates the JA ODE
    on each and interpolates onto H. Returns (H, B) with B = mu0*(M + H).
    """
    H = np.asarray(H, dtype=float).ravel()
    if len(H) < 2:
        raise ValueError("H must have at least two points")
    M = np.zeros(len(H))
    M[0] = M0

    ip, ik, m0 = 0, 1, M0
    while ik < len(H) - 1:
        rising = H[ik] > H[ip]
        falling = H[ik] < H[ip]
        if rising and H[ik + 1] >= H[ik]:
            ik += 1
        elif falling and H[ik + 1] <= H[ik]:
            ik += 1
        elif rising or falling:
            Hi, Mi = _solve_segment(a, k, c, Ms, alpha, H[ip], H[ik], m0)
            M[ip + 1:ik + 1] = _interp(Hi, Mi, H[ip + 1:ik + 1])
            m0 = Mi[-1]
            ip, ik = ik, ik + 1
        else:  # equal fields
            ik += 1

    Hi, Mi = _solve_segment(a, k, c, Ms, alpha, H[ip], H[ik], m0)
    M[ip + 1:ik + 1] = _interp(Hi, Mi, H[ip + 1:ik + 1])
    return H, MU0 * (M + H)


def calc_metrics(B_meas, B_sim, n_params):
    """Return RMSE, R2 and adjusted R2 between measured and simulated B."""
    B_meas = np.asarray(B_meas, dtype=float).ravel()
    B_sim = np.asarray(B_sim, dtype=float).ravel()
    n = len(B_meas)
    sse = float(np.sum((B_meas - B_sim) ** 2))
    sst = float(np.sum((B_meas - B_meas.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    r2_adj = (1.0 - (sse / (n - n_params)) / (sst / (n - 1))
              if n > n_params and sst > 0 else r2)
    return {'rmse': np.sqrt(sse / n), 'r2': r2, 'r2_adj': r2_adj,
            'sse': sse, 'sst': sst, 'n_points': n}
