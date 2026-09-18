"""Guarded JA loop for alpha > 0: floors the irreversible denominator so the ODE cannot become
singular where delta*k ~ alpha*(Man - M), caps the step count, and aborts slow evaluations."""
import os, time, numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ja_model import man, dman, MU0, _interp

DENOM_FLOOR = 0.0      # 0 = time cap only (identical physics to ja_model where it finishes)
MAX_SECONDS = 3.0      # per loop evaluation

class TooSlow(Exception):
    pass

def dMdH_g(a, k, c, Ms, alpha, M, H, Hstart, Hend):
    He = H + alpha * M
    Man = man(a, Ms, He)
    dM1 = Man - M
    delta = 1.0 if Hend >= Hstart else -1.0
    if Hend > Hstart: dM1 = max(dM1, 0.0)
    elif Hend < Hstart: dM1 = min(dM1, 0.0)
    denom = (1.0 + c) * (delta * k - alpha * (Man - M))
    floor = max(DENOM_FLOOR * k * (1.0 + c), 1e-70)
    if abs(denom) < floor:
        denom = floor * (delta if denom == 0 else np.sign(denom))
    return dM1 / denom + c / (1.0 + c) * dman(a, Ms, He)

def _solve_segment(a, k, c, Ms, alpha, Hstart, Hend, M0, t0):
    if Hend == Hstart:
        return np.array([Hstart, Hend]), np.array([M0, M0])
    span = abs(Hend - Hstart)
    def rhs(H, M):
        if time.time() - t0 > MAX_SECONDS: raise TooSlow()
        return dMdH_g(a, k, c, Ms, alpha, M[0], H, Hstart, Hend)
    sol = solve_ivp(rhs, (Hstart, Hend), [M0], method='RK45', rtol=1e-4, atol=1e-6,
                    max_step=span / 10.0, first_step=span / 10.0)
    return sol.t, sol.y[0]

def single_loop_g(a, k, c, Ms, alpha, H, M0=0.0):
    H = np.asarray(H, dtype=float).ravel(); t0 = time.time()
    M = np.zeros(len(H)); M[0] = M0
    ip, ik, m0 = 0, 1, M0
    while ik < len(H) - 1:
        rising = H[ik] > H[ip]; falling = H[ik] < H[ip]
        if rising and H[ik + 1] >= H[ik]: ik += 1
        elif falling and H[ik + 1] <= H[ik]: ik += 1
        elif rising or falling:
            Hi, Mi = _solve_segment(a, k, c, Ms, alpha, H[ip], H[ik], m0, t0)
            M[ip + 1:ik + 1] = _interp(Hi, Mi, H[ip + 1:ik + 1]); m0 = Mi[-1]; ip, ik = ik, ik + 1
        else: ik += 1
    Hi, Mi = _solve_segment(a, k, c, Ms, alpha, H[ip], H[ik], m0, t0)
    M[ip + 1:ik + 1] = _interp(Hi, Mi, H[ip + 1:ik + 1])
    return H, MU0 * (M + H)
