# Author: Roman Szewczyk
# License: MIT License (see LICENSE file)

"""Per-temperature Jiles-Atherton identification (Stage A).

Each major loop is fitted independently for (a, k, c, Ms) with alpha = 0 and
a scalar k. This is the honest single-loop fit: it reaches R2_adj ~ 0.999 at
every temperature, but the resulting k and c scatter from loop to loop because
a single loop does not determine them uniquely near the Curie point. The
values therefore serve as a diagnostic and as priors for the global fit
(global_physical_ja_fit.py), which resolves the scatter into smooth trends.

Usage:
    python3 identify_per_temperature.py [start_idx] [end_idx]
"""

import os
import sys
import json
import time
import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ja_model import MU0, single_loop, calc_metrics
from paths import PROCESSED_NPZ, results_json

ALPHA = 0.0


def coercivity(H, B):
    """Mean absolute coercivity from the zero crossings of B(H)."""
    hc = []
    for i in range(1, len(B)):
        if B[i - 1] * B[i] < 0:
            t = abs(B[i - 1]) / (abs(B[i - 1]) + abs(B[i]))
            hc.append(abs(H[i - 1] + t * (H[i] - H[i - 1])))
    return float(np.mean(hc)) if hc else 100.0


def saturation_estimate(H, B):
    """Saturation magnetisation estimated from the high-field loop tip."""
    j = int(np.argmax(H))
    return float(B[j] / MU0 - H[j])


class _Objective:
    """SSE of one loop; a top-level class so it is picklable for DE workers."""

    def __init__(self, H, B, M0):
        self.H, self.B, self.M0 = H, B, M0

    def __call__(self, p):
        a, k, c, Ms = p
        if a <= 0 or k <= 0 or c < 0 or c > 0.95 or Ms <= 0:
            return 1e30
        try:
            _, Bsim = single_loop(a, k, c, Ms, ALPHA, self.H, M0=self.M0)
        except Exception:
            return 1e30
        if len(Bsim) != len(self.B) or not np.all(np.isfinite(Bsim)):
            return 1e30
        return float(np.sum((self.B - Bsim) ** 2))


def identify(T, H, B):
    H = np.asarray(H, dtype=float).ravel()
    B = np.asarray(B, dtype=float).ravel()
    Hc = coercivity(H, B)
    Ms_tip = saturation_estimate(H, B)
    M0 = B[0] / MU0 - H[0]

    # Physical bounds; c and Ms are kept in physical ranges so the optimiser
    # cannot reach high-R2 but non-physical corners near the Curie point.
    bounds = [(max(200.0, Hc * 0.5), max(6000.0, Hc * 16.0)),
              (max(50.0, Hc * 0.3), max(3500.0, Hc * 20.0)),
              (0.0, 0.35),
              (Ms_tip * 0.98, Ms_tip * 1.18)]

    t0 = time.time()
    res = differential_evolution(_Objective(H, B, M0), bounds, popsize=12,
                                 maxiter=50, tol=1e-9, mutation=(0.5, 1.0),
                                 recombination=0.7, seed=42, polish=True,
                                 workers=-1, updating='deferred')
    a, k, c, Ms = res.x
    _, Bsim = single_loop(a, k, c, Ms, ALPHA, H, M0=M0)
    m = calc_metrics(B, Bsim, 4)
    print(f"T={T:>5.0f} C  a={a:>7.0f}  k={k:>6.0f}  c={c:>6.4f}  "
          f"Ms={Ms / 1e6:.4f}M  R2adj={m['r2_adj']:.5f}  ({time.time() - t0:.1f}s)",
          flush=True)
    return {'temperature': float(T), 'M0': float(M0),
            'params': {'a': float(a), 'k': float(k), 'c': float(c),
                       'Ms': float(Ms), 'alpha': ALPHA},
            'metrics': {kk: float(vv) for kk, vv in m.items()},
            'B_sim': Bsim.tolist()}


def main():
    data = np.load(PROCESSED_NPZ, allow_pickle=True)
    T = np.array(data['temperatures'], dtype=float)
    mH, mB = data['major_H'], data['major_B']
    n = len(T)

    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else n
    lo, hi = max(0, lo), min(n, hi)

    out = results_json('parameters_per_temperature.json')
    results = json.load(open(out)) if os.path.exists(out) else []
    while len(results) < n:
        results.append(None)

    for i in range(lo, hi):
        results[i] = identify(T[i], mH[i], mB[i])
        json.dump(results, open(out, 'w'), indent=2)


if __name__ == '__main__':
    main()
