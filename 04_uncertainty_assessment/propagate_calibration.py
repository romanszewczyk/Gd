# Author: Roman Szewczyk
# License: MIT License (see LICENSE file)

"""Component A: propagation of the measurement-system (calibration) uncertainty.

This is the measurement-uncertainty propagation requested in the paper. The
two Type B input quantities are the calibration scale (gain) uncertainties of
the flux density B and the field H (see uncertainty_model.py):

    u_rel(B) = REL_U_B = 0.1 % ,   u_rel(H) = REL_U_H = 0.01 % .

Both are *systematic*: a single unknown gain multiplies every point of every
loop, so they are common to all twelve temperatures. They are propagated
through the well-conditioned global fit by the GUM law of propagation of
uncertainty. Linearising the global weighted-least-squares estimator about the
published optimum gives the operator

    K = (G^T W G)^{-1} G^T W ,      G = dB_sim/dx ,

which maps a perturbation of the data to the resulting change of the twelve
global parameters x. A B-gain g_B perturbs the data by dB = g_B * B_meas, and
an H-gain g_H rescales the field axis, perturbing the model by
dB_sim = (d B_sim / d g_H). Hence the sensitivities

    s_B = K @ B_meas ,             s_H = -K @ (dB_sim/dg_H) ,

and the combined calibration standard uncertainty of each parameter is

    u_cal(x_j) = sqrt( (s_B,j * u_rel(B))^2 + (s_H,j * u_rel(H))^2 ) .

(This linear operator was cross-checked against a full non-linear global
re-fit of gain-rescaled data: the well-determined direction -- the saturation
amplitude -- agrees to better than 1 %, dMs0/dg_B ~ 0.77. The remaining
sensitivities lie in the flat k--c valley of the objective, where the
contribution is negligible at the 0.1 %/0.01 % input level regardless of its
exact value. Per-loop re-fitting is degenerate and is therefore not used.)

Because the gains rescale the axes without changing the loop shape, the result
is dominated by the saturation-magnetization amplitude M_s0 (~0.1 %) and is
negligible for the dimensionless exponent beta, the Curie temperature T_c and
the parameter shapes.
"""

import json
import time
import numpy as np

import uncertainty_model as um

SUBSAMPLE = 300
H_EPS = 1e-3                # finite step for the H-gain model derivative


def main():
    t0 = time.time()
    T_C, T_K, loops = um.load_major_loops(subsample=SUBSAMPLE)
    x0 = um.x0_from_coeffs()
    Bmeas = np.concatenate([lp['B'] for lp in loops])

    G, base = um.global_jacobian(x0, loops, T_C, T_K)
    w = um.weights_vector(loops)
    K, GtWG, cond = um.estimator_operator(G, w, scale=np.abs(x0))
    print(f"global Jacobian {G.shape}, cond(scaled G^T W G) = {cond:.3e} "
          f"({time.time() - t0:.0f}s)")

    # B-gain: data perturbation dB = g_B * B_meas  ->  s_B = K @ B_meas
    s_B = K @ Bmeas
    # H-gain: field axis rescaled -> model perturbation dB_sim/dg_H
    dB_dgH = (um.global_simulate(x0, loops, T_C, T_K, H_scale=1 + H_EPS)
              - um.global_simulate(x0, loops, T_C, T_K, H_scale=1 - H_EPS)) / (2 * H_EPS)
    s_H = -K @ dB_dgH

    u_cal = np.sqrt((s_B * um.REL_U_B) ** 2 + (s_H * um.REL_U_H) ** 2)

    print(f"\n{'parameter':>10} {'value':>12} {'u_cal':>12} {'u_cal/val':>10}  "
          f"{'dx/dgB':>11} {'dx/dgH':>11}")
    out = {}
    for nm, val, uc, sb, sh in zip(um.X_NAMES, x0, u_cal, s_B, s_H):
        rel = uc / abs(val) if val else 0.0
        print(f"{nm:>10} {val:>12.4g} {uc:>12.4g} {rel:>9.3%}  {sb:>11.3g} {sh:>11.3g}")
        out[nm] = {'value': float(val), 'u_cal': float(uc), 'u_cal_rel': float(rel),
                   'dx_dgB': float(sb), 'dx_dgH': float(sh)}

    summary = {
        'description': 'Component A: measurement-system calibration uncertainty '
                       'propagated through the global fit (linearised GUM LPU).',
        'inputs': {'u_rel_B': um.REL_U_B, 'u_rel_H': um.REL_U_H,
                   'subsample': SUBSAMPLE},
        'cond_scaled_GtWG': cond,
        'parameters': out,
    }
    json.dump(summary, open(um.rj('calibration_uncertainty.json'), 'w'), indent=2)
    print(f"\nsaved {um.rj('calibration_uncertainty.json')}  "
          f"(total {time.time() - t0:.0f}s)")


if __name__ == '__main__':
    main()
