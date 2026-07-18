# Author: Roman Szewczyk
# License: MIT License (see LICENSE file)

"""Robustness cross-check of the effective saturation exponent beta.

The global fit returns an apparent exponent beta = 0.281, well below the
asymptotic critical value of gadolinium. This script tests whether that low
value is an artefact of the finite distance to T_c: it refits the power law
Ms(T_K) = Ms0 (1 - T_K/Tc)^beta to the *independent* per-temperature Ms values
(parameters_per_temperature.json) over windows that progressively approach T_c,
and over a range of assumed T_c.

Findings reproduced here and reported in the manuscript (Sec. 7.1):
  * with T_c fixed at the determined 291.48 K, beta stays ~0.28-0.30 for every
    window down to tau_min ~ 0.012 -- restricting to the loops nearest T_c does
    NOT raise it toward 0.36, so the low value is not a finite-window artefact;
  * beta is instead controlled by the assumed T_c (the beta-T_c fit
    correlation): it rises from 0.28 at the determined T_c = 291.48 K to about
    0.31 when T_c is held at the single-crystal bulk value 293 K, whereas a
    free fit of these polycrystalline data prefers the lower T_c ~ 291 K used
    throughout.

The script reads committed results only and writes a small JSON summary; it does
not modify the fit.
"""

import os
import sys
import json
import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import results_json

TC_FIXED = 291.48          # determined Curie temperature (global fit), in K
T_WINDOWS_C = [-38, -30, -20, -15, -10, -5, 0]   # include loops with T >= cut
TC_SCAN_K = [290.0, 291.48, 293.0, 293.6]


def _powerlaw(T_K, Ms0, beta, Tc):
    return Ms0 * np.clip(1.0 - T_K / Tc, 1e-9, None) ** beta


def fit_fixed_tc(T_K, Ms, Tc):
    """Fit (Ms0, beta) with Tc held fixed."""
    f = lambda Tk, Ms0, beta: _powerlaw(Tk, Ms0, beta, Tc)
    popt, pcov = curve_fit(f, T_K, Ms, p0=[2.0e6, 0.30], maxfev=20000)
    return popt[1], float(np.sqrt(np.diag(pcov))[1]), popt[0]


def main():
    d = json.load(open(results_json('parameters_per_temperature.json')))
    T_C = np.array([e['temperature'] for e in d], dtype=float)
    Ms = np.array([e['params']['Ms'] for e in d], dtype=float)
    T_K = T_C + 273.15
    tau = (TC_FIXED - T_K) / TC_FIXED

    # Windowed fits at the determined T_c.
    windows = []
    print(f"Windowed power-law fits, T_c fixed at {TC_FIXED} K:")
    print(f"{'T>= [C]':>8} {'n':>3} {'tau_min':>8} {'beta':>14} {'Ms0[MA/m]':>10}")
    for cut in T_WINDOWS_C:
        m = T_C >= cut
        if m.sum() < 3:
            continue
        beta, berr, Ms0 = fit_fixed_tc(T_K[m], Ms[m], TC_FIXED)
        print(f"{cut:>8} {m.sum():>3} {tau[m].min():>8.4f} "
              f"{beta:>7.3f}+-{berr:.3f} {Ms0/1e6:>10.3f}")
        windows.append({'cut_C': cut, 'n': int(m.sum()),
                        'tau_min': float(tau[m].min()), 'tau_max': float(tau[m].max()),
                        'beta': float(beta), 'beta_err': float(berr), 'Ms0': float(Ms0)})

    # Sensitivity of the full-range beta to the assumed T_c.
    tc_scan = []
    print("\nFull-range beta vs assumed T_c:")
    for Tc in TC_SCAN_K:
        beta, berr, Ms0 = fit_fixed_tc(T_K, Ms, Tc)
        print(f"  T_c={Tc:7.2f} K -> beta={beta:.3f}")
        tc_scan.append({'Tc_K': Tc, 'beta': float(beta), 'beta_err': float(berr)})

    # Free-T_c baseline (sanity: should come out close to the global beta, T_c).
    popt, _ = curve_fit(_powerlaw, T_K, Ms, p0=[2.0e6, 0.30, 292.0],
                        bounds=([1e6, 0.1, 289.0], [4e6, 0.6, 320.0]), maxfev=20000)
    free = {'Ms0': float(popt[0]), 'beta': float(popt[1]), 'Tc_K': float(popt[2])}
    print(f"\nFree-Tc full-range baseline: beta={free['beta']:.3f}, "
          f"Tc={free['Tc_K']:.2f} K, Ms0={free['Ms0']/1e6:.3f} MA/m")

    out = {'description': 'Robustness of the apparent exponent beta to the fitting '
                          'window and to the assumed Tc, from the independent '
                          'per-temperature Ms values.',
           'Tc_fixed_K': TC_FIXED, 'windows': windows, 'tc_scan': tc_scan,
           'free_tc_baseline': free}
    json.dump(out, open(results_json('beta_window_crosscheck.json'), 'w'), indent=2)
    print(f"\nsaved {results_json('beta_window_crosscheck.json')}")


if __name__ == '__main__':
    main()
