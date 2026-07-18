# Uncertainty assessment

Propagation of measurement and identifiability uncertainty into the global
Jiles-Atherton model of gadolinium (the model fitted in `../02_scripts`, results
in `../03_results`, described in the accompanying manuscript submitted to
Physica B: Condensed Matter).

The assessment follows the *Guide to the Expression of Uncertainty in Measurement*
(GUM, JCGM 100) and its Monte Carlo supplement (JCGM 101): identify the input
quantities and their standard uncertainties, propagate them through the model,
and report combined standard uncertainties and coverage intervals.

## The parameters are not independent

The defining feature of this assessment is that the four Jiles-Atherton
parameters `(a, k, c, Ms)` are not statistically independent. For a single
loop the field scale `a`, the pinning `k` and the reversibility `c` trade off
against one another (the ill-conditioning of JA identification), and `a` is tied
to `Ms` through the loop amplitude. The joint distribution is estimated
empirically (`correlation_analysis.py`) from the log-residuals
`ln(independent fit / global law)` of the twelve per-temperature fits about the
smooth global laws; its centred covariance `Sigma` carries the full correlation
structure. The strongest correlations are `a`-`Ms` (approx. +0.9) and `a`-`k`
(approx. -0.35).

The joint distribution is used for two reasons. The marginal uncertainty of each
temperature law (`Ms0, Tc, beta`, and the `a, k, c` quadratics) is essentially
unchanged by the correlation, because each law is fitted from a single parameter
channel. The dependency matters instead for any quantity that combines the
parameters: the per-loop scatter lies mostly along the degenerate direction that
leaves the loop almost unchanged, so independent draws would overstate the spread
of such combined quantities. Only the joint treatment can establish which of the
two cases applies, and here it shows the order-parameter quantities hold up.

## Two contributions

A. Measurement-system (calibration) uncertainty, Type B.
The loops were recorded with the calibrated ferrograph/hysteresisgraph system of
Charubin, Nowicki, Marusenkov, Szewczyk *et al.*, *JAMRIS* **12**(3), 2018
calibrated against certified standards to a relative accuracy of 0.1 % for B
and 0.01 % for H. A calibration accuracy is a scale (gain) uncertainty: a single
unknown factor multiplies every point of every loop, fully correlated across all
twelve temperatures. Propagated through the well-posed global fit by the GUM law
of propagation of uncertainty, it rescales the axes without changing the loop
shape, so it appears almost entirely in the saturation amplitude `Ms0`
(approx. 0.08 %) and is negligible for `Tc`, `beta` and the parameter trends.

B. Identifiability uncertainty, Type A.
The dominant contribution, propagated by a correlated Monte Carlo (JCGM 101,
1e5 trials): in each trial the twelve per-temperature `(a, k, c, Ms)` are
perturbed by a multivariate-lognormal factor `exp(delta)`,
`delta ~ N(0, Sigma)` drawn jointly from the empirical covariance, and the
`Ms(T)` power law is re-fitted. The spread of the re-fitted `Ms0, Tc, beta` is
the identifiability contribution to those quantities, and the fraction of trials
preserving the decreasing `k(T)` trend is recorded.

The combined standard uncertainty is `u_c = sqrt(u_cal^2 + u_idn^2)`.

## Why per-loop re-fitting is not used for propagation

Re-fitting a single loop from slightly different starting points returns `(a,k,c)`
differing by several to tens of per cent at constant fit quality, which is the
degeneracy the global fit removes. The global fit is well posed, so propagation is
done at the global level. See the module docstrings for the numerical checks
(solver-jitter-aware finite-difference steps, condition number of the scaled
normal matrix, linear-vs-nonlinear cross-check of the gain sensitivities).

## Files

| Script | Purpose |
|--------|---------|
| `uncertainty_model.py` | Input uncertainties, loop loading, global forward model `B_sim(x)`, Jacobian `G`, the WLS estimator operator `K`. |
| `correlation_analysis.py` | The empirical covariance `Sigma` and correlation matrix of `(a, k, c, Ms)`. Writes `results/json/parameter_correlations.json`. |
| `propagate_calibration.py` | Component A. Writes `results/json/calibration_uncertainty.json`. |
| `propagate_identifiability.py` | Component B (correlated Monte Carlo over the power-law fit). Writes `results/json/identifiability_uncertainty.json`. |
| `combine_and_report.py` | Combines A+B. Writes `uncertainty_summary.{json,csv}`, `uncertainty_budget.tex`, `parameter_correlations.tex`. |
| `plot_uncertainty.py` | `Ms_powerlaw_uncertainty.jpg`: the `Ms(T)` power law with its 95 % band. |

```bash
cd 04_uncertainty_assessment
python3 correlation_analysis.py
python3 propagate_calibration.py        # ~25 s (one global Jacobian)
python3 propagate_identifiability.py    # correlated Monte Carlo (about 2 min)
python3 combine_and_report.py
python3 plot_uncertainty.py
```

The scripts import `ja_model.py` and `global_physical_ja_fit.py` from
`../02_scripts` and read the published results from `../03_results`; they do not
modify either.

See `uncertainty_method.md` for the full step-by-step method.
