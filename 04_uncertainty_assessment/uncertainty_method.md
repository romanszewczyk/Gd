# Uncertainty assessment: step-by-step method

How the uncertainty of the global Jiles-Atherton model of gadolinium was identified and
assessed. The procedure follows the *Guide to the Expression of Uncertainty in Measurement*
(GUM, JCGM 100) and its Monte Carlo supplement (JCGM 101): identify the input quantities and
their standard uncertainties, propagate them through the model, and report combined standard
uncertainties and coverage intervals.

The scripts referenced below live in this folder; they import the forward model from
`../02_scripts` and read the published fit from `../03_results`, and they do not modify either.

---

## Step 0: What is the measurand?

The reported quantities are the twelve parameters of the global model, i.e. the smooth
temperature laws

$$M_s(T_K)=M_{s0}\,(1-T_K/T_c)^{\beta},\qquad a(T),\ k(T),\ c(T)\ \text{quadratic in }T\,[^{\circ}\mathrm{C}],$$

with $\alpha=0$. The order-parameter scalars $M_{s0}, T_c, \beta$ are the quantities for which a
marginal standard uncertainty is reported.

---

## Step 1: Identify the input uncertainties

**1A. Measurement-system uncertainty (Type B).** From the calibrated hysteresisgraph (see
`hysteresisgraph_accuracy.md`): a flux-density scale uncertainty and a field scale uncertainty,
$u_\mathrm{rel}(B)=0.1\,\%$, $u_\mathrm{rel}(H)=0.01\,\%$, both systematic (a single gain per
channel, common to every point and every temperature). Coded as `REL_U_B`, `REL_U_H`.

**1B. Identifiability uncertainty (Type A).** Jiles-Atherton identification is ill-conditioned:
$a$, $k$, $c$ trade off against one another and $a$ is tied to $M_s$ through the loop amplitude.
Quantified empirically (Step 2) from the scatter between the per-temperature independent fits
(`parameters_per_temperature.json`) and the global laws (`parameters_physical.json`).

---

## Step 2: The parameters are not independent (the key methodological point)

`correlation_analysis.py` forms, for each parameter $p\in\{a,k,c,M_s\}$ and temperature $T_i$,
the log-residual $L_p(T_i)=\ln\!\big(p_\text{independent}(T_i)/p_\text{law}(T_i)\big)$. The twelve
temperatures give twelve samples of the 4-vector $(L_a,L_k,L_c,L_{M_s})$; its centred covariance
`Sigma` (4x4, log space) carries the full correlation structure, and its diagonal is the
(squared) relative scatter used by a naive independent analysis. The empirical correlation matrix is

|        |  a    |  k    |  c    | $M_s$ |
|--------|------:|------:|------:|------:|
| a      |  1.00 | -0.35 | -0.11 |  0.92 |
| k      | -0.35 |  1.00 |  0.18 | -0.32 |
| c      | -0.11 |  0.18 |  1.00 | -0.03 |
| $M_s$  |  0.92 | -0.32 | -0.03 |  1.00 |

so $a$ and $M_s$ are strongly correlated and $a,k,c$ are mutually correlated at a moderate level.
Working in log space keeps every positive parameter positive under perturbation and makes the
independent model the exact special case `Sigma -> diag(Sigma)`.

---

## Step 3: Decide the level of propagation: global, not per-loop

Re-fitting a single loop for $(a,k,c,M_s)$ from slightly different starting points returns
$k$ and $c$ moving by several to tens of per cent at essentially constant fit quality; the
per-loop problem is degenerate, so propagating measurement uncertainty through it is
meaningless. The global fit is well posed (a local re-fit started from the optimum reported
here returns to it, with no parameter shifted by more than 0.4 %), so propagation is done at
the global level.

A subtlety found and fixed along the way: the forward model integrates an ODE with an adaptive
solver (`atol=1e-6`), so `B_sim` carries jitter at the microtesla level. Finite-difference
derivatives must use a step larger than this jitter (`rel_step = 1e-3`); the default
step of about $10^{-8}$ measures only solver noise. All derivatives below use the jitter-aware
step.

---

## Step 4: Propagate the measurement uncertainty (Component A)

`propagate_calibration.py` linearises the global weighted-least-squares fit about the optimum:
Jacobian $G=\partial B_\text{sim}/\partial x$; per-point weights $w=1/(12\,\mathrm{SST}_i)$;
estimator operator $K=(G^{\mathsf T}WG)^{-1}G^{\mathsf T}W$ solved in scaled parameters (condition
number of the scaled normal matrix $\approx1.1\times10^4$). A B-gain perturbs the data by
$\mathrm dB=g_B B_\text{meas}$ ($s_B=K B_\text{meas}$); an H-gain rescales the field axis
($s_H=-K\,\mathrm dB_\text{sim}/\mathrm dg_H$). The calibration uncertainty
$u_\mathrm{cal}(x_j)=\sqrt{(s_{B,j}u_\mathrm{rel}(B))^2+(s_{H,j}u_\mathrm{rel}(H))^2}$ is dominated
by the saturation amplitude ($u_\mathrm{cal}(M_{s0})\approx0.08\,\%$) and is below $0.05\,\%$
elsewhere. Cross-checked against a full non-linear re-fit of $\pm1\%$ gain-rescaled data
($\mathrm dM_{s0}/\mathrm dg_B\approx0.77$ from both the linearised operator and the non-linear
re-fit, an agreement to better than 1 %).

---

## Step 5: Propagate the identifiability uncertainty (Component B), correlated

`propagate_identifiability.py` runs a correlated Monte Carlo (JCGM 101; $N=10^5$ trials)
drawn from the joint distribution of Step 2. In each trial the twelve per-temperature parameters
are perturbed by a multivariate-lognormal factor $\exp(\delta)$, $\delta\sim N(0,\Sigma)$ drawn
jointly per temperature, the $M_s(T)$ power law is re-fitted by a bounded least-squares fit, and a
quadratic is fitted to the perturbed $k(T)$ to check whether it still decreases toward $T_c$.

**Result.** $M_{s0}=2.14\pm0.08\,\mathrm{MA/m}$ (3.6 %), $T_c=291.5\pm0.7\,\mathrm{K}$ (0.25 %),
$\beta=0.281\pm0.019$ (6.6 %); and $k(T)$ decreasing in more than 99 % of trials.

**Effect of the correlations.** The marginal uncertainty of each order-parameter scalar is set by
the saturation channel alone, so the reported $M_{s0},T_c,\beta$ are essentially insensitive to the
off-diagonal structure of $\Sigma$. The correlations matter only for quantities that combine the
parameters, where the per-loop scatter lies mostly along the degenerate direction. The strongly
correlated pair $(a,M_s)$ are the well-determined ones, while $(k,c)$ are only loosely determined;
the joint treatment is what establishes that the order-parameter quantities hold up.

---

## Step 6: Combine and report

`combine_and_report.py` forms $u_c=\sqrt{u_\mathrm{cal}^2+u_\mathrm{idn}^2}$ for each reported
quantity (the value column is the global-fit point estimate; the identifiability term is the
standard deviation of the per-trial re-fitted scalar) and writes
`uncertainty_summary.{json,csv}`, `uncertainty_budget.tex` and `parameter_correlations.tex`.
`plot_uncertainty.py` draws the $M_s(T)$ power law with its 95 % identifiability band.

**Reading of the budget.** Component A (measurement) is one to two orders of magnitude smaller
than Component B (identifiability) for every reported quantity. The order-parameter quantities and
$a(T)$ are well determined; the absolute levels of $k$ and $c$ are not, but the decreasing trend of
$k(T)$ toward $T_c$, the physically meaningful result, survives in more than 99 % of trials. The
measurement system is not the limiting factor; the identifiability of the model is.

---

## Step 7: Assumptions and limitations

- Instrument accuracies are taken directly as relative standard uncertainties (conservative;
  a rectangular-bound reading would divide them by $\sqrt3$).
- Calibration uncertainties are modelled as fully correlated gains; no independent point-wise
  random term is added (the recorded `u(H)` files are not in the processed dataset).
- The joint identifiability distribution is estimated from twelve temperatures, so `Sigma` itself
  carries sampling uncertainty; the conclusions use only its gross structure (which pairs are
  strongly vs weakly correlated), which is stable.
- No temperature-stability or demagnetising-correction uncertainty is budgeted. The
  manufacturer-stated stability of the circulator bath (0.005 K, PolyScience AP07R-40)
  implies a relative change of $M_s$ below 0.05 % even at the warmest loop
  ($\beta\,\delta T/(T_c\,\tau_{\min})\approx4\times10^{-4}$), so the omission is harmless.

---

## Reproduce

```bash
cd 04_uncertainty_assessment
python3 correlation_analysis.py
python3 propagate_calibration.py        # Component A  (~25 s)
python3 propagate_identifiability.py    # Component B, correlated Monte Carlo (about 2 min)
python3 combine_and_report.py
python3 plot_uncertainty.py
```

Inputs: `../01_data/processed/gd_processed.npz`, `../03_results/json/*.json`.
Outputs: `results/json/`, `results/tables/`, `results/figures/`.
