# Accuracy of the hysteresisgraph measurement system

This document describes the accuracy of the measurement system used to record the
gadolinium $B(H)$ hysteresis loops, and explains how its accuracy figures enter the
uncertainty assessment (`uncertainty_method.md`). The system and its calibration are
documented in:

> T. Charubin, M. Nowicki, A. Marusenkov, R. Szewczyk, A. Nosenko, V. Kyrylchuk,
> *Mobile Ferrograph System for Ultrahigh Permeability Alloys*, Journal of Automation,
> Mobile Robotics and Intelligent Systems **12**(3), 40-42 (2018),
> DOI: 10.14313/JAMRIS_3-2018/16.

## 1. The instrument

The loops are measured with a computer-controlled ferrograph / hysteresisgraph (the
"Blacktower Ferrograph System"). Its measuring chain is:

| Block | Device | Role |
|-------|--------|------|
| Control / acquisition | PC with NI PCI-6221 DAQ card | generates the excitation waveform, digitises the signals |
| Field excitation | KEPCO BOP 36-6M voltage/current (U/I) converter, magnetizing coil/rod | drives the magnetizing current that sets $H$ |
| Flux measurement | Lakeshore 480 fluxmeter fed by the measurement (search) coil | integrates the induced voltage to give the flux, hence $B$ |
| Integrity | continuity tester | verifies input/output separation (guards against insulation breakdown shorting the coils) |

The magnetizing field $H$ is obtained from the magnetizing current and the coil/rod
geometry (a straight magnetizing rod is treated as a single-turn coil); the flux density
$B$ is obtained from the integrated search-coil voltage and the sample cross-section. Both
$H$ and $B$ are thus derived quantities whose accuracy is fixed by the calibration of the
current and flux channels and of the geometric constants.

## 2. Calibration and quoted accuracy

> *"Calibration and adjustment of the system were carried out with certified standards,
> allowing for 0.1 % accuracy in measurement of flux density B, and 0.01 % accuracy in
> measurement of magnetizing field H."* (Charubin et al., 2018)

The headline accuracy figures are therefore

| Quantity | Relative accuracy | Absolute value at the loop tip (major loop) |
|----------|-------------------|---------------------------------------------|
| Flux density $B$ | 0.1 % | $\approx 1.6\ \mathrm{mT}$ at $B_\mathrm{tip}\approx1.65\ \mathrm{T}$ |
| Magnetizing field $H$ | 0.01 % | $\approx 5.5\ \mathrm{A/m}$ at $H_\mathrm{tip}=55\ \mathrm{kA/m}$ |

Two points matter for how these figures are used:

1. They originate from a calibration against certified standards. A calibration fixes
   the scale factor (gain) of each channel: the flux channel (fluxmeter + integrator +
   coil area) and the field channel (current source + coil constant). The quoted accuracy
   is therefore predominantly a gain (scale) uncertainty, not random point-to-point
   noise.

2. A gain uncertainty is systematic and correlated. A single unknown scale factor
   multiplies every recorded point of a loop, and the same instrument records every
   temperature, so the same scale factor applies to all twelve loops:
   $$B_\mathrm{meas}=(1+g_B)\,B_\mathrm{true},\qquad H_\mathrm{meas}=(1+g_H)\,H_\mathrm{true},$$
   with relative standard uncertainties $u(g_B)=0.1\,\%$ and $u(g_H)=0.01\,\%$. This
   correlation is essential to the propagation: it means the calibration error rescales
   the loops without distorting their shape, so it propagates into amplitude-type
   parameters (the saturation magnetization) but not into shape-type parameters (the
   critical exponent $\beta$, the Curie temperature $T_c$, or the temperature trends of
   the parameters).

The field channel is ten times more accurate than the flux channel (0.01 % vs 0.1 %),
reflecting that the magnetizing current and coil geometry are easier to tie to standards
than the integrated flux.

## 3. Interpretation as a standard uncertainty (GUM)

The quoted "accuracy" is converted to a standard uncertainty for propagation. We adopt the
conservative reading and take the percentages directly as relative standard
uncertainties:
$$u_\mathrm{rel}(B)=1.0\times10^{-3},\qquad u_\mathrm{rel}(H)=1.0\times10^{-4}.$$
(If the accuracy were instead interpreted as the half-width $a$ of a rectangular
distribution, the GUM Type B rule $u=a/\sqrt{3}$ would make the standard uncertainties
smaller by a factor $\sqrt{3}\approx1.73$; our choice is therefore an upper bound and does
not understate the measurement uncertainty.) These two numbers are coded as `REL_U_B` and
`REL_U_H` in `uncertainty_model.py`.

## 4. Other accuracy-relevant features of the system

The cited paper documents several design measures that protect the accuracy; they are
listed here for completeness, although their detailed metrological budget is not separately
quantified in the source:

- External-field shielding. Three-axis Helmholtz coils with a magnetometer cancel the
  ambient field (including the Earth's $\sim 40\ \mu\mathrm{T}$) to below $0.1\ \mu\mathrm{T}$.
  This keeps the applied field equal to the intended field, which matters most for
  high-permeability, low-coercivity materials.
- Straight magnetizing rod. For low fields a straight rod (equivalent to a single-turn
  coil) replaces a wound coil, giving better field uniformity and avoiding a very-low-current
  regime in which the U/I converter would add noise and error.
- Thermal separation. The sample is separated from the current-carrying rod, so
  resistive self-heating of the magnetizing element does not bias the measurement. This is
  relevant here, because the gadolinium loops are recorded at controlled temperatures from
  $-38$ to $+15\,^{\circ}\mathrm{C}$, close to the Curie point where $M_s$ is strongly
  temperature-dependent.

## 5. What feeds the uncertainty budget

Only the two calibrated accuracy figures are used as quantitative inputs to the
measurement (Type B) part of the uncertainty assessment:

| Input quantity | Symbol | Standard uncertainty | Character |
|----------------|--------|----------------------|-----------|
| Flux-density scale | $g_B$ | $0.1\,\%$ (relative) | systematic, common to all loops |
| Field scale | $g_H$ | $0.01\,\%$ (relative) | systematic, common to all loops |

Not used (because the data are not available in this repository):

- Recorded point-wise uncertainty `u(H)` files. The raw-data description
  (`../01_data/raw_data/experimental_description.txt`) notes that each measurement directory
  contains a companion `u(H)` file with point-wise uncertainty data. These per-point files
  are not part of the processed dataset (`../01_data/processed/gd_processed.npz`), so a
  per-point random term is not propagated; if those files are added, the method in
  `uncertainty_method.md` accommodates an extra independent (random) component directly.
- Temperature uncertainty. The refrigerated circulator bath (PolyScience AP07R-40, 7 L)
  has a manufacturer-stated stability of 0.005 K. Its effect enters through $M_s(T)$ and
  is below 0.05 % of $M_s$ even at the warmest loop, so it is not separately budgeted.

Because the calibrated accuracy is small and systematic, the measurement system turns out
not to be the factor that limits the extracted parameters; the identifiability of the
Jiles-Atherton model is (see `uncertainty_method.md`, Step 6).
