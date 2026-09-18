# Model-free hysteresis observables of polycrystalline gadolinium up to the Curie point

Update of the repository *Temperature-dependent Jiles-Atherton model of gadolinium*
(`Gd_isotropic_repo5`, MIT License, Roman Szewczyk; measurements Michał Nowicki). It adds,
without changing anything in that repository:

1. **Model-free loop observables of the main series** (twelve temperatures, -38 to +15 °C):
   coercive field, remanence, loss per cycle, tip magnetization and the loss-averaged pinning
   field `k_eff = W/(4 mu0 M_amp)`, compared with the loops of the global Jiles-Atherton fit.
2. **A supplementary loop series through the Curie point** (6.06 kA/m peak field, -30 to
   +20 °C, plus one 54.8 kA/m check loop at -30 °C) and its observables.
3. **A model-adequacy test**: per-loop Jiles-Atherton re-fits with the low-field window
   weighted equally, with and without the inter-domain coupling.

## Main results

- The global fit reproduces the approach to saturation but not the hysteretic part of the
  loops: modelled coercive fields 264 to 66 A/m against 520-640 A/m measured (473 A/m at
  +15 °C). A change of the pinning parameter k by a factor of three alters the full-loop
  objective by 1e-4, and the low-field-weighted re-fit reaches only 60-66 % of the measured
  coercive field with or without a free alpha. The single-pinning-parameter isotropic model
  cannot be wide at H = 0 and closed by 10 kA/m at once; its k(T) and c(T) are properties of the
  temperature-constrained fit, not measurements.
- Model-free: the loss per cycle of the drift-free 10 kA/m loops falls 969 -> 259 J/m3
  (3.7-fold) in proportion to the magnetization swing (867 -> 249 kA/m); the loss-averaged
  pinning field is constant (207-284 A/m); the coercive field stays within 12 % of 580 A/m to
  +5 °C and falls by a quarter to +15 °C; the remanence falls 4.5-fold. The loss extrapolates
  to zero at Tc = 292.6 K (292.0-293.1 K), consistent with Tc = 291.5 ± 0.7 K of the global fit
  and 292.2 ± 0.4 K of the tip-magnetization power law (beta_tip = 0.335 ± 0.008).
- Through the Curie point (supplementary series): the loop is still open at 17.5 °C (coercive
  field 402 A/m, loss 149 J/m3) and closed at 20 °C (branch separation below 0.1 mT), so
  hysteresis is extinguished between 290.65 and 293.15 K; the pinning fields collapse only in
  the last two kelvin, after the magnetization has fallen fourfold. The supplementary loops are
  systematically 10-20 % wider than the main-series loops at the same temperature (offset of
  that run); the series is used for the temperature dependence, not for absolute values.

## Layout

```
data/
  gd_processed.npz                     main series: 55, 10 and 0.4 kA/m loops at 12 temperatures
                                       (allow_pickle=True; temperatures, major_H/B, mid_H/B, minor_H/B)
  gd_supplementary_loops_near_Tc.csv   supplementary series: T_C; nominal peak field; branch; i; H; B
global_fit/
  parameters_physical.json             per-temperature parameters and B_sim of the global fit (input)
  physical_model_coeffs.json           the twelve global parameters (input)
code/
  ja_model.py                          isotropic Jiles-Atherton model (copy of the original)
  loop_observables.py                  observables of the main series -> results/json/loop_observables.json
  near_tc_loops.py                     supplementary series -> results/json/near_tc_loops.json
  plot_figures.py                      six square figures -> results/figures/
  ja_guarded.py, lowfield_refit_test.py  model-adequacy re-fit -> results/pilot_refit/pilot_results.json
  run_all.sh                           runs everything (add --with-refit for the re-fit, minutes)
results/
  json/, figures/, pilot_refit/
```

## Reproduction

```bash
pip install -r requirements.txt       # numpy, scipy, matplotlib
./code/run_all.sh                     # seconds
./code/run_all.sh --with-refit        # + model-adequacy re-fit (minutes, parallel)
```

## Notes on the evaluation

- Each main-series record holds an initial magnetization curve followed by one full cycle;
  the cycle is isolated from the first field maximum to the next. Coercive field: mean |H| of
  the two B = 0 crossings; remanence: mean |B| of the two H = 0 crossings; loss per cycle:
  `|oint B dH|` (path integral; an open-polygon shoelace sum is origin-dependent and wrong).
- The 55 kA/m records do not close (fluxmeter drift up to 20 mT per cycle), which adds a
  spurious area comparable to the loop area; their loss is therefore not used. The 10 kA/m
  records close to better than 4 mT and give the same area with and without detrending; the
  supplementary records close to better than 0.5 mT.
- Tc of the loss power law is obtained by a chi-square profile in Tc (log-linear fit of
  A and x for each Tc on a grid), because the three-parameter fit is ill-conditioned this close
  to Tc.
- In the model-adequacy re-fit the Jiles-Atherton solver is wrapped with a 3 s wall-time cap
  per loop evaluation (`ja_guarded.py`, identical physics): with a free alpha the search visits
  the corner a < 1 kA/m, k < 50 A/m, alpha ~ 3e-3, where the equation becomes singular and one
  evaluation takes minutes.

## License

MIT License (see `LICENSE`). Author: Roman Szewczyk. Measurement data: Michał Nowicki.
