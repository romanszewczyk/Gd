#!/usr/bin/env bash
# Reproduce every result of this repository (seconds): loop observables of the main series,
# supplementary series through the Curie point, figures. The model-adequacy re-fit
# (lowfield_refit_test.py, minutes) is run only with --with-refit.
set -e
cd "$(dirname "$0")"
python3 loop_observables.py
python3 near_tc_loops.py
python3 plot_figures.py
if [ "$1" = "--with-refit" ]; then python3 lowfield_refit_test.py; fi
echo "done"
