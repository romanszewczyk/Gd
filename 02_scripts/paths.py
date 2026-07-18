# Author: Roman Szewczyk
# License: MIT License (see LICENSE file)

"""Repository paths, resolved relative to this file so scripts run from anywhere."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_NPZ = os.path.join(ROOT, '01_data', 'processed', 'gd_processed.npz')
RESULTS_JSON = os.path.join(ROOT, '03_results', 'json')
FIGURES = os.path.join(ROOT, '03_results', 'figures')
TABLES = os.path.join(ROOT, '03_results', 'tables')

for _d in (RESULTS_JSON, FIGURES, TABLES):
    os.makedirs(_d, exist_ok=True)


def results_json(name):
    return os.path.join(RESULTS_JSON, name)


def figure(name):
    return os.path.join(FIGURES, name)


def table(name):
    return os.path.join(TABLES, name)
