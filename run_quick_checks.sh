#!/bin/sh
# Quick end-to-end checks (under a minute). Leaves results/ untouched.
set -e
PY=${PYTHON:-python3}
echo "== self-checks: closed formulas against full enumeration =="
$PY -c "
import flatness, signal_decay, fine_bins, finer_bins, quad_stats
for mod in (flatness, signal_decay, fine_bins, finer_bins, quad_stats):
    mod.self_check(); print(' ', mod.__name__, 'passed')"
echo "== results rebuilt from the raw batches =="
$PY check_results.py
echo "== figures and derived numbers =="
( cd figures && $PY make_figures.py )
echo "ALL QUICK CHECKS PASSED"
