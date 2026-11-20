#!/bin/sh
# Regenerates every file in results/ from scratch with the seeds of the shipped data.
# About 2.5 hours on a 16-core machine (four processes per group); the 16-photon group of finer_bins.py takes about 50 minutes.
set -e
PY=${PYTHON:-python3}


# Fig. 3(b), (c): lexicographic bins on three entries
for s in 1001 1002 1003 1004; do $PY finer_bins.py --batch 10 50 $s & done; wait
for s in 1201 1202 1203 1204; do $PY finer_bins.py --batch 12 25 $s & done; wait
for s in 1401 1402 1403 1404; do $PY finer_bins.py --batch 14 25 $s & done; wait
for s in 1601 1602 1603 1604; do $PY finer_bins.py --batch 16 25 $s & done; wait
for s in 601 602 603 604; do $PY finer_bins.py --batch 6 100 $s & done; wait
for s in 801 802 803 804; do $PY finer_bins.py --batch 8 75 $s & done; wait

# Fig. 2(a): flatness of permuted binning
for s in 30901 30902 30903 30904; do $PY flatness.py --batch 3 9 50 $s & done; wait
for s in 41601 41602 41603 41604; do $PY flatness.py --batch 4 16 50 $s & done; wait
for s in 52501 52502 52503 52504; do $PY flatness.py --batch 5 25 25 $s & done; wait
for s in 61201 61202 61203 61204; do $PY flatness.py --batch 6 12 25 $s & done; wait
for s in 63601 63602 63603 63604; do $PY flatness.py --batch 6 36 10 $s & done; wait
for s in 63605 63606 63607 63608; do $PY flatness.py --batch 6 36 15 $s & done; wait
for s in 71401 71402 71403 71404; do $PY flatness.py --batch 7 14 25 $s & done; wait
for s in 81601 81602 81603 81604; do $PY flatness.py --batch 8 16 10 $s & done; wait
for s in 81605 81606 81607 81608; do $PY flatness.py --batch 8 16 15 $s & done; wait

# Fig. 4: quadratic statistics
for s in 309001 309002 309003 309004; do $PY quad_stats.py --batch 3 9 200 $s & done; wait
for s in 416001 416002 416003 416004; do $PY quad_stats.py --batch 4 16 100 $s & done; wait
for s in 408001 408002 408003 408004; do $PY quad_stats.py --batch 4 8 200 $s & done; wait
for s in 510001 510002 510003 510004; do $PY quad_stats.py --batch 5 10 100 $s & done; wait
for s in 525001 525002 525003 525004; do $PY quad_stats.py --batch 5 25 50 $s & done; wait
for s in 612001 612002 612003 612004; do $PY quad_stats.py --batch 6 12 50 $s & done; wait
for s in 636001 636002 636003 636004; do $PY quad_stats.py --batch 6 36 25 $s & done; wait
for s in 714001 714002 714003 714004; do $PY quad_stats.py --batch 7 14 50 $s & done; wait
for s in 816001 816002 816003 816004; do $PY quad_stats.py --batch 8 16 25 $s & done; wait

# Fig. 2(b): reward test
for s in 91641 91642 91643 91644; do $PY flatness.py --reward-batch 4 16 4 25 100 $s & done; wait
for s in 92541 92542 92543 92544; do $PY flatness.py --reward-batch 5 25 4 25 100 $s & done; wait
for s in 93661 93662 93663 93664; do $PY flatness.py --reward-batch 6 36 16 10 100 $s & done; wait
for s in 93665 93666 93667 93668; do $PY flatness.py --reward-batch 6 36 16 15 100 $s & done; wait
for s in 93641 93642 93643 93644; do $PY flatness.py --reward-batch 6 36 4 10 100 $s & done; wait
for s in 93645 93646 93647 93648; do $PY flatness.py --reward-batch 6 36 4 15 100 $s & done; wait

# pool the batches into results/*.json
$PY finer_bins.py --pool 10
$PY finer_bins.py --pool 12
$PY finer_bins.py --pool 14
$PY finer_bins.py --pool 16
$PY finer_bins.py --pool 6
$PY finer_bins.py --pool 8
$PY flatness.py --pool 3 9
$PY flatness.py --pool 4 16
$PY flatness.py --pool 5 25
$PY flatness.py --pool 6 12
$PY flatness.py --pool 6 36
$PY flatness.py --pool 7 14
$PY flatness.py --pool 8 16
$PY quad_stats.py --pool 3 9
$PY quad_stats.py --pool 4 16
$PY quad_stats.py --pool 4 8
$PY quad_stats.py --pool 5 10
$PY quad_stats.py --pool 5 25
$PY quad_stats.py --pool 6 12
$PY quad_stats.py --pool 6 36
$PY quad_stats.py --pool 7 14
$PY quad_stats.py --pool 8 16
$PY flatness.py --reward-pool 4 16 4 100
$PY flatness.py --reward-pool 5 25 4 100
$PY flatness.py --reward-pool 6 36 16 100
$PY flatness.py --reward-pool 6 36 4 100

# Fig. 3(a): first-entry bins (full run, then the three sizes with enlarged samples)
$PY signal_decay.py
$PY signal_decay.py --replace 14 100 14010
$PY signal_decay.py --replace 18 100 1801
$PY signal_decay.py --replace 20 100 2001

# Sec. 4: classical evaluation of the Gaussian boson sampling hash
$PY gbs_hash_classical.py

# figures and derived numbers
( cd figures && $PY make_figures.py )
