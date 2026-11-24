# cgbs-constraints-repro

Reproduction repository for the paper

> H. Wakaura, *Signal and interference constraints on cryptographic uses of
> coarse-grained boson sampling* (manuscript in preparation, 2026).

The paper formulates three conditions that a coarse-graining of a boson
sampling distribution must satisfy to support a cryptographic function (a
readable signal, a signal beyond low-order correlations, no classical
estimator) and tests published schemes against them. Every number and figure
in the paper is produced by the scripts here from the files in `results/`.

## Layout

```
flatness.py              Lemma 1 (flatness of permuted binning) and the reward test of the proof-of-work protocol
signal_decay.py          lexicographic bins with boundaries on the first entry: signal and interference part, 4 to 20 photons
finer_bins.py            lexicographic bins with boundaries on three entries: up to d = M^(3/2) bins, 6 to 16 photons
fine_bins.py             helper functions for finer_bins.py (two-entry boundaries)
quad_stats.py            quadratic statistics: signal and residual after matching low moments
gbs_hash_classical.py    classical evaluation of the Gaussian boson sampling hash from the covariance matrix
figures/make_figures.py  Figs. 2-4 and the derived numbers (exponents, closed-form collision probability)
results/*.json           the data behind the paper
results/*_raw_*          per-interferometer raw batches from which the JSON files are pooled
reproduce_all.sh         regenerates everything in results/ from scratch
run_quick_checks.sh      self-checks, rebuild of results/*.json from the raw batches, figures
check_results.py         used by run_quick_checks.sh
pool_commands.txt        the pooling commands, one per line
```

## Claim-to-script map

| Paper item | Script | Data | Sample |
|---|---|---|---|
| Lemma 1, Fig. 2(a): variance of a randomly permuted bin; collision probability | `flatness.py --batch`, `--pool` | `results/flatness.json` (`flatness`) | 100 to 200 Haar-random interferometers per size, 40 permutations each |
| Fig. 2(b): a uniform sampler passes the reward test | `flatness.py --reward-batch`, `--reward-pool` | `results/flatness.json` (`reward`) | 100 interferometers x 100 trials per case |
| Sec. 4: the hash of three-mode correlations is classically exact | `gbs_hash_classical.py` | `results/gbs_hash.json` | 16, 64, 256 modes |
| Fig. 3(a): signal and interference part of first-entry bins | `signal_decay.py`, `--replace` | `results/signal_decay.json` | 100 to 200 interferometers per size |
| Fig. 3(b), (c): interference share and product-formula ranking for fine bins | `finer_bins.py --batch`, `--pool` | `results/finer_bins.json` | 100 to 400 interferometers per size |
| Fig. 4, Appendix A: quadratic statistics | `quad_stats.py --batch`, `--pool` | `results/quad_stats.json` | 100 to 800 interferometers per size |
| Power-law exponents; closed-form collision probability (Mhiri et al., arXiv:2604.14323) against the measured values | `figures/make_figures.py` | `figures/numbers.json` | derived from the files above |

All probabilities are exact: full enumeration of the output distribution
(`flatness.py`, `quad_stats.py`) or permanents evaluated with Glynn's formula
(`signal_decay.py`, `finer_bins.py`). The only randomness is the choice of
Haar-random interferometers, permutations and, in the reward test, samples;
every run is seeded.

## Quick start (under a minute)

```sh
pip install -r requirements.txt
./run_quick_checks.sh
```

This runs

1. the self-checks of each script (the closed formulas for bin probabilities
   against full enumeration at small sizes, the flatness lemma against all 252
   bipartitions of a 10-outcome distribution, the Haar mean against counting);
2. the classical evaluation of the hash (bits compared with the shipped ones);
3. a rebuild of `results/flatness.json`, `results/finer_bins.json` and
   `results/quad_stats.json` from the raw batches in a temporary directory,
   compared with the shipped files;
4. the figures (`figures/fig2.pdf`, `fig3.pdf`, `fig4.pdf`) and
   `figures/numbers.json`.

## Full reproduction (about 2.5 hours on a 16-core machine)

```sh
./reproduce_all.sh
```

The script lists every batch with its size, count and seed, four processes
per group, then pools the batches and draws the figures. The seeds are those
of the shipped data, so the output should agree with `results/` up to
floating-point differences between linear-algebra libraries. The longest
group is the 16-photon run of `finer_bins.py` (about 50 minutes); the
enumeration at 6 photons in 36 modes needs about 1 GB of memory per process.

Running a script without arguments (for example `python flatness.py`) starts
a smaller single-process run with seed 0 and overwrites the corresponding
file in `results/`. Only `signal_decay.py` is meant to be used that way, as
the first step of its reproduction; for the others use the batch and pool
commands of `reproduce_all.sh`.

## Requirements

Python 3.10 or later with NumPy, SciPy and Matplotlib; The Walrus is needed
only for `gbs_hash_classical.py`. The shipped data were produced with Python
3.13.9, NumPy 2.2.6, SciPy 1.17.1, Matplotlib 3.10.6 and The Walrus 0.22.0.

## Data format

- `results/flatness.json`: `flatness` holds one row per size (`n` photons,
  `m` modes, `K` outcomes, `d` bins, `n_u` interferometers) with the normalised
  collision probability `S2_times_K` and its standard error, the mean largest
  deviation of a bin probability from `1/d` under lexicographic order
  (`dev_lex`) and under random permutations (`dev_rnd`) with their standard
  errors over interferometers (`dev_lex_sem`, `dev_rnd_sem`), and the measured
  variance divided by the lemma (`var_ratio`). `reward` holds, per case, the
  fraction of trials in which an honest (`boson`) and a uniform miner are
  rewarded, with the range over interferometers.
- `results/signal_decay.json`: per regime and size, the mean largest deviation
  of a bin probability from its Haar mean for bosons (`dev`) and for
  distinguishable particles (`dev_dist`), the fluctuation of their difference
  (`resid`), their correlation and the agreement of the most probable bin.
- `results/finer_bins.json`: per size and bin number, the r.m.s. ratio of the
  interference part to the signal (`inter_frac`) with its standard error
  (`inter_sem`, delta method over interferometers; it does not include the
  finite sample of 1000 interferometers behind the distinguishable-particle
  Haar mean, which raises `inter_frac` by 0.001 to 0.003, as
  `python finer_bins.py --pool N 8000` run in a copy shows) and the fractions of
  interferometers for which the product formula ranks the true most probable
  bin first (`mpb_same`), in its top three and in its top 1%.
- `results/quad_stats.json`: per size and statistic, the total variation
  distance from the Haar mean (`signal`) and the residual after matching one,
  two and three moments, divided by the signal (`res1`, `res2`, `res3`), with
  the standard error of the last ratio (`res3_sem`, delta method).
- `results/gbs_hash.json`: hash bits and wall-clock time per number of modes.

`SHA256SUMS` lists the checksums of the files in `results/`.

## Licence

MIT, see `LICENSE`. If you use this repository, please cite the paper; see
`CITATION.cff`.
