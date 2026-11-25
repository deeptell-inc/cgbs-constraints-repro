"""Does the signal of lexicographic (linear) binning survive at large photon number?

Bins are contiguous in lexicographic order with boundaries on the first digit, i.e. on the lowest
occupied mode i_1:  bin_j = {a_{j-1} <= i_1 < a_j}.  Then
    Pr[i_1 >= a] = Per( (U[a:, :n])^dagger U[a:, :n] )            (one n x n permanent, exact)
and its Haar mean is the counting fraction C(m-a+n-1, n) / C(m+n-1, n)   (Schur's lemma).
Signal = deviation of a bin probability from its Haar mean.  The same quantity for distinguishable
particles, prod_i sum_{j>=a} |U_ji|^2, tells how much of the signal is interference.

Run:  python signal_decay.py        (~6 min; writes results/signal_decay.json)
"""
import itertools, json, math, os, sys
import numpy as np
from flatness import haar, probs, rng


def permanent(A, chunk=1 << 15):
    """Glynn's formula, vectorised over the 2^(n-1) sign vectors."""
    n = len(A)
    total, bits = 0j, np.arange(n - 1)
    with np.errstate(all='ignore'):      # Accelerate's matmul raises spurious FP warnings on this machine
        for start in range(0, 1 << (n - 1), chunk):
            idx = np.arange(start, min(start + chunk, 1 << (n - 1)))
            delta = np.ones((len(idx), n))
            delta[:, 1:] = 1 - 2 * ((idx[:, None] >> bits) & 1)
            total += (delta[:, 1:].prod(1) * (delta @ A).prod(1)).sum()
    assert np.isfinite(total)
    return total / 2 ** (n - 1)


def tail_haar(n, m, a):
    return math.comb(m - a + n - 1, n) / math.comb(m + n - 1, n)


def boundaries(n, m, d):
    """a_1 < ... < a_{d'-1}: first-digit thresholds whose Haar-mean tails are closest to 1 - j/d."""
    tails = np.array([tail_haar(n, m, a) for a in range(m + 1)])
    return sorted({int(abs(tails - (1 - j / d)).argmin()) for j in range(1, d)} - {0, m})


def tails(U, n, cuts):
    """(boson, distinguishable) Pr[i_1 >= a] for each threshold a in cuts."""
    V = U[:, :n]
    with np.errstate(all='ignore'):   # same spurious matmul warnings as in permanent()
        grams = [V[a:].conj().T @ V[a:] for a in cuts]
    bos = [permanent(g).real for g in grams]
    dis = [(abs(V[a:]) ** 2).sum(0).prod() for a in cuts]
    return np.array(bos), np.array(dis)


def bins(t):                              # tail probabilities -> bin probabilities
    t = np.atleast_2d(t)
    full = np.hstack([np.ones((len(t), 1)), t, np.zeros((len(t), 1))])
    return np.squeeze(-np.diff(full, axis=1))


def self_check():
    U, n, m = haar(16), 4, 16             # against exact enumeration of all 3876 outcomes
    p = probs(U, n)
    first = np.array([o[0] for o in itertools.combinations_with_replacement(range(m), n)])
    for a in (1, 3, 7):
        b, _ = tails(U, n, [a])
        assert np.isclose(b[0], p[first >= a].sum(), rtol=1e-9), (a, b[0], p[first >= a].sum())
    x = np.array([tails(haar(9), 3, [2])[0][0] for _ in range(3000)])   # Haar mean = counting fraction
    assert abs(x.mean() - tail_haar(3, 9, 2)) < 4 * x.std() / math.sqrt(len(x))


def measure(n, m, d, n_u):
    cuts = boundaries(n, m, d)
    w = bins(np.array([tail_haar(n, m, a) for a in cuts]))
    B, D = map(np.array, zip(*(tails(haar(m), n, cuts) for _ in range(n_u))))
    dB = bins(B) - w                                  # boson deviation from exact Haar mean
    dD = bins(D) - bins(D).mean(0)                    # distinguishable, centred empirically
    diff = bins(B) - bins(D)
    sem = lambda x: float(x.std() / math.sqrt(n_u))   # standard error over interferometers
    row = dict(n=n, m=m, d=len(w), n_u=n_u, w_min=float(w.min()), w_max=float(w.max()),
               dev=float(abs(dB).max(1).mean()), dev_sem=sem(abs(dB).max(1)),
               dev_dist_sem=sem(abs(dD).max(1)), resid_sem=sem(abs(diff - diff.mean(0)).max(1)),
               rel=float((abs(dB) / w).max(1).mean()),
               dev_dist=float(abs(dD).max(1).mean()),
               resid=float(abs(diff - diff.mean(0)).max(1).mean()),   # fluctuation of the interference part P_boson - P_dist
               corr=float(np.corrcoef(dB.ravel(), dD.ravel())[0, 1]),
               mpb_same=float(((bins(B) / w).argmax(1) == (bins(D) / bins(D).mean(0)).argmax(1)).mean()))
    print(f"{n:3d} {m:4d} d={row['d']:2d} w=[{row['w_min']:.3f},{row['w_max']:.3f}]  dev={row['dev']:.4f}±{row['dev_sem']:.4f}"
          f"  rel={row['rel']:.3f}  dist dev={row['dev_dist']:.4f}  resid={row['resid']:.4f}  corr={row['corr']:.2f}  same MPB={row['mpb_same']:.2f}", flush=True)
    return row


if __name__ == '__main__':
    self_check()
    print('self-check passed')
    quick = '--quick' in sys.argv
    regimes = (('m = n^2, d = 4', lambda n: n * n, lambda n: 4),
               ('m = n^2, d = n', lambda n: n * n, lambda n: n),
               ('m = 2n,  d = 2', lambda n: 2 * n, lambda n: 2))
    os.makedirs('results', exist_ok=True)
    if '--replace' in sys.argv:      # python signal_decay.py --replace N N_U SEED: redo one size with a larger sample
        n, n_u, seed = map(int, sys.argv[2:5])
        rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state
        rows = json.load(open('results/signal_decay.json'))
        for label, mode_count, dfun in regimes:
            i = next(k for k, x in enumerate(rows) if x['regime'] == label and x['n'] == n)
            rows[i] = dict(regime=label, **measure(n, mode_count(n), dfun(n), n_u))
    else:
        plan = [(4, 200), (6, 200), (8, 200), (10, 200), (12, 100), (14, 60)] + ([] if quick else [(16, 100), (18, 60), (20, 30)])
        rows = []
        for label, mode_count, dfun in regimes:
            print(f"\n## {label}   (dev = mean over unitaries of max_j |P_j - w_j|)")
            rows += [dict(regime=label, **measure(n, mode_count(n), dfun(n), n_u)) for n, n_u in plan]
    for label, _, _ in regimes:
        r = [x for x in rows if x['regime'] == label and x['n'] >= 8]
        slope = np.polyfit(np.log([x['n'] for x in r]), np.log([x['dev'] for x in r]), 1)[0]
        slope_r = np.polyfit(np.log([x['n'] for x in r]), np.log([x['resid'] for x in r]), 1)[0]
        print(f"{label}: log-log slope vs n (n >= 8): dev {slope:.2f}, resid {slope_r:.2f}")
        [x for x in rows if x['regime'] == label][-1].update(slope_dev=float(slope), slope_resid=float(slope_r))
    json.dump(rows, open('results/signal_decay.json', 'w'), indent=1)
