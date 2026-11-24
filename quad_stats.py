"""Quadratic coarse-grainings f(s) = sum_jk J_jk s_j s_k: signal, and how much of it low moments explain.

For each statistic the exact distribution P_k = Pr[f(s) = k] is obtained by full enumeration.
Signal = TV(P, w), w = Haar mean (uniform over Fock states, exact).
Classical predictor of order p: tilt w to match the first p moments of f,
    P_hat_k ~ w_k exp(a_1 k + ... + a_p k^p).
The q-th moment of a quadratic f is a 2q-point photon-number correlator, computable classically in
polynomial time for fixed q, so a small residual TV(P, P_hat) means the signal is classically predictable.

Run:  python quad_stats.py        (~15 min; writes results/quad_stats.json)
"""
import glob, itertools, json, os, sys
import numpy as np
from scipy.optimize import minimize
from flatness import haar, probs, rng as haar_rng

rng = np.random.default_rng(1)


def occupations(n, m):
    outs = np.array(list(itertools.combinations_with_replacement(range(m), n)), dtype=np.int16)
    s = np.zeros((len(outs), m), dtype=np.float32)
    rows = np.arange(len(outs))
    for c in range(n):
        np.add.at(s, (rows, outs[:, c]), 1)
    return s


def statistics(s):
    """name -> integer value of the statistic on every outcome."""
    K, m = s.shape
    blocks = lambda r: np.stack([b.sum(1) for b in np.array_split(s, r, axis=1)], axis=1)
    J = np.triu(rng.integers(0, 2, (m, m)), 1).astype(np.float32)
    with np.errstate(all='ignore'):          # spurious matmul warnings from Accelerate
        rand = np.einsum('kj,kj->k', s @ J, s)
    out = {
        'linear: photons in first half': blocks(2)[:, 0],
        'blocks r=2: sum N_b^2': (blocks(2) ** 2).sum(1),
        'blocks r=4: sum N_b^2': (blocks(4) ** 2).sum(1),
        'collisions: sum s_j^2 (r=m)': (s ** 2).sum(1),
        'neighbours: sum s_j s_j+1': (s * np.roll(s, -1, axis=1)).sum(1),
        'random graph: sum_{j<k} J_jk s_j s_k': rand,
    }
    return {k: np.rint(v).astype(np.int64) for k, v in out.items()}


def tilt(w, x, mom):
    """Distribution ~ w exp(a . x^q) whose first len(mom) moments of x equal mom."""
    phi = np.stack([x ** q for q in range(1, len(mom) + 1)], axis=1)

    def dual(a):
        z = phi @ a
        z0 = z.max()
        e = w * np.exp(z - z0)
        return np.log(e.sum()) + z0 - a @ mom, phi.T @ (e / e.sum()) - mom

    a = minimize(dual, np.zeros(len(mom)), jac=True, method='BFGS', options={'gtol': 1e-10}).x
    e = w * np.exp(phi @ a - (phi @ a).max())
    return e / e.sum()


def self_check():
    w, x = np.array([.1, .2, .3, .4]), np.arange(4.)
    target = np.array([.4, .3, .2, .1])
    for p in (1, 2, 3):                       # tilt reproduces the requested moments; 3 moments fix 4 points
        q = tilt(w, x, np.array([(target * x ** k).sum() for k in range(1, p + 1)]))
        assert np.allclose([(q * x ** k).sum() for k in range(1, p + 1)], [(target * x ** k).sum() for k in range(1, p + 1)], atol=1e-7)
    assert np.allclose(q, target, atol=1e-6)


def sample(n, m, n_u):
    """Per interferometer and statistic: [number of values, signal TV, residual TV after 1, 2, 3 moments]."""
    s = occupations(n, m)
    stats = statistics(s)
    del s
    acc = {name: [] for name in stats}
    for _ in range(n_u):
        p = probs(haar(m), n)
        for name, f in stats.items():
            vals, inv = np.unique(f, return_inverse=True)
            w = np.bincount(inv) / len(f)
            P = np.bincount(inv, weights=p)
            x = (vals - (w * vals).sum()) / np.sqrt((w * vals ** 2).sum() - (w * vals).sum() ** 2)
            tv = lambda q: 0.5 * abs(P - q).sum()
            res = [tv(tilt(w, x, np.array([(P * x ** k).sum() for k in range(1, o + 1)]))) for o in (1, 2, 3) if len(vals) > o + 1]
            acc[name].append([len(vals), tv(w)] + res + [np.nan] * (3 - len(res)))
    return {name: np.array(a) for name, a in acc.items()}


def rows_from(n, m, acc):
    rows = []
    for name, a in acc.items():
        n_u = len(a)
        nv, sig, r1, r2, r3 = a.mean(0)
        rows.append(dict(n=n, m=m, n_u=n_u, stat=name, values=int(nv), signal=float(sig), sig_sem=float(a[:, 1].std() / np.sqrt(n_u)),
                         res1=float(r1 / sig), res2=float(r2 / sig), res3=float(r3 / sig),
                         res3_sem=float((a[:, 4] - r3 / sig * a[:, 1]).std() / (np.sqrt(n_u) * sig))))   # delta method for a ratio of means
        print(f"{n:2d} {m:3d}  {name:38s} values={int(nv):3d}  signal TV={sig:.4f}  residual/signal after 1,2,3 moments: "
              f"{r1 / sig:.2f} {r2 / sig:.2f} {r3 / sig:.2f}", flush=True)
    return rows


def measure(n, m, n_u):
    return rows_from(n, m, sample(n, m, n_u))


if __name__ == '__main__':
    self_check()
    os.makedirs('results', exist_ok=True)
    if '--batch' in sys.argv:      # python quad_stats.py --batch N M N_U SEED: raw data of one size, for running in parallel
        n, m, n_u, seed = map(int, sys.argv[2:6])
        rng.bit_generator.state = np.random.default_rng(1000 * n + m).bit_generator.state   # same random graph in every batch
        haar_rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state
        acc = sample(n, m, n_u)
        np.savez(f'results/quad_raw_n{n}_m{m}_s{seed}.npz', acc=np.stack(list(acc.values())), names=np.array(list(acc)))
    elif '--pool' in sys.argv:     # python quad_stats.py --pool N M: replace the rows of that size by the pooled batches
        n, m = map(int, sys.argv[2:4])
        raw = [np.load(f) for f in sorted(glob.glob(f'results/quad_raw_n{n}_m{m}_s*.npz'))]
        acc = {str(name): np.concatenate([r['acc'][i] for r in raw]) for i, name in enumerate(raw[0]['names'])}
        rows = json.load(open('results/quad_stats.json')) if os.path.exists('results/quad_stats.json') else []
        rows = [x for x in rows if (x['n'], x['m']) != (n, m)] + rows_from(n, m, acc)
        json.dump(sorted(rows, key=lambda x: (x['m'] != x['n'] ** 2, x['n'])), open('results/quad_stats.json', 'w'), indent=1)
    else:
        quick = '--quick' in sys.argv
        plan = [(3, 9, 200), (4, 16, 100), (5, 25, 40)] + ([] if quick else [(6, 36, 12)]) \
             + [(4, 8, 200), (5, 10, 100), (6, 12, 60), (7, 14, 30)] + ([] if quick else [(8, 16, 30)])
        rows = []
        for n, m, n_u in plan:
            print(f"\n## n={n}, m={m}, {n_u} unitaries")
            rows += measure(n, m, n_u)
        json.dump(rows, open('results/quad_stats.json', 'w'), indent=1)
