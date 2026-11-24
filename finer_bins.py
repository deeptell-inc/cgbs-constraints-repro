"""Lexicographic bins with boundaries on the first three digits: d up to ~m^1.5.

Tail probability of a boundary (x1 <= x2 <= x3), with
    P0(a)       = Pr[all photons in modes >= a]
    P1(x; a)    = Pr[one photon in mode x, the rest in modes >= a]
    P2(x; a)    = Pr[two photons in mode x, the rest in modes >= a]
    P11(x,y; a) = Pr[one photon in x, one in y, the rest in modes >= a]:
    x1 = x2 = x3 :  P0(x1)
    x1 < x2 = x3 :  P0(x1+1) + P1(x1; x2)
    x1 = x2 < x3 :  P0(x1+1) + P1(x1; x1+1) + P2(x1; x3)
    x1 < x2 < x3 :  P0(x1+1) + P1(x1; x2+1) + P11(x1,x2; x3)
For bosons these are coefficients of Per(V^dag (Pi_{>=a} + z1 Pi_x + z2 Pi_y) V), read off inside
Glynn's sum; for distinguishable particles, of prod_l (q_l(a) + z1 w_xl + z2 w_yl).

Run:  python finer_bins.py       (~1 h; writes results/finer_bins.json)
More interferometers for one size, in parallel:  python finer_bins.py --batch N N_U SEED  (several seeds),
then  python finer_bins.py --pool N.
"""
import glob, itertools, json, math, os, sys, time
import numpy as np
from flatness import probs, rng
from signal_decay import permanent
from fine_bins import haar_cols, bins, merge


def glynn_coefs(A, c1, c2=None, chunk=1 << 13):
    """([z1], [z1^2]) of Per(A + z1 c1*c1^T), or [z1 z2] of Per(A + z1 c1*c1^T + z2 c2*c2^T) if c2 is given."""
    n = len(A)
    bits, acc = np.arange(n - 1), np.zeros(2, complex)
    with np.errstate(all='ignore'):      # spurious matmul warnings from Accelerate
        for start in range(0, 1 << (n - 1), chunk):
            idx = np.arange(start, min(start + chunk, 1 << (n - 1)))
            delta = np.ones((len(idx), n))
            delta[:, 1:] = 1 - 2 * ((idx[:, None] >> bits) & 1)
            sign, r = delta[:, 1:].prod(1), delta @ A
            a = (delta @ c1.conj())[:, None] * c1
            e0, e1, e2 = np.ones(len(idx), complex), 0, 0
            if c2 is None:
                for j in range(n):
                    e2, e1, e0 = e2 * r[:, j] + e1 * a[:, j], e1 * r[:, j] + e0 * a[:, j], e0 * r[:, j]
                acc += (sign * e1).sum(), (sign * e2).sum()
            else:
                b, f1 = (delta @ c2.conj())[:, None] * c2, 0      # e1: z1, f1: z2, e2: z1 z2
                for j in range(n):
                    e2 = e2 * r[:, j] + e1 * b[:, j] + f1 * a[:, j]
                    e1, f1, e0 = e1 * r[:, j] + e0 * a[:, j], f1 * r[:, j] + e0 * b[:, j], e0 * r[:, j]
                acc[0] += (sign * e2).sum()
    assert np.isfinite(acc).all()
    return acc.real / 2 ** (n - 1)


def tails_boson(V, cuts):
    m = len(V)
    gram, p0, p1 = {}, {}, {}

    def G(a):
        if a not in gram:
            with np.errstate(all='ignore'):      # spurious matmul warnings from Accelerate
                gram[a] = V[a:].conj().T @ V[a:]
        return gram[a]

    def P0(a):
        if a not in p0:
            p0[a] = permanent(G(a)).real if a < m else 0.0
        return p0[a]

    def P1(x, a):
        if (x, a) not in p1:
            p1[x, a] = glynn_coefs(G(a), V[x])[0] if a < m else 0.0
        return p1[x, a]

    out = []
    for x1, x2, x3 in cuts:
        if x1 == x2 == x3:
            out.append(P0(x1))
        elif x2 == x3:
            out.append(P0(x1 + 1) + P1(x1, x2))
        elif x1 == x2:
            out.append(P0(x1 + 1) + P1(x1, x1 + 1) + glynn_coefs(G(x3), V[x1])[1])
        else:
            out.append(P0(x1 + 1) + P1(x1, x2 + 1) + glynn_coefs(G(x3), V[x1], V[x2])[0])
    return np.array(out)


def tails_dist(V, cuts):
    """Same events for distinguishable particles, vectorised over boundaries."""
    m, n = V.shape
    w = np.vstack([abs(V) ** 2, np.zeros((1, n))])
    q = np.vstack([np.cumsum(w[:m][::-1], axis=0)[::-1], np.zeros((1, n))])     # q[a, l] = sum_{k >= a} w[k, l]
    x1, x2, x3 = np.asarray(cuts).T
    k0, k1, k2, k3 = (x1 == x2) & (x2 == x3), (x1 < x2) & (x2 == x3), (x1 == x2) & (x2 < x3), (x1 < x2) & (x2 < x3)

    def dp(qa, a, b):                                  # coefficients [z1], [z1^2], [z1 z2] of prod_l (qa + z1 a + z2 b)
        e0, e1, f1, e2, e11 = np.ones(len(qa)), 0, 0, 0, 0
        for l in range(n):
            e11 = e11 * qa[:, l] + e1 * b[:, l] + f1 * a[:, l]
            e2 = e2 * qa[:, l] + e1 * a[:, l]
            e1, f1, e0 = e1 * qa[:, l] + e0 * a[:, l], f1 * qa[:, l] + e0 * b[:, l], e0 * qa[:, l]
        return e1, e2, e11

    p1 = dp(q[np.where(k1, x2, np.minimum(x2 + 1, m))], w[x1], w[x2])[0]
    _, p2, p11 = dp(q[x3], w[x1], w[x2])
    return q[np.where(k0, x1, x1 + 1)].prod(1) + np.where(k0, 0, p1) + np.where(k2, p2, 0) + np.where(k3, p11, 0)


def boundaries(n, m, d):
    """Three-digit boundaries whose Haar tails are closest to 1 - j/d; returns (cuts, Haar tails)."""
    K = math.comb(m + n - 1, n)
    A1 = np.array([math.comb(m - x + n - 2, n) / K for x in range(m)])
    A2 = np.array([math.comb(m - x + n - 3, n - 1) / K for x in range(m)])
    A3 = np.array([math.comb(m - x + n - 3, n - 2) / K for x in range(m)])
    cand, tail = [], []
    for x1 in range(m):                                 # lexicographic order; tails strictly decreasing
        x2, x3 = np.triu_indices(m - x1)
        cand.append(np.stack([np.full(len(x2), x1), x2 + x1, x3 + x1], axis=1))
        tail.append(A1[x1] + A2[x2 + x1] + A3[x3 + x1])
    cand, tail = np.concatenate(cand), np.concatenate(tail)
    assert np.diff(tail).max() < 1e-12                  # lexicographic order; only float rounding can break monotonicity
    tail = np.minimum.accumulate(tail)
    pos = np.searchsorted(-tail, -(1 - np.arange(1, d) / d))
    pos = np.clip(pos, 1, len(tail) - 1)
    pos = np.where(abs(tail[pos - 1] - (1 - np.arange(1, d) / d)) < abs(tail[pos] - (1 - np.arange(1, d) / d)), pos - 1, pos)
    pos = np.unique(pos[pos > 0])
    return [tuple(int(v) for v in c) for c in cand[pos]], tail[pos]


def self_check():
    n, m = 4, 16                                        # bosons and Haar counting vs enumeration of all 3876 outcomes
    U = np.linalg.qr(rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m)))[0]
    p = probs(U, n)
    lists = list(itertools.combinations_with_replacement(range(m), n))
    cuts = [(1, 1, 1), (0, 3, 3), (2, 2, 5), (0, 2, 7), (3, 4, 9), (5, 5, 6), (0, 0, 1), (0, 1, 2)]
    ref = [sum(pi for pi, o in zip(p, lists) if o[:3] >= c) for c in cuts]
    assert np.allclose(tails_boson(U[:, :n], cuts), ref, rtol=1e-8), (tails_boson(U[:, :n], cuts), ref)
    all_cuts, all_tails = boundaries(n, m, 10 ** 6)     # every boundary
    count = {c: sum(1 for o in lists if o[:3] >= c) / len(lists) for c in cuts}
    lookup = dict(zip(all_cuts, all_tails))
    assert all(np.isclose(lookup[c], count[c]) for c in cuts)
    n, m = 4, 6                                         # distinguishable vs all m^n assignments
    V = haar_cols(m, n)
    dcuts = [(1, 1, 1), (0, 3, 3), (2, 2, 5), (0, 2, 4), (1, 2, 3), (0, 0, 1)]
    ref = [sum(np.prod([abs(V[k, l]) ** 2 for l, k in enumerate(ks)])
               for ks in itertools.product(range(m), repeat=n) if tuple(sorted(ks))[:3] >= c) for c in dcuts]
    assert np.allclose(tails_dist(V, dcuts), ref, rtol=1e-9), (tails_dist(V, dcuts), ref)


def sample(n, m, n_u, d_fine):
    """Finest-level bin probabilities of bosons and of distinguishable particles for n_u Haar interferometers."""
    cuts, _ = boundaries(n, m, d_fine)
    Vs = [haar_cols(m, n) for _ in range(n_u)]
    return bins(np.array([tails_boson(V, cuts) for V in Vs])), bins(np.array([tails_dist(V, cuts) for V in Vs]))


def measure(n, m, n_u, d_fine, n_mean=1000, raw=None):
    t0 = time.time()
    cuts, t_h = boundaries(n, m, d_fine)
    PB, PD = raw if raw is not None else sample(n, m, n_u, d_fine)
    n_u = len(PB)
    wB = bins(t_h)[0]
    wD = bins(np.array([tails_dist(haar_cols(m, n), cuts) for _ in range(n_mean)])).mean(0)
    assert np.allclose(PB.sum(1), 1) and PB.min() > -1e-9
    rows = []
    for d in sorted({m, 4 * m, d_fine} & set(range(d_fine + 1))):
        g = merge(PB, wB, d) if d < d_fine else [[j] for j in range(len(wB))]   # finest level: raw bins
        col = lambda X: np.stack([X[..., idx].sum(-1) for idx in g], axis=-1)
        pb, pd, wb, wd = col(PB), col(PD), col(wB), col(wD)
        rb, rd = pb / wb - 1, pd / wd - 1
        inter = (pb - wb) - (pd - wd)
        a_u, b_u = (inter ** 2).mean(1), ((pb - wb) ** 2).mean(1)      # per interferometer; inter_frac^2 = mean(a_u) / mean(b_u)
        frac = math.sqrt(a_u.mean() / b_u.mean())
        rank = np.array([(np.argsort(-rd[u]) == rb[u].argmax()).argmax() for u in range(n_u)])   # classical rank of true MPB
        row = dict(n=n, m=m, d_target=d, d=len(g), n_u=n_u,
                   rel=float(abs(rb).max(1).mean()),
                   inter_frac=float(np.sqrt((inter ** 2).mean() / ((pb - wb) ** 2).mean())),
                   inter_sem=float((a_u - frac ** 2 * b_u).std() / (math.sqrt(n_u) * b_u.mean() * 2 * frac)),   # delta method
                   corr=float(np.corrcoef(rb.ravel(), rd.ravel())[0, 1]),
                   mpb_same=float((rank == 0).mean()), mpb_sem=float((rank == 0).std() / math.sqrt(n_u)),
                   mpb_top3=float((rank < 3).mean()), mpb_top1pct=float((rank < max(1, len(g) // 100)).mean()))
        rows.append(row)
        print(f"{n:3d} {m:4d}  d={row['d']:5d} (d/m={row['d'] / m:5.2f})  rel max={row['rel']:.2f}  interference/signal={row['inter_frac']:.3f}"
              f"  corr={row['corr']:.3f}  MPB same={row['mpb_same']:.2f}±{row['mpb_sem']:.2f}  top-3 {row['mpb_top3']:.2f}"
              f"  top-1% {row['mpb_top1pct']:.2f}  chance {1 / row['d']:.4f}   [{time.time() - t0:.0f} s]", flush=True)
    return rows


if __name__ == '__main__':
    self_check()
    print('self-check passed', flush=True)
    os.makedirs('results', exist_ok=True)
    if '--batch' in sys.argv:      # python finer_bins.py --batch N N_U SEED: raw bins of one size, for running in parallel
        n, n_u, seed = map(int, sys.argv[2:5])
        rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state
        PB, PD = sample(n, n * n, n_u, n ** 3)
        np.savez(f'results/finer_raw_n{n}_s{seed}.npz', PB=PB, PD=PD)
    elif '--pool' in sys.argv:     # python finer_bins.py --pool N [N_MEAN]: replace the rows of size N by the pooled batches
        n = int(sys.argv[2])
        raw = [np.load(f) for f in sorted(glob.glob(f'results/finer_raw_n{n}_s*.npz'))]
        rng.bit_generator.state = np.random.default_rng(n).bit_generator.state
        new = measure(n, n * n, None, n ** 3, n_mean=int(sys.argv[3]) if len(sys.argv) > 3 else 1000,   # N_MEAN: size of the distinguishable Haar sample
                      raw=tuple(np.concatenate([r[k] for r in raw]) for k in ('PB', 'PD')))
        old = json.load(open('results/finer_bins.json')) if os.path.exists('results/finer_bins.json') else []
        rows = [r for r in old if r['n'] != n] + new
        json.dump(sorted(rows, key=lambda r: (r['n'], r['d'])), open('results/finer_bins.json', 'w'), indent=1)
    else:
        plan = [(6, 200), (8, 100)] if '--quick' in sys.argv else [(6, 200), (8, 150), (10, 100), (12, 80), (14, 50), (16, 30)]
        rows = []
        for n, n_u in plan:
            rows += measure(n, n * n, n_u, n ** 3)
            json.dump(rows, open('results/finer_bins.json', 'w'), indent=1)      # save after every size
