"""Fine lexicographic bins (d up to ~m): how much of the signal is interference?

Outcomes are sorted mode lists i_1 <= i_2 <= ... in lexicographic order; bin boundaries sit on the
first two digits (x1, x2).  Tail probability of a boundary:
    T(x1, x1)      = Pr[i_1 >= x1]
    T(x1, x2 > x1) = Pr[i_1 >= x1 + 1] + Pr[one photon in mode x1, all others in modes >= x2]
The last term is the z^1 coefficient of Per(V^dag (Pi_{>=x2} + z Pi_{x1}) V), read off inside Glynn's
sum.  Distinguishable particles: same events with the product formula.  Haar mean for bosons is the
counting fraction (exact); for distinguishable particles it is averaged over many Haar samples.

Run:  python fine_bins.py        (~5 min; writes results/fine_bins.json)
"""
import itertools, json, math, os, sys
import numpy as np
from flatness import probs, rng
from signal_decay import permanent


def haar_cols(m, n):
    """First n columns of a Haar-random m x m unitary."""
    z = (rng.normal(size=(m, n)) + 1j * rng.normal(size=(m, n))) / math.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / abs(np.diag(r)))


def glynn_lin(A, c, chunk=1 << 14):
    """z^1 coefficient of Per(A + z * outer(conj(c), c))."""
    n = len(A)
    total, bits = 0j, np.arange(n - 1)
    with np.errstate(all='ignore'):      # spurious matmul warnings from Accelerate
        for start in range(0, 1 << (n - 1), chunk):
            idx = np.arange(start, min(start + chunk, 1 << (n - 1)))
            delta = np.ones((len(idx), n))
            delta[:, 1:] = 1 - 2 * ((idx[:, None] >> bits) & 1)
            r = delta @ A
            excl = np.ones_like(r)                       # excl[:, j] = prod_{k != j} r[:, k]
            excl[:, 1:] = np.cumprod(r, axis=1)[:, :-1]
            excl[:, :-1] *= np.cumprod(r[:, ::-1], axis=1)[:, ::-1][:, 1:]
            total += (delta[:, 1:].prod(1) * (delta @ c.conj()) * (excl @ c)).sum()
    assert np.isfinite(total)
    return total / 2 ** (n - 1)


def tail_haar(n, m, x1, x2):
    K = math.comb(m + n - 1, n)
    if x2 == x1:
        return math.comb(m - x1 + n - 1, n) / K
    return (math.comb(m - x1 + n - 2, n) + math.comb(m - x2 + n - 2, n - 1)) / K


def boundaries(n, m, d):
    """Two-digit boundaries whose Haar tails are closest to 1 - j/d (lexicographic order)."""
    cand = [(x1, x2) for x1 in range(m) for x2 in range(x1, m)]
    t = np.array([tail_haar(n, m, *c) for c in cand])          # decreasing
    pick = sorted({int(abs(t - (1 - j / d)).argmin()) for j in range(1, d)} - {0})
    return [cand[i] for i in pick], t[pick]


def tails_boson(V, cuts):
    with np.errstate(all='ignore'):
        gram = lambda a: V[a:].conj().T @ V[a:]
        cache, out = {}, []
        for x1, x2 in cuts:
            a = x1 if x2 == x1 else x1 + 1
            if a not in cache:
                cache[a] = permanent(gram(a)).real
            out.append(cache[a] + (0 if x2 == x1 else glynn_lin(gram(x2), V[x1]).real))
    return np.array(out)


def tails_dist(V, cuts):
    w = abs(V) ** 2
    q = np.vstack([np.cumsum(w[::-1], axis=0)[::-1], np.zeros(w.shape[1])])   # q[a, l] = sum_{k >= a} |V_kl|^2
    out = []
    for x1, x2 in cuts:
        if x2 == x1:
            out.append(q[x1].prod())
        else:
            out.append(q[x1 + 1].prod() + sum(w[x1, i] * np.delete(q[x2], i).prod() for i in range(w.shape[1])))
    return np.array(out)


def bins(t):
    t = np.atleast_2d(t)
    return -np.diff(np.hstack([np.ones((len(t), 1)), t, np.zeros((len(t), 1))]), axis=1)


def self_check():
    n, m = 4, 16                                             # bosons vs enumeration of all 3876 outcomes
    U = np.linalg.qr(rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m)))[0]
    p = probs(U, n)
    lists = list(itertools.combinations_with_replacement(range(m), n))
    cuts = [(0, 3), (1, 1), (1, 2), (2, 9), (5, 5), (6, 7)]
    ref = [sum(pi for pi, o in zip(p, lists) if o[:2] >= c) for c in cuts]
    assert np.allclose(tails_boson(U[:, :n], cuts), ref, rtol=1e-9), (tails_boson(U[:, :n], cuts), ref)
    hw = [sum(1 for o in lists if o[:2] >= c) / len(lists) for c in cuts]
    assert np.allclose([tail_haar(n, m, *c) for c in cuts], hw)
    n, m = 3, 7                                              # distinguishable vs all m^n assignments
    V, dcuts = haar_cols(m, n), [(0, 3), (1, 1), (1, 2), (2, 5), (4, 4)]
    ref = [sum(np.prod([abs(V[k, l]) ** 2 for l, k in enumerate(ks)])
               for ks in itertools.product(range(m), repeat=n) if tuple(sorted(ks))[:2] >= c) for c in dcuts]
    assert np.allclose(tails_dist(V, dcuts), ref, rtol=1e-9)


def merge(P, w, d):
    """Coarsen contiguous bins to about d bins of Haar weight ~1/d; returns a list of index groups."""
    groups, cur, acc = [], [], 0.0
    for j, wj in enumerate(w):
        cur.append(j); acc += wj
        if acc >= 1 / d - 1e-12:
            groups.append(cur); cur, acc = [], 0.0
    if cur:
        groups[-1] += cur
    return groups


def measure(n, m, n_u, n_mean=2000):
    cuts, t_h = boundaries(n, m, m)
    Vs = [haar_cols(m, n) for _ in range(n_u)]
    PB = bins(np.array([tails_boson(V, cuts) for V in Vs]))
    PD = bins(np.array([tails_dist(V, cuts) for V in Vs]))
    wB = bins(t_h)[0]
    wD = bins(np.array([tails_dist(haar_cols(m, n), cuts) for _ in range(n_mean)])).mean(0)
    assert np.allclose(PB.sum(1), 1) and PB.min() > -1e-9
    rows = []
    for d in sorted({n, round(n ** 1.5), m}):
        g = merge(PB, wB, d)
        col = lambda X: np.stack([X[..., idx].sum(-1) for idx in g], axis=-1)
        pb, pd, wb, wd = col(PB), col(PD), col(wB), col(wD)
        rb, rd = pb / wb - 1, pd / wd - 1                       # relative deviations from Haar mean
        inter = (pb - wb) - (pd - wd)                           # interference part
        row = dict(n=n, m=m, d_target=d, d=len(g), n_u=n_u,
                   rel=float(abs(rb).max(1).mean()), rel_rms=float(np.sqrt((rb ** 2).mean())),
                   inter_frac=float(np.sqrt((inter ** 2).mean() / ((pb - wb) ** 2).mean())),
                   corr=float(np.corrcoef(rb.ravel(), rd.ravel())[0, 1]),
                   mpb_same=float((rb.argmax(1) == rd.argmax(1)).mean()),
                   mpb_top3=float(np.mean([b in t for b, t in zip(rb.argmax(1), np.argsort(-rd, axis=1)[:, :3])])))
        rows.append(row)
        print(f"{n:3d} {m:4d}  d={row['d']:4d}  rel max={row['rel']:.3f} rms={row['rel_rms']:.3f}  interference/signal={row['inter_frac']:.3f}"
              f"  corr={row['corr']:.3f}  MPB same={row['mpb_same']:.2f} (top-3 {row['mpb_top3']:.2f}, chance {1 / row['d']:.3f})", flush=True)
    return rows


if __name__ == '__main__':
    self_check()
    print('self-check passed')
    plan = [(4, 200), (6, 200), (8, 100), (10, 100)] + ([] if '--quick' in sys.argv else [(12, 60), (14, 40), (16, 30)])
    rows = [r for n, n_u in plan for r in measure(n, n * n, n_u)]
    os.makedirs('results', exist_ok=True)
    json.dump(rows, open('results/fine_bins.json', 'w'), indent=1)
