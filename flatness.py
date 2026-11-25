"""Flatness of permuted state binning in coarse-grained boson sampling.

Lemma checked here: for ANY distribution p on K outcomes and a uniformly random
partition into d equal bins,  Var[P(bin)] = (d-1)/d^2 * K/(K-1) * (sum p^2 - 1/K).

Run:  python flatness.py          (~3 min; writes results/flatness.json)
"""
import glob, itertools, json, math, os, sys
import numpy as np

rng = np.random.default_rng(0)


def haar(m):
    z = (rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))) / math.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / abs(np.diag(r)))


def probs(U, n, batch=20000):
    """Exact output distribution, n photons in the first n modes; outcomes in lexicographic order."""
    m = U.shape[0]
    outs = np.array(list(itertools.combinations_with_replacement(range(m), n)), dtype=np.int16)
    deltas = np.array([(1,) + d for d in itertools.product((1, -1), repeat=n - 1)])  # Glynn formula
    sign = deltas.prod(1)
    p = np.empty(len(outs))
    for i in range(0, len(outs), batch):
        o = outs[i:i + batch]
        A = U[o][:, :, :n]                                   # (b, n, n) submatrices
        perm = ((np.einsum('dk,bkj->bdj', deltas, A)).prod(2) * sign).sum(1) / 2 ** (n - 1)
        s = np.sort(o, axis=1)                               # multiplicities prod s_i!
        run = np.ones(len(o)); mult = np.ones(len(o))
        for c in range(1, n):
            run = np.where(s[:, c] == s[:, c - 1], run + 1, 1)
            mult *= run
        p[i:i + batch] = abs(perm) ** 2 / mult
    return p


def bin_probs(p, order, d):
    return np.array([c.sum() for c in np.array_split(p[order], d)])


def var_formula(p, d):
    K = len(p)
    return (d - 1) / d ** 2 * K / (K - 1) * ((p ** 2).sum() - 1 / K)


def self_check():
    """Exact enumeration of all equal bipartitions of a 10-outcome distribution."""
    p = probs(haar(4), 2)
    assert abs(p.sum() - 1) < 1e-10
    sums = [p[list(c)].sum() for c in itertools.combinations(range(10), 5)]
    assert np.isclose(np.var(sums), var_formula(p, 2), rtol=1e-9), (np.var(sums), var_formula(p, 2))


def flat_sample(n, m, n_u, d=4, n_perm=40):
    """Per interferometer: [sum p^2, lexicographic deviation, mean random-permutation deviation, Var measured, Var from the lemma]."""
    acc = []
    for _ in range(n_u):
        p = probs(haar(m), n)
        assert abs(p.sum() - 1) < 1e-8
        P = np.array([bin_probs(p, rng.permutation(len(p)), d) for _ in range(n_perm)])
        acc.append([(p ** 2).sum(), abs(bin_probs(p, np.arange(len(p)), d) - 1 / d).max(), abs(P - 1 / d).max(1).mean(),
                    ((P - 1 / d) ** 2).mean(), var_formula(p, d)])
    return np.array(acc)


def flat_row(n, m, acc, d=4):
    K = math.comb(m + n - 1, n)
    S2, lex, rnd, var_emp, var_th = acc.mean(0)
    row = dict(n=n, m=m, K=K, d=d, n_u=len(acc), S2_times_K=float(S2 * K), S2_times_K_sem=float(acc[:, 0].std() * K / math.sqrt(len(acc))),
               dev_lex=float(lex), dev_rnd=float(rnd), var_ratio=float(var_emp / var_th),
               dev_lex_sem=float(acc[:, 1].std() / math.sqrt(len(acc))), dev_rnd_sem=float(acc[:, 2].std() / math.sqrt(len(acc))))
    print(f"{n:2d} {m:3d} {K:8d}  n_u={len(acc):3d}  K*sum p^2={row['S2_times_K']:6.3f}±{row['S2_times_K_sem']:.3f} (n+1={n + 1})  "
          f"lex={row['dev_lex']:.4f}  rnd={row['dev_rnd']:.5f}  Var emp/theory={row['var_ratio']:.3f}", flush=True)
    return row


def flatness_table():
    plan = [(3, 9, 20), (4, 16, 20), (5, 25, 10), (6, 36, 3), (6, 12, 10), (7, 14, 10), (8, 16, 3)]
    return [flat_row(n, m, flat_sample(n, m, n_u)) for n, m, n_u in plan]


def reward_sample(n, m, d, n_u, trials):
    """Reward step of the PoW protocol of Singh et al. in isolation: is a uniform sampler rewarded?

    Sample size and tolerance follow their Eq. (33) and 2 d eps^0.8 = 0.1; a miner holds 1/10 of the pool.
    Per interferometer: number of rewarded trials [random permutation: honest, uniform; lexicographic: honest, uniform].
    """
    eps, n_tot = (0.05 / d) ** 1.25, int(1.8e5 * d ** 3.5)
    out = []
    for _ in range(n_u):
        p = probs(haar(m), n)
        cnt = []
        for name in ('random_perm', 'lexicographic'):
            ok = [0, 0]
            for _ in range(trials):
                order = rng.permutation(len(p)) if name == 'random_perm' else np.arange(len(p))
                P = bin_probs(p, order, d); P /= P.sum()
                mu_net = rng.multinomial(n_tot, P).max() / n_tot
                for i, q in enumerate((P, np.full(d, 1 / d))):
                    mu_i = rng.multinomial(n_tot // 10, q).max() / (n_tot // 10)
                    ok[i] += abs(mu_i - mu_net) <= eps
            cnt += ok
        out.append(cnt)
    return np.array(out)


def reward_row(n, m, d, cnt, trials):
    f = cnt / trials                                   # per interferometer
    sem = lambda x: float(x.std() / math.sqrt(len(x)))   # standard error over interferometers
    out = dict(n=n, m=m, d=d, eps=(0.05 / d) ** 1.25, n_tot=int(1.8e5 * d ** 3.5), n_u=len(cnt), trials=trials,
               random_perm=dict(boson=float(f[:, 0].mean()), uniform=float(f[:, 1].mean()),
                                uniform_min=float(f[:, 1].min()), uniform_max=float(f[:, 1].max()), uniform_sem=sem(f[:, 1])),
               lexicographic=dict(boson=float(f[:, 2].mean()), uniform=float(f[:, 3].mean()), uniform_max=float(f[:, 3].max()),
                                  uniform_sem=sem(f[:, 3])))
    r, l = out['random_perm'], out['lexicographic']
    print(f"n={n} m={m} d={d} eps={out['eps']:.1e} {len(cnt)} interferometers x {trials} trials  random permutation: honest {r['boson']:.3f}, "
          f"uniform {r['uniform']:.3f} [{r['uniform_min']:.2f}, {r['uniform_max']:.2f}]   lexicographic: honest {l['boson']:.3f}, "
          f"uniform {l['uniform']:.3f} (max {l['uniform_max']:.2f})", flush=True)
    return out


def reward_test(n, m, d, n_u=1, trials=400):
    return reward_row(n, m, d, reward_sample(n, m, d, n_u, trials), trials)


if __name__ == '__main__':
    self_check()
    os.makedirs('results', exist_ok=True)
    if '--batch' in sys.argv:      # python flatness.py --batch N M N_U SEED: raw data of one size, for running in parallel
        n, m, n_u, seed = map(int, sys.argv[2:6])
        rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state
        np.save(f'results/flat_raw_n{n}_m{m}_s{seed}.npy', flat_sample(n, m, n_u))
    elif '--reward-batch' in sys.argv:   # python flatness.py --reward-batch N M D N_U TRIALS SEED
        n, m, d, n_u, trials, seed = map(int, sys.argv[2:8])
        rng.bit_generator.state = np.random.default_rng(seed).bit_generator.state
        np.save(f'results/reward_raw_n{n}_m{m}_d{d}_t{trials}_s{seed}.npy', reward_sample(n, m, d, n_u, trials))
    elif '--reward-pool' in sys.argv:    # python flatness.py --reward-pool N M D TRIALS: replace that row by the pooled batches
        n, m, d, trials = map(int, sys.argv[2:6])
        cnt = np.concatenate([np.load(f) for f in sorted(glob.glob(f'results/reward_raw_n{n}_m{m}_d{d}_t{trials}_s*.npy'))])
        res = json.load(open('results/flatness.json')) if os.path.exists('results/flatness.json') else {'flatness': [], 'reward': []}
        res['reward'] = sorted([x for x in res['reward'] if (x['n'], x['m'], x['d']) != (n, m, d)] + [reward_row(n, m, d, cnt, trials)],
                               key=lambda x: (x['n'], x['d']))
        json.dump(res, open('results/flatness.json', 'w'), indent=1)
    elif '--pool' in sys.argv:     # python flatness.py --pool N M: replace the row of that size by the pooled batches
        n, m = map(int, sys.argv[2:4])
        acc = np.concatenate([np.load(f) for f in sorted(glob.glob(f'results/flat_raw_n{n}_m{m}_s*.npy'))])
        res = json.load(open('results/flatness.json')) if os.path.exists('results/flatness.json') else {'flatness': [], 'reward': []}
        res['flatness'] = sorted([x for x in res['flatness'] if (x['n'], x['m']) != (n, m)] + [flat_row(n, m, acc)],
                                 key=lambda x: (x['m'] != x['n'] ** 2, x['n']))
        json.dump(res, open('results/flatness.json', 'w'), indent=1)
    else:
        print(' n   m        K')
        res = {'flatness': flatness_table(), 'reward': [reward_test(4, 16, 4), reward_test(5, 25, 4), reward_test(6, 36, 4), reward_test(6, 36, 16)]}
        json.dump(res, open('results/flatness.json', 'w'), indent=1)
