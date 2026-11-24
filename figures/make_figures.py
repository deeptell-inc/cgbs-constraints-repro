"""Figures 2-4 of the manuscript, drawn from ../results/*.json only.

Run from this directory:  python make_figures.py   (writes fig2.pdf, fig3.pdf, fig4.pdf, numbers.json)
numbers.json holds the derived quantities quoted in the text (power-law exponents).
"""
import json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.linewidth': 0.6, 'lines.linewidth': 1.0,
                     'lines.markersize': 4, 'legend.frameon': False, 'legend.handlelength': 1.6,
                     'xtick.direction': 'in', 'ytick.direction': 'in', 'pdf.fonttype': 42})
BLUE, RED, GREY, GREEN = '#1f5fa8', '#c0392b', '#666666', '#2e8b57'
load = lambda name: json.load(open(f'../results/{name}.json'))
numbers = {}


def slope(x, y):
    """Least-squares exponent of y ~ x^a and its standard error."""
    (a, _), cov = np.polyfit(np.log(x), np.log(y), 1, cov=True)
    return float(a), float(math.sqrt(cov[0, 0]))


def label(ax, s):
    ax.set_title(s, loc='left', fontweight='bold', fontsize=9, pad=3)


def fig2():
    d = load('flatness')
    rows = d['flatness']
    K = np.array([r['K'] for r in rows], float)
    dilute = np.array([r['m'] == r['n'] ** 2 for r in rows])
    rms = np.sqrt([(r['d'] - 1) / r['d'] ** 2 * (r['S2_times_K'] - 1) / (r['K'] - 1) for r in rows])
    eps = d['reward'][0]['eps']
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={'width_ratios': [1.5, 1]})
    for mask, fill, tag in ((dilute, 'full', r'$M=N^2$'), (~dilute, 'none', r'$M=2N$')):
        for key, mk, col, lab in (('dev_lex', 's', RED, 'lexicographic'), ('dev_rnd', 'o', BLUE, 'random permutation')):
            a.errorbar(K[mask], [r[key] for r, k in zip(rows, mask) if k], yerr=[r[key + '_sem'] for r, k in zip(rows, mask) if k],
                       fmt=mk, color=col, fillstyle=fill, lw=0.6, capsize=1.5, label=f'{lab}, {tag}')
    a.set_xscale('log'); a.set_yscale('log')
    o = np.argsort(K)
    a.loglog(K[o], rms[o], '--', color=BLUE, lw=0.8, label='r.m.s. from the lemma')
    a.axhline(eps, color=GREY, lw=0.6, ls=':')
    a.text(K.max(), eps * 1.15, r'tolerance $\epsilon$', ha='right', color=GREY)
    a.set_xlabel(r'number of outcomes $|Y|$'); a.set_ylabel(r'$\langle\max_j|P_j-1/d|\rangle$')
    a.set_ylim(3e-5, 0.3); a.legend(fontsize=7, loc='lower left'); label(a, 'a')
    rew = [r for r in d['reward'] if r['d'] == 4]
    x = np.arange(len(rew))
    b.bar(x - 0.2, [r['random_perm']['uniform'] for r in rew], 0.4, color=BLUE, label='random permutation')
    b.bar(x + 0.2, [r['lexicographic']['uniform'] for r in rew], 0.4, color=RED, label='lexicographic')
    b.set_xticks(x, [f"$N={r['n']}$\n$M={r['m']}$" for r in rew])
    b.set_ylabel('uniform sampler rewarded'); b.set_ylim(0, 1.25); b.set_yticks([0, 0.5, 1])
    b.legend(fontsize=7, loc='upper right'); label(b, 'b')
    fig.tight_layout(pad=0.4); fig.savefig('fig2.pdf')
    numbers['fig2_dev_exponent_vs_K'] = slope(K, [r['dev_rnd'] for r in rows])


def fig3():
    sd = [r for r in load('signal_decay') if r['regime'] == 'm = n^2, d = 4']
    n = np.array([r['n'] for r in sd], float)
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.0, 2.4))
    for key, col, mk, lab in (('dev', BLUE, 'o', 'bosons'), ('dev_dist', GREY, 's', 'distinguishable'),
                              ('resid', RED, '^', 'interference part')):
        y = np.array([r[key] for r in sd])
        a.loglog(n, y, mk, color=col, label=lab, fillstyle='none' if key == 'dev_dist' else 'full')
        s, _ = slope(n[n >= 8], y[n >= 8])
        if key != 'dev_dist':
            a.loglog(n, y[-1] * (n / n[-1]) ** s, '-', color=col, lw=0.6)
            a.text(n[4], y[4] * 1.3, f'$N^{{{s:.2f}}}$', color=col)
    a.set_xlabel('photons $N$'); a.set_ylabel('deviation from Haar mean')
    a.set_xticks([4, 8, 16], ['4', '8', '16']); a.set_xticks([], minor=True)
    a.legend(fontsize=7, loc='lower left'); label(a, 'a')
    fb = load('finer_bins')
    levels = [('$d\\approx M$', lambda r: r['d_target'] == r['m'], GREEN, 'o'),
              ('$d\\approx 3.5M$', lambda r: r['d_target'] == 4 * r['m'], GREY, 's'),
              ('$d=M^{3/2}$', lambda r: r['d_target'] == r['n'] ** 3, RED, '^')]
    for lab, pick, col, mk in levels:
        rows = [r for r in fb if pick(r)]
        b.errorbar([r['n'] for r in rows], [r['inter_frac'] for r in rows], yerr=[r['inter_sem'] for r in rows], fmt=mk + '-',
                   color=col, label=lab, lw=0.6, capsize=1.5)
    b.set_xlabel('photons $N$'); b.set_ylabel('interference / signal (r.m.s.)'); b.set_ylim(0, 0.5)
    b.set_xticks(range(6, 17, 2)); c.set_xticks(range(6, 17, 2))
    b.legend(fontsize=7, loc='lower left'); label(b, 'b')
    fine = [r for r in fb if r['d_target'] == r['n'] ** 3]
    x = [r['n'] for r in fine]
    c.errorbar(x, [r['mpb_same'] for r in fine], yerr=[r['mpb_sem'] for r in fine], fmt='o-', color=BLUE, lw=0.6,
               capsize=1.5, label='same bin')
    for key, mk, col, lab in (('mpb_top3', 's', GREY, 'top 3'), ('mpb_top1pct', '^', RED, 'top 1%')):
        f = np.array([r[key] for r in fine])
        c.errorbar(x, f, yerr=np.sqrt(f * (1 - f) / [r['n_u'] for r in fine]), fmt=mk + '-', color=col, lw=0.6,
                   capsize=1.5, label=lab)   # binomial standard error
    c.plot(x, [1 / r['d'] for r in fine], ':', color='k', lw=0.8, label='chance, same bin')
    c.set_xlabel('photons $N$'); c.set_ylabel('product formula finds the MPB'); c.set_ylim(-0.03, 1.5)
    c.set_yticks([0, 0.5, 1]); c.legend(fontsize=7, loc='upper left', ncol=2, columnspacing=0.8); label(c, 'c')
    fig.tight_layout(pad=0.4); fig.savefig('fig3.pdf')
    for reg in ('m = n^2, d = 4', 'm = n^2, d = n', 'm = 2n,  d = 2'):
        rows = [r for r in load('signal_decay') if r['regime'] == reg and r['n'] >= 8]
        nn = [r['n'] for r in rows]
        numbers[f'signal_exponent[{reg}]'] = slope(nn, [r['dev'] for r in rows])
        numbers[f'interference_exponent[{reg}]'] = slope(nn, [r['resid'] for r in rows])


def fig4():
    q = load('quad_stats')
    names = {'linear': ('linear', GREY, 's'), 'collisions': ('collisions', BLUE, 'o'),
             'neighbours': ('neighbours', GREEN, 'v'), 'random graph': ('random graph', RED, '^'),
             'blocks r=4': ('blocks, $r=4$', 'k', 'D')}
    pick = lambda stat, dilute: [r for r in q if r['stat'].startswith(stat) and (r['m'] == r['n'] ** 2) == dilute
                                 and not math.isnan(r['res3'])]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.0, 2.4))
    for stat in ('linear', 'collisions', 'random graph'):
        lab, col, mk = names[stat]
        rows = [r for r in q if r['stat'].startswith(stat) and r['m'] == r['n'] ** 2]
        a.errorbar([r['n'] for r in rows], [r['signal'] for r in rows], yerr=[r['sig_sem'] for r in rows],
                   fmt=mk + '-', color=col, lw=0.6, capsize=1.5, label=lab)
    a.set_xscale('log'); a.set_yscale('log'); a.set_xticks([3, 4, 5, 6], ['3', '4', '5', '6']); a.set_xticks([], minor=True)
    a.set_xlabel('photons $N$'); a.set_ylabel('signal (total variation)'); a.legend(fontsize=7, loc='lower left')
    label(a, 'a')
    for ax, dilute, tag, stats in ((b, True, 'c', ('linear', 'collisions', 'random graph')),
                                   (c, False, 'c', ('linear', 'collisions', 'neighbours', 'random graph', 'blocks r=4'))):
        for stat in stats:
            lab, col, mk = names[stat]
            rows = pick(stat, dilute)
            ax.errorbar([r['n'] for r in rows], [r['res3'] for r in rows], yerr=[r['res3_sem'] for r in rows], fmt=mk + '-',
                        color=col, lw=0.6, capsize=1.5, label=lab)
        ax.set_xlabel('photons $N$'); ax.set_ylim(0, 0.8)
        ax.set_xticks(sorted({r['n'] for r in pick('linear', dilute)}))
    b.set_ylabel('residual / signal, 3 moments'); label(b, 'b'); label(c, 'c')
    b.set_title('$M=N^2$', loc='right', fontsize=8, pad=3); c.set_title('$M=2N$', loc='right', fontsize=8, pad=3)
    b.legend(fontsize=7, loc='upper left'); c.legend(fontsize=7, loc='upper left', ncol=2, columnspacing=0.8)
    fig.tight_layout(pad=0.4); fig.savefig('fig4.pdf')
    for stat, dilute in (('collisions', True), ('collisions', False), ('neighbours', False), ('linear', True)):
        rows = [r for r in q if r['stat'].startswith(stat) and (r['m'] == r['n'] ** 2) == dilute]
        numbers[f"quad_signal_exponent[{stat}, {'dilute' if dilute else 'dense'}]"] = \
            slope([r['n'] for r in rows], [r['signal'] for r in rows])


def collision():
    """Haar mean of |Y| sum_s p(s)^2 in closed form (Mhiri et al., arXiv:2604.14323, Eq. (45)) against our exact distributions."""
    from scipy.integrate import quad
    P2 = lambda m, n: (n + m - 1) * quad(lambda t: math.cos(t) ** (n + m - 2) * math.sin((n + 1) * t), 0, math.pi / 2, limit=400)[0]
    assert abs(P2(5, 1) - 10 / 6) < 1e-9                     # one photon: m E|U_j1|^4 = 2m/(m+1)
    out = {f"measured, closed form [N={r['n']}, M={r['m']}]": [r['S2_times_K'], P2(r['m'], r['n'])] for r in load('flatness')['flatness']}
    out['closed form [N=20, M=400]'] = P2(400, 20)
    out['max over N<=40 of closed form / (N+1), M=N^2'] = max(P2(n * n, n) / (n + 1) for n in range(2, 41))
    out['max over N<=40 of closed form, M=2N'] = max(P2(2 * n, n) for n in range(2, 41))
    return out


if __name__ == '__main__':
    fig2(); fig3(); fig4()
    for k, (v, e) in numbers.items():
        print(f'{k:55s} {v:+.2f} ± {e:.2f}')
    numbers['collision'] = collision()
    for k, v in numbers['collision'].items():
        print(f'{k:50s}', np.round(v, 3))
    json.dump(numbers, open('numbers.json', 'w'), indent=1)
