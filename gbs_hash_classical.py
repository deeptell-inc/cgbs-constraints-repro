"""Classical evaluation of the GBS hash of Hatanaka et al. (arXiv:2409.19932).

Their hash bit is c_j = floor(10^k * mu_j) mod 2 with mu_j = <n_j n_{j+1} n_{j+2}> on a Gaussian
state (squeezed vacua r = b_j through a linear interferometer). A fixed-order photon-number moment
of a Gaussian state follows from the covariance matrix by Wick's theorem, so no sampling is needed.

Run:  python gbs_hash_classical.py
"""
import json, math, os, time
import numpy as np
from thewalrus.quantum import photon_number_mean, photon_number_moment

rng = np.random.default_rng(0)


def brick_unitary(N, depth):
    """Brickwork of beamsplitters with periodic boundary, angles drawn as in their Sec. II."""
    U = np.eye(N, dtype=complex)
    for l in range(depth):
        for j in range(l % 2, N, 2):
            a, b = j, (j + 1) % N
            th, ph = rng.normal(math.pi / 4, math.pi / 16), rng.uniform(0, 2 * math.pi)
            c, s = math.cos(th), np.exp(1j * ph) * math.sin(th)
            U[[a, b]] = c * U[a] - s.conjugate() * U[b], s * U[a] + c * U[b]
    assert np.allclose(np.einsum("ik,jk->ij", U, U.conj()), np.eye(N))
    return U


def covariance(U, r):
    """xp-ordered covariance (hbar = 2) of squeezed vacua with parameters r after interferometer U."""
    S = np.block([[U.real, -U.imag], [U.imag, U.real]])
    # einsum, not @: Accelerate's matmul raises spurious floating-point warnings on this machine
    return np.einsum('ik,k,jk->ij', S, np.concatenate([np.exp(-2 * r), np.exp(2 * r)]), S, optimize=False)


def gbs_hash(bits, U, k=1):
    N = len(bits)
    cov, mu0 = covariance(U, np.asarray(bits, float)), np.zeros(2 * N)
    mus = [photon_number_moment(mu0, cov, {j: 1, (j + 1) % N: 1, (j + 2) % N: 1}) for j in range(N)]
    return [int(math.floor(10 ** k * x)) % 2 for x in mus], mus


if __name__ == '__main__':
    # checks: identity interferometer gives a product state; mean photon number matches the closed form
    _, mus = gbs_hash([1, 1, 1, 0], np.eye(4))
    assert np.isclose(mus[0], math.sinh(1) ** 6) and np.isclose(mus[1], 0, atol=1e-12)
    U, r = brick_unitary(8, 8), rng.integers(0, 2, 8).astype(float)
    assert np.isclose(photon_number_mean(np.zeros(16), covariance(U, r), 3), (abs(U[3]) ** 2 * np.sinh(r) ** 2).sum())

    rows = []
    for N in (16, 64, 256):
        U, bits = brick_unitary(N, N), rng.integers(0, 2, N)
        t0 = time.time()
        h, _ = gbs_hash(bits, U)
        rows.append(dict(modes=N, seconds=time.time() - t0, hash=''.join(map(str, h))))
        print(f"N={N:4d} modes: hash {''.join(map(str, h[:32]))}{'...' if N > 32 else ''}  ({rows[-1]['seconds']:.2f} s, no sampling)")
    os.makedirs('results', exist_ok=True)
    json.dump(rows, open('results/gbs_hash.json', 'w'), indent=1)
