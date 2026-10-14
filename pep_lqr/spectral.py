"""Structured epsilon-pseudospectrum of the perturbed closed loop Phi + Delta_A - Delta_B K.

Exact border
------------
For fixed Delta_B, put M = zI - Phi + Delta_B K. The smallest real Delta_A (spectral norm) that makes M - Delta_A
singular has the closed form of L. Qiu, B. Bernhardsson, A. Rantzer, E. J. Davison, P. M. Young and J. C. Doyle,
"A formula for computation of the real stability radius", Automatica 31(6), 1995:

    tau_R(M) = sup_{gamma in (0, 1]} sigma_{2n-1}( [[Re M, -gamma Im M], [Im M / gamma, Re M]] ),

which is unimodal in gamma. Hence z lies in the structured pseudospectrum iff

    phi(z) = min_{||Delta_B||_2 <= eps_B} tau_R(zI - Phi + Delta_B K) - eps_A <= 0,

and the border is the zero level set of phi. Only Delta_B (a 2-vector here) has to be searched.

Fast cloud
----------
`generate_perturbation_cloud` samples random admissible perturbations and returns their eigenvalues. It is
instantaneous and used by the interactive dashboard; the exact border is computed on demand.

Spectral-radius maximiser
-------------------------
`structured_ps_radius` maximises |lambda| over admissible perturbations by projected gradient ascent on the
simple-eigenvalue derivative (Guglielmi, Lubich, Sicilia, SIAM J. Numer. Anal. 61, 2023). Its dominant eigenvalue is
marked in Fig. 2, and it is one of the warm starts of the worst-case search. The maximiser is not unique (spectral-norm
bounds leave the other eigenvalue free): `spectral_radius_maximisers` samples the maximisers, and the pseudospectral
comparison curve of Fig. 3 is the largest finite-horizon cost among them.
"""
from types import SimpleNamespace

import numpy as np
import scipy.linalg as sla

from .worst_case import closed_loop, multistart, ascend, _batch_delta


# ----------------------------------------------------------------------------------------------------------------------
# Spectral-radius maximiser (pseudospectral approach)
# ----------------------------------------------------------------------------------------------------------------------
def spectral_radius_grad(p, DA, DB):
    """|lambda|^2 for the eigenvalue of largest modulus and its gradient 2 Re(conj(lam) / (x^H y) conj(x) y^T)."""
    F = closed_loop(p, DA, DB)
    lam, vl, vr = sla.eig(F, left=True, right=True)
    i = int(np.argmax(np.abs(lam)))
    l, x, y = lam[i], vl[:, i], vr[:, i]
    c = np.conj(l) / (np.conj(x) @ y)
    G = 2.0 * np.real(c * np.outer(np.conj(x), y))
    return float(abs(l) ** 2), G, -G @ p.K.T


def structured_ps_radius(p, n_random=8, rng=None, maxit=4000):
    """Perturbation maximising the spectral radius of Phi + Delta_A - Delta_B K (multistart projected ascent)."""
    rng = np.random.default_rng() if rng is None else rng
    _, GA, GB = spectral_radius_grad(p, np.zeros((p.n, p.n)), np.zeros((p.n, p.m)))
    starts = [("eig-gradient", (p.eps_A * GA / max(np.linalg.norm(GA, 2), 1e-300),
                                p.eps_B * GB / max(np.linalg.norm(GB, 2), 1e-300)))]
    starts += [(f"random{i}", (_batch_delta(1, p.n, p.n, p.eps_A, rng)[0], _batch_delta(1, p.n, p.m, p.eps_B, rng)[0]))
               for i in range(n_random)]
    r = multistart(p, spectral_radius_grad, starts, maxit=maxit)
    r.rho = np.sqrt(r.J)
    return r


def spectral_radius_maximisers(p, n_starts=2000, rng=None, tol=1e-6, maxit=4000):
    """Sample of perturbations attaining the maximal spectral radius.

    With spectral-norm bounds the maximiser is not unique: many perturbations reach the same maximal |lambda| while
    their other eigenvalue, and hence their finite-horizon cost, differ. Runs the projected ascent from `n_starts`
    random starts and returns all results within `tol` of the best spectral radius, as a list of (Delta_A, Delta_B).
    """
    rng = np.random.default_rng() if rng is None else rng
    best = structured_ps_radius(p, n_random=0, rng=rng, maxit=maxit)
    found = [best]
    for _ in range(n_starts):
        r = ascend(p, spectral_radius_grad, _batch_delta(1, p.n, p.n, p.eps_A, rng)[0],
                   _batch_delta(1, p.n, p.m, p.eps_B, rng)[0], maxit=maxit)
        found.append(r)
    rho_max = max(np.sqrt(r.J) for r in found)
    return [(r.DA, r.DB) for r in found if np.sqrt(r.J) >= rho_max - tol]



# ----------------------------------------------------------------------------------------------------------------------
# Exact border
# ----------------------------------------------------------------------------------------------------------------------
_LOGG = np.linspace(-4.0, 0.0, 13)          # coarse log10(gamma) grid used to bracket the unimodal maximum
_GR = (np.sqrt(5) - 1) / 2


def _sigma2(M, logg):
    """sigma_{2n-1} of the Qiu matrix, batched over M (..., n, n) complex and logg (...)."""
    g = (10.0 ** logg)[..., None, None]
    R, I = M.real, M.imag
    P = np.concatenate([np.concatenate([R, -g * I], -1), np.concatenate([I / g, R], -1)], -2)
    return np.linalg.svd(P, compute_uv=False)[..., -2]


def real_distance(M, n_golden=18):
    """tau_R(M): real spectral-norm distance to singularity of the complex matrices M (..., n, n) (Qiu et al. 1995)."""
    shp = M.shape[:-2]
    vals = np.stack([_sigma2(M, np.full(shp, t)) for t in _LOGG], -1)
    k = np.argmax(vals, -1)
    a = _LOGG[np.maximum(k - 1, 0)]; b = _LOGG[np.minimum(k + 1, len(_LOGG) - 1)]
    best = np.max(vals, -1)
    for _ in range(n_golden):                   # golden-section search for the maximum in [a, b]
        c, d = b - _GR * (b - a), a + _GR * (b - a)
        left = _sigma2(M, c) > _sigma2(M, d)    # maximum lies in [a, d]
        b = np.where(left, d, b)
        a = np.where(left, a, c)
    return np.maximum(best, _sigma2(M, (a + b) / 2))


def _check_shape(p):
    if p.eps_B > 0 and p.n * p.m != 2:
        raise NotImplementedError("the Delta_B search is implemented for a 2-vector Delta_B (n = 2, m = 1)")


def _DB_candidates(p, n_r=4, n_t=16):
    """Polar grid of the Delta_B ball (plus 0)."""
    if p.eps_B == 0:
        return np.zeros((1, p.n, p.m))
    C = [np.zeros((p.n, p.m))]
    for r in np.linspace(p.eps_B / n_r, p.eps_B, n_r):
        for t in np.linspace(0, 2 * np.pi, n_t, endpoint=False):
            C.append(r * np.array([np.cos(t), np.sin(t)]).reshape(p.n, p.m))
    return np.array(C)


def ps_phi(p, z, DB_cand, chunk=400):
    """phi(z) minimised over the candidate set (an upper bound of the true phi), and the argmin index."""
    z = np.atleast_1d(z)
    out, arg = np.empty(z.size), np.empty(z.size, int)
    BK = DB_cand @ p.K
    for i in range(0, z.size, chunk):
        zz = z[i:i + chunk]
        M = zz[:, None, None, None] * np.eye(p.n) - p.Phi + BK[None]
        d = real_distance(M)
        arg[i:i + chunk] = np.argmin(d, 1)
        out[i:i + chunk] = np.min(d, 1) - p.eps_A
    return out, arg


def ps_phi_refine_batch(p, zs, n_r=12, n_t=64, levels=4, chunk=48):
    """Refined min over Delta_B for many points z at once.

    The minimiser typically lies on the boundary of the Delta_B ball, so the search is done in polar coordinates:
    a fine polar grid, then `levels` zoom steps (a 9x9 local grid shrunk 4x per level, radius clipped to [0, eps_B])."""
    zs = np.atleast_1d(zs)
    zI = zs[:, None, None] * np.eye(p.n) - p.Phi
    if p.eps_B == 0:
        return real_distance(zI) - p.eps_A
    _check_shape(p)
    R, T = np.meshgrid(np.linspace(0, p.eps_B, n_r + 1), np.linspace(0, 2 * np.pi, n_t, endpoint=False))
    R, T = R.ravel(), T.ravel()
    BK = np.stack([R * np.cos(T), R * np.sin(T)], -1).reshape(-1, p.n, p.m) @ p.K
    off = np.linspace(-1, 1, 9)
    out = np.empty(zs.size)
    for i in range(0, zs.size, chunk):
        zc = zI[i:i + chunk]
        vals = real_distance(zc[:, None] + BK[None])
        k = np.argmin(vals, 1)
        idx = np.arange(len(k))
        best, r0, t0 = vals[idx, k], R[k], T[k]
        dr, dt = p.eps_B / n_r, 2 * np.pi / n_t
        for _ in range(levels):
            rr = np.clip(r0[:, None, None] + dr * off[None, :, None], 0, p.eps_B) + 0 * off[None, None, :]
            tt = t0[:, None, None] + dt * off[None, None, :] + 0 * off[None, :, None]
            rr, tt = rr.reshape(len(k), -1), tt.reshape(len(k), -1)
            D = np.stack([rr * np.cos(tt), rr * np.sin(tt)], -1).reshape(len(k), -1, p.n, p.m)
            v = real_distance(zc[:, None] + D @ p.K)
            kk = np.argmin(v, 1)
            vb = v[idx, kk]
            upd = vb < best
            best = np.where(upd, vb, best)
            r0 = np.where(upd, rr[idx, kk], r0)
            t0 = np.where(upd, tt[idx, kk], t0)
            dr, dt = dr / 4, dt / 4
        out[i:i + chunk] = best - p.eps_A
    return out


def structured_pseudospectrum(p, re=(-1.15, 1.15), im_max=1.15, n_re=161, n_real=2001):
    """Signed membership function phi on a grid (phi <= 0 inside the structured pseudospectrum).

    Computed on the upper half-plane and mirrored (the set is symmetric since the perturbations are real); the
    intervals of the real axis inside the set are returned separately in `real_segments`.
    Exclusion test used to skip most of the plane: phi(z) >= sigma_min(zI - Phi) - eps_B ||K|| - eps_A.
    The polar candidate grid gives an upper bound of phi; tau_R is 1-Lipschitz, so the candidates' covering radius
    times ||K|| bounds the error and only points where the sign is uncertain are refined.

    Returns:
        SimpleNamespace(X, Y, phi) ready for contour plotting.
    """
    _check_shape(p)
    h = (re[1] - re[0]) / (n_re - 1)
    # rows at Im z = h/2, 3h/2, ...: the real axis lies between rows (its intervals are drawn separately as lines)
    X, Y = np.meshgrid(np.linspace(*re, n_re), np.arange(h / 2, im_max + h / 2, h))
    Zc = (X + 1j * Y).ravel()
    phi = np.full(Zc.size, np.inf)
    smin0 = np.linalg.svd(Zc[:, None, None] * np.eye(p.n) - p.Phi, compute_uv=False)[:, -1]
    lb = smin0 - p.eps_B * np.linalg.norm(p.K, 2) - p.eps_A
    cand = lb <= 0
    DBc = _DB_candidates(p)
    phi[cand], _ = ps_phi(p, Zc[cand], DBc)
    phi[~cand] = lb[~cand]
    if p.eps_B > 0:
        n_r, n_t = 4, 16
        spacing = 0.5 * p.eps_B * np.hypot(1 / n_r, 2 * np.pi / n_t)
        in_band = (phi[cand] > 0) & (phi[cand] < spacing * np.linalg.norm(p.K, 2))
        band = np.where(cand)[0][in_band]
        if band.size:
            phi[band] = np.minimum(phi[band], ps_phi_refine_batch(p, Zc[band]))
    PHI = phi.reshape(X.shape)
    return SimpleNamespace(X=np.vstack([X[::-1], X]), Y=np.vstack([-Y[::-1], Y]), phi=np.vstack([PHI[::-1], PHI]),
                           real_segments=_real_axis_segments(p, re, n_real))


def _real_axis_segments(p, re, n=2001):
    """Intervals of the real axis inside the pseudospectrum (real eigenvalues). They have no area, so a 2D grid
    only shows them as thin spikes; they are computed on a fine 1D grid and drawn as lines."""
    x = np.linspace(*re, n).astype(complex)
    phi, _ = ps_phi(p, x, _DB_candidates(p))
    if p.eps_B > 0:
        n_r, n_t = 4, 16
        spacing = 0.5 * p.eps_B * np.hypot(1 / n_r, 2 * np.pi / n_t)
        band = (phi > 0) & (phi < spacing * np.linalg.norm(p.K, 2))
        if band.any():
            phi[band] = np.minimum(phi[band], ps_phi_refine_batch(p, x[band]))
    inside = np.concatenate([[False], phi <= 0, [False]])
    starts, ends = np.where(np.diff(inside.astype(int)) == 1)[0], np.where(np.diff(inside.astype(int)) == -1)[0] - 1
    return [(x[s].real, x[e].real) for s, e in zip(starts, ends)]


def plot_pseudospectrum(ax, grid, color="#ffaa0f", alpha=0.7, lw=1.0, lw_real=2.0, zorder=2):
    """Filled structured pseudospectrum with its border, and its real-axis intervals as lines."""
    ax.contourf(grid.X, grid.Y, grid.phi, levels=[-1e9, 0], colors=[color], alpha=alpha, zorder=zorder)
    ax.contour(grid.X, grid.Y, grid.phi, levels=[0], colors=[color], linewidths=lw, zorder=zorder)
    for a, b in getattr(grid, "real_segments", []):
        ax.plot([a, b], [0, 0], color=color, lw=lw_real, solid_capstyle='butt', zorder=zorder)


# ----------------------------------------------------------------------------------------------------------------------
# Fast random cloud (dashboard)
# ----------------------------------------------------------------------------------------------------------------------
def generate_2D_rotations(N, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    theta = rng.uniform(0, 2 * np.pi, N)
    C, S = np.cos(theta), np.sin(theta)
    R = np.zeros((N, 2, 2))
    R[:, 0, 0], R[:, 0, 1] = C, -S
    R[:, 1, 0], R[:, 1, 1] = S, C
    flips = rng.choice([1, -1], size=(N, 1, 1))
    return R * flips


def generate_perturbation_cloud(Phi, K, eps_A, eps_B, N_samples=30000, n=2, m=1, rng=None):
    """Eigenvalues of Phi + Delta_A - Delta_B K for random admissible perturbations (fast visual estimate)."""
    rng = np.random.default_rng() if rng is None else rng
    U_A, _, Vh_A = np.linalg.svd(rng.standard_normal((N_samples, n, n)))
    S_rand_A = rng.uniform(0, 1, size=(N_samples, n))
    Delta_A_1 = (U_A * S_rand_A[:, np.newaxis, :]) @ Vh_A

    M_B = rng.standard_normal((N_samples, n, m))
    Delta_B_1 = (M_B / np.linalg.norm(M_B, axis=1, keepdims=True)) * rng.uniform(0, 1, size=(N_samples, 1, 1))

    K_batch = np.broadcast_to(K, (N_samples, m, n))
    eigs_batch_1 = np.linalg.eigvals(Phi + eps_A * Delta_A_1 - eps_B * (Delta_B_1 @ K_batch)).flatten()

    if n == 2:
        U_str, V_str = generate_2D_rotations(N_samples, rng), generate_2D_rotations(N_samples, rng)
        Sig_str = np.zeros((N_samples, n, n))
        Sig_str[:, 0, 0] = 1.0
        Sig_str[:, 1, 1] = rng.uniform(-1, 1, N_samples)
        Delta_A_2 = U_str @ Sig_str @ np.transpose(V_str, (0, 2, 1))

        theta_b = rng.uniform(0, 2 * np.pi, N_samples)
        Delta_B_2 = np.zeros((N_samples, n, m))
        Delta_B_2[:, 0, 0] = np.cos(theta_b)
        Delta_B_2[:, 1, 0] = np.sin(theta_b)

        eigs_batch_2 = np.linalg.eigvals(Phi + eps_A * Delta_A_2 - eps_B * (Delta_B_2 @ K_batch)).flatten()
        return np.concatenate((eigs_batch_1, eigs_batch_2))
    return eigs_batch_1
