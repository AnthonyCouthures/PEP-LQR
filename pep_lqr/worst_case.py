"""Worst-case LTI perturbation by norm-constrained projected gradient ascent.

For fixed perturbations (Delta_A, Delta_B), the worst initial condition on the unit ball is the top eigenvector of the
finite-horizon Gramian, so

    J_N(Delta) = lambda_max(P_N),   P_0 = S,   P_{j+1} = Q_K + Phi_D^T P_j Phi_D,   Phi_D = Phi + Delta_A - Delta_B K.

Its gradient follows from one forward and one adjoint (costate) sweep:

    grad_{Delta_A} J = 2 sum_k p_{k+1} x_k^T,    grad_{Delta_B} J = -grad_{Delta_A} J K^T,
    p_N = S x_N,   p_k = Q_K x_k + Phi_D^T p_{k+1}.

The perturbations are kept in their spectral-norm balls by projection (singular-value clipping), and the step size is
controlled by an Armijo rule, following the norm-constrained gradient flows of
N. Guglielmi, C. Lubich, S. Sicilia, SIAM J. Numer. Anal. 61 (2023) and N. Guglielmi, C. Lubich, SIAM J. Numer. Anal. 63 (2025).
The stationarity residual returned with each solution is a first-order (KKT) certificate.
"""
from types import SimpleNamespace

import numpy as np


def make_problem(Phi, K, Q_K, S, eps_A, eps_B, N):
    """Bundles the data of one worst-case problem."""
    Phi = np.asarray(Phi, float)
    K = np.atleast_2d(K)
    return SimpleNamespace(Phi=Phi, K=K, Q_K=np.asarray(Q_K, float), S=np.asarray(S, float),
                           eps_A=float(eps_A), eps_B=float(eps_B), N=int(N), n=Phi.shape[0], m=K.shape[0])


def closed_loop(p, DA, DB):
    return p.Phi + DA - DB @ p.K


def wc_cost(p, DA, DB):
    """Worst-case cost over ||x0|| <= 1 for fixed perturbations, and the maximising x0."""
    F = closed_loop(p, DA, DB)
    P = p.S.copy()
    for _ in range(p.N):
        P = p.Q_K + F.T @ P @ F
    w, V = np.linalg.eigh((P + P.T) / 2)
    return w[-1], V[:, -1]


def rollout(p, DA, DB, x0):
    """Simulates the perturbed closed loop from x0. Returns (cost, list of states, list of state norms)."""
    F = closed_loop(p, DA, DB)
    x, J = np.asarray(x0, float), 0.0
    traj_s, traj_n = [x], [np.linalg.norm(x)]
    for _ in range(p.N):
        J += x @ p.Q_K @ x
        x = F @ x
        traj_s.append(x); traj_n.append(np.linalg.norm(x))
    return J + x @ p.S @ x, traj_s, traj_n


def wc_cost_and_grad(p, DA, DB):
    """Worst-case cost and its gradient with respect to (Delta_A, Delta_B) (adjoint sweep)."""
    F = closed_loop(p, DA, DB)
    J, x0 = wc_cost(p, DA, DB)
    xs = [x0]
    for _ in range(p.N):
        xs.append(F @ xs[-1])
    lam = p.S @ xs[-1]
    G = np.zeros_like(F)
    for k in reversed(range(p.N)):
        G += 2.0 * np.outer(lam, xs[k])
        lam = p.Q_K @ xs[k] + F.T @ lam
    return J, G, -G @ p.K.T


def proj_2(D, eps):
    """Projection onto the spectral-norm ball of radius eps (singular-value clipping)."""
    if eps == 0:
        return np.zeros_like(D)
    U, sv, Vt = np.linalg.svd(D, full_matrices=False)
    return (U * np.minimum(sv, eps)) @ Vt


def ascend(p, f, DA, DB, maxit=4000, tol=1e-10, theta=2.0, c_armijo=1e-4):
    """Projected gradient ascent of f(p, DA, DB) -> (value, grad_A, grad_B) with Armijo step control."""
    DA, DB = proj_2(DA, p.eps_A), proj_2(DB, p.eps_B)
    J, GA, GB = f(p, DA, DB)
    scale = p.eps_A + p.eps_B
    h = 0.5 * scale / (np.linalg.norm(GA) + np.linalg.norm(GB) + 1e-300)
    it = 0
    for it in range(maxit):
        rejected = False
        while True:
            nA, nB = proj_2(DA + h * GA, p.eps_A), proj_2(DB + h * GB, p.eps_B)
            pred = np.sum(GA * (nA - DA)) + np.sum(GB * (nB - DB))
            Jn, GAn, GBn = f(p, nA, nB)
            if Jn >= J + c_armijo * pred or h < 1e-16:
                break
            h /= theta
            rejected = True
        step = np.linalg.norm(nA - DA) + np.linalg.norm(nB - DB)
        DA, DB, J, GA, GB = nA, nB, Jn, GAn, GBn
        if not rejected:
            h *= theta
        if step < tol * scale:
            break
    tau = scale / (np.linalg.norm(GA) + np.linalg.norm(GB) + 1e-300)
    residual = (np.linalg.norm(DA - proj_2(DA + tau * GA, p.eps_A)) +
                np.linalg.norm(DB - proj_2(DB + tau * GB, p.eps_B))) / scale
    return SimpleNamespace(J=J, DA=DA, DB=DB, GA=GA, GB=GB, iters=it + 1, residual=residual)


def multistart(p, f, starts, maxit=4000):
    """Runs `ascend` from each named start and keeps the best result."""
    best = None
    for name, (DA, DB) in starts:
        r = ascend(p, f, DA, DB, maxit=maxit)
        r.start = name
        if best is None or r.J > best.J:
            best = r
    return best


def random_delta(rows, cols, rng=None):
    """Random matrix with largest singular value 1 and the others uniform in [0, 1]."""
    rng = np.random.default_rng() if rng is None else rng
    u, s, vh = np.linalg.svd(rng.standard_normal((rows, cols)), full_matrices=False)
    s_rand = rng.uniform(0, 1, size=s.shape)
    if s_rand.size:
        s_rand[0] = 1.0
    return (u * s_rand) @ vh


def _batch_delta(num, rows, cols, eps, rng):
    if eps == 0:
        return np.zeros((num, rows, cols))
    U, s, Vh = np.linalg.svd(rng.standard_normal((num, rows, cols)), full_matrices=False)
    s_rand = rng.uniform(0, 1, size=s.shape); s_rand[:, 0] = 1.0
    return eps * (U * s_rand[:, None, :]) @ Vh


def monte_carlo_best(p, num_sims=1000, rng=None):
    """Best of `num_sims` random admissible LTI perturbations (exact worst x0 for each). Used as a warm start.

    Returns:
        tuple: (cost, Delta_A, Delta_B, x0).
    """
    rng = np.random.default_rng() if rng is None else rng
    DA = _batch_delta(num_sims, p.n, p.n, p.eps_A, rng)
    DB = _batch_delta(num_sims, p.n, p.m, p.eps_B, rng)
    F = p.Phi[None] + DA - DB @ p.K[None]
    P = np.broadcast_to(p.S, F.shape).copy()
    Ft = np.transpose(F, (0, 2, 1))
    for _ in range(p.N):
        P = p.Q_K[None] + Ft @ P @ F
    w, V = np.linalg.eigh((P + np.transpose(P, (0, 2, 1))) / 2)
    i = int(np.argmax(w[:, -1]))
    return w[i, -1], DA[i], DB[i], V[i, :, -1]


def extract_from_sdp(p, sdp):
    """Warm start from the principal eigenvector of the SDP solution: Delta_A = V_A X^+, Delta_B = V_B (K X)^+."""
    L = sdp['lift']
    w, V = np.linalg.eigh(sdp['Z'])
    z = V[:, -1] * np.sqrt(max(w[-1], 0.0))
    X = np.column_stack([H @ z for H in L.H_x[:p.N]])
    VA = p.eps_A * np.column_stack([H @ z for H in L.H_wA])
    VB = p.eps_B * np.column_stack([H @ z for H in L.H_wB])
    DA = VA @ np.linalg.pinv(X)
    DB = VB @ np.linalg.pinv(p.K @ X)
    return proj_2(DA, p.eps_A), proj_2(DB, p.eps_B)


def worst_case_search(p, sdp=None, mc=None, prev=None, ps=None, n_random=6, rng=None, maxit=4000):
    """Worst-case LTI perturbation by projected gradient ascent from several warm starts.

    Warm starts: principal eigenvector of the SDP solution (`sdp`), best Monte Carlo sample (`mc`, as returned by
    monte_carlo_best), the solution at a neighbouring horizon (`prev`), the spectral-radius maximiser (`ps`), and
    `n_random` random boundary perturbations.

    Returns:
        SimpleNamespace with J (cost), DA, DB, x0, Phi_wc, traj_s, traj_n, residual (stationarity) and start (name).
    """
    rng = np.random.default_rng() if rng is None else rng
    if p.eps_A == 0 and p.eps_B == 0:
        DA, DB = np.zeros((p.n, p.n)), np.zeros((p.n, p.m))
        r = SimpleNamespace(J=wc_cost(p, DA, DB)[0], DA=DA, DB=DB, residual=0.0, start="nominal")
    else:
        starts = []
        if sdp is not None:
            starts.append(("sdp", extract_from_sdp(p, sdp)))
        if mc is not None:
            starts.append(("monte-carlo", (mc[1], mc[2])))
        if prev is not None:
            starts.append(("continuation", (prev.DA, prev.DB)))
        if ps is not None:
            starts.append(("spectral-radius", (ps.DA, ps.DB)))
        for i in range(n_random):
            starts.append((f"random{i}", (_batch_delta(1, p.n, p.n, p.eps_A, rng)[0],
                                          _batch_delta(1, p.n, p.m, p.eps_B, rng)[0])))
        r = multistart(p, wc_cost_and_grad, starts, maxit=maxit)
    J, x0 = wc_cost(p, r.DA, r.DB)
    cost, traj_s, traj_n = rollout(p, r.DA, r.DB, x0)
    return SimpleNamespace(J=cost, DA=r.DA, DB=r.DB, x0=x0, Phi_wc=closed_loop(p, r.DA, r.DB),
                           traj_s=traj_s, traj_n=traj_n, residual=r.residual, start=r.start)


def run_monte_carlo_simulations(x0, p, num_sims=120, rng=None):
    """Nominal trajectory from x0 and `num_sims` trajectories under random admissible LTI perturbations.

    Returns:
        tuple: (nominal states, nominal norms, nominal cost, list of sampled state lists, list of sampled norm lists).
    """
    rng = np.random.default_rng() if rng is None else rng
    zA, zB = np.zeros((p.n, p.n)), np.zeros((p.n, p.m))
    cost_nom, traj_nom_s, traj_nom_n = rollout(p, zA, zB, x0)
    sim_s, sim_n = [], []
    for _ in range(num_sims):
        DA = p.eps_A * random_delta(p.n, p.n, rng)
        DB = p.eps_B * random_delta(p.n, p.m, rng)
        _, t_s, t_n = rollout(p, DA, DB, x0)
        sim_s.append(t_s); sim_n.append(t_n)
    return traj_nom_s, traj_nom_n, cost_nom, sim_s, sim_n
