"""PEP and IQC semidefinite programs over the lifted Gram matrix Z.

Lifted vector z = [x_0; w_A,0..N-1; w_B,0..N-1] with pre-scaled perturbation signals
v_A,k = eps_A w_A,k and v_B,k = eps_B w_B,k, so that

    x_{k+1} = Phi x_k + eps_A w_A,k - eps_B w_B,k,     x_k = H_x,k z.

* PEP (LTI, operator interpolation):  G_x - G_wA >= 0  and  G_Kx - G_wB >= 0  (N x N Gram matrices).
* IQC (LTV, pointwise-in-time norm bounds): only the diagonals of these matrices are constrained.
"""
from types import SimpleNamespace

import numpy as np
import cvxpy as cp

_original_solve = cp.Problem.solve
_preferred_solver = None


def enable_default_solver(verbose=True):
    """Patch cvxpy.Problem.solve to prefer MOSEK, then SCS, then the CVXPY default."""
    global _preferred_solver
    if cp.MOSEK in cp.installed_solvers():
        _preferred_solver, msg = cp.MOSEK, "MOSEK"
    elif cp.SCS in cp.installed_solvers():
        _preferred_solver, msg = cp.SCS, "SCS (MOSEK not available)"
    else:
        _preferred_solver, msg = None, "default CVXPY solver (MOSEK and SCS not available)"

    def _global_solve(self, *args, **kwargs):
        if 'solver' not in kwargs and _preferred_solver is not None:
            kwargs['solver'] = _preferred_solver
        return _original_solve(self, *args, **kwargs)

    cp.Problem.solve = _global_solve
    if verbose:
        print(f"CVXPY is now globally configured to use {msg} by default.")


def build_lift(Phi, K, Q_K, S, eps_A, eps_B, N):
    """Block-selector and state-extraction matrices of the lifted problem.

    Returns:
        SimpleNamespace with n_tot, E0, H_x (list of N+1 matrices n x n_tot), H_wA, H_wB (lists of N),
        and the symmetric cost matrix M_J such that J_N = Tr(M_J Z).
    """
    n = Phi.shape[0]
    n_tot = n * (1 + 2 * N)
    E0 = np.zeros((n, n_tot)); E0[:, :n] = np.eye(n)
    H_wA, H_wB = [], []
    for k in range(N):
        a = np.zeros((n, n_tot)); a[:, n + k * n: n + (k + 1) * n] = np.eye(n); H_wA.append(a)
        b = np.zeros((n, n_tot)); b[:, n + (N + k) * n: n + (N + k + 1) * n] = np.eye(n); H_wB.append(b)
    H_x = [E0]
    for k in range(N):
        H_x.append(Phi @ H_x[-1] + eps_A * H_wA[k] - eps_B * H_wB[k])
    M_J = sum(H.T @ Q_K @ H for H in H_x[:N]) + H_x[N].T @ S @ H_x[N]
    return SimpleNamespace(n=n, N=N, n_tot=n_tot, E0=E0, H_x=H_x, H_wA=H_wA, H_wB=H_wB, M_J=(M_J + M_J.T) / 2)


def _block_gram(blocks, Z):
    """N x N affine expression G[i, j] = Tr(blocks[i] Z blocks[j]^T)."""
    G = 0
    for r in range(blocks[0].shape[0]):
        R_r = np.stack([b[r] for b in blocks])
        G = G + R_r @ Z @ R_r.T
    return G


def solve_sdp(Phi, K, Q_K, S, eps_A, eps_B, N, relaxation="pep"):
    """Solves the relaxed PEP SDP (``relaxation="pep"``) or the IQC/LTV SDP (``relaxation="iqc"``).

    Returns:
        dict with 'cost' (upper bound), 'Z', 'rank' (numerical rank, 1e-4 relative threshold),
        'lift' (see build_lift) and 'status'.

    Raises:
        ValueError: If the SDP solver fails.
    """
    L = build_lift(Phi, K, Q_K, S, eps_A, eps_B, N)
    n = L.n
    Z = cp.Variable((L.n_tot, L.n_tot), PSD=True)
    M_A = _block_gram(L.H_x[:N], Z) - _block_gram(L.H_wA, Z)
    M_B = _block_gram([K @ H for H in L.H_x[:N]], Z) - _block_gram(L.H_wB, Z)
    cons = [cp.trace(Z[:n, :n]) <= 1]
    if relaxation == "pep":
        cons += [(M_A + M_A.T) / 2 >> 0, (M_B + M_B.T) / 2 >> 0]
    elif relaxation == "iqc":
        cons += [cp.diag(M_A) >= 0, cp.diag(M_B) >= 0]
    else:
        raise ValueError(f"unknown relaxation {relaxation!r}")
    prob = cp.Problem(cp.Maximize(cp.trace(L.M_J @ Z)), cons)
    try:
        prob.solve()
    except cp.error.SolverError:
        raise ValueError("SDP Solver Error")
    if Z.value is None:
        raise ValueError(f"SDP Solver Error (status {prob.status})")
    Z_val = (Z.value + Z.value.T) / 2
    eigs = np.linalg.eigvalsh(Z_val)
    rank = int(np.sum(eigs > 1e-4 * max(np.max(eigs), 1e-12)))
    return {'cost': prob.value, 'Z': Z_val, 'rank': rank, 'lift': L, 'status': prob.status}
