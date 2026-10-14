"""End-to-end computations used by the dashboard and the paper figures."""
import time
import warnings

import numpy as np

from .system import build_system_matrices, solve_nominal_lqr
from .sdp import solve_sdp
from .worst_case import make_problem, monte_carlo_best, worst_case_search, run_monte_carlo_simulations, wc_cost
from .spectral import structured_ps_radius, spectral_radius_maximisers


def _check_worst_case(p, wc, sdp_cost=None, rtol=1e-3):
    """Feasibility of the extracted perturbation, and consistency with the SDP upper bound."""
    assert np.linalg.norm(wc.DA, 2) <= p.eps_A * (1 + 1e-9) + 1e-14
    assert np.linalg.norm(wc.DB, 2) <= p.eps_B * (1 + 1e-9) + 1e-14
    if sdp_cost is not None and wc.J > sdp_cost * (1 + rtol):
        warnings.warn(f"N={p.N}: LTI worst case {wc.J:.6f} exceeds the SDP bound {sdp_cost:.6f} "
                      f"by more than {rtol:.0e} (inaccurate SDP solve?)")


def solve_pep_2d(A11, A12, A21, A22, B1, B2, Q11, Q22, R11, S11, S22, eps_A, eps_B, N, num_sims=1000,
                 rng=None, prev=None):
    """
    Solves the PEP problem for a 2D system: SDP upper bound, and worst-case LTI perturbation by projected
    gradient ascent (warm-started from the SDP solution, the best of `num_sims` Monte Carlo samples, the
    spectral-radius maximiser, an optional previous solution `prev` and random perturbations).

    Returns:
        tuple: (results dict or None, status message).
    """
    rng = np.random.default_rng() if rng is None else rng
    A, B, Q, R, S_user = build_system_matrices(A11, A12, A21, A22, B1, B2, Q11, Q22, R11, S11, S22)
    try:
        _, K, Phi, Q_K = solve_nominal_lqr(A, B, Q, R)
    except ValueError as e:
        return None, str(e)

    p = make_problem(Phi, K, Q_K, S_user, eps_A, eps_B, N)
    try:
        sdp = solve_sdp(Phi, K, Q_K, S_user, eps_A, eps_B, N, relaxation="pep")
    except ValueError as e:
        return None, str(e)

    mc = monte_carlo_best(p, num_sims, rng)
    ps = structured_ps_radius(p, rng=rng) if (eps_A > 0 or eps_B > 0) else None
    wc = worst_case_search(p, sdp=sdp, mc=mc, prev=prev, ps=ps, rng=rng)
    _check_worst_case(p, wc, sdp['cost'])

    t_nom_s, t_nom_n, c_nom, sim_s, sim_n = run_monte_carlo_simulations(wc.x0, p, num_sims=num_sims, rng=rng)

    return {
        'N': N, 'mode_str': "User S", 'wc_method': "gradient ascent",
        'gamma_N': sdp['cost'], 'cost_ext': wc.J, 'residual': wc.residual,
        'traj_ext_s': wc.traj_s, 'traj_ext_n': wc.traj_n,
        'rank_Z': sdp['rank'], 'is_tight': (sdp['rank'] == 1),
        'traj_nom_s': t_nom_s, 'traj_nom_n': t_nom_n, 'cost_nom': c_nom,
        'sim_s': sim_s, 'sim_n': sim_n,
        'Phi': Phi, 'Phi_wc': wc.Phi_wc, 'K': K, 'problem': p, 'wc': wc,
    }, "Success"


def compare_bounds(A, B, Q, R, S, eps_A, eps_B, horizons, num_sims=8000, rng=None, verbose=True):
    """IQC (LTV) bound, relaxed PEP bound, LTI worst case (projected gradient ascent), best Monte Carlo sample, and
    the cost of the spectral-radius maximising perturbation (pseudospectral approach of Guglielmi, Lubich and
    Sicilia), for each horizon in `horizons`. Consecutive horizons are warm-started from each other.

    The spectral-radius maximiser is not unique; the pseudospectral cost is the largest finite-horizon cost among
    the sampled maximisers (the most favourable choice for that approach).

    Returns:
        dict of lists: horizons, iqc, pep, lti, mc, ps, rank, residual, t_sdp, t_wc.
    """
    rng = np.random.default_rng() if rng is None else rng
    _, K, Phi, Q_K = solve_nominal_lqr(A, B, Q, R)
    out = {k: [] for k in ['horizons', 'iqc', 'pep', 'lti', 'mc', 'ps', 'rank', 'residual', 't_sdp', 't_wc']}
    prev, ps, maximisers = None, None, None
    for N in horizons:
        p = make_problem(Phi, K, Q_K, S, eps_A, eps_B, N)
        if ps is None:
            ps = structured_ps_radius(p, rng=rng)
            maximisers = spectral_radius_maximisers(p, rng=rng)
            if verbose:
                print(f"Spectral-radius maximisers: {len(maximisers)} sampled with rho = {ps.rho:.6f}\n")
        t0 = time.time()
        iqc = solve_sdp(Phi, K, Q_K, S, eps_A, eps_B, N, relaxation="iqc")
        pep = solve_sdp(Phi, K, Q_K, S, eps_A, eps_B, N, relaxation="pep")
        t_sdp = time.time() - t0
        mc = monte_carlo_best(p, num_sims, rng)
        t0 = time.time()
        wc = worst_case_search(p, sdp=pep, mc=mc, prev=prev, ps=ps, rng=rng)
        t_wc = time.time() - t0
        _check_worst_case(p, wc, pep['cost'])
        prev = wc
        cost_ps = max(wc_cost(p, DA, DB)[0] for DA, DB in maximisers)
        for key, val in [('horizons', N), ('iqc', iqc['cost']), ('pep', pep['cost']), ('lti', wc.J), ('mc', mc[0]),
                         ('ps', cost_ps), ('rank', pep['rank']), ('residual', wc.residual), ('t_sdp', t_sdp),
                         ('t_wc', t_wc)]:
            out[key].append(val)
        if verbose:
            print(f"N={N:2d} | Rank(Z): {pep['rank']} | SDP time: {t_sdp:.2f}s | Gradient ascent: {t_wc:.2f}s "
                  f"(residual {wc.residual:.0e})")
            print(f"      Upper bounds -> IQC (LTV): {iqc['cost']:.4f} | PEP (relaxed): {pep['cost']:.4f}")
            print(f"      Lower bounds -> LTI worst case: {wc.J:.4f} | Max. spectral radius: {cost_ps:.4f} "
                  f"| MC: {mc[0]:.4f}\n")
    return out
