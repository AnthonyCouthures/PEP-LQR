"""Worst-case LTI perturbations of finite-horizon LQR: a Performance Estimation Problem approach."""
from .system import build_system_matrices, solve_nominal_lqr
from .sdp import enable_default_solver, build_lift, solve_sdp
from .worst_case import (make_problem, closed_loop, wc_cost, wc_cost_and_grad, rollout, proj_2, ascend,
                         monte_carlo_best, extract_from_sdp, worst_case_search, run_monte_carlo_simulations)
from .spectral import (structured_ps_radius, real_distance, structured_pseudospectrum, plot_pseudospectrum,
                       generate_perturbation_cloud)
from .pipeline import solve_pep_2d, compare_bounds
from .plotting import set_paper_style, plot_function, plot_pole_migration, plot_comparison
