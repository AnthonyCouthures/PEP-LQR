"""Interactive ipywidgets dashboard."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import ipywidgets as widgets

from .pipeline import solve_pep_2d
from .spectral import generate_perturbation_cloud, structured_pseudospectrum, plot_pseudospectrum


def update_plot(A11, A12, A21, A22, B1, B2, Q11, Q22, R11, S11, S22, eps_A, eps_B, N, exact_border, num_sims, seed=0):
    """
    Recomputes and draws the dashboard: time evolution, nominal flow, worst-case flow, and the spectral picture.
    Intended to be used with ipywidgets' interactive_output.

    Args:
        exact_border (bool): Draw the exact structured pseudospectrum (slower) instead of the random cloud.
        num_sims (int): Monte Carlo samples (warm start of the worst-case search and plotted trajectories).
    """
    with mpl.rc_context({'text.usetex': False}):     # independent of the paper style (LaTeX) set for the figures
        _update_plot(A11, A12, A21, A22, B1, B2, Q11, Q22, R11, S11, S22, eps_A, eps_B, N, exact_border, num_sims, seed)


def _update_plot(A11, A12, A21, A22, B1, B2, Q11, Q22, R11, S11, S22, eps_A, eps_B, N, exact_border, num_sims, seed):
    rng = np.random.default_rng(seed)
    res, msg = solve_pep_2d(A11, A12, A21, A22, B1, B2, Q11, Q22, R11, S11, S22, eps_A, eps_B, int(N),
                            num_sims=num_sims, rng=rng)

    fig, ax_dict = plt.subplot_mosaic([['ax1', 'ax4'], ['ax2', 'ax4'], ['ax3', 'ax4']], figsize=(15, 9))
    ax1, ax2, ax3, ax4 = ax_dict['ax1'], ax_dict['ax2'], ax_dict['ax3'], ax_dict['ax4']

    if res is None:
        fig.suptitle(msg, color='red', fontsize=18, fontweight='bold')
        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xticks([]); ax.set_yticks([])
        plt.show(); return

    # --- Subplot 1: Time Evolution ---
    for traj in res['sim_n']:
        ax1.plot(range(res['N'] + 1), traj, color='#1f77b4', alpha=0.08)
    ax1.plot(range(res['N'] + 1), res['traj_nom_n'], color='#2ca02c', lw=3, label="Nominal Trajectory", zorder=5)
    ax1.plot(range(res['N'] + 1), res['traj_ext_n'], color='#d62728', lw=3, linestyle='--',
             label="Worst-Case (LTI)", zorder=6)
    ax1.set_title(r"Time Evolution (State Norm $\|x_k\|_2$)", fontsize=13)
    ax1.set_xlabel("Time Step $k$")
    ax1.set_ylabel(r"$\|x_k\|$")
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend(loc='upper right')

    t_str = "Exact Bound" if res['is_tight'] else "LTV Conservatism"

    # Phase plot preparation
    all_x = [s[0] for t in res['sim_s'] for s in t] + [s[0] for s in res['traj_nom_s']] + [s[0] for s in res['traj_ext_s']]
    all_y = [s[1] for t in res['sim_s'] for s in t] + [s[1] for s in res['traj_nom_s']] + [s[1] for s in res['traj_ext_s']]
    min_x, max_x, min_y, max_y = min(all_x), max(all_x), min(all_y), max(all_y)
    pad_x, pad_y = max(0.1 * (max_x - min_x), 0.1), max(0.1 * (max_y - min_y), 0.1)
    nx1, nx2 = [s[0] for s in res['traj_nom_s']], [s[1] for s in res['traj_nom_s']]
    ex1, ex2 = [s[0] for s in res['traj_ext_s']], [s[1] for s in res['traj_ext_s']]
    Y, X = np.mgrid[min_y - pad_y: max_y + pad_y: 20j, min_x - pad_x: max_x + pad_x: 20j]

    # --- Subplot 2: Nominal Flow ---
    ax2.set_title(r"Nominal Flow Field & Phase Portrait", fontsize=13)
    U_nom = (res['Phi'][0, 0] - 1) * X + res['Phi'][0, 1] * Y
    V_nom = res['Phi'][1, 0] * X + (res['Phi'][1, 1] - 1) * Y
    ax2.quiver(X, Y, U_nom, V_nom, color='#2ca02c', alpha=0.3)
    for traj in res['sim_s']:
        ax2.plot([s[0] for s in traj], [s[1] for s in traj], color='#1f77b4', alpha=0.08, lw=1)
    ax2.plot(nx1, nx2, color='#2ca02c', lw=3, marker='o', markersize=4, zorder=5, label="Nominal Traj")
    ax2.plot(ex1, ex2, color='#d62728', lw=3, linestyle='--', marker='x', markersize=6, zorder=6, label="Worst-Case Traj")
    ax2.legend(loc='best')

    # --- Subplot 3: Worst-Case Flow ---
    ax3.set_title("Worst-Case Flow Field (LTI)", fontsize=13)
    U_wc = (res['Phi_wc'][0, 0] - 1) * X + res['Phi_wc'][0, 1] * Y
    V_wc = res['Phi_wc'][1, 0] * X + (res['Phi_wc'][1, 1] - 1) * Y
    ax3.quiver(X, Y, U_wc, V_wc, color='indigo', alpha=0.20)
    ax3.plot(nx1, nx2, color='#2ca02c', lw=2, linestyle=':', marker='o', markersize=4, alpha=0.6, label="Nominal Traj")
    ax3.plot(ex1, ex2, color='#d62728', lw=3, linestyle='--', marker='x', markersize=6, zorder=6, label="Worst-Case Traj")
    ax3.legend(loc='best')

    for ax in [ax2, ax3]:
        ax.set_xlim(min_x - pad_x, max_x + pad_x); ax.set_ylim(min_y - pad_y, max_y + pad_y)
        ax.set_xlabel("State $x_1$"); ax.set_ylabel("State $x_2$")
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.axhline(0, color='black', lw=0.5); ax.axvline(0, color='black', lw=0.5)

    # --- Subplot 4: Structured pseudospectrum and eigenvalues ---
    ax4.set_title("Structured Uncertainty & Worst-Case Eigenvalues", fontsize=13)
    eigs_nom = np.linalg.eigvals(res['Phi'])
    eigs_wc = np.linalg.eigvals(res['Phi_wc'])
    theta_uc = np.linspace(0, 2 * np.pi, 200)
    ax4.plot(np.cos(theta_uc), np.sin(theta_uc), 'k-', lw=1.5, alpha=0.4, label='Unit Circle')

    if exact_border:
        plot_pseudospectrum(ax4, structured_pseudospectrum(res['problem'], n_re=121, n_real=301), color='#ffaa0f', alpha=0.45)
        ax4.plot([], [], color='#ffaa0f', lw=6, alpha=0.6, label='Structured pseudospectrum (exact)')
    else:
        eigs_batch = generate_perturbation_cloud(res['Phi'], res['K'], eps_A, eps_B, N_samples=6000, rng=rng)
        ax4.scatter(eigs_batch.real, eigs_batch.imag, c='black', s=2, alpha=0.35, zorder=2, edgecolors='none',
                    label='Structured Cloud')

    for en, ew in zip(np.sort_complex(eigs_nom), np.sort_complex(eigs_wc)):
        ax4.plot([en.real, ew.real], [en.imag, ew.imag], 'k:', lw=1.8, alpha=0.6, zorder=4)
    ax4.scatter(eigs_nom.real, eigs_nom.imag, s=110, c='#2ca02c', edgecolors='white', marker='o', zorder=5,
                label='Nominal Eigs')
    ax4.scatter(eigs_wc.real, eigs_wc.imag, s=150, c='#d62728', edgecolors='white', marker='X', zorder=6,
                label='Worst-Case (LTI) Eigs')
    ax4.set_xlim(-1.2, 1.2); ax4.set_ylim(-1.2, 1.2)
    ax4.set_aspect('equal')
    ax4.set_xlabel("Real Part"); ax4.set_ylabel("Imaginary Part")
    ax4.grid(True, linestyle=':', alpha=0.7)
    ax4.axhline(0, color='black', lw=0.5); ax4.axvline(0, color='black', lw=0.5)
    ax4.legend(loc='best', fontsize=8, framealpha=0.9, edgecolor='black')

    lines = [
        (f"Verification Dashboard ({res['mode_str']}):", 'black'),
        (f"• Original SDP Rank:      {res['rank_Z']} ({t_str})", 'black'),
        (f"• SDP Upper Bound (LTV):  {res['gamma_N']:.4f}", 'purple'),
        (f"• LTI Lower Bound:        {res['cost_ext']:.4f}", '#d62728'),
        (f"• Stationarity residual:  {res['residual']:.1e}", 'black'),
    ]
    for i, (txt, col) in enumerate(lines):
        ax4.text(0.025, 1.36 - i * 0.05, txt, color=col, transform=ax4.transAxes,
                 fontsize=11, fontweight='bold', family='monospace', zorder=11)

    plt.tight_layout(pad=2.0)
    plt.show()


sl_layout, sl_style = widgets.Layout(width='95%'), {'description_width': '40%'}


def make_slider(desc, val, min_v, max_v, step_v=0.01):
    """Creates a FloatSlider widget with the dashboard layout and style."""
    return widgets.FloatSlider(value=val, min=min_v, max=max_v, step=step_v, description=desc, style=sl_style,
                               layout=sl_layout)
