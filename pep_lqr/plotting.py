"""Paper figures: Fig. 1 (spectral picture and trajectories), Fig. 2 (pole migration), Fig. 3 (bound comparison)."""
import io
import os
import shutil

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .spectral import structured_pseudospectrum, plot_pseudospectrum, structured_ps_radius

# ==========================================
# Plotting configuration
# ==========================================
FIG_SIZE_RECT = (2.3, 2.3)
FIG_SIZE_SQ = (2.3, 2.3)
FIG_DPI = 300

C_NOM = '#5B9A64'
C_WC = '#C1121F'
C_SIM = '#006AA3'
C_FLOW_WC = '#d62728'
C_PSEUDO = '#ffaa0f'

LW_NOM = .75
LW_WC = .75
LW_SIM = 0.5

LS_NOM = '-'
LS_WC = 'dashed'
LS_NOM_ALT = '-'

M_NOM = 'o'
M_WC = 's'
M_WC_EIG = '*'
MS_NOM = 4
MS_WC = 3
MS_NOM_EIG = 60
MS_WC_EIG = 100

ALPHA_SIM = 0.08
ALPHA_FLOW_NOM = 0.75
ALPHA_FLOW_WC = 0.75
FLOW_GRID = 22          # quiver grid density (per axis)

ADD_ARROWS = True
ARROW_STYLE = '->'
ARROW_SIZE = 10

FIG_DIR = "figures"   # all paper figures are saved in this folder


def _fig_path(filename):
    os.makedirs(FIG_DIR, exist_ok=True)
    return os.path.join(FIG_DIR, filename)


def _latex_works():
    """True if matplotlib can render text with LaTeX and the newtxtext package."""
    if not (shutil.which("latex") and shutil.which("dvipng")):
        return False
    try:
        with mpl.rc_context({'text.usetex': True, 'text.latex.preamble': r'\usepackage{newtxtext}'}):
            fig = plt.figure()
            fig.text(0.5, 0.5, r"Test $\varepsilon_A$")
            fig.savefig(io.BytesIO(), format="png")
            plt.close(fig)
        return True
    except Exception:
        plt.close('all')
        return False


def set_paper_style(use_tex="auto"):
    """Matplotlib settings used for the paper figures.

    Args:
        use_tex: True / False, or "auto" to render text with LaTeX (newtxtext font, as in the paper) whenever a working
            LaTeX installation is found, and with matplotlib's mathtext otherwise (e.g. on Colab).
    """
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams.update({
        'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 7,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'xtick.major.size': 3.5, 'xtick.minor.size': 2.0, 'xtick.major.width': 0.8, 'xtick.minor.width': 0.6,
        'xtick.direction': 'out', 'xtick.top': False, 'xtick.bottom': True,
        'ytick.direction': 'out', 'ytick.left': True, 'ytick.right': False,
        'grid.linewidth': .8, 'grid.color': 'lightgrey', 'grid.linestyle': '6,2',
        'axes.linewidth': 0.5, 'axes.labelpad': 4.0, 'axes.spines.top': True, 'axes.spines.right': True,
        'axes.spines.left': True, 'axes.spines.bottom': True, 'axes.axisbelow': True, 'axes.xmargin': 0,
    })
    if use_tex == "auto":
        use_tex = _latex_works()
    if use_tex:
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['text.latex.preamble'] = r'\usepackage{newtxtext}'
        print("Paper style: text rendered with LaTeX (newtxtext).")
    else:
        print("Paper style: LaTeX not available, text rendered with matplotlib's mathtext.")


def _speed_cmap(color):
    """Light tint -> color -> dark shade of `color` (slow -> fast)."""
    c = np.array(mpl.colors.to_rgb(color))
    return mpl.colors.LinearSegmentedColormap.from_list("speed", [0.72 + 0.28 * c, c, 0.6 * c])


def flow_quiver(ax, Phi, X, Y, color, alpha):
    """Increment field x -> (Phi - I) x on the grid (X, Y).

    All arrows have the same length (directions normalised in screen units), which keeps the field smooth; the colour
    encodes the local speed, from a light tint (slow) to a dark shade (fast) of `color`, so that slow invariant
    manifolds appear as light bands."""
    U = (Phi[0, 0] - 1) * X + Phi[0, 1] * Y
    V = Phi[1, 0] * X + (Phi[1, 1] - 1) * Y
    Us, Vs = U / np.ptp(X), V / np.ptp(Y)                     # screen-proportional components
    speed = np.hypot(Us, Vs)
    with np.errstate(invalid='ignore', divide='ignore'):
        Un = np.where(speed > 0, Us / speed, 0.0)
        Vn = np.where(speed > 0, Vs / speed, 0.0)
    level = np.sqrt(speed / max(np.max(speed), 1e-300))
    ax.quiver(X, Y, Un, Vn, level, cmap=_speed_cmap(color), clim=(0, 1), alpha=alpha, pivot='mid', angles='uv',
              scale=1.25 * FLOW_GRID, scale_units='width', width=0.004, headwidth=4, headlength=4.5, zorder=1)


def add_trajectory_arrows(ax, x_coords, y_coords, color, lw, ls, zorder):
    if not ADD_ARROWS:
        return
    for i in range(len(x_coords) - 1):
        ax.annotate('', xy=(x_coords[i + 1], y_coords[i + 1]), xytext=(x_coords[i], y_coords[i]),
                    arrowprops=dict(arrowstyle=ARROW_STYLE, color=color, lw=lw, linestyle=ls,
                                    mutation_scale=ARROW_SIZE, shrinkA=0, shrinkB=0),
                    zorder=zorder)


# ==========================================
# Fig. 1
# ==========================================
def plot_function(res, name):
    """Four Fig. 1 panels for one system: time evolution, nominal flow, worst-case flow, and the spectral picture
    (exact structured pseudospectrum, nominal and worst-case eigenvalues)."""
    all_x = [s[0] for t in res['sim_s'] for s in t] + [s[0] for s in res['traj_nom_s']] + [s[0] for s in res['traj_ext_s']]
    all_y = [s[1] for t in res['sim_s'] for s in t] + [s[1] for s in res['traj_nom_s']] + [s[1] for s in res['traj_ext_s']]
    min_x, max_x, min_y, max_y = min(all_x), max(all_x), min(all_y), max(all_y)
    pad_x, pad_y = max(0.15 * (max_x - min_x), 0.1), max(0.15 * (max_y - min_y), 0.1)

    nx1, nx2 = [s[0] for s in res['traj_nom_s']], [s[1] for s in res['traj_nom_s']]
    ex1, ex2 = [s[0] for s in res['traj_ext_s']], [s[1] for s in res['traj_ext_s']]
    Y, X = np.mgrid[min_y - pad_y: max_y + pad_y: FLOW_GRID * 1j, min_x - pad_x: max_x + pad_x: FLOW_GRID * 1j]

    # Plot 1: time evolution
    fig1, ax1 = plt.subplots(figsize=FIG_SIZE_RECT, dpi=FIG_DPI)
    for traj in res['sim_n']:
        ax1.plot(range(res['N'] + 1), traj, color=C_SIM, alpha=ALPHA_SIM)
    ax1.plot(range(res['N'] + 1), res['traj_nom_n'], color=C_NOM, lw=LW_NOM, ls=LS_NOM, label="Nominal Trajectory", zorder=5)
    ax1.plot(range(res['N'] + 1), res['traj_ext_n'], color=C_WC, lw=LW_WC, ls=LS_WC, label="Worst-Case", zorder=6)
    ax1.set_xlabel("Time Step $k$")
    ax1.set_ylabel(r"$\|x_k\|$")
    ax1.grid(True)
    ax1.legend(loc='upper right')

    t_str = "Exact Bound" if res['is_tight'] else "LTV Conservatism"
    sanity_text = (
        f"Verification Dashboard:\n"
        f"$\\bullet$ SDP Upper Bound: {res['gamma_N']:.4f}\n"
        f"$\\bullet$ LTI Lower Bound: {res['cost_ext']:.4f}\n"
        f"$\\bullet$ Rank(Z): {res['rank_Z']} ({t_str})"
    )
    t_color = 'green' if res['is_tight'] else 'red'
    ax1.text(0.02, 0.02, sanity_text, transform=ax1.transAxes, fontsize=5, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=t_color, lw=2))
    fig1.savefig(_fig_path(f"{name}_time_evolution.pdf"), bbox_inches='tight', format="pdf")
    plt.show()
    plt.close(fig1)

    # Plot 2: nominal flow
    fig2, ax2 = plt.subplots(figsize=FIG_SIZE_RECT, dpi=FIG_DPI)
    flow_quiver(ax2, res['Phi'], X, Y, C_NOM, ALPHA_FLOW_NOM)
    for traj in res['sim_s']:
        ax2.plot([s[0] for s in traj], [s[1] for s in traj], color=C_SIM, alpha=ALPHA_SIM, lw=LW_SIM)
    add_trajectory_arrows(ax2, nx1, nx2, C_NOM, LW_NOM, LS_NOM, zorder=5)
    add_trajectory_arrows(ax2, ex1, ex2, C_WC, LW_WC, LS_WC, zorder=6)
    ax2.plot([], [], color=C_NOM, lw=LW_NOM, ls=LS_NOM, marker=M_NOM, markersize=MS_NOM, label="Nominal Traj")
    ax2.plot([], [], color=C_WC, lw=LW_WC, ls=LS_WC, marker=M_WC, markersize=MS_WC, label="Worst-Case Traj")
    ax2.legend(loc='best')
    ax2.set_xlim(min_x - pad_x, max_x + pad_x); ax2.set_ylim(min_y - pad_y, max_y + pad_y)
    ax2.set_xlabel("State $x_1$"); ax2.set_ylabel("State $x_2$")
    ax2.grid(True)
    ax2.axhline(0, color='black', lw=0.5); ax2.axvline(0, color='black', lw=0.5)
    fig2.savefig(_fig_path(f"{name}_nominal_flow.pdf"), bbox_inches='tight', format="pdf")
    plt.show()
    plt.close(fig2)

    # Plot 3: worst-case flow
    fig3, ax3 = plt.subplots(figsize=FIG_SIZE_RECT, dpi=FIG_DPI)
    ax3.set_xlim(min_x - pad_x, max_x + pad_x); ax3.set_ylim(min_y - pad_y, max_y + pad_y)
    ax3.set_xlabel("State $x_1$"); ax3.set_ylabel("State $x_2$")
    ax3.grid(True)
    ax3.axhline(0, color='black', lw=0.5); ax3.axvline(0, color='black', lw=0.5)
    flow_quiver(ax3, res['Phi_wc'], X, Y, C_FLOW_WC, ALPHA_FLOW_WC)
    ax3.scatter(nx1, nx2, color=C_NOM, marker=M_NOM, s=MS_NOM, alpha=1, zorder=5)
    add_trajectory_arrows(ax3, nx1, nx2, C_NOM, LW_NOM, LS_NOM_ALT, zorder=5)
    ax3.scatter(ex1, ex2, color=C_WC, marker=M_WC, s=MS_WC, zorder=6)
    add_trajectory_arrows(ax3, ex1, ex2, C_WC, LW_WC, LS_WC, zorder=6)
    fig3.savefig(_fig_path(f"{name}_worst_case_flow.pdf"), bbox_inches='tight', format="pdf")
    plt.show()
    plt.close(fig3)

    # Plot 4: exact structured pseudospectrum and eigenvalues
    fig4, ax4 = plt.subplots(figsize=FIG_SIZE_SQ, dpi=FIG_DPI)
    eigs_nom = np.linalg.eigvals(res['Phi'])
    eigs_wc = np.linalg.eigvals(res['Phi_wc'])
    theta_uc = np.linspace(0, 2 * np.pi, 200)
    ax4.plot(np.cos(theta_uc), np.sin(theta_uc), '-', c='black', lw=1, label='Unit Circle')
    ax4.axhline(0, color='black', lw=0.5); ax4.axvline(0, color='black', lw=0.5)
    plot_pseudospectrum(ax4, structured_pseudospectrum(res['problem']), color=C_PSEUDO)
    for en, ew in zip(np.sort_complex(eigs_nom), np.sort_complex(eigs_wc)):
        ax4.plot([en.real, ew.real], [en.imag, ew.imag], 'k:', lw=1.8, alpha=0.8, zorder=4)
    ax4.scatter(eigs_nom.real, eigs_nom.imag, s=MS_NOM_EIG, c=C_NOM, marker=M_NOM, zorder=5, label='Nominal')
    ax4.scatter(eigs_wc.real, eigs_wc.imag, s=MS_WC_EIG, c=C_WC, marker=M_WC_EIG, zorder=6, label='Worst-Case')
    ax4.set_xlim(-1.1, 1.1); ax4.set_ylim(-1.1, 1.1)
    ax4.set_aspect('equal')
    ax4.set_xlabel("Real Part")
    ax4.set_ylabel("Imaginary Part")
    ax4.grid(True)
    fig4.savefig(_fig_path(f"{name}_spectral_targeting.pdf"), bbox_inches='tight', format="pdf")
    plt.show()
    plt.close(fig4)

    print(f"Success! Saved 4 files in '{FIG_DIR}/': \n"
          f"- '{name}_time_evolution.pdf'\n"
          f"- '{name}_nominal_flow.pdf'\n"
          f"- '{name}_worst_case_flow.pdf'\n"
          f"- '{name}_spectral_targeting.pdf'")


# ==========================================
# Fig. 2
# ==========================================
def match_poles(p_prev, p_curr):
    """Sorts poles to ensure continuous lines when drawing trajectory arrows."""
    d1 = abs(p_prev[0] - p_curr[0]) + abs(p_prev[1] - p_curr[1])
    d2 = abs(p_prev[0] - p_curr[1]) + abs(p_prev[1] - p_curr[0])
    return p_curr if d1 <= d2 else np.array([p_curr[1], p_curr[0]])


def plot_pole_migration(N_values, wc_poles, problem, filename="spectral_migration_trajectory_2D.pdf", seed=0,
                        inset_N_max=5, inset_box=(-0.9, -0.98, 0.95, 0.85)):
    """Worst-case closed-loop poles as a function of the horizon N, over the exact structured pseudospectrum.

    Also marks the dominant eigenvalue of the spectral-radius maximising perturbation (asymptotic worst case),
    computed by the projected-gradient eigenvalue optimisation of Guglielmi, Lubich and Sicilia.

    An inset (position `inset_box` = (x, y, width, height) in data coordinates; None to disable) zooms on the lower
    branch of the worst-case poles for N <= inset_N_max, with a finer pseudospectrum grid, to show that these poles
    lie strictly inside the pseudospectrum."""
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    ax.grid(True)
    ax.axhline(0, color='black', lw=0.5, zorder=1)
    ax.axvline(0, color='black', lw=0.5, zorder=1)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=0.8, label='Unit Circle', zorder=1)
    plot_pseudospectrum(ax, structured_pseudospectrum(problem), color=C_PSEUDO)

    eigs_nom = np.linalg.eigvals(problem.Phi)
    ax.scatter(eigs_nom.real, eigs_nom.imag, s=40, c=C_NOM, marker='o', zorder=5, label='Nominal eigenvalues')

    ps = structured_ps_radius(problem, rng=np.random.default_rng(seed))
    eigs_ps = np.linalg.eigvals(problem.Phi + ps.DA - ps.DB @ problem.K)
    eigs_ps = eigs_ps[np.abs(eigs_ps) >= np.max(np.abs(eigs_ps)) - 1e-9]   # dominant eigenvalue(s) only
    ax.scatter(eigs_ps.real, eigs_ps.imag, s=70, c='black', marker='*', zorder=7, label='Max. spectral radius')

    cmap = mpl.cm.plasma
    norm = mpl.colors.Normalize(vmin=min(N_values), vmax=max(N_values))
    prev_poles = None
    list_length = len(N_values)
    matched = {}
    for N in N_values:
        if N not in wc_poles:
            continue
        p = wc_poles[N]
        if prev_poles is not None:
            p = match_poles(prev_poles, p)
        matched[N] = p
        if prev_poles is not None:
            for i in range(len(p)):
                ax.annotate('', xy=(p[i].real, p[i].imag), xytext=(prev_poles[i].real, prev_poles[i].imag),
                            arrowprops=dict(arrowstyle='->', color='black', lw=0.6, alpha=0.9, shrinkA=2.5, shrinkB=2.5),
                            zorder=4)
        ax.scatter(p.real, p.imag, s=18, color=cmap(norm(N)), marker='D', edgecolors='black', linewidths=0.4, zorder=6)

        if N > 5:
            if N == 6:
                ax.annotate(f"6--{list_length + 1}", xy=(p[0].real, p[0].imag), xytext=(0, 4), textcoords='offset points',
                            ha='left', va='bottom', fontsize=8, color='black', zorder=10)
                ax.annotate(f"{list_length + 1}--6", xy=(p[1].real, p[1].imag), xytext=(0, 4), textcoords='offset points',
                            ha='right', va='bottom', fontsize=8, color='black', zorder=10)
        else:
            ax.annotate(str(N), xy=(p[0].real, p[0].imag), xytext=(3, 4), textcoords='offset points',
                        ha='left', va='bottom', fontsize=8, color='black', zorder=10)
            ax.annotate(str(N), xy=(p[1].real, p[1].imag), xytext=(4, -2), textcoords='offset points',
                        ha='left', va='top', fontsize=8, color='black', zorder=10)
        prev_poles = p

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Planning Horizon ($N$)', rotation=270, labelpad=12, fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")

    handles, labels = ax.get_legend_handles_labels()
    pseudo_patch = mpatches.Patch(facecolor=C_PSEUDO, alpha=0.8, label=r'Struct. $\varepsilon$-pseudospectrum')
    wc_marker = mpl.lines.Line2D([0], [0], marker='D', color='w', markerfacecolor=cmap(0.5), markeredgecolor='k',
                                 markersize=5, markeredgewidth=0.5, label='Worst-case eigenvalues')
    handles.extend([wc_marker, pseudo_patch])
    labels.extend([wc_marker.get_label(), pseudo_patch.get_label()])
    ax.legend(handles=handles, labels=labels, loc='upper left', framealpha=0.95, edgecolor='black')

    if inset_box is not None:
        _pole_inset(ax, problem, matched, [N for N in N_values if N in matched and N <= inset_N_max], cmap, norm,
                    inset_box)

    plt.tight_layout()
    fig.savefig(_fig_path(filename), bbox_inches='tight', format="pdf")
    plt.show()
    print(f"\nSuccess! Plot saved as {_fig_path(filename)}.")


def _pole_inset(ax, problem, matched, Ns, cmap, norm, box, pad_left=0.015, pad_right=0.03, pad_y=0.04):
    """Zoom on the lower branch (Im < 0) of the worst-case poles for the horizons Ns.

    The poles lie only a few thousandths inside the border, so the inset does not keep equal axes: the window is
    fitted to the poles (plus `pad_right` to show the border on their right), which stretches the horizontal axis.
    The inset keeps its own tick labels."""
    pts = np.array([q for N in Ns for q in matched[N] if q.imag < -1e-9])
    if pts.size == 0:
        return
    x0, x1 = pts.real.min() - pad_left, pts.real.max() + pad_right
    y0, y1 = pts.imag.min() - pad_y, pts.imag.max() + pad_y

    axins = ax.inset_axes(box, transform=ax.transData)
    # finer than the main grid, but only around the window: rows from |y1| to |y0| are all that is needed
    grid = structured_pseudospectrum(problem, re=(x0, x1), im_max=max(abs(y0), abs(y1)), n_re=61, n_real=2)
    plot_pseudospectrum(axins, grid, color=C_PSEUDO)
    prev = None
    for N in Ns:
        p = matched[N]
        if prev is not None:
            for i in range(len(p)):
                axins.annotate('', xy=(p[i].real, p[i].imag), xytext=(prev[i].real, prev[i].imag),
                               arrowprops=dict(arrowstyle='->', color='black', lw=0.5, alpha=0.9, shrinkA=2.0,
                                               shrinkB=2.5), zorder=4)
        axins.scatter(p.real, p.imag, s=7, color=cmap(norm(N)), marker='D', edgecolors='black', linewidths=0.3,
                      zorder=6)
        for q in p:
            if q.imag < -1e-9:
                axins.annotate(str(N), xy=(q.real, q.imag), xytext=(5, 0), textcoords='offset points',
                               ha='left', va='center', fontsize=7, color='black', zorder=10)
        prev = p
    axins.set_xlim(x0, x1)
    axins.set_ylim(y0, y1)
    axins.set_aspect('auto')
    axins.tick_params(labelsize=5)
    axins.grid(True)
    ax.indicate_inset_zoom(axins, edgecolor='black', lw=0.6)


# ==========================================
# Fig. 3
# ==========================================
def plot_comparison(results, filename="cost_comparaison.pdf"):
    """Bound comparison over the horizon: IQC/LTV and relaxed PEP upper bounds, LTI worst case, Monte Carlo, and the
    spectral-radius maximising perturbation (pseudospectral approach of Guglielmi, Lubich and Sicilia)."""
    myblue, myorange, myred, mygreen = '#006AA3', "#FAA00F", "#C1121F", "#5B9A64"
    horizons = results['horizons']
    plt.figure(figsize=(3.5, 2.5), dpi=300)
    plt.plot(horizons, results['iqc'], '--v', color=myred, lw=2.0, markersize=5, label='IQC / LTV (Upper Bound)')
    plt.plot(horizons, results['pep'], '-s', color=myorange, lw=2.0, markersize=5, markerfacecolor='white',
             markeredgewidth=2, label=r'PEP SDP (Relax. / Rank $> 1$)')
    plt.plot(horizons, results['lti'], '-o', color=myblue, lw=2.0, markersize=5, label='Worst-case LTI (gradient ascent)')
    plt.plot(horizons, results['mc'], '*k', markersize=5, zorder=10, label='MC LTI Samples')
    plt.plot(horizons, results['ps'], '-', color=mygreen, lw=1.5, label='Max. spectral radius (pseudospectral)')
    plt.fill_between(horizons, results['pep'], results['iqc'], color=myred, alpha=0.1, label='IQC conservatism')
    plt.fill_between(horizons, results['lti'], results['pep'], color=myorange, alpha=0.15, label='PEP relaxation gap')
    plt.xlabel("Finite Horizon ($N$)")
    plt.ylabel("Worst-Case Accumulated Cost")
    plt.xticks(horizons)
    plt.grid(True)
    plt.legend(loc='lower right', framealpha=0.9, edgecolor='black', fontsize=7)
    plt.tight_layout()
    plt.savefig(_fig_path(filename), transparent=True, bbox_inches='tight', format="pdf")
    plt.show()
    print(f"Success! Plot saved as {_fig_path(filename)}.")
