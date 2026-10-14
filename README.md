# Worst-Case LTI Perturbations of Finite-Horizon LQR

### A Performance Estimation Problem (PEP) Approach

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1D8yb0V3SLUVH80DLC78VKvGildCxSr2Y)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)

Companion code for the paper **"Worst-Case LTI Perturbations of Finite-Horizon LQR: A Performance Estimation Problem Approach"** by Anthony Couthures, Gianluca Bianchin and Julien M. Hendrickx (ICTEAM, UCLouvain).

This repository contains the numerical implementation and an interactive dashboard used to produce every figure in the paper.

## What this is about

An LQR gain is designed for a nominal model and an infinite horizon, but it is deployed for a finite time on a plant whose parameters are slightly off. How bad can the finite-horizon cost get when the true dynamics are

$$x_{k+1} = (\Phi + \Delta_A + \Delta_B K) x_k,    \|\Delta_A\|_2 ≤ \varepsilon_A,  \|\Delta_B\|_2 ≤ \varepsilon_B$$

with **static (LTI)** perturbations?

Classical tools (μ-analysis, IQC / dissipation inequalities) either look only at asymptotic behavior or relax the static perturbation into a *time-varying* one, which is conservative and does not return the worst-case perturbation itself. We instead:

1. Recast the worst-case search as a **QCQP over sequences**, using operator interpolation to enforce LTI behavior exactly.
2. Lift it to a **rank-constrained SDP** over a Gram matrix $Z$ (PEP framework).
3. Drop the rank constraint to get an **upper bound**. If `rank(Z) = 1` the bound is exact and the worst-case initial state and $\Delta_A$, $\Delta_B$ can be read off directly.
4. If `rank(Z) > 1`, recover a valid LTI worst case by **projected gradient ascent** on the perturbations. For fixed $(\Delta_A, \Delta_B)$ the worst initial state is the top eigenvector of the finite-horizon Gramian, so the cost is an eigenvalue whose gradient is obtained from one forward and one adjoint sweep. The perturbations are kept in their norm balls by projection, with an Armijo step-size rule, following the norm-constrained gradient flows of Guglielmi, Lubich and Sicilia [1, 2]. The search is warm-started from the SDP solution, Monte Carlo samples and random perturbations, and returns a stationarity residual as a first-order certificate.

The structured $\varepsilon$-pseudospectrum shown in the figures is computed **exactly**: for fixed $\Delta_B$, the smallest real $\Delta_A$ that makes $z$ an eigenvalue is given by the real-stability-radius formula of Qiu et al. [3], and only $\Delta_B$ has to be searched.

### Main observations

- Finite-horizon worst-case LTI perturbations do **not** simply follow the asymptotic (spectral-radius-maximizing) strategy. They align slowly decaying invariant manifolds with specific initial conditions to drive transient growth.
- This produces a **bifurcation** in the worst-case behavior as the horizon $N$ changes (see the pole-migration figure).
- The PEP bounds sit between the IQC (LTV) upper bound and the cost of the spectral-radius-maximising perturbation (the pseudospectral approach of [1], computed by projected gradient ascent on the dominant eigenvalue). With spectral-norm bounds this maximiser is not unique; the comparison uses the largest finite-horizon cost among the sampled maximisers, i.e. the most favourable choice for the pseudospectral approach.

## Repository contents

| File | Purpose |
|---|---|
| `PEP_LQR-ECC2027.ipynb` | Minimal notebook: interactive dashboard and paper figures |
| `pep_lqr/` | Python package with all the code (see below) |
| `pyproject.toml`, `requirements.txt` | Package metadata and Python dependencies |
| `CITATION.cff` | Citation metadata |

| Module | Content |
|---|---|
| `pep_lqr/system.py` | System matrices and nominal LQR |
| `pep_lqr/sdp.py` | Relaxed PEP and IQC (LTV) SDPs over the Gram matrix $Z$, solver configuration |
| `pep_lqr/worst_case.py` | Worst-case cost and adjoint gradient, projected gradient ascent, Monte Carlo warm start |
| `pep_lqr/spectral.py` | Exact structured pseudospectrum, fast random cloud (dashboard) |
| `pep_lqr/pipeline.py` | `solve_pep_2d` (one system) and `compare_bounds` (bounds over the horizon) |
| `pep_lqr/plotting.py` | Paper figures |
| `pep_lqr/dashboard.py` | Plotting function of the interactive dashboard (the `ipywidgets` cell itself is in the notebook) |

The notebook is organised in sections:

| Section | Description |
|---|---|
| **Simulator** | Interactive dashboard. The spectral panel shows a fast random cloud of perturbed eigenvalues; tick *Exact pseudospectrum border* for the exact set (slower) |
| **Figure 1** | Structured pseudospectrum and worst-case trajectories for three 2D systems $A_1,A_2,A_3$ (monotonic, oscillatory, spiraling worst cases) |
| **Figure 2** | Movement of the worst-case eigenvalues as $N$ grows |
| **Figure 3** | IQC (LTV) vs. relaxed PEP vs. worst-case LTI vs. MC vs. spectral-radius-maximising perturbation, over $N = 2,..., 20$ |

## Getting started

### Option 1: Google Colab

Click the badge at the top. The first cell installs the `pep_lqr` package from this repository if it is not available. Everything except MOSEK runs out of the box (the code falls back to SCS).

### Option 2: Local

```bash
git clone https://github.com/AnthonyCouthures/PEP-LQR.git
cd PEP-LQR
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
jupyter lab PEP_LQR-ECC2027.ipynb
```

### Solvers

`enable_default_solver()` (called in the first notebook cell) patches `cvxpy.Problem.solve` to prefer **MOSEK**, then **SCS**, then CVXPY's default. MOSEK is recommended for accuracy and speed on the larger SDPs (`N` up to 20) and offers a [free academic license](https://www.mosek.com/products/academic-licenses/). With SCS, expect lower accuracy and possibly different rank estimates.

## Reproducing the paper figures

Run the notebook top to bottom. The figure cells save PDFs to the `figures/` folder. Text is rendered with LaTeX (`newtxtext`, as in the paper) when a LaTeX installation is found, and with matplotlib's mathtext otherwise (e.g. on Colab):

| Output | Paper figure |
|---|---|
| `A_{1,2,3}_spectral_targeting.pdf` | Fig. 1, top row |
| `A_{1,2,3}_time_evolution.pdf`, `_nominal_flow.pdf`, `_worst_case_flow.pdf` | Fig. 1, bottom row and supporting plots |
| `spectral_migration_trajectory_2D.pdf` | Worst-case pole migration vs. horizon |
| `cost_comparaison.pdf` | Bound comparison (IQC / PEP / worst-case LTI / MC / max. spectral radius) |

Parameters used: `A₁ = [[1,1],[0,1]]` (ε_A = 0.22, ε_B = 0), `A₂ = [[-0.9,1],[0,1]]` (0.2, 0.1), `A₃ = [[1,0.9],[-0.8,1]]` (0.1, 0.25), `B = [0 1]ᵀ`, `Q = R = S = I`, `N = 20` (see the cells for exact values).

> The Monte Carlo warm starts and random starts are seeded in the first notebook cell (`rng = np.random.default_rng(0)`), so runs are repeatable.

## References

1. N. Guglielmi, C. Lubich, S. Sicilia, "Rank-1 matrix differential equations for structured eigenvalue optimization," *SIAM Journal on Numerical Analysis*, 61(4):1737–1762, 2023. doi:[10.1137/22M1498735](https://doi.org/10.1137/22M1498735)
2. N. Guglielmi, C. Lubich, "Transient dynamics under structured perturbations: bridging unstructured and structured pseudospectra," *SIAM Journal on Numerical Analysis*, 63(2):908–930, 2025. doi:[10.1137/24M1630876](https://doi.org/10.1137/24M1630876)
3. L. Qiu, B. Bernhardsson, A. Rantzer, E. J. Davison, P. M. Young, J. C. Doyle, "A formula for computation of the real stability radius," *Automatica*, 31(6):879–890, 1995.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{couthures2026worstcase,
  title     = {Worst-Case {LTI} Perturbations of Finite-Horizon {LQR}: A Performance Estimation Problem Approach},
  author    = {Couthures, Anthony and Bianchin, Gianluca and Hendrickx, Julien M.},
  year      = {2026},
  note      = {Code: https://github.com/AnthonyCouthures/PEP-LQR}
}
```

(Update the entry with the venue and DOI once available.)

## Acknowledgements

Supported by the "SIDDARTA" Concerted Research Action (ARC) granted by UCLouvain, and by "InterpoControl" F.R.S.-FNRS.

## License

Released under the [MIT License](LICENSE).

## Contact

Anthony Couthures, ICTEAM, UCLouvain: anthony.couthures@uclouvain.be
