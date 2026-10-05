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
4. If `rank(Z) > 1`, recover a valid LTI worst case with two warm-started local searches: a **Burer–Monteiro** rank-1 factorization (seeded by the SDP) and a **Monte-Carlo-seeded NLP** (seeded by random LTI samples).

### Main observations

- Finite-horizon worst-case LTI perturbations do **not** simply follow the asymptotic (spectral-radius-maximizing) strategy. They align slowly decaying invariant manifolds with specific initial conditions to drive transient growth.
- This produces a **bifurcation** in the worst-case behavior as the horizon $N$ changes (see the pole-migration figure).
- The PEP bounds sit between the IQC (LTV) upper bound and a pseudospectral-heuristic lower bound.

## Repository contents

| File | Purpose |
|---|---|
| `ECC_test.ipynb` | Everything: solvers, interactive dashboard, paper figures |
| `requirements.txt` | Python dependencies |
| `CITATION.cff` | Citation metadata |

The notebook is organised in sections:

| Section | Description |
|---|---|
| **Simulator** | Core functions (nominal LQR, PEP matrices, Gram-matrix SDP, Burer–Monteiro QCQP, MC/NLP/pseudospectral heuristics) plus an `ipywidgets` dashboard to vary $A, B, Q, R, S, \varepsilon_A, \varepsilon_B, N$ live |
| **Figure 1** | Spectral targeting and worst-case trajectories for three 2D systems $A_1,A_2,A_3$ (monotonic, oscillatory, spiraling worst cases) |
| **Pole migration** | Movement of the worst-case eigenvalues as $N$ grows |
| **Comparison** | IQC (LTV) vs. relaxed PEP vs. rank-1 PEP vs. MC vs. pseudospectral heuristic, over $N = 2,..., 20$ |

## Getting started

### Option 1: Google Colab

Click the badge at the top. Everything except MOSEK runs out of the box (the code falls back to SCS).

### Option 2: Local

```bash
git clone https://github.com/AnthonyCouthures/PEP-LQR.git
cd <repo-name>
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab ECC_test.ipynb
```

### Solvers

The notebook patches `cvxpy.Problem.solve` to prefer **MOSEK**, then **SCS**, then CVXPY's default. MOSEK is recommended for accuracy and speed on the larger SDPs (`N` up to 20) and offers a [free academic license](https://www.mosek.com/products/academic-licenses/). With SCS, expect lower accuracy and possibly different rank estimates.

## Reproducing the paper figures

Run the notebook top to bottom. The figure cells save PDFs to the working directory:

| Output | Paper figure |
|---|---|
| `A_{1,2,3}_spectral_targeting.pdf` | Fig. 1, top row |
| `A_{1,2,3}_time_evolution.pdf`, `_nominal_flow.pdf`, `_worst_case_flow.pdf` | Fig. 1, bottom row and supporting plots |
| `spectral_migration_trajectory_2D.pdf` | Worst-case pole migration vs. horizon |
| `cost_comparaison.pdf` | Bound comparison (IQC / PEP / MC / pseudospectral) |

Parameters used: `A₁ = [[1,1],[0,1]]` (ε_A = 0.22, ε_B = 0), `A₂ = [[-0.9,1],[0,1]]` (0.2, 0.1), `A₃ = [[1,0.9],[-0.8,1]]` (0.1, 0.25), `B = [0 1]ᵀ`, `Q = R = S = I`, `N = 20` (see the cells for exact values).

> Results from the Monte-Carlo and NLP heuristics are stochastic. Set `np.random.seed(...)` at the top of the notebook for repeatable runs.

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
