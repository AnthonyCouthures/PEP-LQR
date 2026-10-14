"""Nominal system and LQR controller."""
import numpy as np
from scipy.linalg import solve_discrete_are


def build_system_matrices(A11, A12, A21, A22, B1, B2, Q11, Q22, R11, S11, S22):
    """
    Constructs the system matrices A, B, Q, R, and S from individual scalar components.

    Returns:
        tuple: (A, B, Q, R, S_user).
    """
    A = np.array([[A11, A12], [A21, A22]], dtype=float)
    B = np.array([[B1], [B2]], dtype=float)
    Q = np.array([[Q11, 0.0], [0.0, Q22]], dtype=float)
    R = np.array([[R11]], dtype=float)
    S_user = np.array([[S11, 0.0], [0.0, S22]], dtype=float)
    return A, B, Q, R, S_user


def solve_nominal_lqr(A, B, Q, R):
    """
    Solves the discrete-time algebraic Riccati equation and returns the infinite-horizon LQR gain.

    The control law is u = -K x, so the nominal closed loop is Phi = A - B K and the
    equivalent stage cost is Q_K = Q + K^T R K.

    Returns:
        tuple: (P, K, Phi, Q_K).

    Raises:
        ValueError: If the system is not stabilizable.
    """
    try:
        P = solve_discrete_are(A, B, Q, R)
    except Exception:
        raise ValueError("System not stabilizable (Check A and B)")
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    Phi = A - B @ K
    Q_K = Q + K.T @ R @ K
    return P, K, Phi, Q_K
