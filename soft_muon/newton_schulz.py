"""
Newton-Schulz and related iterations for matrix inverse square root.

This module provides multiple backends for computing B^{-1/2} or the polar factor,
used in the SoftMuon optimizer's regularized polar computation.
"""

import torch
from typing import Tuple


def coupled_newton_invsqrt(
    B: torch.Tensor,
    n_iters: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Compute B^{-1/2} via coupled Newton iteration.

    This is the recommended method for computing the matrix inverse square root
    needed in the regularized polar: Q_λ(C) = C @ (C^T C + λI)^{-1/2}.

    The iteration is:
        Y₀ = B_scaled, Z₀ = I
        Y_{k+1} = 0.5 * Y_k @ (3I - Z_k @ Y_k)
        Z_{k+1} = 0.5 * (3I - Z_k @ Y_k) @ Z_k

    At convergence: Y → B^{1/2}, Z → B^{-1/2}

    Args:
        B: Symmetric positive definite matrix of shape (n, n)
        n_iters: Number of iterations. Default: 5
        eps: Small constant for numerical stability. Default: 1e-7

    Returns:
        B^{-1/2} of shape (n, n)
    """
    n = B.shape[0]
    device, dtype = B.device, B.dtype

    # Scale for convergence: need eigenvalues in (0, 3) for the iteration
    # Use power iteration estimate for spectral norm (max eigenvalue)
    # This is more robust than trace/n for ill-conditioned matrices
    v = torch.randn(n, device=device, dtype=dtype)
    for _ in range(3):  # A few power iterations
        v = B @ v
        v = v / (v.norm() + eps)
    spectral_norm_est = (v @ B @ v).abs() + eps

    # Scale so max eigenvalue is ~1 (well within the (0, 3) convergence range)
    scale = 1.0 / spectral_norm_est
    B_scaled = B * scale

    # Initialize: Y = B_scaled, Z = I
    I = torch.eye(n, device=device, dtype=dtype)
    Y = B_scaled.clone()
    Z = I.clone()

    # Coupled Newton iteration for simultaneous sqrt and inverse sqrt
    for _ in range(n_iters):
        ZY = Z @ Y
        T = 3.0 * I - ZY
        Y_new = 0.5 * Y @ T
        Z_new = 0.5 * T @ Z
        Y, Z = Y_new, Z_new

    # Z converges to (B_scaled)^{-1/2}
    # Undo scaling: (scale * B)^{-1/2} = scale^{-1/2} * B^{-1/2}
    return Z * (scale**0.5)


def newton_schulz_invsqrt_simple(
    B: torch.Tensor,
    n_iters: int = 10,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Simple Newton iteration for B^{-1/2}.

    Uses the iteration: X_{k+1} = 0.5 * X_k @ (3I - B @ X_k @ X_k)

    This converges to B^{-1/2} when ||I - B|| < 1.

    Args:
        B: Symmetric positive definite matrix
        n_iters: Number of iterations
        eps: Numerical stability constant

    Returns:
        B^{-1/2}
    """
    n = B.shape[0]
    device, dtype = B.device, B.dtype

    # Scale B so eigenvalues are near 1
    norm_est = torch.trace(B) / n + eps
    scale = 1.0 / norm_est
    B_scaled = B * scale

    I = torch.eye(n, device=device, dtype=dtype)
    X = I.clone()

    for _ in range(n_iters):
        X = 0.5 * X @ (3.0 * I - B_scaled @ X @ X)

    return X * (scale**0.5)


def newton_schulz_polar(
    X: torch.Tensor,
    n_iters: int = 5,
    coeffs: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
) -> torch.Tensor:
    """
    Original Muon Newton-Schulz iteration for polar factor.

    Computes Q from polar decomposition X = QH where Q is orthogonal.
    The iteration is: X ← aX + b(XX^T)X + c(XX^T)(XX^T)X

    This is Muon's approach which applies φ(σ) = 1 to all singular values,
    making them all equal to 1.

    Args:
        X: Input matrix of shape (m, n)
        n_iters: Number of iterations. Default: 5
        coeffs: Coefficients (a, b, c) for the iteration.
            Default: (3.4445, -4.7750, 2.0315) from Muon paper

    Returns:
        Polar factor Q of shape (m, n)

    Note:
        Input should be pre-scaled so singular values are near 1.
        The Muon paper recommends scaling by 1/||X||_F * sqrt(n).
    """
    a, b, c = coeffs

    # Pre-scale for convergence
    m, n = X.shape
    norm = X.norm()
    X = X * ((min(m, n) ** 0.5) / (norm + 1e-7))

    for _ in range(n_iters):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X


def newton_schulz_invsqrt(
    B: torch.Tensor,
    n_iters: int = 5,
    coeffs: Tuple[float, float, float] = (3.4445, -4.7750, 2.0315),
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Newton-Schulz style iteration for inverse square root.

    Uses polynomial iteration similar to Muon's polar factor computation.

    Args:
        B: Symmetric positive definite matrix of shape (n, n)
        n_iters: Number of iterations
        coeffs: Coefficients for the iteration
        eps: Numerical stability constant

    Returns:
        Approximation of B^{-1/2}
    """
    n = B.shape[0]
    device, dtype = B.device, B.dtype

    # Scale B
    norm_est = torch.trace(B) / n + eps
    scale = 1.0 / norm_est
    B_scaled = B * scale

    I = torch.eye(n, device=device, dtype=dtype)
    X = I.clone()

    # Use simple Newton iteration
    for _ in range(n_iters):
        X2B = X @ X @ B_scaled
        X = 0.5 * X @ (3.0 * I - X2B)

    return X * (scale**0.5)


def denman_beavers(
    A: torch.Tensor,
    n_iters: int = 10,
    eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Denman-Beavers iteration for simultaneous sqrt and inverse sqrt.

    Returns (A^{1/2}, A^{-1/2}).

    The iteration is:
        Y_{k+1} = 0.5 * (Y_k + Z_k^{-1})
        Z_{k+1} = 0.5 * (Z_k + Y_k^{-1})

    Starting from Y_0 = A, Z_0 = I.

    Args:
        A: Symmetric positive definite matrix
        n_iters: Number of iterations
        eps: Regularization for stability

    Returns:
        Tuple of (A^{1/2}, A^{-1/2})

    Note:
        Requires matrix inversion at each step, making it less GPU-friendly
        than Newton-Schulz methods. Provided for comparison and validation.
    """
    n = A.shape[0]
    device, dtype = A.device, A.dtype
    I = torch.eye(n, device=device, dtype=dtype)

    Y = A.clone()
    Z = I.clone()

    for _ in range(n_iters):
        # Regularize for stability
        Y_reg = Y + eps * I
        Z_reg = Z + eps * I

        Y_new = 0.5 * (Y + torch.linalg.inv(Z_reg))
        Z_new = 0.5 * (Z + torch.linalg.inv(Y_reg))
        Y, Z = Y_new, Z_new

    return Y, Z  # A^{1/2}, A^{-1/2}


def regularized_polar(
    C: torch.Tensor,
    lambda_reg: float,
    n_iters: int = 5,
    eps: float = 1e-7,
    backend: str = "coupled_newton",
) -> torch.Tensor:
    """
    Compute regularized polar factor Q_λ(C) = C @ (C^T C + λI)^{-1/2}.

    This applies the spectral filter φ_λ(σ) = σ / √(σ² + λ) to the singular
    values of C, which interpolates between:
    - λ → 0: Muon's polar decomposition (φ(σ) → 1)
    - λ → ∞: Scaled gradient descent (φ(σ) → 0)

    For an m×n matrix C:
    - If m < n (wide): Uses Q_λ(C) = (CC^T + λI)^{-1/2} @ C
    - If m >= n (tall): Uses Q_λ(C) = C @ (C^T C + λI)^{-1/2}

    Args:
        C: Input matrix of shape (m, n)
        lambda_reg: Regularization parameter λ >= 0
        n_iters: Number of iterations for inverse sqrt computation
        eps: Numerical stability constant
        backend: 'coupled_newton' or 'newton_schulz'

    Returns:
        Q_λ(C) of shape (m, n)
    """
    m, n = C.shape

    # Scale input for numerical stability
    # This is critical for convergence of Newton iterations
    # Use Frobenius norm normalized by sqrt(min dimension)
    scale = C.norm() / (min(m, n) ** 0.5) + eps
    C_scaled = C / scale

    # Lambda needs to be scaled relative to the squared norm
    # After scaling, ||C_scaled||_F^2 ≈ min(m,n), so we scale lambda accordingly
    lambda_scaled = lambda_reg / (scale ** 2)

    # Ensure lambda provides sufficient regularization
    # This prevents numerical issues when C is ill-conditioned
    # The minimum lambda should be high enough to ensure all eigenvalues
    # of B + lambda*I are bounded away from zero
    min_lambda = 0.01  # Ensures eigenvalues > 0.01 for stability
    lambda_scaled = max(lambda_scaled, min_lambda)

    if backend == "coupled_newton":
        invsqrt_fn = lambda B: coupled_newton_invsqrt(B, n_iters, eps)
    elif backend == "newton_schulz":
        invsqrt_fn = lambda B: newton_schulz_invsqrt(B, n_iters, eps=eps)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    if m < n:
        # Wide matrix: work with CC^T (m×m)
        B = C_scaled @ C_scaled.T
        # Ensure symmetry and add regularization to diagonal
        B = 0.5 * (B + B.T)  # Ensure symmetry
        B.diagonal().add_(lambda_scaled)
        B_invsqrt = invsqrt_fn(B)
        return B_invsqrt @ C_scaled
    else:
        # Tall matrix: work with C^T C (n×n)
        B = C_scaled.T @ C_scaled
        # Ensure symmetry and add regularization to diagonal
        B = 0.5 * (B + B.T)  # Ensure symmetry
        B.diagonal().add_(lambda_scaled)
        B_invsqrt = invsqrt_fn(B)
        return C_scaled @ B_invsqrt
