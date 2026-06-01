#!/usr/bin/env python3
"""Minimal solver module for matrix completion experiments.

This file contains only the reusable numerical pieces that are needed by
`synthetic_api.py`. It intentionally avoids the old benchmark / CLI layer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass
class SolverResult:
    method: str
    X_hat: Array
    history: list[dict[str, float]]
    runtime_sec: float
    iterations: int
    l2_reg: float = 0.0
    method_family: str = ""
    selected_l2_reg: float | None = None
    validation_rmse: float | None = None


def fro_norm(X: Array) -> float:
    return float(np.linalg.norm(X, ord="fro"))


def masked_rmse(X_hat: Array, X_true: Array, mask: Array) -> float:
    if not np.any(mask):
        return float("nan")
    diff = X_hat[mask] - X_true[mask]
    return float(np.sqrt(np.mean(diff**2)))


def masked_mae(X_hat: Array, X_true: Array, mask: Array) -> float:
    if not np.any(mask):
        return float("nan")
    diff = np.abs(X_hat[mask] - X_true[mask])
    return float(np.mean(diff))


def relative_fro_error(X_hat: Array, X_true: Array) -> float:
    denom = fro_norm(X_true)
    if denom == 0.0:
        return float("nan")
    return fro_norm(X_hat - X_true) / denom


def observed_objective(X: Array, Y: Array, mask: Array) -> float:
    residual = np.where(mask, X - Y, 0.0)
    return 0.5 * float(np.sum(residual**2))


def truncated_svd(X: Array, rank: int) -> tuple[Array, Array, Array]:
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    rank = min(rank, len(s))
    return U[:, :rank], s[:rank], Vt[:rank, :]


def best_rank_approximation(X: Array, rank: int) -> Array:
    U, s, Vt = truncated_svd(X, rank)
    return (U * s) @ Vt


def soft_threshold_svd(X: Array, tau: float, max_rank: int | None = None) -> Array:
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    s = np.maximum(s - tau, 0.0)
    positive = int(np.sum(s > 0.0))
    if max_rank is not None:
        positive = min(positive, max_rank)
    if positive == 0:
        return np.zeros_like(X)
    return (U[:, :positive] * s[:positive]) @ Vt[:positive, :]


def orthonormal_matrix(rows: int, cols: int, rng: np.random.Generator) -> Array:
    Q, _ = np.linalg.qr(rng.normal(size=(rows, cols)))
    return Q[:, :cols]


def initialize_low_rank(
    Y: Array,
    mask: Array,
    rank: int,
    rng: np.random.Generator,
    init: str = "spectral",
) -> Array:
    m, n = Y.shape
    rank = min(rank, m, n)
    observed_fraction = max(float(np.mean(mask)), 1e-8)
    if init == "spectral":
        scaled = Y / observed_fraction
        X0 = best_rank_approximation(scaled, rank)
        if fro_norm(X0) > 0.0:
            return X0
    scale = np.std(Y[mask]) if np.any(mask) else 1.0
    scale = max(float(scale), 1e-2)
    A = rng.normal(scale=scale / np.sqrt(rank), size=(m, rank))
    B = rng.normal(scale=scale / np.sqrt(rank), size=(n, rank))
    return best_rank_approximation(A @ B.T, rank)


def factorize_from_matrix(X: Array, rank: int) -> tuple[Array, Array]:
    U, s, Vt = truncated_svd(X, rank)
    rank = len(s)
    sqrt_s = np.sqrt(np.maximum(s, 0.0))
    A = U[:, :rank] * sqrt_s
    B = Vt[:rank, :].T * sqrt_s
    return A, B


def tangent_projection_fixed_rank(X: Array, Z: Array, rank: int) -> Array:
    U, _, Vt = truncated_svd(X, rank)
    V = Vt.T
    P_U = U @ U.T
    P_V = V @ V.T
    return P_U @ Z + Z @ P_V - P_U @ Z @ P_V


def retract_fixed_rank(X: Array, Xi: Array, rank: int) -> Array:
    return best_rank_approximation(X + Xi, rank)


def compact_matrix(U: Array, S: Array, V: Array) -> Array:
    return U @ S @ V.T


def initialize_compact_factors(
    Y: Array,
    mask: Array,
    rank: int,
    rng: np.random.Generator,
    init: str = "spectral",
) -> tuple[Array, Array, Array]:
    m, n = Y.shape
    rank = min(rank, m, n)
    observed_fraction = max(float(np.mean(mask)), 1e-8)

    if init == "spectral":
        scaled = Y / observed_fraction
        U, singular_values, Vt = truncated_svd(scaled, rank)
        if len(singular_values) == rank and float(np.max(singular_values)) > 0.0:
            singular_values = np.maximum(singular_values, 1e-8)
            return U, np.diag(singular_values), Vt.T

    scale = np.std(Y[mask]) if np.any(mask) else 1.0
    scale = max(float(scale), 1e-2)
    U = orthonormal_matrix(m, rank, rng)
    V = orthonormal_matrix(n, rank, rng)
    S = np.diag(np.full(rank, scale))
    return U, S, V


def tangent_components_compact(
    U: Array,
    S: Array,
    V: Array,
    Y: Array,
    mask: Array,
    l2_reg: float,
) -> tuple[Array, Array, Array, Array, Array]:
    X = compact_matrix(U, S, V)
    grad_euclid = np.where(mask, X - Y, 0.0) + l2_reg * X
    M = U.T @ grad_euclid @ V
    U_perp = grad_euclid @ V - U @ M
    V_perp = grad_euclid.T @ U - V @ M.T
    return X, M, U_perp, V_perp, grad_euclid


def tangent_norm_from_components(M: Array, U_perp: Array, V_perp: Array) -> float:
    norm_sq = fro_norm(M) ** 2 + fro_norm(U_perp) ** 2 + fro_norm(V_perp) ** 2
    return float(np.sqrt(max(norm_sq, 0.0)))


def thin_qr_nonzero_columns(Z: Array, tol: float = 1e-12) -> tuple[Array, Array]:
    rows, cols = Z.shape
    if cols == 0:
        return np.zeros((rows, 0)), np.zeros((0, 0))
    if fro_norm(Z) <= tol:
        return np.zeros((rows, 0)), np.zeros((0, cols))

    Q, R = np.linalg.qr(Z, mode="reduced")
    diag = np.abs(np.diag(R))
    threshold = tol * max(1.0, float(np.max(diag)) if len(diag) else 1.0)
    keep = diag > threshold
    if not np.any(keep):
        return np.zeros((rows, 0)), np.zeros((0, cols))
    return Q[:, keep], R[keep, :]


def retract_compact_usv(
    U: Array,
    S: Array,
    V: Array,
    M_step: Array,
    U_perp_step: Array,
    V_perp_step: Array,
    rank: int,
) -> tuple[Array, Array, Array]:
    Q_u, R_u = thin_qr_nonzero_columns(U_perp_step)
    Q_v, R_v = thin_qr_nonzero_columns(V_perp_step)
    r = S.shape[0]
    q_u = Q_u.shape[1]
    q_v = Q_v.shape[1]

    core = np.zeros((r + q_u, r + q_v))
    core[:r, :r] = S + M_step
    if q_v:
        core[:r, r:] = R_v.T
    if q_u:
        core[r:, :r] = R_u

    U_core, singular_values, Vt_core = np.linalg.svd(core, full_matrices=False)
    rank = min(rank, len(singular_values))
    U_aug = np.hstack([U, Q_u]) if q_u else U
    V_aug = np.hstack([V, Q_v]) if q_v else V

    U_new = U_aug @ U_core[:, :rank]
    S_new = np.diag(singular_values[:rank])
    V_new = V_aug @ Vt_core[:rank, :].T
    return U_new, S_new, V_new


def solve_riemannian_gradient_descent(
    Y: Array,
    mask: Array,
    rank: int,
    rng: np.random.Generator,
    init: str = "spectral",
    max_iter: int = 300,
    tol: float = 1e-6,
    initial_step: float = 1.0,
    backtracking_beta: float = 0.5,
    armijo_c: float = 1e-4,
    l2_reg: float = 0.0,
    method_name: str = "riemannian_gd",
    method_family: str | None = None,
) -> SolverResult:
    start = time.perf_counter()
    X = initialize_low_rank(Y, mask, rank, rng, init=init)
    history: list[dict[str, float]] = []
    initial_grad_norm: float | None = None

    for iteration in range(1, max_iter + 1):
        train_obj = observed_objective(X, Y, mask) + 0.5 * l2_reg * fro_norm(X) ** 2
        grad_euclid = np.where(mask, X - Y, 0.0) + l2_reg * X
        grad_riem = tangent_projection_fixed_rank(X, grad_euclid, rank)
        grad_norm = fro_norm(grad_riem)

        if initial_grad_norm is None:
            initial_grad_norm = max(grad_norm, 1.0)

        step = initial_step
        candidate = X
        grad_norm_sq = grad_norm**2

        if grad_norm_sq > 0.0:
            for _ in range(30):
                candidate = retract_fixed_rank(X, -step * grad_riem, rank)
                candidate_obj = observed_objective(candidate, Y, mask) + 0.5 * l2_reg * fro_norm(candidate) ** 2
                if candidate_obj <= train_obj - armijo_c * step * grad_norm_sq:
                    break
                step *= backtracking_beta

        relative_change = fro_norm(candidate - X) / max(fro_norm(X), 1.0)
        history.append(
            {
                "iteration": float(iteration),
                "train_objective": float(train_obj),
                "gradient_norm": float(grad_norm),
                "step_size": float(step),
                "relative_change": float(relative_change),
            }
        )

        X = candidate

        if grad_norm <= tol * initial_grad_norm or relative_change <= tol:
            break

    runtime_sec = time.perf_counter() - start
    return SolverResult(
        method=method_name,
        X_hat=X,
        history=history,
        runtime_sec=runtime_sec,
        iterations=len(history),
        l2_reg=l2_reg,
        method_family=method_family or method_name,
    )


def solve_riemannian_gradient_descent_compact(
    Y: Array,
    mask: Array,
    rank: int,
    rng: np.random.Generator,
    init: str = "spectral",
    max_iter: int = 300,
    tol: float = 1e-6,
    initial_step: float = 1.0,
    backtracking_beta: float = 0.5,
    armijo_c: float = 1e-4,
    l2_reg: float = 0.0,
    method_name: str = "riemannian_gd_compact",
    method_family: str | None = None,
) -> SolverResult:
    start = time.perf_counter()
    U, S, V = initialize_compact_factors(Y, mask, rank, rng, init=init)
    history: list[dict[str, float]] = []
    initial_grad_norm: float | None = None

    for iteration in range(1, max_iter + 1):
        X, M, U_perp, V_perp, _ = tangent_components_compact(U, S, V, Y, mask, l2_reg)
        train_obj = observed_objective(X, Y, mask) + 0.5 * l2_reg * fro_norm(S) ** 2
        grad_norm = tangent_norm_from_components(M, U_perp, V_perp)

        if initial_grad_norm is None:
            initial_grad_norm = max(grad_norm, 1.0)

        step = initial_step
        candidate_U, candidate_S, candidate_V = U, S, V
        candidate_X = X
        grad_norm_sq = grad_norm**2

        if grad_norm_sq > 0.0:
            for _ in range(30):
                candidate_U, candidate_S, candidate_V = retract_compact_usv(
                    U,
                    S,
                    V,
                    M_step=-step * M,
                    U_perp_step=-step * U_perp,
                    V_perp_step=-step * V_perp,
                    rank=rank,
                )
                candidate_X = compact_matrix(candidate_U, candidate_S, candidate_V)
                candidate_obj = observed_objective(candidate_X, Y, mask) + 0.5 * l2_reg * fro_norm(candidate_S) ** 2
                if candidate_obj <= train_obj - armijo_c * step * grad_norm_sq:
                    break
                step *= backtracking_beta

        relative_change = fro_norm(candidate_X - X) / max(fro_norm(X), 1.0)
        history.append(
            {
                "iteration": float(iteration),
                "train_objective": float(train_obj),
                "gradient_norm": float(grad_norm),
                "step_size": float(step),
                "relative_change": float(relative_change),
            }
        )

        U, S, V = candidate_U, candidate_S, candidate_V

        if grad_norm <= tol * initial_grad_norm or relative_change <= tol:
            break

    runtime_sec = time.perf_counter() - start
    return SolverResult(
        method=method_name,
        X_hat=compact_matrix(U, S, V),
        history=history,
        runtime_sec=runtime_sec,
        iterations=len(history),
        l2_reg=l2_reg,
        method_family=method_family or method_name,
    )


def solve_als(
    Y: Array,
    mask: Array,
    rank: int,
    rng: np.random.Generator,
    init: str = "spectral",
    reg: float = 1e-3,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> SolverResult:
    start = time.perf_counter()
    X0 = initialize_low_rank(Y, mask, rank, rng, init=init)
    A, B = factorize_from_matrix(X0, rank)
    m, n = Y.shape
    history: list[dict[str, float]] = []
    eye = np.eye(rank)
    previous_obj: float | None = None

    for iteration in range(1, max_iter + 1):
        for i in range(m):
            idx = np.flatnonzero(mask[i])
            if len(idx) == 0:
                A[i] = 0.0
                continue
            B_obs = B[idx]
            lhs = B_obs.T @ B_obs + reg * eye
            rhs = B_obs.T @ Y[i, idx]
            A[i] = np.linalg.solve(lhs, rhs)

        for j in range(n):
            idx = np.flatnonzero(mask[:, j])
            if len(idx) == 0:
                B[j] = 0.0
                continue
            A_obs = A[idx]
            lhs = A_obs.T @ A_obs + reg * eye
            rhs = A_obs.T @ Y[idx, j]
            B[j] = np.linalg.solve(lhs, rhs)

        X = A @ B.T
        train_obj = observed_objective(X, Y, mask)
        relative_improvement = 0.0
        if previous_obj is not None:
            relative_improvement = abs(previous_obj - train_obj) / max(previous_obj, 1.0)
        history.append(
            {
                "iteration": float(iteration),
                "train_objective": float(train_obj),
                "relative_improvement": float(relative_improvement),
            }
        )
        if previous_obj is not None and relative_improvement <= tol:
            break
        previous_obj = train_obj

    runtime_sec = time.perf_counter() - start
    return SolverResult(
        method="als",
        X_hat=A @ B.T,
        history=history,
        runtime_sec=runtime_sec,
        iterations=len(history),
        l2_reg=0.0,
        method_family="als",
    )


def solve_soft_impute(
    Y: Array,
    mask: Array,
    rank_hint: int,
    max_rank: int | None = None,
    lambda_scale: float = 0.15,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> SolverResult:
    start = time.perf_counter()
    X = np.zeros_like(Y)
    history: list[dict[str, float]] = []
    spectral_norm = float(np.linalg.svd(Y, compute_uv=False)[0]) if np.any(Y) else 1.0
    tau = lambda_scale * spectral_norm
    if max_rank is None:
        max_rank = max(rank_hint + 4, 2 * rank_hint)

    for iteration in range(1, max_iter + 1):
        filled = np.where(mask, Y, X)
        X_new = soft_threshold_svd(filled, tau=tau, max_rank=max_rank)
        train_obj = observed_objective(X_new, Y, mask)
        relative_change = fro_norm(X_new - X) / max(fro_norm(X), 1.0)
        history.append(
            {
                "iteration": float(iteration),
                "train_objective": float(train_obj),
                "relative_change": float(relative_change),
                "tau": float(tau),
            }
        )
        X = X_new
        if relative_change <= tol:
            break

    runtime_sec = time.perf_counter() - start
    return SolverResult(
        method="soft_impute",
        X_hat=X,
        history=history,
        runtime_sec=runtime_sec,
        iterations=len(history),
        l2_reg=0.0,
        method_family="soft_impute",
    )


__all__ = [
    "Array",
    "SolverResult",
    "masked_rmse",
    "masked_mae",
    "relative_fro_error",
    "observed_objective",
    "solve_riemannian_gradient_descent",
    "solve_riemannian_gradient_descent_compact",
    "solve_als",
    "solve_soft_impute",
]
