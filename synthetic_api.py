#!/usr/bin/env python3
"""Основной API для синтетических экспериментов по matrix completion.

Модуль устроен как набор небольших конструкторов: сценарий, маска пропусков,
наблюдения, методы, метрики и графики. Такой формат удобнее для ноутбука и
исследовательских запусков, чем отдельный CLI-бенчмарк.

Общая схема:
    матрица -> структура -> поле пропусков -> маска наблюдений ->
    train/validation/test -> шум -> методы -> метрики
"""

from __future__ import annotations

import copy
import csv
import itertools
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib_config")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/xdg_cache")


def _running_inside_ipython() -> bool:
    try:
        from IPython import get_ipython  # type: ignore

        return get_ipython() is not None
    except Exception:
        return False


try:
    import matplotlib

    if not _running_inside_ipython():
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

from matrix_completion_methods import (
    SolverResult,
    masked_mae,
    masked_rmse,
    observed_objective,
    relative_fro_error,
    solve_als,
    solve_riemannian_gradient_descent,
    solve_riemannian_gradient_descent_compact,
    solve_soft_impute,
)


Array = np.ndarray


@dataclass
class MatrixConfig:
    m: int = 60
    n: int = 60
    rank: int = 4
    factor_distribution: str = "gaussian"
    singular_value_profile: str = "linear"
    singular_scale: float = 1.0
    coherence_mode: str = "incoherent"
    coherence_strength: float = 0.9
    sparse_additive_fraction: float = 0.0
    sparse_additive_scale: float = 0.0


@dataclass
class StructureConfig:
    mode: str = "none"
    strength: float = 0.0
    secondary_strength: float = 0.0


@dataclass
class MissingnessFieldConfig:
    mode: str = "random"
    concentration: float = 1.0
    width: float = 0.15
    row_index: int | None = None
    col_index: int | None = None
    value_dependence: float = 0.0


@dataclass
class MaskSamplingConfig:
    observed_fraction: float = 0.35
    exact_fraction: bool = True
    threshold: float | None = None
    invert_scores: bool = False


@dataclass
class SplitConfig:
    validation_fraction: float = 0.15
    test_fraction: float = 0.15


@dataclass
class NoiseConfig:
    mode: str = "gaussian"
    std: float = 0.0
    outlier_fraction: float = 0.0
    outlier_scale: float = 0.0


@dataclass
class ScenarioConfig:
    matrix: MatrixConfig = field(default_factory=MatrixConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    missingness_field: MissingnessFieldConfig = field(default_factory=MissingnessFieldConfig)
    mask_sampling: MaskSamplingConfig = field(default_factory=MaskSamplingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)


@dataclass
class MaskSplit:
    observed_mask: Array
    train_mask: Array
    fit_mask: Array
    validation_mask: Array
    test_mask: Array


@dataclass
class Scenario:
    config: ScenarioConfig
    seed: int
    X_true: Array
    X_structured: Array
    missingness_field: Array
    mask_split: MaskSplit
    Y_observed: Array
    metadata: dict[str, Any]


@dataclass
class MethodConfig:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    label: str | None = None


@dataclass
class MethodResult:
    method_config: MethodConfig
    name: str
    label: str
    X_hat: Array
    history: list[dict[str, float]]
    runtime_sec: float
    iterations: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    scenario: Scenario
    method_results: list[MethodResult]
    records: list[dict[str, Any]]


VALID_METHODS = {
    "soft_impute",
    "als",
    "rgd",
    "rgd_l2",
    "rgd_l2_selected",
    "compact_rgd",
    "compact_rgd_l2",
    "compact_l2_selected",
}


def make_rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _require_positive_int(name: str, value: int) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def _require_fraction(name: str, value: float) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}.")


def validate_scenario_config(config: ScenarioConfig) -> None:
    """Проверяет настройки сценария до запуска численных методов."""
    _require_positive_int("matrix.m", config.matrix.m)
    _require_positive_int("matrix.n", config.matrix.n)
    _require_positive_int("matrix.rank", config.matrix.rank)
    if config.matrix.rank > min(config.matrix.m, config.matrix.n):
        raise ValueError(
            "matrix.rank must be no larger than min(matrix.m, matrix.n), "
            f"got rank={config.matrix.rank}, m={config.matrix.m}, n={config.matrix.n}."
        )

    if config.matrix.singular_scale < 0.0:
        raise ValueError(f"matrix.singular_scale must be non-negative, got {config.matrix.singular_scale}.")
    _require_fraction("matrix.sparse_additive_fraction", config.matrix.sparse_additive_fraction)
    if config.matrix.sparse_additive_scale < 0.0:
        raise ValueError(
            f"matrix.sparse_additive_scale must be non-negative, got {config.matrix.sparse_additive_scale}."
        )

    if config.missingness_field.width <= 0.0:
        raise ValueError(f"missingness_field.width must be positive, got {config.missingness_field.width}.")
    if config.missingness_field.concentration < 0.0:
        raise ValueError(
            f"missingness_field.concentration must be non-negative, got {config.missingness_field.concentration}."
        )

    _require_fraction("mask_sampling.observed_fraction", config.mask_sampling.observed_fraction)
    _require_fraction("split.validation_fraction", config.split.validation_fraction)
    _require_fraction("split.test_fraction", config.split.test_fraction)
    if config.split.validation_fraction + config.split.test_fraction >= 1.0:
        raise ValueError(
            "split.validation_fraction + split.test_fraction must be less than 1, "
            f"got {config.split.validation_fraction + config.split.test_fraction}."
        )

    if config.noise.std < 0.0:
        raise ValueError(f"noise.std must be non-negative, got {config.noise.std}.")
    _require_fraction("noise.outlier_fraction", config.noise.outlier_fraction)
    if config.noise.outlier_scale < 0.0:
        raise ValueError(f"noise.outlier_scale must be non-negative, got {config.noise.outlier_scale}.")


def _require_plotting() -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib недоступен в текущем Python-окружении. "
            "Нужно открыть notebook в ядре, где установлен matplotlib."
        )


def _as_sortable(value: Any) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _format_tick_value(value: float, sig_digits: int = 4) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.{sig_digits}g}"


def _numeric_series_overlap(
    grouped: dict[str, list[tuple[Any, float]]],
) -> bool:
    per_x: dict[float, list[float]] = {}
    for points in grouped.values():
        for value, metric in points:
            x_value = float(value)
            per_x.setdefault(x_value, []).append(float(metric))
    for metrics in per_x.values():
        seen: set[float] = set()
        for metric in metrics:
            rounded = round(metric, 12)
            if rounded in seen:
                return True
            seen.add(rounded)
    return False


def _series_offsets(
    count: int,
    *,
    reference_values: Sequence[float],
) -> list[float]:
    if count <= 1:
        return [0.0]
    unique_x = sorted(set(reference_values))
    if len(unique_x) > 1:
        min_gap = min(b - a for a, b in zip(unique_x, unique_x[1:], strict=False) if b > a)
        base_offset = 0.12 * min_gap
    else:
        scale = abs(unique_x[0]) if unique_x else 1.0
        base_offset = 0.03 * max(scale, 1.0)
    center = 0.5 * (count - 1)
    return [(index - center) * base_offset for index in range(count)]


def _dedupe_values(seq: Sequence[float]) -> list[float]:
    out: list[float] = []
    for value in seq:
        if value not in out:
            out.append(value)
    return out


def _nearest_values_on_scale(
    values: Sequence[float],
    *,
    max_ticks: int,
    transform: Callable[[float], float],
) -> list[float]:
    unique_values = _dedupe_values(values)
    if max_ticks <= 0 or len(unique_values) <= max_ticks:
        return unique_values
    if max_ticks == 1:
        return [unique_values[0]]
    if max_ticks == 2:
        return [unique_values[0], unique_values[-1]]

    transformed = [transform(value) for value in unique_values]
    left = transformed[0]
    right = transformed[-1]
    chosen: list[float] = []
    for index in range(max_ticks):
        target = left + (right - left) * index / (max_ticks - 1)
        nearest = min(
            unique_values,
            key=lambda value: (
                abs(transform(value) - target),
                abs(value - unique_values[0]),
            ),
        )
        chosen.append(nearest)

    chosen.extend([unique_values[0], unique_values[-1]])
    return _dedupe_values(chosen)


def _select_tick_values(
    values: Sequence[float],
    max_ticks: int,
    *,
    scale: str = "linear",
) -> list[float]:
    unique_values = _dedupe_values(values)
    if max_ticks <= 0 or len(unique_values) <= max_ticks:
        return unique_values
    if max_ticks == 1:
        return [unique_values[0]]
    if max_ticks == 2:
        return [unique_values[0], unique_values[-1]]

    if scale == "log":
        positive = [value for value in unique_values if value > 0]
        if positive:
            return _nearest_values_on_scale(
                positive,
                max_ticks=max_ticks,
                transform=lambda value: math.log10(value),
            )

    if scale == "symlog":
        zero_values = [value for value in unique_values if abs(value) < 1e-15]
        positive = [value for value in unique_values if value > 0]
        negative = [value for value in unique_values if value < 0]
        if positive and not negative:
            slots = max_ticks - (1 if zero_values else 0)
            selected = _nearest_values_on_scale(
                positive,
                max_ticks=max(2, slots),
                transform=lambda value: math.log10(value),
            )
            if zero_values:
                selected = [0.0] + selected
            return _dedupe_values(selected)

    selected: list[float] = []
    for index in range(max_ticks):
        pos = round(index * (len(unique_values) - 1) / (max_ticks - 1))
        selected.append(unique_values[pos])
    return _dedupe_values(selected)


def _draw_distribution(shape: tuple[int, int], distribution: str, rng: np.random.Generator) -> Array:
    if distribution == "gaussian":
        return rng.normal(size=shape)
    if distribution == "uniform":
        return rng.uniform(-1.0, 1.0, size=shape)
    if distribution == "rademacher":
        return rng.choice([-1.0, 1.0], size=shape)
    raise ValueError(f"Unknown factor distribution: {distribution}")


def _orthonormalize_columns(Z: Array) -> Array:
    Q, _ = np.linalg.qr(Z)
    return Q[:, : Z.shape[1]]


def _build_singular_values(rank: int, profile: str, scale: float) -> Array:
    if rank <= 0:
        return np.zeros(0)
    if profile == "linear":
        return scale * np.linspace(rank, 1.0, rank)
    if profile == "flat":
        return scale * np.ones(rank)
    if profile == "geometric":
        return scale * rank * np.geomspace(1.0, 0.1, rank)
    raise ValueError(f"Unknown singular value profile: {profile}")


def _inject_coherence(Q: Array, strength: float, anchor_index: int = 0) -> Array:
    if strength <= 0.0 or Q.shape[1] == 0:
        return Q
    spike = np.zeros(Q.shape[0])
    spike[anchor_index % Q.shape[0]] = 1.0
    mixed = (1.0 - strength) * Q[:, 0] + strength * spike
    norm = np.linalg.norm(mixed)
    if norm == 0.0:
        return Q
    candidate = Q.copy()
    candidate[:, 0] = mixed / norm
    return _orthonormalize_columns(candidate)


def generate_base_matrix(config: MatrixConfig, rng: np.random.Generator) -> Array:
    rank = min(config.rank, config.m, config.n)
    U = _orthonormalize_columns(_draw_distribution((config.m, rank), config.factor_distribution, rng))
    V = _orthonormalize_columns(_draw_distribution((config.n, rank), config.factor_distribution, rng))

    if config.coherence_mode in {"coherent_rows", "coherent_both"}:
        U = _inject_coherence(U, config.coherence_strength, anchor_index=0)
    if config.coherence_mode in {"coherent_cols", "coherent_both"}:
        V = _inject_coherence(V, config.coherence_strength, anchor_index=0)
    if config.coherence_mode not in {"incoherent", "coherent_rows", "coherent_cols", "coherent_both"}:
        raise ValueError(f"Unknown coherence_mode: {config.coherence_mode}")

    singular_values = _build_singular_values(rank, config.singular_value_profile, config.singular_scale)
    X = U @ np.diag(singular_values) @ V.T

    if config.sparse_additive_fraction > 0.0 and config.sparse_additive_scale > 0.0:
        total = config.m * config.n
        count = min(total, max(1, int(round(config.sparse_additive_fraction * total))))
        chosen = rng.choice(total, size=count, replace=False)
        sparse = np.zeros_like(X)
        sparse.reshape(-1)[chosen] = rng.normal(scale=config.sparse_additive_scale, size=count)
        X = X + sparse
    return X


def apply_structure(X: Array, config: StructureConfig, rng: np.random.Generator) -> Array:
    if config.mode == "none" or config.strength == 0.0:
        return X.copy()

    m, n = X.shape
    if config.mode == "row_scale":
        weights = np.linspace(1.0 - config.strength, 1.0 + config.strength, m)
        return weights[:, None] * X
    if config.mode == "column_scale":
        weights = np.linspace(1.0 - config.strength, 1.0 + config.strength, n)
        return X * weights[None, :]
    if config.mode == "checkerboard":
        pattern = np.fromfunction(lambda i, j: (-1.0) ** (i + j), X.shape)
        return X * (1.0 + config.strength * pattern)
    if config.mode == "additive_block":
        out = X.copy()
        rows = max(1, int(round(max(config.strength, 0.1) * 0.25 * m)))
        cols = max(1, int(round(max(config.strength, 0.1) * 0.25 * n)))
        start_row = int(rng.integers(0, max(1, m - rows + 1)))
        start_col = int(rng.integers(0, max(1, n - cols + 1)))
        out[start_row : start_row + rows, start_col : start_col + cols] += max(config.secondary_strength, 1.0)
        return out
    raise ValueError(f"Unknown structure mode: {config.mode}")


def build_missingness_field(X: Array, config: MissingnessFieldConfig, rng: np.random.Generator) -> Array:
    m, n = X.shape
    rr, cc = np.indices((m, n))
    field = 1e-6 * rng.normal(size=(m, n))

    if config.mode == "random":
        field += rng.random(size=(m, n))
    elif config.mode == "block":
        row_center = int(rng.integers(0, m))
        col_center = int(rng.integers(0, n))
        row_width = max(1.0, config.width * m)
        col_width = max(1.0, config.width * n)
        field += config.concentration * np.exp(
            -0.5 * (((rr - row_center) / row_width) ** 2 + ((cc - col_center) / col_width) ** 2)
        )
    elif config.mode == "clustered":
        row_width = max(1.0, config.width * m)
        col_width = max(1.0, config.width * n)
        for _ in range(3):
            row_center = int(rng.integers(0, m))
            col_center = int(rng.integers(0, n))
            field += config.concentration * np.exp(
                -0.5 * (((rr - row_center) / row_width) ** 2 + ((cc - col_center) / col_width) ** 2)
            )
    elif config.mode in {"row_focus", "row_stripe"}:
        row = config.row_index if config.row_index is not None else int(rng.integers(0, m))
        field[row, :] += config.concentration
        field += 0.05 * rng.random(size=(m, n))
    elif config.mode in {"column_focus", "column_stripe"}:
        col = config.col_index if config.col_index is not None else int(rng.integers(0, n))
        field[:, col] += config.concentration
        field += 0.05 * rng.random(size=(m, n))
    elif config.mode == "value_based":
        values = np.abs(X)
        denom = max(float(np.max(values)), 1e-12)
        field += values / denom
    else:
        raise ValueError(f"Unknown missingness_field mode: {config.mode}")

    if config.value_dependence != 0.0:
        values = np.abs(X)
        denom = max(float(np.max(values)), 1e-12)
        field += config.value_dependence * (values / denom)
    return field


def sample_observed_mask(
    missingness_field: Array,
    config: MaskSamplingConfig,
    rng: np.random.Generator,
) -> Array:
    total = missingness_field.size
    observed_count = min(total, max(1, int(round(config.observed_fraction * total))))
    scores = missingness_field.reshape(-1).copy()
    scores += 1e-9 * rng.normal(size=scores.shape)
    if config.invert_scores:
        scores = -scores

    if config.exact_fraction:
        order = np.argsort(scores)  # малые значения чаще попадают в наблюдения
        observed = np.zeros(total, dtype=bool)
        observed[order[:observed_count]] = True
        return observed.reshape(missingness_field.shape)

    threshold = config.threshold
    if threshold is None:
        threshold = float(np.quantile(scores, config.observed_fraction))
    observed = scores <= threshold
    if not np.any(observed):
        observed[np.argmin(scores)] = True
    return observed.reshape(missingness_field.shape)


def split_observed_mask(
    observed_mask: Array,
    config: SplitConfig,
    rng: np.random.Generator,
) -> MaskSplit:
    observed_idx = np.flatnonzero(observed_mask.reshape(-1))
    if len(observed_idx) < 3:
        raise ValueError("Not enough observed entries to create fit / validation / test splits.")

    permutation = rng.permutation(observed_idx)
    total_observed = len(observed_idx)
    test_count = min(max(1, int(round(config.test_fraction * total_observed))), total_observed - 2)
    test_idx = permutation[:test_count]
    remaining = permutation[test_count:]
    val_count = 0
    if config.validation_fraction > 0.0 and len(remaining) >= 2:
        val_count = min(max(1, int(round(config.validation_fraction * len(remaining)))), len(remaining) - 1)
    validation_idx = remaining[:val_count]
    fit_idx = remaining[val_count:]

    flat_size = observed_mask.size
    train_mask = np.zeros(flat_size, dtype=bool)
    fit_mask = np.zeros(flat_size, dtype=bool)
    validation_mask = np.zeros(flat_size, dtype=bool)
    test_mask = np.zeros(flat_size, dtype=bool)
    fit_mask[fit_idx] = True
    validation_mask[validation_idx] = True
    test_mask[test_idx] = True
    train_mask[fit_idx] = True
    train_mask[validation_idx] = True
    shape = observed_mask.shape

    return MaskSplit(
        observed_mask=observed_mask.copy(),
        train_mask=train_mask.reshape(shape),
        fit_mask=fit_mask.reshape(shape),
        validation_mask=validation_mask.reshape(shape),
        test_mask=test_mask.reshape(shape),
    )


def build_observed_matrix(
    X_structured: Array,
    observed_mask: Array,
    config: NoiseConfig,
    rng: np.random.Generator,
) -> Array:
    Y = np.zeros_like(X_structured)
    if config.mode not in {"gaussian", "none"}:
        raise ValueError(f"Unknown noise mode: {config.mode}")

    if config.mode == "gaussian" and config.std > 0.0:
        noise = rng.normal(scale=config.std, size=X_structured.shape)
    else:
        noise = np.zeros_like(X_structured)

    Y[observed_mask] = X_structured[observed_mask] + noise[observed_mask]

    if config.outlier_fraction > 0.0 and config.outlier_scale > 0.0:
        candidate_idx = np.flatnonzero(observed_mask.reshape(-1))
        outlier_count = min(
            len(candidate_idx),
            max(1, int(round(config.outlier_fraction * len(candidate_idx)))),
        )
        chosen = rng.choice(candidate_idx, size=outlier_count, replace=False)
        Y.reshape(-1)[chosen] += rng.normal(scale=config.outlier_scale, size=outlier_count)
    return Y


def package_scenario(
    *,
    config: ScenarioConfig,
    seed: int,
    X_true: Array,
    X_structured: Array,
    missingness_field: Array,
    mask_split: MaskSplit,
    Y_observed: Array,
    metadata: dict[str, Any] | None = None,
) -> Scenario:
    combined_metadata = {
        "observed_fraction_effective": float(np.mean(mask_split.observed_mask)),
        "train_fraction_effective": float(np.mean(mask_split.train_mask)),
        "fit_fraction_effective": float(np.mean(mask_split.fit_mask)),
        "validation_fraction_effective": float(np.mean(mask_split.validation_mask)),
        "test_fraction_effective": float(np.mean(mask_split.test_mask)),
    }
    if metadata:
        combined_metadata.update(metadata)
    return Scenario(
        config=copy.deepcopy(config),
        seed=seed,
        X_true=np.array(X_true, copy=True),
        X_structured=np.array(X_structured, copy=True),
        missingness_field=np.array(missingness_field, copy=True),
        mask_split=MaskSplit(
            observed_mask=np.array(mask_split.observed_mask, copy=True),
            train_mask=np.array(mask_split.train_mask, copy=True),
            fit_mask=np.array(mask_split.fit_mask, copy=True),
            validation_mask=np.array(mask_split.validation_mask, copy=True),
            test_mask=np.array(mask_split.test_mask, copy=True),
        ),
        Y_observed=np.array(Y_observed, copy=True),
        metadata=combined_metadata,
    )


def assemble_scenario(config: ScenarioConfig, seed: int = 42) -> Scenario:
    validate_scenario_config(config)
    rng = make_rng(seed)
    X_true = generate_base_matrix(config.matrix, rng)
    X_structured = apply_structure(X_true, config.structure, rng)
    field = build_missingness_field(X_structured, config.missingness_field, rng)
    observed_mask = sample_observed_mask(field, config.mask_sampling, rng)
    mask_split = split_observed_mask(observed_mask, config.split, rng)
    Y_observed = build_observed_matrix(X_structured, mask_split.observed_mask, config.noise, rng)
    return package_scenario(
        config=config,
        seed=seed,
        X_true=X_true,
        X_structured=X_structured,
        missingness_field=field,
        mask_split=mask_split,
        Y_observed=Y_observed,
        metadata={
            "assembly_mode": "from_config",
        },
    )


def make_method(name: str, label: str | None = None, **params: Any) -> MethodConfig:
    if name not in VALID_METHODS:
        raise ValueError(f"Unknown method: {name}")
    return MethodConfig(name=name, label=label, params=dict(params))


def _solver_result_to_method_result(
    method: MethodConfig,
    solver_result: SolverResult,
    *,
    name: str | None = None,
    label: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> MethodResult:
    return MethodResult(
        method_config=method,
        name=name or method.name,
        label=label or method.label or method.name,
        X_hat=solver_result.X_hat,
        history=solver_result.history,
        runtime_sec=solver_result.runtime_sec,
        iterations=solver_result.iterations,
        metadata=metadata or {},
    )


def _scenario_rng(scenario: Scenario, offset: int) -> np.random.Generator:
    return make_rng(scenario.seed + offset)


def _resolve_rank(method: MethodConfig, scenario: Scenario) -> int:
    requested = method.params.get("rank", scenario.config.matrix.rank)
    return max(1, min(int(requested), scenario.config.matrix.m, scenario.config.matrix.n))


def _resolve_init(method: MethodConfig) -> str:
    return str(method.params.get("init", "spectral"))


def run_soft_impute_method(scenario: Scenario, method: MethodConfig) -> MethodResult:
    rank = _resolve_rank(method, scenario)
    result = solve_soft_impute(
        scenario.Y_observed,
        scenario.mask_split.train_mask,
        rank_hint=rank,
        max_iter=int(method.params.get("max_iter", 110)),
        lambda_scale=float(method.params.get("lambda_scale", 0.15)),
    )
    return _solver_result_to_method_result(
        method,
        result,
        metadata={"assumed_rank": rank},
    )


def run_als_method(scenario: Scenario, method: MethodConfig) -> MethodResult:
    rank = _resolve_rank(method, scenario)
    init = _resolve_init(method)
    result = solve_als(
        scenario.Y_observed,
        scenario.mask_split.train_mask,
        rank=rank,
        rng=_scenario_rng(scenario, 201),
        init=init,
        reg=float(method.params.get("reg", 1e-3)),
        max_iter=int(method.params.get("max_iter", 60)),
    )
    return _solver_result_to_method_result(method, result, metadata={"assumed_rank": rank})


def run_rgd_method(scenario: Scenario, method: MethodConfig) -> MethodResult:
    rank = _resolve_rank(method, scenario)
    init = _resolve_init(method)
    result = solve_riemannian_gradient_descent(
        scenario.Y_observed,
        scenario.mask_split.train_mask,
        rank=rank,
        rng=_scenario_rng(scenario, 301),
        init=init,
        max_iter=int(method.params.get("max_iter", 140)),
        method_name=method.label or method.name,
        method_family="rgd",
    )
    return _solver_result_to_method_result(method, result, metadata={"assumed_rank": rank})


def run_rgd_l2_method(scenario: Scenario, method: MethodConfig) -> MethodResult:
    rank = _resolve_rank(method, scenario)
    init = _resolve_init(method)
    l2_reg = float(method.params["l2_reg"])
    result = solve_riemannian_gradient_descent(
        scenario.Y_observed,
        scenario.mask_split.train_mask,
        rank=rank,
        rng=_scenario_rng(scenario, 311),
        init=init,
        max_iter=int(method.params.get("max_iter", 140)),
        l2_reg=l2_reg,
        method_name=method.label or method.name,
        method_family="rgd_l2",
    )
    return _solver_result_to_method_result(
        method,
        result,
        metadata={"assumed_rank": rank, "l2_reg": l2_reg},
    )


def run_compact_rgd_method(scenario: Scenario, method: MethodConfig) -> MethodResult:
    rank = _resolve_rank(method, scenario)
    init = _resolve_init(method)
    result = solve_riemannian_gradient_descent_compact(
        scenario.Y_observed,
        scenario.mask_split.train_mask,
        rank=rank,
        rng=_scenario_rng(scenario, 401),
        init=init,
        max_iter=int(method.params.get("max_iter", 120)),
        method_name=method.label or method.name,
        method_family="compact_rgd",
    )
    return _solver_result_to_method_result(method, result, metadata={"assumed_rank": rank})


def run_compact_rgd_l2_method(scenario: Scenario, method: MethodConfig) -> MethodResult:
    rank = _resolve_rank(method, scenario)
    init = _resolve_init(method)
    l2_reg = float(method.params["l2_reg"])
    result = solve_riemannian_gradient_descent_compact(
        scenario.Y_observed,
        scenario.mask_split.train_mask,
        rank=rank,
        rng=_scenario_rng(scenario, 411),
        init=init,
        max_iter=int(method.params.get("max_iter", 120)),
        l2_reg=l2_reg,
        method_name=method.label or method.name,
        method_family="compact_rgd_l2",
    )
    return _solver_result_to_method_result(
        method,
        result,
        metadata={"assumed_rank": rank, "l2_reg": l2_reg},
    )


def _select_l2_and_refit(
    scenario: Scenario,
    method: MethodConfig,
    *,
    compact: bool,
) -> MethodResult:
    rank = _resolve_rank(method, scenario)
    init = _resolve_init(method)
    l2_grid = list(method.params.get("l2_grid", [1e-3, 1e-2, 3e-2]))
    max_iter = int(method.params.get("max_iter", 120 if compact else 140))

    candidate_runs: list[tuple[float, float, SolverResult]] = []
    for index, l2_value in enumerate(l2_grid):
        if compact:
            candidate = solve_riemannian_gradient_descent_compact(
                scenario.Y_observed,
                scenario.mask_split.fit_mask,
                rank=rank,
                rng=_scenario_rng(scenario, 501 + index),
                init=init,
                max_iter=max_iter,
                l2_reg=float(l2_value),
                method_name=f"candidate_{l2_value:.0e}",
                method_family="compact_rgd_l2_candidate",
            )
        else:
            candidate = solve_riemannian_gradient_descent(
                scenario.Y_observed,
                scenario.mask_split.fit_mask,
                rank=rank,
                rng=_scenario_rng(scenario, 601 + index),
                init=init,
                max_iter=max_iter,
                l2_reg=float(l2_value),
                method_name=f"candidate_{l2_value:.0e}",
                method_family="rgd_l2_candidate",
            )
        validation_rmse = masked_rmse(candidate.X_hat, scenario.X_structured, scenario.mask_split.validation_mask)
        candidate_runs.append((validation_rmse, float(l2_value), candidate))

    best_validation_rmse, best_l2, _ = min(candidate_runs, key=lambda item: (item[0], item[1]))

    if compact:
        final = solve_riemannian_gradient_descent_compact(
            scenario.Y_observed,
            scenario.mask_split.train_mask,
            rank=rank,
            rng=_scenario_rng(scenario, 701),
            init=init,
            max_iter=max_iter,
            l2_reg=best_l2,
            method_name=method.label or method.name,
            method_family="compact_l2_selected",
        )
    else:
        final = solve_riemannian_gradient_descent(
            scenario.Y_observed,
            scenario.mask_split.train_mask,
            rank=rank,
            rng=_scenario_rng(scenario, 801),
            init=init,
            max_iter=max_iter,
            l2_reg=best_l2,
            method_name=method.label or method.name,
            method_family="rgd_l2_selected",
        )

    return _solver_result_to_method_result(
        method,
        final,
        metadata={
            "assumed_rank": rank,
            "selected_l2_reg": best_l2,
            "selection_validation_rmse": best_validation_rmse,
        },
    )


def run_method(scenario: Scenario, method: MethodConfig) -> MethodResult:
    if method.name == "soft_impute":
        return run_soft_impute_method(scenario, method)
    if method.name == "als":
        return run_als_method(scenario, method)
    if method.name == "rgd":
        return run_rgd_method(scenario, method)
    if method.name == "rgd_l2":
        return run_rgd_l2_method(scenario, method)
    if method.name == "rgd_l2_selected":
        return _select_l2_and_refit(scenario, method, compact=False)
    if method.name == "compact_rgd":
        return run_compact_rgd_method(scenario, method)
    if method.name == "compact_rgd_l2":
        return run_compact_rgd_l2_method(scenario, method)
    if method.name == "compact_l2_selected":
        return _select_l2_and_refit(scenario, method, compact=True)
    raise ValueError(f"Unknown method: {method.name}")


def run_methods(scenario: Scenario, methods: Sequence[MethodConfig]) -> list[MethodResult]:
    return [run_method(scenario, method) for method in methods]


def flatten_config(config: ScenarioConfig) -> dict[str, Any]:
    nested = asdict(config)
    flat: dict[str, Any] = {}
    for section_name, section_values in nested.items():
        for key, value in section_values.items():
            flat[f"{section_name}.{key}"] = value
    return flat


def evaluate_method_result(scenario: Scenario, method_result: MethodResult) -> dict[str, Any]:
    final_train_objective = (
        method_result.history[-1]["train_objective"]
        if method_result.history and "train_objective" in method_result.history[-1]
        else observed_objective(method_result.X_hat, scenario.Y_observed, scenario.mask_split.train_mask)
    )
    record = {
        "seed": scenario.seed,
        "method": method_result.name,
        "label": method_result.label,
        "m": scenario.config.matrix.m,
        "n": scenario.config.matrix.n,
        "true_rank": scenario.config.matrix.rank,
        "assumed_rank": method_result.metadata.get("assumed_rank", scenario.config.matrix.rank),
        "iterations": method_result.iterations,
        "runtime_sec": round(method_result.runtime_sec, 6),
        "final_train_objective": round(float(final_train_objective), 6),
        "train_rmse_observed": round(masked_rmse(method_result.X_hat, scenario.Y_observed, scenario.mask_split.train_mask), 6),
        "validation_rmse": round(masked_rmse(method_result.X_hat, scenario.X_structured, scenario.mask_split.validation_mask), 6),
        "test_rmse": round(masked_rmse(method_result.X_hat, scenario.X_structured, scenario.mask_split.test_mask), 6),
        "test_mae": round(masked_mae(method_result.X_hat, scenario.X_structured, scenario.mask_split.test_mask), 6),
        "relative_fro_error": round(relative_fro_error(method_result.X_hat, scenario.X_structured), 6),
        "selected_l2_reg": method_result.metadata.get("selected_l2_reg", ""),
        "selection_validation_rmse": method_result.metadata.get("selection_validation_rmse", ""),
    }
    record.update(flatten_config(scenario.config))
    return record


def run_experiment_on_scenario(
    scenario: Scenario,
    methods: Sequence[MethodConfig],
) -> ExperimentResult:
    results = run_methods(scenario, methods)
    records = [evaluate_method_result(scenario, result) for result in results]
    return ExperimentResult(scenario=scenario, method_results=results, records=records)


def run_single_experiment(
    config: ScenarioConfig,
    methods: Sequence[MethodConfig],
    seed: int = 42,
) -> ExperimentResult:
    scenario = assemble_scenario(config, seed=seed)
    return run_experiment_on_scenario(scenario, methods)


def run_many_experiments(
    configs: Sequence[ScenarioConfig],
    methods: Sequence[MethodConfig],
    seeds: Sequence[int],
) -> list[ExperimentResult]:
    runs: list[ExperimentResult] = []
    for config in configs:
        for seed in seeds:
            runs.append(run_single_experiment(config, methods, seed=seed))
    return runs


def collect_records(runs: Sequence[ExperimentResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        rows.extend(run.records)
    return rows


def apply_overrides(config: ScenarioConfig, overrides: dict[str, Any]) -> ScenarioConfig:
    clone = copy.deepcopy(config)
    for dotted_path, value in overrides.items():
        target: Any = clone
        parts = dotted_path.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        setattr(target, parts[-1], value)
    return clone


def build_config_variants(
    base_config: ScenarioConfig,
    grid: dict[str, Sequence[Any]],
) -> list[ScenarioConfig]:
    if not grid:
        return [copy.deepcopy(base_config)]
    keys = list(grid.keys())
    variants: list[ScenarioConfig] = []
    for values in itertools.product(*(grid[key] for key in keys)):
        overrides = {key: value for key, value in zip(keys, values, strict=True)}
        variants.append(apply_overrides(base_config, overrides))
    return variants


def one_factor_sweep(
    base_config: ScenarioConfig,
    parameter_path: str,
    values: Sequence[Any],
    methods: Sequence[MethodConfig],
    seeds: Sequence[int],
) -> list[ExperimentResult]:
    configs = [apply_overrides(base_config, {parameter_path: value}) for value in values]
    return run_many_experiments(configs, methods, seeds)


def aggregate_records(
    records: Sequence[dict[str, Any]],
    by: Sequence[str],
    metrics: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if metrics is None:
        metrics = (
            "test_rmse",
            "test_mae",
            "relative_fro_error",
            "runtime_sec",
            "iterations",
            "train_rmse_observed",
        )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in records:
        grouped.setdefault(tuple(row[key] for key in by), []).append(dict(row))

    output: list[dict[str, Any]] = []
    for key_values, rows in grouped.items():
        item = {key: value for key, value in zip(by, key_values, strict=True)}
        item["n_runs"] = len(rows)
        for metric in metrics:
            values = [float(row[metric]) for row in rows]
            item[f"avg_{metric}"] = round(float(np.mean(values)), 6)
            item[f"std_{metric}"] = round(float(np.std(values, ddof=1)), 6) if len(values) > 1 else 0.0
        output.append(item)
    return sorted(output, key=lambda row: tuple(_as_sortable(row[key]) for key in by))


def to_dataframe(rows: Sequence[dict[str, Any]]):
    if pd is None:
        return list(rows)
    return pd.DataFrame(rows)


def write_csv(rows: Sequence[dict[str, Any]], path: str | Path) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(
    rows: Sequence[dict[str, Any]],
    columns: Sequence[tuple[str, str]],
    caption: str,
    label: str,
    path: str | Path,
) -> None:
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{'l' + 'r' * (len(columns) - 1)}}}",
        r"\toprule",
        " & ".join(title for _, title in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        values = [str(row[key]).replace("_", r"\_") for key, _ in columns]
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def singular_values(X: Array) -> Array:
    return np.linalg.svd(X, compute_uv=False)


def plot_matrix(X: Array, title: str = "", cmap: str = "viridis", save_path: str | Path | None = None):
    _require_plotting()
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(X, aspect="auto", cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180)
    if _running_inside_ipython():
        plt.show()
    return fig, ax


def plot_mask(mask: Array, title: str = "", save_path: str | Path | None = None):
    return plot_matrix(mask.astype(float), title=title, cmap="gray_r", save_path=save_path)


def plot_singular_values(
    X: Array,
    title: str = "",
    log_scale: bool = False,
    save_path: str | Path | None = None,
):
    _require_plotting()
    values = singular_values(X)
    plotted_values = np.maximum(values, np.finfo(float).tiny) if log_scale else values
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(values) + 1), plotted_values, marker="o", linewidth=2)
    ax.set_title(title or ("Singular values (log scale)" if log_scale else "Singular values"))
    ax.set_xlabel("Index")
    ax.set_ylabel("Singular value")
    if log_scale:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180)
    if _running_inside_ipython():
        plt.show()
    return fig, ax


def plot_metric(
    rows: Sequence[dict[str, Any]],
    *,
    x: str,
    y: str,
    hue: str = "method",
    title: str = "",
    xlabel: str | None = None,
    ylabel: str | None = None,
    x_tick_rotation: float = 0.0,
    x_tick_sig_digits: int = 4,
    max_xticks: int | None = None,
    x_scale: str = "linear",
    symlog_linthresh: float | None = None,
    separate_series_on_overlap: bool = True,
    figsize: tuple[float, float] = (8.4, 4.8),
    save_path: str | Path | None = None,
):
    _require_plotting()
    grouped: dict[str, list[tuple[Any, float]]] = {}
    for row in rows:
        grouped.setdefault(str(row[hue]), []).append((row[x], float(row[y])))

    all_x = [value for values in grouped.values() for value, _ in values]
    numeric_x = True
    for value in all_x:
        try:
            float(value)
        except (TypeError, ValueError):
            numeric_x = False
            break

    fig, ax = plt.subplots(figsize=figsize)
    if numeric_x:
        unique_x = sorted({float(value) for value in all_x})
        overlap = separate_series_on_overlap and _numeric_series_overlap(grouped)
        offsets = _series_offsets(len(grouped), reference_values=unique_x) if overlap else [0.0] * len(grouped)
        markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
        for series_index, (label, points) in enumerate(grouped.items()):
            numeric_points = [(float(value), metric) for value, metric in points]
            numeric_points.sort(key=lambda item: item[0])
            offset = offsets[series_index]
            ax.plot(
                [item[0] + offset for item in numeric_points],
                [item[1] for item in numeric_points],
                marker=markers[series_index % len(markers)],
                linewidth=2,
                label=label,
            )
        tick_values = (
            _select_tick_values(unique_x, max_xticks, scale=x_scale)
            if max_xticks is not None
            else unique_x
        )
        if x_scale == "log":
            ax.set_xscale("log")
        elif x_scale == "symlog":
            positive_x = [value for value in unique_x if value > 0]
            min_positive = min(positive_x) if positive_x else 1e-6
            ax.set_xscale("symlog", linthresh=symlog_linthresh or min_positive)
        ax.set_xticks(tick_values)
        ax.set_xticklabels(
            [_format_tick_value(value, sig_digits=x_tick_sig_digits) for value in tick_values],
            rotation=x_tick_rotation,
        )
        ax.margins(x=0.03)
        ax.minorticks_off()
    else:
        categories = sorted({str(value) for value in all_x}, key=_as_sortable)
        index_map = {category: idx for idx, category in enumerate(categories)}
        markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
        for series_index, (label, points) in enumerate(grouped.items()):
            cat_points = [(str(value), metric) for value, metric in points]
            cat_points.sort(key=lambda item: _as_sortable(item[0]))
            ax.plot(
                [index_map[item[0]] for item in cat_points],
                [item[1] for item in cat_points],
                marker=markers[series_index % len(markers)],
                linewidth=2,
                label=label,
            )
        category_positions = list(range(len(categories)))
        if max_xticks is not None:
            shown_positions = _select_tick_values(category_positions, max_xticks)
            shown_labels = [categories[int(pos)] for pos in shown_positions]
        else:
            shown_positions = category_positions
            shown_labels = categories
        ax.set_xticks(shown_positions)
        ax.set_xticklabels(shown_labels, rotation=x_tick_rotation or 20)

    ax.set_title(title or f"{y} vs {x}")
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180)
    if _running_inside_ipython():
        plt.show()
    return fig, ax


def plot_histories(
    method_results: Sequence[MethodResult],
    metric_key: str = "train_objective",
    title: str = "",
    save_path: str | Path | None = None,
):
    _require_plotting()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for result in method_results:
        xs = []
        ys = []
        for idx, item in enumerate(result.history, start=1):
            if metric_key in item:
                xs.append(idx)
                ys.append(float(item[metric_key]))
        if xs:
            ax.plot(xs, ys, linewidth=2, label=result.label)
    ax.set_title(title or metric_key)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric_key)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180)
    if _running_inside_ipython():
        plt.show()
    return fig, ax


__all__ = [
    "ExperimentResult",
    "MaskSamplingConfig",
    "MaskSplit",
    "MatrixConfig",
    "MethodConfig",
    "MethodResult",
    "MissingnessFieldConfig",
    "NoiseConfig",
    "Scenario",
    "ScenarioConfig",
    "SplitConfig",
    "StructureConfig",
    "aggregate_records",
    "apply_overrides",
    "apply_structure",
    "assemble_scenario",
    "build_config_variants",
    "build_missingness_field",
    "build_observed_matrix",
    "collect_records",
    "evaluate_method_result",
    "generate_base_matrix",
    "make_method",
    "make_rng",
    "one_factor_sweep",
    "package_scenario",
    "plot_histories",
    "plot_mask",
    "plot_matrix",
    "plot_singular_values",
    "plot_metric",
    "run_many_experiments",
    "run_experiment_on_scenario",
    "run_method",
    "run_methods",
    "run_single_experiment",
    "sample_observed_mask",
    "singular_values",
    "split_observed_mask",
    "to_dataframe",
    "validate_scenario_config",
    "write_csv",
    "write_latex_table",
]
