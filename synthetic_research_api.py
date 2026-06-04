"""Однофакторные эксперименты для сравнения методов восстановления матриц."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import synthetic_api as api


@dataclass
class ScenarioPreset:
    name: str
    config: api.ScenarioConfig
    description: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class MethodSpec:
    name: str
    label: str
    params: dict[str, Any] = field(default_factory=dict)
    use_true_rank: bool = True


@dataclass
class SweepSpec:
    name: str
    path: str
    values: list[Any] = field(default_factory=list)
    alias: str | None = None
    title: str = ""


@dataclass
class SuiteSpec:
    name: str
    base_scenario: ScenarioPreset
    methods: list[MethodSpec]
    sweeps: list[SweepSpec]
    seeds: list[int] = field(default_factory=lambda: [42])
    description: str = ""
    output_dir: str = ""


@dataclass
class SweepRun:
    sweep: SweepSpec
    scenarios: list[ScenarioPreset]
    manifest: list[dict[str, Any]]
    records: list[dict[str, Any]]
    summary: list[dict[str, Any]]
    runs: list[api.ExperimentResult]


@dataclass
class SuiteRun:
    suite: SuiteSpec
    root: Path
    sweep_runs: list[SweepRun]
    manifest: list[dict[str, Any]]
    records: list[dict[str, Any]]
    summary: list[dict[str, Any]]


def make_scenario(
    name: str,
    *,
    m: int = 60,
    n: int = 60,
    rank: int = 4,
    observed_fraction: float = 0.35,
    noise_std: float = 0.02,
    missingness: str = "random",
    description: str = "",
    tags: Sequence[str] | None = None,
    config: api.ScenarioConfig | None = None,
) -> ScenarioPreset:
    """Создает базовый сценарий без длинного перечисления вложенных настроек."""

    scenario = copy.deepcopy(config) if config is not None else api.ScenarioConfig()
    scenario.matrix.m = m
    scenario.matrix.n = n
    scenario.matrix.rank = rank
    scenario.mask_sampling.observed_fraction = observed_fraction
    scenario.missingness_field.mode = missingness
    scenario.noise.std = noise_std
    return ScenarioPreset(name=name, config=scenario, description=description, tags=list(tags or []))


def make_method(
    name: str,
    label: str | None = None,
    *,
    use_true_rank: bool = True,
    **params: Any,
) -> MethodSpec:
    """Описывает метод; ранг по умолчанию берется из текущего сценария."""

    return MethodSpec(
        name=name,
        label=label or name,
        params=dict(params),
        use_true_rank=use_true_rank,
    )


def make_sweep(
    name: str,
    path: str,
    values: Sequence[Any] | None = None,
    *,
    alias: str | None = None,
    title: str = "",
) -> SweepSpec:
    """Описывает однофакторный эксперимент: меняется только один параметр."""

    return SweepSpec(name=name, path=path, values=list(values or []), alias=alias, title=title)


def make_suite(
    name: str,
    base_scenario: ScenarioPreset,
    methods: Sequence[MethodSpec],
    sweeps: Sequence[SweepSpec],
    *,
    seeds: Sequence[int] | None = None,
    description: str = "",
    output_dir: str = "",
) -> SuiteSpec:
    return SuiteSpec(
        name=name,
        base_scenario=copy.deepcopy(base_scenario),
        methods=list(copy.deepcopy(methods)),
        sweeps=list(copy.deepcopy(sweeps)),
        seeds=[int(seed) for seed in (seeds or [42])],
        description=description,
        output_dir=output_dir,
    )


def rank_grid(
    scenario: api.ScenarioConfig,
    *,
    dense_cutoff: int = 5,
    growth: float = 1.5,
    max_rank: int = 100,
) -> list[int]:
    """Сетка рангов: малые значения подряд, дальше шаг растет."""

    limit = max(1, min(int(max_rank), scenario.matrix.m, scenario.matrix.n))
    dense_limit = min(limit, max(1, int(dense_cutoff)))
    values = list(range(1, dense_limit + 1))

    while values[-1] < limit:
        next_value = max(values[-1] + 1, int(math.ceil(values[-1] * growth)))
        values.append(min(next_value, limit))

    return values


def signal_rms(scenario: api.ScenarioConfig) -> float:
    rank = min(scenario.matrix.rank, scenario.matrix.m, scenario.matrix.n)
    singulars = api.build_singular_values(
        rank,
        scenario.matrix.singular_value_profile,
        scenario.matrix.singular_scale,
    )
    if len(singulars) == 0:
        return 1.0
    fro_sq = sum(float(value) ** 2 for value in singulars)
    return math.sqrt(fro_sq / max(1, scenario.matrix.m * scenario.matrix.n))


def noise_grid(
    scenario: api.ScenarioConfig,
    levels: Sequence[float] | None = None,
    *,
    digits: int = 6,
) -> list[float]:
    """Сетка шума относительно среднего масштаба сигнала."""

    levels = levels or [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    values = [round(float(level) * signal_rms(scenario), digits) for level in levels]
    return list(dict.fromkeys(values))


def missingness_grid() -> list[str]:
    return ["random", "block", "row_stripe"]


def values_for_sweep(sweep: SweepSpec, base_scenario: ScenarioPreset) -> list[Any]:
    if sweep.values:
        return list(sweep.values)
    if sweep.path == "matrix.rank":
        return rank_grid(base_scenario.config)
    if sweep.path == "noise.std":
        return noise_grid(base_scenario.config)
    if sweep.path == "missingness_field.mode":
        return missingness_grid()
    if sweep.path == "mask_sampling.observed_fraction":
        return [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 0.95]
    raise ValueError(f"Для '{sweep.path}' нет автоматической сетки. Передайте values=[...] явно.")


def sweep_alias(sweep: SweepSpec) -> str:
    return sweep.alias or sweep.path


def scenario_value(scenario: api.ScenarioConfig, path: str) -> Any:
    flat = api.flatten_config(scenario)
    if path not in flat:
        raise KeyError(f"Неизвестный параметр сценария: {path}")
    return flat[path]


def scenario_for_value(base_scenario: ScenarioPreset, sweep: SweepSpec, value: Any) -> ScenarioPreset:
    alias = sweep_alias(sweep)
    config = api.apply_overrides(base_scenario.config, {sweep.path: value})
    return ScenarioPreset(
        name=f"{base_scenario.name}__{alias}_{value}",
        description=f"{base_scenario.description} | {alias}={value}".strip(" |"),
        tags=[*base_scenario.tags, alias],
        config=config,
    )


def build_scenarios(base_scenario: ScenarioPreset, sweep: SweepSpec) -> list[ScenarioPreset]:
    return [scenario_for_value(base_scenario, sweep, value) for value in values_for_sweep(sweep, base_scenario)]


def materialize_methods(methods: Sequence[MethodSpec], scenario: api.ScenarioConfig) -> list[api.MethodConfig]:
    result: list[api.MethodConfig] = []
    for method in methods:
        params = dict(method.params)
        if method.use_true_rank and "rank" not in params:
            params["rank"] = scenario.matrix.rank
        result.append(api.make_method(method.name, label=method.label, **params))
    return result


def preview_suite(suite: SuiteSpec) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sweep in suite.sweeps:
        alias = sweep_alias(sweep)
        for i, scenario in enumerate(build_scenarios(suite.base_scenario, sweep), start=1):
            rows.append(
                {
                    "suite_name": suite.name,
                    "sweep_name": sweep.name,
                    "scenario_index": i,
                    "scenario_name": scenario.name,
                    alias: scenario_value(scenario.config, sweep.path),
                    **api.flatten_config(scenario.config),
                }
            )
    return rows


def run_sweep(
    base_scenario: ScenarioPreset,
    methods: Sequence[MethodSpec],
    sweep: SweepSpec,
    seeds: Sequence[int],
) -> SweepRun:
    scenarios = build_scenarios(base_scenario, sweep)
    alias = sweep_alias(sweep)
    manifest: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    runs: list[api.ExperimentResult] = []

    for scenario_index, scenario in enumerate(scenarios, start=1):
        value = scenario_value(scenario.config, sweep.path)
        metadata = {
            "sweep_name": sweep.name,
            "sweep_title": sweep.title,
            "scenario_index": scenario_index,
            "scenario_name": scenario.name,
            alias: value,
        }
        manifest.append({**metadata, **api.flatten_config(scenario.config)})

        current_methods = materialize_methods(methods, scenario.config)
        for seed in seeds:
            run = api.run_single_experiment(scenario.config, current_methods, seed=seed)
            runs.append(run)
            for row in run.records:
                records.append({**metadata, **row})

    summary = api.aggregate_records(records, by=[alias, "method"])
    for row in summary:
        row["sweep_name"] = sweep.name
        row["sweep_title"] = sweep.title
    return SweepRun(sweep=sweep, scenarios=scenarios, manifest=manifest, records=records, summary=summary, runs=runs)


def run_suite(suite: SuiteSpec, root: str | Path = ".") -> SuiteRun:
    root_path = Path(root).expanduser().resolve()
    sweep_runs = [run_sweep(suite.base_scenario, suite.methods, sweep, suite.seeds) for sweep in suite.sweeps]

    manifest: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for sweep_run in sweep_runs:
        for row in sweep_run.manifest:
            manifest.append({"suite_name": suite.name, **row})
        for row in sweep_run.records:
            records.append({"suite_name": suite.name, **row})
        for row in sweep_run.summary:
            summary.append({"suite_name": suite.name, **row})

    return SuiteRun(
        suite=copy.deepcopy(suite),
        root=root_path,
        sweep_runs=sweep_runs,
        manifest=manifest,
        records=records,
        summary=summary,
    )


def summary_for(run: SuiteRun, sweep_name: str) -> list[dict[str, Any]]:
    return [row for row in run.summary if row["sweep_name"] == sweep_name]


def label_methods(rows: Sequence[dict[str, Any]], methods: Sequence[MethodSpec]) -> list[dict[str, Any]]:
    labels = {method.name: method.label for method in methods}
    result: list[dict[str, Any]] = []
    for row in rows:
        current = dict(row)
        current["method"] = labels.get(str(current.get("method")), current.get("method"))
        result.append(current)
    return result


def _plotting():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Для построения графиков нужен matplotlib.") from exc
    return plt


def _as_sortable(value: Any) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _format_tick(value: float, digits: int) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.{digits}g}"


def _select_ticks(values: Sequence[float], max_count: int | None) -> list[float]:
    unique = sorted(dict.fromkeys(float(value) for value in values))
    if max_count is None or max_count <= 0 or len(unique) <= max_count:
        return unique
    if max_count == 1:
        return [unique[0]]
    return [unique[round(index * (len(unique) - 1) / (max_count - 1))] for index in range(max_count)]


def _needs_x_offsets(grouped: dict[str, list[tuple[Any, float]]]) -> bool:
    by_x: dict[float, set[float]] = {}
    for points in grouped.values():
        for x_value, y_value in points:
            key = float(x_value)
            rounded = round(float(y_value), 12)
            if rounded in by_x.setdefault(key, set()):
                return True
            by_x[key].add(rounded)
    return False


def _x_offsets(series_count: int, x_values: Sequence[float]) -> list[float]:
    if series_count <= 1:
        return [0.0]
    unique_x = sorted(set(float(value) for value in x_values))
    gaps = [b - a for a, b in zip(unique_x, unique_x[1:], strict=False) if b > a]
    step = 0.08 * min(gaps) if gaps else 0.03 * max(abs(unique_x[0]) if unique_x else 1.0, 1.0)
    center = 0.5 * (series_count - 1)
    return [(index - center) * step for index in range(series_count)]


def plot_suite_metric(
    run: SuiteRun,
    sweep_name: str,
    *,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    x_scale: str = "linear",
    x_tick_rotation: float = 0,
    x_tick_sig_digits: int = 4,
    max_xticks: int | None = None,
):
    rows = label_methods(summary_for(run, sweep_name), run.suite.methods)
    grouped: dict[str, list[tuple[Any, float]]] = {}
    for row in rows:
        grouped.setdefault(str(row["method"]), []).append((row[x], float(row[y])))

    plt = _plotting()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    all_x = [point_x for points in grouped.values() for point_x, _ in points]

    numeric_x = True
    for value in all_x:
        try:
            float(value)
        except (TypeError, ValueError):
            numeric_x = False
            break

    if numeric_x:
        x_values = sorted(float(value) for value in all_x)
        offsets = _x_offsets(len(grouped), x_values) if _needs_x_offsets(grouped) else [0.0] * len(grouped)
        for index, (label, points) in enumerate(grouped.items()):
            numeric_points = sorted((float(point_x), point_y) for point_x, point_y in points)
            ax.plot(
                [point_x + offsets[index] for point_x, _ in numeric_points],
                [point_y for _, point_y in numeric_points],
                marker=markers[index % len(markers)],
                linewidth=2,
                label=label,
            )
        ticks = _select_ticks(x_values, max_xticks)
        if x_scale == "log":
            ax.set_xscale("log")
        elif x_scale == "symlog":
            positive = [value for value in x_values if value > 0]
            linthresh = min(positive) if positive else 1e-6
            ax.set_xscale("symlog", linthresh=linthresh)
        ax.set_xticks(ticks)
        ax.set_xticklabels([_format_tick(value, x_tick_sig_digits) for value in ticks], rotation=x_tick_rotation)
        ax.margins(x=0.03)
        ax.minorticks_off()
    else:
        categories = sorted({str(value) for value in all_x}, key=_as_sortable)
        positions = {category: index for index, category in enumerate(categories)}
        for index, (label, points) in enumerate(grouped.items()):
            categorical_points = sorted((str(point_x), point_y) for point_x, point_y in points)
            ax.plot(
                [positions[point_x] for point_x, _ in categorical_points],
                [point_y for _, point_y in categorical_points],
                marker=markers[index % len(markers)],
                linewidth=2,
                label=label,
            )
        tick_positions = _select_ticks(list(range(len(categories))), max_xticks)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([categories[int(pos)] for pos in tick_positions], rotation=x_tick_rotation or 15)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_sweep_report(
    run: SuiteRun,
    sweep_name: str,
    *,
    x: str,
    title: str,
    xlabel: str,
    x_scale: str = "linear",
    x_tick_rotation: float = 0,
    x_tick_sig_digits: int = 4,
    max_xticks: int | None = None,
) -> None:
    plot_suite_metric(
        run,
        sweep_name,
        x=x,
        y="avg_test_rmse",
        title=f"{title}: RMSE на тестовой части",
        xlabel=xlabel,
        ylabel="Средний RMSE на тестовой части",
        x_scale=x_scale,
        x_tick_rotation=x_tick_rotation,
        x_tick_sig_digits=x_tick_sig_digits,
        max_xticks=max_xticks,
    )
    plot_suite_metric(
        run,
        sweep_name,
        x=x,
        y="avg_runtime_sec",
        title=f"{title}: время работы",
        xlabel=xlabel,
        ylabel="Среднее время, сек",
        x_scale=x_scale,
        x_tick_rotation=x_tick_rotation,
        x_tick_sig_digits=x_tick_sig_digits,
        max_xticks=max_xticks,
    )
    plot_suite_metric(
        run,
        sweep_name,
        x=x,
        y="avg_iterations",
        title=f"{title}: число итераций",
        xlabel=xlabel,
        ylabel="Среднее число итераций",
        x_scale=x_scale,
        x_tick_rotation=x_tick_rotation,
        x_tick_sig_digits=x_tick_sig_digits,
        max_xticks=max_xticks,
    )


def research_tree(root: str | Path) -> dict[str, Path]:
    root_path = Path(root).expanduser().resolve()
    configs = root_path / "research_configs"
    paths = {
        "root": root_path,
        "presets": configs / "presets" / "scenarios",
        "suites": configs / "suites",
        "outputs": root_path / "research_outputs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def scenario_to_dict(scenario: ScenarioPreset) -> dict[str, Any]:
    return {
        "name": scenario.name,
        "description": scenario.description,
        "tags": list(scenario.tags),
        "scenario": asdict(scenario.config),
    }


def scenario_from_dict(data: dict[str, Any]) -> ScenarioPreset:
    scenario_data = data.get("scenario", {})
    config = api.ScenarioConfig(
        matrix=api.MatrixConfig(**scenario_data.get("matrix", {})),
        structure=api.StructureConfig(**scenario_data.get("structure", {})),
        missingness_field=api.MissingnessFieldConfig(**scenario_data.get("missingness_field", {})),
        mask_sampling=api.MaskSamplingConfig(**scenario_data.get("mask_sampling", {})),
        split=api.SplitConfig(**scenario_data.get("split", {})),
        noise=api.NoiseConfig(**scenario_data.get("noise", {})),
    )
    return ScenarioPreset(
        name=data["name"],
        description=data.get("description", ""),
        tags=list(data.get("tags", [])),
        config=config,
    )


def method_to_dict(method: MethodSpec) -> dict[str, Any]:
    return asdict(method)


def method_from_dict(data: dict[str, Any]) -> MethodSpec:
    return MethodSpec(
        name=data["name"],
        label=data.get("label") or data["name"],
        params=dict(data.get("params", {})),
        use_true_rank=bool(data.get("use_true_rank", True)),
    )


def sweep_to_dict(sweep: SweepSpec) -> dict[str, Any]:
    return asdict(sweep)


def sweep_from_dict(data: dict[str, Any]) -> SweepSpec:
    return SweepSpec(
        name=data["name"],
        path=data["path"],
        values=list(data.get("values", [])),
        alias=data.get("alias"),
        title=data.get("title", ""),
    )


def suite_to_dict(suite: SuiteSpec) -> dict[str, Any]:
    return {
        "name": suite.name,
        "description": suite.description,
        "output_dir": suite.output_dir,
        "seeds": list(suite.seeds),
        "base_scenario": scenario_to_dict(suite.base_scenario),
        "methods": [method_to_dict(method) for method in suite.methods],
        "sweeps": [sweep_to_dict(sweep) for sweep in suite.sweeps],
    }


def suite_from_dict(data: dict[str, Any]) -> SuiteSpec:
    return SuiteSpec(
        name=data["name"],
        description=data.get("description", ""),
        output_dir=data.get("output_dir", ""),
        seeds=[int(seed) for seed in data.get("seeds", [42])],
        base_scenario=scenario_from_dict(data["base_scenario"]),
        methods=[method_from_dict(item) for item in data.get("methods", [])],
        sweeps=[sweep_from_dict(item) for item in data.get("sweeps", [])],
    )


def save_scenario(scenario: ScenarioPreset, root: str | Path, filename: str | None = None) -> Path:
    path = research_tree(root)["presets"] / (filename or f"{scenario.name}.json")
    path.write_text(json.dumps(scenario_to_dict(scenario), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_scenario(path: str | Path) -> ScenarioPreset:
    return scenario_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_suite(suite: SuiteSpec, root: str | Path, filename: str | None = None) -> Path:
    path = research_tree(root)["suites"] / (filename or f"{suite.name}.json")
    path.write_text(json.dumps(suite_to_dict(suite), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_suite(path: str | Path) -> SuiteSpec:
    return suite_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def output_dir_for(run: SuiteRun, output_dir: str | Path | None = None) -> Path:
    if output_dir is not None:
        path = Path(output_dir)
    elif run.suite.output_dir:
        path = Path(run.suite.output_dir)
    else:
        path = research_tree(run.root)["outputs"] / run.suite.name
    return path if path.is_absolute() else run.root / path


def save_run(run: SuiteRun, output_dir: str | Path | None = None) -> dict[str, Path]:
    path = output_dir_for(run, output_dir)
    path.mkdir(parents=True, exist_ok=True)

    suite_path = path / f"{run.suite.name}_suite.json"
    manifest_path = path / f"{run.suite.name}_manifest.csv"
    records_path = path / f"{run.suite.name}_records.csv"
    summary_path = path / f"{run.suite.name}_summary.csv"

    suite_path.write_text(json.dumps(suite_to_dict(run.suite), ensure_ascii=False, indent=2), encoding="utf-8")
    api.write_csv(run.manifest, manifest_path)
    api.write_csv(run.records, records_path)
    api.write_csv(run.summary, summary_path)

    return {
        "suite_json": suite_path,
        "manifest_csv": manifest_path,
        "records_csv": records_path,
        "summary_csv": summary_path,
    }
