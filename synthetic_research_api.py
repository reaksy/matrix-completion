#!/usr/bin/env python3
"""Удобный слой для однофакторных синтетических экспериментов.

Основной сценарий работы:
    базовый сценарий -> несколько однофакторных экспериментов -> общий запуск.

В этом файле нет отдельного CLI и старого grid-search слоя: все запускается
напрямую из Python или ноутбука, чтобы код оставался коротким и проверяемым.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import synthetic_api as base


@dataclass
class ScenarioPreset:
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    scenario: base.ScenarioConfig = field(default_factory=base.ScenarioConfig)


@dataclass
class ParamBinding:
    path: str
    scale: float = 1.0
    offset: float = 0.0
    cast: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    round_digits: int | None = None


@dataclass
class ResearchMethod:
    name: str
    label: str | None = None
    key: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    bindings: dict[str, ParamBinding] = field(default_factory=dict)


@dataclass
class OneFactorExperiment:
    name: str
    vary_path: str
    values: list[Any] = field(default_factory=list)
    base_scenario: ScenarioPreset | None = None
    base_scenario_ref: str | None = None
    methods: list[ResearchMethod] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [42])
    vary_alias: str | None = None
    description: str = ""
    tags: list[str] = field(default_factory=list)
    output_dir: str = ""
    aggregate_by: list[str] = field(default_factory=list)

    def resolved_values(self, root: str | Path) -> list[Any]:
        return resolve_experiment_values(self, root)

    def generated_scenarios(self, root: str | Path) -> list[ScenarioPreset]:
        return generate_experiment_scenarios(self, root)

    def preview(self, root: str | Path) -> list[dict[str, Any]]:
        return preview_experiment(self, root)

    def run(self, root: str | Path) -> "OneFactorExperimentRun":
        return run_experiment(self, root)

    def save(self, root: str | Path, filename: str | None = None) -> Path:
        return save_experiment(self, root, filename=filename)


@dataclass
class OneFactorExperimentRun:
    experiment: OneFactorExperiment
    root: Path
    manifest: list[dict[str, Any]]
    records: list[dict[str, Any]]
    summary: list[dict[str, Any]]
    runs: list[base.ExperimentResult]

    def save(self, output_dir: str | Path | None = None) -> dict[str, Path]:
        return save_experiment_run(self, root=self.root, output_dir=output_dir)


@dataclass
class ExperimentSuite:
    name: str
    experiments: list[OneFactorExperiment] = field(default_factory=list)
    base_scenario: ScenarioPreset | None = None
    base_scenario_ref: str | None = None
    methods: list[ResearchMethod] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [42])
    description: str = ""
    tags: list[str] = field(default_factory=list)
    output_dir: str = ""
    aggregate_by: list[str] = field(default_factory=list)

    def resolved_experiments(self, root: str | Path) -> list[OneFactorExperiment]:
        return resolve_suite_experiments(self, root)

    def preview(self, root: str | Path) -> list[dict[str, Any]]:
        return preview_suite(self, root)

    def run(self, root: str | Path) -> "ExperimentSuiteRun":
        return run_suite(self, root)

    def save(self, root: str | Path, filename: str | None = None) -> Path:
        return save_suite(self, root, filename=filename)


@dataclass
class ExperimentSuiteRun:
    suite: ExperimentSuite
    root: Path
    experiment_runs: list[OneFactorExperimentRun]
    manifest: list[dict[str, Any]]
    records: list[dict[str, Any]]
    summary: list[dict[str, Any]]

    def save(self, output_dir: str | Path | None = None) -> dict[str, Path]:
        return save_suite_run(self, root=self.root, output_dir=output_dir)


def _root_path(root: str | Path) -> Path:
    return Path(root).expanduser().resolve()


def ensure_research_tree(root: str | Path) -> dict[str, Path]:
    root_path = _root_path(root)
    config_dir = root_path / "research_configs"
    preset_dir = config_dir / "presets" / "scenarios"
    experiment_dir = config_dir / "experiments"
    suite_dir = config_dir / "suites"
    output_dir = root_path / "research_outputs"

    for path in (config_dir, preset_dir, experiment_dir, suite_dir, output_dir):
        path.mkdir(parents=True, exist_ok=True)

    return {
        "root": root_path,
        "configs": config_dir,
        "presets": preset_dir,
        "experiments": experiment_dir,
        "suites": suite_dir,
        "outputs": output_dir,
    }


def resolve_preset_path(reference: str | Path, root: str | Path) -> Path:
    ref = Path(reference)
    if ref.is_absolute():
        return ref
    return _root_path(root) / ref


def resolve_experiment_base_preset(
    experiment: OneFactorExperiment,
    root: str | Path,
) -> ScenarioPreset:
    if experiment.base_scenario is not None:
        return copy.deepcopy(experiment.base_scenario)
    if experiment.base_scenario_ref:
        return load_scenario(resolve_preset_path(experiment.base_scenario_ref, root))
    raise ValueError("Для эксперимента нужно задать base_scenario или base_scenario_ref.")


def resolve_experiment_base_scenario(
    experiment: OneFactorExperiment,
    root: str | Path,
) -> base.ScenarioConfig:
    return copy.deepcopy(resolve_experiment_base_preset(experiment, root).scenario)


def _deduplicate_preserve_order(values: Sequence[Any]) -> list[Any]:
    output: list[Any] = []
    for value in values:
        if value not in output:
            output.append(value)
    return output


def default_rank_values(
    scenario_config: base.ScenarioConfig,
    *,
    dense_cutoff: int = 5,
    growth: float = 1.5,
    default_cap: int = 100,
    r_max: int | None = None,
) -> list[int]:
    """Строит разумную сетку рангов для матриц разных размеров.

    Малые ранги проверяются подряд, дальше шаг растет геометрически. Так сетка
    остается читаемой и для 10x10, и для 1000x1000.
    """

    d = min(scenario_config.matrix.m, scenario_config.matrix.n)
    requested_limit = default_cap if r_max is None else int(r_max)
    limit = max(1, min(requested_limit, d))
    dense_limit = min(limit, max(1, dense_cutoff))

    values = list(range(1, dense_limit + 1))
    current = values[-1]
    while current < limit:
        next_rank = max(current + 1, int(math.ceil(current * growth)))
        next_rank = min(next_rank, limit)
        if next_rank == values[-1]:
            break
        values.append(int(next_rank))
        current = int(next_rank)
    return values


def estimate_signal_rms(scenario_config: base.ScenarioConfig) -> float:
    rank = min(
        scenario_config.matrix.rank,
        scenario_config.matrix.m,
        scenario_config.matrix.n,
    )
    singulars = base._build_singular_values(  # type: ignore[attr-defined]
        rank,
        scenario_config.matrix.singular_value_profile,
        scenario_config.matrix.singular_scale,
    )
    if len(singulars) == 0:
        return 0.0
    fro_sq = float(sum(float(value) * float(value) for value in singulars))
    return float(math.sqrt(fro_sq / max(1, scenario_config.matrix.m * scenario_config.matrix.n)))


def default_noise_values(
    scenario_config: base.ScenarioConfig,
    *,
    relative_levels: Sequence[float] | None = None,
    round_digits: int = 6,
) -> list[float]:
    """Строит сетку шума относительно среднего масштаба сигнала."""

    levels = list(relative_levels or [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0])
    rms = estimate_signal_rms(scenario_config)
    if rms <= 0.0:
        rms = 1.0
    values = [round(float(level) * rms, round_digits) for level in levels]
    return _deduplicate_preserve_order(values)


def default_observed_fraction_values() -> list[float]:
    return [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 0.95]


def default_missingness_mode_values() -> list[str]:
    return ["random", "block", "row_stripe"]


def resolve_experiment_values(
    experiment: OneFactorExperiment,
    root: str | Path,
) -> list[Any]:
    if experiment.values:
        return list(experiment.values)

    if experiment.base_scenario is None and experiment.base_scenario_ref is None:
        raise ValueError(
            "У эксперимента нет ручной сетки values и еще нет базового сценария. "
            "Передайте values=[...], задайте base_scenario/base_scenario_ref "
            "или сначала вызовите suite.resolved_experiments(...)."
        )

    scenario_config = resolve_experiment_base_scenario(experiment, root)
    if experiment.vary_path == "matrix.rank":
        return default_rank_values(scenario_config)
    if experiment.vary_path == "noise.std":
        return default_noise_values(scenario_config)
    if experiment.vary_path == "mask_sampling.observed_fraction":
        return default_observed_fraction_values()
    if experiment.vary_path == "missingness_field.mode":
        return default_missingness_mode_values()

    raise ValueError(
        f"Для параметра '{experiment.vary_path}' нет автоматической сетки. "
        "Передайте values=[...] явно."
    )


def make_scenario(
    name: str,
    *,
    description: str = "",
    tags: Sequence[str] | None = None,
    config: base.ScenarioConfig | None = None,
    matrix: base.MatrixConfig | None = None,
    structure: base.StructureConfig | None = None,
    missingness_field: base.MissingnessFieldConfig | None = None,
    mask_sampling: base.MaskSamplingConfig | None = None,
    split: base.SplitConfig | None = None,
    noise: base.NoiseConfig | None = None,
) -> ScenarioPreset:
    scenario_config = copy.deepcopy(config) if config is not None else base.ScenarioConfig()
    if matrix is not None:
        scenario_config.matrix = copy.deepcopy(matrix)
    if structure is not None:
        scenario_config.structure = copy.deepcopy(structure)
    if missingness_field is not None:
        scenario_config.missingness_field = copy.deepcopy(missingness_field)
    if mask_sampling is not None:
        scenario_config.mask_sampling = copy.deepcopy(mask_sampling)
    if split is not None:
        scenario_config.split = copy.deepcopy(split)
    if noise is not None:
        scenario_config.noise = copy.deepcopy(noise)

    return ScenarioPreset(
        name=name,
        description=description,
        tags=list(tags or []),
        scenario=scenario_config,
    )


def clone_scenario(
    scenario: ScenarioPreset,
    *,
    name: str | None = None,
    description: str | None = None,
    tags: Sequence[str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> ScenarioPreset:
    cloned = copy.deepcopy(scenario)
    cloned.name = name or cloned.name
    if description is not None:
        cloned.description = description
    if tags is not None:
        cloned.tags = list(tags)
    if overrides:
        cloned.scenario = base.apply_overrides(cloned.scenario, overrides)
    return cloned


def sweep_scenarios(
    base_scenario: ScenarioPreset,
    parameter_path: str,
    values: Sequence[Any],
    *,
    alias: str | None = None,
    name_template: str | None = None,
    description_template: str | None = None,
    extra_tags: Sequence[str] | None = None,
) -> list[ScenarioPreset]:
    output: list[ScenarioPreset] = []
    alias = alias or parameter_path

    for value in values:
        scenario_name = name_template.format(value=value, alias=alias) if name_template else f"{base_scenario.name}__{alias}_{value}"
        scenario_description = (
            description_template.format(value=value, alias=alias)
            if description_template
            else f"{base_scenario.description} | {alias}={value}".strip(" |")
        )
        tags = list(base_scenario.tags)
        if extra_tags:
            tags.extend(extra_tags)
        tags.append(alias)

        output.append(
            clone_scenario(
                base_scenario,
                name=scenario_name,
                description=scenario_description,
                tags=tags,
                overrides={parameter_path: value},
            )
        )
    return output


def generate_experiment_scenarios(
    experiment: OneFactorExperiment,
    root: str | Path,
) -> list[ScenarioPreset]:
    base_preset = resolve_experiment_base_preset(experiment, root)
    values = resolve_experiment_values(experiment, root)
    return sweep_scenarios(
        base_preset,
        experiment.vary_path,
        values,
        alias=_experiment_axis_alias(experiment),
        extra_tags=experiment.tags,
    )


def make_binding(
    path: str,
    *,
    scale: float = 1.0,
    offset: float = 0.0,
    cast: str | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
    round_digits: int | None = None,
) -> ParamBinding:
    return ParamBinding(
        path=path,
        scale=scale,
        offset=offset,
        cast=cast,
        minimum=minimum,
        maximum=maximum,
        round_digits=round_digits,
    )


def make_method(
    name: str,
    *,
    label: str | None = None,
    key: str | None = None,
    params: dict[str, Any] | None = None,
    bindings: dict[str, ParamBinding] | None = None,
) -> ResearchMethod:
    return ResearchMethod(
        name=name,
        label=label,
        key=key,
        params=dict(params or {}),
        bindings=dict(bindings or {}),
    )


def make_experiment(
    name: str,
    *,
    vary_path: str,
    values: Sequence[Any] | None = None,
    base_scenario: ScenarioPreset | None = None,
    base_scenario_ref: str | None = None,
    methods: Sequence[ResearchMethod] | None = None,
    seeds: Sequence[int] | None = None,
    vary_alias: str | None = None,
    description: str = "",
    tags: Sequence[str] | None = None,
    output_dir: str = "",
    aggregate_by: Sequence[str] | None = None,
) -> OneFactorExperiment:
    return OneFactorExperiment(
        name=name,
        vary_path=vary_path,
        values=list(values or []),
        base_scenario=copy.deepcopy(base_scenario),
        base_scenario_ref=base_scenario_ref,
        methods=list(copy.deepcopy(list(methods or []))),
        seeds=[int(seed) for seed in (seeds or [42])],
        vary_alias=vary_alias,
        description=description,
        tags=list(tags or []),
        output_dir=output_dir,
        aggregate_by=list(aggregate_by or []),
    )


def make_suite(
    name: str,
    *,
    experiments: Sequence[OneFactorExperiment] | None = None,
    base_scenario: ScenarioPreset | None = None,
    base_scenario_ref: str | None = None,
    methods: Sequence[ResearchMethod] | None = None,
    seeds: Sequence[int] | None = None,
    description: str = "",
    tags: Sequence[str] | None = None,
    output_dir: str = "",
    aggregate_by: Sequence[str] | None = None,
) -> ExperimentSuite:
    return ExperimentSuite(
        name=name,
        experiments=list(copy.deepcopy(list(experiments or []))),
        base_scenario=copy.deepcopy(base_scenario),
        base_scenario_ref=base_scenario_ref,
        methods=list(copy.deepcopy(list(methods or []))),
        seeds=[int(seed) for seed in (seeds or [42])],
        description=description,
        tags=list(tags or []),
        output_dir=output_dir,
        aggregate_by=list(aggregate_by or []),
    )


def _get_flat_value(scenario_config: base.ScenarioConfig, path: str) -> Any:
    flat = base.flatten_config(scenario_config)
    if path not in flat:
        raise KeyError(f"Неизвестный путь в сценарии: {path}")
    return flat[path]


def resolve_binding(binding: ParamBinding, scenario_config: base.ScenarioConfig) -> Any:
    value = _get_flat_value(scenario_config, binding.path)
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, (int, float)):
        value = binding.scale * float(value) + binding.offset
        if binding.minimum is not None:
            value = max(value, binding.minimum)
        if binding.maximum is not None:
            value = min(value, binding.maximum)
        if binding.round_digits is not None:
            value = round(value, binding.round_digits)
        if binding.cast == "int":
            value = int(round(value))
        elif binding.cast == "float":
            value = float(value)
    elif binding.cast == "str":
        value = str(value)
    return value


def materialize_methods(
    methods: Sequence[ResearchMethod],
    scenario_config: base.ScenarioConfig,
) -> list[base.MethodConfig]:
    output: list[base.MethodConfig] = []
    for method in methods:
        params = dict(method.params)
        for param_name, binding in method.bindings.items():
            params[param_name] = resolve_binding(binding, scenario_config)
        output.append(base.make_method(method.name, label=method.label, **params))
    return output


def _merge_tags(base_tags: Sequence[str], extra_tags: Sequence[str]) -> list[str]:
    merged: list[str] = []
    for tag in list(base_tags) + list(extra_tags):
        if tag not in merged:
            merged.append(tag)
    return merged


def resolve_suite_experiments(
    suite: ExperimentSuite,
    root: str | Path,
) -> list[OneFactorExperiment]:
    output: list[OneFactorExperiment] = []
    for experiment in suite.experiments:
        resolved = copy.deepcopy(experiment)
        if resolved.base_scenario is None and resolved.base_scenario_ref is None:
            resolved.base_scenario = copy.deepcopy(suite.base_scenario)
            resolved.base_scenario_ref = suite.base_scenario_ref
        if not resolved.methods:
            resolved.methods = copy.deepcopy(suite.methods)
        if resolved.seeds == [42] and suite.seeds != [42]:
            resolved.seeds = list(suite.seeds)
        if not resolved.output_dir and suite.output_dir:
            resolved.output_dir = str(Path(suite.output_dir) / resolved.name)
        if not resolved.aggregate_by and suite.aggregate_by:
            resolved.aggregate_by = list(suite.aggregate_by)
        resolved.tags = _merge_tags(suite.tags, resolved.tags)
        if not resolved.values:
            resolved.values = resolve_experiment_values(resolved, root)
        output.append(resolved)
    return output


def _experiment_axis_alias(experiment: OneFactorExperiment) -> str:
    return experiment.vary_alias or experiment.vary_path


def _scenario_tags(tags: Sequence[str]) -> str:
    return ",".join(tags)


def _experiment_metadata(
    experiment: OneFactorExperiment,
    scenario: ScenarioPreset,
    scenario_index: int,
) -> dict[str, Any]:
    axis_alias = _experiment_axis_alias(experiment)
    axis_value = _get_flat_value(scenario.scenario, experiment.vary_path)
    return {
        "experiment_name": experiment.name,
        "experiment_description": experiment.description,
        "scenario_index": scenario_index,
        "scenario_name": scenario.name,
        "scenario_description": scenario.description,
        "scenario_tags": _scenario_tags(scenario.tags),
        axis_alias: axis_value,
    }


def _experiment_groupby(experiment: OneFactorExperiment) -> list[str]:
    if experiment.aggregate_by:
        return list(experiment.aggregate_by)
    return [_experiment_axis_alias(experiment), "method"]


def _add_scenario_names_to_summary(
    summary: list[dict[str, Any]],
    experiment: OneFactorExperiment,
    scenarios: Sequence[ScenarioPreset],
) -> list[dict[str, Any]]:
    axis_alias = _experiment_axis_alias(experiment)
    names_by_value = {
        _get_flat_value(scenario.scenario, experiment.vary_path): scenario.name
        for scenario in scenarios
    }

    output: list[dict[str, Any]] = []
    for row in summary:
        updated = dict(row)
        if axis_alias in updated:
            updated["scenario_name"] = names_by_value.get(updated[axis_alias], "")
        output.append(updated)
    return output


def preview_experiment(
    experiment: OneFactorExperiment,
    root: str | Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_index, scenario in enumerate(generate_experiment_scenarios(experiment, root), start=1):
        rows.append(
            {
                **_experiment_metadata(experiment, scenario, scenario_index),
                **base.flatten_config(scenario.scenario),
            }
        )
    return rows


def run_experiment(
    experiment: OneFactorExperiment,
    root: str | Path,
) -> OneFactorExperimentRun:
    root_path = _root_path(root)
    scenarios = generate_experiment_scenarios(experiment, root_path)

    runs: list[base.ExperimentResult] = []
    manifest: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    for scenario_index, scenario in enumerate(scenarios, start=1):
        metadata = _experiment_metadata(experiment, scenario, scenario_index)
        manifest.append({**metadata, **base.flatten_config(scenario.scenario)})
        methods = materialize_methods(experiment.methods, scenario.scenario)

        for seed in experiment.seeds:
            run = base.run_single_experiment(scenario.scenario, methods, seed=seed)
            runs.append(run)
            for row in run.records:
                records.append({**metadata, **row})

    summary = base.aggregate_records(records, by=_experiment_groupby(experiment))
    summary = _add_scenario_names_to_summary(summary, experiment, scenarios)
    for row in summary:
        row["experiment_name"] = experiment.name
        row["experiment_description"] = experiment.description

    return OneFactorExperimentRun(
        experiment=copy.deepcopy(experiment),
        root=root_path,
        manifest=manifest,
        records=records,
        summary=summary,
        runs=runs,
    )


def preview_suite(
    suite: ExperimentSuite,
    root: str | Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment in resolve_suite_experiments(suite, root):
        for row in experiment.preview(root):
            rows.append(
                {
                    "suite_name": suite.name,
                    "suite_description": suite.description,
                    **row,
                }
            )
    return rows


def run_suite(
    suite: ExperimentSuite,
    root: str | Path,
) -> ExperimentSuiteRun:
    root_path = _root_path(root)
    experiment_runs: list[OneFactorExperimentRun] = []
    manifest: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []

    for experiment in resolve_suite_experiments(suite, root_path):
        experiment_run = experiment.run(root_path)
        experiment_runs.append(experiment_run)

        for row in experiment_run.manifest:
            manifest.append(
                {
                    "suite_name": suite.name,
                    "suite_description": suite.description,
                    **row,
                }
            )
        for row in experiment_run.records:
            records.append(
                {
                    "suite_name": suite.name,
                    "suite_description": suite.description,
                    **row,
                }
            )
        for row in experiment_run.summary:
            summary.append(
                {
                    "suite_name": suite.name,
                    "suite_description": suite.description,
                    **row,
                }
            )

    return ExperimentSuiteRun(
        suite=copy.deepcopy(suite),
        root=root_path,
        experiment_runs=experiment_runs,
        manifest=manifest,
        records=records,
        summary=summary,
    )


def scenario_config_to_dict(config: base.ScenarioConfig) -> dict[str, Any]:
    return asdict(config)


def scenario_config_from_dict(payload: dict[str, Any]) -> base.ScenarioConfig:
    return base.ScenarioConfig(
        matrix=base.MatrixConfig(**payload.get("matrix", {})),
        structure=base.StructureConfig(**payload.get("structure", {})),
        missingness_field=base.MissingnessFieldConfig(**payload.get("missingness_field", {})),
        mask_sampling=base.MaskSamplingConfig(**payload.get("mask_sampling", {})),
        split=base.SplitConfig(**payload.get("split", {})),
        noise=base.NoiseConfig(**payload.get("noise", {})),
    )


def scenario_preset_to_dict(preset: ScenarioPreset) -> dict[str, Any]:
    return {
        "name": preset.name,
        "description": preset.description,
        "tags": list(preset.tags),
        "scenario": scenario_config_to_dict(preset.scenario),
    }


def scenario_preset_from_dict(payload: dict[str, Any]) -> ScenarioPreset:
    return ScenarioPreset(
        name=payload["name"],
        description=payload.get("description", ""),
        tags=list(payload.get("tags", [])),
        scenario=scenario_config_from_dict(payload.get("scenario", {})),
    )


def binding_to_dict(binding: ParamBinding) -> dict[str, Any]:
    return asdict(binding)


def binding_from_dict(payload: dict[str, Any]) -> ParamBinding:
    return ParamBinding(**payload)


def method_to_dict(method: ResearchMethod) -> dict[str, Any]:
    return {
        "name": method.name,
        "label": method.label,
        "key": method.key,
        "params": dict(method.params),
        "bindings": {name: binding_to_dict(binding) for name, binding in method.bindings.items()},
    }


def method_from_dict(payload: dict[str, Any]) -> ResearchMethod:
    return ResearchMethod(
        name=payload["name"],
        label=payload.get("label"),
        key=payload.get("key"),
        params=dict(payload.get("params", {})),
        bindings={
            name: binding_from_dict(binding_payload)
            for name, binding_payload in payload.get("bindings", {}).items()
        },
    )


def experiment_to_dict(experiment: OneFactorExperiment) -> dict[str, Any]:
    return {
        "name": experiment.name,
        "description": experiment.description,
        "tags": list(experiment.tags),
        "vary_path": experiment.vary_path,
        "values": list(experiment.values),
        "vary_alias": experiment.vary_alias,
        "base_scenario": (
            None if experiment.base_scenario is None else scenario_preset_to_dict(experiment.base_scenario)
        ),
        "base_scenario_ref": experiment.base_scenario_ref,
        "methods": [method_to_dict(method) for method in experiment.methods],
        "seeds": list(experiment.seeds),
        "output_dir": experiment.output_dir,
        "aggregate_by": list(experiment.aggregate_by),
    }


def experiment_from_dict(payload: dict[str, Any]) -> OneFactorExperiment:
    base_scenario_payload = payload.get("base_scenario")
    return OneFactorExperiment(
        name=payload["name"],
        description=payload.get("description", ""),
        tags=list(payload.get("tags", [])),
        vary_path=payload["vary_path"],
        values=list(payload.get("values", [])),
        vary_alias=payload.get("vary_alias"),
        base_scenario=(
            None
            if base_scenario_payload in (None, {})
            else scenario_preset_from_dict(base_scenario_payload)
        ),
        base_scenario_ref=payload.get("base_scenario_ref"),
        methods=[method_from_dict(item) for item in payload.get("methods", [])],
        seeds=[int(seed) for seed in payload.get("seeds", [42])],
        output_dir=str(payload.get("output_dir", "")),
        aggregate_by=list(payload.get("aggregate_by", [])),
    )


def suite_to_dict(suite: ExperimentSuite) -> dict[str, Any]:
    return {
        "name": suite.name,
        "description": suite.description,
        "tags": list(suite.tags),
        "base_scenario": None if suite.base_scenario is None else scenario_preset_to_dict(suite.base_scenario),
        "base_scenario_ref": suite.base_scenario_ref,
        "methods": [method_to_dict(method) for method in suite.methods],
        "seeds": list(suite.seeds),
        "experiments": [experiment_to_dict(experiment) for experiment in suite.experiments],
        "output_dir": suite.output_dir,
        "aggregate_by": list(suite.aggregate_by),
    }


def suite_from_dict(payload: dict[str, Any]) -> ExperimentSuite:
    base_scenario_payload = payload.get("base_scenario")
    return ExperimentSuite(
        name=payload["name"],
        description=payload.get("description", ""),
        tags=list(payload.get("tags", [])),
        base_scenario=(
            None
            if base_scenario_payload in (None, {})
            else scenario_preset_from_dict(base_scenario_payload)
        ),
        base_scenario_ref=payload.get("base_scenario_ref"),
        methods=[method_from_dict(item) for item in payload.get("methods", [])],
        seeds=[int(seed) for seed in payload.get("seeds", [42])],
        experiments=[experiment_from_dict(item) for item in payload.get("experiments", [])],
        output_dir=str(payload.get("output_dir", "")),
        aggregate_by=list(payload.get("aggregate_by", [])),
    )


def save_scenario(
    scenario: ScenarioPreset,
    root: str | Path,
    filename: str | None = None,
) -> Path:
    tree = ensure_research_tree(root)
    path = tree["presets"] / (filename or f"{scenario.name}.json")
    path.write_text(
        json.dumps(scenario_preset_to_dict(scenario), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_scenario(path: str | Path) -> ScenarioPreset:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return scenario_preset_from_dict(payload)


def save_experiment(
    experiment: OneFactorExperiment,
    root: str | Path,
    filename: str | None = None,
) -> Path:
    tree = ensure_research_tree(root)
    path = tree["experiments"] / (filename or f"{experiment.name}.json")
    path.write_text(
        json.dumps(experiment_to_dict(experiment), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_experiment(path: str | Path) -> OneFactorExperiment:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return experiment_from_dict(payload)


def save_suite(
    suite: ExperimentSuite,
    root: str | Path,
    filename: str | None = None,
) -> Path:
    tree = ensure_research_tree(root)
    path = tree["suites"] / (filename or f"{suite.name}.json")
    path.write_text(
        json.dumps(suite_to_dict(suite), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_suite(path: str | Path) -> ExperimentSuite:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return suite_from_dict(payload)


def _resolve_output_dir(
    root: str | Path,
    *,
    configured_output_dir: str = "",
    default_name: str,
    output_dir: str | Path | None = None,
) -> Path:
    tree = ensure_research_tree(root)
    if output_dir is not None:
        candidate = Path(output_dir)
    elif configured_output_dir:
        candidate = Path(configured_output_dir)
    else:
        candidate = tree["outputs"] / default_name
    return candidate if candidate.is_absolute() else tree["root"] / candidate


def save_experiment_run(
    experiment_run: OneFactorExperimentRun,
    root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    final_output_dir = _resolve_output_dir(
        root,
        configured_output_dir=experiment_run.experiment.output_dir,
        default_name=experiment_run.experiment.name,
        output_dir=output_dir,
    )
    final_output_dir.mkdir(parents=True, exist_ok=True)

    experiment_path = final_output_dir / f"{experiment_run.experiment.name}_experiment.json"
    manifest_path = final_output_dir / f"{experiment_run.experiment.name}_manifest.csv"
    records_path = final_output_dir / f"{experiment_run.experiment.name}_records.csv"
    summary_path = final_output_dir / f"{experiment_run.experiment.name}_summary.csv"

    experiment_path.write_text(
        json.dumps(experiment_to_dict(experiment_run.experiment), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    base.write_csv(experiment_run.manifest, manifest_path)
    base.write_csv(experiment_run.records, records_path)
    base.write_csv(experiment_run.summary, summary_path)

    return {
        "experiment_json": experiment_path,
        "manifest_csv": manifest_path,
        "records_csv": records_path,
        "summary_csv": summary_path,
    }


def save_suite_run(
    suite_run: ExperimentSuiteRun,
    root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    final_output_dir = _resolve_output_dir(
        root,
        configured_output_dir=suite_run.suite.output_dir,
        default_name=suite_run.suite.name,
        output_dir=output_dir,
    )
    final_output_dir.mkdir(parents=True, exist_ok=True)

    suite_path = final_output_dir / f"{suite_run.suite.name}_suite.json"
    manifest_path = final_output_dir / f"{suite_run.suite.name}_manifest.csv"
    records_path = final_output_dir / f"{suite_run.suite.name}_records.csv"
    summary_path = final_output_dir / f"{suite_run.suite.name}_summary.csv"

    suite_path.write_text(
        json.dumps(suite_to_dict(suite_run.suite), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    base.write_csv(suite_run.manifest, manifest_path)
    base.write_csv(suite_run.records, records_path)
    base.write_csv(suite_run.summary, summary_path)

    experiment_dir = final_output_dir / "experiments"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    for experiment_run in suite_run.experiment_runs:
        save_experiment_run(
            experiment_run,
            root=root,
            output_dir=experiment_dir / experiment_run.experiment.name,
        )

    return {
        "suite_json": suite_path,
        "manifest_csv": manifest_path,
        "records_csv": records_path,
        "summary_csv": summary_path,
        "experiments_dir": experiment_dir,
    }


def save_scenario_preset(scenario: ScenarioPreset, root: str | Path, filename: str | None = None) -> Path:
    return save_scenario(scenario, root=root, filename=filename)


def load_scenario_preset(path: str | Path) -> ScenarioPreset:
    return load_scenario(path)


def save_suite_config(suite: ExperimentSuite, root: str | Path, filename: str | None = None) -> Path:
    return save_suite(suite, root=root, filename=filename)


def load_suite_config(path: str | Path) -> ExperimentSuite:
    return load_suite(path)


__all__ = [
    "ExperimentSuite",
    "ExperimentSuiteRun",
    "OneFactorExperiment",
    "OneFactorExperimentRun",
    "ParamBinding",
    "ResearchMethod",
    "ScenarioPreset",
    "binding_from_dict",
    "binding_to_dict",
    "clone_scenario",
    "default_missingness_mode_values",
    "default_noise_values",
    "default_observed_fraction_values",
    "default_rank_values",
    "ensure_research_tree",
    "estimate_signal_rms",
    "experiment_from_dict",
    "experiment_to_dict",
    "generate_experiment_scenarios",
    "load_experiment",
    "load_scenario",
    "load_scenario_preset",
    "load_suite",
    "load_suite_config",
    "make_binding",
    "make_experiment",
    "make_method",
    "make_scenario",
    "make_suite",
    "materialize_methods",
    "method_from_dict",
    "method_to_dict",
    "preview_experiment",
    "preview_suite",
    "resolve_experiment_base_preset",
    "resolve_experiment_base_scenario",
    "resolve_experiment_values",
    "resolve_preset_path",
    "resolve_suite_experiments",
    "run_experiment",
    "run_suite",
    "save_experiment",
    "save_experiment_run",
    "save_scenario",
    "save_scenario_preset",
    "save_suite",
    "save_suite_config",
    "save_suite_run",
    "scenario_config_from_dict",
    "scenario_config_to_dict",
    "scenario_preset_from_dict",
    "scenario_preset_to_dict",
    "suite_from_dict",
    "suite_to_dict",
    "sweep_scenarios",
]
