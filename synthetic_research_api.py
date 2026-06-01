#!/usr/bin/env python3
"""Research-first config layer for synthetic matrix completion experiments.

This module is intentionally separate from the existing working workflow.
It builds on top of `synthetic_api.py`, but the main object here is a single
research config that can be saved to JSON, loaded back, and executed.

Design goals:
1. one JSON file = one research study;
2. the study may either embed a base scenario or reference a reusable preset;
3. methods can bind parameters to scenario fields without editing code;
4. sweeps can target both scenario parameters and method parameters;
5. the runner returns raw records + summary tables and can save them.
"""

from __future__ import annotations

import copy
import itertools
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
class SweepAxis:
    target: str = "scenario"  # scenario or method
    path: str = ""
    values: list[Any] = field(default_factory=list)
    alias: str | None = None
    method_key: str | None = None


@dataclass
class ResearchConfig:
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    scenarios: list[ScenarioPreset] = field(default_factory=list)
    base_scenario: base.ScenarioConfig | None = None
    base_scenario_ref: str | None = None
    methods: list[ResearchMethod] = field(default_factory=list)
    sweeps: list[SweepAxis] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [42])
    output_dir: str = ""
    aggregate_by: list[str] = field(default_factory=list)


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

    def to_research_config(self, root: str | Path) -> ResearchConfig:
        scenario_config = resolve_experiment_base_scenario(self, root)
        return ResearchConfig(
            name=self.name,
            description=self.description,
            tags=list(self.tags),
            base_scenario=scenario_config,
            methods=copy.deepcopy(self.methods),
            sweeps=[
                SweepAxis(
                    target="scenario",
                    path=self.vary_path,
                    values=self.resolved_values(root),
                    alias=self.vary_alias,
                )
            ],
            seeds=list(self.seeds),
            output_dir=self.output_dir,
            aggregate_by=list(self.aggregate_by),
        )

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


def resolve_experiment_base_preset(
    experiment: OneFactorExperiment,
    root: str | Path,
) -> ScenarioPreset:
    if experiment.base_scenario is not None:
        return copy.deepcopy(experiment.base_scenario)
    if experiment.base_scenario_ref:
        return load_scenario_preset(resolve_preset_path(experiment.base_scenario_ref, root))
    raise ValueError("OneFactorExperiment must define either base_scenario or base_scenario_ref.")


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
    r_max: int | None = None,
) -> list[int]:
    d = min(scenario_config.matrix.m, scenario_config.matrix.n)
    limit = d if r_max is None else max(1, min(int(r_max), d))
    dense_limit = min(limit, max(1, dense_cutoff))
    values = list(range(1, dense_limit + 1))
    current = values[-1]
    while current < limit:
        nxt = max(current + 1, int(math.ceil(current * growth)))
        nxt = min(nxt, limit)
        if nxt == values[-1]:
            break
        values.append(int(nxt))
        current = int(nxt)
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
            "Experiment has no explicit values and is not yet attached to a base scenario. "
            "Either pass values=[...], set base_scenario/base_scenario_ref, or resolve it through "
            "suite.resolved_experiments(...)."
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
        f"No default value grid is defined for '{experiment.vary_path}'. "
        "Pass explicit values=[...] for this experiment."
    )


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
        alias=experiment.vary_alias,
        extra_tags=experiment.tags,
    )


def _experiment_axis_alias(experiment: OneFactorExperiment) -> str:
    return experiment.vary_alias or experiment.vary_path


def _experiment_scenario_name(experiment: OneFactorExperiment, row: dict[str, Any], root: str | Path) -> str:
    alias = _experiment_axis_alias(experiment)
    value = row.get(alias)
    base_name = resolve_experiment_base_preset(experiment, root).name
    return f"{base_name}__{alias}_{value}"


def _augment_experiment_rows(
    rows: list[dict[str, Any]],
    experiment: OneFactorExperiment,
    root: str | Path,
) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for row in rows:
        updated = dict(row)
        updated["scenario_name"] = _experiment_scenario_name(experiment, updated, root)
        updated["experiment_name"] = experiment.name
        updated["experiment_description"] = experiment.description
        augmented.append(updated)
    return augmented


def preview_experiment(
    experiment: OneFactorExperiment,
    root: str | Path,
) -> list[dict[str, Any]]:
    rows = preview_research_grid(experiment.to_research_config(root), root)
    return _augment_experiment_rows(rows, experiment, root)


def run_experiment(
    experiment: OneFactorExperiment,
    root: str | Path,
) -> OneFactorExperimentRun:
    root_path = _root_path(root)
    research_run = run_research(experiment.to_research_config(root_path), root=root_path)
    return OneFactorExperimentRun(
        experiment=copy.deepcopy(experiment),
        root=root_path,
        manifest=_augment_experiment_rows(research_run["manifest"], experiment, root_path),
        records=_augment_experiment_rows(research_run["records"], experiment, root_path),
        summary=_augment_experiment_rows(research_run["summary"], experiment, root_path),
        runs=research_run["runs"],
    )


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
                    "experiment_name": experiment.name,
                    "experiment_description": experiment.description,
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
                    "experiment_name": experiment.name,
                    "experiment_description": experiment.description,
                    **row,
                }
            )
        for row in experiment_run.records:
            records.append(
                {
                    "suite_name": suite.name,
                    "suite_description": suite.description,
                    "experiment_name": experiment.name,
                    "experiment_description": experiment.description,
                    **row,
                }
            )
        for row in experiment_run.summary:
            summary.append(
                {
                    "suite_name": suite.name,
                    "suite_description": suite.description,
                    "experiment_name": experiment.name,
                    "experiment_description": experiment.description,
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


def _root_path(root: str | Path) -> Path:
    return Path(root).expanduser().resolve()


def ensure_research_tree(root: str | Path) -> dict[str, Path]:
    root_path = _root_path(root)
    config_dir = root_path / "research_configs"
    preset_dir = config_dir / "presets" / "scenarios"
    study_dir = config_dir / "studies"
    experiment_dir = config_dir / "experiments"
    suite_dir = config_dir / "suites"
    output_dir = root_path / "research_outputs"
    for path in (config_dir, preset_dir, study_dir, experiment_dir, suite_dir, output_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "root": root_path,
        "configs": config_dir,
        "presets": preset_dir,
        "studies": study_dir,
        "experiments": experiment_dir,
        "suites": suite_dir,
        "outputs": output_dir,
    }


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


def make_study(
    name: str,
    *,
    description: str = "",
    tags: Sequence[str] | None = None,
    scenarios: Sequence[ScenarioPreset] | None = None,
    base_scenario: base.ScenarioConfig | None = None,
    base_scenario_ref: str | None = None,
    methods: Sequence[ResearchMethod] | None = None,
    sweeps: Sequence[SweepAxis] | None = None,
    seeds: Sequence[int] | None = None,
    output_dir: str = "",
    aggregate_by: Sequence[str] | None = None,
) -> ResearchConfig:
    return ResearchConfig(
        name=name,
        description=description,
        tags=list(tags or []),
        scenarios=list(copy.deepcopy(list(scenarios or []))),
        base_scenario=copy.deepcopy(base_scenario),
        base_scenario_ref=base_scenario_ref,
        methods=list(copy.deepcopy(list(methods or []))),
        sweeps=list(copy.deepcopy(list(sweeps or []))),
        seeds=[int(seed) for seed in (seeds or [42])],
        output_dir=output_dir,
        aggregate_by=list(aggregate_by or []),
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


def sweep_to_dict(sweep: SweepAxis) -> dict[str, Any]:
    return asdict(sweep)


def sweep_from_dict(payload: dict[str, Any]) -> SweepAxis:
    return SweepAxis(**payload)


def research_config_to_dict(config: ResearchConfig) -> dict[str, Any]:
    return {
        "name": config.name,
        "description": config.description,
        "tags": list(config.tags),
        "scenarios": [scenario_preset_to_dict(item) for item in config.scenarios],
        "base_scenario": None if config.base_scenario is None else scenario_config_to_dict(config.base_scenario),
        "base_scenario_ref": config.base_scenario_ref,
        "methods": [method_to_dict(method) for method in config.methods],
        "sweeps": [sweep_to_dict(sweep) for sweep in config.sweeps],
        "seeds": list(config.seeds),
        "output_dir": config.output_dir,
        "aggregate_by": list(config.aggregate_by),
    }


def research_config_from_dict(payload: dict[str, Any]) -> ResearchConfig:
    base_scenario_payload = payload.get("base_scenario")
    return ResearchConfig(
        name=payload["name"],
        description=payload.get("description", ""),
        tags=list(payload.get("tags", [])),
        scenarios=[scenario_preset_from_dict(item) for item in payload.get("scenarios", [])],
        base_scenario=None if base_scenario_payload in (None, {}) else scenario_config_from_dict(base_scenario_payload),
        base_scenario_ref=payload.get("base_scenario_ref"),
        methods=[method_from_dict(item) for item in payload.get("methods", [])],
        sweeps=[sweep_from_dict(item) for item in payload.get("sweeps", [])],
        seeds=[int(seed) for seed in payload.get("seeds", [42])],
        output_dir=str(payload.get("output_dir", "")),
        aggregate_by=list(payload.get("aggregate_by", [])),
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


def save_scenario_preset(
    preset: ScenarioPreset,
    root: str | Path,
    filename: str | None = None,
) -> Path:
    tree = ensure_research_tree(root)
    path = tree["presets"] / (filename or f"{preset.name}.json")
    path.write_text(
        json.dumps(scenario_preset_to_dict(preset), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_scenario_preset(path: str | Path) -> ScenarioPreset:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return scenario_preset_from_dict(payload)


def save_research_config(
    config: ResearchConfig,
    root: str | Path,
    filename: str | None = None,
) -> Path:
    tree = ensure_research_tree(root)
    path = tree["studies"] / (filename or f"{config.name}.json")
    path.write_text(
        json.dumps(research_config_to_dict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_research_config(path: str | Path) -> ResearchConfig:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return research_config_from_dict(payload)


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


def method_key(method: ResearchMethod) -> str:
    return method.key or method.label or method.name


def resolve_preset_path(reference: str | Path, root: str | Path) -> Path:
    ref = Path(reference)
    if ref.is_absolute():
        return ref
    return _root_path(root) / ref


def resolve_base_scenario(config: ResearchConfig, root: str | Path) -> base.ScenarioConfig:
    if config.base_scenario is not None:
        return copy.deepcopy(config.base_scenario)
    if config.base_scenario_ref:
        preset = load_scenario_preset(resolve_preset_path(config.base_scenario_ref, root))
        return copy.deepcopy(preset.scenario)
    raise ValueError("ResearchConfig must define either base_scenario or base_scenario_ref.")


def resolve_scenarios(config: ResearchConfig, root: str | Path) -> list[ScenarioPreset]:
    if config.scenarios:
        return [copy.deepcopy(item) for item in config.scenarios]
    return [
        ScenarioPreset(
            name="base",
            description="Base scenario",
            tags=["base"],
            scenario=resolve_base_scenario(config, root),
        )
    ]


def _get_flat_value(scenario_config: base.ScenarioConfig, path: str) -> Any:
    flat = base.flatten_config(scenario_config)
    if path not in flat:
        raise KeyError(f"Unknown scenario path in binding: {path}")
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
    else:
        if binding.cast == "str":
            value = str(value)
    return value


def normalize_method_param_path(path: str) -> str:
    return path.removeprefix("params.")


def materialize_methods(
    config: ResearchConfig,
    scenario_config: base.ScenarioConfig,
    method_overrides: dict[str, dict[str, Any]] | None = None,
) -> list[base.MethodConfig]:
    method_overrides = method_overrides or {}
    output: list[base.MethodConfig] = []
    for method in config.methods:
        params = dict(method.params)
        overrides = method_overrides.get(method_key(method), {})
        for name, value in overrides.items():
            params[normalize_method_param_path(name)] = value
        for param_name, binding in method.bindings.items():
            params[param_name] = resolve_binding(binding, scenario_config)
        output.append(base.make_method(method.name, label=method.label, **params))
    return output


def default_groupby(config: ResearchConfig) -> list[str]:
    if config.aggregate_by:
        return list(config.aggregate_by)
    keys = []
    if config.scenarios:
        keys.append("scenario_name")
    keys.extend(sweep.alias or _default_axis_alias(sweep) for sweep in config.sweeps)
    keys.append("method")
    return keys


def _default_axis_alias(sweep: SweepAxis) -> str:
    if sweep.target == "scenario":
        return sweep.path
    method_part = sweep.method_key or "method"
    return f"{method_part}.{normalize_method_param_path(sweep.path)}"


def expand_research_grid(
    config: ResearchConfig,
    root: str | Path,
) -> list[tuple[base.ScenarioConfig, dict[str, dict[str, Any]], dict[str, Any]]]:
    output: list[tuple[base.ScenarioConfig, dict[str, dict[str, Any]], dict[str, Any]]] = []
    scenario_specs = resolve_scenarios(config, root)
    axes = list(config.sweeps)

    if not axes:
        for scenario_spec in scenario_specs:
            metadata = {
                "scenario_name": scenario_spec.name,
                "scenario_description": scenario_spec.description,
            }
            if scenario_spec.tags:
                metadata["scenario_tags"] = ",".join(scenario_spec.tags)
            output.append((copy.deepcopy(scenario_spec.scenario), {}, metadata))
        return output

    for scenario_spec in scenario_specs:
        for combo in itertools.product(*(axis.values for axis in axes)):
            scenario_config = copy.deepcopy(scenario_spec.scenario)
            method_overrides: dict[str, dict[str, Any]] = {}
            metadata: dict[str, Any] = {
                "scenario_name": scenario_spec.name,
                "scenario_description": scenario_spec.description,
            }
            if scenario_spec.tags:
                metadata["scenario_tags"] = ",".join(scenario_spec.tags)
            for axis, value in zip(axes, combo, strict=True):
                metadata[axis.alias or _default_axis_alias(axis)] = value
                if axis.target == "scenario":
                    scenario_config = base.apply_overrides(scenario_config, {axis.path: value})
                elif axis.target == "method":
                    if not axis.method_key:
                        raise ValueError("Method sweep axis requires method_key.")
                    method_overrides.setdefault(axis.method_key, {})[normalize_method_param_path(axis.path)] = value
                else:
                    raise ValueError("Sweep target must be either 'scenario' or 'method'.")
            output.append((scenario_config, method_overrides, metadata))
    return output


def preview_research_grid(
    config: ResearchConfig,
    root: str | Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_index, (scenario_config, method_overrides, metadata) in enumerate(expand_research_grid(config, root), start=1):
        row = {
            "research_name": config.name,
            "scenario_index": scenario_index,
            **metadata,
            **base.flatten_config(scenario_config),
        }
        if method_overrides:
            row["method_overrides"] = json.dumps(method_overrides, ensure_ascii=False, sort_keys=True)
        rows.append(row)
    return rows


def run_research(
    config: ResearchConfig,
    root: str | Path,
) -> dict[str, Any]:
    expanded = expand_research_grid(config, root)
    runs: list[base.ExperimentResult] = []
    records: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for scenario_index, (scenario_config, method_overrides, metadata) in enumerate(expanded, start=1):
        manifest_row = {
            "research_name": config.name,
            "scenario_index": scenario_index,
            **metadata,
            **base.flatten_config(scenario_config),
        }
        if method_overrides:
            manifest_row["method_overrides"] = json.dumps(method_overrides, ensure_ascii=False, sort_keys=True)
        manifest.append(manifest_row)

        methods = materialize_methods(config, scenario_config, method_overrides=method_overrides)
        for seed in config.seeds:
            run = base.run_single_experiment(scenario_config, methods, seed=seed)
            runs.append(run)
            for row in run.records:
                augmented = dict(row)
                augmented["research_name"] = config.name
                augmented["research_description"] = config.description
                augmented["scenario_index"] = scenario_index
                for key, value in metadata.items():
                    augmented[key] = value
                if method_overrides:
                    augmented["method_overrides"] = json.dumps(method_overrides, ensure_ascii=False, sort_keys=True)
                records.append(augmented)

    summary = base.aggregate_records(records, by=default_groupby(config))
    return {
        "research_config": config,
        "manifest": manifest,
        "records": records,
        "summary": summary,
        "runs": runs,
    }


def run_research_from_file(
    config_path: str | Path,
    root: str | Path,
) -> dict[str, Any]:
    config = load_research_config(config_path)
    return run_research(config, root=root)


def resolve_output_dir(
    config: ResearchConfig,
    root: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    tree = ensure_research_tree(root)
    if output_dir is not None:
        candidate = Path(output_dir)
        return candidate if candidate.is_absolute() else tree["root"] / candidate
    if config.output_dir:
        candidate = Path(config.output_dir)
        return candidate if candidate.is_absolute() else tree["root"] / candidate
    return tree["outputs"] / config.name


def save_research_run(
    research_run: dict[str, Any],
    root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    config: ResearchConfig = research_run["research_config"]
    final_output_dir = resolve_output_dir(config, root, output_dir=output_dir)
    final_output_dir.mkdir(parents=True, exist_ok=True)

    config_path = final_output_dir / f"{config.name}_research.json"
    manifest_path = final_output_dir / f"{config.name}_manifest.csv"
    records_path = final_output_dir / f"{config.name}_records.csv"
    summary_path = final_output_dir / f"{config.name}_summary.csv"

    config_path.write_text(
        json.dumps(research_config_to_dict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    base.write_csv(research_run["manifest"], manifest_path)
    base.write_csv(research_run["records"], records_path)
    base.write_csv(research_run["summary"], summary_path)

    return {
        "research_json": config_path,
        "manifest_csv": manifest_path,
        "records_csv": records_path,
        "summary_csv": summary_path,
    }


def save_experiment_run(
    experiment_run: OneFactorExperimentRun,
    root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    research_config = experiment_run.experiment.to_research_config(root)
    research_paths = save_research_run(
        {
            "research_config": research_config,
            "manifest": experiment_run.manifest,
            "records": experiment_run.records,
            "summary": experiment_run.summary,
            "runs": experiment_run.runs,
        },
        root=root,
        output_dir=output_dir,
    )
    final_output_dir = resolve_output_dir(research_config, root, output_dir=output_dir)
    experiment_path = final_output_dir / f"{experiment_run.experiment.name}_experiment.json"
    experiment_path.write_text(
        json.dumps(experiment_to_dict(experiment_run.experiment), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "experiment_json": experiment_path,
        **research_paths,
    }


def save_suite_run(
    suite_run: ExperimentSuiteRun,
    root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    tree = ensure_research_tree(root)
    if output_dir is not None:
        final_output_dir = Path(output_dir)
        if not final_output_dir.is_absolute():
            final_output_dir = tree["root"] / final_output_dir
    elif suite_run.suite.output_dir:
        final_output_dir = Path(suite_run.suite.output_dir)
        if not final_output_dir.is_absolute():
            final_output_dir = tree["root"] / final_output_dir
    else:
        final_output_dir = tree["outputs"] / suite_run.suite.name

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


def create_example_research_bundle(root: str | Path) -> dict[str, Path]:
    tree = ensure_research_tree(root)

    preset = ScenarioPreset(
        name="base_60x60_rank4",
        description="Базовый шаблон 60x60 с rank=4, случайными пропусками и слабым шумом.",
        tags=["synthetic", "preset", "rank4"],
        scenario=base.ScenarioConfig(
            matrix=base.MatrixConfig(m=60, n=60, rank=4),
            structure=base.StructureConfig(mode="none"),
            missingness_field=base.MissingnessFieldConfig(mode="random"),
            mask_sampling=base.MaskSamplingConfig(observed_fraction=0.35, exact_fraction=True),
            split=base.SplitConfig(validation_fraction=0.15, test_fraction=0.15),
            noise=base.NoiseConfig(mode="gaussian", std=0.02),
        ),
    )
    preset_path = save_scenario_preset(preset, tree["root"])

    research = ResearchConfig(
        name="rank_sweep_research",
        description="Перебор истинного ранга с привязкой ранга методов к рангу сценария.",
        tags=["synthetic", "rank_sweep", "research"],
        base_scenario_ref=str(preset_path.relative_to(tree["root"])),
        methods=[
            ResearchMethod(
                name="soft_impute",
                label="Soft-Impute",
                key="soft",
                params={"max_iter": 80},
                bindings={"rank": ParamBinding(path="matrix.rank", cast="int")},
            ),
            ResearchMethod(
                name="rgd",
                label="RGD",
                key="rgd",
                params={"max_iter": 120, "init": "spectral"},
                bindings={"rank": ParamBinding(path="matrix.rank", cast="int")},
            ),
            ResearchMethod(
                name="rgd_l2",
                label="RGD + L2",
                key="rgd_l2",
                params={"max_iter": 120, "init": "spectral", "l2_reg": 0.03},
                bindings={"rank": ParamBinding(path="matrix.rank", cast="int")},
            ),
        ],
        sweeps=[
            SweepAxis(target="scenario", path="matrix.rank", values=[4, 6, 8, 10]),
        ],
        seeds=[41, 42],
        output_dir="research_outputs/rank_sweep_research",
    )
    research_path = save_research_config(research, tree["root"])
    return {
        "preset": preset_path,
        "research": research_path,
    }


def save_scenario(scenario: ScenarioPreset, root: str | Path, filename: str | None = None) -> Path:
    return save_scenario_preset(scenario, root=root, filename=filename)


def load_scenario(path: str | Path) -> ScenarioPreset:
    return load_scenario_preset(path)


def save_research(config: ResearchConfig, root: str | Path, filename: str | None = None) -> Path:
    return save_research_config(config, root=root, filename=filename)


def load_research(path: str | Path) -> ResearchConfig:
    return load_research_config(path)


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
    "ResearchConfig",
    "ResearchMethod",
    "ScenarioPreset",
    "SweepAxis",
    "create_example_research_bundle",
    "clone_scenario",
    "default_groupby",
    "default_missingness_mode_values",
    "default_noise_values",
    "default_observed_fraction_values",
    "default_rank_values",
    "ensure_research_tree",
    "estimate_signal_rms",
    "experiment_from_dict",
    "experiment_to_dict",
    "expand_research_grid",
    "generate_experiment_scenarios",
    "load_experiment",
    "load_research",
    "load_research_config",
    "load_scenario",
    "load_scenario_preset",
    "load_suite_config",
    "load_suite",
    "make_binding",
    "make_experiment",
    "make_method",
    "make_scenario",
    "make_study",
    "make_suite",
    "preview_experiment",
    "preview_research_grid",
    "preview_suite",
    "resolve_base_scenario",
    "resolve_experiment_base_preset",
    "resolve_experiment_base_scenario",
    "resolve_experiment_values",
    "resolve_output_dir",
    "resolve_preset_path",
    "resolve_scenarios",
    "resolve_suite_experiments",
    "research_config_from_dict",
    "research_config_to_dict",
    "run_experiment",
    "run_research",
    "run_research_from_file",
    "run_suite",
    "save_experiment",
    "save_experiment_run",
    "save_research",
    "save_research_config",
    "save_research_run",
    "save_scenario",
    "save_scenario_preset",
    "save_suite_config",
    "save_suite",
    "save_suite_run",
    "scenario_preset_from_dict",
    "scenario_preset_to_dict",
    "suite_from_dict",
    "suite_to_dict",
    "sweep_scenarios",
]
