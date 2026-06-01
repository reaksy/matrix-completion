#!/usr/bin/env python3
"""Research-oriented config layer for synthetic experiments.

This module does not replace `synthetic_api.py`. It sits on top of it and adds:
1. serializable study configs;
2. serializable method specs;
3. one or more sweep axes;
4. save/load helpers for configs;
5. a study runner that expands configs and aggregates results.

The stable low-level workflow in `synthetic_api.py` remains untouched.
"""

from __future__ import annotations

import copy
import itertools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import synthetic_api as base


@dataclass
class MethodSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    label: str | None = None


@dataclass
class SweepAxis:
    parameter_path: str
    values: list[Any]
    alias: str | None = None


@dataclass
class StudyConfig:
    name: str
    description: str = ""
    base_scenario: base.ScenarioConfig = field(default_factory=base.ScenarioConfig)
    methods: list[MethodSpec] = field(default_factory=list)
    sweeps: list[SweepAxis] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [42])
    output_dir: str = "study_outputs"
    aggregate_by: list[str] = field(default_factory=list)


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


def study_config_to_dict(config: StudyConfig) -> dict[str, Any]:
    return {
        "name": config.name,
        "description": config.description,
        "base_scenario": scenario_config_to_dict(config.base_scenario),
        "methods": [asdict(method) for method in config.methods],
        "sweeps": [asdict(axis) for axis in config.sweeps],
        "seeds": list(config.seeds),
        "output_dir": config.output_dir,
        "aggregate_by": list(config.aggregate_by),
    }


def study_config_from_dict(payload: dict[str, Any]) -> StudyConfig:
    return StudyConfig(
        name=payload["name"],
        description=payload.get("description", ""),
        base_scenario=scenario_config_from_dict(payload.get("base_scenario", {})),
        methods=[MethodSpec(**item) for item in payload.get("methods", [])],
        sweeps=[SweepAxis(**item) for item in payload.get("sweeps", [])],
        seeds=[int(seed) for seed in payload.get("seeds", [42])],
        output_dir=str(payload.get("output_dir", "study_outputs")),
        aggregate_by=list(payload.get("aggregate_by", [])),
    )


def save_study_config(config: StudyConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(study_config_to_dict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_study_config(path: str | Path) -> StudyConfig:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return study_config_from_dict(payload)


def materialize_methods(method_specs: Sequence[MethodSpec]):
    return [
        base.make_method(method.name, label=method.label, **dict(method.params))
        for method in method_specs
    ]


def default_groupby_for_study(config: StudyConfig) -> list[str]:
    if config.aggregate_by:
        return list(config.aggregate_by)
    group_by = [axis.parameter_path for axis in config.sweeps]
    group_by.append("method")
    return group_by


def expand_study_scenarios(
    config: StudyConfig,
) -> list[tuple[base.ScenarioConfig, dict[str, Any]]]:
    if not config.sweeps:
        return [(copy.deepcopy(config.base_scenario), {})]

    scenarios: list[tuple[base.ScenarioConfig, dict[str, Any]]] = []
    axes = list(config.sweeps)
    for combo in itertools.product(*(axis.values for axis in axes)):
        scenario = copy.deepcopy(config.base_scenario)
        metadata: dict[str, Any] = {}
        for axis, value in zip(axes, combo, strict=True):
            scenario = base.apply_overrides(scenario, {axis.parameter_path: value})
            metadata[axis.alias or axis.parameter_path] = value
        scenarios.append((scenario, metadata))
    return scenarios


def preview_study_grid(config: StudyConfig) -> list[dict[str, Any]]:
    preview_rows: list[dict[str, Any]] = []
    for index, (scenario, metadata) in enumerate(expand_study_scenarios(config), start=1):
        row = {
            "scenario_index": index,
            "study_name": config.name,
            **metadata,
        }
        row.update(base.flatten_config(scenario))
        preview_rows.append(row)
    return preview_rows


def run_study(config: StudyConfig) -> dict[str, Any]:
    method_configs = materialize_methods(config.methods)
    scenario_specs = expand_study_scenarios(config)

    runs: list[base.ExperimentResult] = []
    records: list[dict[str, Any]] = []
    scenario_manifest: list[dict[str, Any]] = []

    for scenario_index, (scenario_config, metadata) in enumerate(scenario_specs, start=1):
        scenario_manifest.append(
            {
                "scenario_index": scenario_index,
                "study_name": config.name,
                **metadata,
                **base.flatten_config(scenario_config),
            }
        )
        for seed in config.seeds:
            run = base.run_single_experiment(scenario_config, method_configs, seed=seed)
            runs.append(run)
            for row in run.records:
                augmented = dict(row)
                augmented["study_name"] = config.name
                augmented["study_description"] = config.description
                augmented["scenario_index"] = scenario_index
                for key, value in metadata.items():
                    augmented[key] = value
                records.append(augmented)

    summary = base.aggregate_records(records, by=default_groupby_for_study(config))
    return {
        "study_config": config,
        "method_configs": method_configs,
        "scenario_manifest": scenario_manifest,
        "runs": runs,
        "records": records,
        "summary": summary,
    }


def save_study_outputs(
    study_result: dict[str, Any],
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    config: StudyConfig = study_result["study_config"]
    output_root = Path(output_dir or config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    config_path = output_root / f"{config.name}_config.json"
    manifest_path = output_root / f"{config.name}_manifest.csv"
    records_path = output_root / f"{config.name}_records.csv"
    summary_path = output_root / f"{config.name}_summary.csv"

    save_study_config(config, config_path)
    base.write_csv(study_result["scenario_manifest"], manifest_path)
    base.write_csv(study_result["records"], records_path)
    base.write_csv(study_result["summary"], summary_path)

    return {
        "config": config_path,
        "manifest": manifest_path,
        "records": records_path,
        "summary": summary_path,
    }


def plot_study_metric(
    study_result: dict[str, Any],
    *,
    x: str,
    y: str,
    hue: str = "method",
    title: str = "",
    xlabel: str | None = None,
    ylabel: str | None = None,
    save_path: str | Path | None = None,
):
    return base.plot_metric(
        study_result["summary"],
        x=x,
        y=y,
        hue=hue,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        save_path=save_path,
    )


__all__ = [
    "MethodSpec",
    "SweepAxis",
    "StudyConfig",
    "default_groupby_for_study",
    "expand_study_scenarios",
    "load_study_config",
    "materialize_methods",
    "plot_study_metric",
    "preview_study_grid",
    "run_study",
    "save_study_config",
    "save_study_outputs",
    "scenario_config_from_dict",
    "scenario_config_to_dict",
    "study_config_from_dict",
    "study_config_to_dict",
]
