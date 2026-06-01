#!/usr/bin/env python3
"""Config-first workflow for synthetic matrix completion research.

This module is intentionally separate from the existing working files.
It does not replace `synthetic_api.py` or `synthetic_study_api.py`.

Main idea:
1. one JSON file = one scenario config;
2. one JSON file = one research study config;
3. studies reference scenarios by path;
4. experiments can be reproduced by loading JSON files rather than rebuilding
   everything manually in a notebook.
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
class ScenarioDocument:
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    scenario: base.ScenarioConfig = field(default_factory=base.ScenarioConfig)


@dataclass
class MethodDocument:
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    label: str | None = None


@dataclass
class SweepDocument:
    parameter_path: str
    values: list[Any]
    alias: str | None = None


@dataclass
class StudyDocument:
    name: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    base_scenario_path: str = ""
    methods: list[MethodDocument] = field(default_factory=list)
    sweeps: list[SweepDocument] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [42])
    output_dir: str = ""
    aggregate_by: list[str] = field(default_factory=list)


def _root_path(root: str | Path) -> Path:
    return Path(root).expanduser().resolve()


def ensure_study_tree(root: str | Path) -> dict[str, Path]:
    root_path = _root_path(root)
    configs_dir = root_path / "study_configs"
    scenarios_dir = configs_dir / "scenarios"
    studies_dir = configs_dir / "studies"
    outputs_dir = root_path / "study_outputs"
    for path in (configs_dir, scenarios_dir, studies_dir, outputs_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "root": root_path,
        "configs": configs_dir,
        "scenarios": scenarios_dir,
        "studies": studies_dir,
        "outputs": outputs_dir,
    }


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


def scenario_document_to_dict(document: ScenarioDocument) -> dict[str, Any]:
    return {
        "name": document.name,
        "description": document.description,
        "tags": list(document.tags),
        "scenario": scenario_config_to_dict(document.scenario),
    }


def scenario_document_from_dict(payload: dict[str, Any]) -> ScenarioDocument:
    return ScenarioDocument(
        name=payload["name"],
        description=payload.get("description", ""),
        tags=list(payload.get("tags", [])),
        scenario=scenario_config_from_dict(payload.get("scenario", {})),
    )


def study_document_to_dict(document: StudyDocument) -> dict[str, Any]:
    return {
        "name": document.name,
        "description": document.description,
        "tags": list(document.tags),
        "base_scenario_path": document.base_scenario_path,
        "methods": [asdict(method) for method in document.methods],
        "sweeps": [asdict(sweep) for sweep in document.sweeps],
        "seeds": list(document.seeds),
        "output_dir": document.output_dir,
        "aggregate_by": list(document.aggregate_by),
    }


def study_document_from_dict(payload: dict[str, Any]) -> StudyDocument:
    return StudyDocument(
        name=payload["name"],
        description=payload.get("description", ""),
        tags=list(payload.get("tags", [])),
        base_scenario_path=str(payload.get("base_scenario_path", "")),
        methods=[MethodDocument(**method) for method in payload.get("methods", [])],
        sweeps=[SweepDocument(**sweep) for sweep in payload.get("sweeps", [])],
        seeds=[int(seed) for seed in payload.get("seeds", [42])],
        output_dir=str(payload.get("output_dir", "")),
        aggregate_by=list(payload.get("aggregate_by", [])),
    )


def save_scenario_document(
    document: ScenarioDocument,
    root: str | Path,
    filename: str | None = None,
) -> Path:
    tree = ensure_study_tree(root)
    path = tree["scenarios"] / (filename or f"{document.name}.json")
    path.write_text(
        json.dumps(scenario_document_to_dict(document), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_scenario_document(path: str | Path) -> ScenarioDocument:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return scenario_document_from_dict(payload)


def save_study_document(
    document: StudyDocument,
    root: str | Path,
    filename: str | None = None,
) -> Path:
    tree = ensure_study_tree(root)
    path = tree["studies"] / (filename or f"{document.name}.json")
    path.write_text(
        json.dumps(study_document_to_dict(document), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_study_document(path: str | Path) -> StudyDocument:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return study_document_from_dict(payload)


def resolve_scenario_path(study_document: StudyDocument, root: str | Path) -> Path:
    root_path = _root_path(root)
    raw = Path(study_document.base_scenario_path)
    if raw.is_absolute():
        return raw
    return root_path / raw


def load_base_scenario_for_study(study_document: StudyDocument, root: str | Path) -> ScenarioDocument:
    return load_scenario_document(resolve_scenario_path(study_document, root))


def materialize_methods(methods: Sequence[MethodDocument]) -> list[base.MethodConfig]:
    return [
        base.make_method(method.name, label=method.label, **dict(method.params))
        for method in methods
    ]


def default_groupby(study_document: StudyDocument) -> list[str]:
    if study_document.aggregate_by:
        return list(study_document.aggregate_by)
    keys = [sweep.alias or sweep.parameter_path for sweep in study_document.sweeps]
    keys.append("method")
    return keys


def expand_study(
    study_document: StudyDocument,
    root: str | Path,
) -> list[tuple[base.ScenarioConfig, dict[str, Any]]]:
    scenario_document = load_base_scenario_for_study(study_document, root)
    base_scenario = scenario_document.scenario
    if not study_document.sweeps:
        return [(copy.deepcopy(base_scenario), {})]

    expanded: list[tuple[base.ScenarioConfig, dict[str, Any]]] = []
    axes = list(study_document.sweeps)
    for combo in itertools.product(*(axis.values for axis in axes)):
        scenario = copy.deepcopy(base_scenario)
        metadata: dict[str, Any] = {}
        for axis, value in zip(axes, combo):
            scenario = base.apply_overrides(scenario, {axis.parameter_path: value})
            metadata[axis.alias or axis.parameter_path] = value
        expanded.append((scenario, metadata))
    return expanded


def preview_study_document(
    study_document: StudyDocument,
    root: str | Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario_index, (scenario, metadata) in enumerate(expand_study(study_document, root), start=1):
        row = {
            "study_name": study_document.name,
            "scenario_index": scenario_index,
            **metadata,
            **base.flatten_config(scenario),
        }
        rows.append(row)
    return rows


def run_study_document(
    study_document: StudyDocument,
    root: str | Path,
) -> dict[str, Any]:
    method_configs = materialize_methods(study_document.methods)
    scenario_rows = expand_study(study_document, root)

    manifest: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    runs: list[base.ExperimentResult] = []

    for scenario_index, (scenario_config, metadata) in enumerate(scenario_rows, start=1):
        manifest.append(
            {
                "study_name": study_document.name,
                "scenario_index": scenario_index,
                **metadata,
                **base.flatten_config(scenario_config),
            }
        )
        for seed in study_document.seeds:
            run = base.run_single_experiment(scenario_config, method_configs, seed=seed)
            runs.append(run)
            for row in run.records:
                augmented = dict(row)
                augmented["study_name"] = study_document.name
                augmented["study_description"] = study_document.description
                augmented["scenario_index"] = scenario_index
                for key, value in metadata.items():
                    augmented[key] = value
                records.append(augmented)

    summary = base.aggregate_records(records, by=default_groupby(study_document))
    return {
        "study_document": study_document,
        "method_configs": method_configs,
        "manifest": manifest,
        "records": records,
        "summary": summary,
        "runs": runs,
    }


def run_study_from_file(
    study_path: str | Path,
    root: str | Path,
) -> dict[str, Any]:
    document = load_study_document(study_path)
    return run_study_document(document, root=root)


def resolve_output_dir(
    study_document: StudyDocument,
    root: str | Path,
    output_dir: str | Path | None = None,
) -> Path:
    tree = ensure_study_tree(root)
    if output_dir is not None:
        output_path = Path(output_dir)
        if output_path.is_absolute():
            return output_path
        return tree["root"] / output_path

    if study_document.output_dir:
        configured = Path(study_document.output_dir)
        if configured.is_absolute():
            return configured
        return tree["root"] / configured

    return tree["outputs"] / study_document.name


def save_study_run(
    study_run: dict[str, Any],
    root: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Path]:
    study_document: StudyDocument = study_run["study_document"]
    final_output_dir = resolve_output_dir(study_document, root, output_dir=output_dir)
    final_output_dir.mkdir(parents=True, exist_ok=True)

    config_path = final_output_dir / f"{study_document.name}_study.json"
    manifest_path = final_output_dir / f"{study_document.name}_manifest.csv"
    records_path = final_output_dir / f"{study_document.name}_records.csv"
    summary_path = final_output_dir / f"{study_document.name}_summary.csv"

    config_path.write_text(
        json.dumps(study_document_to_dict(study_document), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    base.write_csv(study_run["manifest"], manifest_path)
    base.write_csv(study_run["records"], records_path)
    base.write_csv(study_run["summary"], summary_path)

    return {
        "study_json": config_path,
        "manifest_csv": manifest_path,
        "records_csv": records_path,
        "summary_csv": summary_path,
    }


def create_example_config_bundle(root: str | Path) -> dict[str, Path]:
    tree = ensure_study_tree(root)

    scenario_doc = ScenarioDocument(
        name="base_60x60_rank4",
        description="Базовый синтетический сценарий 60x60, rank=4, случайные пропуски.",
        tags=["synthetic", "base", "rank4"],
        scenario=base.ScenarioConfig(
            matrix=base.MatrixConfig(m=60, n=60, rank=4),
            structure=base.StructureConfig(mode="none"),
            missingness_field=base.MissingnessFieldConfig(mode="random"),
            mask_sampling=base.MaskSamplingConfig(observed_fraction=0.35, exact_fraction=True),
            split=base.SplitConfig(validation_fraction=0.15, test_fraction=0.15),
            noise=base.NoiseConfig(mode="gaussian", std=0.02),
        ),
    )
    scenario_path = save_scenario_document(scenario_doc, tree["root"])

    study_doc = StudyDocument(
        name="rank_sweep_demo",
        description="Перебор истинного ранга при фиксированных остальных параметрах.",
        tags=["synthetic", "rank_sweep"],
        base_scenario_path=str(scenario_path.relative_to(tree["root"])),
        methods=[
            MethodDocument(name="soft_impute", label="Soft-Impute", params={"rank": 4, "max_iter": 80}),
            MethodDocument(name="rgd", label="RGD", params={"rank": 4, "max_iter": 120, "init": "spectral"}),
            MethodDocument(name="rgd_l2", label="RGD + L2", params={"rank": 4, "max_iter": 120, "init": "spectral", "l2_reg": 0.03}),
        ],
        sweeps=[
            SweepDocument(parameter_path="matrix.rank", values=[4, 6, 8, 10]),
        ],
        seeds=[41, 42],
        output_dir="study_outputs/rank_sweep_demo",
    )
    study_path = save_study_document(study_doc, tree["root"])

    return {
        "scenario": scenario_path,
        "study": study_path,
    }


__all__ = [
    "MethodDocument",
    "ScenarioDocument",
    "StudyDocument",
    "SweepDocument",
    "create_example_config_bundle",
    "default_groupby",
    "ensure_study_tree",
    "expand_study",
    "load_base_scenario_for_study",
    "load_scenario_document",
    "load_study_document",
    "materialize_methods",
    "preview_study_document",
    "resolve_output_dir",
    "resolve_scenario_path",
    "run_study_document",
    "run_study_from_file",
    "save_scenario_document",
    "save_study_document",
    "save_study_run",
    "scenario_config_from_dict",
    "scenario_config_to_dict",
    "scenario_document_from_dict",
    "scenario_document_to_dict",
    "study_document_from_dict",
    "study_document_to_dict",
]
