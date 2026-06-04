import math

import numpy as np
import pytest

import synthetic_api as api
import synthetic_research_api as research


def small_config() -> api.ScenarioConfig:
    return api.ScenarioConfig(
        matrix=api.MatrixConfig(m=12, n=10, rank=2),
        mask_sampling=api.MaskSamplingConfig(observed_fraction=0.5),
        split=api.SplitConfig(validation_fraction=0.1, test_fraction=0.1),
        noise=api.NoiseConfig(std=0.01),
    )


def test_assemble_scenario_shapes_and_splits_are_valid() -> None:
    scenario = api.assemble_scenario(small_config(), seed=7)

    assert scenario.X_true.shape == (12, 10)
    assert scenario.Y_observed.shape == (12, 10)
    assert scenario.mask_split.observed_mask.shape == (12, 10)
    assert np.any(scenario.mask_split.fit_mask)
    assert np.any(scenario.mask_split.validation_mask)
    assert np.any(scenario.mask_split.test_mask)
    assert not np.any(scenario.mask_split.fit_mask & scenario.mask_split.test_mask)
    assert not np.any(scenario.mask_split.validation_mask & scenario.mask_split.test_mask)


def test_invalid_scenario_config_fails_early() -> None:
    config = small_config()
    config.mask_sampling.observed_fraction = 1.5

    with pytest.raises(ValueError, match="observed_fraction"):
        api.assemble_scenario(config, seed=7)


def test_compact_rgd_smoke_run_has_finite_metrics() -> None:
    method = api.make_method("compact_rgd", rank=2, max_iter=2)
    result = api.run_single_experiment(small_config(), [method], seed=11)

    assert len(result.records) == 1
    record = result.records[0]
    assert record["method"] == "compact_rgd"
    assert math.isfinite(float(record["test_rmse"]))
    assert record["iterations"] >= 1


def test_als_and_row_stripe_missingness_are_supported() -> None:
    config = small_config()
    config.missingness_field.mode = "row_stripe"
    method = api.make_method("als", rank=2, max_iter=2)

    result = api.run_single_experiment(config, [method], seed=13)

    assert len(result.records) == 1
    assert result.records[0]["method"] == "als"
    assert math.isfinite(float(result.records[0]["test_rmse"]))


def test_suite_resolves_default_rank_values_from_base_scenario() -> None:
    base = research.ScenarioPreset(name="small", config=small_config())
    sweep = research.make_sweep("rank_sweep", "matrix.rank", alias="rank")

    values = research.values_for_sweep(sweep, base)

    assert values == [1, 2, 3, 4, 5, 8, 10]


def test_default_rank_grid_is_capped_for_large_matrices() -> None:
    config = api.ScenarioConfig(matrix=api.MatrixConfig(m=1000, n=1000, rank=3))

    values = research.rank_grid(config)

    assert values[:5] == [1, 2, 3, 4, 5]
    assert values[-1] == 100
    assert len(values) < 20


def test_simple_suite_smoke_run_has_summary_rows() -> None:
    base = research.ScenarioPreset(name="small", config=small_config())
    methods = [research.make_method("compact_rgd", "Compact RGD", max_iter=2)]
    sweeps = [research.make_sweep("rank_sweep", "matrix.rank", values=[1, 2], alias="rank")]
    suite = research.make_suite("small_suite", base, methods, sweeps, seeds=[3])

    run = research.run_suite(suite)

    assert len(run.sweep_runs) == 1
    assert len(run.manifest) == 2
    assert len(run.records) == 2
    assert len(run.summary) == 2


def test_suite_run_can_be_saved_with_non_uniform_columns(tmp_path) -> None:
    base = research.ScenarioPreset(name="small", config=small_config())
    methods = [research.make_method("compact_rgd", "Compact RGD", max_iter=2)]
    sweeps = [
        research.make_sweep("rank_sweep", "matrix.rank", values=[1], alias="rank"),
        research.make_sweep("noise_sweep", "noise.std", values=[0.0], alias="noise"),
    ]
    suite = research.make_suite("small_suite", base, methods, sweeps, seeds=[3])

    run = research.run_suite(suite, root=tmp_path)
    paths = research.save_run(run)

    assert paths["manifest_csv"].exists()
    assert paths["records_csv"].exists()
    assert paths["summary_csv"].exists()
