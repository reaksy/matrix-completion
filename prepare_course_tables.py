#!/usr/bin/env python3
"""Prepare compact result tables for the course paper.

The benchmark script intentionally saves many detailed CSV files. This helper
turns them into a smaller set of publication-style CSV and LaTeX tables that
are easier to discuss in the course text.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


Row = dict[str, str]


METHOD_LABELS = {
    "als": "ALS",
    "soft_impute": "Soft-Impute",
    "riemannian_gd_l2_selected": r"RGD, full SVD, validation $\lambda$",
    "riemannian_gd_compact": r"Compact RGD",
    "riemannian_gd_compact_l2_1e-02": r"Compact RGD, $\lambda=10^{-2}$",
    "riemannian_gd_compact_l2_3e-02": r"Compact RGD, $\lambda=3\cdot 10^{-2}$",
    "riemannian_gd_compact_l2_selected": r"Compact RGD, validation $\lambda$",
}

RANK_LABELS = {
    "under": "underestimated",
    "correct": "correct",
    "over": "overestimated",
}


def read_csv(path: Path) -> list[Row]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def to_float(row: Row, key: str) -> float:
    return float(row[key])


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}\\%"


def index_by(rows: list[Row], key: str) -> dict[str, Row]:
    return {row[key]: row for row in rows}


def latex_table(
    rows: list[dict[str, Any]],
    columns: list[tuple[str, str]],
    caption: str,
    label: str,
    path: Path,
    column_spec: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    align = column_spec or ("l" + "r" * (len(columns) - 1))
    header = " & ".join(title for _, title in columns) + r" \\"
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(row[key]) for key, _ in columns) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def make_main_scalable_table(summary_by_method: list[Row], output_dir: Path) -> None:
    methods = [
        "als",
        "soft_impute",
        "riemannian_gd_compact",
        "riemannian_gd_compact_l2_1e-02",
        "riemannian_gd_compact_l2_3e-02",
        "riemannian_gd_compact_l2_selected",
    ]
    by_method = index_by(summary_by_method, "method")
    soft_rmse = to_float(by_method["soft_impute"], "avg_test_rmse")
    soft_time = to_float(by_method["soft_impute"], "avg_runtime_sec")
    rows: list[dict[str, Any]] = []
    for method in methods:
        item = by_method[method]
        rmse = to_float(item, "avg_test_rmse")
        runtime = to_float(item, "avg_runtime_sec")
        rows.append(
            {
                "method": METHOD_LABELS[method],
                "rmse": fmt(rmse),
                "std": fmt(to_float(item, "std_test_rmse")),
                "rel_fro": fmt(to_float(item, "avg_relative_fro_error")),
                "time": fmt(runtime),
                "rmse_vs_soft": pct((soft_rmse - rmse) / soft_rmse),
                "time_vs_soft": f"{runtime / soft_time:.2f}x",
            }
        )

    write_csv(rows, output_dir / "course_main_scalable_comparison.csv")
    latex_table(
        rows,
        [
            ("method", "Метод"),
            ("rmse", "RMSE"),
            ("rel_fro", "Отн. ошибка"),
            ("time", "Время, c"),
            ("rmse_vs_soft", "RMSE vs SI"),
            ("time_vs_soft", "Время vs SI"),
        ],
        "Основное сравнение масштабируемых методов на синтетических данных",
        "tab:main-scalable-results",
        output_dir / "course_main_scalable_comparison.tex",
        column_spec=r"p{0.34\linewidth}rrrrr",
    )


def make_upper_bound_table(summary_by_method: list[Row], output_dir: Path) -> None:
    methods = [
        "soft_impute",
        "riemannian_gd_l2_selected",
        "riemannian_gd_compact_l2_selected",
    ]
    by_method = index_by(summary_by_method, "method")
    rows: list[dict[str, Any]] = []
    for method in methods:
        item = by_method[method]
        rows.append(
            {
                "method": METHOD_LABELS[method],
                "rmse": fmt(to_float(item, "avg_test_rmse")),
                "std": fmt(to_float(item, "std_test_rmse")),
                "rel_fro": fmt(to_float(item, "avg_relative_fro_error")),
                "time": fmt(to_float(item, "avg_runtime_sec")),
                "iterations": f"{to_float(item, 'avg_iterations'):.1f}",
            }
        )

    write_csv(rows, output_dir / "course_quality_upper_bound.csv")
    latex_table(
        rows,
        [
            ("method", "Метод"),
            ("rmse", "RMSE"),
            ("std", "Std"),
            ("rel_fro", "Отн. ошибка"),
            ("time", "Время, c"),
            ("iterations", "Итерации"),
        ],
        "Validation-selected версии как верхняя планка качества",
        "tab:validation-selected-upper-bound",
        output_dir / "course_quality_upper_bound.tex",
        column_spec=r"p{0.38\linewidth}rrrrr",
    )


def make_research_progression_table(summary_by_method: list[Row], output_dir: Path) -> None:
    by_method = index_by(summary_by_method, "method")
    steps = [
        (
            "1",
            "riemannian_gd",
            "Простой RGD",
            "Учебная схема: градиент, проекция, SVD-ретракция.",
        ),
        (
            "2",
            "soft_impute",
            "Soft-Impute",
            "Baseline с ядерной нормой и shrinkage сингулярных значений.",
        ),
        (
            "3",
            "riemannian_gd_compact_l2_3e-02",
            r"Compact RGD, $\lambda=3\cdot 10^{-2}$",
            "Регуляризация снижает переобучение под шум.",
        ),
        (
            "4",
            "riemannian_gd_compact_l2_selected",
            r"Compact RGD, validation $\lambda$",
            "Параметр выбирается по отложенной части наблюдений.",
        ),
        (
            "5",
            "riemannian_gd_l2_selected",
            r"Full SVD RGD, validation $\lambda$",
            "Верхняя планка качества, но дорогая ретракция.",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for step, method, label, role in steps:
        item = by_method[method]
        rows.append(
            {
                "step": step,
                "method": label,
                "role": role,
                "rmse": fmt(to_float(item, "avg_test_rmse")),
                "rel_fro": fmt(to_float(item, "avg_relative_fro_error")),
                "time": fmt(to_float(item, "avg_runtime_sec")),
            }
        )

    write_csv(rows, output_dir / "course_research_progression.csv")
    latex_table(
        rows,
        [
            ("step", "Шаг"),
            ("method", "Метод"),
            ("role", "Идея"),
            ("rmse", "RMSE"),
            ("rel_fro", "Отн. ошибка"),
            ("time", "Время, c"),
        ],
        "Логика развития эксперимента",
        "tab:research-progression",
        output_dir / "course_research_progression.tex",
        column_spec=r"rp{0.20\linewidth}p{0.30\linewidth}rrr",
    )


def make_noise_table(summary_by_noise: list[Row], output_dir: Path) -> None:
    methods = [
        "soft_impute",
        "riemannian_gd_compact_l2_1e-02",
        "riemannian_gd_compact_l2_3e-02",
        "riemannian_gd_compact_l2_selected",
    ]
    grouped: dict[str, dict[str, Row]] = defaultdict(dict)
    for row in summary_by_noise:
        grouped[row["noise_std"]][row["method"]] = row

    rows: list[dict[str, Any]] = []
    for noise in sorted(grouped, key=float):
        for method in methods:
            item = grouped[noise][method]
            rows.append(
                {
                    "noise": noise,
                    "method": METHOD_LABELS[method],
                    "rmse": fmt(to_float(item, "avg_test_rmse")),
                    "rel_fro": fmt(to_float(item, "avg_relative_fro_error")),
                    "time": fmt(to_float(item, "avg_runtime_sec")),
                }
            )

    write_csv(rows, output_dir / "course_noise_scalable_comparison.csv")
    latex_table(
        rows,
        [
            ("noise", "Шум"),
            ("method", "Метод"),
            ("rmse", "RMSE"),
            ("rel_fro", "Отн. ошибка"),
            ("time", "Время, c"),
        ],
        "Поведение компактного RGD и Soft-Impute при разном уровне шума",
        "tab:noise-scalable-results",
        output_dir / "course_noise_scalable_comparison.tex",
        column_spec=r"rp{0.36\linewidth}rrr",
    )


def make_rank_table(summary_by_rank_case: list[Row], output_dir: Path) -> None:
    methods = [
        "soft_impute",
        "riemannian_gd_compact_l2_1e-02",
        "riemannian_gd_compact_l2_3e-02",
        "riemannian_gd_compact_l2_selected",
    ]
    grouped: dict[str, dict[str, Row]] = defaultdict(dict)
    for row in summary_by_rank_case:
        grouped[row["rank_case"]][row["method"]] = row

    order = ["under", "correct", "over"]
    rows: list[dict[str, Any]] = []
    for rank_case in order:
        for method in methods:
            item = grouped[rank_case][method]
            rows.append(
                {
                    "rank_case": RANK_LABELS[rank_case],
                    "method": METHOD_LABELS[method],
                    "rmse": fmt(to_float(item, "avg_test_rmse")),
                    "rel_fro": fmt(to_float(item, "avg_relative_fro_error")),
                    "time": fmt(to_float(item, "avg_runtime_sec")),
                }
            )

    write_csv(rows, output_dir / "course_rank_scalable_comparison.csv")
    latex_table(
        rows,
        [
            ("rank_case", "Ранг"),
            ("method", "Метод"),
            ("rmse", "RMSE"),
            ("rel_fro", "Отн. ошибка"),
            ("time", "Время, c"),
        ],
        "Зависимость качества от выбранного ранга",
        "tab:rank-scalable-results",
        output_dir / "course_rank_scalable_comparison.tex",
        column_spec=r"p{0.18\linewidth}p{0.34\linewidth}rrr",
    )


def make_compact_lambda_table(input_dir: Path, output_dir: Path) -> None:
    summary = read_csv(input_dir / "summary_by_compact_l2_reg.csv")
    wins = read_csv(input_dir / "best_compact_l2_counts.csv")
    wins_by_l2 = {row["l2_reg"]: row for row in wins}
    rows: list[dict[str, Any]] = []
    for row in summary:
        win_row = wins_by_l2.get(row["l2_reg"], {})
        rows.append(
            {
                "lambda": row["l2_reg"],
                "rmse": fmt(to_float(row, "avg_test_rmse")),
                "std": fmt(to_float(row, "std_test_rmse")),
                "rel_fro": fmt(to_float(row, "avg_relative_fro_error")),
                "time": fmt(to_float(row, "avg_runtime_sec")),
                "wins": win_row.get("wins", "0"),
                "win_rate": pct(float(win_row.get("win_rate", 0.0))),
            }
        )

    write_csv(rows, output_dir / "course_compact_lambda_sweep.csv")
    latex_table(
        rows,
        [
            ("lambda", r"$\lambda$"),
            ("rmse", "RMSE"),
            ("std", "Std"),
            ("rel_fro", "Отн. ошибка"),
            ("time", "Время, c"),
            ("wins", "Побед"),
            ("win_rate", "Доля"),
        ],
        "Сетка регуляризации для компактного RGD",
        "tab:compact-lambda-sweep",
        output_dir / "course_compact_lambda_sweep.tex",
        column_spec="rrrrrrr",
    )


def make_selected_lambda_table(input_dir: Path, output_dir: Path) -> None:
    rows_raw = read_csv(input_dir / "selected_compact_l2_counts.csv")
    rows = [
        {
            "lambda": row["selected_l2_reg"],
            "count": row["selected_count"],
            "rate": pct(float(row["selected_rate"])),
        }
        for row in rows_raw
    ]
    write_csv(rows, output_dir / "course_compact_selected_lambda_counts.csv")
    latex_table(
        rows,
        [
            ("lambda", r"Выбранное $\lambda$"),
            ("count", "Сценариев"),
            ("rate", "Доля"),
        ],
        "Какие значения регуляризации выбирает validation-процедура для компактного RGD",
        "tab:compact-selected-lambda-counts",
        output_dir / "course_compact_selected_lambda_counts.tex",
        column_spec="rrr",
    )


def make_compact_winner_table(input_dir: Path, output_dir: Path) -> None:
    rows_raw = read_csv(input_dir / "compact_validation_win_counts.csv")
    rows = [
        {
            "method": METHOD_LABELS.get(row["method"], row["method"]),
            "wins": row["wins"],
            "win_rate": pct(float(row["win_rate"])),
        }
        for row in rows_raw
    ]
    write_csv(rows, output_dir / "course_compact_validation_wins.csv")
    latex_table(
        rows,
        [
            ("method", "Метод"),
            ("wins", "Побед"),
            ("win_rate", "Доля"),
        ],
        "Число побед по сценариям для масштабируемого сравнения",
        "tab:compact-validation-wins",
        output_dir / "course_compact_validation_wins.tex",
        column_spec=r"p{0.42\linewidth}rr",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build course-ready tables from benchmark CSV outputs.")
    parser.add_argument("--input-dir", default="results_compact_l2_validation_r3")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "course_tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_by_method = read_csv(input_dir / "summary_by_method.csv")
    summary_by_noise = read_csv(input_dir / "summary_by_noise.csv")
    summary_by_rank_case = read_csv(input_dir / "summary_by_rank_case.csv")

    make_main_scalable_table(summary_by_method, output_dir)
    make_upper_bound_table(summary_by_method, output_dir)
    make_research_progression_table(summary_by_method, output_dir)
    make_noise_table(summary_by_noise, output_dir)
    make_rank_table(summary_by_rank_case, output_dir)
    make_compact_lambda_table(input_dir, output_dir)
    make_selected_lambda_table(input_dir, output_dir)
    make_compact_winner_table(input_dir, output_dir)

    print(f"Saved course-ready tables to {output_dir}")


if __name__ == "__main__":
    main()
