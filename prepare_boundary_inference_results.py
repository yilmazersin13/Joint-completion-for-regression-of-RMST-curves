#!/usr/bin/env python3
"""Validate and prepare the final JC--SP boundary-inference results.

The script reads a completed boundary-only run or the combined main run. It
does not refit the estimator or change a retained replication. It writes one
compact paper table, one vector-PDF coverage figure, CSV summaries, and an
audit file.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from prepare_simulation_results import PdfCanvas, draw_axes


PATTERN_ORDER = ["null", "delayed", "crossing"]
PATTERN_LABEL = {
    "null": "Null",
    "delayed": "Delayed",
    "crossing": "Crossing",
}


def read_csv(path):
    return pd.read_csv(path, keep_default_na=False, low_memory=False)


def numeric(frame, columns):
    frame[columns] = frame[columns].apply(pd.to_numeric, errors="coerce")


def validate_output(
    root,
    allow_pilot=False,
    screening=False,
    combined_main=False,
):
    required = [
        "run_manifest.json",
        "cell_status.csv",
        "horizon_results.csv",
        "replication_results.csv",
        "completion_diagnostics.csv",
        "failures.csv",
        "summary_horizon.csv",
        "summary_method.csv",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError("missing output files: " + ", ".join(missing))

    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    status = read_csv(root / "cell_status.csv")
    failures = read_csv(root / "failures.csv")
    horizon = read_csv(root / "horizon_results.csv")
    replication = read_csv(root / "replication_results.csv")
    completion = read_csv(root / "completion_diagnostics.csv")

    numeric(
        horizon,
        [
            "n",
            "censor_target",
            "replication",
            "replacement",
            "tau",
            "estimate",
            "truth",
            "error",
            "boundary_standard_error",
            "boundary_lower",
            "boundary_upper",
            "boundary_covered",
            "boundary_rejected",
        ],
    )
    numeric(
        replication,
        [
            "n",
            "censor_target",
            "replication",
            "replacement",
            "boundary_simultaneous_covered",
            "boundary_global_reject",
            "boundary_candidate_constraints",
            "boundary_cone_constraints",
            "boundary_exact_active_constraints",
            "boundary_valid_draws",
            "boundary_projection_failures",
            "boundary_projection_fallbacks",
            "boundary_projection_max_violation",
        ],
    )
    numeric(
        completion,
        [
            "n",
            "censor_target",
            "replication",
            "replacement",
            "dual_root_norm",
            "completed_path_violation",
        ],
    )
    status_columns = ["requested", "valid", "attempts", "failures", "complete"]
    numeric(status, status_columns)

    config = manifest.get("config", {})
    if combined_main:
        expected_design = "main"
        expected_reps = 500
        expected_draws = 399
        expected_cells = 26
        expected_valid = 13000
        expected_patterns = set(PATTERN_ORDER)
        expected_boundary_valid = 6000
    elif screening:
        expected_design = "boundary_main"
        expected_reps = 100
        expected_draws = 199
        expected_cells = 12
        expected_valid = 1200
        expected_patterns = set(PATTERN_ORDER)
        expected_boundary_valid = 1200
    elif allow_pilot:
        expected_design = "inference_pilot"
        expected_reps = 100
        expected_draws = 199
        expected_cells = 8
        expected_valid = 800
        expected_patterns = {"null", "crossing"}
        expected_boundary_valid = 800
    else:
        expected_design = "boundary_main"
        expected_reps = 500
        expected_draws = 399
        expected_cells = 12
        expected_valid = 6000
        expected_patterns = set(PATTERN_ORDER)
        expected_boundary_valid = 6000
    errors = []
    if manifest.get("status") != "complete":
        errors.append("manifest is not complete")
    if config.get("design") != expected_design:
        errors.append(f"run is not the {expected_design} design")
    if int(config.get("reps", -1)) != expected_reps:
        errors.append(
            f"run does not use {expected_reps} replications per cell"
        )
    if int(config.get("multiplier_draws", -1)) != expected_draws:
        errors.append(f"run does not use {expected_draws} multiplier draws")
    if not config.get("boundary_inference", False):
        errors.append("boundary inference was not enabled")
    if not allow_pilot:
        if not np.isclose(float(config.get("support_rate", np.nan)), 0.60):
            errors.append("final run does not use support rate 0.60")
        if not np.isclose(float(config.get("support_scale", np.nan)), 0.50):
            errors.append("final run does not use support scale 0.50")
    if (
        len(status) != expected_cells
        or int(status["valid"].sum()) != expected_valid
    ):
        errors.append("cell or retained-replication count is incorrect")
    if not (status["valid"] == status["requested"]).all():
        errors.append("one or more cells are incomplete")
    failure_count = int(status["failures"].sum())
    if len(failures) != failure_count:
        errors.append("failure log and cell-status counts disagree")
    if int(manifest.get("failed_attempts", -1)) != failure_count:
        errors.append("failure log and manifest counts disagree")
    if not (status["attempts"] == status["valid"] + status["failures"]).all():
        errors.append("attempt, valid, and failure counts disagree")
    if len(horizon) != expected_valid * 5 * 10:
        errors.append("horizon row count is incorrect")
    if (
        len(replication) != expected_valid * 5
        or len(completion) != expected_valid
    ):
        errors.append("method or completion row count is incorrect")
    if set(horizon["pattern"]) != expected_patterns:
        errors.append("treatment-pattern labels are incomplete")
    if not combined_main and set(horizon["nuisance"]) != {"correct"}:
        errors.append(
            "the boundary run contains an unexpected model-specification condition"
        )
    if not (
        horizon["replacement"]
        == (horizon["attempt"] != horizon["replication"]).astype(int)
    ).all():
        errors.append("the horizon replacement flags are inconsistent")
    if not (
        replication["replacement"]
        == (replication["attempt"] != replication["replication"]).astype(int)
    ).all():
        errors.append("the method replacement flags are inconsistent")
    if not (
        completion["replacement"]
        == (completion["attempt"] != completion["replication"]).astype(int)
    ).all():
        errors.append("the completion replacement flags are inconsistent")

    boundary_horizon = (
        (horizon["method"] == "JC-SP")
        & (horizon["nuisance"] == "correct")
    )
    boundary_replication = (
        (replication["method"] == "JC-SP")
        & (replication["nuisance"] == "correct")
    )
    jc_horizon = horizon[boundary_horizon].copy()
    jc_replication = replication[boundary_replication].copy()
    other_horizon = horizon[~boundary_horizon]
    other_replication = replication[~boundary_replication]
    if (
        len(jc_horizon) != expected_boundary_valid * 10
        or len(jc_replication) != expected_boundary_valid
    ):
        errors.append("JC--SP boundary row count is incorrect")
    if jc_horizon["boundary_covered"].isna().any():
        errors.append("a JC--SP pointwise boundary result is missing")
    if jc_replication["boundary_simultaneous_covered"].isna().any():
        errors.append("a JC--SP simultaneous boundary result is missing")
    if other_horizon["boundary_covered"].notna().any():
        errors.append("a comparison method has a boundary interval")
    if other_replication["boundary_simultaneous_covered"].notna().any():
        errors.append("a comparison method has a boundary band")
    if not (jc_replication["boundary_valid_draws"] == expected_draws).all():
        errors.append(
            "a JC--SP replication has fewer than the declared valid draws"
        )
    if int(jc_replication["boundary_projection_failures"].sum()) != 0:
        errors.append("a tangent-cone projection failed")
    if jc_replication["boundary_projection_max_violation"].max() > 2e-6:
        errors.append("a tangent-cone projection exceeds the tolerance")
    if not np.isfinite(
        jc_horizon[
            [
                "estimate",
                "truth",
                "error",
                "boundary_lower",
                "boundary_upper",
                "boundary_covered",
                "boundary_rejected",
            ]
        ].to_numpy()
    ).all():
        errors.append("a JC--SP boundary result is non-finite")
    if (jc_horizon["boundary_lower"] > jc_horizon["boundary_upper"]).any():
        errors.append("a pointwise boundary interval has reversed endpoints")
    if errors:
        raise RuntimeError("\n".join(errors))

    return manifest, status, jc_horizon, jc_replication, completion


def summarize_boundary(horizon, replication):
    pointwise = (
        horizon.groupby(
            ["n", "censor_target", "pattern", "tau"],
            as_index=False,
            dropna=False,
        )
        .agg(
            replications=("estimate", "size"),
            bias=("error", "mean"),
            rmse=("error", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
            empirical_sd=("estimate", "std"),
            average_boundary_se=("boundary_standard_error", "mean"),
            average_interval_width=(
                "boundary_upper",
                lambda x: float(
                    np.mean(
                        np.asarray(x)
                        - horizon.loc[x.index, "boundary_lower"].to_numpy()
                    )
                ),
            ),
            pointwise_coverage=("boundary_covered", "mean"),
            pointwise_rejection=("boundary_rejected", "mean"),
        )
    )
    pointwise["coverage_mcse"] = np.sqrt(
        pointwise["pointwise_coverage"]
        * (1.0 - pointwise["pointwise_coverage"])
        / pointwise["replications"]
    )
    pointwise["rejection_mcse"] = np.sqrt(
        pointwise["pointwise_rejection"]
        * (1.0 - pointwise["pointwise_rejection"])
        / pointwise["replications"]
    )

    simultaneous = (
        replication.groupby(
            ["n", "censor_target", "pattern"],
            as_index=False,
            dropna=False,
        )
        .agg(
            replications=("replication", "size"),
            simultaneous_coverage=("boundary_simultaneous_covered", "mean"),
            global_rejection=("boundary_global_reject", "mean"),
            mean_candidate_constraints=(
                "boundary_candidate_constraints",
                "mean",
            ),
            mean_exact_active_constraints=(
                "boundary_exact_active_constraints",
                "mean",
            ),
            projection_failures=("boundary_projection_failures", "sum"),
            projection_fallbacks=("boundary_projection_fallbacks", "sum"),
            maximum_projection_violation=(
                "boundary_projection_max_violation",
                "max",
            ),
        )
    )
    simultaneous["simultaneous_coverage_mcse"] = np.sqrt(
        simultaneous["simultaneous_coverage"]
        * (1.0 - simultaneous["simultaneous_coverage"])
        / simultaneous["replications"]
    )
    simultaneous["global_rejection_mcse"] = np.sqrt(
        simultaneous["global_rejection"]
        * (1.0 - simultaneous["global_rejection"])
        / simultaneous["replications"]
    )
    return pointwise, simultaneous


def make_paper_summary(pointwise, simultaneous):
    selected = pointwise[pointwise["tau"].isin([0.5, 2.5, 5.0])]
    coverage = selected.pivot(
        index=["n", "censor_target", "pattern"],
        columns="tau",
        values="pointwise_coverage",
    ).reset_index()
    coverage = coverage.rename(
        columns={
            0.5: "coverage_0_5",
            2.5: "coverage_2_5",
            5.0: "coverage_5_0",
        }
    )
    paper = coverage.merge(
        simultaneous,
        on=["n", "censor_target", "pattern"],
        how="left",
        validate="one_to_one",
    )
    paper["pattern_order"] = paper["pattern"].map(
        {name: index for index, name in enumerate(PATTERN_ORDER)}
    )
    return paper.sort_values(
        ["n", "censor_target", "pattern_order"]
    ).reset_index(drop=True)


def write_latex_table(path, paper, replications, multiplier_draws):
    displayed_probabilities = paper[
        [
            "coverage_0_5",
            "coverage_2_5",
            "coverage_5_0",
            "simultaneous_coverage",
            "global_rejection",
        ]
    ].to_numpy(dtype=float)
    maximum_mcse = float(
        np.sqrt(
            displayed_probabilities
            * (1.0 - displayed_probabilities)
            / replications
        ).max()
    )
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Boundary-aware inference for JC--SP when the event-time "
        "and censoring models are correctly specified.}",
        "\\label{tab:boundary-inference}",
        "\\small",
        "\\begin{tabular}{rrlrrrrr}",
        "\\toprule",
        "$n$ & Censoring & Pattern & "
        "\\multicolumn{3}{c}{Pointwise coverage} & "
        "Simultaneous & Global \\\\",
        "\\cmidrule(lr){4-6}",
        " &  &  & $\\tau=0.5$ & $\\tau=2.5$ & $\\tau=5$ & "
        "coverage & rejection \\\\",
        "\\midrule",
    ]
    previous_n = None
    previous_group = None
    for row in paper.itertuples():
        group = (int(row.n), float(row.censor_target))
        if previous_group is not None and group != previous_group:
            lines.append("\\addlinespace")
        n_label = ""
        if int(row.n) != previous_n:
            span = int((paper["n"] == row.n).sum())
            n_label = f"\\multirow{{{span}}}{{*}}{{{int(row.n)}}}"
        censor_label = ""
        if group != previous_group:
            span = int(
                (
                    (paper["n"] == row.n)
                    & np.isclose(paper["censor_target"], row.censor_target)
                ).sum()
            )
            censor_label = (
                f"\\multirow{{{span}}}{{*}}"
                f"{{{100 * row.censor_target:.0f}\\%}}"
            )
        lines.append(
            f"{n_label} & {censor_label} & "
            f"{PATTERN_LABEL[row.pattern]} & {row.coverage_0_5:.3f} & "
            f"{row.coverage_2_5:.3f} & {row.coverage_5_0:.3f} & "
            f"{row.simultaneous_coverage:.3f} & "
            f"{row.global_rejection:.3f} \\\\"
        )
        previous_n = int(row.n)
        previous_group = group
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "",
            "\\vspace{2pt}",
            "\\begin{minipage}{0.96\\linewidth}",
            f"\\footnotesize Each row contains {replications} replications. "
            "Pointwise "
            "intervals use the wider of the regular Gaussian and tangent-cone "
            "multiplier half-widths. Simultaneous bands use the regular and "
            "projected multiplier envelope. Global rejection is "
            "type-I error under the null and power under the delayed and "
            "crossing alternatives. Each multiplier calculation uses "
            f"{multiplier_draws} subject-level Gaussian draws. The largest "
            "Bernoulli Monte Carlo standard error among the displayed "
            f"coverage and rejection proportions is {maximum_mcse:.3f}. "
            "Boldface is not used because the rows represent different "
            "design settings rather than competing procedures.",
            "\\end{minipage}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_coverage_figure(path, pointwise):
    canvas = PdfCanvas(612, 470)
    canvas.text(
        306,
        452,
        "JC-SP pointwise coverage from the boundary-aware envelope",
        11,
        "F2",
        align="center",
    )
    styles = {
        "null": ((0.12, 0.31, 0.64), (), 1.4),
        "delayed": ((0.14, 0.55, 0.28), (7, 2), 1.4),
        "crossing": ((0.73, 0.19, 0.16), (2, 2), 1.4),
    }
    for index, pattern in enumerate(PATTERN_ORDER):
        x = 155 + 110 * index
        color, dash, width = styles[pattern]
        canvas.line(x, 430, x + 24, 430, width, color, dash)
        canvas.text(x + 30, 427, PATTERN_LABEL[pattern], 8)

    panel_width = 235
    panel_height = 155
    left_values = [55, 322]
    bottom_values = [245, 55]
    panels = [
        (250, 0.20, "n=250, 20% censoring"),
        (250, 0.50, "n=250, 50% censoring"),
        (500, 0.20, "n=500, 20% censoring"),
        (500, 0.50, "n=500, 50% censoring"),
    ]
    for index, (n, censor, title) in enumerate(panels):
        row = index // 2
        column = index % 2
        x0 = left_values[column]
        y0 = bottom_values[row]
        canvas.text(
            x0 + panel_width / 2,
            y0 + panel_height + 11,
            title,
            9,
            "F2",
            align="center",
        )
        map_x, map_y = draw_axes(
            canvas,
            x0,
            y0,
            panel_width,
            panel_height,
            (0.5, 5.0),
            (0.85, 1.005),
            [0.5, 2.5, 5.0],
            [0.85, 0.90, 0.95, 1.00],
            show_x_labels=True,
            show_y_labels=column == 0,
            y_decimals=2,
        )
        canvas.line(
            map_x(0.5),
            map_y(0.95),
            map_x(5.0),
            map_y(0.95),
            0.8,
            (0.35, 0.35, 0.35),
            (4, 2),
        )
        panel = pointwise[
            (pointwise["n"] == n)
            & np.isclose(pointwise["censor_target"], censor)
        ]
        for pattern in PATTERN_ORDER:
            group = panel[panel["pattern"] == pattern].sort_values("tau")
            points = [
                (map_x(tau), map_y(coverage))
                for tau, coverage in group[
                    ["tau", "pointwise_coverage"]
                ].itertuples(index=False)
            ]
            color, dash, width = styles[pattern]
            canvas.polyline(points, width, color, dash)
            for point_x, point_y in points:
                canvas.circle(point_x, point_y, 1.2, color, color)
    canvas.text(306, 20, "Time horizon (years)", 9, align="center")
    canvas.text(
        15,
        235,
        "Pointwise coverage",
        9,
        rotate=90,
        align="center",
    )
    canvas.save(path)


def write_audit(path, manifest, status, pointwise, simultaneous, replication):
    null_global = simultaneous[simultaneous["pattern"] == "null"][
        "global_rejection"
    ]
    horizon_means = pointwise.groupby("tau")["pointwise_coverage"].mean()
    lines = [
        "Boundary-inference simulation audit",
        f"script version: {manifest.get('script_version')}",
        f"cells: {len(status)}",
        f"valid replications: {int(status['valid'].sum())}",
        f"failed attempts: {int(status['failures'].sum())}",
        "multiplier draws per replication: "
        f"{int(manifest['config']['multiplier_draws'])}",
        "projection failures: "
        f"{int(replication['boundary_projection_failures'].sum())}",
        "projection fallbacks: "
        f"{int(replication['boundary_projection_fallbacks'].sum())}",
        "maximum projection violation: "
        f"{replication['boundary_projection_max_violation'].max():.3e}",
        "candidate constraint count: "
        f"mean={replication['boundary_candidate_constraints'].mean():.2f}, "
        f"range={int(replication['boundary_candidate_constraints'].min())}-"
        f"{int(replication['boundary_candidate_constraints'].max())}",
        "pointwise coverage range: "
        f"{pointwise['pointwise_coverage'].min():.3f}-"
        f"{pointwise['pointwise_coverage'].max():.3f}",
        "mean pointwise coverage across cells by horizon: "
        f"{horizon_means.min():.3f}-{horizon_means.max():.3f}",
        "simultaneous coverage range: "
        f"{simultaneous['simultaneous_coverage'].min():.3f}-"
        f"{simultaneous['simultaneous_coverage'].max():.3f}",
        "null global rejection range: "
        f"{null_global.min():.3f}-{null_global.max():.3f}",
        "all structural and numerical boundary-output checks passed",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        default="simulation_boundary_main_output",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="validate the locked 100-replication inference pilot",
    )
    parser.add_argument(
        "--combined-main",
        action="store_true",
        help="extract boundary results from a complete 26-cell main run",
    )
    parser.add_argument(
        "--screening",
        action="store_true",
        help="validate the 12-cell corrected-grid boundary screen",
    )
    args = parser.parse_args()
    if sum([args.pilot, args.screening, args.combined_main]) > 1:
        parser.error(
            "--pilot, --screening, and --combined-main are mutually exclusive"
        )
    root = Path(args.output)

    manifest, status, horizon, replication, completion = validate_output(
        root,
        allow_pilot=args.pilot,
        screening=args.screening,
        combined_main=args.combined_main,
    )
    pointwise, simultaneous = summarize_boundary(horizon, replication)
    paper = make_paper_summary(pointwise, simultaneous)

    pointwise.to_csv(root / "boundary_pointwise_summary.csv", index=False)
    simultaneous.to_csv(root / "boundary_simultaneous_summary.csv", index=False)
    paper.to_csv(root / "paper_boundary_inference_summary.csv", index=False)
    write_latex_table(
        root / "paper_table_boundary_inference.tex",
        paper,
        int(manifest["config"]["reps"]),
        int(manifest["config"]["multiplier_draws"]),
    )
    write_coverage_figure(root / "figure_boundary_coverage.pdf", pointwise)
    write_audit(
        root / "boundary_results_audit.txt",
        manifest,
        status,
        pointwise,
        simultaneous,
        replication,
    )

    print(f"output: {root.resolve()}")
    print(f"valid replications: {int(status['valid'].sum())}")
    print(
        "pointwise coverage range: "
        f"{pointwise['pointwise_coverage'].min():.3f} to "
        f"{pointwise['pointwise_coverage'].max():.3f}"
    )
    print(
        "simultaneous coverage range: "
        f"{simultaneous['simultaneous_coverage'].min():.3f} to "
        f"{simultaneous['simultaneous_coverage'].max():.3f}"
    )
    print("boundary-inference paper files written")


if __name__ == "__main__":
    main()
