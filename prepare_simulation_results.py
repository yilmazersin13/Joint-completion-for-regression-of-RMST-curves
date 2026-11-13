#!/usr/bin/env python3
"""Prepare manuscript tables and figures from the saved main simulation.

The script reads the raw retained-replication files, checks their agreement
with the saved summaries, and writes the paper artifacts into the same output
folder.  It does not rerun or alter any simulation replication.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_ORDER = ["JC-SP", "JC-P", "CT-SP", "IPCW-SP", "ORACLE-SP"]
PATTERN_ORDER = ["null", "delayed", "crossing"]
NUISANCE_ORDER = ["correct", "event_miss", "censor_miss", "both_miss"]

METHOD_LABEL = {
    "JC-SP": "JC--SP",
    "JC-P": "JC--P",
    "CT-SP": "CT--SP",
    "IPCW-SP": "IPCW--SP",
    "ORACLE-SP": "ORACLE--SP",
}
PATTERN_LABEL = {
    "null": "Null",
    "delayed": "Delayed",
    "crossing": "Crossing",
}
NUISANCE_LABEL = {
    "correct": "Both correct",
    "event_miss": "Event-time misspecified",
    "censor_miss": "Censoring misspecified",
    "both_miss": "Both misspecified",
}


def read_csv(path: Path) -> pd.DataFrame:
    """Read labels such as 'null' literally rather than as missing values."""
    return pd.read_csv(path, keep_default_na=False, low_memory=False)


def numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    frame[columns] = frame[columns].apply(pd.to_numeric, errors="coerce")


def maximum_absolute_difference(left, right) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    finite = np.isfinite(left) & np.isfinite(right)
    both_missing = ~np.isfinite(left) & ~np.isfinite(right)
    if not np.all(finite | both_missing):
        return math.inf
    if not finite.any():
        return 0.0
    return float(np.max(np.abs(left[finite] - right[finite])))


def validate_output(root: Path):
    required = [
        "run_manifest.json",
        "cell_status.csv",
        "horizon_results.csv",
        "replication_results.csv",
        "completion_diagnostics.csv",
        "failures.csv",
        "summary_horizon.csv",
        "summary_method.csv",
        "summary_completion.csv",
        "censoring_calibration.csv",
    ]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise RuntimeError("missing output files: " + ", ".join(missing))

    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    status = read_csv(root / "cell_status.csv")
    horizon = read_csv(root / "horizon_results.csv")
    replication = read_csv(root / "replication_results.csv")
    completion = read_csv(root / "completion_diagnostics.csv")
    failures = read_csv(root / "failures.csv")
    summary_horizon = read_csv(root / "summary_horizon.csv")
    summary_method = read_csv(root / "summary_method.csv")
    summary_completion = read_csv(root / "summary_completion.csv")

    horizon_numeric = [
        "n",
        "censor_target",
        "replication",
        "attempt",
        "seed",
        "replacement",
        "tau",
        "estimate",
        "truth",
        "error",
        "standard_error",
        "covered",
        "rejected",
    ]
    replication_numeric = [
        "n",
        "censor_target",
        "replication",
        "attempt",
        "seed",
        "replacement",
        "ise",
        "lambda",
        "gcv",
        "active_constraints",
        "qp_iterations",
        "qp_violation",
        "simultaneous_covered",
        "global_reject",
        "v_range",
        "v_concave",
        "any_shape_violation",
        "max_shape_violation",
    ]
    completion_numeric = [
        "n",
        "censor_target",
        "replication",
        "attempt",
        "seed",
        "replacement",
        "observed_censoring",
        "n_completed",
        "tail_ipcw_gap",
        "fitted_ipcw_gap",
        "gap_ratio",
        "mean_kl",
        "kl_per_subject",
        "mean_effective_support",
        "dual_iterations",
        "dual_root_norm",
        "completed_path_violation",
        "tail_path_violation",
        "event_fit_iterations",
        "censor_fit_iterations",
        "runtime_seconds",
    ]
    numeric(horizon, horizon_numeric)
    numeric(replication, replication_numeric)
    numeric(completion, completion_numeric)
    numeric(status, ["requested", "valid", "attempts", "failures", "complete"])

    if manifest.get("status") != "complete":
        raise RuntimeError("the saved manifest is not complete")
    if manifest.get("config", {}).get("design") != "main":
        raise RuntimeError("the saved run is not the declared main design")
    if int(manifest.get("config", {}).get("reps", -1)) != 500:
        raise RuntimeError("the saved run does not use 500 replications per cell")
    if not np.isclose(
        float(manifest.get("config", {}).get("support_rate", np.nan)),
        0.60,
    ):
        raise RuntimeError("the saved run does not use support rate 0.60")
    if not np.isclose(
        float(manifest.get("config", {}).get("support_scale", np.nan)),
        0.50,
    ):
        raise RuntimeError("the saved run does not use support scale 0.50")
    if len(status) != 26 or int(status["valid"].sum()) != 13000:
        raise RuntimeError("the retained cell or replication count is incorrect")
    if not (status["valid"] == status["requested"]).all():
        raise RuntimeError("one or more cells are incomplete")
    failure_count = int(status["failures"].sum())
    if len(failures) != failure_count:
        raise RuntimeError("the failure log and cell-status counts disagree")
    if int(manifest.get("failed_attempts", -1)) != failure_count:
        raise RuntimeError("the failure log and manifest counts disagree")
    if not (status["attempts"] == status["valid"] + status["failures"]).all():
        raise RuntimeError("attempt, valid, and failure counts disagree")
    if not (
        horizon["replacement"]
        == (horizon["attempt"] != horizon["replication"]).astype(int)
    ).all():
        raise RuntimeError("the horizon replacement flags are inconsistent")
    if not (
        replication["replacement"]
        == (replication["attempt"] != replication["replication"]).astype(int)
    ).all():
        raise RuntimeError("the method replacement flags are inconsistent")
    if not (
        completion["replacement"]
        == (completion["attempt"] != completion["replication"]).astype(int)
    ).all():
        raise RuntimeError("the completion replacement flags are inconsistent")
    if len(horizon) != 650000 or len(replication) != 65000:
        raise RuntimeError("the raw horizon or method row count is incorrect")
    if len(completion) != 13000:
        raise RuntimeError("the completion row count is incorrect")
    if set(horizon["pattern"]) != set(PATTERN_ORDER):
        raise RuntimeError("the treatment-pattern labels are incomplete")
    if set(horizon["method"]) != set(METHOD_ORDER):
        raise RuntimeError("the method labels are incomplete")

    error_difference = maximum_absolute_difference(
        horizon["error"], horizon["estimate"] - horizon["truth"]
    )
    if error_difference > 1e-12:
        raise RuntimeError("estimate, truth, and error are inconsistent")

    ise_check = (
        horizon.assign(error_squared=horizon["error"] ** 2)
        .groupby(["cell_id", "replication", "method"], as_index=False)
        .agg(ise_from_horizons=("error_squared", "mean"))
    )
    ise_check = ise_check.merge(
        replication[["cell_id", "replication", "method", "ise"]],
        on=["cell_id", "replication", "method"],
        validate="one_to_one",
    )
    ise_difference = maximum_absolute_difference(
        ise_check["ise_from_horizons"], ise_check["ise"]
    )
    if ise_difference > 1e-12:
        raise RuntimeError("stored ISE does not agree with the horizon errors")

    horizon_group = [
        "n",
        "censor_target",
        "pattern",
        "nuisance",
        "method",
        "tau",
    ]
    recalculated_horizon = (
        horizon.groupby(horizon_group, as_index=False)
        .agg(
            replications=("estimate", "size"),
            mean_estimate=("estimate", "mean"),
            truth=("truth", "first"),
            bias=("error", "mean"),
            rmse=("error", lambda x: float(np.sqrt(np.mean(x**2)))),
            empirical_sd=("estimate", "std"),
            average_se=("standard_error", "mean"),
            inference_replications=("standard_error", "count"),
            coverage=("covered", "mean"),
            rejection=("rejected", "mean"),
        )
        .sort_values(horizon_group)
        .reset_index(drop=True)
    )
    horizon_summary_numeric = [
        "replications",
        "mean_estimate",
        "truth",
        "bias",
        "rmse",
        "empirical_sd",
        "average_se",
        "inference_replications",
        "coverage",
        "rejection",
    ]
    numeric(summary_horizon, horizon_summary_numeric)
    saved_horizon = summary_horizon.sort_values(horizon_group).reset_index(drop=True)
    if recalculated_horizon[horizon_group].astype(str).to_numpy().tolist() != (
        saved_horizon[horizon_group].astype(str).to_numpy().tolist()
    ):
        raise RuntimeError("saved horizon summary has inconsistent grouping labels")
    horizon_summary_difference = max(
        maximum_absolute_difference(
            recalculated_horizon[column], saved_horizon[column]
        )
        for column in horizon_summary_numeric
    )
    if horizon_summary_difference > 5e-12:
        raise RuntimeError("saved horizon summary disagrees with the raw output")

    method_group = ["n", "censor_target", "pattern", "nuisance", "method"]
    recalculated_method = (
        replication.groupby(method_group, as_index=False)
        .agg(
            replications=("ise", "size"),
            mean_ise=("ise", "mean"),
            median_ise=("ise", "median"),
            shape_violation_rate=("any_shape_violation", "mean"),
            mean_v_range=("v_range", "mean"),
            mean_v_concave=("v_concave", "mean"),
            mean_active_constraints=("active_constraints", "mean"),
            mean_qp_iterations=("qp_iterations", "mean"),
            simultaneous_inference_replications=("simultaneous_covered", "count"),
            simultaneous_coverage=("simultaneous_covered", "mean"),
            global_rejection=("global_reject", "mean"),
        )
        .sort_values(method_group)
        .reset_index(drop=True)
    )
    method_summary_numeric = [
        "replications",
        "mean_ise",
        "median_ise",
        "shape_violation_rate",
        "mean_v_range",
        "mean_v_concave",
        "mean_active_constraints",
        "mean_qp_iterations",
        "simultaneous_inference_replications",
        "simultaneous_coverage",
        "global_rejection",
    ]
    numeric(summary_method, method_summary_numeric)
    saved_method = summary_method.sort_values(method_group).reset_index(drop=True)
    if recalculated_method[method_group].astype(str).to_numpy().tolist() != (
        saved_method[method_group].astype(str).to_numpy().tolist()
    ):
        raise RuntimeError("saved method summary has inconsistent grouping labels")
    method_summary_difference = max(
        maximum_absolute_difference(recalculated_method[column], saved_method[column])
        for column in method_summary_numeric
    )
    if method_summary_difference > 5e-12:
        raise RuntimeError("saved method summary disagrees with the raw output")

    completion_group = ["n", "censor_target", "pattern", "nuisance"]
    recalculated_completion = (
        completion.groupby(completion_group, as_index=False)
        .agg(
            replications=("gap_ratio", "size"),
            observed_censoring=("observed_censoring", "mean"),
            tail_ipcw_gap=("tail_ipcw_gap", "mean"),
            fitted_ipcw_gap=("fitted_ipcw_gap", "mean"),
            gap_ratio=("gap_ratio", "mean"),
            mean_kl=("mean_kl", "mean"),
            mean_effective_support=("mean_effective_support", "mean"),
            completed_path_max=("completed_path_violation", "max"),
            tail_path_max=("tail_path_violation", "max"),
            mean_dual_iterations=("dual_iterations", "mean"),
            max_dual_root=("dual_root_norm", "max"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
        )
        .sort_values(completion_group)
        .reset_index(drop=True)
    )
    completion_summary_numeric = [
        "replications",
        "observed_censoring",
        "tail_ipcw_gap",
        "fitted_ipcw_gap",
        "gap_ratio",
        "mean_kl",
        "mean_effective_support",
        "completed_path_max",
        "tail_path_max",
        "mean_dual_iterations",
        "max_dual_root",
        "mean_runtime_seconds",
    ]
    numeric(summary_completion, completion_summary_numeric)
    saved_completion = summary_completion.sort_values(completion_group).reset_index(
        drop=True
    )
    if recalculated_completion[completion_group].astype(str).to_numpy().tolist() != (
        saved_completion[completion_group].astype(str).to_numpy().tolist()
    ):
        raise RuntimeError("saved completion summary has inconsistent grouping labels")
    completion_summary_difference = max(
        maximum_absolute_difference(
            recalculated_completion[column], saved_completion[column]
        )
        for column in completion_summary_numeric
    )
    if completion_summary_difference > 5e-12:
        raise RuntimeError("saved completion summary disagrees with the raw output")

    if (
        completion["fitted_ipcw_gap"]
        > completion["tail_ipcw_gap"] + 1e-8
    ).any():
        raise RuntimeError("calibration increased a moment gap")
    if completion["dual_root_norm"].max() > 1e-6:
        raise RuntimeError("a dual residual exceeds the numerical tolerance")
    if completion["completed_path_violation"].max() > 1e-8:
        raise RuntimeError("a completed path violates the RMST shape")
    constrained = replication[replication["method"].str.endswith("-SP")]
    if constrained["max_shape_violation"].max() > 1e-8:
        raise RuntimeError("a constrained fitted curve violates the RMST shape")

    jcsp = replication[replication["method"] == "JC-SP"]
    invalid_jcsp_inference = jcsp[
        (jcsp["active_constraints"] > 0)
        & np.isfinite(jcsp["simultaneous_covered"])
    ]
    if not invalid_jcsp_inference.empty:
        raise RuntimeError("ordinary JC-SP inference was used at an active boundary")
    jcp_correct = replication[
        (replication["method"] == "JC-P")
        & (replication["nuisance"] == "correct")
    ]
    if jcp_correct["simultaneous_covered"].count() != 6000:
        raise RuntimeError("JC-P simultaneous inference is incomplete")

    audit = {
        "manifest": manifest,
        "status": status,
        "horizon": horizon,
        "replication": replication,
        "completion": completion,
        "failures": failures,
        "summary_horizon": saved_horizon,
        "summary_method": saved_method,
        "error_difference": error_difference,
        "ise_difference": ise_difference,
        "horizon_summary_difference": horizon_summary_difference,
        "method_summary_difference": method_summary_difference,
        "completion_summary_difference": completion_summary_difference,
    }
    return audit


def derive_estimation_table(audit) -> pd.DataFrame:
    summary_horizon = audit["summary_horizon"]
    summary_method = audit["summary_method"]
    main_h = summary_horizon[
        (summary_horizon["nuisance"] == "correct")
        & summary_horizon["pattern"].isin(PATTERN_ORDER)
    ]
    main_m = summary_method[
        (summary_method["nuisance"] == "correct")
        & summary_method["pattern"].isin(PATTERN_ORDER)
    ]
    curve = (
        main_m.groupby(["n", "censor_target", "method"], as_index=False)
        .agg(
            mean_ise=("mean_ise", "mean"),
            shape_violation_rate=("shape_violation_rate", "mean"),
            mean_active_constraints=("mean_active_constraints", "mean"),
        )
    )
    curve["curve_rmse"] = np.sqrt(curve["mean_ise"])
    bias = (
        main_h.assign(absolute_bias=main_h["bias"].abs())
        .groupby(["n", "censor_target", "method"], as_index=False)
        .agg(mean_absolute_bias=("absolute_bias", "mean"))
    )
    table = curve.merge(bias, on=["n", "censor_target", "method"])
    table["method_order"] = table["method"].map(
        {name: order for order, name in enumerate(METHOD_ORDER)}
    )
    return table.sort_values(
        ["n", "censor_target", "method_order"]
    ).reset_index(drop=True)


def derive_inference_table(audit) -> pd.DataFrame:
    summary_horizon = audit["summary_horizon"]
    summary_method = audit["summary_method"]
    rows = []
    for (n, censor, pattern), group in summary_horizon[
        (summary_horizon["method"] == "JC-P")
        & (summary_horizon["nuisance"] == "correct")
    ].groupby(["n", "censor_target", "pattern"]):
        method_row = summary_method[
            (summary_method["n"] == n)
            & (summary_method["censor_target"] == censor)
            & (summary_method["pattern"] == pattern)
            & (summary_method["nuisance"] == "correct")
            & (summary_method["method"] == "JC-P")
        ].iloc[0]
        rows.append(
            {
                "n": n,
                "censor_target": censor,
                "pattern": pattern,
                "pointwise_coverage_min": group["coverage"].min(),
                "pointwise_coverage_max": group["coverage"].max(),
                "tau5_coverage": group.loc[group["tau"] == 5, "coverage"].iloc[0],
                "simultaneous_coverage": method_row["simultaneous_coverage"],
                "global_rejection": method_row["global_rejection"],
            }
        )
    table = pd.DataFrame(rows)
    table["pattern_order"] = table["pattern"].map(
        {name: order for order, name in enumerate(PATTERN_ORDER)}
    )
    return table.sort_values(
        ["n", "censor_target", "pattern_order"]
    ).reset_index(drop=True)


def derive_misspecification_table(audit) -> pd.DataFrame:
    summary_horizon = audit["summary_horizon"]
    summary_method = audit["summary_method"]
    selected_method = summary_method[
        (summary_method["n"] == 500)
        & (summary_method["censor_target"] == 0.5)
        & (summary_method["pattern"] == "delayed")
    ][["nuisance", "method", "mean_ise"]].copy()
    selected_method["curve_rmse"] = np.sqrt(selected_method["mean_ise"])
    selected_horizon = summary_horizon[
        (summary_horizon["n"] == 500)
        & (summary_horizon["censor_target"] == 0.5)
        & (summary_horizon["pattern"] == "delayed")
        & (summary_horizon["tau"] == 5)
    ][["nuisance", "method", "bias", "rmse"]]
    table = selected_method.merge(
        selected_horizon, on=["nuisance", "method"], validate="one_to_one"
    )
    table["nuisance_order"] = table["nuisance"].map(
        {name: order for order, name in enumerate(NUISANCE_ORDER)}
    )
    table["method_order"] = table["method"].map(
        {name: order for order, name in enumerate(METHOD_ORDER)}
    )
    return table.sort_values(
        ["nuisance_order", "method_order"]
    ).reset_index(drop=True)


def derive_completion_table(audit) -> pd.DataFrame:
    completion = audit["completion"]
    main = completion[completion["nuisance"].isin(["correct", "both_miss"])]
    table = (
        main.groupby(["n", "censor_target", "nuisance"], as_index=False)
        .agg(
            gap_before=("tail_ipcw_gap", "mean"),
            gap_after=("fitted_ipcw_gap", "mean"),
            gap_ratio=("gap_ratio", "mean"),
            gap_ratio_q05=("gap_ratio", lambda x: x.quantile(0.05)),
            gap_ratio_q25=("gap_ratio", lambda x: x.quantile(0.25)),
            gap_ratio_median=("gap_ratio", "median"),
            gap_ratio_q75=("gap_ratio", lambda x: x.quantile(0.75)),
            gap_ratio_q95=("gap_ratio", lambda x: x.quantile(0.95)),
            mean_kl=("mean_kl", "mean"),
            mean_effective_support=("mean_effective_support", "mean"),
        )
    )
    table["nuisance_order"] = table["nuisance"].map(
        {"correct": 0, "both_miss": 1}
    )
    return table.sort_values(
        ["n", "censor_target", "nuisance_order"]
    ).reset_index(drop=True)


def table_start(
    caption: str,
    label: str,
    columns: str,
    size: str = "small",
    tabcolsep: float | None = None,
) -> list[str]:
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\{size}",
    ]
    if tabcolsep is not None:
        lines.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep:g}pt}}")
    lines.extend(
        [
            "\\begin{tabular}{" + columns + "}",
            "\\toprule",
        ]
    )
    return lines


def table_end(note: str) -> list[str]:
    return [
        "\\bottomrule",
        "\\end{tabular}",
        "",
        "\\vspace{2pt}",
        "\\begin{minipage}{0.96\\linewidth}",
        "\\footnotesize " + note,
        "\\end{minipage}",
        "\\end{table}",
        "",
    ]


def latex_number(value: float, digits: int, bold: bool = False) -> str:
    text = f"{value:.{digits}f}"
    return f"\\textbf{{{text}}}" if bold else text


def write_estimation_table(path: Path, table: pd.DataFrame) -> None:
    lines = table_start(
        "Estimation and fitted-curve shape when the event-time and censoring "
        "models are correctly specified.",
        "tab:sim-estimation",
        "rrlrrrr",
        "small",
        4,
    )
    lines.extend(
        [
            "$n$ & Censoring & Method & "
            "\\shortstack{Mean absolute\\\\bias} & "
            "\\shortstack{Curve\\\\RMSE} & "
            "\\shortstack{Shape\\\\violation (\\%)} & "
            "\\shortstack{Active\\\\constraints} \\\\",
            "\\midrule",
        ]
    )
    feasible = table.loc[table["method"] != "ORACLE-SP"]
    best = feasible.groupby(["n", "censor_target"])[
        ["mean_absolute_bias", "curve_rmse"]
    ].min()
    previous_n = None
    previous_group = None
    for row in table.itertuples():
        group = (int(row.n), float(row.censor_target))
        if previous_group is not None and group != previous_group:
            lines.append("\\addlinespace")
        n_label = ""
        if int(row.n) != previous_n:
            span = int((table["n"] == row.n).sum())
            n_label = f"\\multirow{{{span}}}{{*}}{{{int(row.n)}}}"
        censor_label = ""
        if group != previous_group:
            span = int(
                (
                    (table["n"] == row.n)
                    & np.isclose(table["censor_target"], row.censor_target)
                ).sum()
            )
            censor_label = (
                f"\\multirow{{{span}}}{{*}}"
                f"{{{100 * row.censor_target:.0f}\\%}}"
            )
        active = (
            "--"
            if row.method == "JC-P"
            else f"{row.mean_active_constraints:.2f}"
        )
        best_row = best.loc[group]
        bias = latex_number(
            row.mean_absolute_bias,
            4,
            row.method != "ORACLE-SP"
            and np.isclose(row.mean_absolute_bias, best_row.mean_absolute_bias),
        )
        rmse = latex_number(
            row.curve_rmse,
            4,
            row.method != "ORACLE-SP"
            and np.isclose(row.curve_rmse, best_row.curve_rmse),
        )
        lines.append(
            f"{n_label} & {censor_label} & "
            f"{METHOD_LABEL[row.method]} & {bias} & {rmse} & "
            f"{100 * row.shape_violation_rate:.1f} & {active} \\\\"
        )
        previous_n = int(row.n)
        previous_group = group
    lines.extend(
        table_end(
            "The absolute bias is averaged over the three treatment patterns "
            "and ten horizons. Curve RMSE is the square root of the mean ISE "
            "over the three patterns. A shape violation means that at least "
            "one fitted profile exceeded the $10^{-8}$ tolerance. JC--P is "
            "unconstrained, so the number of active constraints is not defined. "
            "Within each sample-size and censoring setting, boldface marks the "
            "smallest error among the feasible estimators. ORACLE--SP is not "
            "ranked because it is unavailable in practice."
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_inference_table(path: Path, table: pd.DataFrame) -> None:
    lines = table_start(
        "JC--P inference when the event-time and censoring models are "
        "correctly specified.",
        "tab:sim-inference",
        "rrlrrrr",
        "scriptsize",
        4,
    )
    lines.extend(
        [
            "$n$ & Censoring & Pattern & "
            "\\shortstack{Pointwise\\\\range} & "
            "\\shortstack{Coverage\\\\at 5} & "
            "\\shortstack{Simultaneous\\\\coverage} & "
            "\\shortstack{Global\\\\rejection} \\\\",
            "\\midrule",
        ]
    )
    previous_n = None
    previous_group = None
    for row in table.itertuples():
        group = (int(row.n), float(row.censor_target))
        if previous_group is not None and group != previous_group:
            lines.append("\\addlinespace")
        n_label = ""
        if int(row.n) != previous_n:
            span = int((table["n"] == row.n).sum())
            n_label = f"\\multirow{{{span}}}{{*}}{{{int(row.n)}}}"
        censor_label = ""
        if group != previous_group:
            span = int(
                (
                    (table["n"] == row.n)
                    & np.isclose(table["censor_target"], row.censor_target)
                ).sum()
            )
            censor_label = (
                f"\\multirow{{{span}}}{{*}}"
                f"{{{100 * row.censor_target:.0f}\\%}}"
            )
        coverage_range = (
            f"{row.pointwise_coverage_min:.3f}--"
            f"{row.pointwise_coverage_max:.3f}"
        )
        lines.append(
            f"{n_label} & {censor_label} & "
            f"{PATTERN_LABEL[row.pattern]} & {coverage_range} & "
            f"{row.tau5_coverage:.3f} & "
            f"{row.simultaneous_coverage:.3f} & "
            f"{row.global_rejection:.3f} \\\\"
        )
        previous_n = int(row.n)
        previous_group = group
    lines.extend(
        table_end(
            "Each entry is based on 500 replications and 399 multiplier draws. "
            "The pointwise range is over the ten horizons. The global rejection "
            "proportion is type-I error in the null setting and power in the "
            "delayed and crossing settings. The largest Bernoulli Monte Carlo "
            "standard error among the displayed coverage and rejection "
            "proportions is 0.022. Boldface is not used because the rows "
            "represent different design settings rather than competing methods."
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_misspecification_table(path: Path, table: pd.DataFrame) -> None:
    lines = table_start(
        "Focused model-misspecification comparison for the delayed-effect "
        "setting with $n=500$ and 50\\% censoring.",
        "tab:sim-misspecification",
        "llrrr",
    )
    lines.extend(
        [
            "Model specification & Method & Curve RMSE & Bias at 5 & RMSE at 5 \\\\",
            "\\midrule",
        ]
    )
    feasible = table.loc[table["method"] != "ORACLE-SP"].copy()
    feasible["absolute_bias"] = feasible["bias"].abs()
    best = feasible.groupby("nuisance")[
        ["curve_rmse", "absolute_bias", "rmse"]
    ].min()
    previous = None
    for row in table.itertuples():
        if previous is not None and row.nuisance != previous:
            lines.append("\\addlinespace")
        nuisance = ""
        if row.nuisance != previous:
            span = int((table["nuisance"] == row.nuisance).sum())
            nuisance = (
                f"\\multirow{{{span}}}{{*}}"
                f"{{{NUISANCE_LABEL[row.nuisance]}}}"
            )
        best_row = best.loc[row.nuisance]
        eligible = row.method != "ORACLE-SP"
        curve_rmse = latex_number(
            row.curve_rmse,
            4,
            eligible and np.isclose(row.curve_rmse, best_row.curve_rmse),
        )
        bias = latex_number(
            row.bias,
            4,
            eligible and np.isclose(abs(row.bias), best_row.absolute_bias),
        )
        rmse = latex_number(
            row.rmse,
            4,
            eligible and np.isclose(row.rmse, best_row.rmse),
        )
        lines.append(
            f"{nuisance} & {METHOD_LABEL[row.method]} & "
            f"{curve_rmse} & {bias} & {rmse} \\\\"
        )
        previous = row.nuisance
    lines.extend(
        table_end(
            "Curve RMSE is the square root of the mean ISE over the ten "
            "horizons. Event-time misspecified means that the censoring model is "
            "correct, and censoring misspecified means that the event-time "
            "model is correct. ORACLE--SP uses the complete restricted event "
            "times and is included only as a reference. Within each model "
            "specification, boldface marks the smallest curve RMSE, smallest "
            "absolute five-year bias, and smallest five-year RMSE among the "
            "feasible estimators. ORACLE--SP is not ranked."
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_completion_table(path: Path, table: pd.DataFrame) -> None:
    lines = table_start(
        "Joint-completion diagnostics in the main grid.",
        "tab:sim-completion",
        "rrlrrrrr",
        "scriptsize",
        4,
    )
    lines.extend(
        [
            "$n$ & Censoring & Model specification & "
            "\\shortstack{Gap\\\\before} & "
            "\\shortstack{Gap\\\\after} & "
            "\\shortstack{Gap\\\\ratio} & Mean KL & "
            "\\shortstack{Effective\\\\support} \\\\",
            "\\midrule",
        ]
    )
    previous_n = None
    previous_group = None
    for row in table.itertuples():
        group = (int(row.n), float(row.censor_target))
        if previous_group is not None and group != previous_group:
            lines.append("\\addlinespace")
        n_label = ""
        if int(row.n) != previous_n:
            span = int((table["n"] == row.n).sum())
            n_label = f"\\multirow{{{span}}}{{*}}{{{int(row.n)}}}"
        censor_label = ""
        if group != previous_group:
            span = int(
                (
                    (table["n"] == row.n)
                    & np.isclose(table["censor_target"], row.censor_target)
                ).sum()
            )
            censor_label = (
                f"\\multirow{{{span}}}{{*}}"
                f"{{{100 * row.censor_target:.0f}\\%}}"
            )
        lines.append(
            f"{n_label} & {censor_label} & "
            f"{NUISANCE_LABEL[row.nuisance]} & {row.gap_before:.3f} & "
            f"{row.gap_after:.3f} & {row.gap_ratio:.3f} & "
            f"{row.mean_kl:.3f} & {row.mean_effective_support:.2f} \\\\"
        )
        previous_n = int(row.n)
        previous_group = group
    lines.extend(
        table_end(
            "Each row averages 1,500 replications over the three treatment "
            "patterns. The gaps use the standardized regression-score "
            "coordinates. Mean KL is the recipient mean KL divergence from "
            "the fitted conditional-tail reference. Effective support is the "
            "exponential-entropy support size among completed subjects. "
            "Boldface is not used because these columns are diagnostics with "
            "different scales and no single best direction."
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class PdfCanvas:
    """Small vector-PDF canvas sufficient for the two simulation figures."""

    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        self.commands: list[str] = []

    def line(
        self,
        x1,
        y1,
        x2,
        y2,
        width=1.0,
        color=(0.0, 0.0, 0.0),
        dash=(),
    ):
        dash_text = " ".join(f"{value:g}" for value in dash)
        self.commands.append(
            "q "
            f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} RG "
            f"{width:.3f} w [{dash_text}] 0 d "
            f"{x1:.3f} {y1:.3f} m {x2:.3f} {y2:.3f} l S Q"
        )

    def polyline(
        self,
        points,
        width=1.0,
        color=(0.0, 0.0, 0.0),
        dash=(),
    ):
        if len(points) < 2:
            return
        dash_text = " ".join(f"{value:g}" for value in dash)
        path = [f"{points[0][0]:.3f} {points[0][1]:.3f} m"]
        path.extend(f"{x:.3f} {y:.3f} l" for x, y in points[1:])
        self.commands.append(
            "q "
            f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} RG "
            f"{width:.3f} w [{dash_text}] 0 d "
            + " ".join(path)
            + " S Q"
        )

    def rectangle(
        self,
        x,
        y,
        width,
        height,
        stroke=(0.0, 0.0, 0.0),
        fill=None,
        line_width=0.8,
    ):
        commands = ["q"]
        if fill is not None:
            commands.append(f"{fill[0]:.3f} {fill[1]:.3f} {fill[2]:.3f} rg")
        commands.append(f"{stroke[0]:.3f} {stroke[1]:.3f} {stroke[2]:.3f} RG")
        commands.append(f"{line_width:.3f} w")
        commands.append(f"{x:.3f} {y:.3f} {width:.3f} {height:.3f} re")
        commands.append("B" if fill is not None else "S")
        commands.append("Q")
        self.commands.append(" ".join(commands))

    def circle(
        self,
        x,
        y,
        radius=2.0,
        stroke=(0.0, 0.0, 0.0),
        fill=None,
        line_width=0.6,
    ):
        kappa = 0.5522847498 * radius
        path = (
            f"{x + radius:.3f} {y:.3f} m "
            f"{x + radius:.3f} {y + kappa:.3f} "
            f"{x + kappa:.3f} {y + radius:.3f} "
            f"{x:.3f} {y + radius:.3f} c "
            f"{x - kappa:.3f} {y + radius:.3f} "
            f"{x - radius:.3f} {y + kappa:.3f} "
            f"{x - radius:.3f} {y:.3f} c "
            f"{x - radius:.3f} {y - kappa:.3f} "
            f"{x - kappa:.3f} {y - radius:.3f} "
            f"{x:.3f} {y - radius:.3f} c "
            f"{x + kappa:.3f} {y - radius:.3f} "
            f"{x + radius:.3f} {y - kappa:.3f} "
            f"{x + radius:.3f} {y:.3f} c"
        )
        commands = ["q"]
        if fill is not None:
            commands.append(f"{fill[0]:.3f} {fill[1]:.3f} {fill[2]:.3f} rg")
        commands.append(f"{stroke[0]:.3f} {stroke[1]:.3f} {stroke[2]:.3f} RG")
        commands.append(f"{line_width:.3f} w")
        commands.append(path)
        commands.append("B" if fill is not None else "S")
        commands.append("Q")
        self.commands.append(" ".join(commands))

    def text(
        self,
        x,
        y,
        text,
        size=8,
        font="F1",
        color=(0.0, 0.0, 0.0),
        align="left",
        rotate=0,
    ):
        approximate_width = 0.52 * size * len(text)
        if align == "center":
            x -= approximate_width / 2
        elif align == "right":
            x -= approximate_width
        radians = math.radians(rotate)
        a = math.cos(radians)
        b = math.sin(radians)
        c = -math.sin(radians)
        d = math.cos(radians)
        self.commands.append(
            "BT "
            f"/{font} {size:.3f} Tf "
            f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg "
            f"{a:.5f} {b:.5f} {c:.5f} {d:.5f} {x:.3f} {y:.3f} Tm "
            f"({pdf_escape(text)}) Tj ET"
        )

    def save(self, path: Path):
        content = ("\n".join(self.commands) + "\n").encode("latin-1")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                f"<< /Type /Page /Parent 2 0 R "
                f"/MediaBox [0 0 {self.width:g} {self.height:g}] "
                f"/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> "
                f"/Contents 6 0 R >>"
            ).encode("ascii"),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
            b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n"
            + content
            + b"endstream",
        ]
        data = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for number, body in enumerate(objects, start=1):
            offsets.append(len(data))
            data.extend(f"{number} 0 obj\n".encode("ascii"))
            data.extend(body)
            data.extend(b"\nendobj\n")
        xref = len(data)
        data.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        data.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            data.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        data.extend(
            (
                f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref}\n%%EOF\n"
            ).encode("ascii")
        )
        path.write_bytes(data)


def draw_axes(
    canvas: PdfCanvas,
    x,
    y,
    width,
    height,
    x_limits,
    y_limits,
    x_ticks,
    y_ticks,
    show_x_labels=True,
    show_y_labels=True,
    y_decimals=1,
):
    def map_x(value):
        return x + width * (value - x_limits[0]) / (x_limits[1] - x_limits[0])

    def map_y(value):
        return y + height * (value - y_limits[0]) / (y_limits[1] - y_limits[0])

    for tick in y_ticks:
        location = map_y(tick)
        canvas.line(
            x,
            location,
            x + width,
            location,
            width=0.35,
            color=(0.82, 0.82, 0.82),
        )
        if show_y_labels:
            canvas.text(
                x - 5,
                location - 2.5,
                f"{tick:.{y_decimals}f}",
                7,
                align="right",
            )
    for tick in x_ticks:
        location = map_x(tick)
        canvas.line(
            location,
            y,
            location,
            y + height,
            width=0.25,
            color=(0.90, 0.90, 0.90),
        )
        if show_x_labels:
            canvas.text(location, y - 12, f"{tick:g}", 7, align="center")
    canvas.rectangle(x, y, width, height, line_width=0.7)
    return map_x, map_y


def write_curve_figure(path: Path, summary_horizon: pd.DataFrame) -> None:
    selected = summary_horizon[
        (summary_horizon["n"] == 500)
        & (summary_horizon["nuisance"] == "correct")
        & summary_horizon["pattern"].isin(PATTERN_ORDER)
        & summary_horizon["method"].isin(METHOD_ORDER)
    ]
    canvas = PdfCanvas(612, 500)
    canvas.text(
        306,
        482,
        "Mean treatment-difference curves: n=500, both models correct",
        12,
        "F2",
        align="center",
    )

    styles = {
        "Truth": ((0.0, 0.0, 0.0), (), 1.8),
        "JC-SP": ((0.10, 0.10, 0.10), (6, 2), 1.2),
        "JC-P": ((0.45, 0.45, 0.45), (2, 2), 1.0),
        "CT-SP": ((0.12, 0.35, 0.70), (8, 2, 2, 2), 1.1),
        "IPCW-SP": ((0.75, 0.18, 0.16), (8, 3), 1.1),
        "ORACLE-SP": ((0.10, 0.55, 0.25), (1, 2), 1.2),
    }
    legend_order = ["Truth"] + METHOD_ORDER
    legend_x = [35, 135, 235, 335, 435, 535]
    for name, location in zip(legend_order, legend_x):
        color, dash, width = styles[name]
        canvas.line(location, 457, location + 24, 457, width, color, dash)
        if name not in {"Truth", "JC-P"}:
            canvas.circle(location + 12, 457, 1.5, color, color)
        canvas.text(location + 29, 454, name, 8)

    left = 53
    panel_width = 169
    panel_height = 156
    column_gap = 17
    row_y = {0.2: 257, 0.5: 55}
    y_limits = (-0.25, 0.25)
    for column, pattern in enumerate(PATTERN_ORDER):
        x0 = left + column * (panel_width + column_gap)
        canvas.text(
            x0 + panel_width / 2,
            425,
            PATTERN_LABEL[pattern],
            9,
            "F2",
            align="center",
        )
        for censor in [0.2, 0.5]:
            y0 = row_y[censor]
            map_x, map_y = draw_axes(
                canvas,
                x0,
                y0,
                panel_width,
                panel_height,
                (0.5, 5.0),
                y_limits,
                [0.5, 2.5, 5.0],
                [-0.2, -0.1, 0.0, 0.1, 0.2],
                show_x_labels=censor == 0.5,
                show_y_labels=column == 0,
            )
            group = selected[
                (selected["pattern"] == pattern)
                & (selected["censor_target"] == censor)
            ].sort_values("tau")
            truth = group[group["method"] == "JC-SP"][["tau", "truth"]]
            truth_points = [
                (map_x(tau), map_y(value))
                for tau, value in truth.itertuples(index=False)
            ]
            color, dash, width = styles["Truth"]
            canvas.polyline(truth_points, width, color, dash)
            for method in METHOD_ORDER:
                method_group = group[group["method"] == method]
                points = [
                    (map_x(tau), map_y(value))
                    for tau, value in method_group[
                        ["tau", "mean_estimate"]
                    ].itertuples(index=False)
                ]
                color, dash, width = styles[method]
                canvas.polyline(points, width, color, dash)
                if method not in {"JC-P"}:
                    for point_x, point_y in points:
                        canvas.circle(point_x, point_y, 1.15, color, color)
            if column == 2:
                canvas.text(
                    x0 + panel_width - 4,
                    y0 + panel_height - 12,
                    f"{100 * censor:.0f}% censoring",
                    7,
                    "F2",
                    align="right",
                )

    canvas.text(306, 17, "Time horizon (years)", 9, align="center")
    canvas.text(
        15,
        239,
        "Mean estimated treatment difference",
        9,
        rotate=90,
        align="center",
    )
    canvas.save(path)


def write_bias_figure(path: Path, summary_horizon: pd.DataFrame) -> None:
    selected = summary_horizon[
        (summary_horizon["n"] == 500)
        & (summary_horizon["censor_target"] == 0.5)
        & (summary_horizon["nuisance"] == "correct")
        & summary_horizon["pattern"].isin(PATTERN_ORDER)
        & summary_horizon["method"].isin(
            ["JC-SP", "JC-P", "CT-SP", "IPCW-SP"]
        )
    ]
    canvas = PdfCanvas(612, 260)
    canvas.text(
        306,
        244,
        "Pointwise estimation error: n=500, 50% censoring",
        11,
        "F2",
        align="center",
    )

    styles = {
        "JC-SP": ((0.0, 0.0, 0.0), (), 1.8),
        "JC-P": ((0.40, 0.40, 0.40), (3, 2), 1.1),
        "CT-SP": ((0.12, 0.35, 0.70), (8, 2, 2, 2), 1.2),
        "IPCW-SP": ((0.75, 0.18, 0.16), (8, 3), 1.2),
    }
    legend_x = [94, 206, 318, 430]
    for method, location in zip(styles, legend_x):
        color, dash, width = styles[method]
        canvas.line(location, 224, location + 24, 224, width, color, dash)
        if method != "JC-P":
            canvas.circle(location + 12, 224, 1.4, color, color)
        canvas.text(location + 29, 221, method, 8)

    left = 53
    panel_width = 169
    panel_height = 145
    column_gap = 17
    for column, pattern in enumerate(PATTERN_ORDER):
        x0 = left + column * (panel_width + column_gap)
        y0 = 47
        canvas.text(
            x0 + panel_width / 2,
            205,
            PATTERN_LABEL[pattern],
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
            (-0.06, 0.06),
            [0.5, 2.5, 5.0],
            [-0.06, -0.03, 0.0, 0.03, 0.06],
            show_y_labels=column == 0,
            y_decimals=2,
        )
        canvas.line(
            map_x(0.5),
            map_y(0.0),
            map_x(5.0),
            map_y(0.0),
            width=0.8,
            color=(0.30, 0.30, 0.30),
            dash=(4, 3),
        )
        group = selected[selected["pattern"] == pattern]
        for method in styles:
            method_group = group[group["method"] == method].sort_values("tau")
            points = [
                (map_x(tau), map_y(bias))
                for tau, bias in method_group[["tau", "bias"]].itertuples(
                    index=False
                )
            ]
            color, dash, width = styles[method]
            canvas.polyline(points, width, color, dash)
            if method != "JC-P":
                for point_x, point_y in points:
                    canvas.circle(point_x, point_y, 1.05, color, color)

    canvas.text(306, 17, "Time horizon (years)", 9, align="center")
    canvas.text(
        15,
        119,
        "Mean estimate minus truth (years)",
        9,
        rotate=90,
        align="center",
    )
    canvas.save(path)


def write_gap_figure(path: Path, completion_table: pd.DataFrame) -> None:
    canvas = PdfCanvas(612, 360)
    canvas.text(
        306,
        341,
        "Distribution of the fitted-to-initial moment-gap ratio",
        12,
        "F2",
        align="center",
    )
    panels = [
        ("correct", "Both models correct", 55),
        ("both_miss", "Both models misspecified", 333),
    ]
    y_min, y_max = 0.25, 1.02
    for nuisance, title, x0 in panels:
        y0, width, height = 55, 230, 245
        canvas.text(x0 + width / 2, 314, title, 9, "F2", align="center")
        map_x, map_y = draw_axes(
            canvas,
            x0,
            y0,
            width,
            height,
            (0.5, 4.5),
            (y_min, y_max),
            [],
            [0.25, 0.50, 0.75, 1.00],
            show_x_labels=False,
            show_y_labels=nuisance == "correct",
        )
        canvas.line(
            x0,
            map_y(1.0),
            x0 + width,
            map_y(1.0),
            width=1.0,
            color=(0.35, 0.35, 0.35),
            dash=(5, 3),
        )
        panel = completion_table[
            completion_table["nuisance"] == nuisance
        ].sort_values(["censor_target", "n"])
        for index, row in enumerate(panel.itertuples(), start=1):
            center = map_x(float(index))
            q05 = map_y(row.gap_ratio_q05)
            q25 = map_y(row.gap_ratio_q25)
            median = map_y(row.gap_ratio_median)
            q75 = map_y(row.gap_ratio_q75)
            q95 = map_y(row.gap_ratio_q95)
            fill = (
                (0.75, 0.85, 0.96)
                if int(row.n) == 250
                else (0.48, 0.68, 0.88)
            )
            canvas.line(center, q05, center, q95, 0.8)
            canvas.line(center - 5, q05, center + 5, q05, 0.8)
            canvas.line(center - 5, q95, center + 5, q95, 0.8)
            canvas.rectangle(
                center - 12,
                q25,
                24,
                q75 - q25,
                fill=fill,
                line_width=0.8,
            )
            canvas.line(center - 12, median, center + 12, median, 1.4)
            canvas.circle(center, map_y(row.gap_ratio), 2.0, fill=(0.0, 0.0, 0.0))
            canvas.text(
                center,
                y0 - 14,
                f"{100 * row.censor_target:.0f}%",
                7,
                align="center",
            )
            canvas.text(
                center,
                y0 - 25,
                f"n={int(row.n)}",
                7,
                align="center",
            )
    canvas.text(
        15,
        177,
        "Fitted gap / initial tail-IPCW gap",
        9,
        rotate=90,
        align="center",
    )
    canvas.text(
        306,
        14,
        "Boxes: interquartile range; whiskers: 5th and 95th percentiles; dots: means",
        7.5,
        align="center",
    )
    canvas.save(path)


def write_audit(path: Path, audit, estimation, inference, misspec, completion) -> None:
    manifest = audit["manifest"]
    raw_completion = audit["completion"]
    replication = audit["replication"]
    correct_jcp = audit["summary_horizon"][
        (audit["summary_horizon"]["method"] == "JC-P")
        & (audit["summary_horizon"]["nuisance"] == "correct")
    ]
    jcsp = replication[replication["method"] == "JC-SP"]
    constrained = replication[replication["method"].str.endswith("-SP")]
    lines = [
        "Simulation results preparation audit",
        "",
        f"script version: {manifest.get('script_version')}",
        f"main-run seed: {manifest['config']['seed']}",
        f"cells: {len(audit['status'])}",
        f"valid replications: {int(audit['status']['valid'].sum())}",
        f"failed attempts: {int(audit['status']['failures'].sum())}",
        f"failure-log rows: {len(audit['failures'])}",
        "retained completion rows with shifted attempt identifiers: "
        f"{int(raw_completion['replacement'].sum())}",
        f"horizon rows: {len(audit['horizon'])}",
        f"method rows: {len(replication)}",
        f"completion rows: {len(raw_completion)}",
        "",
        "Raw-to-summary agreement",
        f"maximum estimate-error identity difference: {audit['error_difference']:.3e}",
        f"maximum ISE identity difference: {audit['ise_difference']:.3e}",
        f"maximum horizon-summary difference: {audit['horizon_summary_difference']:.3e}",
        f"maximum method-summary difference: {audit['method_summary_difference']:.3e}",
        f"maximum completion-summary difference: {audit['completion_summary_difference']:.3e}",
        "",
        "Numerical and structural checks",
        f"maximum moment-gap ratio: {raw_completion['gap_ratio'].max():.6f}",
        f"maximum dual residual: {raw_completion['dual_root_norm'].max():.3e}",
        "maximum completed-path violation: "
        f"{raw_completion['completed_path_violation'].max():.3e}",
        "maximum constrained fitted-curve violation: "
        f"{constrained['max_shape_violation'].max():.3e}",
        "JC-SP replications eligible for ordinary interior inference: "
        f"{int((jcsp['active_constraints'] == 0).sum())} of {len(jcsp)}",
        "JC-SP both-models-correct replications eligible for ordinary interior "
        "inference: "
        f"{int(((jcsp['nuisance'] == 'correct') & (jcsp['active_constraints'] == 0)).sum())} "
        f"of {int((jcsp['nuisance'] == 'correct').sum())}",
        "",
        "Inference findings used in the manuscript",
        "mean JC-P pointwise coverage at tau=0.5: "
        f"{correct_jcp.loc[correct_jcp['tau'] == 0.5, 'coverage'].mean():.4f}",
        "mean JC-P pointwise coverage at tau=5: "
        f"{correct_jcp.loc[correct_jcp['tau'] == 5, 'coverage'].mean():.4f}",
        "JC-P simultaneous coverage range: "
        f"{inference['simultaneous_coverage'].min():.3f} to "
        f"{inference['simultaneous_coverage'].max():.3f}",
        "JC-P null global rejection range: "
        f"{inference.loc[inference['pattern'] == 'null', 'global_rejection'].min():.3f} "
        "to "
        f"{inference.loc[inference['pattern'] == 'null', 'global_rejection'].max():.3f}",
        "JC-P alternative global rejection range: "
        f"{inference.loc[inference['pattern'] != 'null', 'global_rejection'].min():.3f} "
        "to "
        f"{inference.loc[inference['pattern'] != 'null', 'global_rejection'].max():.3f}",
        "",
        "Artifacts",
        f"estimation rows: {len(estimation)}",
        f"inference rows: {len(inference)}",
        f"misspecification rows: {len(misspec)}",
        f"completion rows: {len(completion)}",
        "",
        "All preparation checks passed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", default="simulation_main_output")
    args = parser.parse_args()
    root = Path(args.output)

    audit = validate_output(root)
    estimation = derive_estimation_table(audit)
    inference = derive_inference_table(audit)
    misspecification = derive_misspecification_table(audit)
    completion = derive_completion_table(audit)

    estimation.to_csv(root / "paper_estimation_summary.csv", index=False)
    inference.to_csv(root / "paper_inference_summary.csv", index=False)
    misspecification.to_csv(
        root / "paper_misspecification_summary.csv", index=False
    )
    completion.to_csv(root / "paper_completion_summary.csv", index=False)

    write_estimation_table(root / "paper_table_estimation.tex", estimation)
    write_inference_table(root / "paper_table_inference.tex", inference)
    write_misspecification_table(
        root / "paper_table_misspecification.tex", misspecification
    )
    write_completion_table(root / "paper_table_completion.tex", completion)
    write_curve_figure(
        root / "figure_mean_treatment_curves.pdf", audit["summary_horizon"]
    )
    write_bias_figure(
        root / "figure_pointwise_estimation_error.pdf",
        audit["summary_horizon"],
    )
    write_gap_figure(root / "figure_moment_gap_ratio.pdf", completion)
    write_audit(
        root / "paper_results_audit.txt",
        audit,
        estimation,
        inference,
        misspecification,
        completion,
    )

    print(f"output: {root.resolve()}")
    print("validated raw output and saved summaries")
    print("wrote 4 manuscript tables, 3 vector PDF figures, and 4 derived CSV files")
    print("all preparation checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
