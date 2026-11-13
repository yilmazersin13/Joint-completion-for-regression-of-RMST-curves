#!/usr/bin/env python3
"""Shape of the fitted curves at the simulated subjects' own covariate values.

The main simulation imposes the shape restrictions at 18 fixed covariate
combinations. This diagnostic refits the first replications of the 12
correctly specified cells of the main design, with the same seeds, and
evaluates the fitted JC-SP and CT-SP curves at every simulated subject.
No inference is run and no main result is changed.

    python code/simulated_subject_shape_diagnostic.py --reps 25

Dependencies: numpy, scipy, pandas.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_joint_rmst_simulation as sim


def subject_violation(design, gamma):
    slopes = (design @ gamma) @ sim.slope_operator().T
    parts = {
        "below_zero": np.maximum(-slopes, 0.0).max(axis=1),
        "above_one": np.maximum(slopes - 1.0, 0.0).max(axis=1),
        "first_above_one": np.maximum(slopes[:, 0] - 1.0, 0.0),
        "concavity": np.maximum(np.diff(slopes, axis=1), 0.0).max(axis=1),
    }
    largest = np.maximum.reduce(
        [parts["below_zero"], parts["above_one"], parts["concavity"]]
    )
    return largest, parts


def one_replication(cell, seed, config):
    rng = np.random.default_rng(seed)
    data = sim.generate_data(cell["n"], cell["pattern"], cell["censor_lambda"], rng)
    event_fit = sim.fit_event_model(data, True)
    censor_fit = sim.fit_censor_model(data, True)
    design, penalty, basis = sim.build_basis(data, n_tau_basis=6, n_z_basis=5)
    profile_x, _ = sim.make_evaluation_profiles(basis)
    width = config["support_scale"] * cell["n"] ** (-config["support_rate"])
    endpoints = sim.construct_endpoint_responses(data, design, event_fit, censor_fit, width)
    completed, _ = sim.calibrate_completion(data, design, endpoints, config["rho"])
    grid = np.asarray([0.25, 0.5, 1.0, 2.0]) * cell["n"] ** (-0.75)
    fits = sim.fit_all_methods(data, design, penalty, profile_x, endpoints, completed, grid)
    outside = (np.abs(data["z1"]) > 0.75) | (np.abs(data["z2"]) > 0.75)
    row = {}
    for method in ("JC-SP", "CT-SP"):
        v, parts = subject_violation(design, fits[method]["gamma"])
        key = method.replace("-", "_").lower()
        for name, values in parts.items():
            row[key + "_share_" + name] = float(np.mean(values > 1e-8))
        row[key + "_median_among_violators"] = (
            float(np.median(v[v > 1e-8])) if np.any(v > 1e-8) else 0.0
        )
        row[key + "_share_any"] = float(np.mean(v > 1e-8))
        row[key + "_share_above_0.01"] = float(np.mean(v > 0.01))
        row[key + "_share_above_0.05"] = float(np.mean(v > 0.05))
        row[key + "_largest"] = float(v.max())
        row[key + "_share_of_violators_outside_grid"] = (
            float(np.mean(outside[v > 0.01])) if np.any(v > 0.01) else np.nan
        )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--out", default="simulation_subject_shape_output")
    args = parser.parse_args()
    config = {"rho": 0.7, "support_scale": 0.5, "support_rate": 0.60}
    rows = []
    for cell in sim.make_cells("main"):
        if cell["nuisance"] != "correct":
            continue
        done = 0
        attempt = 0
        while done < args.reps and attempt < args.reps + 10:
            attempt += 1
            seed = int(args.seed + cell["cell_id"] * 1_000_000 + attempt)
            try:
                row = one_replication(cell, seed, config)
            except Exception:
                continue
            done += 1
            rows.append({"cell_id": cell["cell_id"], "n": cell["n"],
                         "censor_target": cell["censor_target"],
                         "pattern": cell["pattern"], "seed": seed, **row})
        print(f"cell {cell['cell_id']} done: {done}", flush=True)
    frame = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "subject_shape_replications.csv", index=False)
    numeric = [c for c in frame.columns if c.startswith(("jc_sp", "ct_sp"))]
    summary = frame.groupby(["n", "censor_target"])[numeric].mean().reset_index()
    summary["replications"] = frame.groupby(["n", "censor_target"]).size().values
    summary["jc_sp_largest_max"] = frame.groupby(["n", "censor_target"])["jc_sp_largest"].max().values
    summary.to_csv(out / "subject_shape_summary.csv", index=False)
    overall = {"replications": int(len(frame)), "reps_per_cell": args.reps,
               **{c: float(frame[c].mean()) for c in numeric},
               "jc_sp_largest_max": float(frame["jc_sp_largest"].max())}
    (out / "subject_shape_overall.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(summary.round(4).to_string(index=False))
    print(json.dumps(overall, indent=1))


if __name__ == "__main__":
    main()
