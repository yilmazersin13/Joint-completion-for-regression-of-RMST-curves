#!/usr/bin/env python3
"""Real-data analysis for the joint multi-horizon RMST paper.

The analysis uses the stage C colon cancer trial distributed with the R
package survival.  It compares levamisole plus 5-fluorouracil with observation
for all-cause mortality.  The program applies the same completion parameter,
candidate-grid rate, spline dimensions, penalty grid, and shape restrictions
as the main simulation.  The event-time and censoring models are
finite-dimensional piecewise-exponential models, so the model-adjusted
influence calculation
and the active-boundary multiplier procedure in the manuscript apply.

Run after exporting the documented R data:

    Rscript export_colon_trial_data.R
    python run_colon_real_data.py

Dependencies: numpy, scipy, pandas.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.linalg import block_diag
from scipy.optimize import LinearConstraint, minimize
from scipy.special import logsumexp


DAYS_PER_YEAR = 365.25
TAU = np.arange(0.5, 7.0 + 1e-12, 0.5)
TAU_MAX = 7.0
RHO = 0.7
SUPPORT_SCALE = 0.5
SUPPORT_RATE = 0.60
EVENT_CUTS = np.arange(0.0, 7.0 + 1e-12, 1.0)
CENSOR_CUTS = np.array([0.0, 4.0, 5.0, 5.5, 6.0, 6.5, 7.0])
LAMBDA_MULTIPLIERS = np.array([0.25, 0.5, 1.0, 2.0])
MULTIPLIER_DRAWS = 1999
RANDOM_SEED = 731947
SCRIPT_VERSION = "2026-11-13.1"
METHOD_NAMES = ["JC-SP", "JC-P", "CT-SP", "IPCW-SP", "PV-P", "KM"]


def load_colon_data(path):
    raw = pd.read_csv(path)
    selected = raw.loc[
        (raw["etype"] == 2)
        & raw["rx"].isin(["Obs", "Lev+5FU"])
        & raw["nodes"].notna()
    ].copy()
    selected = selected.sort_values("id").reset_index(drop=True)
    if selected["id"].duplicated().any():
        raise RuntimeError("overall-survival data contain duplicate patient IDs")

    selected["a"] = (selected["rx"] == "Lev+5FU").astype(float)
    selected["time_years"] = selected["time"] / DAYS_PER_YEAR
    selected["y"] = np.minimum(selected["time_years"], TAU_MAX)
    selected["delta"] = (
        (selected["status"] == 1) & (selected["time_years"] <= TAU_MAX)
    ).astype(float)
    selected["censor_event"] = (
        (selected["status"] == 0) & (selected["time_years"] < TAU_MAX)
    ).astype(float)

    age_min, age_max = selected["age"].min(), selected["age"].max()
    log_nodes = np.log1p(selected["nodes"].to_numpy(dtype=float))
    nodes_min, nodes_max = log_nodes.min(), log_nodes.max()
    selected["z_age"] = (
        2.0 * (selected["age"] - age_min) / (age_max - age_min) - 1.0
    )
    selected["z_nodes"] = (
        2.0 * (log_nodes - nodes_min) / (nodes_max - nodes_min) - 1.0
    )
    selected["extent4"] = (selected["extent"] == 4).astype(float)

    linear_names = ["a", "sex", "obstruct", "surg", "extent4"]
    nuisance_names = [
        "z_age",
        "z_nodes",
        "sex",
        "obstruct",
        "surg",
        "extent4",
    ]
    data = {
        "frame": selected,
        "id": selected["id"].to_numpy(dtype=int),
        "a": selected["a"].to_numpy(dtype=float),
        "z1": selected["z_age"].to_numpy(dtype=float),
        "z2": selected["z_nodes"].to_numpy(dtype=float),
        "u": selected[linear_names].to_numpy(dtype=float),
        "linear_names": linear_names,
        "nuisance": selected[nuisance_names].to_numpy(dtype=float),
        "nuisance_names": nuisance_names,
        "y": selected["y"].to_numpy(dtype=float),
        "delta": selected["delta"].to_numpy(dtype=float),
        "censor_event": selected["censor_event"].to_numpy(dtype=float),
        "status": selected["status"].to_numpy(dtype=float),
        "time_years": selected["time_years"].to_numpy(dtype=float),
        "scaling": {
            "age_min": float(age_min),
            "age_max": float(age_max),
            "log_nodes_min": float(nodes_min),
            "log_nodes_max": float(nodes_max),
        },
    }
    return data


def interval_exposure(times, cuts):
    times = np.asarray(times, dtype=float).reshape(-1, 1)
    return np.maximum(
        0.0,
        np.minimum(times, cuts[1:].reshape(1, -1))
        - cuts[:-1].reshape(1, -1),
    )


def log_likelihood_rate_model(eta, x, exposure, event):
    linear = np.clip(x @ eta, -30.0, 30.0)
    rate = np.exp(linear)
    return float(np.sum(event * linear - exposure * rate))


def fit_rate_model(x, exposure, event, subject_id, n_subjects, start):
    eta = np.asarray(start, dtype=float).copy()
    converged = False

    for iteration in range(100):
        linear = np.clip(x @ eta, -30.0, 30.0)
        rate = np.exp(linear)
        residual = event - exposure * rate
        score = x.T @ residual
        information = x.T @ ((exposure * rate)[:, None] * x)
        ridge = 1e-8 * np.eye(x.shape[1])
        step = np.linalg.solve(information + ridge, score)
        if np.max(np.abs(step)) < 1e-9:
            converged = True
            break
        old_value = log_likelihood_rate_model(eta, x, exposure, event)
        accepted = False
        step_size = 1.0
        for _ in range(30):
            candidate = eta + step_size * step
            new_value = log_likelihood_rate_model(
                candidate, x, exposure, event
            )
            if np.isfinite(new_value) and new_value >= old_value - 1e-10:
                eta = candidate
                accepted = True
                break
            step_size *= 0.5
        if not accepted:
            break

    linear = np.clip(x @ eta, -30.0, 30.0)
    rate = np.exp(linear)
    residual = event - exposure * rate
    information = x.T @ ((exposure * rate)[:, None] * x)
    subject_scores = np.zeros((n_subjects, x.shape[1]))
    np.add.at(subject_scores, subject_id, x * residual[:, None])
    if not converged and np.linalg.norm(x.T @ residual, ord=np.inf) < 2e-6:
        converged = True
    condition = np.linalg.cond(information + 1e-8 * np.eye(x.shape[1]))
    if not converged:
        raise RuntimeError(
            "piecewise-exponential event-time or censoring model did not converge"
        )
    if condition > 1e12:
        raise RuntimeError(
            "event-time or censoring information matrix is nearly singular"
        )
    return {
        "eta": eta,
        "scores": subject_scores,
        "J": -information / n_subjects,
        "iterations": iteration + 1,
        "condition": float(condition),
    }


def fit_piecewise_model(data, cuts, event_name, treatment_by_interval):
    n = len(data["y"])
    q = len(cuts) - 1
    rows = []
    exposures = []
    events = []
    subject_ids = []
    event_vector = data[event_name]

    for i in range(n):
        event_interval = int(
            np.searchsorted(cuts[1:-1], data["y"][i], side="right")
        )
        for j in range(q):
            exposure = max(
                0.0,
                min(data["y"][i], cuts[j + 1]) - cuts[j],
            )
            event = float(event_vector[i] == 1.0 and event_interval == j)
            if exposure <= 0.0 and event == 0.0:
                continue
            baseline = np.zeros(q)
            baseline[j] = 1.0
            parts = [baseline]
            if treatment_by_interval:
                parts.append(baseline * data["a"][i])
                parts.append(data["nuisance"][i])
            else:
                parts.append(
                    np.concatenate(
                        [[data["a"][i]], data["nuisance"][i]]
                    )
                )
            rows.append(np.concatenate(parts))
            exposures.append(exposure)
            events.append(event)
            subject_ids.append(i)

    x = np.asarray(rows, dtype=float)
    exposure = np.asarray(exposures, dtype=float)
    event = np.asarray(events, dtype=float)
    subject_id = np.asarray(subject_ids, dtype=int)
    start = np.zeros(x.shape[1])
    for j in range(q):
        mask = x[:, j] == 1.0
        start[j] = math.log(
            (event[mask].sum() + 0.25) / (exposure[mask].sum() + 0.25)
        )
    fit = fit_rate_model(x, exposure, event, subject_id, n, start)
    fit["cuts"] = np.asarray(cuts, dtype=float)
    fit["treatment_by_interval"] = bool(treatment_by_interval)
    fit["event_name"] = event_name
    fit["interval_events"] = [
        int(event[x[:, j] == 1.0].sum()) for j in range(q)
    ]
    return fit


def piecewise_survival_and_gradient(times, i, data, fit):
    times = np.asarray(times, dtype=float)
    cuts = fit["cuts"]
    q = len(cuts) - 1
    lengths = interval_exposure(times, cuts)
    rows = np.zeros((q, len(fit["eta"])))
    for j in range(q):
        rows[j, j] = 1.0
        if fit["treatment_by_interval"]:
            rows[j, q + j] = data["a"][i]
            rows[j, 2 * q :] = data["nuisance"][i]
        else:
            rows[j, q:] = np.concatenate(
                [[data["a"][i]], data["nuisance"][i]]
            )
    rates = np.exp(np.clip(rows @ fit["eta"], -30.0, 30.0))
    cumulative = lengths @ rates
    gradient_hazard = (lengths * rates.reshape(1, -1)) @ rows
    survival = np.exp(-cumulative)
    gradient = -survival[:, None] * gradient_hazard
    return survival, gradient


def make_knots(lower, upper, n_basis, degree=3):
    n_internal = n_basis - degree - 1
    if n_internal > 0:
        internal = np.linspace(lower, upper, n_internal + 2)[1:-1]
    else:
        internal = np.array([])
    return np.concatenate(
        [
            np.repeat(lower, degree + 1),
            internal,
            np.repeat(upper, degree + 1),
        ]
    )


def raw_bspline(x, knots, degree=3):
    return BSpline.design_matrix(
        np.asarray(x, dtype=float), knots, degree, extrapolate=True
    ).toarray()


def reduced_difference_penalty(n_full):
    d2 = np.diff(np.eye(n_full), n=2, axis=0)
    reduced = d2[:, :-1]
    return reduced.T @ reduced


def build_basis(data, n_tau_basis=6, n_z_basis=5):
    degree = 3
    tau_knots = make_knots(0.0, TAU_MAX, n_tau_basis, degree)
    z_knots = make_knots(-1.0, 1.0, n_z_basis, degree)
    tau_full = raw_bspline(TAU, tau_knots, degree)
    tau_zero = raw_bspline(np.array([0.0]), tau_knots, degree)[0]
    b_tau = (tau_full - tau_zero.reshape(1, -1))[:, :-1]

    z1_full = raw_bspline(data["z1"], z_knots, degree)
    z2_full = raw_bspline(data["z2"], z_knots, degree)
    mean_z1 = z1_full.mean(axis=0)
    mean_z2 = z2_full.mean(axis=0)
    b_z1 = (z1_full - mean_z1.reshape(1, -1))[:, :-1]
    b_z2 = (z2_full - mean_z2.reshape(1, -1))[:, :-1]

    n = len(data["a"])
    dt = b_tau.shape[1]
    dz = b_z1.shape[1]
    baseline = np.broadcast_to(b_tau, (n, len(TAU), dt)).copy()
    linear_blocks = [
        baseline * data["u"][:, j, None, None]
        for j in range(data["u"].shape[1])
    ]
    tensor1 = np.einsum("ka,ib->ikab", b_tau, b_z1).reshape(
        n, len(TAU), dt * dz
    )
    tensor2 = np.einsum("ka,ib->ikab", b_tau, b_z2).reshape(
        n, len(TAU), dt * dz
    )
    design = np.concatenate(
        [baseline, *linear_blocks, tensor1, tensor2], axis=2
    )

    p_tau = reduced_difference_penalty(n_tau_basis)
    p_z = reduced_difference_penalty(n_z_basis)
    p_tensor = np.kron(p_tau, np.eye(dz)) + np.kron(
        np.eye(dt), p_z
    )
    penalty = block_diag(
        p_tau,
        *([p_tau] * data["u"].shape[1]),
        p_tensor,
        p_tensor,
    )
    basis = {
        "tau_knots": tau_knots,
        "z_knots": z_knots,
        "degree": degree,
        "mean_z1": mean_z1,
        "mean_z2": mean_z2,
        "b_tau": b_tau,
        "dt": dt,
        "dz": dz,
        "n_linear": data["u"].shape[1],
    }
    return design, penalty, basis


def profile_design(u, z1, z2, basis):
    u = np.asarray(u, dtype=float)
    z1 = np.asarray(z1, dtype=float)
    z2 = np.asarray(z2, dtype=float)
    b_tau = basis["b_tau"]
    z1_full = raw_bspline(z1, basis["z_knots"], basis["degree"])
    z2_full = raw_bspline(z2, basis["z_knots"], basis["degree"])
    b_z1 = (z1_full - basis["mean_z1"].reshape(1, -1))[:, :-1]
    b_z2 = (z2_full - basis["mean_z2"].reshape(1, -1))[:, :-1]
    m = len(z1)
    baseline = np.broadcast_to(
        b_tau, (m, len(TAU), basis["dt"])
    ).copy()
    linear_blocks = [
        baseline * u[:, j, None, None] for j in range(u.shape[1])
    ]
    tensor1 = np.einsum("ka,ib->ikab", b_tau, b_z1).reshape(
        m, len(TAU), basis["dt"] * basis["dz"]
    )
    tensor2 = np.einsum("ka,ib->ikab", b_tau, b_z2).reshape(
        m, len(TAU), basis["dt"] * basis["dz"]
    )
    return np.concatenate(
        [baseline, *linear_blocks, tensor1, tensor2], axis=2
    )


def make_profiles_and_contrasts(data, basis):
    z1_levels = np.quantile(data["z1"], [0.10, 0.50, 0.90])
    z2_levels = np.quantile(data["z2"], [0.10, 0.50, 0.90])
    z1_grid, z2_grid = np.meshgrid(z1_levels, z2_levels, indexing="ij")
    binary = np.unique(data["u"][:, 1:], axis=0)
    profile_u = []
    profile_z1 = []
    profile_z2 = []
    for values in binary:
        for treatment in (0.0, 1.0):
            for z1_value, z2_value in zip(z1_grid.ravel(), z2_grid.ravel()):
                profile_u.append(np.concatenate([[treatment], values]))
                profile_z1.append(z1_value)
                profile_z2.append(z2_value)
    profile_x = profile_design(
        np.asarray(profile_u),
        np.asarray(profile_z1),
        np.asarray(profile_z2),
        basis,
    )

    u0 = data["u"].copy()
    u1 = data["u"].copy()
    u0[:, 0] = 0.0
    u1[:, 0] = 1.0
    x0 = profile_design(u0, data["z1"], data["z2"], basis)
    x1 = profile_design(u1, data["z1"], data["z2"], basis)
    patient_x = np.concatenate([x0, x1], axis=0)
    flat = np.round(patient_x.reshape(len(patient_x), -1), 12)
    _, keep = np.unique(flat, axis=0, return_index=True)
    profile_x = np.concatenate(
        [profile_x, patient_x[np.sort(keep)]], axis=0
    )
    contrast = np.mean(x1 - x0, axis=0)
    mean_curve0 = np.mean(x0, axis=0)
    mean_curve1 = np.mean(x1, axis=0)
    return profile_x, contrast, mean_curve0, mean_curve1


def tail_reference(i, data, event_fit, support_width):
    grid = np.arange(
        support_width, TAU_MAX + support_width / 2.0, support_width
    )
    grid[-1] = TAU_MAX
    support = grid[grid > data["y"][i] + 1e-12]
    if len(support) == 0 or abs(support[-1] - TAU_MAX) > 1e-10:
        support = np.append(support, TAU_MAX)
    times = np.concatenate([[data["y"][i]], support])
    survival, gradient = piecewise_survival_and_gradient(
        times, i, data, event_fit
    )
    denominator = survival[0]
    if denominator < 1e-10:
        raise RuntimeError("estimated conditional event survival violates positivity")
    j_count = len(support)
    q = np.zeros(j_count)
    dq = np.zeros((j_count, len(event_fit["eta"])))
    if j_count > 1:
        numerator = survival[: j_count - 1] - survival[1:j_count]
        d_numerator = gradient[: j_count - 1] - gradient[1:j_count]
        q[: j_count - 1] = numerator / denominator
        dq[: j_count - 1] = (
            d_numerator / denominator
            - q[: j_count - 1, None] * gradient[0] / denominator
        )
    q[-1] = survival[j_count - 1] / denominator
    dq[-1] = (
        gradient[j_count - 1] / denominator
        - q[-1] * gradient[0] / denominator
    )
    if np.min(q) <= 0.0 or abs(q.sum() - 1.0) > 1e-8:
        raise RuntimeError("invalid conditional-tail fractions")
    h_support = np.minimum(support[:, None], TAU.reshape(1, -1))
    return {"support": support, "q0": q, "dq0": dq, "h": h_support}


def construct_endpoint_responses(
    data, design, event_fit, censor_fit, support_width
):
    n, k_count, _ = design.shape
    full_vector = (data["delta"] == 1.0) | (
        data["y"] >= TAU_MAX - 1e-12
    )
    restricted_y = np.minimum(data["y"][:, None], TAU.reshape(1, -1))
    tail_response = np.zeros((n, k_count))
    tail_response[full_vector] = restricted_y[full_vector]
    references = {}

    for i in np.flatnonzero(~full_vector):
        reference = tail_reference(i, data, event_fit, support_width)
        reference["B"] = reference["h"] @ design[i]
        references[int(i)] = reference
        tail_response[i] = reference["q0"] @ reference["h"]

    observed_at_horizon = (
        (data["y"][:, None] >= TAU.reshape(1, -1))
        | (
            (data["delta"][:, None] == 1.0)
            & (data["y"][:, None] < TAU.reshape(1, -1))
        )
    )
    ipcw_response = np.zeros_like(restricted_y)
    censor_gradients = []
    minimum_censor_survival = 1.0
    for i in range(n):
        survival, gradient = piecewise_survival_and_gradient(
            restricted_y[i], i, data, censor_fit
        )
        minimum_censor_survival = min(
            minimum_censor_survival, float(np.min(survival))
        )
        if np.min(survival) < 0.05:
            raise RuntimeError("estimated censoring survival is below 0.05")
        ipcw_response[i] = (
            observed_at_horizon[i] * restricted_y[i] / survival
        )
        censor_gradients.append((survival, gradient))

    return {
        "tail": tail_response,
        "ipcw": ipcw_response,
        "references": references,
        "full_vector": full_vector,
        "phi0": np.einsum("nkp,nk->np", design, tail_response),
        "phi1": np.einsum("nkp,nk->np", design, ipcw_response),
        "censor_gradients": censor_gradients,
        "minimum_censor_survival": minimum_censor_survival,
    }


def dual_components(theta, c_std, m1_std, references, scale, n, rho):
    p = len(theta)
    moment = c_std.copy()
    curvature = np.zeros((p, p))
    log_normalizers = 0.0
    fitted_q = {}
    for i, reference in references.items():
        b_std = reference["B"] / scale.reshape(1, -1)
        log_weights = np.log(reference["q0"]) - b_std @ theta
        log_z = logsumexp(log_weights)
        q = np.exp(log_weights - log_z)
        mean_b = q @ b_std
        centered = b_std - mean_b.reshape(1, -1)
        moment += mean_b / n
        curvature += centered.T @ (q[:, None] * centered) / n
        log_normalizers += log_z
        fitted_q[i] = q
    root = moment - m1_std - rho * theta
    dual = (
        theta @ (c_std - m1_std)
        - log_normalizers / n
        - 0.5 * rho * theta @ theta
    )
    return root, curvature, dual, fitted_q, moment


def calibrate_completion(data, design, endpoints, rho):
    n, _, p = design.shape
    gram = np.einsum("nkp,nkq->pq", design, design) / n
    scale = np.maximum(np.sqrt(np.diag(gram)), 1e-6)
    full = endpoints["full_vector"]
    c_original = endpoints["phi0"][full].sum(axis=0) / n
    c_std = c_original / scale
    m0_std = endpoints["phi0"].mean(axis=0) / scale
    m1_std = endpoints["phi1"].mean(axis=0) / scale
    theta = np.zeros(p)
    converged = False
    _, curvature0, _, _, _ = dual_components(
        theta, c_std, m1_std, endpoints["references"], scale, n, rho
    )

    for iteration in range(80):
        root, curvature, dual, _, _ = dual_components(
            theta, c_std, m1_std, endpoints["references"], scale, n, rho
        )
        if np.linalg.norm(root, ord=np.inf) < 1e-8:
            converged = True
            break
        step = np.linalg.solve(curvature + rho * np.eye(p), root)
        step_size = 1.0
        accepted = False
        for _ in range(30):
            candidate = theta + step_size * step
            _, _, candidate_dual, _, _ = dual_components(
                candidate,
                c_std,
                m1_std,
                endpoints["references"],
                scale,
                n,
                rho,
            )
            if np.isfinite(candidate_dual) and candidate_dual >= dual - 1e-11:
                theta = candidate
                accepted = True
                break
            step_size *= 0.5
        if not accepted:
            break

    root, _, _, fitted_q, fitted_moment = dual_components(
        theta, c_std, m1_std, endpoints["references"], scale, n, rho
    )
    if not converged and np.linalg.norm(root, ord=np.inf) < 2e-7:
        converged = True
    if not converged:
        raise RuntimeError("dual completion did not converge")

    completed = endpoints["tail"].copy()
    kl_values = []
    ess_values = []
    for i, q in fitted_q.items():
        reference = endpoints["references"][i]
        completed[i] = q @ reference["h"]
        kl_values.append(float(np.sum(q * np.log(q / reference["q0"]))))
        ess_values.append(float(np.exp(-np.sum(q * np.log(q)))))
    gap0 = float(np.linalg.norm(m0_std - m1_std))
    gap_fitted = float(np.linalg.norm(fitted_moment - m1_std))
    return completed, {
        "theta": theta,
        "scale": scale,
        "gram": gram,
        "curvature0_std": curvature0,
        "iterations": iteration + 1,
        "root_norm": float(np.linalg.norm(root)),
        "gap0": gap0,
        "gap_fitted": gap_fitted,
        "gap_ratio": gap_fitted / gap0 if gap0 > 0.0 else 0.0,
        "mean_kl": float(np.mean(kl_values)),
        "kl_per_subject": float(np.sum(kl_values) / n),
        "mean_ess": float(np.mean(ess_values)),
        "n_completed": len(fitted_q),
        "rho": float(rho),
    }


def fit_penalized_response(response, design, penalty, lambda_grid):
    n, k_count, p = design.shape
    gram = np.einsum("nkp,nkq->pq", design, design) / n
    moment = np.einsum("nkp,nk->p", design, response) / n
    best = None
    for lam in lambda_grid:
        hessian = gram + lam * penalty + 1e-8 * np.eye(p)
        gamma = np.linalg.solve(hessian, moment)
        residual = response - np.einsum("nkp,p->nk", design, gamma)
        rss = float(np.sum(residual**2))
        edf = float(np.trace(np.linalg.solve(hessian, gram)))
        denominator = max(1.0 - edf / (n * k_count), 1e-6)
        gcv = (rss / (n * k_count)) / denominator**2
        candidate = {
            "gamma": gamma,
            "lambda": float(lam),
            "gcv": float(gcv),
            "gram": gram,
            "hessian": hessian,
        }
        if best is None or candidate["gcv"] < best["gcv"]:
            best = candidate
    return best


def slope_operator():
    operator = np.zeros((len(TAU), len(TAU)))
    for k in range(len(TAU)):
        width = TAU[k] - (TAU[k - 1] if k > 0 else 0.0)
        operator[k, k] = 1.0 / width
        if k > 0:
            operator[k, k - 1] = -1.0 / width
    return operator


def shape_constraint_matrix(profile_x):
    d1 = slope_operator()
    d2 = np.diff(np.eye(len(TAU)), axis=0)
    matrices = []
    lower = []
    upper = []
    for x_g in profile_x:
        slopes = d1 @ x_g
        concavity = d2 @ slopes
        matrices.extend([slopes, concavity])
        lower.extend(
            [np.zeros(len(TAU)), np.full(len(TAU) - 1, -np.inf)]
        )
        upper.extend([np.ones(len(TAU)), np.zeros(len(TAU) - 1)])
    return np.vstack(matrices), np.concatenate(lower), np.concatenate(upper)


def constrain_fit(unconstrained, profile_x, tolerance=1e-8):
    gamma = unconstrained["gamma"]
    matrix, lower, upper = shape_constraint_matrix(profile_x)
    values = matrix @ gamma
    violation = max(
        float(np.max(np.maximum(lower - values, 0.0))),
        float(np.max(np.maximum(values - upper, 0.0))),
    )
    if violation <= tolerance:
        active = int(
            np.sum(np.isfinite(lower) & (np.abs(values - lower) <= 1e-6))
            + np.sum(np.isfinite(upper) & (np.abs(values - upper) <= 1e-6))
        )
        return gamma.copy(), active, 0, violation

    hessian = unconstrained["hessian"]
    target = hessian @ gamma

    def objective(coef):
        return 0.5 * coef @ hessian @ coef - target @ coef

    def gradient(coef):
        return hessian @ coef - target

    result = minimize(
        objective,
        np.zeros_like(gamma),
        jac=gradient,
        method="SLSQP",
        constraints=[LinearConstraint(matrix, lower, upper)],
        options={"ftol": 1e-10, "maxiter": 800, "disp": False},
    )
    fitted_values = matrix @ result.x
    final_violation = max(
        float(np.max(np.maximum(lower - fitted_values, 0.0))),
        float(np.max(np.maximum(fitted_values - upper, 0.0))),
    )
    if not result.success or final_violation > 2e-6:
        raise RuntimeError(
            f"shape quadratic program failed: {result.message}; "
            f"violation={final_violation:.3e}"
        )
    active = int(
        np.sum(
            np.isfinite(lower) & (np.abs(fitted_values - lower) <= 1e-6)
        )
        + np.sum(
            np.isfinite(upper) & (np.abs(fitted_values - upper) <= 1e-6)
        )
    )
    return result.x, active, int(result.nit), final_violation


def patient_curve_check(data, basis, gamma, tolerance=1e-8):
    u0 = data["u"].copy()
    u1 = data["u"].copy()
    u0[:, 0] = 0.0
    u1[:, 0] = 1.0
    worst = 0.0
    count = 0
    for u in (data["u"], u0, u1):
        x = profile_design(u, data["z1"], data["z2"], basis)
        slopes = (x @ gamma) @ slope_operator().T
        violation = np.maximum.reduce(
            [
                np.maximum(-slopes, 0.0).max(axis=1),
                np.maximum(slopes - 1.0, 0.0).max(axis=1),
                np.maximum(np.diff(slopes, axis=1), 0.0).max(axis=1),
            ]
        )
        worst = max(worst, float(violation.max()))
        if u is data["u"]:
            count = int(np.sum(violation > tolerance))
    return {"observed_patients_violating": count, "maximum_violation": worst}


def response_path_violation(response):
    slopes = response @ slope_operator().T
    range_violation = np.maximum(-slopes, 0.0) + np.maximum(
        slopes - 1.0, 0.0
    )
    concavity = np.maximum(np.diff(slopes, axis=1), 0.0)
    return max(
        float(np.max(range_violation)),
        float(np.max(concavity)) if concavity.size else 0.0,
    )


def fit_km_rmst(time, event):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    order = np.argsort(time)
    time = time[order]
    event = event[order]
    event_times = np.unique(time[event == 1.0])
    survival = 1.0
    previous = 0.0
    area = 0.0
    result = np.zeros(len(TAU))
    event_index = 0
    for k, horizon in enumerate(TAU):
        while (
            event_index < len(event_times)
            and event_times[event_index] <= horizon
        ):
            current = event_times[event_index]
            area += survival * (current - previous)
            risk = np.sum(time >= current)
            deaths = np.sum((time == current) & (event == 1.0))
            survival *= 1.0 - deaths / risk
            previous = current
            event_index += 1
        result[k] = area + survival * (horizon - previous)
    return result


def pseudo_rmst(data):
    n = len(data["y"])
    full = fit_km_rmst(data["y"], data["delta"])
    pseudo = np.zeros((n, len(TAU)))
    for i in range(n):
        keep = np.ones(n, dtype=bool)
        keep[i] = False
        leave_one_out = fit_km_rmst(
            data["y"][keep], data["delta"][keep]
        )
        pseudo[i] = n * full - (n - 1) * leave_one_out
    return pseudo


def estimate_jc_influence(
    data,
    design,
    endpoints,
    event_fit,
    censor_fit,
    completion_info,
    jc_fit,
    contrast_design,
):
    n, _, p = design.shape
    h0_terms = np.zeros((n, p, len(event_fit["eta"])))
    h1_terms = np.zeros((n, p, len(censor_fit["eta"])))
    for i, reference in endpoints["references"].items():
        derivative_response = reference["h"].T @ reference["dq0"]
        h0_terms[i] = design[i].T @ derivative_response
    for i in range(n):
        survival, gradient = endpoints["censor_gradients"][i]
        derivative_response = (
            -endpoints["ipcw"][i, :, None]
            * gradient
            / survival[:, None]
        )
        h1_terms[i] = design[i].T @ derivative_response

    h0 = h0_terms.mean(axis=0)
    h1 = h1_terms.mean(axis=0)
    fitted_mean = np.einsum("nkp,p->nk", design, jc_fit["gamma"])
    a_i = np.einsum("nkp,nk->np", design, fitted_mean)
    psi0 = endpoints["phi0"] - a_i
    psi1 = endpoints["phi1"] - a_i
    correction0 = event_fit["scores"] @ np.linalg.solve(
        event_fit["J"].T, h0.T
    )
    correction1 = censor_fit["scores"] @ np.linalg.solve(
        censor_fit["J"].T, h1.T
    )
    xi0 = psi0 - correction0
    xi1 = psi1 - correction1

    k_std = completion_info["curvature0_std"]
    rho = completion_info["rho"]
    l_std = k_std @ np.linalg.inv(k_std + rho * np.eye(p))
    scale = completion_info["scale"]
    l_original = scale[:, None] * l_std / scale[None, :]
    combined = xi0 @ (np.eye(p) - l_original).T + xi1 @ l_original.T
    coefficient_if = combined @ np.linalg.inv(jc_fit["hessian"]).T
    coefficient_if -= coefficient_if.mean(axis=0, keepdims=True)
    curve_if = coefficient_if @ contrast_design.T
    standard_error = curve_if.std(axis=0, ddof=1) / math.sqrt(n)
    return {
        "standard_error": standard_error,
        "coefficient_if": coefficient_if,
        "curve_if": curve_if,
    }


def remove_duplicate_cone_rows(matrix, decimals=10):
    if len(matrix) == 0:
        return np.empty((0, matrix.shape[1]))
    norms = np.linalg.norm(matrix, axis=1)
    normalized = matrix[norms > 1e-12] / norms[norms > 1e-12, None]
    rounded = np.round(normalized, decimals=decimals)
    _, first = np.unique(rounded, axis=0, return_index=True)
    return normalized[np.sort(first)]


def estimate_tangent_cone(
    gamma, profile_x, coefficient_if, kappa, active_tolerance=1e-7
):
    n = coefficient_if.shape[0]
    matrix, lower, upper = shape_constraint_matrix(profile_x)
    values = matrix @ gamma
    constraint_if = coefficient_if @ matrix.T
    constraint_se = constraint_if.std(axis=0, ddof=1) / math.sqrt(n)
    threshold = np.maximum(active_tolerance, kappa * constraint_se)
    lower_slack = values - lower
    upper_slack = upper - values
    selected_lower = np.isfinite(lower) & (lower_slack <= threshold)
    selected_upper = np.isfinite(upper) & (upper_slack <= threshold)
    rows = []
    if np.any(selected_lower):
        rows.append(-matrix[selected_lower])
    if np.any(selected_upper):
        rows.append(matrix[selected_upper])
    cone = (
        remove_duplicate_cone_rows(np.vstack(rows))
        if rows
        else np.empty((0, matrix.shape[1]))
    )
    return {
        "matrix": cone,
        "candidate_rows": int(selected_lower.sum() + selected_upper.sum()),
        "unique_rows": int(len(cone)),
        "exact_active_rows": int(
            np.sum(
                np.isfinite(lower) & (np.abs(lower_slack) <= 1e-6)
            )
            + np.sum(
                np.isfinite(upper) & (np.abs(upper_slack) <= 1e-6)
            )
        ),
        "maximum_threshold": float(np.max(threshold)),
    }


def project_draws_to_tangent_cone(draws, hessian, cone):
    if len(cone) == 0:
        return draws.copy(), {
            "failures": 0,
            "fallbacks": 0,
            "maximum_violation": 0.0,
            "mean_iterations": 0.0,
        }
    hessian_inverse_cone = np.linalg.solve(hessian, cone.T)
    dual_hessian = cone @ hessian_inverse_cone
    dual_hessian = 0.5 * (dual_hessian + dual_hessian.T)
    constraint = LinearConstraint(
        cone, np.full(len(cone), -np.inf), np.zeros(len(cone))
    )
    projected = np.zeros_like(draws)
    failures = 0
    fallbacks = 0
    maximum_violation = 0.0
    iterations = []

    for b, draw in enumerate(draws):
        if np.max(cone @ draw) <= 1e-10:
            projected[b] = draw
            iterations.append(0)
            continue
        dual_linear = cone @ draw

        def dual_objective(multiplier):
            return (
                0.5 * multiplier @ dual_hessian @ multiplier
                - dual_linear @ multiplier
            )

        def dual_gradient(multiplier):
            return dual_hessian @ multiplier - dual_linear

        dual_result = minimize(
            dual_objective,
            np.zeros(len(cone)),
            jac=dual_gradient,
            method="L-BFGS-B",
            bounds=[(0.0, None)] * len(cone),
            options={"ftol": 1e-13, "gtol": 1e-10, "maxiter": 500},
        )
        candidate = draw - hessian_inverse_cone @ dual_result.x
        violation = float(max(np.max(cone @ candidate), 0.0))
        iteration_count = int(dual_result.nit)
        success = dual_result.success
        if not success or violation > 2e-6:
            fallbacks += 1

            def objective(coef):
                difference = coef - draw
                return 0.5 * difference @ hessian @ difference

            def gradient(coef):
                return hessian @ (coef - draw)

            primal_result = minimize(
                objective,
                np.zeros_like(draw),
                jac=gradient,
                method="SLSQP",
                constraints=[constraint],
                options={"ftol": 1e-11, "maxiter": 300, "disp": False},
            )
            candidate = primal_result.x
            violation = float(max(np.max(cone @ candidate), 0.0))
            success = primal_result.success
            iteration_count += int(primal_result.nit)
        maximum_violation = max(maximum_violation, violation)
        iterations.append(iteration_count)
        if not success or violation > 2e-6:
            failures += 1
            projected[b] = np.nan
        else:
            projected[b] = candidate
    return projected, {
        "failures": failures,
        "fallbacks": fallbacks,
        "maximum_violation": maximum_violation,
        "mean_iterations": float(np.mean(iterations)),
    }


def estimate_boundary_inference(
    gamma,
    profile_x,
    contrast_design,
    hessian,
    coefficient_if,
    multiplier_draws,
    rng,
):
    n = coefficient_if.shape[0]
    cone = estimate_tangent_cone(
        gamma,
        profile_x,
        coefficient_if,
        kappa=0.10 * math.sqrt(math.log(n)),
    )
    multipliers = rng.normal(size=(multiplier_draws, n))
    root_draws = multipliers @ coefficient_if / math.sqrt(n)
    projected, projection = project_draws_to_tangent_cone(
        root_draws, hessian, cone["matrix"]
    )
    valid = np.all(np.isfinite(projected), axis=1)
    if valid.sum() < math.ceil(0.95 * multiplier_draws):
        raise RuntimeError("too few valid tangent-cone multiplier projections")
    curve_draws = projected[valid] @ contrast_design.T / math.sqrt(n)
    regular_draws = root_draws[valid] @ contrast_design.T / math.sqrt(n)
    projected_half = np.quantile(np.abs(curve_draws), 0.95, axis=0)
    regular_curve_if = coefficient_if @ contrast_design.T
    regular_se = regular_curve_if.std(axis=0, ddof=1) / math.sqrt(n)
    regular_half = 1.96 * regular_se
    pointwise_half = np.maximum(projected_half, regular_half)

    projected_scale = np.maximum(curve_draws.std(axis=0, ddof=1), 1e-12)
    projected_max = np.max(
        np.abs(curve_draws) / projected_scale.reshape(1, -1), axis=1
    )
    projected_sim = (
        np.quantile(projected_max, 0.95) * projected_scale
    )
    regular_scale = np.maximum(regular_se, 1e-12)
    regular_max = np.max(
        np.abs(regular_draws) / regular_scale.reshape(1, -1), axis=1
    )
    regular_sim = np.quantile(regular_max, 0.95) * regular_scale
    simultaneous_half = np.maximum(projected_sim, regular_sim)
    return {
        "pointwise_half_width": pointwise_half,
        "simultaneous_half_width": simultaneous_half,
        "regular_se": regular_se,
        "valid_draws": int(valid.sum()),
        "cone": {key: value for key, value in cone.items() if key != "matrix"},
        "projection": projection,
    }


def fit_analysis(data, rho=RHO, run_inference=True):
    start_time = time.perf_counter()
    event_fit = fit_piecewise_model(
        data, EVENT_CUTS, "delta", treatment_by_interval=True
    )
    censor_fit = fit_piecewise_model(
        data, CENSOR_CUTS, "censor_event", treatment_by_interval=False
    )
    design, penalty, basis = build_basis(data)
    profile_x, contrast, mean_curve0, mean_curve1 = (
        make_profiles_and_contrasts(data, basis)
    )
    support_width = SUPPORT_SCALE * len(data["y"]) ** (-SUPPORT_RATE)
    endpoints = construct_endpoint_responses(
        data, design, event_fit, censor_fit, support_width
    )
    completed, completion = calibrate_completion(
        data, design, endpoints, rho
    )
    lambda_grid = LAMBDA_MULTIPLIERS * len(data["y"]) ** (-0.75)
    jc = fit_penalized_response(completed, design, penalty, lambda_grid)
    ct = fit_penalized_response(
        endpoints["tail"], design, penalty, lambda_grid
    )
    ipcw = fit_penalized_response(
        endpoints["ipcw"], design, penalty, lambda_grid
    )
    pseudo = pseudo_rmst(data)
    pv = fit_penalized_response(pseudo, design, penalty, lambda_grid)

    fits = {
        "JC-P": {**jc, "active": 0, "qp_iterations": 0},
        "PV-P": {**pv, "active": 0, "qp_iterations": 0},
    }
    for name, fit in [("JC-SP", jc), ("CT-SP", ct), ("IPCW-SP", ipcw)]:
        gamma, active, iterations, violation = constrain_fit(fit, profile_x)
        fits[name] = {
            **fit,
            "gamma": gamma,
            "active": active,
            "qp_iterations": iterations,
            "qp_violation": violation,
        }

    km0 = fit_km_rmst(
        data["y"][data["a"] == 0.0], data["delta"][data["a"] == 0.0]
    )
    km1 = fit_km_rmst(
        data["y"][data["a"] == 1.0], data["delta"][data["a"] == 1.0]
    )
    estimates = {}
    curves0 = {}
    curves1 = {}
    for name, fit in fits.items():
        estimates[name] = contrast @ fit["gamma"]
        curves0[name] = mean_curve0 @ fit["gamma"]
        curves1[name] = mean_curve1 @ fit["gamma"]
    estimates["KM"] = km1 - km0
    curves0["KM"] = km0
    curves1["KM"] = km1

    inference = None
    if run_inference:
        influence = estimate_jc_influence(
            data,
            design,
            endpoints,
            event_fit,
            censor_fit,
            completion,
            fits["JC-P"],
            contrast,
        )
        rng = np.random.default_rng(RANDOM_SEED)
        inference = estimate_boundary_inference(
            fits["JC-SP"]["gamma"],
            profile_x,
            contrast,
            fits["JC-SP"]["hessian"],
            influence["coefficient_if"],
            MULTIPLIER_DRAWS,
            rng,
        )
        inference["interior_standard_error"] = influence["standard_error"]

    diagnostics = {
        "n": len(data["y"]),
        "tau_max": TAU_MAX,
        "support_width": float(support_width),
        "support_maximum_width_bound": float(1.5 * support_width),
        "rho": float(rho),
        "n_profiles": int(len(profile_x)),
        "n_selected_profiles": int(
            18 * len(np.unique(data["u"][:, 1:], axis=0))
        ),
        "patient_curve_check": {
            name: patient_curve_check(data, basis, fits[name]["gamma"])
            for name in ("JC-SP", "CT-SP", "IPCW-SP", "JC-P")
        },
        "n_design_columns": int(design.shape[2]),
        "event_model_iterations": event_fit["iterations"],
        "event_model_condition": event_fit["condition"],
        "event_interval_events": event_fit["interval_events"],
        "censor_model_iterations": censor_fit["iterations"],
        "censor_model_condition": censor_fit["condition"],
        "censor_interval_events": censor_fit["interval_events"],
        "minimum_censor_survival": endpoints["minimum_censor_survival"],
        "completion": {
            key: value
            for key, value in completion.items()
            if key
            not in {"theta", "scale", "gram", "curvature0_std"}
        },
        "completed_path_maximum_violation": response_path_violation(completed),
        "methods": {
            name: {
                "lambda": float(fit["lambda"]),
                "gcv": float(fit["gcv"]),
                "active": int(fit.get("active", 0)),
                "qp_iterations": int(fit.get("qp_iterations", 0)),
                "qp_violation": float(fit.get("qp_violation", np.nan)),
            }
            for name, fit in fits.items()
        },
        "elapsed_seconds": float(time.perf_counter() - start_time),
    }
    return {
        "estimates": estimates,
        "curves0": curves0,
        "curves1": curves1,
        "fits": fits,
        "inference": inference,
        "diagnostics": diagnostics,
    }


def make_data_audit(data):
    frame = data["frame"]
    rows = []
    for treatment, label in [(0.0, "Observation"), (1.0, "Lev+5-FU")]:
        mask = data["a"] == treatment
        group = frame.loc[mask]
        rows.append(
            {
                "group": label,
                "n": int(mask.sum()),
                "deaths_by_7": int(data["delta"][mask].sum()),
                "censored_before_7": int(data["censor_event"][mask].sum()),
                "under_observation_at_7": int(
                    np.sum(data["time_years"][mask] >= TAU_MAX)
                ),
                "median_age": float(group["age"].median()),
                "median_nodes": float(group["nodes"].median()),
            }
        )
    return pd.DataFrame(rows)


def make_at_risk_table(data):
    rows = []
    for horizon in TAU:
        for treatment, label in [(0.0, "Observation"), (1.0, "Lev+5-FU")]:
            mask = data["a"] == treatment
            rows.append(
                {
                    "tau": horizon,
                    "group": label,
                    "at_risk": int(
                        np.sum(data["time_years"][mask] >= horizon)
                    ),
                    "events": int(
                        np.sum(
                            (data["status"][mask] == 1.0)
                            & (data["time_years"][mask] <= horizon)
                        )
                    ),
                    "censored": int(
                        np.sum(
                            (data["status"][mask] == 0.0)
                            & (data["time_years"][mask] < horizon)
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_nonlinear_component_curves(data, basis, gamma):
    dt = basis["dt"]
    dz = basis["dz"]
    first_tensor = (1 + basis["n_linear"]) * dt
    age_gamma = gamma[first_tensor : first_tensor + dt * dz]
    nodes_gamma = gamma[first_tensor + dt * dz : first_tensor + 2 * dt * dz]
    horizon_basis = basis["b_tau"][-1]

    age_min = data["scaling"]["age_min"]
    age_max = data["scaling"]["age_max"]
    age_limits = np.quantile(data["frame"]["age"], [0.05, 0.95])
    age_values = np.linspace(age_limits[0], age_limits[1], 101)
    age_scaled = 2.0 * (age_values - age_min) / (age_max - age_min) - 1.0

    log_nodes_min = data["scaling"]["log_nodes_min"]
    log_nodes_max = data["scaling"]["log_nodes_max"]
    nodes_limits = np.quantile(data["frame"]["nodes"], [0.05, 0.95])
    nodes_values = np.linspace(nodes_limits[0], nodes_limits[1], 101)
    log_nodes = np.log1p(nodes_values)
    nodes_scaled = (
        2.0 * (log_nodes - log_nodes_min)
        / (log_nodes_max - log_nodes_min)
        - 1.0
    )

    rows = []
    for name, values, scaled, mean_basis, coefficients in [
        ("Age", age_values, age_scaled, basis["mean_z1"], age_gamma),
        (
            "Positive nodes",
            nodes_values,
            nodes_scaled,
            basis["mean_z2"],
            nodes_gamma,
        ),
    ]:
        full = raw_bspline(scaled, basis["z_knots"], basis["degree"])
        covariate_basis = (full - mean_basis.reshape(1, -1))[:, :-1]
        tensor = np.einsum("a,ib->iab", horizon_basis, covariate_basis)
        component = tensor.reshape(len(values), dt * dz) @ coefficients
        for value, estimate in zip(values, component):
            rows.append(
                {
                    "covariate": name,
                    "value": float(value),
                    "component_years": float(estimate),
                    "component_months": float(12.0 * estimate),
                }
            )
    return pd.DataFrame(rows)


def write_latex_tables(output, audit, estimates):
    description_lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Analysis sample and seven-year follow-up.}",
        r"\label{tab:colon-description}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (
            r"Group & $n$ & Deaths & Censored & At risk "
            r"& Median age & Median nodes\\"
        ),
        r"\midrule",
    ]
    for row in audit.itertuples(index=False):
        description_lines.append(
            f"{row.group} & {row.n} & {row.deaths_by_7} & "
            f"{row.censored_before_7} & {row.under_observation_at_7} & "
            f"{row.median_age:.0f} & {row.median_nodes:.0f}\\\\"
        )
    description_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\begin{minipage}{0.94\textwidth}\small "
                r"Censored denotes loss to follow-up before seven years. "
                r"At risk denotes subjects whose recorded follow-up extends "
                r"to seven years. Twelve subjects with missing numbers of "
                r"positive lymph nodes were excluded."
                r" Boldface is not used because the rows describe treatment "
                r"groups rather than competing procedures."
                r"\end{minipage}"
            ),
            r"\end{table}",
        ]
    )
    (output / "paper_table_colon_description.tex").write_text(
        "\n".join(description_lines) + "\n", encoding="utf-8"
    )

    selected = estimates.loc[estimates["tau"].isin([2.0, 4.0, 6.0, 7.0])]
    wide = selected.pivot(index="tau", columns="method", values="estimate")
    jc = selected.loc[selected["method"] == "JC-SP"].set_index("tau")
    result_lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Estimated gain in RMST for levamisole plus 5-FU versus observation.}",
        r"\label{tab:colon-results}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (
            r"Horizon & JC--SP (95\% pointwise) & JC--P & CT--SP "
            r"& IPCW--SP & PV--P & KM\\"
        ),
        r"\midrule",
    ]
    for horizon in [2.0, 4.0, 6.0, 7.0]:
        estimate_months = 12.0 * wide.loc[horizon, "JC-SP"]
        lower_months = 12.0 * jc.loc[horizon, "pointwise_lower"]
        upper_months = 12.0 * jc.loc[horizon, "pointwise_upper"]
        result_lines.append(
            f"{horizon:.0f} years & {estimate_months:.2f} "
            f"({lower_months:.2f}, {upper_months:.2f}) & "
            f"{12.0 * wide.loc[horizon, 'JC-P']:.2f} & "
            f"{12.0 * wide.loc[horizon, 'CT-SP']:.2f} & "
            f"{12.0 * wide.loc[horizon, 'IPCW-SP']:.2f} & "
            f"{12.0 * wide.loc[horizon, 'PV-P']:.2f} & "
            f"{12.0 * wide.loc[horizon, 'KM']:.2f}\\\\"
        )
    result_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\begin{minipage}{0.96\textwidth}\small "
                r"Values are months gained within the stated time horizon; "
                r"positive values favour levamisole plus 5-FU. "
                r"JC--SP intervals use the active-boundary multiplier "
                r"envelope. Comparator columns are point estimates. No value "
                r"is bolded because the true effect is unknown in real data."
                r"\end{minipage}"
            ),
            r"\end{table}",
        ]
    )
    (output / "paper_table_colon_results.tex").write_text(
        "\n".join(result_lines) + "\n", encoding="utf-8"
    )


def main():
    input_path = Path("colon_trial_raw.csv")
    output = Path("real_data_output")
    output.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(
            "colon_trial_raw.csv is missing; run export_colon_trial_data.R"
        )

    data = load_colon_data(input_path)
    audit = make_data_audit(data)
    at_risk = make_at_risk_table(data)
    analysis = fit_analysis(data, rho=RHO, run_inference=True)
    inference = analysis["inference"]
    nonlinear_components = make_nonlinear_component_curves(
        data,
        build_basis(data)[2],
        analysis["fits"]["JC-SP"]["gamma"],
    )

    rows = []
    for method in METHOD_NAMES:
        for k, horizon in enumerate(TAU):
            row = {
                "method": method,
                "tau": horizon,
                "estimate": analysis["estimates"][method][k],
                "curve_observation": analysis["curves0"][method][k],
                "curve_treatment": analysis["curves1"][method][k],
                "pointwise_lower": np.nan,
                "pointwise_upper": np.nan,
                "simultaneous_lower": np.nan,
                "simultaneous_upper": np.nan,
            }
            if method == "JC-SP":
                row["pointwise_lower"] = (
                    row["estimate"]
                    - inference["pointwise_half_width"][k]
                )
                row["pointwise_upper"] = (
                    row["estimate"]
                    + inference["pointwise_half_width"][k]
                )
                row["simultaneous_lower"] = (
                    row["estimate"]
                    - inference["simultaneous_half_width"][k]
                )
                row["simultaneous_upper"] = (
                    row["estimate"]
                    + inference["simultaneous_half_width"][k]
                )
            rows.append(row)
    estimates = pd.DataFrame(rows)

    sensitivity_rows = []
    for rho in (0.2, 2.0):
        sensitivity = fit_analysis(data, rho=rho, run_inference=False)
        for k, horizon in enumerate(TAU):
            sensitivity_rows.append(
                {
                    "rho": rho,
                    "tau": horizon,
                    "estimate": sensitivity["estimates"]["JC-SP"][k],
                    "gap_ratio": sensitivity["diagnostics"]["completion"][
                        "gap_ratio"
                    ],
                    "mean_kl": sensitivity["diagnostics"]["completion"][
                        "mean_kl"
                    ],
                }
            )

    audit.to_csv(output / "colon_sample_audit.csv", index=False)
    at_risk.to_csv(output / "colon_at_risk.csv", index=False)
    estimates.to_csv(output / "colon_estimates.csv", index=False)
    nonlinear_components.to_csv(
        output / "colon_nonlinear_components.csv", index=False
    )
    pd.DataFrame(sensitivity_rows).to_csv(
        output / "colon_rho_sensitivity.csv", index=False
    )
    write_latex_tables(output, audit, estimates)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "input_file": str(input_path.resolve()),
        "input_rows": 1858,
        "analysis_rows": len(data["y"]),
        "selection": (
            "etype=2, rx in {Obs, Lev+5FU}, nonmissing nodes"
        ),
        "endpoint": "all-cause mortality",
        "treatment": "Lev+5FU versus Obs",
        "tau": TAU.tolist(),
        "random_seed": RANDOM_SEED,
        "multiplier_draws": MULTIPLIER_DRAWS,
        "scaling": data["scaling"],
        "diagnostics": analysis["diagnostics"],
        "inference": {
            key: value
            for key, value in inference.items()
            if key
            not in {
                "pointwise_half_width",
                "simultaneous_half_width",
                "regular_se",
                "interior_standard_error",
            }
        },
    }
    (output / "colon_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"output: {output.resolve()}")
    print(f"analysis sample: {len(data['y'])}")
    print(
        "completed subjects:",
        analysis["diagnostics"]["completion"]["n_completed"],
    )
    print(
        "minimum estimated censoring survival:",
        f"{analysis['diagnostics']['minimum_censor_survival']:.6f}",
    )
    print(
        "moment-gap ratio:",
        f"{analysis['diagnostics']['completion']['gap_ratio']:.6f}",
    )
    print(
        "maximum completed-path violation:",
        f"{analysis['diagnostics']['completed_path_maximum_violation']:.3e}",
    )
    print(
        "JC-SP active restrictions:",
        analysis["diagnostics"]["methods"]["JC-SP"]["active"],
    )
    print(
        "valid multiplier draws:",
        inference["valid_draws"],
        "of",
        MULTIPLIER_DRAWS,
    )
    print(
        "seven-year JC-SP difference (months):",
        f"{12.0 * analysis['estimates']['JC-SP'][-1]:.3f}",
    )


if __name__ == "__main__":
    main()
