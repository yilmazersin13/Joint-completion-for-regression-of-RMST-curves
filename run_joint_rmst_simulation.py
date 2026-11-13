#!/usr/bin/env python3
"""Simulation study for shape-preserving joint RMST completion.

The program follows the estimator in joint_rmst_manuscript.tex.  It uses one
tail distribution for every subject censored before five years, calibrates the
resulting response vector toward the stacked IPCW score, and fits the same
partially linear P-spline model to all comparison responses.

Main run
--------
    python run_joint_rmst_simulation.py --design main --reps 500 --jobs 4

Run with shape restrictions also at every simulated subject (version 5)
-----------------------------------------------------------------------
    python run_joint_rmst_simulation.py --design main_b --reps 500 --jobs 4 \
        --support-rate 0.60 --support-scale 0.5 --boundary-inference \
        --subject-constraints --out simulation_subject_output

Small implementation check
--------------------------
    python run_joint_rmst_simulation.py --pilot

The main output directory contains raw replication results, failure logs,
summary tables, calibrated censoring rates, and a JSON run manifest.  Failed
replications are recorded and replaced until the requested number of valid
replications is reached or the declared attempt limit is exhausted.

Dependencies: numpy, scipy, pandas.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from scipy.linalg import block_diag, cholesky, solve_triangular
from scipy.optimize import LinearConstraint, brentq, minimize, nnls
from scipy.special import logsumexp


TAU = np.arange(0.5, 5.0 + 1e-12, 0.5)
TAU_MAX = 5.0
INTERVAL_STARTS = np.array([0.0, 1.5, 3.0])
INTERVAL_ENDS = np.array([1.5, 3.0, 5.0])
BASE_HAZARDS = np.array([0.15, 0.10, 0.08])

TREATMENT_PATTERNS = {
    "null": np.array([0.0, 0.0, 0.0]),
    "proportional": np.log(np.array([0.75, 0.75, 0.75])),
    "delayed": np.log(np.array([1.0, 0.65, 0.65])),
    "crossing": np.log(np.array([1.40, 0.65, 0.65])),
}

NUISANCE_CONDITIONS = {
    "correct": (True, True),
    "event_miss": (False, True),
    "censor_miss": (True, False),
    "both_miss": (False, False),
    "event_omit": (False, True),
}

METHOD_NAMES = ["JC-SP", "JC-P", "CT-SP", "IPCW-SP", "ORACLE-SP"]
SCRIPT_VERSION = "2026-11-13.1"


def interval_lengths(times):
    """Exposure time in the three event-hazard intervals."""
    t = np.asarray(times, dtype=float).reshape(-1, 1)
    return np.maximum(
        0.0,
        np.minimum(t, INTERVAL_ENDS.reshape(1, -1))
        - INTERVAL_STARTS.reshape(1, -1),
    )


def nonlinear_predictor(z1, z2):
    return 0.6 * (np.asarray(z1) ** 2 - 1.0 / 3.0) + 0.4 * np.sin(
        np.pi * np.asarray(z2)
    )


def true_event_rates(a, z1, z2, pattern):
    psi = TREATMENT_PATTERNS[pattern]
    g = nonlinear_predictor(z1, z2)
    return BASE_HAZARDS.reshape(1, -1) * np.exp(
        g.reshape(-1, 1) + np.asarray(a).reshape(-1, 1) * psi.reshape(1, -1)
    )


def simulate_event_times(a, z1, z2, pattern, rng):
    """Generate piecewise-exponential event times by inverse cumulative hazard."""
    rates = true_event_rates(a, z1, z2, pattern)
    remaining = rng.exponential(size=len(a))
    event_time = np.full(len(a), np.inf)
    unresolved = np.ones(len(a), dtype=bool)

    for j in range(3):
        if j < 2:
            width = INTERVAL_ENDS[j] - INTERVAL_STARTS[j]
            capacity = rates[:, j] * width
            hit = unresolved & (remaining <= capacity)
            event_time[hit] = INTERVAL_STARTS[j] + remaining[hit] / rates[hit, j]
            remaining[unresolved & ~hit] -= capacity[unresolved & ~hit]
            unresolved[hit] = False
        else:
            event_time[unresolved] = (
                INTERVAL_STARTS[j] + remaining[unresolved] / rates[unresolved, j]
            )
            unresolved[:] = False
    return event_time


def generate_data(n, pattern, censor_lambda, rng):
    a = rng.binomial(1, 0.5, size=n).astype(float)
    z1 = rng.uniform(-1.0, 1.0, size=n)
    z2 = rng.uniform(-1.0, 1.0, size=n)
    event_time = simulate_event_times(a, z1, z2, pattern, rng)
    censor_rate = censor_lambda * np.exp(0.25 * z1 - 0.20 * z2 + 0.15 * a)
    censor_time = rng.exponential(1.0 / censor_rate)
    y = np.minimum(np.minimum(event_time, censor_time), TAU_MAX)
    delta = ((event_time <= censor_time) & (event_time <= TAU_MAX)).astype(float)
    censor_event = ((censor_time < event_time) & (censor_time < TAU_MAX)).astype(float)
    return {
        "a": a,
        "z1": z1,
        "z2": z2,
        "event_time": event_time,
        "censor_time": censor_time,
        "y": y,
        "delta": delta,
        "censor_event": censor_event,
    }


def gauss_legendre_profiles(order=15):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    z1, z2 = np.meshgrid(nodes, nodes, indexing="ij")
    w1, w2 = np.meshgrid(weights / 2.0, weights / 2.0, indexing="ij")
    return z1.ravel(), z2.ravel(), (w1 * w2).ravel()


def expected_censoring_rate(censor_lambda, pattern, quadrature_order=15):
    z1, z2, weights = gauss_legendre_profiles(quadrature_order)
    total = 0.0
    for treatment in (0.0, 1.0):
        a = np.full_like(z1, treatment)
        event_rates = true_event_rates(a, z1, z2, pattern)
        censor_rates = censor_lambda * np.exp(0.25 * z1 - 0.20 * z2 + 0.15 * a)
        probabilities = np.zeros_like(z1)
        joint_survival = np.ones_like(z1)
        for j in range(3):
            width = INTERVAL_ENDS[j] - INTERVAL_STARTS[j]
            total_rate = event_rates[:, j] + censor_rates
            probabilities += (
                joint_survival
                * censor_rates
                / total_rate
                * (1.0 - np.exp(-total_rate * width))
            )
            joint_survival *= np.exp(-total_rate * width)
        total += 0.5 * np.sum(weights * probabilities)
    return float(total)


def calibrate_censoring_lambda(target, pattern):
    objective = lambda log_rate: expected_censoring_rate(
        math.exp(log_rate), pattern
    ) - target
    root = brentq(objective, -8.0, 2.5, xtol=1e-11)
    value = math.exp(root)
    return value, expected_censoring_rate(value, pattern)


def true_rmst_for_profiles(a, z1, z2, pattern, tau=TAU):
    rates = true_event_rates(a, z1, z2, pattern)
    answer = np.zeros((len(a), len(tau)))
    for k, horizon in enumerate(tau):
        survival_at_start = np.ones(len(a))
        integral = np.zeros(len(a))
        for j in range(3):
            width = max(
                0.0,
                min(horizon, INTERVAL_ENDS[j]) - INTERVAL_STARTS[j],
            )
            if width > 0.0:
                rate = rates[:, j]
                integral += survival_at_start * (1.0 - np.exp(-rate * width)) / rate
                survival_at_start *= np.exp(-rate * width)
            if horizon <= INTERVAL_ENDS[j]:
                break
        answer[:, k] = integral
    return answer


def marginal_treatment_truth(pattern, tau=TAU, quadrature_order=15):
    z1, z2, weights = gauss_legendre_profiles(quadrature_order)
    rmst0 = true_rmst_for_profiles(np.zeros_like(z1), z1, z2, pattern, tau)
    rmst1 = true_rmst_for_profiles(np.ones_like(z1), z1, z2, pattern, tau)
    return np.sum(weights[:, None] * (rmst1 - rmst0), axis=0)


def log_likelihood_rate_model(eta, x, exposure, event):
    linear = np.clip(x @ eta, -30.0, 30.0)
    rate = np.exp(linear)
    return float(np.sum(event * linear - exposure * rate))


def fit_rate_model(x, exposure, event, subject_id, n_subjects, start=None):
    """Fit a multiplicative rate model and return subject scores and J."""
    p = x.shape[1]
    if start is None:
        start = np.zeros(p)
        start[0] = math.log((event.sum() + 0.25) / (exposure.sum() + 0.25))
    eta = np.asarray(start, dtype=float).copy()
    converged = False

    for iteration in range(80):
        linear = np.clip(x @ eta, -30.0, 30.0)
        rate = np.exp(linear)
        residual = event - exposure * rate
        score = x.T @ residual
        information = x.T @ ((exposure * rate)[:, None] * x)
        ridge = 1e-8 * np.eye(p)
        step = np.linalg.solve(information + ridge, score)
        if np.max(np.abs(step)) < 1e-9:
            converged = True
            break
        old_value = log_likelihood_rate_model(eta, x, exposure, event)
        accepted = False
        step_size = 1.0
        for _ in range(25):
            candidate = eta + step_size * step
            new_value = log_likelihood_rate_model(candidate, x, exposure, event)
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
    subject_scores = np.zeros((n_subjects, p))
    np.add.at(subject_scores, subject_id, x * residual[:, None])
    j_matrix = -information / n_subjects
    if not converged and np.linalg.norm(x.T @ residual, ord=np.inf) < 1e-6:
        converged = True
    if not converged:
        raise RuntimeError("event-time or censoring rate model did not converge")
    if np.linalg.cond(information + 1e-8 * np.eye(p)) > 1e12:
        raise RuntimeError(
            "event-time or censoring information matrix is nearly singular"
        )
    return {
        "eta": eta,
        "scores": subject_scores,
        "J": j_matrix,
        "iterations": iteration + 1,
    }


def fit_event_model(data, correct, omit_treatment=False):
    y = data["y"]
    delta = data["delta"]
    a = data["a"]
    z1 = data["z1"]
    z2 = data["z2"]
    n = len(y)

    if not correct:
        if omit_treatment:
            x = np.column_stack([np.ones(n), z1, z2])
        else:
            x = np.column_stack([np.ones(n), a, z1, z2])
        fit = fit_rate_model(x, y, delta, np.arange(n), n)
        fit["correct"] = False
        fit["omit_treatment"] = bool(omit_treatment)
        return fit

    rows = []
    exposure = []
    events = []
    subject_ids = []
    for i in range(n):
        event_interval = int(np.searchsorted(INTERVAL_ENDS[:-1], y[i], side="right"))
        for j in range(3):
            duration = max(0.0, min(y[i], INTERVAL_ENDS[j]) - INTERVAL_STARTS[j])
            event_j = float(delta[i] == 1.0 and event_interval == j)
            if duration <= 0.0 and event_j == 0.0:
                continue
            row = np.zeros(8)
            row[j] = 1.0
            row[3 + j] = a[i]
            row[6] = z1[i] ** 2 - 1.0 / 3.0
            row[7] = math.sin(math.pi * z2[i])
            rows.append(row)
            exposure.append(duration)
            events.append(event_j)
            subject_ids.append(i)
    x = np.asarray(rows)
    exposure = np.asarray(exposure)
    events = np.asarray(events)
    start = np.zeros(8)
    global_rate = (events.sum() + 0.25) / (exposure.sum() + 0.25)
    start[:3] = math.log(global_rate)
    fit = fit_rate_model(
        x, exposure, events, np.asarray(subject_ids, dtype=int), n, start=start
    )
    fit["correct"] = True
    return fit


def fit_censor_model(data, correct):
    n = len(data["y"])
    if correct:
        x = np.column_stack(
            [np.ones(n), data["z1"], data["z2"], data["a"]]
        )
    else:
        x = np.column_stack([np.ones(n), data["z1"]])
    fit = fit_rate_model(
        x,
        data["y"],
        data["censor_event"],
        np.arange(n),
        n,
    )
    fit["correct"] = bool(correct)
    return fit


def event_survival_and_gradient(times, a, z1, z2, fit):
    times = np.asarray(times, dtype=float)
    eta = fit["eta"]
    if fit["correct"]:
        lengths = interval_lengths(times)
        x_interval = np.zeros((3, 8))
        for j in range(3):
            x_interval[j, j] = 1.0
            x_interval[j, 3 + j] = a
            x_interval[j, 6] = z1**2 - 1.0 / 3.0
            x_interval[j, 7] = math.sin(math.pi * z2)
        rates = np.exp(np.clip(x_interval @ eta, -30.0, 30.0))
        cumulative = lengths @ rates
        gradient_hazard = (lengths * rates.reshape(1, -1)) @ x_interval
    else:
        if fit.get("omit_treatment", False):
            x_row = np.array([1.0, z1, z2])
        else:
            x_row = np.array([1.0, a, z1, z2])
        rate = math.exp(float(np.clip(x_row @ eta, -30.0, 30.0)))
        cumulative = times * rate
        gradient_hazard = times[:, None] * rate * x_row.reshape(1, -1)
    survival = np.exp(-cumulative)
    gradient = -survival[:, None] * gradient_hazard
    return survival, gradient


def censor_survival_and_gradient(times, a, z1, z2, fit):
    times = np.asarray(times, dtype=float)
    if fit["correct"]:
        x_row = np.array([1.0, z1, z2, a])
    else:
        x_row = np.array([1.0, z1])
    rate = math.exp(float(np.clip(x_row @ fit["eta"], -30.0, 30.0)))
    survival = np.exp(-times * rate)
    gradient = -survival[:, None] * times[:, None] * rate * x_row.reshape(1, -1)
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


def build_basis(data, tau=TAU, n_tau_basis=6, n_z_basis=5):
    degree = 3
    tau_knots = make_knots(0.0, TAU_MAX, n_tau_basis, degree)
    z_knots = make_knots(-1.0, 1.0, n_z_basis, degree)
    tau_full = raw_bspline(tau, tau_knots, degree)
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
    baseline = np.broadcast_to(b_tau, (n, len(tau), dt)).copy()
    treatment = baseline * data["a"][:, None, None]
    tensor1 = np.einsum("ka,ib->ikab", b_tau, b_z1).reshape(n, len(tau), dt * dz)
    tensor2 = np.einsum("ka,ib->ikab", b_tau, b_z2).reshape(n, len(tau), dt * dz)
    design = np.concatenate([baseline, treatment, tensor1, tensor2], axis=2)

    p_tau = reduced_difference_penalty(n_tau_basis)
    p_z = reduced_difference_penalty(n_z_basis)
    p_tensor = np.kron(p_tau, np.eye(dz)) + np.kron(np.eye(dt), p_z)
    penalty = block_diag(p_tau, p_tau, p_tensor, p_tensor)

    basis = {
        "tau_knots": tau_knots,
        "z_knots": z_knots,
        "degree": degree,
        "mean_z1": mean_z1,
        "mean_z2": mean_z2,
        "b_tau": b_tau,
        "dt": dt,
        "dz": dz,
    }
    return design, penalty, basis


def profile_design(a, z1, z2, basis, tau=TAU):
    a = np.asarray(a, dtype=float)
    z1 = np.asarray(z1, dtype=float)
    z2 = np.asarray(z2, dtype=float)
    b_tau = basis["b_tau"]
    z1_full = raw_bspline(z1, basis["z_knots"], basis["degree"])
    z2_full = raw_bspline(z2, basis["z_knots"], basis["degree"])
    b_z1 = (z1_full - basis["mean_z1"].reshape(1, -1))[:, :-1]
    b_z2 = (z2_full - basis["mean_z2"].reshape(1, -1))[:, :-1]
    m = len(a)
    baseline = np.broadcast_to(b_tau, (m, len(tau), basis["dt"])).copy()
    treatment = baseline * a[:, None, None]
    tensor1 = np.einsum("ka,ib->ikab", b_tau, b_z1).reshape(
        m, len(tau), basis["dt"] * basis["dz"]
    )
    tensor2 = np.einsum("ka,ib->ikab", b_tau, b_z2).reshape(
        m, len(tau), basis["dt"] * basis["dz"]
    )
    return np.concatenate([baseline, treatment, tensor1, tensor2], axis=2)


def tail_reference(y, a, z1, z2, event_fit, support_width):
    grid = np.arange(support_width, TAU_MAX + support_width / 2.0, support_width)
    grid[-1] = TAU_MAX
    support = grid[grid > y + 1e-12]
    if len(support) == 0 or abs(support[-1] - TAU_MAX) > 1e-10:
        support = np.append(support, TAU_MAX)
    times = np.concatenate([[y], support])
    survival, gradient = event_survival_and_gradient(times, a, z1, z2, event_fit)
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


def construct_endpoint_responses(data, design, event_fit, censor_fit, support_width):
    n, k_count, p = design.shape
    full_vector = (data["delta"] == 1.0) | (data["y"] >= TAU_MAX - 1e-12)
    restricted_y = np.minimum(data["y"][:, None], TAU.reshape(1, -1))
    tail_response = np.zeros((n, k_count))
    tail_response[full_vector] = restricted_y[full_vector]
    references = {}

    for i in np.flatnonzero(~full_vector):
        reference = tail_reference(
            data["y"][i],
            data["a"][i],
            data["z1"][i],
            data["z2"][i],
            event_fit,
            support_width,
        )
        reference["B"] = reference["h"] @ design[i]
        references[int(i)] = reference
        tail_response[i] = reference["q0"] @ reference["h"]

    u = restricted_y
    observed_at_horizon = (
        (data["y"][:, None] >= TAU.reshape(1, -1))
        | (
            (data["delta"][:, None] == 1.0)
            & (data["y"][:, None] < TAU.reshape(1, -1))
        )
    )
    ipcw_response = np.zeros_like(u)
    censor_gradients = []
    for i in range(n):
        survival, gradient = censor_survival_and_gradient(
            u[i], data["a"][i], data["z1"][i], data["z2"][i], censor_fit
        )
        if np.min(survival) < 0.02:
            raise RuntimeError("estimated censoring survival is below 0.02")
        ipcw_response[i] = observed_at_horizon[i] * u[i] / survival
        censor_gradients.append((survival, gradient))

    phi0 = np.einsum("nkp,nk->np", design, tail_response)
    phi1 = np.einsum("nkp,nk->np", design, ipcw_response)
    return {
        "tail": tail_response,
        "ipcw": ipcw_response,
        "references": references,
        "full_vector": full_vector,
        "phi0": phi0,
        "phi1": phi1,
        "censor_gradients": censor_gradients,
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
    m0_original = endpoints["phi0"].mean(axis=0)
    m1_original = endpoints["phi1"].mean(axis=0)
    m0_std = m0_original / scale
    m1_std = m1_original / scale
    theta = np.zeros(p)
    converged = False

    _, curvature0, _, _, _ = dual_components(
        theta, c_std, m1_std, endpoints["references"], scale, n, rho
    )

    for iteration in range(60):
        root, curvature, dual, _, _ = dual_components(
            theta, c_std, m1_std, endpoints["references"], scale, n, rho
        )
        if np.linalg.norm(root, ord=np.inf) < 1e-8:
            converged = True
            break
        step = np.linalg.solve(curvature + rho * np.eye(p), root)
        step_size = 1.0
        accepted = False
        for _ in range(25):
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

    root, curvature, _, fitted_q, fitted_moment = dual_components(
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
        "mean_kl": float(np.mean(kl_values)) if kl_values else 0.0,
        "kl_per_subject": float(np.sum(kl_values) / n),
        "mean_ess": float(np.mean(ess_values)) if ess_values else 1.0,
        "n_completed": len(fitted_q),
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


def slope_operator(tau=TAU):
    k_count = len(tau)
    operator = np.zeros((k_count, k_count))
    for k in range(k_count):
        width = tau[k] - (tau[k - 1] if k > 0 else 0.0)
        operator[k, k] = 1.0 / width
        if k > 0:
            operator[k, k - 1] = -1.0 / width
    return operator


def shape_constraint_matrix(profile_x, tau=TAU):
    d1 = slope_operator(tau)
    d2 = np.diff(np.eye(len(tau)), axis=0)
    matrices = []
    lower = []
    upper = []
    for x_g in profile_x:
        slopes = d1 @ x_g
        concavity = d2 @ slopes
        matrices.extend([slopes, concavity])
        lower.extend([np.zeros(len(tau)), np.full(len(tau) - 1, -np.inf)])
        upper.extend([np.ones(len(tau)), np.zeros(len(tau) - 1)])
    return np.vstack(matrices), np.concatenate(lower), np.concatenate(upper)


def least_distance_solution(inequality, bound, hessian, center):
    """Minimize 0.5 (c-center)'H(c-center) subject to inequality @ c <= bound.

    Least-distance programming of Lawson and Hanson, solved by nonnegative
    least squares. Returns the minimizer and the number of positive weights.
    """
    root = cholesky(hessian, lower=False)
    scaled = -solve_triangular(root, inequality.T, trans="T", lower=False)
    offset = inequality @ center - bound
    stacked = np.vstack([scaled, offset.reshape(1, -1)])
    target = np.zeros(stacked.shape[0])
    target[-1] = 1.0
    weights, _ = nnls(stacked, target, maxiter=50 * stacked.shape[0])
    residual = stacked @ weights - target
    if abs(residual[-1]) < 1e-12:
        raise RuntimeError("least-distance program is infeasible")
    step = -residual[:-1] / residual[-1]
    solution = center + solve_triangular(root, step, lower=False)
    return solution, int(np.sum(weights > 0.0))


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
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    try:
        candidate, positive = least_distance_solution(
            np.vstack([-matrix[finite_lower], matrix[finite_upper]]),
            np.concatenate([-lower[finite_lower], upper[finite_upper]]),
            hessian,
            gamma,
        )
        fitted_values = matrix @ candidate
        final_violation = max(
            float(np.max(np.maximum(lower - fitted_values, 0.0))),
            float(np.max(np.maximum(fitted_values - upper, 0.0))),
        )
        if np.all(np.isfinite(candidate)) and final_violation <= 2e-6:
            active = int(
                np.sum(finite_lower & (np.abs(fitted_values - lower) <= 1e-6))
                + np.sum(finite_upper & (np.abs(fitted_values - upper) <= 1e-6))
            )
            return candidate, active, positive, final_violation
    except Exception:
        pass

    target = hessian @ gamma

    def objective(coef):
        return 0.5 * coef @ hessian @ coef - target @ coef

    def gradient(coef):
        return hessian @ coef - target

    constraint = LinearConstraint(matrix, lower, upper)
    result = minimize(
        objective,
        np.zeros_like(gamma),
        jac=gradient,
        method="SLSQP",
        constraints=[constraint],
        options={"ftol": 1e-10, "maxiter": 600, "disp": False},
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
        np.sum(np.isfinite(lower) & (np.abs(fitted_values - lower) <= 1e-6))
        + np.sum(np.isfinite(upper) & (np.abs(fitted_values - upper) <= 1e-6))
    )
    return result.x, active, int(result.nit), final_violation


def shape_metrics(gamma, profile_x, tau=TAU):
    curves = np.einsum("mkp,p->mk", profile_x, gamma)
    slopes = curves @ slope_operator(tau).T
    range_violation = np.maximum(-slopes, 0.0) + np.maximum(slopes - 1.0, 0.0)
    concavity_violation = np.maximum(np.diff(slopes, axis=1), 0.0)
    maximum = max(
        float(np.max(range_violation)),
        float(np.max(concavity_violation)) if concavity_violation.size else 0.0,
    )
    return {
        "v_range": float(np.mean(range_violation)),
        "v_concave": float(np.mean(concavity_violation)),
        "any_shape_violation": int(maximum > 1e-8),
        "max_shape_violation": maximum,
    }


def response_path_metrics(response, tau=TAU):
    slopes = response @ slope_operator(tau).T
    range_violation = np.maximum(-slopes, 0.0) + np.maximum(slopes - 1.0, 0.0)
    concavity_violation = np.maximum(np.diff(slopes, axis=1), 0.0)
    maximum = max(
        float(np.max(range_violation)),
        float(np.max(concavity_violation)) if concavity_violation.size else 0.0,
    )
    return maximum


def estimate_jc_influence(
    data,
    design,
    endpoints,
    event_fit,
    censor_fit,
    completion_info,
    jc_fit,
    contrast_design,
    multiplier_draws,
    rng,
):
    """Model-adjusted fixed-basis influence estimate for interior JC-P."""
    n, _, p = design.shape
    d_event = len(event_fit["eta"])
    d_censor = len(censor_fit["eta"])
    h0_terms = np.zeros((n, p, d_event))
    h1_terms = np.zeros((n, p, d_censor))

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
    correction0 = event_fit["scores"] @ np.linalg.solve(event_fit["J"].T, h0.T)
    correction1 = censor_fit["scores"] @ np.linalg.solve(censor_fit["J"].T, h1.T)
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

    critical = np.nan
    if multiplier_draws > 0 and np.all(standard_error > 0.0):
        multipliers = rng.normal(size=(multiplier_draws, n))
        perturbations = multipliers @ curve_if / n
        maxima = np.max(np.abs(perturbations / standard_error.reshape(1, -1)), axis=1)
        critical = float(np.quantile(maxima, 0.95))
    return {
        "standard_error": standard_error,
        "simultaneous_critical": critical,
        "coefficient_if": coefficient_if,
        "curve_if": curve_if,
    }


def remove_duplicate_cone_rows(matrix, decimals=10):
    """Remove repeated oriented halfspaces without changing their direction."""
    if len(matrix) == 0:
        return np.empty((0, matrix.shape[1]))
    norms = np.linalg.norm(matrix, axis=1)
    keep = norms > 1e-12
    normalized = matrix[keep] / norms[keep, None]
    rounded = np.round(normalized, decimals=decimals)
    _, first = np.unique(rounded, axis=0, return_index=True)
    return normalized[np.sort(first)]


def estimate_tangent_cone(
    gamma,
    profile_x,
    coefficient_if,
    kappa,
    active_tolerance=1e-7,
):
    """Estimate active rows and write the tangent cone as C u <= 0.

    For a lower restriction M_j gamma >= l_j, a tangent direction satisfies
    -M_j u <= 0.  For an upper restriction M_j gamma <= u_j, it satisfies
    M_j u <= 0.  A row is retained when its fitted slack is no larger than
    kappa times its estimated standard error.  The slowly increasing default
    kappa=0.1*sqrt(log(n)) is supplied by the caller.
    """
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
    exact_active = int(
        np.sum(np.isfinite(lower) & (np.abs(lower_slack) <= 1e-6))
        + np.sum(np.isfinite(upper) & (np.abs(upper_slack) <= 1e-6))
    )
    return {
        "matrix": cone,
        "candidate_rows": int(selected_lower.sum() + selected_upper.sum()),
        "unique_rows": int(len(cone)),
        "exact_active_rows": exact_active,
        "maximum_threshold": float(np.max(threshold)),
    }


def project_draws_to_tangent_cone(draws, hessian, cone):
    """Project root-n Gaussian coefficient draws in the Hessian metric."""
    draws = np.asarray(draws, dtype=float)
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
        cone,
        np.full(len(cone), -np.inf),
        np.zeros(len(cone)),
    )
    projected = np.zeros_like(draws)
    failures = 0
    fallbacks = 0
    maximum_violation = 0.0
    iterations = []
    root = cholesky(hessian, lower=False)
    scaled = -solve_triangular(root, cone.T, trans="T", lower=False)
    unit = np.zeros(scaled.shape[0] + 1)
    unit[-1] = 1.0

    for b, draw in enumerate(draws):
        if np.max(cone @ draw) <= 1e-10:
            projected[b] = draw
            iterations.append(0)
            continue

        dual_linear = cone @ draw
        try:
            stacked = np.vstack([scaled, dual_linear.reshape(1, -1)])
            weights, _ = nnls(stacked, unit, maxiter=50 * stacked.shape[0])
            residual = stacked @ weights - unit
            candidate = draw + solve_triangular(
                root, -residual[:-1] / residual[-1], lower=False
            )
            violation = float(max(np.max(cone @ candidate), 0.0))
            if np.all(np.isfinite(candidate)) and violation <= 2e-6:
                maximum_violation = max(maximum_violation, violation)
                iterations.append(int(np.sum(weights > 0.0)))
                projected[b] = candidate
                continue
        except Exception:
            pass

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

        if not dual_result.success or violation > 2e-6:
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
        else:
            success = True

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
    cone_kappa,
):
    """Projected multiplier approximation for an active shape boundary."""
    n, p = coefficient_if.shape
    cone_info = estimate_tangent_cone(
        gamma,
        profile_x,
        coefficient_if,
        kappa=cone_kappa,
    )
    if multiplier_draws <= 0:
        raise ValueError("boundary inference requires positive multiplier draws")

    multipliers = rng.normal(size=(multiplier_draws, n))
    root_draws = multipliers @ coefficient_if / math.sqrt(n)
    projected, projection_info = project_draws_to_tangent_cone(
        root_draws,
        hessian,
        cone_info["matrix"],
    )
    valid = np.all(np.isfinite(projected), axis=1)
    if valid.sum() < max(20, math.ceil(0.90 * multiplier_draws)):
        raise RuntimeError(
            "too few valid tangent-cone multiplier projections: "
            f"{valid.sum()} of {multiplier_draws}"
        )

    curve_draws = projected[valid] @ contrast_design.T / math.sqrt(n)
    regular_curve_draws = root_draws[valid] @ contrast_design.T / math.sqrt(n)
    lower_quantile = np.quantile(curve_draws, 0.025, axis=0)
    upper_quantile = np.quantile(curve_draws, 0.975, axis=0)
    projected_pointwise_half_width = np.quantile(
        np.abs(curve_draws),
        0.95,
        axis=0,
    )
    regular_curve_if = coefficient_if @ contrast_design.T
    regular_se = regular_curve_if.std(axis=0, ddof=1) / math.sqrt(n)
    regular_pointwise_half_width = 1.96 * regular_se
    pointwise_half_width = np.maximum(
        projected_pointwise_half_width,
        regular_pointwise_half_width,
    )
    projected_se = curve_draws.std(axis=0, ddof=1)
    scale = np.maximum(projected_se, 1e-12)
    maxima = np.max(np.abs(curve_draws) / scale.reshape(1, -1), axis=1)
    simultaneous_critical = float(np.quantile(maxima, 0.95))
    projected_simultaneous_half_width = simultaneous_critical * scale

    regular_scale = np.maximum(regular_se, 1e-12)
    regular_maxima = np.max(
        np.abs(regular_curve_draws) / regular_scale.reshape(1, -1),
        axis=1,
    )
    regular_simultaneous_critical = float(np.quantile(regular_maxima, 0.95))
    regular_simultaneous_half_width = (
        regular_simultaneous_critical * regular_scale
    )
    simultaneous_half_width = np.maximum(
        projected_simultaneous_half_width,
        regular_simultaneous_half_width,
    )

    return {
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
        "pointwise_half_width": pointwise_half_width,
        "projected_pointwise_half_width": projected_pointwise_half_width,
        "regular_pointwise_half_width": regular_pointwise_half_width,
        "projected_se": projected_se,
        "regular_se": regular_se,
        "simultaneous_critical": simultaneous_critical,
        "regular_simultaneous_critical": regular_simultaneous_critical,
        "simultaneous_half_width": simultaneous_half_width,
        "projected_simultaneous_half_width": (
            projected_simultaneous_half_width
        ),
        "regular_simultaneous_half_width": regular_simultaneous_half_width,
        "valid_draws": int(valid.sum()),
        "cone": cone_info,
        "projection": projection_info,
    }


def make_evaluation_profiles(basis):
    levels = np.array([-0.75, 0.0, 0.75])
    z1, z2 = np.meshgrid(levels, levels, indexing="ij")
    z1 = z1.ravel()
    z2 = z2.ravel()
    a0 = np.zeros_like(z1)
    a1 = np.ones_like(z1)
    profile_x = profile_design(
        np.concatenate([a0, a1]),
        np.concatenate([z1, z1]),
        np.concatenate([z2, z2]),
        basis,
    )
    x0 = profile_design(a0, z1, z2, basis)
    x1 = profile_design(a1, z1, z2, basis)
    contrast = np.mean(x1 - x0, axis=0)
    return profile_x, contrast


def add_subject_profiles(profile_x, data, basis):
    """Append every simulated subject, under both treatments, to the constrained set."""
    n = len(data["a"])
    x0 = profile_design(np.zeros(n), data["z1"], data["z2"], basis)
    x1 = profile_design(np.ones(n), data["z1"], data["z2"], basis)
    return np.concatenate([profile_x, x0, x1], axis=0)


def fit_all_methods(
    data,
    design,
    penalty,
    profile_x,
    endpoints,
    completed,
    lambda_grid,
):
    oracle = np.minimum(data["event_time"][:, None], TAU.reshape(1, -1))
    base_fits = {
        "JC": fit_penalized_response(completed, design, penalty, lambda_grid),
        "CT": fit_penalized_response(endpoints["tail"], design, penalty, lambda_grid),
        "IPCW": fit_penalized_response(endpoints["ipcw"], design, penalty, lambda_grid),
        "ORACLE": fit_penalized_response(oracle, design, penalty, lambda_grid),
    }
    fitted = {
        "JC-P": {
            **base_fits["JC"],
            "active": 0,
            "qp_iterations": 0,
            "qp_violation": shape_metrics(base_fits["JC"]["gamma"], profile_x)[
                "max_shape_violation"
            ],
        }
    }
    for source, method in [
        ("JC", "JC-SP"),
        ("CT", "CT-SP"),
        ("IPCW", "IPCW-SP"),
        ("ORACLE", "ORACLE-SP"),
    ]:
        gamma, active, iterations, violation = constrain_fit(base_fits[source], profile_x)
        fitted[method] = {
            **base_fits[source],
            "gamma": gamma,
            "active": active,
            "qp_iterations": iterations,
            "qp_violation": violation,
        }
    return fitted


def run_replication(cell, replication, attempt, seed, config):
    start_time = time.perf_counter()
    rng = np.random.default_rng(seed)
    data = generate_data(cell["n"], cell["pattern"], cell["censor_lambda"], rng)
    event_correct, censor_correct = NUISANCE_CONDITIONS[cell["nuisance"]]
    event_fit = fit_event_model(
        data, event_correct, omit_treatment=cell["nuisance"] == "event_omit"
    )
    censor_fit = fit_censor_model(data, censor_correct)
    design, penalty, basis = build_basis(
        data,
        n_tau_basis=config["n_tau_basis"],
        n_z_basis=config["n_z_basis"],
    )
    profile_x, contrast_design = make_evaluation_profiles(basis)
    if config.get("subject_constraints", False):
        profile_x = add_subject_profiles(profile_x, data, basis)
    support_width = config["support_width"]
    if config["support_rate"] is not None:
        support_width = (
            config["support_scale"] * cell["n"] ** (-config["support_rate"])
        )
    endpoints = construct_endpoint_responses(
        data,
        design,
        event_fit,
        censor_fit,
        support_width,
    )
    completed, completion_info = calibrate_completion(
        data, design, endpoints, config["rho"]
    )
    completion_info["rho"] = config["rho"]
    lambda_grid = (
        np.asarray(config["lambda_multipliers"], dtype=float)
        * cell["n"] ** (-0.75)
    )
    fits = fit_all_methods(
        data,
        design,
        penalty,
        profile_x,
        endpoints,
        completed,
        lambda_grid,
    )
    truth = np.asarray(cell["truth"], dtype=float)

    influence = None
    se_jc = np.full(len(TAU), np.nan)
    simultaneous_critical = np.nan
    boundary = None
    if cell["nuisance"] == "correct":
        influence = estimate_jc_influence(
            data,
            design,
            endpoints,
            event_fit,
            censor_fit,
            completion_info,
            fits["JC-P"],
            contrast_design,
            config["multiplier_draws"],
            rng,
        )
        se_jc = influence["standard_error"]
        simultaneous_critical = influence["simultaneous_critical"]
        if config["boundary_inference"]:
            cone_kappa = config["cone_kappa"]
            if cone_kappa is None:
                cone_kappa = 0.10 * math.sqrt(math.log(cell["n"]))
            boundary = estimate_boundary_inference(
                fits["JC-SP"]["gamma"],
                profile_x,
                contrast_design,
                fits["JC-SP"]["hessian"],
                influence["coefficient_if"],
                config["multiplier_draws"],
                rng,
                cone_kappa,
            )

    common = {
        "cell_id": cell["cell_id"],
        "n": cell["n"],
        "censor_target": cell["censor_target"],
        "pattern": cell["pattern"],
        "nuisance": cell["nuisance"],
        "replication": replication,
        "attempt": attempt,
        "seed": seed,
        "replacement": int(attempt != replication),
    }
    horizon_rows = []
    replication_rows = []
    for method in METHOD_NAMES:
        fit = fits[method]
        estimate = contrast_design @ fit["gamma"]
        error = estimate - truth
        metrics = shape_metrics(fit["gamma"], profile_x)
        use_interior_inference = method == "JC-P" or (
            method == "JC-SP" and fit["active"] == 0
        )
        standard_error = se_jc if use_interior_inference else np.full(len(TAU), np.nan)
        covered = np.where(
            np.isfinite(standard_error),
            np.abs(error) <= 1.96 * standard_error,
            np.nan,
        )
        rejected = np.where(
            np.isfinite(standard_error),
            np.abs(estimate) > 1.96 * standard_error,
            np.nan,
        )
        simultaneous_covered = np.nan
        global_reject = np.nan
        if use_interior_inference and np.isfinite(simultaneous_critical):
            simultaneous_covered = float(
                np.all(np.abs(error) <= simultaneous_critical * standard_error)
            )
            global_reject = float(
                np.any(np.abs(estimate) > simultaneous_critical * standard_error)
            )

        boundary_standard_error = np.full(len(TAU), np.nan)
        boundary_lower = np.full(len(TAU), np.nan)
        boundary_upper = np.full(len(TAU), np.nan)
        boundary_covered = np.full(len(TAU), np.nan)
        boundary_rejected = np.full(len(TAU), np.nan)
        boundary_simultaneous_covered = np.nan
        boundary_global_reject = np.nan
        if method == "JC-SP" and boundary is not None:
            boundary_standard_error = boundary["projected_se"]
            boundary_lower = estimate - boundary["pointwise_half_width"]
            boundary_upper = estimate + boundary["pointwise_half_width"]
            boundary_covered = (
                (truth >= boundary_lower) & (truth <= boundary_upper)
            ).astype(float)
            boundary_rejected = (
                (boundary_lower > 0.0) | (boundary_upper < 0.0)
            ).astype(float)
            half_width = boundary["simultaneous_half_width"]
            boundary_simultaneous_covered = float(
                np.all(np.abs(error) <= half_width)
            )
            boundary_global_reject = float(
                np.any(np.abs(estimate) > half_width)
            )
        for k, tau_k in enumerate(TAU):
            horizon_rows.append(
                {
                    **common,
                    "method": method,
                    "tau": tau_k,
                    "estimate": estimate[k],
                    "truth": truth[k],
                    "error": error[k],
                    "standard_error": standard_error[k],
                    "covered": covered[k],
                    "rejected": rejected[k],
                    "boundary_standard_error": boundary_standard_error[k],
                    "boundary_lower": boundary_lower[k],
                    "boundary_upper": boundary_upper[k],
                    "boundary_covered": boundary_covered[k],
                    "boundary_rejected": boundary_rejected[k],
                }
            )
        boundary_diagnostics = {
            "boundary_simultaneous_covered": boundary_simultaneous_covered,
            "boundary_global_reject": boundary_global_reject,
            "boundary_candidate_constraints": np.nan,
            "boundary_cone_constraints": np.nan,
            "boundary_exact_active_constraints": np.nan,
            "boundary_valid_draws": np.nan,
            "boundary_projection_failures": np.nan,
            "boundary_projection_fallbacks": np.nan,
            "boundary_projection_max_violation": np.nan,
            "boundary_projection_mean_iterations": np.nan,
            "boundary_simultaneous_critical": np.nan,
            "boundary_regular_simultaneous_critical": np.nan,
            "boundary_regular_pointwise_dominance": np.nan,
            "boundary_regular_simultaneous_dominance": np.nan,
        }
        if method == "JC-SP" and boundary is not None:
            boundary_diagnostics.update(
                {
                    "boundary_candidate_constraints": boundary["cone"][
                        "candidate_rows"
                    ],
                    "boundary_cone_constraints": boundary["cone"]["unique_rows"],
                    "boundary_exact_active_constraints": boundary["cone"][
                        "exact_active_rows"
                    ],
                    "boundary_valid_draws": boundary["valid_draws"],
                    "boundary_projection_failures": boundary["projection"][
                        "failures"
                    ],
                    "boundary_projection_fallbacks": boundary["projection"][
                        "fallbacks"
                    ],
                    "boundary_projection_max_violation": boundary["projection"][
                        "maximum_violation"
                    ],
                    "boundary_projection_mean_iterations": boundary["projection"][
                        "mean_iterations"
                    ],
                    "boundary_simultaneous_critical": boundary[
                        "simultaneous_critical"
                    ],
                    "boundary_regular_simultaneous_critical": boundary[
                        "regular_simultaneous_critical"
                    ],
                    "boundary_regular_pointwise_dominance": float(
                        np.mean(
                            boundary["regular_pointwise_half_width"]
                            >= boundary["projected_pointwise_half_width"]
                        )
                    ),
                    "boundary_regular_simultaneous_dominance": float(
                        np.mean(
                            boundary["regular_simultaneous_half_width"]
                            >= boundary["projected_simultaneous_half_width"]
                        )
                    ),
                }
            )
        replication_rows.append(
            {
                **common,
                "method": method,
                "ise": float(np.mean(error**2)),
                "lambda": fit["lambda"],
                "gcv": fit["gcv"],
                "active_constraints": fit["active"],
                "qp_iterations": fit["qp_iterations"],
                "qp_violation": fit["qp_violation"],
                "simultaneous_covered": simultaneous_covered,
                "global_reject": global_reject,
                **boundary_diagnostics,
                **metrics,
            }
        )

    completion_row = {
        **common,
        "observed_censoring": float(np.mean(data["censor_event"])),
        "n_completed": completion_info["n_completed"],
        "tail_ipcw_gap": completion_info["gap0"],
        "fitted_ipcw_gap": completion_info["gap_fitted"],
        "gap_ratio": completion_info["gap_fitted"]
        / max(completion_info["gap0"], 1e-14),
        "mean_kl": completion_info["mean_kl"],
        "kl_per_subject": completion_info["kl_per_subject"],
        "mean_effective_support": completion_info["mean_ess"],
        "dual_iterations": completion_info["iterations"],
        "dual_root_norm": completion_info["root_norm"],
        "completed_path_violation": response_path_metrics(completed),
        "tail_path_violation": response_path_metrics(endpoints["tail"]),
        "event_fit_iterations": event_fit["iterations"],
        "censor_fit_iterations": censor_fit["iterations"],
        "runtime_seconds": time.perf_counter() - start_time,
        "support_width": support_width,
    }
    return horizon_rows, replication_rows, completion_row


def make_cells(design_name):
    cells = []
    if design_name == "main":
        for n in (250, 500):
            for censor_target in (0.20, 0.50):
                for pattern in ("null", "delayed", "crossing"):
                    for nuisance in ("correct", "both_miss"):
                        cells.append((n, censor_target, pattern, nuisance))
        for nuisance in ("event_miss", "censor_miss"):
            cells.append((500, 0.50, "delayed", nuisance))
    elif design_name == "main_b":
        for n in (250, 500):
            for censor_target in (0.20, 0.50):
                for pattern in ("null", "delayed", "crossing"):
                    for nuisance in ("correct", "both_miss"):
                        cells.append((n, censor_target, pattern, nuisance))
        for nuisance in ("event_miss", "censor_miss"):
            cells.append((500, 0.50, "delayed", nuisance))
        for nuisance in ("event_miss", "event_omit"):
            for n in (250, 500):
                for censor_target in (0.20, 0.50):
                    for pattern in ("null", "delayed", "crossing"):
                        if (n, censor_target, pattern, nuisance) not in cells:
                            cells.append((n, censor_target, pattern, nuisance))
    elif design_name == "full":
        for n in (250, 500, 1000):
            for censor_target in (0.20, 0.40, 0.60):
                for pattern in TREATMENT_PATTERNS:
                    for nuisance in NUISANCE_CONDITIONS:
                        cells.append((n, censor_target, pattern, nuisance))
    elif design_name == "inference_pilot":
        for n in (250, 500):
            for censor_target in (0.20, 0.50):
                for pattern in ("null", "crossing"):
                    cells.append((n, censor_target, pattern, "correct"))
    elif design_name == "boundary_main":
        for n in (250, 500):
            for censor_target in (0.20, 0.50):
                for pattern in ("null", "delayed", "crossing"):
                    cells.append((n, censor_target, pattern, "correct"))
    elif design_name == "pilot":
        cells.append((120, 0.40, "delayed", "correct"))
    else:
        raise ValueError(f"unknown design {design_name}")

    calibration_cache = {}
    output = []
    for cell_id, (n, censor_target, pattern, nuisance) in enumerate(cells, start=1):
        key = (censor_target, pattern)
        if key not in calibration_cache:
            calibration_cache[key] = calibrate_censoring_lambda(censor_target, pattern)
        censor_lambda, achieved = calibration_cache[key]
        output.append(
            {
                "cell_id": cell_id,
                "n": n,
                "censor_target": censor_target,
                "pattern": pattern,
                "nuisance": nuisance,
                "censor_lambda": censor_lambda,
                "calibrated_censoring": achieved,
                "truth": marginal_treatment_truth(pattern).tolist(),
            }
        )
    return output


def run_cell(cell, config):
    horizon_rows = []
    replication_rows = []
    completion_rows = []
    failure_rows = []
    valid = 0
    attempt = 0
    max_attempts = config["reps"] + max(10, math.ceil(0.20 * config["reps"]))
    while valid < config["reps"] and attempt < max_attempts:
        attempt += 1
        seed = int(config["seed"] + cell["cell_id"] * 1_000_000 + attempt)
        try:
            rows_h, rows_r, row_c = run_replication(
                cell, valid + 1, attempt, seed, config
            )
            horizon_rows.extend(rows_h)
            replication_rows.extend(rows_r)
            completion_rows.append(row_c)
            valid += 1
        except Exception as error:
            failure_rows.append(
                {
                    "cell_id": cell["cell_id"],
                    "n": cell["n"],
                    "censor_target": cell["censor_target"],
                    "pattern": cell["pattern"],
                    "nuisance": cell["nuisance"],
                    "attempt": attempt,
                    "seed": seed,
                    "error_type": type(error).__name__,
                    "message": str(error),
                    "traceback": " | ".join(traceback.format_exc().splitlines()[-4:]),
                }
            )
    status = {
        "cell_id": cell["cell_id"],
        "requested": config["reps"],
        "valid": valid,
        "attempts": attempt,
        "failures": len(failure_rows),
        "complete": int(valid == config["reps"]),
    }
    return horizon_rows, replication_rows, completion_rows, failure_rows, status


def save_cell_checkpoint(output_directory, cell_id, result):
    """Save one completed cell immediately so an interrupted run is auditable."""
    checkpoint = output_directory / "cell_checkpoints"
    checkpoint.mkdir(parents=True, exist_ok=True)
    horizon, replication, completion, failures, status = result
    prefix = checkpoint / f"cell_{cell_id:03d}"
    pd.DataFrame(horizon).to_csv(str(prefix) + "_horizon.csv", index=False)
    pd.DataFrame(replication).to_csv(str(prefix) + "_replication.csv", index=False)
    pd.DataFrame(completion).to_csv(str(prefix) + "_completion.csv", index=False)
    failure_columns = [
        "cell_id",
        "n",
        "censor_target",
        "pattern",
        "nuisance",
        "attempt",
        "seed",
        "error_type",
        "message",
        "traceback",
    ]
    pd.DataFrame(failures, columns=failure_columns).to_csv(
        str(prefix) + "_failures.csv", index=False
    )
    pd.DataFrame([status]).to_csv(str(prefix) + "_status.csv", index=False)


def mean_or_nan(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else np.nan


def summarize_outputs(horizon, replication, completion):
    group_h = ["n", "censor_target", "pattern", "nuisance", "method", "tau"]
    summary_h = (
        horizon.groupby(group_h, dropna=False)
        .agg(
            replications=("estimate", "count"),
            mean_estimate=("estimate", "mean"),
            truth=("truth", "mean"),
            bias=("error", "mean"),
            rmse=("error", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
            empirical_sd=("estimate", "std"),
            average_se=("standard_error", mean_or_nan),
            inference_replications=("covered", lambda x: int(x.notna().sum())),
            coverage=("covered", mean_or_nan),
            rejection=("rejected", mean_or_nan),
            boundary_average_se=("boundary_standard_error", mean_or_nan),
            boundary_inference_replications=(
                "boundary_covered",
                lambda x: int(x.notna().sum()),
            ),
            boundary_coverage=("boundary_covered", mean_or_nan),
            boundary_rejection=("boundary_rejected", mean_or_nan),
        )
        .reset_index()
    )
    for prefix in ("", "boundary_"):
        denominator = summary_h[
            prefix + "inference_replications"
        ].replace(0, np.nan)
        for outcome in ("coverage", "rejection"):
            column = prefix + outcome
            summary_h[column + "_mcse"] = np.sqrt(
                summary_h[column] * (1.0 - summary_h[column]) / denominator
            )

    group_r = ["n", "censor_target", "pattern", "nuisance", "method"]
    summary_r = (
        replication.groupby(group_r, dropna=False)
        .agg(
            replications=("ise", "count"),
            mean_ise=("ise", "mean"),
            median_ise=("ise", "median"),
            shape_violation_rate=("any_shape_violation", "mean"),
            mean_v_range=("v_range", "mean"),
            mean_v_concave=("v_concave", "mean"),
            mean_active_constraints=("active_constraints", "mean"),
            mean_qp_iterations=("qp_iterations", "mean"),
            simultaneous_inference_replications=(
                "simultaneous_covered",
                lambda x: int(x.notna().sum()),
            ),
            simultaneous_coverage=("simultaneous_covered", mean_or_nan),
            global_rejection=("global_reject", mean_or_nan),
            boundary_simultaneous_inference_replications=(
                "boundary_simultaneous_covered",
                lambda x: int(x.notna().sum()),
            ),
            boundary_simultaneous_coverage=(
                "boundary_simultaneous_covered",
                mean_or_nan,
            ),
            boundary_global_rejection=("boundary_global_reject", mean_or_nan),
            mean_boundary_candidate_constraints=(
                "boundary_candidate_constraints",
                mean_or_nan,
            ),
            mean_boundary_cone_constraints=(
                "boundary_cone_constraints",
                mean_or_nan,
            ),
            total_boundary_projection_failures=(
                "boundary_projection_failures",
                lambda x: float(pd.to_numeric(x, errors="coerce").sum()),
            ),
            total_boundary_projection_fallbacks=(
                "boundary_projection_fallbacks",
                lambda x: float(pd.to_numeric(x, errors="coerce").sum()),
            ),
            max_boundary_projection_violation=(
                "boundary_projection_max_violation",
                lambda x: float(pd.to_numeric(x, errors="coerce").max()),
            ),
            mean_boundary_regular_pointwise_dominance=(
                "boundary_regular_pointwise_dominance",
                mean_or_nan,
            ),
            mean_boundary_regular_simultaneous_dominance=(
                "boundary_regular_simultaneous_dominance",
                mean_or_nan,
            ),
        )
        .reset_index()
    )

    group_c = ["n", "censor_target", "pattern", "nuisance"]
    summary_c = (
        completion.groupby(group_c, dropna=False)
        .agg(
            replications=("replication", "count"),
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
            support_width=("support_width", "mean"),
        )
        .reset_index()
    )
    return summary_h, summary_r, summary_c


def write_simple_latex_table(frame, path, columns, digits=4):
    selected = frame.loc[:, columns].copy()
    for column in selected.select_dtypes(include=[np.number]).columns:
        selected[column] = selected[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
        )
    column_spec = "l" * len(columns)
    lines = ["\\begin{tabular}{" + column_spec + "}", "\\toprule"]
    lines.append(" & ".join(columns) + " \\\\")
    lines.append("\\midrule")
    for row in selected.itertuples(index=False, name=None):
        lines.append(" & ".join(str(value) for value in row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design",
        choices=["main", "main_b", "full", "inference_pilot", "boundary_main"],
        default="main",
    )
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--rho", type=float, default=0.7)
    parser.add_argument("--support-width", type=float, default=0.10)
    parser.add_argument(
        "--support-rate",
        type=float,
        default=None,
        help="if set, use support_scale*n^(-support_rate) instead of a fixed width",
    )
    parser.add_argument(
        "--support-scale",
        type=float,
        default=1.0,
        help="scale in the n-dependent support-width rule",
    )
    parser.add_argument("--multiplier-draws", type=int, default=399)
    parser.add_argument("--boundary-inference", action="store_true")
    parser.add_argument(
        "--subject-constraints",
        action="store_true",
        help="also impose the shape restrictions at every subject under both treatments",
    )
    parser.add_argument(
        "--cone-kappa",
        type=float,
        default=None,
        help="near-active threshold multiplier; default is 0.1*sqrt(log(n))",
    )
    parser.add_argument("--out", default="simulation_output")
    parser.add_argument("--pilot", action="store_true")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.reps <= 0 or args.jobs <= 0:
        raise ValueError("reps and jobs must be positive")
    if args.rho <= 0.0 or args.support_width <= 0.0:
        raise ValueError("rho and support width must be positive")
    if args.support_rate is not None and args.support_rate <= 0.5:
        raise ValueError("support rate must be greater than 0.5")
    if args.support_scale <= 0.0:
        raise ValueError("support scale must be positive")
    if args.cone_kappa is not None and args.cone_kappa <= 0.0:
        raise ValueError("cone kappa must be positive")

    design_name = "pilot" if args.pilot else args.design
    reps = 2 if args.pilot else args.reps
    multiplier_draws = 49 if args.pilot else args.multiplier_draws
    output_directory = Path(args.out)
    if args.pilot and args.out == "simulation_output":
        output_directory = Path("simulation_pilot_output")
    output_directory.mkdir(parents=True, exist_ok=True)

    config = {
        "design": design_name,
        "reps": reps,
        "jobs": args.jobs,
        "seed": args.seed,
        "rho": args.rho,
        "support_width": 0.20 if args.pilot else args.support_width,
        "support_rate": args.support_rate,
        "support_scale": args.support_scale,
        "multiplier_draws": multiplier_draws,
        "boundary_inference": bool(args.boundary_inference),
        "subject_constraints": bool(args.subject_constraints),
        "cone_kappa": (
            float(args.cone_kappa) if args.cone_kappa is not None else None
        ),
        "n_tau_basis": 6,
        "n_z_basis": 5,
        "lambda_multipliers": [0.25, 0.5, 1.0, 2.0],
    }
    cells = make_cells(design_name)
    calibration = pd.DataFrame(
        [
            {
                "cell_id": cell["cell_id"],
                "pattern": cell["pattern"],
                "censor_target": cell["censor_target"],
                "censor_lambda": cell["censor_lambda"],
                "calibrated_censoring": cell["calibrated_censoring"],
            }
            for cell in cells
        ]
    ).drop_duplicates(["pattern", "censor_target"])
    calibration.to_csv(output_directory / "censoring_calibration.csv", index=False)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "started": pd.Timestamp.now().isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "config": config,
        "cells": cells,
        "status": "running",
    }
    (output_directory / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    all_horizon = []
    all_replication = []
    all_completion = []
    all_failures = []
    statuses = []
    started = time.perf_counter()

    if args.jobs == 1 or len(cells) == 1:
        for cell in cells:
            print(
                f"cell {cell['cell_id']}/{len(cells)}: n={cell['n']}, "
                f"censor={cell['censor_target']:.0%}, {cell['pattern']}, "
                f"{cell['nuisance']}",
                flush=True,
            )
            result = run_cell(cell, config)
            save_cell_checkpoint(output_directory, cell["cell_id"], result)
            h, r, c, f, status = result
            all_horizon.extend(h)
            all_replication.extend(r)
            all_completion.extend(c)
            all_failures.extend(f)
            statuses.append(status)
            print(
                f"  valid={status['valid']}, failures={status['failures']}, "
                f"attempts={status['attempts']}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(run_cell, cell, config): cell for cell in cells}
            for future in as_completed(futures):
                cell = futures[future]
                result = future.result()
                save_cell_checkpoint(output_directory, cell["cell_id"], result)
                h, r, c, f, status = result
                all_horizon.extend(h)
                all_replication.extend(r)
                all_completion.extend(c)
                all_failures.extend(f)
                statuses.append(status)
                print(
                    f"cell {cell['cell_id']} complete: valid={status['valid']}, "
                    f"failures={status['failures']}",
                    flush=True,
                )

    horizon = pd.DataFrame(all_horizon)
    replication = pd.DataFrame(all_replication)
    completion = pd.DataFrame(all_completion)
    failure_columns = [
        "cell_id",
        "n",
        "censor_target",
        "pattern",
        "nuisance",
        "attempt",
        "seed",
        "error_type",
        "message",
        "traceback",
    ]
    failures = pd.DataFrame(all_failures, columns=failure_columns)
    status_frame = pd.DataFrame(statuses).sort_values("cell_id")

    horizon.to_csv(output_directory / "horizon_results.csv", index=False)
    replication.to_csv(output_directory / "replication_results.csv", index=False)
    completion.to_csv(output_directory / "completion_diagnostics.csv", index=False)
    failures.to_csv(output_directory / "failures.csv", index=False)
    status_frame.to_csv(output_directory / "cell_status.csv", index=False)

    if len(horizon) and len(replication) and len(completion):
        summary_h, summary_r, summary_c = summarize_outputs(
            horizon, replication, completion
        )
        summary_h.to_csv(output_directory / "summary_horizon.csv", index=False)
        summary_r.to_csv(output_directory / "summary_method.csv", index=False)
        summary_c.to_csv(output_directory / "summary_completion.csv", index=False)
        table_h = summary_h[summary_h["tau"].isin([2.0, 3.5, 5.0])]
        write_simple_latex_table(
            table_h,
            output_directory / "table_horizon.tex",
            [
                "n",
                "censor_target",
                "pattern",
                "nuisance",
                "method",
                "tau",
                "bias",
                "rmse",
                "coverage",
            ],
        )
        write_simple_latex_table(
            summary_c,
            output_directory / "table_completion.tex",
            [
                "n",
                "censor_target",
                "pattern",
                "nuisance",
                "gap_ratio",
                "mean_kl",
                "completed_path_max",
                "mean_runtime_seconds",
            ],
        )

    elapsed = time.perf_counter() - started
    manifest["finished"] = pd.Timestamp.now().isoformat()
    manifest["elapsed_seconds"] = elapsed
    manifest["status"] = (
        "complete" if len(status_frame) and status_frame["complete"].all() else "incomplete"
    )
    manifest["valid_replications"] = int(status_frame["valid"].sum())
    manifest["failed_attempts"] = int(status_frame["failures"].sum())
    (output_directory / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"finished in {elapsed / 60.0:.1f} minutes; "
        f"valid={manifest['valid_replications']}, "
        f"failed attempts={manifest['failed_attempts']}; "
        f"status={manifest['status']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
