#!/usr/bin/env python3
"""Create the vector figure for the colon trial application."""

from pathlib import Path
import math

import pandas as pd

from prepare_simulation_results import PdfCanvas, draw_axes


def polygon(canvas, points, fill, stroke=None, line_width=0.0):
    if len(points) < 3:
        return
    path = [f"{points[0][0]:.3f} {points[0][1]:.3f} m"]
    path.extend(f"{x:.3f} {y:.3f} l" for x, y in points[1:])
    path.append("h")
    commands = ["q", f"{fill[0]:.3f} {fill[1]:.3f} {fill[2]:.3f} rg"]
    if stroke is not None and line_width > 0.0:
        commands.extend(
            [
                f"{stroke[0]:.3f} {stroke[1]:.3f} {stroke[2]:.3f} RG",
                f"{line_width:.3f} w",
            ]
        )
    commands.extend(path)
    commands.extend(["B" if stroke is not None else "f", "Q"])
    canvas.commands.append(" ".join(commands))


def write_figure(path, estimates):
    estimates = estimates.copy()
    numeric = [
        "estimate",
        "pointwise_lower",
        "pointwise_upper",
        "simultaneous_lower",
        "simultaneous_upper",
    ]
    estimates[numeric] = 12.0 * estimates[numeric]
    primary = estimates.loc[estimates["method"] == "JC-SP"].sort_values("tau")

    canvas = PdfCanvas(612, 420)
    canvas.text(
        306,
        398,
        "Gain in restricted mean survival: levamisole plus 5-FU versus observation",
        11,
        "F2",
        align="center",
    )
    x0, y0, width, height = 70, 75, 505, 285
    map_x, map_y = draw_axes(
        canvas,
        x0,
        y0,
        width,
        height,
        (0.5, 7.0),
        (-2.0, 12.5),
        [0.5, 1, 2, 3, 4, 5, 6, 7],
        [-2, 0, 2, 4, 6, 8, 10, 12],
        y_decimals=0,
    )

    upper = [
        (map_x(row.tau), map_y(row.simultaneous_upper))
        for row in primary.itertuples()
    ]
    lower = [
        (map_x(row.tau), map_y(row.simultaneous_lower))
        for row in reversed(list(primary.itertuples()))
    ]
    polygon(canvas, upper + lower, fill=(0.91, 0.91, 0.91))
    upper = [
        (map_x(row.tau), map_y(row.pointwise_upper))
        for row in primary.itertuples()
    ]
    lower = [
        (map_x(row.tau), map_y(row.pointwise_lower))
        for row in reversed(list(primary.itertuples()))
    ]
    polygon(canvas, upper + lower, fill=(0.80, 0.80, 0.80))
    canvas.line(
        map_x(0.5),
        map_y(0.0),
        map_x(7.0),
        map_y(0.0),
        width=0.8,
        color=(0.30, 0.30, 0.30),
        dash=(4, 3),
    )

    styles = {
        "JC-SP": ((0.0, 0.0, 0.0), (), 2.0),
        "JC-P": ((0.38, 0.38, 0.38), (3, 2), 1.1),
        "CT-SP": ((0.12, 0.35, 0.70), (8, 2, 2, 2), 1.2),
        "IPCW-SP": ((0.75, 0.18, 0.16), (8, 3), 1.2),
        "PV-P": ((0.10, 0.55, 0.25), (1, 2), 1.2),
        "KM": ((0.52, 0.23, 0.63), (10, 3), 1.2),
    }
    for method in ["KM", "PV-P", "IPCW-SP", "CT-SP", "JC-P", "JC-SP"]:
        subset = estimates.loc[estimates["method"] == method].sort_values("tau")
        points = [
            (map_x(row.tau), map_y(row.estimate))
            for row in subset.itertuples()
        ]
        color, dash, line_width = styles[method]
        canvas.polyline(points, line_width, color, dash)
        if method == "JC-SP":
            for point_x, point_y in points:
                canvas.circle(
                    point_x,
                    point_y,
                    radius=1.5,
                    stroke=color,
                    fill=color,
                )

    canvas.text(322, 44, "Time horizon (years)", 9, align="center")
    canvas.text(
        22,
        215,
        "Estimated RMST gain (months)",
        9,
        rotate=90,
        align="center",
    )
    canvas.text(
        324,
        365,
        "Dark grey: pointwise 95% interval; light grey: simultaneous 95% band",
        7,
        align="center",
    )

    legend_order = ["JC-SP", "JC-P", "CT-SP", "IPCW-SP", "PV-P", "KM"]
    legend_x = [68, 151, 230, 316, 415, 486]
    for method, location in zip(legend_order, legend_x):
        color, dash, line_width = styles[method]
        canvas.line(
            location, 386, location + 20, 386, line_width, color, dash
        )
        canvas.text(location + 24, 383, method, 7)
    canvas.save(path)


def write_nonlinear_figure(path, components):
    canvas = PdfCanvas(612, 270)
    canvas.text(
        306,
        252,
        "Estimated nonlinear RMST components at seven years",
        11,
        "F2",
        align="center",
    )
    age = components.loc[components["covariate"] == "Age", "value"]
    nodes = components.loc[
        components["covariate"] == "Positive nodes", "value"
    ]
    panels = [
        (
            "Age",
            "Age (years)",
            58,
            (float(age.min()), float(age.max())),
            [40, 50, 60, 70],
        ),
        (
            "Positive nodes",
            "Number of positive nodes",
            327,
            (float(nodes.min()), float(nodes.max())),
            [1, 3, 5, 8, 11],
        ),
    ]
    for index, (covariate, x_label, x0, x_limits, x_ticks) in enumerate(panels):
        subset = components.loc[components["covariate"] == covariate].sort_values(
            "value"
        )
        maximum = float(subset["component_months"].abs().max())
        limit = max(0.5, math.ceil(maximum * 2.0) / 2.0)
        y_limits = (-limit, limit)
        y_ticks = [-limit, -limit / 2.0, 0.0, limit / 2.0, limit]
        y0, width, height = 57, 226, 154
        canvas.text(x0 + width / 2, 224, covariate, 9, "F2", align="center")
        map_x, map_y = draw_axes(
            canvas,
            x0,
            y0,
            width,
            height,
            x_limits,
            y_limits,
            x_ticks,
            y_ticks,
            show_y_labels=True,
            y_decimals=1,
        )
        canvas.line(
            map_x(x_limits[0]),
            map_y(0.0),
            map_x(x_limits[1]),
            map_y(0.0),
            width=0.8,
            color=(0.35, 0.35, 0.35),
            dash=(4, 3),
        )
        points = [
            (map_x(row.value), map_y(row.component_months))
            for row in subset.itertuples()
        ]
        canvas.polyline(points, 1.8, (0.05, 0.25, 0.55))
        canvas.text(x0 + width / 2, 29, x_label, 8, align="center")
    canvas.text(
        18,
        135,
        "Centered contribution (months)",
        8,
        rotate=90,
        align="center",
    )
    canvas.save(path)


def main():
    output = Path("real_data_output")
    estimates = pd.read_csv(output / "colon_estimates.csv")
    figure = output / "figure_colon_treatment_difference.pdf"
    write_figure(figure, estimates)
    components = pd.read_csv(output / "colon_nonlinear_components.csv")
    nonlinear_figure = output / "figure_colon_nonlinear_components.pdf"
    write_nonlinear_figure(nonlinear_figure, components)
    print(f"wrote: {figure.resolve()}")
    print(f"wrote: {nonlinear_figure.resolve()}")


if __name__ == "__main__":
    main()
