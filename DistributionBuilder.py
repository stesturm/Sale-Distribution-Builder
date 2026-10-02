"""Distribution Builder for optimal selling of a defaultable asset.

The user draws a target sale-price distribution. The application projects it
onto the attainable/optimal set from Jin and Sturm using a finite-dimensional
extended-f-divergence program.
"""

from __future__ import annotations

import math
import re
import tkinter as tk
from tkinter import ttk

import numpy as np

from SellingProjection import (
    ProjectionError,
    ProjectionResult,
    ShiftedArithmeticBrownian,
    project_distribution,
)


class App:
    CANVAS_W = 630
    CANVAS_H = 570
    LEFT = 62
    TOP = 22
    PLOT_W = 540
    PLOT_H = 455
    DIVERGENCES = (
        "Kullback–Leibler",
        "Rényi",
        "Squared Hellinger",
        "Total variation",
    )

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Distribution Builder — Optimal Selling")
        self.root.minsize(990, 690)
        self.root.option_add("*Font", ("TkDefaultFont", 10))
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.dimension = 20
        self.from_ = 0.0
        self.to = 4.0
        self.range_is_auto = True
        self.col_labels = np.linspace(self.from_, self.to, self.dimension)
        self.col_count = np.zeros(self.dimension, dtype=int)
        self.undo_stack: list[np.ndarray] = []
        self.process: ShiftedArithmeticBrownian | None = None
        self.projection: ProjectionResult | None = None
        self.dragging = False
        self.y_limit = 5
        self.full_height_var = tk.BooleanVar(value=False)
        self.divergence_var = tk.StringVar(value=self.DIVERGENCES[0])
        self.alpha_var = tk.StringVar(value="2")

        self._configure_styles()
        self._create_ui()
        self.root.bind("<Return>", lambda _event: self.apply_inputs())
        self.root.bind("<Control-z>", lambda _event: self.undo())
        self.root.bind("<Control-l>", lambda _event: self.clear_distribution())
        self.apply_inputs()

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.configure("Title.TLabel", font=("TkDefaultFont", 15, "bold"))
        style.configure("Section.TLabel", font=("TkDefaultFont", 10, "bold"))
        style.configure("Muted.TLabel", foreground="#666666")
        style.configure("Result.TLabel", foreground="#9a4f14", font=("TkDefaultFont", 10, "bold"))

    def _create_ui(self) -> None:
        self.parameter_frame = ttk.Frame(self.root, padding=(16, 14))
        self.parameter_frame.grid(row=0, column=0, sticky="ns")

        ttk.Label(self.parameter_frame, text="Selling model", style="Title.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        process_box = ttk.LabelFrame(self.parameter_frame, text="Shifted arithmetic Brownian asset", padding=10)
        process_box.grid(row=1, column=0, sticky="ew")
        self.process_entries = self._make_fields(
            process_box,
            [
                ("Initial value, r₀", "2", "$"),
                ("Drift, b", "0", "$/year"),
                ("Volatility, σ", "1", "$/√year"),
            ],
        )
        ttk.Label(
            process_box,
            text="Rₜ = r₀ + bt + σBₜ, absorbed at 0. No fixed sale horizon.",
            style="Muted.TLabel",
            wraplength=275,
            justify="left",
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))

        projection_box = ttk.LabelFrame(self.parameter_frame, text="Projection", padding=10)
        projection_box.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(projection_box, text="Statistical distance").grid(row=0, column=0, sticky="w")
        self.divergence_combo = ttk.Combobox(
            projection_box,
            textvariable=self.divergence_var,
            values=self.DIVERGENCES,
            state="readonly",
            width=22,
        )
        self.divergence_combo.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.divergence_combo.bind("<<ComboboxSelected>>", self._on_divergence_change)
        self.alpha_frame = ttk.Frame(projection_box)
        self.alpha_frame.grid(row=2, column=0, sticky="w", pady=(7, 0))
        ttk.Label(self.alpha_frame, text="Rényi order α").grid(row=0, column=0, sticky="w")
        self.alpha_entry = tk.Entry(self.alpha_frame, width=8, textvariable=self.alpha_var,
                                    justify="right", relief="solid", borderwidth=1)
        self.alpha_entry.grid(row=0, column=1, padx=(8, 0))
        self.alpha_entry.bind("<KeyRelease>", lambda _event: self._invalidate_projection())
        self.alpha_entry.bind("<FocusOut>", lambda _event: self._invalidate_projection())
        self.alpha_frame.grid_remove()
        ttk.Label(
            projection_box,
            text="Negative scaled mean uses compact CM′; the visible upper range is its disclosed support cap.",
            style="Muted.TLabel",
            wraplength=275,
            justify="left",
        ).grid(row=3, column=0, sticky="w", pady=(7, 0))

        grid_box = ttk.LabelFrame(self.parameter_frame, text="Distribution grid", padding=10)
        grid_box.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        self.grid_entries = self._make_fields(grid_box, [("Number of states", "20", "")])
        ttk.Separator(grid_box).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(7, 8))
        self.range_label = ttk.Label(grid_box, text="Sale-price range: automatic")
        self.range_label.grid(row=2, column=0, columnspan=3, sticky="w")
        range_buttons = ttk.Frame(grid_box)
        range_buttons.grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Button(range_buttons, text="Auto", command=self.reset_auto_range).grid(row=0, column=0)
        ttk.Button(range_buttons, text="Extend upper", command=self.extend_upper_range).grid(row=0, column=1, padx=(6, 0))

        button_row = ttk.Frame(self.parameter_frame)
        button_row.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        button_row.columnconfigure(0, weight=1)
        ttk.Button(button_row, text="Apply inputs", command=self.apply_inputs).grid(row=0, column=0, sticky="ew")
        ttk.Button(button_row, text="Project", command=self.run_projection).grid(row=0, column=1, padx=(6, 0))
        ttk.Button(button_row, text="Undo", command=self.undo).grid(row=0, column=2, padx=(6, 0))
        ttk.Button(button_row, text="Clear", command=self.clear_distribution).grid(row=0, column=3, padx=(6, 0))

        self.error_label = ttk.Label(
            self.parameter_frame, text="", foreground="#b42318", wraplength=290, justify="left"
        )
        self.error_label.grid(row=5, column=0, sticky="ew", pady=(10, 0))
        ttk.Label(
            self.parameter_frame,
            text="Method: Jin–Sturm extended f-divergence projection.\nSee README.md.",
            style="Muted.TLabel",
            wraplength=290,
            justify="left",
        ).grid(row=6, column=0, sticky="w", pady=(8, 0))

        self.builder_frame = ttk.Frame(self.root, padding=(2, 14, 16, 12))
        self.builder_frame.grid(row=0, column=1, sticky="nsew")
        self.builder_frame.columnconfigure(0, weight=1)

        header = ttk.Frame(self.builder_frame)
        header.grid(row=0, column=0, sticky="ew", padx=10)
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Target sale-price distribution", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        self.allocation_label = ttk.Label(header, text="0 / 20 states allocated", style="Muted.TLabel")
        self.allocation_label.grid(row=0, column=1, sticky="e")

        self.attainability_label = ttk.Label(self.builder_frame, text="Scaled mean: —", style="Section.TLabel")
        self.attainability_label.grid(row=1, column=0, sticky="w", padx=10, pady=(11, 2))
        self.result_label = ttk.Label(
            self.builder_frame, text="Build a complete target, then select Project.", style="Muted.TLabel",
            wraplength=650, justify="left"
        )
        self.result_label.grid(row=2, column=0, sticky="w", padx=10)

        chart_options = ttk.Frame(self.builder_frame)
        chart_options.grid(row=3, column=0, sticky="ew", padx=10, pady=(7, 0))
        chart_options.columnconfigure(1, weight=1)
        ttk.Label(chart_options, text="■", foreground="#3b9b70").grid(row=0, column=0)
        ttk.Label(chart_options, text="Target", style="Muted.TLabel").grid(row=0, column=1, sticky="w", padx=(2, 12))
        ttk.Label(chart_options, text="━", foreground="#d2762d").grid(row=0, column=2)
        ttk.Label(chart_options, text="Projection", style="Muted.TLabel").grid(row=0, column=3, sticky="w", padx=(2, 15))
        self.y_scale_label = ttk.Label(chart_options, text="Y-axis: automatic", style="Muted.TLabel")
        self.y_scale_label.grid(row=0, column=4, padx=(0, 8))
        ttk.Checkbutton(
            chart_options, text="Show full height", variable=self.full_height_var, command=self.draw_grid
        ).grid(row=0, column=5)

        self.canvas = tk.Canvas(
            self.builder_frame,
            width=self.CANVAS_W,
            height=self.CANVAS_H,
            background="#ffffff",
            highlightthickness=1,
            highlightbackground="#c7cdd1",
            cursor="crosshair",
        )
        self.canvas.grid(row=4, column=0, padx=10, pady=(7, 4), sticky="n")
        self.canvas.bind("<Button-1>", self._start_draw)
        self.canvas.bind("<B1-Motion>", self._drag_draw)
        self.canvas.bind("<ButtonRelease-1>", self._stop_draw)
        self.canvas.bind("<Motion>", self._show_hover)
        self.canvas.bind("<Leave>", lambda _event: self._set_instruction())

        self.instruction_label = ttk.Label(self.builder_frame, text="", style="Muted.TLabel")
        self.instruction_label.grid(row=5, column=0, sticky="w", padx=10)

    def _make_fields(self, parent, specs) -> list[tk.Entry]:
        entries = []
        validation = (self.root.register(self._validate_typing), "%P")
        for row, (label, initial, unit) in enumerate(specs):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
            entry = tk.Entry(
                parent, width=10, justify="right", relief="solid", borderwidth=1,
                validate="key", validatecommand=validation,
            )
            entry.insert(0, initial)
            entry.grid(row=row, column=1, padx=(10, 6), pady=4)
            entry.bind("<FocusIn>", lambda event: event.widget.select_range(0, tk.END))
            ttk.Label(parent, text=unit, style="Muted.TLabel").grid(row=row, column=2, sticky="w")
            entries.append(entry)
        return entries

    @staticmethod
    def _validate_typing(proposed: str) -> bool:
        return bool(re.fullmatch(r"[+-]?(?:\d*(?:\.\d*)?)?(?:[eE][+-]?\d*)?", proposed))

    def _clear_errors(self) -> None:
        for entry in self.process_entries + self.grid_entries:
            entry.configure(background="white")
        self.error_label.configure(text="")

    def _mark_error(self, entry: tk.Entry, message: str) -> None:
        self._clear_errors()
        entry.configure(background="#fde7e5")
        entry.focus_set()
        entry.select_range(0, tk.END)
        self.error_label.configure(text=message)

    def _parse_number(self, entry: tk.Entry, label: str, index: int) -> float:
        try:
            value = float(entry.get())
        except ValueError:
            raise ValueError(index, f"Enter a valid number for {label}.") from None
        if not math.isfinite(value):
            raise ValueError(index, f"{label} must be finite.")
        return value

    def apply_inputs(self) -> None:
        self._clear_errors()
        try:
            r0 = self._parse_number(self.process_entries[0], "initial value", 0)
            drift = self._parse_number(self.process_entries[1], "drift", 1)
            volatility = self._parse_number(self.process_entries[2], "volatility", 2)
            process = ShiftedArithmeticBrownian(r0, drift, volatility)
            raw_dimension = self._parse_number(self.grid_entries[0], "number of states", 3)
            dimension = int(raw_dimension)
            if raw_dimension != dimension or not 2 <= dimension <= 250:
                raise ValueError(3, "Number of states must be a whole number from 2 to 250.")
        except ValueError as error:
            index = error.args[0] if len(error.args) == 2 and isinstance(error.args[0], int) else None
            message = error.args[1] if index is not None else str(error)
            entries = self.process_entries + self.grid_entries
            if index is not None:
                self._mark_error(entries[index], message)
            else:
                self.error_label.configure(text=message)
            return

        if dimension != self.dimension:
            self.dimension = dimension
            self.col_count = np.zeros(dimension, dtype=int)
            self.undo_stack.clear()
        self.process = process
        if self.range_is_auto:
            self.from_, self.to = 0.0, self._nice_ceiling(2.0 * process.r0)
        self.col_labels = np.linspace(self.from_, self.to, self.dimension)
        self._invalidate_projection(redraw=False)
        self._refresh_range_label()
        self.draw_grid()
        self._update_target_status()

    @staticmethod
    def _nice_ceiling(value: float) -> float:
        value = max(value, 1e-9)
        exponent = math.floor(math.log10(value))
        scale = 10.0**exponent
        fraction = value / scale
        nice = next(candidate for candidate in (1.0, 2.0, 2.5, 5.0, 10.0) if fraction <= candidate)
        return nice * scale

    def _refresh_range_label(self) -> None:
        mode = "auto" if self.range_is_auto else "custom"
        self.range_label.configure(text=f"Sale-price range: $0 to ${self._format_number(self.to)}  ({mode})")

    def reset_auto_range(self) -> None:
        if self.process is None:
            return
        self.range_is_auto = True
        self.from_, self.to = 0.0, self._nice_ceiling(2.0 * self.process.r0)
        self.col_labels = np.linspace(self.from_, self.to, self.dimension)
        self._invalidate_projection(redraw=False)
        self._refresh_range_label()
        self.draw_grid()
        self._update_target_status()

    def extend_upper_range(self) -> None:
        self.to *= 1.5
        self.range_is_auto = False
        self.col_labels = np.linspace(0.0, self.to, self.dimension)
        self._invalidate_projection(redraw=False)
        self._refresh_range_label()
        self.draw_grid()
        self._update_target_status()

    def _on_divergence_change(self, _event=None) -> None:
        if self.divergence_var.get() == "Rényi":
            self.alpha_frame.grid()
        else:
            self.alpha_frame.grid_remove()
        self._invalidate_projection()

    def _invalidate_projection(self, redraw: bool = True) -> None:
        self.projection = None
        if hasattr(self, "result_label"):
            self.result_label.configure(text="Projection not calculated for the current target.", style="Muted.TLabel")
        if redraw and hasattr(self, "canvas"):
            self.draw_grid()

    def _target_distribution(self) -> tuple[np.ndarray, np.ndarray]:
        mask = self.col_count > 0
        return self.col_labels[mask], self.col_count[mask].astype(float) / self.dimension

    def _update_target_status(self) -> None:
        allocated = int(self.col_count.sum())
        self.allocation_label.configure(text=f"{allocated} / {self.dimension} states allocated")
        if self.process is None or allocated != self.dimension:
            remaining = self.dimension - allocated
            self.attainability_label.configure(text="Scaled mean: —")
            self.result_label.configure(
                text=f"Allocate {remaining} more state{'s' if remaining != 1 else ''} to complete the target.",
                style="Muted.TLabel",
            )
            return
        x, p = self._target_distribution()
        try:
            scale = self.process.scale(x)
        except ProjectionError as error:
            self.attainability_label.configure(text="Scaled mean: unavailable")
            self.result_label.configure(text=str(error), style="Muted.TLabel")
            return
        scaled_mean = float(math.fsum(float(s) * float(weight) for s, weight in zip(scale, p)))
        tolerance = 1e-12 * max(1.0, float(np.dot(np.abs(scale), p)))
        if scaled_mean > tolerance:
            status = "not attainable or super-attainable — projection B′ applies"
        elif scaled_mean < -tolerance:
            if self.process.upper_scale_is_infinite:
                status = "attainable but FOSD-improvable — compact projection CM′ applies"
            else:
                status = "not attainable, but super-attainable — compact projection CM′ applies"
        else:
            status = "attainable and FOSD-maximal"
        self.attainability_label.configure(text=f"Scaled mean: {scaled_mean:.6g}  •  {status}")

    def run_projection(self) -> None:
        self._clear_errors()
        if self.process is None or int(self.col_count.sum()) != self.dimension:
            self.error_label.configure(text="Complete the target distribution before projecting it.")
            return
        x, p = self._target_distribution()
        try:
            alpha = 2.0
            if self.divergence_var.get() == "Rényi":
                try:
                    alpha = float(self.alpha_var.get())
                except ValueError:
                    raise ProjectionError("Enter a numeric Rényi order α (0.1–5; choose 1 for the KL limit).") from None
            result = project_distribution(
                x, p, self.process, self.divergence_var.get(), cap=self.to, alpha=alpha
            )
        except ProjectionError as error:
            self.projection = None
            self.error_label.configure(text=str(error))
            self.result_label.configure(text="Projection unavailable.", style="Muted.TLabel")
            self.draw_grid()
            return
        self.projection = result
        additions = result.projected_mass - result.target_mass
        new_mass = float(additions[result.target_mass == 0].sum())
        largest_change_pct = 100 * float(np.max(np.abs(additions)))
        change_text = f"{largest_change_pct:.2f}" if largest_change_pct >= .005 else f"{largest_change_pct:.3g}"
        distance_text = (f"≈0 (below numerical precision)" if result.objective == 0
                         and result.problem != "Already attainable" else f"{result.objective:.6g}")
        detail = (f"{result.problem} • {result.divergence} = {distance_text}"
                  f" • projected scaled mean = {result.projected_scaled_mean:.3g}"
                  f" • largest mass change = {change_text}%")
        if result.cap is not None:
            detail += f" • cap M = {self._format_number(result.cap)}"
        if new_mass > 0:
            locations = result.support[(result.target_mass == 0) & (result.projected_mass > 0)]
            mass_pct = 100 * new_mass
            mass_text = f"{mass_pct:.2f}" if mass_pct >= .005 else f"{mass_pct:.3g}"
            detail += f" • new endpoint mass {mass_text}% at " + ", ".join(self._format_number(v) for v in locations)
        self.result_label.configure(text=detail + ". " + result.message, style="Result.TLabel")
        self.draw_grid()

    def _format_number(self, value: float, step: float | None = None) -> str:
        if abs(value) < 5e-13:
            value = 0.0
        magnitude = abs(value)
        if magnitude and (magnitude >= 1e5 or magnitude < 1e-3):
            return f"{value:.3g}"
        if step is None or step == 0:
            return f"{value:g}"
        decimals = max(0, min(6, int(-math.floor(math.log10(abs(step)))) + 1))
        if decimals == 0:
            return f"{value:.0f}"
        return f"{value:.{decimals}f}".rstrip("0").rstrip(".")

    @staticmethod
    def _tick_indices(count: int, maximum: int) -> list[int]:
        if count <= maximum:
            return list(range(count))
        stride = math.ceil((count - 1) / (maximum - 1))
        indices = list(range(0, count, stride))
        if indices[-1] != count - 1:
            indices.append(count - 1)
        return indices

    def _projected_counts(self) -> np.ndarray:
        values = np.zeros(self.dimension)
        if self.projection is None:
            return values
        for x, mass in zip(self.projection.support, self.projection.projected_mass):
            index = int(np.argmin(np.abs(self.col_labels - x)))
            values[index] += mass * self.dimension
        return values

    def _automatic_y_limit(self) -> int:
        baseline = min(self.dimension, max(4, int(math.ceil(math.sqrt(self.dimension)))))
        tallest = max(float(self.col_count.max()), float(self._projected_counts().max()))
        if tallest <= baseline:
            return baseline
        return min(self.dimension, max(int(math.ceil(tallest)) + 1, int(math.ceil(tallest * 1.2))))

    def _step_points(self, values: np.ndarray, cell_w: float, cell_h: float):
        bottom = self.TOP + self.PLOT_H
        points = [(self.LEFT, bottom)]
        for index, value in enumerate(values):
            x0 = self.LEFT + index * cell_w
            x1 = x0 + cell_w
            y = bottom - value * cell_h
            points.extend(((x0, y), (x1, y)))
        points.append((self.LEFT + self.PLOT_W, bottom))
        return points

    def draw_grid(self) -> None:
        c = self.canvas
        c.delete("all")
        right = self.LEFT + self.PLOT_W
        bottom = self.TOP + self.PLOT_H
        self.y_limit = self.dimension if self.full_height_var.get() else self._automatic_y_limit()
        self.y_scale_label.configure(text=f"Y-axis: 0–{self.y_limit}" + (" (full)" if self.full_height_var.get() else " (auto)"))
        cell_w = self.PLOT_W / self.dimension
        cell_h = self.PLOT_H / self.y_limit

        c.create_rectangle(self.LEFT, self.TOP, right, bottom, fill="#fbfcfd", outline="#8b979e")
        for index in self._tick_indices(self.dimension + 1, 21):
            x = self.LEFT + index * cell_w
            c.create_line(x, self.TOP, x, bottom, fill="#e5e9ec")
        for value in self._tick_indices(self.y_limit + 1, 10):
            y = bottom - value * cell_h
            c.create_line(self.LEFT, y, right, y, fill="#e5e9ec")

        if np.any(self.col_count):
            c.create_polygon(self._step_points(self.col_count, cell_w, cell_h), fill="#3b9b70", outline="#217653", width=1)
        projected = self._projected_counts()
        if np.any(projected):
            c.create_line(self._step_points(projected, cell_w, cell_h)[:-1], fill="#d2762d", width=3, joinstyle="miter")

        x_indices = self._tick_indices(self.dimension, max(2, self.PLOT_W // 78))
        step = (self.to - self.from_) / (self.dimension - 1)
        for index in x_indices:
            x = self.LEFT + (index + 0.5) * cell_w
            c.create_line(x, bottom, x, bottom + 5, fill="#66737a")
            c.create_text(x, bottom + 17, text=self._format_number(float(self.col_labels[index]), step), fill="#34444c", font=("TkDefaultFont", 9))
        for value in self._tick_indices(self.y_limit + 1, 9):
            y = bottom - value * cell_h
            c.create_text(self.LEFT - 9, y, text=str(value), anchor="e", fill="#53636b", font=("TkDefaultFont", 9))
        c.create_text((self.LEFT + right) / 2, bottom + 45, text="Sale price", fill="#27343a")
        c.create_text(16, (self.TOP + bottom) / 2, text="State-equivalent mass", angle=90, fill="#27343a")

    def _event_to_column_count(self, event: tk.Event) -> tuple[int, int] | None:
        if not (self.LEFT <= event.x <= self.LEFT + self.PLOT_W and self.TOP <= event.y <= self.TOP + self.PLOT_H):
            return None
        column = min(self.dimension - 1, int((event.x - self.LEFT) * self.dimension / self.PLOT_W))
        count = min(self.y_limit, max(0, int(math.ceil((self.TOP + self.PLOT_H - event.y) * self.y_limit / self.PLOT_H))))
        return column, count

    def _start_draw(self, event: tk.Event) -> None:
        hit = self._event_to_column_count(event)
        if hit is None:
            return
        self.undo_stack.append(self.col_count.copy())
        self.undo_stack = self.undo_stack[-30:]
        self.dragging = True
        self._apply_draw(*hit)

    def _drag_draw(self, event: tk.Event) -> None:
        if self.dragging:
            hit = self._event_to_column_count(event)
            if hit is not None:
                self._apply_draw(*hit)

    def _stop_draw(self, _event: tk.Event) -> None:
        self.dragging = False

    def _apply_draw(self, column: int, clicked_level: int) -> None:
        current = int(self.col_count[column])
        desired = clicked_level - 1 if 0 < clicked_level <= current else clicked_level
        available = self.dimension - (int(self.col_count.sum()) - current)
        self.col_count[column] = min(max(0, desired), max(0, available))
        self._invalidate_projection(redraw=False)
        self.draw_grid()
        self._update_target_status()

    def _show_hover(self, event: tk.Event) -> None:
        hit = self._event_to_column_count(event)
        if hit is None:
            self._set_instruction()
            return
        column, level = hit
        current = int(self.col_count[column])
        target = level - 1 if 0 < level <= current else level
        self.instruction_label.configure(
            text=f"Sale price {self._format_number(float(self.col_labels[column]))}: click to set {target} states"
        )

    def _set_instruction(self) -> None:
        self.instruction_label.configure(text="Draw the target in green  •  Projected mass appears as an orange outline  •  Ctrl+Z to undo")

    def clear_distribution(self) -> None:
        if np.any(self.col_count):
            self.undo_stack.append(self.col_count.copy())
        self.col_count[:] = 0
        self._invalidate_projection(redraw=False)
        self.draw_grid()
        self._update_target_status()

    def undo(self) -> None:
        if not self.undo_stack:
            return
        previous = self.undo_stack.pop()
        if previous.size == self.dimension:
            self.col_count = previous
            self._invalidate_projection(redraw=False)
            self.draw_grid()
            self._update_target_status()


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
