"""Finite-dimensional extended-f-divergence projections for optimal selling.

This module implements the equality formulations B' and compact CM' from
Jin and Sturm's paper for a shifted arithmetic Brownian asset absorbed at 0.
It contains no GUI code and accepts ordinary NumPy-compatible arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize
from scipy.special import xlogy


class ProjectionError(ValueError):
    """Raised when inputs are invalid or the finite program is infeasible."""


@dataclass(frozen=True)
class ProjectionResult:
    support: np.ndarray
    target_mass: np.ndarray
    projected_mass: np.ndarray
    scale_values: np.ndarray
    target_scaled_mean: float
    projected_scaled_mean: float
    problem: str
    divergence: str
    objective: float
    cap: float | None
    message: str


class ShiftedArithmeticBrownian:
    """R_t = r0 + b t + sigma B_t, absorbed when it reaches zero."""

    def __init__(self, initial_value: float, drift: float, volatility: float):
        self.r0 = float(initial_value)
        self.b = float(drift)
        self.sigma = float(volatility)
        if not all(math.isfinite(v) for v in (self.r0, self.b, self.sigma)):
            raise ProjectionError("Process parameters must be finite.")
        if self.r0 <= 0:
            raise ProjectionError("Initial asset value must be greater than zero.")
        if self.sigma <= 0:
            raise ProjectionError("Volatility must be greater than zero.")

    def scale(self, values: Iterable[float]) -> np.ndarray:
        x = np.asarray(values, dtype=float)
        if np.any(~np.isfinite(x)) or np.any(x < 0):
            raise ProjectionError("Sale-price support must be finite and non-negative.")
        if abs(self.b) <= 1e-12 * max(1.0, self.sigma**2):
            return x - self.r0
        exponent = -2.0 * self.b * (x - self.r0) / self.sigma**2
        if np.any(exponent > 700):
            raise ProjectionError("Scale function exceeds floating-point range on this support; reduce drift, enlarge volatility, or narrow the payoff range.")
        values = -(self.sigma**2 / (2.0 * self.b)) * np.expm1(exponent)
        if np.any(~np.isfinite(values)):
            raise ProjectionError("Scale function overflowed; reduce drift or enlarge volatility.")
        return values

    @property
    def upper_scale_is_infinite(self) -> bool:
        return self.b <= 0


def _combine_support(x: np.ndarray, p: np.ndarray, extra: float | None) -> tuple[np.ndarray, np.ndarray]:
    support = np.unique(np.append(x, [] if extra is None else extra))
    target = np.zeros(support.size, dtype=float)
    for value, mass in zip(x, p):
        index = int(np.argmin(np.abs(support - value)))
        target[index] += mass
    return support, target


def _linear_parts(scale: np.ndarray, target: np.ndarray, fosd: bool):
    scale_norm = max(1.0, float(np.max(np.abs(scale))))
    equalities = np.vstack((np.ones(scale.size), scale / scale_norm))
    rhs = np.array([1.0, 0.0])
    if fosd and scale.size > 1:
        cumulative = np.tril(np.ones((scale.size - 1, scale.size)))
        upper = np.cumsum(target)[:-1]
    else:
        cumulative = np.zeros((0, scale.size))
        upper = np.zeros(0)
    return equalities, rhs, cumulative, upper


def _find_feasible(equalities, rhs, cumulative, upper, bounds) -> np.ndarray:
    result = linprog(
        np.zeros(equalities.shape[1]),
        A_ub=cumulative if cumulative.size else None,
        b_ub=upper if cumulative.size else None,
        A_eq=equalities,
        b_eq=rhs,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise ProjectionError(
            "No feasible projection exists on the selected support. "
            "For a dominating projection, extend the upper support cap; for "
            "KL or Rényi, the original target support must straddle the zero of the scale function."
        )
    return np.maximum(result.x, 0.0)


def _interior_start(target, equalities, rhs, cumulative, upper, bounds, fallback):
    """Maximize the minimum positive-target density ratio before smooth solves."""
    n = len(target)
    rows = np.c_[-np.eye(n), target]
    if cumulative.size:
        rows = np.vstack((rows, np.c_[cumulative, np.zeros(cumulative.shape[0])]))
    limits = np.r_[np.zeros(n), upper] if cumulative.size else np.zeros(n)
    solution = linprog(
        np.r_[np.zeros(n), -1.0], A_ub=rows, b_ub=limits,
        A_eq=np.c_[equalities, np.zeros((2, 1))], b_eq=rhs,
        bounds=bounds + [(0.0, 1.0)], method="highs",
    )
    if solution.success and solution.x[-1] > 1e-10:
        return solution.x[:-1]
    return fallback


def _solve_smooth(name, alpha, target, equalities, rhs, cumulative, upper, bounds, initial):
    positive = target > 0
    tiny = 1e-14

    if name == "Kullback–Leibler":
        def objective(q):
            return float(np.sum(xlogy(q[positive], q[positive] / target[positive])))
        def gradient(q):
            out = np.zeros_like(q)
            out[positive] = np.log(np.maximum(q[positive], tiny) / target[positive]) + 1.0
            return out
    elif name == "Rényi":
        def objective(q):
            ratios = np.maximum(q[positive], 0.0) / target[positive]
            power = float(np.dot(target[positive], ratios ** alpha))
            return power if alpha > 1 else -power
        def gradient(q):
            out = np.zeros_like(q)
            ratios = np.maximum(q[positive], tiny * target[positive]) / target[positive]
            out[positive] = (1 if alpha > 1 else -1) * alpha * ratios ** (alpha - 1)
            return out
    elif name == "Squared Hellinger":
        root_target = np.sqrt(target)
        def objective(q):
            # H² = 1 - BC on probability measures. An artificial clipped
            # sqrt(q) objective can trap SLSQP at oscillatory zero densities.
            return float(1.0 - np.dot(root_target, np.sqrt(np.maximum(q, 0.0))))
        def gradient(q):
            out = np.zeros_like(q)
            out[positive] = -0.5 * root_target[positive] / np.sqrt(
                np.maximum(q[positive], tiny * target[positive])
            )
            return out
    else:
        raise ProjectionError(f"Unknown divergence: {name}")

    constraints = [LinearConstraint(equalities, rhs, rhs)]
    if cumulative.size:
        constraints.append(LinearConstraint(cumulative, -np.inf, upper))
    start = _interior_start(target, equalities, rhs, cumulative, upper, bounds, initial)
    result = minimize(
        objective,
        start,
        jac=gradient,
        method="SLSQP",
        bounds=Bounds([b[0] for b in bounds], [1.0 if b[1] is None else b[1] for b in bounds]),
        constraints=constraints,
        options={"ftol": 1e-11, "maxiter": 2000, "disp": False},
    )
    if not result.success:
        raise ProjectionError(f"{name} optimization failed: {result.message}")
    q = np.maximum(result.x, 0.0)
    if np.any((target > 0) & (q <= 1e-13)) and (name == "Squared Hellinger" or name == "Rényi" and alpha < 1):
        raise ProjectionError("Optimizer reached a zero-density boundary; refine the support or choose another distance.")
    gradient_at_q = gradient(q)
    certificate = linprog(
        gradient_at_q, A_ub=cumulative if cumulative.size else None,
        b_ub=upper if cumulative.size else None, A_eq=equalities, b_eq=rhs,
        bounds=bounds, method="highs",
    )
    if not certificate.success or float(np.dot(gradient_at_q, q) - certificate.fun) > 2e-5:
        raise ProjectionError("Projection did not pass its convex optimality check; no potentially oscillatory result was shown.")
    value = float(objective(q))
    if name == "Rényi":
        power = value if alpha > 1 else -value
        value = math.log(power) / (alpha - 1)
    return q, max(0.0, value)


def _solve_rescaled(name, alpha, target, scale, fosd):
    """Epigraph cutting-plane LP in scale-weighted masses for stiff drifts.

    z_i = q_i max(1, |S_i|). Both moment rows then have coefficients <=1;
    tiny economically necessary probabilities remain representable as ordinary z.
    Smooth convex objectives are supported by tangent underestimators and an
    explicit upper-minus-lower objective gap. TV is an exact LP and its unique
    optimum needs no secondary optimization.
    """
    n = len(target)
    weight = 1.0 / np.maximum(1.0, np.abs(scale))
    base_eq = np.vstack((weight, weight * scale))
    eq = np.c_[base_eq, np.zeros((2, n))]
    cdf = np.tril(np.ones((n - 1, n))) * weight if fosd else np.empty((0, n))
    cdf_limits = np.cumsum(target)[:-1] if fosd else np.empty(0)
    finite_recession = name in {"Squared Hellinger", "Total variation"} or (name == "Rényi" and alpha < 1)
    z_bounds = [(0.0, None if mass > 0 or finite_recession else 0.0) for mass in target]

    def optimize(c, rows, limits, bounds, iterations=1):
        answer = linprog(c, A_ub=np.vstack(rows), b_ub=np.concatenate(limits),
                         A_eq=eq, b_eq=[1.0, 0.0], bounds=bounds, method="highs")
        if not answer.success:
            raise ProjectionError("Scale-rescaled projection infeasible or LP solver failed: " + answer.message)
        return answer

    if name == "Total variation":
        diagonal = np.diag(weight)
        tv_rows = np.vstack((np.c_[diagonal, -np.eye(n)],
                             np.c_[-diagonal, -np.eye(n)]))
        tv_limits = np.r_[target, -target]
        rows = [tv_rows]
        limits = [tv_limits]
        if fosd:
            rows.append(np.c_[cdf, np.zeros((n - 1, n))])
            limits.append(cdf_limits)
        primary = np.r_[np.zeros(n), np.full(n, .5)]
        bounds = z_bounds + [(0.0, None)] * n
        answer = optimize(primary, rows, limits, bounds)
        q = weight * answer.x[:n]
        actual_tv = 0.5 * float(np.sum(np.abs(q - target)))
        if actual_tv > float(answer.fun) + 1e-7:
            raise ProjectionError("The TV LP returned inconsistent absolute-value variables.")
        return q, actual_tv

    positive = np.flatnonzero(target > 0)
    k = len(positive)
    eq = np.c_[base_eq, np.zeros((2, k))]
    rows = [np.c_[cdf, np.zeros((n - 1, k))]] if fosd else []
    limits = [cdf_limits] if fosd else []
    bounds = z_bounds + [(0.0 if name == "Rényi" and alpha > 1 else -1.0, None)] * k
    objective = np.r_[np.zeros(n), np.ones(k)]

    def f_and_slope(z, index):
        p = target[index]
        w = weight[index]
        q = w * z
        ratio = q / p
        if name == "Squared Hellinger":
            return -math.sqrt(p * q), -0.5 * w * math.sqrt(p / q)
        if name == "Kullback–Leibler":
            return q * math.log(ratio), w * (math.log(ratio) + 1)
        power = p * ratio ** alpha
        slope = alpha * w * ratio ** (alpha - 1)
        return (power, slope) if alpha > 1 else (-power, -slope)

    def tangent(z_values):
        matrix = np.zeros((k, n + k))
        limits_here = np.zeros(k)
        for j, i in enumerate(positive):
            z = max(float(z_values[i]), float(target[i]) * 1e-9)
            f, slope = f_and_slope(z, i)
            matrix[j, i] = slope
            matrix[j, n + j] = -1.0
            limits_here[j] = slope * z - f
        rows.append(matrix)
        limits.append(limits_here)

    for trial in (target * 1e-5, target * .01, target, np.ones(n)):
        tangent(trial)
    for _ in range(110):
        answer = optimize(objective, rows, limits, bounds)
        z = answer.x[:n]
        # f(0)=0 for all four convex objectives; evaluate exactly at zero.
        actual = sum(f_and_slope(float(z[i]), i)[0] for i in positive if z[i] > 0)
        gap = actual - float(answer.fun)
        if gap <= 2e-6:
            q = z * weight
            if name == "Squared Hellinger":
                value = 1.0 + actual
            elif name == "Rényi":
                value = math.log(actual if alpha > 1 else -actual) / (alpha - 1)
            else:
                value = actual
            return q, max(0.0, float(value))
        tangent(z)
    raise ProjectionError("Rescaled convex projection did not converge within its certified error tolerance.")


def project_distribution(
    support: Iterable[float],
    masses: Iterable[float],
    process: ShiftedArithmeticBrownian,
    divergence: str = "Kullback–Leibler",
    cap: float | None = None,
    tolerance: float = 1e-12,
    alpha: float = 2.0,
) -> ProjectionResult:
    """Project a target law onto the paper's optimal attainable set.

    Positive scaled mean invokes B' (no FOSD constraint). Negative scaled mean
    invokes compact CM' (FOSD plus a visible cap). Zero scaled mean returns the
    target unchanged. Supported divergences are KL, Rényi (0<α<1 or α>1),
    squared Hellinger and total variation.
    """
    x = np.asarray(list(support), dtype=float)
    p = np.asarray(list(masses), dtype=float)
    if x.ndim != 1 or p.ndim != 1 or x.size == 0 or x.size != p.size:
        raise ProjectionError("Support and masses must be non-empty vectors of equal length.")
    if np.any(~np.isfinite(x)) or np.any(x < 0):
        raise ProjectionError("Target support must be finite and non-negative.")
    if np.any(~np.isfinite(p)) or np.any(p < 0) or p.sum() <= 0:
        raise ProjectionError("Target masses must be finite, non-negative and have positive total mass.")
    p = p / p.sum()
    order = np.argsort(x)
    x, p = x[order], p[order]
    # Coalesce repeated atoms.
    unique_x, inverse = np.unique(x, return_inverse=True)
    unique_p = np.zeros(unique_x.size)
    np.add.at(unique_p, inverse, p)
    x, p = unique_x, unique_p

    scaled_target = process.scale(x)
    target_mean = float(math.fsum(float(a) * float(b) for a, b in zip(scaled_target, p)))
    if divergence == "Rényi (α=2)":  # Compatibility with the earlier API.
        divergence, alpha = "Rényi", 2.0
    if divergence not in {"Kullback–Leibler", "Rényi", "Squared Hellinger", "Total variation"}:
        raise ProjectionError(f"Unsupported divergence: {divergence}")
    if divergence == "Rényi" and (not math.isfinite(alpha) or not (0.1 <= alpha <= 5.0)):
        raise ProjectionError("Rényi α must be between 0.1 and 5.")
    renyi_kl_limit = divergence == "Rényi" and alpha == 1.0
    if renyi_kl_limit:
        divergence = "Kullback–Leibler"
    if divergence == "Rényi" and abs(alpha - 1.0) < 0.01:
        raise ProjectionError("For numerical stability use α=1 (exact KL limit) within 0.01 of one.")
    divergence_label = "Rényi (α=1; KL limit)" if renyi_kl_limit else (f"Rényi (α={alpha:g})" if divergence == "Rényi" else divergence)
    # Test the weighted mean, not the most extreme scale value: otherwise
    # large scales can incorrectly label a visibly nonzero mean as zero.
    weighted_scale = float(np.dot(np.abs(scaled_target), p))
    mean_tolerance = max(64 * np.finfo(float).eps * weighted_scale,
                         tolerance * max(1.0, weighted_scale))
    if abs(target_mean) <= mean_tolerance and abs(target_mean) > 1e-7:
        raise ProjectionError("Scaled mean is numerically indeterminate at this parameter scale; no zero-distance result was shown.")
    if abs(target_mean) <= mean_tolerance:
        return ProjectionResult(
            x, p, p.copy(), process.scale(x), target_mean, target_mean,
            "Already attainable", divergence_label, 0.0, None,
            "The target already has zero scaled mean; no projection is required.",
        )

    finite_recession = divergence in {"Squared Hellinger", "Total variation"} or (divergence == "Rényi" and alpha < 1)

    if target_mean > 0:
        problem = "B′"
        fosd = False
        endpoint = 0.0 if finite_recession else None
        used_cap = None
        explanation = "Closest zero-scaled-mean law to a non-attainable target."
    else:
        problem = "CM′"
        fosd = True
        if cap is None or not math.isfinite(cap):
            raise ProjectionError("A finite upper support cap is required for the dominating projection.")
        if cap < float(x.max()) - tolerance:
            raise ProjectionError("The upper support cap cannot be below the target support.")
        endpoint = float(cap) if finite_recession else None
        used_cap = float(cap)
        explanation = "Closest zero-scaled-mean law that first-order stochastically dominates the target."

    candidate_x, target = _combine_support(x, p, endpoint)
    scale = process.scale(candidate_x)
    equalities, rhs, cumulative, upper = _linear_parts(scale, target, fosd)
    bounds = []
    for mass in target:
        if mass == 0.0 and not finite_recession:
            bounds.append((0.0, 0.0))
        else:
            bounds.append((0.0, 1.0))
    # Large |b| Δx / σ² makes S span tens of orders of magnitude. Direct
    # probability coordinates cause LP/SLSQP to treat required tiny masses as
    # zero. Rescale each probability by its own scale magnitude instead.
    if divergence == "Total variation" or float(np.max(np.abs(scale))) > 1e4:
        projected, objective = _solve_rescaled(divergence, alpha, target, scale, fosd)
    else:
        try:
            initial = _find_feasible(equalities, rhs, cumulative, upper, bounds)
            projected, objective = _solve_smooth(
                divergence, alpha, target, equalities, rhs, cumulative, upper, bounds, initial
            )
        except ProjectionError:
            # A directly scaled SLSQP optimum can be falsely rejected in the
            # intermediate-drift regime. The LP epigraph solver supplies its
            # own global gap certificate, so retry there before giving up.
            projected, objective = _solve_rescaled(divergence, alpha, target, scale, fosd)

    # Explicit post-solve checks: never present a numerically invalid law.
    norm_error = abs(float(projected.sum()) - 1.0)
    mean_error = abs(float(np.dot(scale, projected)))
    fosd_error = float(np.max(np.cumsum(projected)[:-1] - np.cumsum(target)[:-1])) if fosd and projected.size > 1 else 0.0
    if norm_error > 2e-8 or mean_error > 2e-8 * max(1.0, float(np.dot(np.abs(scale), projected))) or fosd_error > 2e-8:
        raise ProjectionError("The numerical solution did not satisfy the constraints accurately enough.")
    if objective <= 1e-12 and float(np.max(np.abs(projected - target))) > 1e-7:
        raise ProjectionError("Inconsistent zero-distance projection; no result was shown.")
    if np.array_equal(projected, target) and abs(target_mean) > mean_tolerance:
        raise ProjectionError("Projection could not change a target with nonzero scaled mean.")
    if float(np.max(np.abs(projected - target))) * len(target) < 0.01:
        explanation += " The mass change is below chart resolution; inspect the numerical values."
    if objective == 0.0 and not np.array_equal(projected, target):
        explanation += " The divergence is below floating-point display resolution (reported as approximately zero)."

    return ProjectionResult(
        candidate_x,
        target,
        projected,
        scale,
        target_mean,
        float(np.dot(scale, projected)),
        problem,
        divergence_label,
        objective,
        used_cap,
        explanation,
    )
