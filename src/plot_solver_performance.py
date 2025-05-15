from __future__ import annotations

import ast
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


_METRIC_LABEL = {
    "time_ms": "Time (ms)",
    "constraint_satisfaction": "Constraint satisfaction",
    "score": "Score",
}

_STAT_LABEL = {
    "mean": "Mean",
    "p90": "90th percentile",
}

_SOLVER_MARKERS = ["o", "s", "D", "^", "v", "X", "P", "*", "h"]
_SOLVER_STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]


def _solver_style(idx: int) -> dict:
    """Consistent marker/linestyle pair based on enumeration index."""
    return dict(
        marker=_SOLVER_MARKERS[idx % len(_SOLVER_MARKERS)],
        linestyle=_SOLVER_STYLES[idx % len(_SOLVER_STYLES)],
        linewidth=2,
        markersize=8,
    )


def _base_ax(
    xlabel: str,
    ylabel: str,
    title: str,
    figsize: tuple[int, int] = (8, 5),
) -> plt.Axes:
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return ax


# -----------------------------------------------------------------------------
# DATA LOADING & PRE‑PROCESSING
# -----------------------------------------------------------------------------


def _parse_constraints(x: str | list[str]) -> tuple[str, ...]:
    """Make *constraints* hashable (tuple) no matter the original format."""
    if isinstance(x, list):
        return tuple(x)
    try:
        # CSV has the list as a stringified Python list
        return tuple(ast.literal_eval(x))
    except Exception:
        # Fallback: treat as single‑string constraint id
        return (str(x),)


def load_csv(path: str | Path) -> pd.DataFrame:
    """Read the CSV and add helper columns."""
    df = pd.read_csv(path)
    df["constraints_tuple"] = df["constraints"].apply(_parse_constraints)
    return df


# -----------------------------------------------------------------------------
# CORE AGGREGATION
# -----------------------------------------------------------------------------

Stat = Literal["mean", "p90"]


def _aggregate(
    df: pd.DataFrame,
    group_cols: list[str],
    metric: str,
    stat: Stat,
) -> pd.DataFrame:
    """Return a tidy DF with one row per group and a single `value` col."""
    if stat == "mean":
        ser = df.groupby(group_cols)[metric].mean()
    elif stat == "p90":
        ser = df.groupby(group_cols)[metric].quantile(0.9)
    else:
        raise ValueError(f"Unknown stat {stat!r}; use 'mean' or 'p90'.")
    return ser.rename("value").reset_index()


# -----------------------------------------------------------------------------
# HIGH‑LEVEL PLOT FUNCTIONS
# -----------------------------------------------------------------------------

def plot_fixed_param(
    df: pd.DataFrame,
    *,
    fixed_col: Literal["N", "M"],
    fixed_val: int,
    varying_col: Literal["N", "M"],
    metric: Literal["constraint_satisfaction", "time_ms", "score"] = "constraint_satisfaction",
    stat: Stat = "mean",
    constraints_mode: Literal["all", "separate"] = "all",
    constraints_filter: Optional[tuple[str, ...]] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    savepath: Optional[str | Path] = None,
) -> list[plt.Axes]:
    """Plot one metric.

    Returns the list of Axes objects created (one or many).
    """
    if varying_col == fixed_col:
        raise ValueError("fixed_col and varying_col must differ.")

    work = df[df[fixed_col] == fixed_val].copy()

    if constraints_filter is not None:
        work = work[work["constraints_tuple"] == constraints_filter]

    # Decide plotting strategy ------------------------------------------------
    if constraints_mode == "separate" and constraints_filter is None:
        axes: list[plt.Axes] = []
        for c_tuple, c_df in work.groupby("constraints_tuple"):
            c_ax = plot_fixed_param(
                c_df,
                fixed_col=fixed_col,
                fixed_val=fixed_val,
                varying_col=varying_col,
                metric=metric,
                stat=stat,
                constraints_mode="all",  # already separated
                constraints_filter=c_tuple,
                ax=None,
                show=False,
            )[0]
            axes.append(c_ax)
        if show:
            plt.show()
        return axes

    # Aggregate --------------------------------------------------------------
    group_cols = [varying_col, "solver"]
    tidy = _aggregate(work, group_cols, metric, stat)

    # Labels ------------------------------------------------------------------
    metric_lbl = _METRIC_LABEL.get(metric, metric)
    stat_lbl = _STAT_LABEL.get(stat, stat)
    xlabel = varying_col
    ylabel = f"{metric_lbl} ({stat_lbl})"
    title = f"{metric_lbl} for varying {varying_col}  |  {fixed_col}={fixed_val}"

    if constraints_filter is not None:
        title += "\n" + ", ".join(constraints_filter)

    # Prepare axis ------------------------------------------------------------
    if ax is None:
        ax = _base_ax(xlabel, ylabel, title)

    # Draw lines --------------------------------------------------------------
    solvers = sorted(tidy["solver"].unique())
    for s_idx, solver in enumerate(solvers):
        subset = tidy[tidy["solver"] == solver]
        ax.plot(
            subset[varying_col],
            subset["value"],
            label=solver,
            **_solver_style(s_idx),
        )

    ax.legend(title="Solver", fontsize=9)

    if savepath:
        ax.figure.savefig(savepath, dpi=300)
    if show:
        plt.show()
    return [ax]


def plot_dual(
    df: pd.DataFrame,
    *,
    fixed_col: Literal["N", "M"],
    fixed_val: int,
    varying_col: Literal["N", "M"],
    metric_left: str,
    stat_left: Stat,
    metric_right: str,
    stat_right: Stat,
    constraints_mode: Literal["all", "separate"] = "all",
) -> plt.Axes:
    """Two metrics on the same chart (left & right y‑axes)."""
    metric_left_lbl = _METRIC_LABEL.get(metric_left, metric_left)
    metric_right_lbl = _METRIC_LABEL.get(metric_right, metric_right)
    stat_left_lbl = _STAT_LABEL.get(stat_left, stat_left)
    stat_right_lbl = _STAT_LABEL.get(stat_right, stat_right)

    ax_left = _base_ax(
        xlabel=varying_col,
        ylabel=f"{metric_left_lbl} ({stat_left_lbl})",
        title=f"{metric_left_lbl} & {metric_right_lbl}  |  {fixed_col}={fixed_val}",
        figsize=(9, 5),
    )
    ax_right = ax_left.twinx()
    ax_right.set_ylabel(f"{metric_right_lbl} ({stat_right_lbl})", fontsize=12)

    plot_fixed_param(
        df,
        fixed_col=fixed_col,
        fixed_val=fixed_val,
        varying_col=varying_col,
        metric=metric_left,
        stat=stat_left,
        constraints_mode=constraints_mode,
        ax=ax_left,
        show=False,
    )
    plot_fixed_param(
        df,
        fixed_col=fixed_col,
        fixed_val=fixed_val,
        varying_col=varying_col,
        metric=metric_right,
        stat=stat_right,
        constraints_mode=constraints_mode,
        ax=ax_right,
        show=False,
    )
    ax_left.legend(title="Solver", fontsize=9, loc="upper left")
    plt.show()
    return ax_left


if __name__ == "__main__":
    df = load_csv("results_id1_N10_solvers.csv")

    # Quick exploration
    print(df.head())
    print("Unique solvers:", df["solver"].unique())
    print("N range:", sorted(df["N"].unique()))
    print("M range:", sorted(df["M"].unique()))
    print("Total distinct constraint-lists:", df["constraints_tuple"].nunique())

    # mean constraint satisfaction, fixed N=10, vary M
    plot_fixed_param(df,
                         fixed_col="N", fixed_val=10, varying_col="M",
                         metric="constraint_satisfaction", stat="mean",
                         constraints_mode="all")

    plot_fixed_param(df,
                     fixed_col="N", fixed_val=10, varying_col="M",
                     metric="time_ms", stat="mean",
                     constraints_mode="all")

    # 90-th percentile runtime, fixed N=0, vary N
    plot_fixed_param(df,
                         fixed_col="N", fixed_val=10, varying_col="M",
                         metric="constraint_satisfaction", stat="p90",
                         constraints_mode="all")
    plot_fixed_param(df,
                     fixed_col="N", fixed_val=10, varying_col="M",
                     metric="time_ms", stat="p90",
                     constraints_mode="all")

    # --------- fixed M, vary N ---------
    df = load_csv("results_id1_M60_solvers.csv")

    # Quick exploration
    print(df.head())
    print("Unique solvers:", df["solver"].unique())
    print("N range:", sorted(df["N"].unique()))
    print("M range:", sorted(df["M"].unique()))
    print("Total distinct constraint-lists:", df["constraints_tuple"].nunique())

    plot_fixed_param(df,
                     fixed_col="M", fixed_val=60, varying_col="N",
                     metric="constraint_satisfaction", stat="mean",
                     constraints_mode="all")
    plot_fixed_param(df,
                     fixed_col="M", fixed_val=60, varying_col="N",
                     metric="time_ms", stat="mean",
                     constraints_mode="all")
    plot_fixed_param(df,
                     fixed_col="M", fixed_val=60, varying_col="N",
                     metric="constraint_satisfaction", stat="p90",
                     constraints_mode="all")
    plot_fixed_param(df,
                     fixed_col="M", fixed_val=60, varying_col="N",
                     metric="time_ms", stat="p90",
                     constraints_mode="all")

    # draw per-constraint lines
    # plot_fixed_param(df,
    #                      fixed_col="N", fixed_val=10, varying_col="M",
    #                      metric="time_ms", stat="mean",
    #                      constraints_mode="separate")