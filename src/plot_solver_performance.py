from __future__ import annotations

import ast
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from adjustText import adjust_text


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
_SOLVER_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # teal
    "#aec7e8",  # light blue
    "#ffbb78",  # light orange
    "#98df8a",  # light green
    "#ff9896",  # light red
    "#c5b0d5",  # light purple
]

_SOLVER_INDEX_MAP = {
    'ilp': 0,
    'ilp-preprocessing': 1,
    'state_space': 2,
    'StateSpace': 2,
    'SimulatedAnnealing': 2,
    'ilp-slicing-s=1': 3,
    'ilp-slicing-s=2': 4,
    'ilp-slicing-s=3': 5,
    'ilp-slicing-s=4': 6,
    'ilp-slicing-s=5': 7,
    'ilp-slicing-s=7': 8,
    'ilp-slicing-s=8': 9,
    'ilp-slicing-s=10': 10,
    'ilp-slicing-s=12': 11,
    'ilp-slicing-s=15': 12,
    'ilp-slicing-s=16': 13,
    'ilp-slicing-s=20': 14,
}


def _solver_style(solver: str) -> dict:
    """Consistent marker/linestyle pair for a given solver name."""
    # Get fixed index for known solvers, or use hash for unknown ones
    if solver in _SOLVER_INDEX_MAP:
        idx = _SOLVER_INDEX_MAP[solver]
    else:
        print("random solver:", solver)
        # For unknown solvers, use hash of name to get consistent index
        idx = hash(solver) % 1000  # Limit range to avoid huge numbers

    return dict(
        marker=_SOLVER_MARKERS[idx % len(_SOLVER_MARKERS)],
        linestyle=_SOLVER_STYLES[idx % len(_SOLVER_STYLES)],
        color=_SOLVER_COLORS[idx % len(_SOLVER_COLORS)],
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
    constraints_filter: Optional[tuple[str, ...]] = None,
    exclude_solvers: Optional[list[str]] = None,
    exclude_empty: bool = False,  # New parameter to filter out rows where empty=True
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

    # change 'state_space' and 'StateSpace' solver names to 'SimulatedAnnealing'
    work.loc[work["solver"].isin(["state_space", "StateSpace"]), "solver"] = "SimulatedAnnealing"

    # Exclude rows where empty is True if requested
    if exclude_empty and "empty" in df.columns:
        work = work[~work["empty"]]

    if constraints_filter is not None:
        work = work[work["constraints_tuple"] == constraints_filter]

    # Exclude specified solvers
    if exclude_solvers:
        work = work[~work["solver"].isin(exclude_solvers)]

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
    for solver in solvers:
        subset = tidy[tidy["solver"] == solver]
        ax.plot(
            subset[varying_col],
            subset["value"],
            label=solver,
            **_solver_style(solver),  # Pass solver name instead of index
        )

    ax.legend(title="Solver", fontsize=9)

    if savepath:
        ax.figure.savefig(savepath, dpi=300)
    if show:
        plt.show()
    return [ax]

def plot_time_vs_satisfaction(
    df: pd.DataFrame,
    *,
    fixed_col: Literal["N", "M"],
    fixed_val: int,
    varying_col: Literal["N", "M"],
    time_stat: Stat = "mean",
    satisfaction_stat: Stat = "mean",
    constraints_filter: Optional[tuple[str, ...]] = None,
    exclude_solvers: Optional[list[str]] = None,
    exclude_empty: bool = False,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    savepath: Optional[str | Path] = None,
    avoid_clutter: bool = True,
) -> list[plt.Axes]:
    """Plot constraint satisfaction vs time for varying parameter values.

    Returns the list of Axes objects created (one or many).
    """
    if varying_col == fixed_col:
        raise ValueError("fixed_col and varying_col must differ.")

    work = df[df[fixed_col] == fixed_val].copy()

    # change 'state_space' and 'StateSpace' solver names to 'SimulatedAnnealing'
    work.loc[work["solver"].isin(["state_space", "StateSpace"]), "solver"] = "SimulatedAnnealing"

    # Filter data
    if exclude_empty and "empty" in df.columns:
        work = work[~work["empty"]]

    if constraints_filter is not None:
        work = work[work["constraints_tuple"] == constraints_filter]

    if exclude_solvers:
        work = work[~work["solver"].isin(exclude_solvers)]

    # Aggregate time and satisfaction data
    group_cols = [varying_col, "solver"]
    time_df = _aggregate(work, group_cols, "time_ms", time_stat)
    satisfaction_df = _aggregate(work, group_cols, "constraint_satisfaction", satisfaction_stat)

    # Merge the data
    tidy = pd.merge(time_df, satisfaction_df, on=group_cols, suffixes=('_time', '_satisfaction'))

    # Labels
    time_lbl = f"{_METRIC_LABEL['time_ms']} ({_STAT_LABEL[time_stat]})"
    satisfaction_lbl = f"{_METRIC_LABEL['constraint_satisfaction']} ({_STAT_LABEL[satisfaction_stat]})"
    title = f"Constraint satisfaction vs. Run time for varying {varying_col}  |  {fixed_col}={fixed_val}"

    if constraints_filter is not None:
        title += "\n" + ", ".join(constraints_filter)

    # Prepare axis
    if ax is None:
        ax = _base_ax(time_lbl, satisfaction_lbl, title)

    # Draw lines
    solvers = sorted(tidy["solver"].unique())

    param_values = sorted(tidy[varying_col].unique())

    for solver in solvers:
        subset = tidy[tidy["solver"] == solver]
        # Sort by varying parameter to ensure points are connected in order
        subset = subset.sort_values(by=varying_col)

        # Plot line and markers
        x_data = subset["value_time"]
        y_data = subset["value_satisfaction"]
        style = _solver_style(solver)

        # Plot line without markers first
        ax.plot(x_data, y_data, label=solver, **{k: v for k, v in style.items() if k != 'marker'})

        # Then add markers with parameter values in hover text
        for _, row in subset.iterrows():
            ax.scatter(
                row["value_time"],
                row["value_satisfaction"],
                marker=style['marker'],
                color=style['color'],
                s=style['markersize'] ** 2,
                zorder=10
            )

    for solver in solvers:
        subset = tidy[tidy["solver"] == solver].sort_values(by=varying_col)
        style = _solver_style(solver)

        # Add only a few labeled points per solver to avoid clutter
        param_subset = param_values
        if len(param_subset) > 5 and avoid_clutter:  # If more than 5 points, select a subset
            indices = np.linspace(0, len(param_subset) - 1, 5).astype(int)
            param_subset = [param_subset[i] for i in indices]
        else:
            param_subset = param_values

        for param in param_subset:
            row = subset[subset[varying_col] == param]
            if not row.empty:
                x, y = row["value_time"].iloc[0], row["value_satisfaction"].iloc[0]
                # Add a small annotation near the point
                ax.annotate(
                    f"{param}",
                    (x, y),
                    xytext=(5, 5),
                    textcoords='offset points',
                    weight='bold',
                    fontsize=8,
                    color=style['color'],
                    bbox=dict(
                        boxstyle="round,pad=0.1",  # minimal padding
                        facecolor='white',
                        alpha=0.8,
                        edgecolor='none'
                    )
                )

    ax.legend(title="Solver", fontsize=9)

    if savepath:
        ax.figure.savefig(savepath, dpi=300)
    if show:
        plt.show()
    return [ax]

if __name__ == "__main__":
    df = load_csv("results_id2_N10_solvers.csv")

    # Quick exploration
    print(df.head())
    print("Unique solvers:", df["solver"].unique())
    print("N range:", sorted(df["N"].unique()))
    print("M range:", sorted(df["M"].unique()))
    print("Total distinct constraint-lists:", df["constraints_tuple"].nunique())

    # replace values M=100 with M=60
    # this is done because for M = 100 tbe average number of candidates was 60
    # df.loc[df["M"] == 100, "M"] = 60

    # plot_time_vs_satisfaction(df,
    #                           fixed_col="N", fixed_val=10, varying_col="M", exclude_solvers=["ilp-slicing-s=2", "ilp-slicing-s=3", "ilp-slicing-s=5"],)
    # plot_time_vs_satisfaction(df,
    #                           fixed_col="N", fixed_val=10, varying_col="M", exclude_solvers=["ilp-slicing-s=7", ], satisfaction_stat="mean",  time_stat="p90")

    # mean constraint satisfaction, fixed N=10, vary M
    # plot_fixed_param(df,
    #                      fixed_col="N", fixed_val=10, varying_col="M",
    #                      metric="constraint_satisfaction", stat="mean",
    #                      constraints_mode="all", exclude_solvers=["ilp-slicing-s=3", "ilp-slicing-s=5", "ilp-slicing-s=8"])
    #
    # plot_fixed_param(df,
    #                  fixed_col="N", fixed_val=10, varying_col="M",
    #                  metric="time_ms", stat="mean",
    #                  constraints_mode="all", exclude_solvers=["ilp-slicing-s=2", "ilp-slicing-s=3"])
    #
    # # 90-th percentile runtime, fixed N=0, vary N
    # plot_fixed_param(df,
    #                      fixed_col="N", fixed_val=10, varying_col="M",
    #                      metric="constraint_satisfaction", stat="p90",
    #                      constraints_mode="all")
    # plot_fixed_param(df,
    #                  fixed_col="N", fixed_val=10, varying_col="M",
    #                  metric="time_ms", stat="p90",
    #                  constraints_mode="all", exclude_solvers=["ilp-slicing-s=3", "ilp-slicing-s=4"])

    # --------- fixed M, vary N ---------
    df = load_csv("results_id2_M60_solvers.csv")

    # Quick exploration
    print(df.head())
    print("Unique solvers:", df["solver"].unique())
    print("N range:", sorted(df["N"].unique()))
    print("M range:", sorted(df["M"].unique()))
    print("Total distinct constraint-lists:", df["constraints_tuple"].nunique())

    # plot_time_vs_satisfaction(df,
    #                           fixed_col="M", fixed_val=60, varying_col="N", exclude_solvers=["ilp-slicing-s=2", "ilp-slicing-s=3", "ilp-slicing-s=5"],)
    # plot_time_vs_satisfaction(df,
    #                           fixed_col="M", fixed_val=60, varying_col="N", exclude_solvers=["ilp-slicing-s=2", "ilp-slicing-s=3", "ilp-slicing-s=5"],
    #                           satisfaction_stat="mean", time_stat="p90")

    df = load_csv("results_id1_M60_solvers.csv")

    # Quick exploration
    print(df.head())
    print("Unique solvers:", df["solver"].unique())
    print("N range:", sorted(df["N"].unique()))
    print("M range:", sorted(df["M"].unique()))
    print("Total distinct constraint-lists:", df["constraints_tuple"].nunique())

    plot_time_vs_satisfaction(df,
                              fixed_col="M", fixed_val=60, varying_col="N",
                              exclude_solvers=["ilp-slicing-s=5", "ilp-slicing-s=3", "ilp-slicing-s=15", "ilp-slicing-s=16", "ilp-slicing-s=20"],)
    plot_time_vs_satisfaction(df,
                              fixed_col="M", fixed_val=60, varying_col="N",
                              satisfaction_stat="mean", time_stat="p90")

    df = load_csv("results_movielens_N10_solvers.csv")

    # Quick exploration
    print(df.head())
    print("Unique solvers:", df["solver"].unique())
    print("N range:", sorted(df["N"].unique()))
    print("M range:", sorted(df["M"].unique()))
    print("Total distinct constraint-lists:", df["constraints_tuple"].nunique())

    plot_time_vs_satisfaction(df,
                              fixed_col="N", fixed_val=10, varying_col="M",
                              exclude_solvers=["ilp-slicing-s=1", "ilp-slicing-s=3", "ilp-slicing-s=5", "ilp-slicing-s=7"],
                              avoid_clutter=False
                              )


