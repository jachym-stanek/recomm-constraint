import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from collections import defaultdict
from typing import Iterable, Dict, Mapping, Sequence, Tuple, List, Any, Callable, Optional

Params  = Dict[str, Any]
Metrics = Mapping[str, float]
Record  = Tuple[Params, Metrics]
SolverRecord  = Tuple[Params, Dict[str, Metrics]]

def plot_constraint_impact():
    file_name = "diversity_experiment_results.txt"
    results = []
    with open(file_name, 'r') as f:
        for line in f:
            results = eval(line)
            break

    print(f"results: {results}")

    # remove the value for max_item=1 as its misleading
    results = [result for result in results if result[0] not in [1, 10]]

    nums_max_items = [result[0] for result in results]
    constrained_recalls = [result[1]["average_recall_constrained"] for result in results]
    constrained_catalog_coverages = [result[1]["catalog_coverage_constrained"] for result in results]

    print(f"nums_max_items: {nums_max_items}")
    print(f"constrained_recalls: {constrained_recalls}")
    print(f"constrained_catalog_coverages: {constrained_catalog_coverages}")

    # Plot results
    # x-axis recall@N, y-axis catalog coverage, plot entry for each max_items
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    line, = ax.plot(constrained_recalls, constrained_catalog_coverages, marker='o', linewidth=2,
                    label=f'Max items per segment: {nums_max_items}')
    # Annotate the points with max_items values
    for i, max_items in enumerate(nums_max_items):
        if max_items == 8:
            ax.annotate(f'{max_items}', (constrained_recalls[i], constrained_catalog_coverages[i]), xytext=(-13, -5),
                        textcoords='offset points', fontsize=13)
        elif max_items == 9:
            ax.annotate(f'{max_items}', (constrained_recalls[i], constrained_catalog_coverages[i]), xytext=(+5, -8),
                        textcoords='offset points', fontsize=13)
        else:
            ax.annotate(f'{max_items}', (constrained_recalls[i], constrained_catalog_coverages[i]), xytext=(3, 3),
                        textcoords='offset points', fontsize=13)

    ax.grid(True)
    ax.set_xlabel(f'Average Recall@N', fontsize=18)
    ax.set_ylabel('Catalog Coverage', fontsize=18)
    # ax.set_title('Impact of Diversity Constraints on Catalog Coverage and Recall')

    plt.show()

def plot_metric_grid(
    data: Iterable[Record],
    *,
    # what to plot
    x_metric: str = "average_recall",
    y_metric: str = "catalog_coverage",
    # two hyper-parameters to explore
    param_a: str,
    param_b: str,
    # pretty labels for them  ↓↓↓
    param_a_label: str | None = None,
    param_b_label: str | None = None,
    # optional omissions
    omit: Optional[Dict[str, Sequence[Any]]] = None,
    # point annotation logic
    labeler: Optional[Callable[[Params, str], str]] = None,
    # constrained variant
    show_constrained: bool = False,
    constrained_suffix: str = "_constrained",
    figsize: Tuple[int, int] = (18, 7),
    plot_panels: str = "both",
):
    param_a_label = param_a_label or param_a.replace("_", " ").title()
    param_b_label = param_b_label or param_b.replace("_", " ").title()

    omit = {k: set(v) for k, v in (omit or {}).items()}
    flat: List[Tuple[Any, Any, Metrics]] = []

    # filter / flatten
    for params, metrics in data:
        a, b = params.get(param_a), params.get(param_b)
        if a is None or b is None:
            continue
        if a in omit.get(param_a, ()) or b in omit.get(param_b, ()):
            continue
        if metrics.get(x_metric, 0.0) <= 0.0:
            continue
        flat.append((a, b, metrics))
    if not flat:
        raise ValueError("Nothing to plot after filtering.")

    by_a, by_b = defaultdict(list), defaultdict(list)
    for a, b, m in flat:
        by_a[a].append((b, m))
        by_b[b].append((a, m))

    def default_label(p: Params, varying: str) -> str:
        return str(p[varying])

    labeler = labeler or default_label

    # plotting
    if plot_panels not in ("left", "right", "both"):
        raise ValueError("plot_panels must be 'left', 'right' or 'both'")

    if plot_panels == "left" or plot_panels == "right":
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig, ax = plt.subplots(1, 2, figsize=figsize)

    def _panel(
        fixed: str, varying: str, groups, axis, title_tmpl: str, fixed_label: str, varying_label: str
    ):
        for fixed_val, pts in sorted(groups.items()):
            xs, ys, xs_c, ys_c, ann_params = [], [], [], [], []
            for varying_val, metrics in pts:
                xs.append(metrics[x_metric])
                ys.append(metrics[y_metric])
                if show_constrained:
                    xs_c.append(metrics.get(x_metric + constrained_suffix))
                    ys_c.append(metrics.get(y_metric + constrained_suffix))
                ann_params.append({fixed: fixed_val, varying: varying_val})

            line, = axis.plot(xs, ys, marker="o",
                              label=f"{fixed_label} = {fixed_val}")
            if show_constrained:
                axis.plot(xs_c, ys_c, linestyle="dotted",
                          color=line.get_color())

            texts = []
            for x, y, p in zip(xs, ys, ann_params):
                txt = axis.text(
                    x, y,  # initial anchor = datapoint
                    labeler(p, varying),
                    fontsize="small",
                    ha="left", va="bottom",
                )
                texts.append(txt)

            # adjust the text labels to avoid overlap
            adjust_text(
                texts,
                ax=axis,
                expand_points=(1.2, 1.4),  # how far to push from points
                expand_text=(1.2, 1.4),  # how far to push from other labels
                arrowprops=dict(arrowstyle="->", lw=.5, alpha=.6),
                only_move={"points": "y", "text": "xy"},  # move mostly up/down
            )

        # assume plotting recall@N and coverage
        axis.set_xlabel("Average recall@N")
        axis.set_ylabel("Catalog coverage")
        axis.set_title(title_tmpl.format(fixed_label, varying_label))
        axis.legend()

    if plot_panels == "both":
        _panel(param_a, param_b, by_a, ax[0],
               "Fixed {} — varying {}", param_a_label, param_b_label)
        _panel(param_b, param_a, by_b, ax[1],
               "Fixed {} — varying {}", param_b_label, param_a_label)
    elif plot_panels == "left":
        _panel(param_a, param_b, by_a, ax,
               "Fixed {} — varying {}", param_a_label, param_b_label)
    elif plot_panels == "right":
        _panel(param_b, param_a, by_b, ax,
               "Fixed {} — varying {}", param_b_label, param_a_label)

    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # results_file = "results_movielens_nearest_neighbors_vs_factors.txt"
    # results_file = "results_id1_nn_vs_b.txt"
    # results_file = "results_id1_reg_vs_nn.txt"
    # results_file = "results_movielens_factors_vs_nn_N10.txt"
    # results_file = "results_id1_regularization_vs_nn.txt"
    results_file = "results_id2_factors_vs_nn.txt"
    records: List[Record] = []
    with open(results_file) as f:
        for params, metrics in map(eval, f):
            records.append((params, metrics))

    skipped_values = {"bm25_B": [0.1, 0.6, 1.0, 1.5],
                      "num_iterations": [10],
                      "nearest_neighbors": [1,2, 15, 30],
                      "num_factors": [1, 2, 4, 8],
                      "regularization": [1000],
                      }

    fig, _ = plot_metric_grid(
        records,
        param_a="num_factors",
        param_b="nearest_neighbors",
        param_a_label="Number of factors",
        param_b_label="Nearest neighbors",
        x_metric="average_recall",
        y_metric="catalog_coverage",
        omit=skipped_values,
        labeler=lambda p, varying: str(p[varying]),  # annotate with varying value
        # figsize=(9, 7),
        plot_panels="both"
    )
    plt.show()
