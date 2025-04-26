import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from collections import defaultdict
from typing import Iterable, Dict, Mapping, Sequence, Tuple, List, Any, Callable, Optional

Params  = Dict[str, Any]
Metrics = Mapping[str, float]
Record  = Tuple[Params, Metrics]



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

    _panel(param_a, param_b, by_a, ax[0],
           "Fixed {} — varying {}", param_a_label, param_b_label)
    _panel(param_b, param_a, by_b, ax[1],
           "Fixed {} — varying {}", param_b_label, param_a_label)

    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # results_file = "results_movielens_nearest_neighbors_vs_factors.txt"
    # results_file = "results_id1_nn_vs_b.txt"
    results_file = "results_id1_reg_vs_nn.txt"
    records: List[Record] = []
    with open(results_file) as f:
        for params, metrics in map(eval, f):
            records.append((params, metrics))

    skipped_values = {"bm25_B": [0.1, 0.6, 1.0, 1.5],
                      "num_iterations": [10],
                      "nearest_neighbors": [2, 10, 20],
                      "regularization": [1000],}

    fig, _ = plot_metric_grid(
        records,
        param_a="nearest_neighbors",
        param_b="regularization",
        param_a_label="Nearest neighbors",
        param_b_label="Regularization",
        x_metric="average_recall",
        y_metric="catalog_coverage",
        omit=skipped_values,
        labeler=lambda p, varying: str(p[varying])  # annotate with varying value
    )
    plt.show()
