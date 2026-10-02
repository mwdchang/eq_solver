"""Per-cell S/I/R/D time series, laid out like the grid."""

import matplotlib.pyplot as plt
import numpy as np

from eq_solver.model import COMPARTMENTS

COLORS = {"S": "#1f77b4", "I": "#d62728", "R": "#2ca02c", "D": "#555555"}
LABELS = {"S": "S: susceptible", "I": "I: infected", "R": "R: recovered", "D": "D: dead"}


def plot_cells(
    states: np.ndarray,
    title: str = "",
    observed: np.ndarray | None = None,
    initial: np.ndarray | None = None,
):
    """states: [days, 4, rows, cols], drawn as solid lines.

    Optional observed data is drawn as dots, and an optional run from the
    initial guesses as dashed lines, to show what calibration changed.
    """
    _, _, rows, cols = states.shape
    fig, axes = plt.subplots(
        rows, cols, figsize=(4 * cols, 3 * rows), sharex=True, sharey=True, squeeze=False
    )
    days = np.arange(states.shape[0])

    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            for k, name in enumerate(COMPARTMENTS):
                ax.plot(days, states[:, k, r, c], color=COLORS[name], label=LABELS[name])
                if initial is not None:
                    ax.plot(days, initial[:, k, r, c], "--", color=COLORS[name], linewidth=1, alpha=0.6)
                if observed is not None:
                    ax.plot(
                        np.arange(observed.shape[0]), observed[:, k, r, c],
                        ".", color=COLORS[name], markersize=3, alpha=0.5,
                    )
            ax.set_title(f"cell ({r}, {c})")
            ax.grid(alpha=0.3)
            if r == rows - 1:
                ax.set_xlabel("day")
            if c == 0:
                ax.set_ylabel("people")

    # Shared legends below the panels so they never hide a curve:
    # one row for compartments, one for line styles
    handles, labels = axes[0, 0].get_legend_handles_labels()
    styles = []
    if initial is not None:
        styles += [(plt.Line2D([], [], color="gray"), "fitted"),
                   (plt.Line2D([], [], linestyle="--", color="gray"), "initial guess")]
    if observed is not None:
        styles.append((plt.Line2D([], [], marker=".", linestyle="", color="gray"), "observed data"))

    if title:
        fig.suptitle(title)
    bottom = 0.14 if styles else 0.08
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, bottom - 0.07),
               ncol=len(labels), frameon=False)
    if styles:
        fig.legend(*zip(*styles), loc="lower center", bbox_to_anchor=(0.5, 0.0),
                   ncol=len(styles), frameon=False)
    return fig
