"""Rank table (LaTeX, ieeeconf style) and the two importance figures of the paper.

Rows follow the first comparison's Gini importance, descending (the patients' order in the paper).
Table: bold = top three; dagger = shuffling the feature did not lower balanced accuracy (drop <= 0).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .importance import COMPARISONS

INK, MUTED, GRID = "#202124", "#62676d", "#e3e6e9"


def feature_order(summary: pd.DataFrame, keys: list[str]) -> list[str]:
    first = summary[summary["comparison"] == keys[0]]
    return list(first.sort_values("gini", ascending=False, kind="stable")["feature"])


def _cell(rank: int, drop: float | None) -> str:
    text = str(rank)
    if drop is not None and drop <= 0:
        text += "$^{\\dagger}$"
    return f"\\textbf{{{text}}}" if rank <= 3 else text


def rank_table(summary: pd.DataFrame, keys: list[str], labels: dict[str, str]) -> str:
    table = summary.set_index(["comparison", "feature"])
    lines = []
    for feature in feature_order(summary, keys):
        cells = []
        for key in keys:
            row = table.loc[(key, feature)]
            cells += [_cell(int(row["gini_rank"]), None), _cell(int(row["held_out_rank"]), float(row["held_out_drop"]))]
        lines.append(f"{labels[feature]} & " + " & ".join(cells) + r" \\")
    body = "\n\\hline\n".join(lines)
    n = len(keys)
    columns = "|l||" + "||".join(["c|c"] * n) + "|"
    head = [COMPARISONS[k].header for k in keys]
    def spans(line):
        parts = [f"\\multicolumn{{2}}{{c{'||' if i < n - 1 else '|'}}}{{\\textbf{{{h[line]}}}}}" for i, h in enumerate(head)]
        return " & " + " & ".join(parts) + r" \\"
    return rf"""\begin{{table}}[t]
\caption{{Feature importance ranks}}
\label{{tab:ranks}}
\begin{{center}}
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{{columns}}}
\hline
{spans(0)}
{spans(1)}
\hline
\textbf{{Feature}} & {" & ".join(["Gini & Held-out"] * n)} \\
\hline\hline
{body}
\hline
\end{{tabular}}
\end{{center}}
{{\footnotesize Rank 1 is the most important of the {len(labels)} features. Bold: top three. $^{{\dagger}}$Shuffling the feature did not lower the balanced accuracy.}}
\end{{table}}
"""


def figures(summary: pd.DataFrame, keys: list[str], labels: dict[str, str], folder: Path) -> list[Path]:
    """Two PNGs sized for one IEEE column: Gini importance and held-out drop."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.edgecolor": "#c2c7cc", "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": INK, "axes.linewidth": .8})
    order = feature_order(summary, keys)
    table = summary.set_index(["comparison", "feature"])
    written = []
    for column, xlabel, name in (("gini", "Gini importance", "importance_gini.png"),
                                 ("held_out_drop", "Drop in balanced accuracy", "importance_held_out.png")):
        fig, ax = plt.subplots(figsize=(3.5, 3.3))
        y = np.arange(len(order))[::-1]
        for i, key in enumerate(keys):
            comparison = COMPARISONS[key]
            values = np.array([table.loc[(key, f), column] for f in order], dtype=float)
            ax.scatter(values, y + (.5 - i) * .26, s=16, marker=comparison.marker, color=comparison.color, edgecolor="white",
                       linewidth=.5, zorder=3, label=comparison.legend)
        ax.axvline(0, color="#aeb4bb", linewidth=.8, zorder=1)
        for edge in np.arange(-.5, len(order)):
            ax.axhline(edge, color="#c2c7cc", linewidth=.5, zorder=1)
        ax.set_ylim(-.5, len(order) - .5)
        ax.set_yticks(y)
        ax.set_yticklabels([labels[f] for f in order])
        ax.set_xlabel(xlabel, color=INK)
        ax.grid(axis="x", color=GRID, linewidth=.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.legend(loc="lower left", bbox_to_anchor=(-.02, 1.0), ncol=2, frameon=False, fontsize=8, handletextpad=.2,
                  columnspacing=1.2, borderaxespad=.2)
        fig.savefig(folder / name, dpi=600, bbox_inches="tight")
        plt.close(fig)
        written.append(folder / name)
    return written
