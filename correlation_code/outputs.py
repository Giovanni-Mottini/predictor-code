"""Correlation table (LaTeX, ieeeconf style) and the plain-text summary of one visit table.

Bold: uncorrected p < 0.05. The note states how many coefficients survive Holm correction.
"""
from __future__ import annotations

import math

import pandas as pd

SCALE_NAMES = {"FMA": "Fugl-Meyer (FMA-UE)", "BBT": "Box and Block Test (BBT)"}
SCALE_SHORT = {"FMA": "FMA-UE", "BBT": "BBT"}


def _num(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    v = round(float(v), 2) + 0.0            # round first so -0.001 prints as 0.00
    return f"\\textminus{{}}{abs(v):.2f}" if v < 0 else f"{v:.2f}"


def _p(p) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "--"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.4f}" if p < 0.05 <= round(p, 3) else f"{p:.3f}"   # a p just under 0.05 never prints as 0.050


def _cell(r, p) -> str:
    text = _num(r)
    if p is not None and not (isinstance(p, float) and math.isnan(p)) and p < 0.05 and text != "--":
        text = f"\\textbf{{{text}}}"
    return f"{text} ({_p(p)})"


def latex_escape(text: str) -> str:
    return text.replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def correlation_table(result: pd.DataFrame, order: list[str], labels: dict[str, str], caption: str, label: str,
                      permutations: int) -> str:
    scales = list(dict.fromkeys(result["scale"]))
    table = result.set_index(["scale", "feature"])
    lines = []
    for feature in order:
        cells = [_cell(table.loc[(s, feature), r], table.loc[(s, feature), p])
                 for s in scales for r, p in (("between", "between_p"), ("rmcorr", "rmcorr_p_shuffle"))]
        lines.append(f"{latex_escape(labels.get(feature, feature))} & " + " & ".join(cells) + r" \\")
    body = "\n\\hline\n".join(lines)
    n = len(scales)
    columns = "|l||" + "||".join(["c|c"] * n) + "|"
    heads = " & ".join(f"\\multicolumn{{2}}{{c{'||' if i < n - 1 else '|'}}}{{\\textbf{{{SCALE_NAMES.get(s, s)}}}}}"
                       for i, s in enumerate(scales))
    sub = " & ".join([r"Between, $\rho$ ($p$) & Within, $r_{\mathrm{rm}}$ ($p$)"] * n)
    patients = int(result["between_patients"].max())
    counts = "; ".join(f"{SCALE_SHORT.get(s, s)}: {int(result.loc[result.scale == s, 'visits'].max())} visits, score changes in "
                       f"{int(result.loc[result.scale == s, 'patients_scale_moves'].max())} patients" for s in scales)
    tests = len(result)
    survivors = int((result["between_p_holm"] < 0.05).sum() + (result["rmcorr_p_holm"] < 0.05).sum())
    smallest = min(result["between_p_holm"].min(), result["rmcorr_p_holm"].min())
    if survivors == 0:
        holm_note = f"no coefficient survives Holm correction over the {tests} tests of each measure (smallest adjusted $p={smallest:.2f}$)"
    else:
        holm_note = f"{survivors} coefficient{'s' if survivors > 1 else ''} survive{'' if survivors > 1 else 's'} Holm correction over the {tests} tests of each measure"
    return rf"""\begin{{table*}}[t]
\caption{{{caption}}}
\label{{{label}}}
\begin{{center}}
\begin{{tabular}}{{{columns}}}
\hline
 & {heads} \\
\hline
\textbf{{Feature}} & {sub} \\
\hline\hline
{body}
\hline
\end{{tabular}}
\end{{center}}
{{\footnotesize Between: Spearman correlation of patient means ({patients} patients). Within: repeated-measures correlation, $p$ from {permutations} within-patient permutations. {counts}. Bold: uncorrected $p<0.05$; {holm_note}.}}
\end{{table*}}
"""


def summary_block(name: str, title: str, visits: pd.DataFrame, result: pd.DataFrame, order: list[str],
                  labels: dict[str, str], notes: list[str]) -> list[str]:
    lines = [title.upper(), "",
             f"visits with a scale and a value: {len(visits)}, patients: {visits['patient'].nunique()}", *notes, ""]
    shown = result.assign(label=result["feature"].map(lambda f: labels.get(f, f)))
    shown["order"] = shown["feature"].map({f: i for i, f in enumerate(order)})
    shown = shown.sort_values(["scale", "order"]).set_index(["scale", "label"])
    lines.append(shown[["visits", "patients", "patients_scale_moves", "between", "between_p", "between_p_holm",
                        "rmcorr", "rmcorr_p_shuffle", "rmcorr_p_holm", "rmcorr_loo_min", "rmcorr_loo_max"]]
                 .round(3).to_string())
    return lines + ["", ""]
