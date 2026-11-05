"""Графики к разделу 6 ноутбуков petri_states_results_*.ipynb (флаги SHORT/LONG_CUSTODY_DAYS).

Только читают готовые CSV из out_states/flags/ и рисуют; ничего не пересчитывают.
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODELS = ["DFG_p0.05", "HM_dep0.95", "SplitMiner"]
COLORS = {"DFG_p0.05": "#2a78d6", "HM_dep0.95": "#eb6834", "SplitMiner": "#1baf7a",
          "среднее": "#555555"}
INK, MUTED, GRID = "#222222", "#777777", "#e6e6e6"
CUR, REC = (15, 365), (15, 180)          # текущие и рекомендованные флаги

plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#999999", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
})


def _bin_label(s):
    a, b = [float(x) for x in re.findall(r"[\d.]+", s)]
    return f"{a:g}+" if b >= 100000 else f"{a:g}–{b:g}"


def _long_label(v):
    return "выкл" if v >= 100000 else str(int(v))


# ---------------------------------------------------------------- перебор флагов
def plot_sweep(sw):
    """sw — таблица перебора одного эксперимента (как в ячейке выше, без колонки exp)."""
    sw = sw.reset_index() if "seed" not in sw.columns else sw
    if len(sw) < 6:
        return _sweep_bars(sw)
    return _sweep_lines(sw)


def _sweep_bars(sw):
    sw = sw.sort_values(["short", "long"])
    cfg = [f"{s} / {l}" for s, l in zip(sw["short"], sw["long"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    x = np.arange(len(MODELS))
    w = 0.8 / len(cfg)
    shades = ["#9ec5f4", "#5598e7", "#1c5cab", "#0d366b"]
    for ax, metric, title in zip(axes, ["F1", "Pnorm"], ["F1 на test", "Precision, нормированная на flower"]):
        for j, (c, (_, r)) in enumerate(zip(cfg, sw.iterrows())):
            vals = [r[f"{metric} {m}"] for m in MODELS]
            bars = ax.bar(x + (j - (len(cfg) - 1) / 2) * w, vals, w * 0.92, color=shades[j],
                          label=f"SHORT / LONG = {c}" + ("  (текущие)" if (r["short"], r["long"]) == CUR else ""))
            ax.bar_label(bars, fmt="%.3f", fontsize=6.5, color=INK, padding=1, rotation=90)
        ax.set_xticks(x, MODELS)
        ax.set_title(title, loc="left", color=INK)
        ax.set_ylim(0, 1.08)
        ax.grid(axis="x", visible=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(cfg), fontsize=9)
    fig.suptitle("Перебор флагов: значения по моделям", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def _sweep_lines(sw):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharey="row")
    main = sw[sw["seed"] == 42]
    other = sw[sw["seed"] != 42]
    panels = [("long", "short", CUR[0], "LONG_CUSTODY_DAYS (SHORT = 15)"),
              ("short", "long", CUR[1], "SHORT_CUSTODY_DAYS (LONG = 365)")]
    for col, (var, fix, fixv, xlabel) in enumerate(panels):
        d = main[main[fix] == fixv].sort_values(var)
        o = other[other[fix] == fixv]
        xs = list(d[var])
        pos = {v: i for i, v in enumerate(xs)}
        labels = [_long_label(v) if var == "long" else str(int(v)) for v in xs]
        for row, metric in enumerate(["F1", "Pnorm"]):
            ax = axes[row, col]
            series = MODELS + (["среднее"] if metric == "F1" else [])
            for m in series:
                c = f"{metric} {m}"
                ax.plot(range(len(xs)), d[c], color=COLORS[m], lw=2 if m != "среднее" else 1.6,
                        ls="-" if m != "среднее" else "--", marker="o", ms=5, label=m)
                if not o.empty:
                    oo = o[o[var].isin(pos)]
                    ax.scatter([pos[v] for v in oo[var]], oo[c], s=34, facecolors="none",
                               edgecolors=COLORS[m], linewidths=1.3, zorder=3)
            cur_v = CUR[0] if var == "short" else CUR[1]
            if cur_v in pos:
                ax.axvline(pos[cur_v], color=INK, lw=1, ls=":")
                ax.annotate("текущее", (pos[cur_v], 1), xycoords=("data", "axes fraction"),
                            xytext=(3, -12), textcoords="offset points", fontsize=8, color=INK)
            if var == "long" and REC[1] in pos:
                ax.axvline(pos[REC[1]], color="#1c5cab", lw=1, ls=":")
                ax.annotate("рекоменд.", (pos[REC[1]], 1), xycoords=("data", "axes fraction"),
                            xytext=(3, -24), textcoords="offset points", fontsize=8, color="#1c5cab")
            ax.set_xticks(range(len(xs)), labels, fontsize=8)
            if row == 1:
                ax.set_xlabel(xlabel + ", дни")
        axes[0, 0].set_ylabel("F1 на test")
        axes[1, 0].set_ylabel("Precision norm на test")
    axes[0, 0].set_title("Влияние LONG", loc="left", color=INK)
    axes[0, 1].set_title("Влияние SHORT", loc="left", color=INK)
    h, l = axes[0, 0].get_legend_handles_labels()
    if not other.empty:
        h.append(plt.Line2D([], [], ls="", marker="o", mfc="none", mec=MUTED))
        l.append(f"seed {', '.join(map(str, sorted(other.seed.unique())))} (кружки)")
    fig.legend(h, l, loc="upper center", ncol=len(l), bbox_to_anchor=(0.5, 1.0), fontsize=9)
    fig.suptitle("Перебор флагов (seed 42 — линии)", x=0.01, y=1.04, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# ---------------------------------------------------------------- медианы (q25–median–q75)
def plot_iqr(tables, label_col=None, title="", names=None):
    """Горизонтальные интервалы q25–q75 с точкой-медианой, логарифмическая шкала дней.
    tables — один DataFrame или список (сравнение вариантов флагов, по строкам совпадают)."""
    if isinstance(tables, pd.DataFrame):
        tables = [tables]
    names = names or [None] * len(tables)
    label_col = label_col or tables[0].columns[0]
    labels = list(tables[0][label_col])
    k = len(tables)
    shades = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, ax = plt.subplots(figsize=(11, 0.62 * len(labels) * (1 + 0.35 * (k - 1)) + 1.2))
    step = 0.32
    for j, (t, nm) in enumerate(zip(tables, names)):
        y = np.arange(len(labels)) + (j - (k - 1) / 2) * step
        q25 = t["q25"].clip(lower=0.1)
        ax.hlines(y, q25, t["q75"], color=shades[j], lw=6, alpha=0.45, capstyle="round",
                  label=f"{nm}: q25–q75" if nm else "q25–q75")
        ax.scatter(t["median"], y, s=46, color=shades[j], edgecolor="white", linewidth=1.5, zorder=3,
                   label=f"{nm}: медиана" if nm else "медиана")
        for yi, (_, r) in zip(y, t.iterrows()):
            txt = f"{r['median']:g} дн.  n={int(r['n'])}"
            if "доля prison" in t.columns:
                txt += f", prison {r['доля prison']:.0%}"
            ax.annotate(txt, (r["q75"], yi), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK)
    ax.set_yticks(np.arange(len(labels)), [_wrap(s) for s in labels], fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("дни (лог. шкала)")
    ax.grid(axis="y", visible=False)
    for v, txt in [(CUR[0], "SHORT 15"), (REC[1], "180"), (CUR[1], "365")]:
        ax.axvline(v, color=MUTED, lw=0.8, ls=":")
        ax.annotate(txt, (v, 1), xycoords=("data", "axes fraction"), xytext=(2, 2),
                    textcoords="offset points", fontsize=7, color=MUTED)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo, hi * 6)          # место под подписи справа
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.02 * k), ncol=2 * k, fontsize=8)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=16)
    fig.tight_layout()
    return fig


def _wrap(s, width=46):
    words, lines, cur = str(s).split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    return "\n".join(lines)


# ---------------------------------------------------------------- профили по длительности
def _bin_edges(df):
    return [float(re.findall(r"[\d.]+", s)[0]) for s in df["dur"]]


def _mark_bins(ax, df, marks):
    starts = _bin_edges(df)
    for v, txt, c in marks:
        if v in starts:
            x = starts.index(v) - 0.5
            ax.axvline(x, color=c, lw=1.1, ls=":")
            ax.annotate(txt, (x, 1), xycoords=("data", "axes fraction"), xytext=(3, -12),
                        textcoords="offset points", fontsize=8, color=c)


def plot_long_profile(lp):
    x = np.arange(len(lp))
    labels = [_bin_label(s) for s in lp["dur"]]
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(12, 6.2), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.2]})
    for c, nm, col in [("доля_prison", "доля с prison", "#2a78d6"),
                       ("доля_с_делом_в_день_входа", "доля с делом в день входа", "#eb6834"),
                       ("доля_одно_дело", "доля с одним делом в день входа", "#1baf7a")]:
        ax.plot(x, lp[c], marker="o", ms=5, lw=2, color=col, label=nm)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("LONG: где задержание по делу становится «сроком»", loc="left", color=INK, fontsize=12)
    ax.legend(loc="center right", fontsize=9)
    _mark_bins(ax, lp, [(180, "180 (рекоменд.)", "#1c5cab"), (365, "365 (текущее)", INK)])
    axn.bar(x, lp["эпизодов"], color="#9ec5f4", width=0.7)
    axn.set_yscale("log")
    axn.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    axn.set_ylabel("эпизодов\n(лог.)")
    axn.grid(axis="x", visible=False)
    _mark_bins(axn, lp, [(180, "", "#1c5cab"), (365, "", INK)])
    axn.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
    axn.set_xlabel("длительность эпизода заключения, дни")
    fig.tight_layout()
    return fig


def plot_short_profile(sp):
    x = np.arange(len(sp))
    labels = [_bin_label(s) for s in sp["dur"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(x, sp["доля_на_след_день"], marker="o", lw=2, color="#2a78d6", label="на следующий день после входа")
    a1.plot(x, sp["доля_то_же_дело"], marker="o", lw=2, color="#eb6834", label="то же дело")
    a1.set_ylim(0, 1.05)
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a1.set_title("Доля событий внутри срока", loc="left", color=INK)
    a1.legend(fontsize=8, loc="upper right")
    a2.bar(x, sp["медиана_дней_от_входа"], color="#5598e7", width=0.7)
    for xi, v in zip(x, sp["медиана_дней_от_входа"]):
        a2.annotate(f"{v:g}", (xi, v), xytext=(0, 2), textcoords="offset points", ha="center",
                    fontsize=8, color=INK)
    a2.set_yscale("log")
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    a2.set_title("Медиана дней от входа до события (лог.)", loc="left", color=INK)
    a2.grid(axis="x", visible=False)
    for ax in (a1, a2):
        _mark_bins(ax, sp, [(15, "SHORT 15", INK)])
        ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
        ax.set_xlabel("длительность срока, дни")
    fig.suptitle("SHORT: события внутри срока", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    return fig
