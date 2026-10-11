"""Общие функции Этапа 1, скопированы без изменений из stage1/stage1_discovery.py:
рисунок матрицы, V Крамера, допустимые моделью переходы, оценка fitness/precision."""
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pm4py
from scipy.stats import chi2_contingency

START = "START"
SEED = 42


def plot_matrix(probs, counts, path, med=None):
    k = probs.shape[1]
    fig, ax = plt.subplots(figsize=(1.25 * k + 2, 0.8 * probs.shape[0] + 1.2))
    im = ax.imshow(probs.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(probs.shape[1]), probs.columns, rotation=30, ha="right")
    ax.set_yticks(range(probs.shape[0]), probs.index)
    for i in range(probs.shape[0]):
        for j in range(probs.shape[1]):
            p, n = probs.iat[i, j], counts.iat[i, j]
            m = med.get((probs.index[i], probs.columns[j])) if med is not None else None
            sub = f"n={n}" + (f", {m:.0f} д" if m is not None and not np.isnan(m) else "")
            ax.text(j, i, f"{p:.2f}\n{sub}", ha="center", va="center", fontsize=7,
                    color="white" if p > 0.55 else "#222")
    ax.set_xlabel("Следующее состояние")
    ax.set_ylabel("Текущее состояние")
    ax.set_title("Вероятность перехода, число и медиана времени до следующего заключения"
                 if med is not None else "Вероятности переходов (без цензурированных)", fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def cramers_v(ct):
    chi2, p, _, _ = chi2_contingency(ct)
    n = ct.values.sum()
    r, k = ct.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1))), p


def allowed_df_pairs(net, im, fm, max_tokens=3, max_states=3_000_000):
    """Отношение directly-follows, допускаемое моделью.
    1) Строим граф достижимости маркировок (для неограниченных сетей Heuristics
       Miner число фишек в месте ограничено max_tokens — нижняя оценка).
    2) Для каждой маркировки m находим E(m) — видимые переходы, которые можно
       выполнить из m через цепочку тихих переходов (неподвижная точка).
    3) Пара (a, b) допустима, если после срабатывания a в некоторой маркировке
       b входит в E(новой маркировки); (START, b) — если b в E(im)."""
    trans = list(net.transitions)
    pre = [[(a.source, a.weight) for a in t.in_arcs] for t in trans]
    post = [[(a.target, a.weight) for a in t.out_arcs] for t in trans]
    key0 = frozenset(im.items())
    idx = {key0: 0}
    keys = [key0]
    vis_edges, sil_edges, enabled_vis = [], [], []
    i = 0
    while i < len(keys) and len(keys) < max_states:
        m = dict(keys[i])
        ev, se = set(), []
        for ti, t in enumerate(trans):
            if not all(m.get(pl, 0) >= w for pl, w in pre[ti]):
                continue
            m2 = dict(m)
            for pl, w in pre[ti]:
                m2[pl] -= w
                if m2[pl] == 0:
                    del m2[pl]
            for pl, w in post[ti]:
                m2[pl] = m2.get(pl, 0) + w
            if max(m2.values(), default=0) > max_tokens:
                continue
            k = frozenset(m2.items())
            j = idx.get(k)
            if j is None:
                j = idx[k] = len(keys)
                keys.append(k)
            if t.label is None:
                se.append(j)
            else:
                ev.add(t.label)
                vis_edges.append((t.label, j))
        enabled_vis.append(ev)
        sil_edges.append(se)
        i += 1
    n = len(enabled_vis)
    E = [set(x) for x in enabled_vis] + [set() for _ in range(len(keys) - n)]
    changed = True
    while changed:
        changed = False
        for u in range(n):
            for v in sil_edges[u]:
                if not E[v] <= E[u]:
                    E[u] |= E[v]
                    changed = True
    pairs = {(START, b) for b in E[0]}
    for a, j in vis_edges:
        pairs |= {(a, b) for b in E[j]}
    return pairs


def log_df_pairs(log_df):
    pairs = Counter()
    for _, g in log_df.groupby("case:concept:name", sort=False):
        acts = [START] + g["concept:name"].tolist()
        pairs.update(zip(acts[:-1], acts[1:]))
    return pairs


def evaluate(models, train_log, test_log, skip_align=(), max_states=3_000_000):
    """skip_align — модели, для которых выравнивания не считаются (fitness_align = NaN)."""
    test_pairs = log_df_pairs(test_log)
    train_pairs = log_df_pairs(train_log)
    rows, edge_rows = [], []
    for name, (net, im, fm) in models.items():
        allowed = allowed_df_pairs(net, im, fm, max_states=max_states)
        # переходы в «конец трассы» не включаем: в журнале конец трассы
        # бывает и у цензурированных объектов
        for split, lg, lp in [("train", train_log, train_pairs), ("test", test_log, test_pairs)]:
            tbr = pm4py.fitness_token_based_replay(lg, net, im, fm)
            ali = ({"log_fitness": float("nan"), "percentage_of_fitting_traces": float("nan")}
                   if name in skip_align else pm4py.fitness_alignments(lg, net, im, fm))
            prec = pm4py.precision_token_based_replay(lg, net, im, fm)
            lost = {p: c for p, c in lp.items() if p not in allowed}
            extra = {p for p in allowed if p not in lp}
            rows.append(dict(
                model=name, split=split,
                fitness_tbr=round(tbr["log_fitness"], 4),
                pct_fit_traces_tbr=round(tbr["percentage_of_fitting_traces"], 2),
                fitness_align=round(ali["log_fitness"], 4),
                pct_fit_traces_align=round(ali["percentage_of_fitting_traces"], 2),
                precision_etc=round(prec, 4),
                f1=round(2 * tbr["log_fitness"] * prec / (tbr["log_fitness"] + prec), 4),
                log_edges=len(lp), model_edges=len(allowed),
                lost_edges=len(lost), lost_edge_occurrences=sum(lost.values()),
                lost_occ_share=round(sum(lost.values()) / sum(lp.values()), 4),
                extra_edges=len(extra),
                places=len(net.places), transitions=len(net.transitions),
                silent=sum(t.label is None for t in net.transitions),
            ))
            if split == "test":
                for p in sorted(set(lp) | allowed):
                    edge_rows.append(dict(model=name, src=p[0], dst=p[1], in_test_log=lp.get(p, 0),
                                          allowed_by_model=p in allowed,
                                          kind=("ok" if (p in allowed and p in lp) else
                                                "lost" if p in lp else "extra")))
    return pd.DataFrame(rows), pd.DataFrame(edge_rows)
