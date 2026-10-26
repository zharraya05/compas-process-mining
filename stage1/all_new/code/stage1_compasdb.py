"""
Этап 1, п. 1.1 плана: граф переходов на таблице compas_custody_charges.csv
(собрана build_from_compas_db.py прямо из compas.db).

Подход тот же, что в stage1/stage1_multi.py (данные Даши):
  * объект = человек (id = person_id из compas.db);
  * событие = заключение под стражу (in_custody);
  * состояние = МНОЖЕСТВО типов преступлений всех обвинений заключения
    (crime_categories.py). Несколько типов — развилка: из каждого типа
    эпизода k идут дуги в каждый тип эпизода k+1 с весом 1/(|S_k| |S_k+1|);
  * в журнале PM4Py по событию на тип, порядок внутри эпизода случайный (seed 42);
  * терминального состояния нет, последнее заключение цензурируется концом
    наблюдения (последняя реальная дата в данных);
  * модели IM / HM / DFG учатся на 80% людей, fitness и precision — на 20%.

Отличие от данных Даши, которое пришлось решить здесь: в compas.db переход
из jail в prison в тот же день записан двумя заключениями (приговор по тому же
делу, а не новый арест). По умолчанию такая запись prison склеивается с
предыдущим заключением (и заключения, начатые в один день, тоже). Вариант
без склейки: аргумент nomerge.

Основной вариант — oldcase: обвинения по делу (case_number), уже встречавшемуся
у человека в прошлом заключении, считаются старыми; заключение только из
старых обвинений получает состояние OldCase, в остальных старые обвинения
не учитываются.

Запуск: python3 stage1_compasdb.py [путь к compas_custody_charges.csv] [oldcase|merge|nomerge]
  -> results_oldcase/ (основной), results/ (все обвинения новые), results_nomerge/
"""
import multiprocessing as mp
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pm4py
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier, export_text

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stage1_common as base                       # noqa: E402
from crime_categories import categorize_fine as categorize, UNKNOWN   # noqa: E402

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "stage1_fixed_input", "compas_custody_charges.csv")
# merge (основной) | nomerge (без склейки jail->prison) | oldcase (обвинения дела, уже
# встречавшегося у человека в прошлом эпизоде, считаются старыми: эпизод только из
# старых обвинений получает состояние OldCase, в остальных старые обвинения не учитываются)
VARIANT = sys.argv[2] if len(sys.argv) > 2 else "merge"
assert VARIANT in ("merge", "nomerge", "oldcase"), VARIANT
MERGE = VARIANT != "nomerge"
# основной вариант — oldcase (решение Greer, 2026-10-09); merge (все обвинения
# считаются новыми) оставлен для сравнения, модели оцениваем в обоих
EVAL = VARIANT in ("merge", "oldcase")
OLDCASE = "OldCase"
JUSTICE = "Justice"
OUT = os.path.join(HERE, "results" if VARIANT == "merge" else f"results_{VARIANT}")
FIG = os.path.join(OUT, "figures")
TAB = os.path.join(OUT, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

SEED = base.SEED
START = base.START
CASE = "id"
PLACEHOLDER_OUT = pd.Timestamp("2020-01-01")       # «ещё в заключении»
DEGREE_RANK = {"F1": 1, "F2": 2, "F3": 3, "F5": 4, "F6": 4, "F7": 4, "M1": 5, "M2": 6, "MO3": 7}
DEG = {1: "F", 2: "F", 3: "F", 4: "F", 5: "M", 6: "M", 7: "M"}
DUR_BINS = [-np.inf, 1, 30, 180, np.inf]
DUR_LABELS = ["S1_<=1d", "S2_2-30d", "S3_31-180d", "S4_>180d"]


def degree_rank(d):
    return DEGREE_RANK.get(str(d).strip("()"), 8)


# ---------------------------------------------------------------- 1. журнал
def load():
    ch = pd.read_csv(DATA, parse_dates=["in_custody", "out_custody"], low_memory=False)
    ch["type"] = ch["c_charge_desc"].map(categorize)
    ch.loc[ch["charge_num"] == 0, "type"] = UNKNOWN      # обвинение не найдено
    ch["rank"] = ch["c_charge_degree"].map(degree_rank)
    ep = (ch.sort_values(["id", "custody_num"])
            .groupby(["id", "custody_num"], as_index=False)
            .agg(in_custody=("in_custody", "first"), out_custody=("out_custody", "first"),
                 custody_source=("custody_source", "first"), age=("age", "first"),
                 sex=("sex", "first"), race=("race", "first")))
    ep["ep_raw"] = np.arange(len(ep))
    ch = ch.merge(ep[["id", "custody_num", "ep_raw"]], on=["id", "custody_num"])
    return ch, ep


def merge_continuations(ep):
    """Склейка: prison, начатый в день выхода из предыдущего заключения, и
    заключения, начатые в тот же день, что предыдущее, — это продолжение того же эпизода."""
    ep = ep.sort_values(["id", "in_custody", "custody_num"]).reset_index(drop=True)
    prev_out = ep.groupby("id")["out_custody"].shift()
    prev_in = ep.groupby("id")["in_custody"].shift()
    cont = (((ep["custody_source"] == "prison") & (ep["in_custody"] == prev_out)) | (ep["in_custody"] == prev_in))
    if not MERGE:
        cont[:] = False
    ep["merged_into_prev"] = cont
    ep["ep_row"] = (~cont).cumsum() - 1
    return ep


def build_log():
    ch, ep = load()
    ep = merge_continuations(ep)
    real = pd.concat([ep["in_custody"], ep.loc[ep["out_custody"] < PLACEHOLDER_OUT, "out_custody"]])
    obs_end = real.max()
    ep["ongoing"] = ep["out_custody"] >= PLACEHOLDER_OUT
    ep.loc[ep["ongoing"], "out_custody"] = obs_end
    df = (ep.groupby("ep_row", as_index=False)
            .agg(id=("id", "first"), in_custody=("in_custody", "first"), out_custody=("out_custody", "max"),
                 ongoing=("ongoing", "max"), age=("age", "first"), sex=("sex", "first"), race=("race", "first"),
                 custodies=("custody_num", "size"),
                 sources=("custody_source", lambda s: "+".join(s))))
    ch = ch.merge(ep[["ep_raw", "ep_row"]], on="ep_raw")
    # дело уже встречалось у этого человека в более раннем эпизоде (повтор старого дела)
    first_ep = ch.dropna(subset=["case_number"]).groupby(["id", "case_number"])["ep_row"].min()
    fe = pd.MultiIndex.from_frame(ch[["id", "case_number"]]).map(first_ep.to_dict().get)
    ch["old_case"] = ch["case_number"].notna().values & (pd.Series(fe, index=ch.index).fillna(np.inf) < ch["ep_row"])
    oc = ch.groupby("ep_row")["old_case"].agg(["any", "all"])
    src = ch.groupby("ep_row")["charge_source"].first()
    old_stats = dict(episodes_with_old_case=int(oc["any"].sum()), episodes_only_old_case=int(oc["all"].sum()),
                     **{f"only_old_case_src_{k}": int(v) for k, v in src[oc["all"]].value_counts().items()},
                     charges_old_case=int(ch["old_case"].sum()))
    if VARIANT == "oldcase":
        ch.loc[ch["old_case"], "type"] = OLDCASE
    t = ch.groupby(["ep_row", "type"], as_index=False).agg(rank=("rank", "min"), n_charges=("type", "size"))
    t["deg"] = t["rank"].map(DEG).fillna("NA")
    # если тип известен хотя бы у одного обвинения, Unknown в эпизоде не нужен
    known = t[t["type"] != UNKNOWN].groupby("ep_row").size()
    t = t[~((t["type"] == UNKNOWN) & t["ep_row"].isin(known.index))]
    # OldCase только когда в эпизоде нет ни одного нового обвинения; если новые есть,
    # состояние = типы новых обвинений (старые не учитываются)
    new = t[t["type"] != OLDCASE].groupby("ep_row").size()
    t = t[~((t["type"] == OLDCASE) & t["ep_row"].isin(new.index))]
    sets = t.groupby("ep_row")["type"].apply(sorted)
    df["types"] = df["ep_row"].map(sets)
    df["n_types"] = df["types"].str.len()
    df["combo"] = df["types"].str.join("+")
    best = t.groupby("ep_row")["rank"].min()
    df["charge_class"] = df["ep_row"].map(best).map(lambda r: "Felony" if r <= 4 else "Misdemeanor" if r <= 7 else "Other/NA")
    df["dur"] = (df["out_custody"] - df["in_custody"]).dt.days.clip(lower=0)
    df["dur_class"] = pd.cut(df["dur"], DUR_BINS, labels=DUR_LABELS).astype(str)
    df["age_group"] = pd.cut(df["age"], [0, 25, 35, 45, 120], labels=["<25", "25-34", "35-44", "45+"]).astype(str)

    # Justice (побег, пронос в тюрьму, экстрадиция...) внутри заключения идёт ПОСЛЕ
    # остальных обвинений: такой эпизод делится на два шага, {прочие типы} -> {Justice}
    df["step"] = 0
    split = df["types"].apply(lambda x: JUSTICE in x and len(x) > 1)
    jr = df[split].assign(types=[[JUSTICE]] * int(split.sum()), step=1)
    df.loc[split, "types"] = df.loc[split, "types"].apply(lambda x: [y for y in x if y != JUSTICE])
    df = pd.concat([df, jr], ignore_index=True)
    df["n_types"] = df["types"].str.len()
    df["combo"] = df["types"].str.join("+")
    df = df.sort_values([CASE, "in_custody", "ep_row", "step"]).reset_index(drop=True)
    df["event_idx"] = df.groupby(CASE).cumcount() + 1
    df["n_events"] = df.groupby(CASE)[CASE].transform("size")
    df["gap_prev"] = (df["in_custody"] - df.groupby(CASE)["out_custody"].shift()).dt.days
    df["gap_next"] = (df.groupby(CASE)["in_custody"].shift(-1) - df["out_custody"]).dt.days
    df["next_types"] = df.groupby(CASE)["types"].shift(-1)
    same_ep = df.groupby(CASE)["ep_row"].shift(-1) == df["ep_row"]
    df.loc[same_ep, "gap_next"] = 0                  # шаг внутри одного заключения
    df.loc[df["step"] == 1, "gap_prev"] = 0
    df["prev_combo"] = df.groupby(CASE)["combo"].shift(1).fillna("first")
    df["censored"] = df["event_idx"] == df["n_events"]
    df["followup_days"] = np.where(df["censored"], (obs_end - df["out_custody"]).dt.days, np.nan)

    ev = df[[CASE, "ep_row", "step", "event_idx", "in_custody", "types"]].explode("types").rename(columns={"types": "type"})
    ev = ev.merge(t[["ep_row", "type", "deg", "n_charges"]], on=["ep_row", "type"], how="left")
    ev["deg"] = ev["deg"].fillna("NA")
    info = dict(source=os.path.basename(DATA), charge_rows=len(ch), custodies_raw=len(ep),
                custodies_merged=int(ep["merged_into_prev"].sum()), episodes=int(df["step"].eq(0).sum()),
                justice_split_steps=int(df["step"].eq(1).sum()), steps=len(df),
                people=df[CASE].nunique(), case_id="id (person_id)", obs_end=str(obs_end.date()),
                ongoing_custody=int(df.loc[df["step"] == 0, "ongoing"].sum()),
                people_repeat=int((df.groupby(CASE).size() > 1).sum()),
                transitions_observed=int(df["next_types"].notna().sum()),
                cases_terminal=0, cases_censored=int(df["censored"].sum()),
                episodes_multi_type=int((df["n_types"] > 1).sum()),
                share_multi_type=round(float((df["n_types"] > 1).mean()), 3),
                type_events=len(ev), negative_gaps=int((df["gap_next"] < 0).sum()), **old_stats)
    return df, ev, info


def to_pm4py_log(ev):
    rng = np.random.default_rng(SEED)
    ev = ev.copy()
    ev["r"] = rng.random(len(ev))
    ev = ev.sort_values([CASE, "in_custody", "ep_row", "step", "r"])
    ev["ts"] = ev["in_custody"] + pd.to_timedelta(ev.groupby("ep_row").cumcount(), unit="s")
    ev = ev.rename(columns={"type": "activity"})[[CASE, "activity", "ts", "ep_row"]]
    ev[CASE] = ev[CASE].astype(str)
    return pm4py.format_dataframe(ev, case_id=CASE, activity_key="activity", timestamp_key="ts")


# ---------------------------------------------------------------- 2. граф (как в stage1_multi.py)
def pair_rows(df):
    d = df[df["next_types"].notna()].copy()
    d["tr_id"] = np.arange(len(d))
    d["w"] = 1.0 / (d["n_types"] * d["next_types"].str.len())
    d = d.assign(src=d["types"]).explode("src").assign(dst=lambda x: x["next_types"]).explode("dst")
    first = df[df["event_idx"] == 1]
    st = first.assign(src=START, dst=first["types"], w=1.0 / first["n_types"]).explode("dst")
    return d.reset_index(drop=True), st.reset_index(drop=True)


def transition_tables(df, labels):
    d, st = pair_rows(df)
    allp = pd.concat([st[["src", "dst", "w"]], d[["src", "dst", "w"]]], ignore_index=True)
    order_from, order_to = [START] + labels, labels
    wcounts = pd.crosstab(allp["src"], allp["dst"], values=allp["w"], aggfunc="sum").fillna(0)
    wcounts = wcounts.reindex(index=order_from, columns=order_to, fill_value=0)
    probs = wcounts.div(wcounts.sum(axis=1), axis=0)
    raw = pd.crosstab(allp["src"], allp["dst"]).reindex(index=order_from, columns=order_to, fill_value=0)
    n_from = d.drop_duplicates(["tr_id", "src"]).groupby("src").size()
    n_from[START] = int((df["event_idx"] == 1).sum())
    contains = raw.div(n_from.reindex(order_from), axis=0)
    cens = df[df["censored"]].explode("types").groupby("types").size().reindex(labels, fill_value=0)
    raw["CENSORED"] = [0] + cens.tolist()
    g = d.groupby(["src", "dst"])["gap_next"]
    times = pd.DataFrame({"median_days": g.median(), "q25_days": g.quantile(0.25), "q75_days": g.quantile(0.75)})
    arcs = probs.stack().rename("p").to_frame()
    arcs.index.names = ["from", "to"]
    arcs["p_contains"] = contains.stack()
    arcs["n_pairs"] = raw.drop(columns="CENSORED").stack()
    arcs["weight"] = wcounts.stack()
    arcs = arcs[arcs["n_pairs"] > 0].join(times.rename_axis(["from", "to"]))
    arcs = arcs.reset_index().sort_values(["from", "p"], ascending=[True, False]).round(3)
    return wcounts, raw, probs, contains, arcs


def fork_tables(df, labels):
    n = df["n_types"].value_counts().sort_index()
    combos = df.loc[df["n_types"] > 1, "combo"].value_counts()
    combos = pd.DataFrame({"episodes": combos, "share_of_multi": (combos / combos.sum()).round(3)})
    e = df[["ep_row", "types"]].explode("types")
    m = e.merge(e, on="ep_row")
    co = pd.crosstab(m["types_x"], m["types_y"]).reindex(index=labels, columns=labels, fill_value=0)
    co_p = co.div(np.diag(co), axis=0)
    alone = df[df["n_types"] == 1].explode("types")["types"].value_counts()
    tot = e["types"].value_counts()
    solo = pd.DataFrame({"episodes_with_type": tot, "alone": alone.reindex(tot.index, fill_value=0)})
    solo["share_alone"] = (solo["alone"] / solo["episodes_with_type"]).round(3)
    return n, combos, co, co_p, solo


def save_episode_dfg(arcs, df, path, perf_path):
    a = arcs[arcs["from"] != START]
    dfg = {(r["from"], r["to"]): int(r["n_pairs"]) for _, r in a.iterrows()}
    sa = {k: int(v) for k, v in df[df["event_idx"] == 1].explode("types")["types"].value_counts().items()}
    ea = {k: int(v) for k, v in df[df["censored"]].explode("types")["types"].value_counts().items()}
    pm4py.save_vis_dfg(dfg, sa, ea, path)
    perf = {(r["from"], r["to"]): float(r["median_days"]) * 86400 for _, r in a.iterrows() if pd.notna(r["median_days"])}
    pm4py.save_vis_performance_dfg(perf, sa, ea, perf_path)


def save_main_graph(arcs, path, min_p=0.05):
    """Итоговый граф переходов: дуги с p >= min_p, подпись — вероятность и медиана дней."""
    import graphviz
    g = graphviz.Digraph(format="png", graph_attr=dict(rankdir="LR", dpi="160", fontname="Helvetica"),
                         node_attr=dict(shape="box", style="rounded,filled", fillcolor="#e8eef7", fontname="Helvetica"),
                         edge_attr=dict(fontname="Helvetica", fontsize="9"))
    a = arcs[arcs["p"] >= min_p]
    for _, r in a.iterrows():
        lbl = f'{r["p"]:.2f}' + (f'\n{r["median_days"]:.0f} д' if pd.notna(r["median_days"]) else "")
        g.edge(r["from"], r["to"], label=lbl, penwidth=f'{0.5 + 5 * r["p"]:.2f}',
               color="#3b6db3" if r["from"] != r["to"] else "#b3473b")
    g.node(START, shape="circle", fillcolor="#cfe8cf")
    g.render(path, cleanup=True)


# ---------------------------------------------------------------- 3. причины
def transition_drivers(df, labels):
    d, _ = pair_rows(df)
    feats = ["sex", "age_group", "race", "charge_class", "dur_class", "gap_prev_bin", "event_idx_bin",
             "n_types_bin", "prev_combo"]
    d["gap_prev_bin"] = pd.cut(d["gap_prev"].fillna(-1), [-np.inf, -0.5, 30, 180, 365, np.inf],
                               labels=["first", "<=30d", "31-180d", "181-365d", ">365d"]).astype(str)
    d["event_idx_bin"] = d["event_idx"].clip(upper=4).astype(str).replace("4", "4+")
    d["n_types_bin"] = d["n_types"].clip(upper=3).astype(str).replace("3", "3+")
    rows = []
    for src in labels:
        sub = d[d["src"] == src]
        for f in feats:
            ct = pd.crosstab(sub[f], sub["dst"], values=sub["w"], aggfunc="sum").fillna(0)
            ct = ct.loc[ct.sum(axis=1) >= 20]
            ct = ct.loc[:, ct.sum(axis=0) > 0]
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            v, p = base.cramers_v(ct)
            rows.append(dict(from_state=src, feature=f, n=round(ct.values.sum(), 1), cramers_v=round(v, 3), p_value=p))
    assoc = pd.DataFrame(rows).sort_values(["from_state", "cramers_v"], ascending=[True, False])
    prof = []
    for f in feats:
        g = pd.crosstab(d[f], d["dst"], values=d["w"], aggfunc="sum", normalize="index").fillna(0)
        g = g.reindex(columns=labels, fill_value=0)
        g.insert(0, "n", d.groupby(f)["w"].sum().round(1))
        g.index = [f"{f}={i}" for i in g.index]
        prof.append(g.round(3))
    profile = pd.concat(prof)
    X = pd.get_dummies(d[["src"] + feats].astype(str), drop_first=False)
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=100, random_state=SEED,
                                  class_weight="balanced").fit(X, d["dst"], sample_weight=d["w"])
    imp = pd.Series(tree.feature_importances_, index=X.columns)
    imp = imp.groupby(lambda c: next(f for f in ["src"] + feats if c.startswith(f + "_"))).sum()
    return assoc, profile, imp.sort_values(ascending=False).round(3), export_text(tree, feature_names=list(X.columns))


# ---------------------------------------------------------------- main
_JOB = None


def _eval_one(name):
    models, train_log, test_log = _JOB
    # выравнивания для Inductive Miner с 15 состояниями и развилками считаются
    # больше 2.5 часов на модель, поэтому для IM они не считаются (fitness по
    # token replay и precision считаются); обход достижимости ограничен 300 000
    # маркировок, иначе сеть HM 0.5 не помещается в память
    return base.evaluate({name: models[name]}, train_log, test_log,
                         skip_align=[n for n in models if n.startswith("IM_")], max_states=300_000)


def main():
    df, ev, info = build_log()
    print("Журнал:", info)
    labels = ev["type"].value_counts().index.tolist()
    df.drop(columns=["types", "next_types"]).assign(types=df["combo"]).to_csv(os.path.join(OUT, "event_table.csv"), index=False)
    ev.to_csv(os.path.join(OUT, "event_types.csv"), index=False)
    log = to_pm4py_log(ev)
    log.to_csv(os.path.join(OUT, "event_log.csv"), index=False)
    pm4py.write_xes(log, os.path.join(OUT, "event_log.xes"))
    pd.Series(info).to_csv(os.path.join(TAB, "log_info.csv"), header=["value"])

    tc = ev["type"].value_counts()
    pd.DataFrame({"episodes_with_type": tc, "share_of_episodes": (tc / len(df)).round(3)}).to_csv(
        os.path.join(TAB, "state_summary.csv"))
    var = df.groupby(CASE)["combo"].agg(" > ".join).value_counts()
    pd.DataFrame({"cases": var, "share": (var / var.sum()).round(4)}).head(30).to_csv(os.path.join(TAB, "top_variants.csv"))
    df.groupby(CASE).size().value_counts().sort_index().to_csv(os.path.join(TAB, "episodes_per_person.csv"), header=["people"])

    n, combos, co, co_p, solo = fork_tables(df, labels)
    n.to_csv(os.path.join(TAB, "fork_types_per_episode.csv"), header=["episodes"])
    combos.head(40).to_csv(os.path.join(TAB, "fork_combinations.csv"))
    co_p.round(3).to_csv(os.path.join(TAB, "fork_cooccurrence.csv"))
    solo.to_csv(os.path.join(TAB, "fork_type_alone.csv"))
    base.plot_matrix(co_p, co, os.path.join(FIG, "fork_cooccurrence.png"))

    wcounts, raw, probs, contains, arcs = transition_tables(df, labels)
    arcs.to_csv(os.path.join(TAB, "transition_arcs.csv"), index=False)
    raw.to_csv(os.path.join(TAB, "transition_pair_counts.csv"))
    wcounts.round(2).to_csv(os.path.join(TAB, "transition_weighted_counts.csv"))
    probs.round(4).to_csv(os.path.join(TAB, "transition_probs.csv"))
    contains.round(4).to_csv(os.path.join(TAB, "transition_contains.csv"))
    med = arcs.set_index(["from", "to"])["median_days"].to_dict()
    base.plot_matrix(probs, raw.drop(columns="CENSORED"), os.path.join(FIG, "transition_matrix.png"), med)
    base.plot_matrix(contains.clip(upper=1), raw.drop(columns="CENSORED"), os.path.join(FIG, "transition_contains.png"))
    save_episode_dfg(arcs, df, os.path.join(FIG, "dfg_frequency.png"), os.path.join(FIG, "dfg_performance.png"))
    save_main_graph(arcs, os.path.join(FIG, "transition_graph_p0.05"))

    assoc, profile, imp, tree_txt = transition_drivers(df, labels)
    assoc.to_csv(os.path.join(TAB, "drivers_cramers_v.csv"), index=False)
    profile.to_csv(os.path.join(TAB, "drivers_next_state_profile.csv"))
    imp.to_csv(os.path.join(TAB, "drivers_tree_importance.csv"), header=["importance"])
    with open(os.path.join(TAB, "drivers_tree.txt"), "w") as f:
        f.write(tree_txt)

    if not EVAL:
        return
    cases = log["case:concept:name"].unique()
    tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(cases, groups=cases))
    train_log = log[log["case:concept:name"].isin(set(cases[tr_idx]))]
    test_log = log[log["case:concept:name"].isin(set(cases[te_idx]))]
    models = {}
    for nt in [0.0, 0.2, 0.5]:
        net, im, fm = pm4py.discover_petri_net_inductive(train_log, noise_threshold=nt)
        models[f"IM_noise{nt}"] = (net, im, fm)
        pm4py.save_vis_petri_net(net, im, fm, os.path.join(FIG, f"petri_IM_noise{nt}.png"))
    pm4py.save_vis_process_tree(pm4py.discover_process_tree_inductive(train_log, noise_threshold=0.2),
                                os.path.join(FIG, "process_tree_IM_noise0.2.png"))
    for dep in [0.5, 0.9]:
        net, im, fm = pm4py.discover_petri_net_heuristics(train_log, dependency_threshold=dep)
        models[f"HM_dep{dep}"] = (net, im, fm)
        pm4py.save_vis_petri_net(net, im, fm, os.path.join(FIG, f"petri_HM_dep{dep}.png"))
    pm4py.save_vis_heuristics_net(pm4py.discover_heuristics_net(train_log, dependency_threshold=0.9),
                                  os.path.join(FIG, "heuristics_net_dep0.9.png"))
    dfg_tr, sa_tr, ea_tr = pm4py.discover_dfg(train_log)
    out_tot = pd.Series({k: v for k, v in dfg_tr.items()}).groupby(level=0).sum()
    sa_tot = sum(sa_tr.values())
    for thr in [0.05, 0.10, 0.20]:
        dfg_f = {k: v for k, v in dfg_tr.items() if v / out_tot[k[0]] >= thr}
        sa_f = {k: v for k, v in sa_tr.items() if v / sa_tot >= thr}
        net, im, fm = pm4py.convert_to_petri_net(dfg_f, sa_f, ea_tr)
        models[f"DFG_p>={thr}"] = (net, im, fm)
    # оценка моделей параллельно: выравнивания для IM с развилками считаются долго
    global _JOB
    _JOB = (models, train_log, test_log)
    with mp.get_context("fork").Pool(min(4, len(models))) as pool:
        res = pool.map(_eval_one, list(models), chunksize=1)
    quality = pd.concat([r[0] for r in res], ignore_index=True)
    edges = pd.concat([r[1] for r in res], ignore_index=True)
    quality.to_csv(os.path.join(TAB, "model_quality.csv"), index=False)
    edges.to_csv(os.path.join(TAB, "model_edges_test.csv"), index=False)
    print(quality[quality["split"] == "test"].to_string())


if __name__ == "__main__":
    main()
