"""Общий код журнала событий варианта «категории × 2» (из stage1\\oldcase_x2\\stage1_categories_x2.ipynb).

build(variant, ch, ep, one_event=False, merge_old=()) — те же правила, что в основном ноутбуке, плюс параметры
экспериментов:
  one_event  — в каждом заключении остаётся одна категория, самая тяжёлая (Justice — только если он один);
  merge_old  — список состояний «OldCase: X», которые сливаются в одно «OldCase: Other».
"""
import numpy as np
import pandas as pd

from crime_categories import categorize_fine as categorize, UNKNOWN

SEED = 42
OLDCASE, JUSTICE = "OldCase", "Justice"
OLD_PREFIX = "OldCase: "
VARIANTS = ["all_new", "oldcase", "no_oldcase", "x2"]
VNAME = {"all_new": "все новые", "oldcase": "OldCase", "no_oldcase": "без OldCase", "x2": "категории × 2"}
PLACEHOLDER_OUT = pd.Timestamp("2020-01-01")
DEGREE_RANK = {"F1": 1, "F2": 2, "F3": 3, "F5": 4, "F6": 4, "F7": 4, "M1": 5, "M2": 6, "MO3": 7}
DEG = {1: "F", 2: "F", 3: "F", 4: "F", 5: "M", 6: "M", 7: "M"}


def degree_rank(d):
    return DEGREE_RANK.get(str(d).strip("()"), 8)


def uniq(s):
    return "; ".join(dict.fromkeys(x for x in s if isinstance(x, str) and x))


def base_type(a):
    # «OldCase: Drug» -> «Drug»
    return a[len(OLD_PREFIX):] if a.startswith(OLD_PREFIX) else a


def load(data):
    ch = pd.read_csv(data, parse_dates=["in_custody", "out_custody"], low_memory=False)
    ch["type"] = ch["c_charge_desc"].map(categorize)
    ch.loc[ch["charge_num"] == 0, "type"] = UNKNOWN      # обвинение не найдено
    ch["rank"] = ch["c_charge_degree"].map(degree_rank)
    ch["degree"] = ch["c_charge_degree"].astype(str).str.strip("()").replace("nan", "")
    ep = (ch.sort_values(["id", "custody_num"])
            .groupby(["id", "custody_num"], as_index=False)
            .agg(in_custody=("in_custody", "first"), out_custody=("out_custody", "first"),
                 custody_source=("custody_source", "first"), age=("age", "first"),
                 sex=("sex", "first"), race=("race", "first")))
    ep["ep_raw"] = np.arange(len(ep))
    ch = ch.merge(ep[["id", "custody_num", "ep_raw"]], on=["id", "custody_num"])
    # склейка: prison с даты выхода из предыдущего заключения и заключения того же дня
    ep = ep.sort_values(["id", "in_custody", "custody_num"]).reset_index(drop=True)
    prev_out = ep.groupby("id")["out_custody"].shift()
    prev_in = ep.groupby("id")["in_custody"].shift()
    cont = ((ep["custody_source"] == "prison") & (ep["in_custody"] == prev_out)) | (ep["in_custody"] == prev_in)
    ep["merged_into_prev"] = cont
    ep["ep_row"] = (~cont).cumsum() - 1
    ch = ch.merge(ep[["ep_raw", "ep_row"]], on="ep_raw")
    # старое дело: case_number уже был у этого id в более раннем заключении
    first_ep = ch.dropna(subset=["case_number"]).groupby(["id", "case_number"])["ep_row"].min()
    fe = pd.MultiIndex.from_frame(ch[["id", "case_number"]]).map(first_ep.to_dict().get)
    ch["old_case"] = ch["case_number"].notna().values & (pd.Series(fe, index=ch.index).fillna(np.inf) < ch["ep_row"])
    return ch, ep



def is_justice(a):
    return a in (JUSTICE, OLD_PREFIX + JUSTICE)


def build(variant, ch, ep, one_event=False, merge_old=()):
    ch = ch.copy()
    # 1. конец наблюдения
    real = pd.concat([ep["in_custody"], ep.loc[ep["out_custody"] < PLACEHOLDER_OUT, "out_custody"]])
    obs_end = real.max()
    ep = ep.assign(ongoing=ep["out_custody"] >= PLACEHOLDER_OUT)
    ep.loc[ep["ongoing"], "out_custody"] = obs_end
    df = (ep.groupby("ep_row", as_index=False)
            .agg(id=("id", "first"), in_custody=("in_custody", "first"), out_custody=("out_custody", "max"),
                 ongoing=("ongoing", "max"), age=("age", "first"), sex=("sex", "first"), race=("race", "first"),
                 custodies=("custody_num", "size"), sources=("custody_source", lambda s: "+".join(s))))
    epi = ch.groupby("ep_row").agg(n_charges_all=("type", "size"), n_old_charges=("old_case", "sum"),
                                   case_numbers_all=("case_number", uniq))
    epi["old_case_numbers"] = ch[ch["old_case"]].groupby("ep_row")["case_number"].agg(uniq)
    epi["old_case_numbers"] = epi["old_case_numbers"].fillna("")
    df = df.merge(epi, left_on="ep_row", right_index=True)
    df["has_old_case"] = df["n_old_charges"] > 0
    df["only_old_case"] = df["n_old_charges"] == df["n_charges_all"]

    # 2. категория старых обвинений
    if variant in ("oldcase", "no_oldcase"):
        ch.loc[ch["old_case"], "type"] = OLDCASE
    elif variant == "x2":
        ch.loc[ch["old_case"], "type"] = OLD_PREFIX + ch.loc[ch["old_case"], "type"]
    # эксперимент «слить редкие»: старые обвинения этих категорий -> общее состояние «OldCase: Other»
    if merge_old:
        ch.loc[ch["type"].isin(list(merge_old)), "type"] = OLD_PREFIX + "Other"
    # 3. набор категорий заключения
    t = ch.groupby(["ep_row", "type"], as_index=False).agg(
        rank=("rank", "min"), n_charges=("type", "size"), case_numbers=("case_number", uniq),
        charges=("c_charge_desc", uniq), degrees=("degree", uniq))
    t["deg"] = t["rank"].map(DEG).fillna("NA")
    unk = t["type"].isin([UNKNOWN, OLD_PREFIX + UNKNOWN])
    known = t[~unk].groupby("ep_row").size()
    t = t[~(unk & t["ep_row"].isin(known.index))]
    if variant in ("oldcase", "no_oldcase"):
        new = t[t["type"] != OLDCASE].groupby("ep_row").size()
        t = t[~((t["type"] == OLDCASE) & t["ep_row"].isin(new.index))]
    # ВЫБОР ДЛЯ СМЕШАННЫХ (x2): старые категории смешанного заключения сохраняются.
    # Чтобы учитывать только новые, раскомментировать две строки:
    # new = t[~t["type"].str.startswith(OLD_PREFIX)].groupby("ep_row").size()
    # t = t[~(t["type"].str.startswith(OLD_PREFIX) & t["ep_row"].isin(new.index))]
    df["types"] = df["ep_row"].map(t.groupby("ep_row")["type"].apply(sorted))
    # эксперимент «одно событие на заключение»: только самая тяжёлая категория (при равной тяжести — случайная)
    if one_event:
        rng1 = np.random.default_rng(SEED)
        t = t.assign(r=rng1.random(len(t))).sort_values(["ep_row", "rank", "r"])
        nonj = t.assign(nj=~t["type"].map(is_justice)).groupby("ep_row")["nj"].transform("any")
        t = t[~(t["type"].map(is_justice) & nonj)]                       # Justice — только если других нет
        t = t.groupby("ep_row", as_index=False).head(1).drop(columns="r")
        df["types"] = df["ep_row"].map(t.groupby("ep_row")["type"].apply(sorted))
    # 4. без OldCase: заключения только из старых дел убираются
    if variant == "no_oldcase":
        df = df[df["types"].apply(lambda x: x != [OLDCASE])].reset_index(drop=True)
        t = t[t["type"] != OLDCASE]
    best = t.groupby("ep_row")["rank"].min()
    df["charge_class"] = df["ep_row"].map(best).map(lambda r: "Felony" if r <= 4 else "Misdemeanor" if r <= 7 else "Other/NA")
    df["dur_days"] = (df["out_custody"] - df["in_custody"]).dt.days.clip(lower=0)

    # 5. Justice отдельным шагом после остальных категорий того же заключения
    df["step"] = 0
    split = df["types"].apply(lambda x: any(map(is_justice, x)) and not all(map(is_justice, x)))
    jr = df[split].copy()
    jr["types"] = jr["types"].apply(lambda x: [y for y in x if is_justice(y)])
    jr["step"] = 1
    df.loc[split, "types"] = df.loc[split, "types"].apply(lambda x: [y for y in x if not is_justice(y)])
    df = pd.concat([df, jr], ignore_index=True)
    df["n_types"] = df["types"].str.len()
    df["state"] = df["types"].str.join("+")
    # 6. порядок шагов, интервалы, соседние состояния, цензурирование
    df = df.sort_values(["id", "in_custody", "ep_row", "step"]).reset_index(drop=True)
    g = df.groupby("id")
    df["step_idx"] = g.cumcount() + 1
    df["n_steps"] = g["id"].transform("size")
    df["gap_prev_days"] = (df["in_custody"] - g["out_custody"].shift()).dt.days
    df["gap_next_days"] = (g["in_custody"].shift(-1) - df["out_custody"]).dt.days
    df.loc[g["ep_row"].shift(-1) == df["ep_row"], "gap_next_days"] = 0
    df.loc[df["step"] == 1, "gap_prev_days"] = 0
    df["prev_state"] = g["state"].shift(1).fillna("")
    df["next_state"] = g["state"].shift(-1).fillna("")
    df["censored"] = df["step_idx"] == df["n_steps"]
    df["followup_days"] = np.where(df["censored"], (obs_end - df["out_custody"]).dt.days, np.nan)
    df["episode_id"] = df["ep_row"]

    # 7. событие = (шаг, категория); порядок внутри шага — по тяжести (rank: 1 = F1 самое тяжёлое),
    #    при равной тяжести — случайный
    ev = df.explode("types").rename(columns={"types": "type"})
    ev = ev.merge(t[["ep_row", "type", "rank", "deg", "n_charges", "case_numbers", "charges", "degrees"]],
                  on=["ep_row", "type"], how="left")
    ev["deg"] = ev["deg"].fillna("NA")
    rng = np.random.default_rng(SEED)
    ev["r"] = rng.random(len(ev))
    ev = ev.sort_values(["id", "in_custody", "ep_row", "step", "rank", "r"]).reset_index(drop=True)
    ev["order_in_episode"] = ev.groupby("ep_row").cumcount()
    ev["time:timestamp"] = (ev["in_custody"] + pd.to_timedelta(ev["order_in_episode"], unit="s")).dt.tz_localize("UTC")
    ev["case:concept:name"] = ev["id"].astype(str)
    ev["concept:name"] = ev["type"]
    ev["base_type"] = ev["type"].map(base_type)
    ev["is_old"] = ev["type"].str.startswith(OLD_PREFIX) | (ev["type"] == OLDCASE)
    ev = ev.rename(columns={"deg": "type_degree_class", "rank": "type_rank", "n_charges": "type_n_charges",
                            "case_numbers": "type_case_numbers", "charges": "type_charges", "degrees": "type_degrees"})
    ev_cols = ["case:concept:name", "concept:name", "time:timestamp", "base_type", "is_old",
               "id", "episode_id", "step", "step_idx", "n_steps", "order_in_episode",
               "in_custody", "out_custody", "ongoing", "dur_days", "custodies", "sources",
               "state", "n_types", "prev_state", "next_state", "gap_prev_days", "gap_next_days",
               "censored", "followup_days",
               "type_rank", "type_degree_class", "type_degrees", "type_n_charges", "type_case_numbers", "type_charges",
               "charge_class", "n_charges_all", "n_old_charges", "has_old_case", "only_old_case",
               "case_numbers_all", "old_case_numbers", "age", "sex", "race"]
    st_cols = ["id", "episode_id", "step", "step_idx", "n_steps", "in_custody", "out_custody", "ongoing",
               "dur_days", "custodies", "sources", "types", "state", "n_types", "prev_state", "next_state",
               "gap_prev_days", "gap_next_days", "censored", "followup_days", "charge_class",
               "n_charges_all", "n_old_charges", "has_old_case", "only_old_case",
               "case_numbers_all", "old_case_numbers", "age", "sex", "race"]
    info = dict(variant=variant, people=df["id"].nunique(), episodes=int(df["step"].eq(0).sum()),
                justice_steps=int(df["step"].eq(1).sum()), steps=len(df), events=len(ev),
                transitions=int((~df["censored"]).sum()), people_repeat=int((df.groupby("id").size() > 1).sum()),
                multi_type_steps=int((df["n_types"] > 1).sum()), activities=ev["concept:name"].nunique(),
                obs_end=str(obs_end.date()))
    return ev[ev_cols], df[st_cols], info

def split_people(log, seed=SEED, test_share=0.2):
    rng = np.random.default_rng(seed)
    ids = log["case:concept:name"].unique()
    test_ids = set(rng.choice(ids, size=int(test_share * len(ids)), replace=False))
    return log[~log["case:concept:name"].isin(test_ids)], log[log["case:concept:name"].isin(test_ids)]


def sample_cases(log, n):
    keep = sorted(set(log["case:concept:name"]))[:n]
    return log[log["case:concept:name"].isin(keep)]


def evaluate(models, train_s, test_s, align_cases=0):
    """fitness/precision (token-based replay) на выборках train_s/test_s; align_cases > 0 — ещё precision
    через выравнивания на первых align_cases людях из test_s."""
    import time
    import pm4py
    rows = []
    for name, (net, im, fm) in models.items():
        for split, lg in (("train", train_s), ("test", test_s)):
            t0 = time.time()
            fit = pm4py.fitness_token_based_replay(lg, net, im, fm)
            prec = pm4py.precision_token_based_replay(lg, net, im, fm)
            row = dict(model=name, split=split, cases=lg["case:concept:name"].nunique(),
                       fitness_tbr=round(fit["log_fitness"], 4), pct_fit_traces=round(fit["percentage_of_fitting_traces"], 2),
                       precision_tbr=round(prec, 4), f1=round(2 * fit["log_fitness"] * prec / (fit["log_fitness"] + prec), 4),
                       places=len(net.places), transitions=len(net.transitions),
                       silent=sum(t.label is None for t in net.transitions))
            if align_cases and split == "test":
                try:
                    row["precision_align"] = round(pm4py.precision_alignments(sample_cases(lg, align_cases), net, im, fm), 4)
                except Exception as ex:          # выравнивания могут не уложиться в память/время
                    row["precision_align"] = float("nan")
                    print(name, "alignments:", type(ex).__name__)
            row["sec"] = round(time.time() - t0)
            rows.append(row)
            print(row)
    return pd.DataFrame(rows)
