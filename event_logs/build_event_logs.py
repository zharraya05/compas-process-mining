"""
Таблицы логов (event log) для PM4Py из compas_custody_charges.csv.

Правила те же, что в Этапе 1 (stage1_compasdb/stage1_compasdb.py и
stage1_compasdb_no_oldcase/stage1_no_oldcase.py), код логики перенесён сюда без
изменений, чтобы скрипт работал сам по себе:
  * объект (case) = человек, id = person_id из compas.db;
  * эпизод = заключение; prison, начатый в день выхода из предыдущего заключения,
    и заключения, начатые в тот же день, склеиваются с предыдущим;
  * 15 типов преступлений (crime_categories.categorize_fine); Unknown убирается,
    если в эпизоде есть известный тип;
  * эпизод с Justice и другими типами делится на два шага {прочие} -> {Justice};
  * в эпизоде из нескольких типов — по событию на тип, порядок внутри эпизода
    случайный (seed 42), к времени добавляются секунды 0, 1, 2... чтобы PM4Py
    сохранил этот порядок;
  * терминального состояния нет, последний шаг человека цензурирован концом
    наблюдения 2016-04-19.

Варианты учёта старых дел (обвинение «старое», если его case_number уже был у этого
id в более раннем эпизоде; в смешанных эпизодах старые обвинения не учитываются):
  oldcase    — основной: эпизод только из старых обвинений = состояние OldCase;
  no_oldcase — такие эпизоды убираются из последовательности человека;
  all_new    — все обвинения считаются новыми (старые дела не выделяются).

Запуск: python build_event_logs.py [путь к compas_custody_charges.csv] [--xes]
  -> event_log_<вариант>.csv / .xlsx (+ .xes с флагом --xes), steps_<вариант>.csv, log_summary.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from crime_categories import categorize_fine as categorize, UNKNOWN   # noqa: E402

DATA = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else os.path.join(HERE, "compas_custody_charges.csv")
if not os.path.exists(DATA):
    DATA = os.path.join(HERE, "..", "stage1_fixed_input", "compas_custody_charges.csv")
VARIANTS = ["oldcase", "no_oldcase", "all_new"]
SEED = 42
OLDCASE, JUSTICE = "OldCase", "Justice"
PLACEHOLDER_OUT = pd.Timestamp("2020-01-01")       # «ещё в заключении»
DEGREE_RANK = {"F1": 1, "F2": 2, "F3": 3, "F5": 4, "F6": 4, "F7": 4, "M1": 5, "M2": 6, "MO3": 7}
DEG = {1: "F", 2: "F", 3: "F", 4: "F", 5: "M", 6: "M", 7: "M"}


def degree_rank(d):
    return DEGREE_RANK.get(str(d).strip("()"), 8)


def uniq(s):
    return "; ".join(dict.fromkeys(x for x in s if isinstance(x, str) and x))


def load():
    ch = pd.read_csv(DATA, parse_dates=["in_custody", "out_custody"], low_memory=False)
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
    # старое дело: case_number уже был у этого id в более раннем эпизоде
    first_ep = ch.dropna(subset=["case_number"]).groupby(["id", "case_number"])["ep_row"].min()
    fe = pd.MultiIndex.from_frame(ch[["id", "case_number"]]).map(first_ep.to_dict().get)
    ch["old_case"] = ch["case_number"].notna().values & (pd.Series(fe, index=ch.index).fillna(np.inf) < ch["ep_row"])
    return ch, ep


def build(variant, ch, ep):
    ch = ch.copy()
    real = pd.concat([ep["in_custody"], ep.loc[ep["out_custody"] < PLACEHOLDER_OUT, "out_custody"]])
    obs_end = real.max()
    ep = ep.assign(ongoing=ep["out_custody"] >= PLACEHOLDER_OUT)
    ep.loc[ep["ongoing"], "out_custody"] = obs_end
    df = (ep.groupby("ep_row", as_index=False)
            .agg(id=("id", "first"), in_custody=("in_custody", "first"), out_custody=("out_custody", "max"),
                 ongoing=("ongoing", "max"), age=("age", "first"), sex=("sex", "first"), race=("race", "first"),
                 custodies=("custody_num", "size"), sources=("custody_source", lambda s: "+".join(s))))
    # сведения об эпизоде по всем его обвинениям (до выбора состояния)
    epi = ch.groupby("ep_row").agg(n_charges_all=("type", "size"), n_old_charges=("old_case", "sum"),
                                   case_numbers_all=("case_number", uniq))
    epi["old_case_numbers"] = ch[ch["old_case"]].groupby("ep_row")["case_number"].agg(uniq)
    epi["old_case_numbers"] = epi["old_case_numbers"].fillna("")
    df = df.merge(epi, left_on="ep_row", right_index=True)
    df["has_old_case"] = df["n_old_charges"] > 0
    df["only_old_case"] = df["n_old_charges"] == df["n_charges_all"]

    if variant != "all_new":
        ch.loc[ch["old_case"], "type"] = OLDCASE
    t = ch.groupby(["ep_row", "type"], as_index=False).agg(
        rank=("rank", "min"), n_charges=("type", "size"), case_numbers=("case_number", uniq),
        charges=("c_charge_desc", uniq), degrees=("degree", uniq))
    t["deg"] = t["rank"].map(DEG).fillna("NA")
    known = t[t["type"] != UNKNOWN].groupby("ep_row").size()
    t = t[~((t["type"] == UNKNOWN) & t["ep_row"].isin(known.index))]
    new = t[t["type"] != OLDCASE].groupby("ep_row").size()
    t = t[~((t["type"] == OLDCASE) & t["ep_row"].isin(new.index))]
    df["types"] = df["ep_row"].map(t.groupby("ep_row")["type"].apply(sorted))
    if variant == "no_oldcase":
        df = df[df["types"].apply(lambda x: x != [OLDCASE])].reset_index(drop=True)
        t = t[t["type"] != OLDCASE]
    best = t.groupby("ep_row")["rank"].min()
    df["charge_class"] = df["ep_row"].map(best).map(lambda r: "Felony" if r <= 4 else "Misdemeanor" if r <= 7 else "Other/NA")
    df["dur_days"] = (df["out_custody"] - df["in_custody"]).dt.days.clip(lower=0)

    # шаг Justice после остальных типов того же заключения
    df["step"] = 0
    split = df["types"].apply(lambda x: JUSTICE in x and len(x) > 1)
    jr = df[split].assign(types=[[JUSTICE]] * int(split.sum()), step=1)
    df.loc[split, "types"] = df.loc[split, "types"].apply(lambda x: [y for y in x if y != JUSTICE])
    df = pd.concat([df, jr], ignore_index=True)
    df["n_types"] = df["types"].str.len()
    df["state"] = df["types"].str.join("+")
    df = df.sort_values(["id", "in_custody", "ep_row", "step"]).reset_index(drop=True)
    g = df.groupby("id")
    df["event_idx"] = g.cumcount() + 1
    df["n_events"] = g["id"].transform("size")
    df["gap_prev_days"] = (df["in_custody"] - g["out_custody"].shift()).dt.days
    df["gap_next_days"] = (g["in_custody"].shift(-1) - df["out_custody"]).dt.days
    df.loc[g["ep_row"].shift(-1) == df["ep_row"], "gap_next_days"] = 0
    df.loc[df["step"] == 1, "gap_prev_days"] = 0
    df["prev_state"] = g["state"].shift(1).fillna("")
    df["next_state"] = g["state"].shift(-1).fillna("")
    df["censored"] = df["event_idx"] == df["n_events"]
    df["followup_days"] = np.where(df["censored"], (obs_end - df["out_custody"]).dt.days, np.nan)
    df["episode_id"] = df["ep_row"]

    # событие = (шаг, тип); порядок внутри шага случайный (как в Этапе 1)
    ev = df.explode("types").rename(columns={"types": "type"})
    ev = ev.merge(t[["ep_row", "type", "deg", "n_charges", "case_numbers", "charges", "degrees"]],
                  on=["ep_row", "type"], how="left")
    ev["deg"] = ev["deg"].fillna("NA")
    rng = np.random.default_rng(SEED)
    ev["r"] = rng.random(len(ev))
    ev = ev.sort_values(["id", "in_custody", "ep_row", "step", "r"]).reset_index(drop=True)
    ev["order_in_episode"] = ev.groupby("ep_row").cumcount()
    ev["time:timestamp"] = (ev["in_custody"] + pd.to_timedelta(ev["order_in_episode"], unit="s")).dt.tz_localize("UTC")
    ev["case:concept:name"] = ev["id"].astype(str)
    ev["concept:name"] = ev["type"]
    ev = ev.rename(columns={"deg": "type_degree_class", "n_charges": "type_n_charges",
                            "case_numbers": "type_case_numbers", "charges": "type_charges",
                            "degrees": "type_degrees", "event_idx": "step_idx", "n_events": "n_steps"})
    ev_cols = ["case:concept:name", "concept:name", "time:timestamp",
               "id", "episode_id", "step", "step_idx", "n_steps", "order_in_episode",
               "in_custody", "out_custody", "ongoing", "dur_days", "custodies", "sources",
               "state", "n_types", "prev_state", "next_state", "gap_prev_days", "gap_next_days",
               "censored", "followup_days",
               "type_degree_class", "type_degrees", "type_n_charges", "type_case_numbers", "type_charges",
               "charge_class", "n_charges_all", "n_old_charges", "has_old_case", "only_old_case",
               "case_numbers_all", "old_case_numbers", "age", "sex", "race"]
    st = df.rename(columns={"event_idx": "step_idx", "n_events": "n_steps"})
    st_cols = ["id", "episode_id", "step", "step_idx", "n_steps", "in_custody", "out_custody", "ongoing",
               "dur_days", "custodies", "sources", "state", "n_types", "prev_state", "next_state",
               "gap_prev_days", "gap_next_days", "censored", "followup_days", "charge_class",
               "n_charges_all", "n_old_charges", "has_old_case", "only_old_case",
               "case_numbers_all", "old_case_numbers", "age", "sex", "race"]
    info = dict(variant=variant, people=df["id"].nunique(), episodes=int(df["step"].eq(0).sum()),
                justice_steps=int(df["step"].eq(1).sum()), steps=len(df), events=len(ev),
                transitions=int((~df["censored"]).sum()), people_repeat=int((df.groupby("id").size() > 1).sum()),
                multi_type_steps=int((df["n_types"] > 1).sum()), activities=ev["concept:name"].nunique(),
                obs_end=str(obs_end.date()))
    return ev[ev_cols], st[st_cols], info


def main():
    ch, ep = load()
    rows = []
    for v in VARIANTS:
        ev, st, info = build(v, ch, ep)
        ev.to_csv(os.path.join(HERE, f"event_log_{v}.csv"), index=False)
        st.to_csv(os.path.join(HERE, f"steps_{v}.csv"), index=False)
        with pd.ExcelWriter(os.path.join(HERE, f"event_log_{v}.xlsx")) as w:
            e = ev.copy()
            e["time:timestamp"] = e["time:timestamp"].dt.tz_localize(None)
            e.to_excel(w, sheet_name="events", index=False)
            st.to_excel(w, sheet_name="steps", index=False)
        if "--xes" in sys.argv:          # XES ~50 МБ на вариант, по умолчанию не пишем
            import pm4py
            pm4py.write_xes(ev, os.path.join(HERE, f"event_log_{v}.xes"), case_id_key="case:concept:name")
        rows.append(info)
        print(info)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "log_summary.csv"), index=False)


if __name__ == "__main__":
    main()
