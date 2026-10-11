"""Общий код эксперимента «данные Даши, состояние = тяжесть обвинения».

Тяжесть для каждого заключения из таблицы Даши (individual_custody_timeline_rfm.csv)
берётся из compas.db по правилам stage1/build_episode_charges.py:
  1. person_id: заключение ищется в jailhistory / prisonhistory по имени и дате входа
     (однофамильцы различаются по дате выхода, затем по аресту ±1 день);
  2. обвинения ареста casearrest.arrest_date в окне [in_custody - 1, in_custody + 1] дней;
  3. иначе обвинения с offense_date в окне [in_custody - 7, in_custody + 1];
  4. иначе последний арест за 365 дней до заключения (приговор, нарушение условий);
  5. из нескольких обвинений берётся самое тяжёлое (F1 > F2 > F3 > прочие F > M1 > M2 > MO3).
"""
import sqlite3

import numpy as np
import pandas as pd

SEED = 42
PLACEHOLDER_OUT = pd.Timestamp("2020-01-01")      # у Даши «ещё в заключении»
OBS_END = pd.Timestamp("2016-04-19")

# состояние = степень обвинения; F5–F7 и MO3 редкие, они объединены с соседями
STATE = {"F1": "F1", "F2": "F2", "F3": "F3", "F5": "F_other", "F6": "F_other", "F7": "F_other",
         "M1": "M1", "M2": "M2", "MO3": "M2"}
STATE_ORDER = ["F1", "F2", "F3", "F_other", "M1", "M2", "Unknown"]


def to_state(d):
    """(F3) -> F3; пусто и редкие коды (CO3, NI0, 0, X ...) -> Unknown."""
    return STATE.get(str(d).strip("()"), "Unknown")


DEGREE_RANK = {"F1": 1, "F2": 2, "F3": 3, "F5": 4, "F6": 4, "F7": 4,
               "M1": 5, "M2": 6, "MO3": 7}


def degree_rank(d):
    d = str(d).strip("()")
    return DEGREE_RANK.get(d, 8)


def pick(cands):
    cands = cands.assign(rank=cands["charge_degree"].map(degree_rank))
    return cands.sort_values(["rank", "charge_number"]).iloc[0]


def assign_person_id(df, con, people, arrests):
    """person_id для каждой строки набора Даши (см. п. 1 в описании)."""
    hist = []
    for t in ["jailhistory", "prisonhistory"]:
        h = pd.read_sql(f"select person_id, in_custody, out_custody from {t}", con)
        h["hist"] = t
        hist.append(h)
    h = pd.concat(hist).merge(people, on="person_id")
    h["in_d"] = pd.to_datetime(h["in_custody"], format="mixed", errors="coerce").dt.normalize()
    h["out_d"] = pd.to_datetime(h["out_custody"], format="mixed", errors="coerce").dt.normalize()
    x = pd.DataFrame({"ep_row": np.arange(len(df)), "name": df["name"],
                      "in_d": pd.to_datetime(df["in_custody"]).dt.normalize(),
                      "out_d": pd.to_datetime(df["out_custody"]).dt.normalize()})
    m = x.merge(h[["name", "in_d", "out_d", "person_id", "hist"]], on=["name", "in_d"], how="inner",
                suffixes=("", "_h")).drop_duplicates(["ep_row", "person_id"])
    n = m.groupby("ep_row")["person_id"].transform("nunique")
    m["out_ok"] = m["out_d"] == m["out_d_h"]
    arr = arrests[["person_id", "arrest_date"]]
    near = m.merge(arr, on="person_id", how="left")
    near = near[(near["arrest_date"] - near["in_d"]).abs() <= pd.Timedelta(days=1)]
    m["arrest_ok"] = m.set_index(["ep_row", "person_id"]).index.isin(near.set_index(["ep_row", "person_id"]).index)
    m = m.sort_values(["ep_row", "out_ok", "arrest_ok", "person_id"], ascending=[True, False, False, True])
    m["person_id_source"] = np.where(n > 1, "ambiguous_resolved", m["hist"])
    best = m.drop_duplicates("ep_row").set_index("ep_row")
    out = pd.DataFrame({"ep_row": x["ep_row"]})
    out["person_id"] = out["ep_row"].map(best["person_id"])
    out["person_id_source"] = out["ep_row"].map(best["person_id_source"]).fillna("none")
    # строки без даты: только если имя однозначно
    uniq = people.groupby("name")["person_id"].agg(lambda s: s.iloc[0] if s.nunique() == 1 else np.nan)
    miss = out["person_id"].isna()
    out.loc[miss, "person_id"] = df.loc[miss, "name"].map(uniq)
    out.loc[miss & out["person_id"].notna(), "person_id_source"] = "unique_name"
    return out



def episode_degree(df, db_path):
    """Тяжесть самого тяжёлого обвинения каждого заключения (п. 1–5 выше).
    Возвращает df + person_id, person_id_source, ep_charge_desc, ep_charge_degree, ep_charge_source, ep_n_charges."""
    con = sqlite3.connect(db_path)
    people = pd.read_sql("select id as person_id, name from people", con)
    arrests = pd.read_sql("select person_id, case_number, arrest_date from casearrest", con)
    charges = pd.read_sql("select person_id, case_number, offense_date, charge_number, "
                          "charge_degree, charge from charge", con)
    arrests["arrest_date"] = pd.to_datetime(arrests["arrest_date"], format="mixed").dt.normalize()
    charges["offense_date"] = pd.to_datetime(charges["offense_date"], format="mixed", errors="coerce").dt.normalize()
    charges = charges.dropna(subset=["charge"])
    arr_ch = arrests.merge(charges, on=["person_id", "case_number"])

    df = df.reset_index(drop=True).copy()
    d = pd.to_datetime(df["in_custody"])
    pid = assign_person_id(df, con, people, arrests)
    df.insert(0, "ep_row", pid["ep_row"])
    df.insert(1, "person_id", pid["person_id"].astype("Int64"))
    df.insert(2, "person_id_source", pid["person_id_source"])
    ids_by_name = people.groupby("name")["person_id"].apply(list)
    arr_by_pid = dict(tuple(arr_ch.groupby("person_id")))
    ch_by_pid = dict(tuple(charges.groupby("person_id")))
    empty = charges.iloc[0:0]
    day1, day7, day365 = pd.Timedelta(days=1), pd.Timedelta(days=7), pd.Timedelta(days=365)

    out = []
    for name, person, day in zip(df["name"], df["person_id"], d):
        rec = dict(ep_charge_desc=np.nan, ep_charge_degree=np.nan, ep_charge_source="none", ep_n_charges=0)
        if pd.notna(day):
            pids = [int(person)] if pd.notna(person) else ids_by_name.get(name, [])
            a = pd.concat([arr_by_pid.get(p, empty) for p in pids]) if pids else empty
            a = a[(a["arrest_date"] >= day - day1) & (a["arrest_date"] <= day + day1)] if len(a) else a
            src = "arrest"
            if not len(a):
                c = pd.concat([ch_by_pid.get(p, empty) for p in pids]) if pids else empty
                a = c[(c["offense_date"] >= day - day7) & (c["offense_date"] <= day + day1)] if len(c) else c
                src = "offense"
            if not len(a) and pids:
                a = pd.concat([arr_by_pid.get(p, empty) for p in pids])
                if len(a):
                    a = a[(a["arrest_date"] < day - day1) & (a["arrest_date"] >= day - day365)]
                    a = a[a["arrest_date"] == a["arrest_date"].max()] if len(a) else a
                src = "prior_case"
            if len(a):
                best = pick(a)
                rec = dict(ep_charge_desc=best["charge"], ep_charge_degree=best["charge_degree"],
                           ep_charge_source=src, ep_n_charges=len(a))
        out.append(rec)
    return pd.concat([df, pd.DataFrame(out)], axis=1)


def build_log(df, degree_col):
    """Журнал событий: одно событие на заключение, состояние = to_state(degree_col).
    Кейс = person_id (если найден), иначе имя. Заключения без даты входа выброшены."""
    x = df[df["in_custody"].notna()].copy()
    x["case:concept:name"] = np.where(x["person_id"].notna(), "p" + x["person_id"].astype(str), "n:" + x["name"])
    x["concept:name"] = x[degree_col].map(to_state)
    x["time:timestamp"] = pd.to_datetime(x["in_custody"])
    out = pd.to_datetime(x["out_custody"])
    x["out_custody"] = out.where(out != PLACEHOLDER_OUT)
    x = x.sort_values(["case:concept:name", "time:timestamp", "ep_row"]).reset_index(drop=True)
    # одинаковые даты входа у одного человека: +1 секунда, чтобы порядок был однозначным
    k = x.groupby(["case:concept:name", "time:timestamp"]).cumcount()
    x["time:timestamp"] = x["time:timestamp"] + pd.to_timedelta(k, unit="s")
    return x[["case:concept:name", "concept:name", "time:timestamp", "out_custody", "ep_row",
              "name", "person_id", degree_col]]


def log_stats(log):
    g = log.groupby("case:concept:name")
    nxt = g["concept:name"].shift(-1)
    pairs = log.assign(nxt=nxt).dropna(subset=["nxt"])
    traces = g["concept:name"].agg(tuple)
    vc = traces.value_counts()
    return pd.Series({
        "состояний": log["concept:name"].nunique(),
        "людей": log["case:concept:name"].nunique(),
        "событий": len(log),
        "людей с 1 заключением": int((g.size() == 1).sum()),
        "переходов": len(pairs),
        "дуг DFG": pairs.groupby(["concept:name", "nxt"]).ngroups,
        "доля петель X→X": round((pairs["concept:name"] == pairs["nxt"]).mean(), 3),
        "вариантов трасс": len(vc),
        "людей с уникальной трассой": int((vc == 1).sum()),
    })


def liberty_pairs(log):
    """Пары соседних заключений и время на свободе: выход из A -> вход в B (дни, < 0 -> 0)."""
    g = log.groupby("case:concept:name")
    p = log.assign(nxt=g["concept:name"].shift(-1), nxt_in=g["time:timestamp"].shift(-1)).dropna(subset=["nxt"])
    p["days"] = ((p["nxt_in"] - p["out_custody"]).dt.total_seconds() / 86400).clip(lower=0)
    return p


def split_people(log, seed=SEED, test_share=0.2):
    rng = np.random.default_rng(seed)
    ids = log["case:concept:name"].unique()
    test_ids = set(rng.choice(ids, size=int(test_share * len(ids)), replace=False))
    return log[~log["case:concept:name"].isin(test_ids)], log[log["case:concept:name"].isin(test_ids)]


def sample_cases(log, n):
    keep = sorted(set(log["case:concept:name"]))[:n]
    return log[log["case:concept:name"].isin(keep)]


def evaluate(models, train_s, test_s, align_cases=0):
    """fitness/precision (token-based replay); align_cases > 0 — ещё precision по выравниваниям на test."""
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
                except Exception as ex:
                    row["precision_align"] = float("nan")
                    print(name, "alignments:", type(ex).__name__)
            row["sec"] = round(time.time() - t0)
            rows.append(row)
    return pd.DataFrame(rows)
