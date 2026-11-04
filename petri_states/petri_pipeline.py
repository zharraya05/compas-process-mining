"""Сети Петри по истории человека в COMPAS DB (ProPublica).

Три этапа:
  1. build_history_table  — таблица 1: одна строка = одна запись базы
     (арест, обвинение, заключение в jail/prison), столбцы по источникам,
     неприменимые ячейки = заглушка STUB.
  2. build_event_log      — таблица 2: журнал событий для сети Петри
     (arrest / charge / jail / prison / free / end), порядок по времени,
     приоритеты при одинаковом времени, около-нулевые интервалы между ними.
  3. discover_models + evaluate_models — сети Петри (Inductive / Heuristics
     Miner + «цветочная» базовая линия) и базовые метрики PM4Py:
     token-based fitness, alignment fitness, ETC precision.

Модуль используется ноутбуком petri_nets.ipynb; его можно запустить и целиком:
    python petri_pipeline.py
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Константы
# ----------------------------------------------------------------------------
STUB = "n/a_src"                         # ячейка не относится к источнику строки
DB_OPEN_STUB = pd.Timestamp("2020-01-01")  # заглушка базы в out_custody: «ещё сидит»
DATA_CAP = pd.Timestamp("2016-12-31")    # позже этой даты реальных данных быть не может (в casearrest есть 2049 год)
EPS = pd.Timedelta(milliseconds=1)       # «почти нулевое» время между событиями с одинаковым временем

# приоритет при одинаковом времени (меньше = раньше)
PRIORITY = {"free": 0, "arrest": 1, "charge": 2, "jail": 3, "prison": 4, "end": 9}

SOURCES = {"jail": "jailhistory", "prison": "prisonhistory", "arrest": "casearrest", "charge": "charge"}

# столбцы таблицы 1: общие + по источникам
HIST_COMMON = ["person_id", "event", "source", "source_id"]
HIST_BY_SOURCE = {
    "jailhistory": ["in_custody", "out_custody"],
    "prisonhistory": ["in_custody", "out_custody"],
    "casearrest": ["case_number", "arrest_id", "arrest_date", "charge_degree"],
    "charge": ["case_number", "offense_date", "charge_number", "charge_degree", "charge", "statute",
               "charge_category"],
}
HIST_COLS = HIST_COMMON + list(dict.fromkeys(c for cols in HIST_BY_SOURCE.values() for c in cols))
DATE_COLS = ["in_custody", "out_custody", "arrest_date", "offense_date"]

SAME_DAY_RULES = ("case_based", "strict")


# ----------------------------------------------------------------------------
# Вспомогательное
# ----------------------------------------------------------------------------
def find_db(candidates) -> Path:
    """Первый существующий путь из списка."""
    for p in candidates:
        if Path(p).expanduser().exists():
            return Path(p).expanduser().resolve()
    raise FileNotFoundError("compas.db не найдена: " + ", ".join(map(str, candidates)))


def to_dt(s: pd.Series) -> pd.Series:
    """Текстовая дата базы -> datetime (битые значения -> NaT)."""
    return pd.to_datetime(s, format="mixed", errors="coerce")


def as_dt(s: pd.Series) -> pd.Series:
    """Столбец таблицы 1 (даты + заглушки) -> datetime; заглушка и пусто -> NaT."""
    return to_dt(s.where(s != STUB))


def import_categorizer(extra_dirs=()):
    """categorize_fine из crime_categories.py (15 типов, как в этапах 0–1)."""
    for d in extra_dirs:
        d = str(Path(d).resolve())
        if d not in sys.path:
            sys.path.insert(0, d)
    from crime_categories import categorize_fine  # noqa: E402
    return categorize_fine


def setup_graphviz(extra=(r"D:\Graphviz\bin",)):
    """Добавить Graphviz в PATH (на Windows он часто не в PATH). True, если dot найден."""
    if shutil.which("dot") is None:
        for p in extra:
            if Path(p).exists():
                os.environ["PATH"] = str(p) + os.pathsep + os.environ["PATH"]
    return shutil.which("dot") is not None


# ----------------------------------------------------------------------------
# Этап 1. Таблица «человек — история — источник»
# ----------------------------------------------------------------------------
def load_tables(db_path) -> dict:
    """Четыре таблицы базы (только чтение, файл не меняется)."""
    con = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        t = {
            "jailhistory": pd.read_sql("select * from jailhistory", con),
            "prisonhistory": pd.read_sql("select * from prisonhistory", con),
            "casearrest": pd.read_sql("select * from casearrest", con),
            "charge": pd.read_sql("select * from charge", con),
        }
    finally:
        con.close()
    return t


def drop_full_duplicates(tables: dict):
    """Удаляет строки, у которых совпадают ВСЕ поля, кроме первичного ключа id.
    Если отличается хоть одно поле (например, другой arrest_id или charge_degree),
    строка остаётся отдельной записью."""
    out, rep = {}, []
    for name, df in tables.items():
        cols = [c for c in df.columns if c != "id"]
        dup = df.duplicated(cols, keep="first")
        out[name] = df[~dup].reset_index(drop=True)
        rep.append(dict(table=name, rows=len(df), full_duplicates=int(dup.sum()), kept=int((~dup).sum())))
    return out, pd.DataFrame(rep)


def drop_invalid_stays(tables: dict):
    """Удаляет пребывания jail/prison, у которых вход позже выхода (in_custody > out_custody).
    Заглушку 2020-01-01 («ещё сидит») не трогает: у неё вход всегда раньше."""
    out, rep, removed = dict(tables), [], []
    for name in ("jailhistory", "prisonhistory"):
        df = tables[name]
        bad = to_dt(df["in_custody"]) > to_dt(df["out_custody"])
        out[name] = df[~bad].reset_index(drop=True)
        removed.append(df[bad].assign(table=name))
        rep.append(dict(table=name, rows=len(df), in_after_out=int(bad.sum()), kept=int((~bad).sum())))
    return out, pd.DataFrame(rep), pd.concat(removed, ignore_index=True)


def build_history_table(tables: dict, categorize=None) -> pd.DataFrame:
    """Таблица 1. Одна строка = одна запись базы.

    person_id, event, source, source_id заполнены у всех строк;
    остальные столбцы — только у своего источника, у чужих = STUB.
    Пустые значения самой базы (NULL) остаются пустыми — это не заглушка.
    """
    parts = []
    j = tables["jailhistory"]
    parts.append(pd.DataFrame({"person_id": j["person_id"], "event": "jail", "source": "jailhistory",
                               "source_id": j["id"], "in_custody": to_dt(j["in_custody"]),
                               "out_custody": to_dt(j["out_custody"])}))
    p = tables["prisonhistory"]
    parts.append(pd.DataFrame({"person_id": p["person_id"], "event": "prison", "source": "prisonhistory",
                               "source_id": p["id"], "in_custody": to_dt(p["in_custody"]),
                               "out_custody": to_dt(p["out_custody"])}))
    a = tables["casearrest"]
    parts.append(pd.DataFrame({"person_id": a["person_id"], "event": "arrest", "source": "casearrest",
                               "source_id": a["id"], "case_number": a["case_number"],
                               "arrest_id": a["arrest_id"], "arrest_date": to_dt(a["arrest_date"]),
                               "charge_degree": a["charge_degree"]}))
    c = tables["charge"]
    cat = c["charge"].map(categorize) if categorize is not None else pd.Series(np.nan, index=c.index)
    parts.append(pd.DataFrame({"person_id": c["person_id"], "event": "charge", "source": "charge",
                               "source_id": c["id"], "case_number": c["case_number"],
                               "offense_date": to_dt(c["offense_date"]), "charge_number": c["charge_number"],
                               "charge_degree": c["charge_degree"], "charge": c["charge"],
                               "statute": c["statute"], "charge_category": cat}))

    filled = []
    for part in parts:
        src = part["source"].iat[0]
        own = set(HIST_BY_SOURCE[src])
        part = part.reindex(columns=HIST_COLS).astype(object)
        for col in HIST_COLS[len(HIST_COMMON):]:
            if col not in own:
                part[col] = STUB
        filled.append(part)
    hist = pd.concat(filled, ignore_index=True)

    # порядок строк: человек -> время события -> приоритет (только для удобства чтения)
    t = event_time(hist)
    hist = (hist.assign(_t=t, _p=hist["event"].map(PRIORITY))
                .sort_values(["person_id", "_t", "_p", "source_id"], na_position="last", kind="mergesort")
                .drop(columns=["_t", "_p"]).reset_index(drop=True))
    hist["person_id"] = hist["person_id"].astype(int)
    hist["source_id"] = hist["source_id"].astype(int)
    return hist


def event_time(hist: pd.DataFrame) -> pd.Series:
    """Время события строки таблицы 1: вход в заключение / дата ареста / дата обвинения."""
    t = pd.Series(pd.NaT, index=hist.index, dtype="datetime64[ns]")
    for ev, col in [("jail", "in_custody"), ("prison", "in_custody"),
                    ("arrest", "arrest_date"), ("charge", "offense_date")]:
        m = hist["event"] == ev
        t[m] = as_dt(hist.loc[m, col]).astype("datetime64[ns]")
    return t


# ----------------------------------------------------------------------------
# Этап 2. Журнал для сети Петри
# ----------------------------------------------------------------------------
def observation_end(hist: pd.DataFrame) -> pd.Timestamp:
    """Конец наблюдения = последняя реальная дата в базе (без заглушки 2020-01-01 и битых дат)."""
    vals = [event_time(hist)]
    stays = hist["event"].isin(["jail", "prison"])
    vals.append(as_dt(hist.loc[stays, "out_custody"]))
    s = pd.concat(vals)
    return s[(s < DB_OPEN_STUB) & (s <= DATA_CAP)].max()


def custody_episodes(hist: pd.DataFrame, obs_end, transfer_gap=pd.Timedelta(days=1)):
    """Склеивает пребывания jail/prison в эпизоды непрерывного заключения.

    Пребывания склеиваются, если пересекаются (вход следующего не позже выхода текущего)
    или если следующее — prison, начатый не позже transfer_gap после выхода (перевод jail -> prison).
    Возвращает (stays с episode_id, episodes с концом эпизода и признаком «ещё сидит»).
    """
    s = hist[hist["event"].isin(["jail", "prison"])][["person_id", "event", "source", "source_id",
                                                      "in_custody", "out_custody"]].copy()
    s["in_t"] = as_dt(s["in_custody"])
    s["out_t"] = as_dt(s["out_custody"])
    s["open"] = s["out_t"].isna() | (s["out_t"] >= DB_OPEN_STUB) | (s["out_t"] > obs_end)
    s.loc[s["out_t"] < s["in_t"], "out_t"] = s["in_t"]            # выход раньше входа -> выход = вход
    s = s[s["in_t"].notna() & (s["in_t"] <= obs_end)].sort_values(["person_id", "in_t", "source_id"])

    ep_ids, ep = [], []
    cur_pid, cur_end, cur_open, eid = None, None, False, -1
    last = None
    for r in s.itertuples(index=False):
        end_r = pd.Timestamp.max if r.open else r.out_t
        new = (r.person_id != cur_pid or not (r.in_t <= cur_end or
               (r.event == "prison" and r.in_t - cur_end <= transfer_gap)))
        if new:
            if last is not None:
                ep.append(last)
            eid += 1
            cur_pid, cur_end = r.person_id, end_r
            last = dict(episode_id=eid, person_id=r.person_id, ep_start=r.in_t, ep_end=end_r,
                        end_source=r.source, end_source_id=r.source_id, n_stays=1)
        else:
            last["n_stays"] += 1
            if end_r > cur_end:
                cur_end = end_r
                last.update(ep_end=end_r, end_source=r.source, end_source_id=r.source_id)
        ep_ids.append(eid)
    if last is not None:
        ep.append(last)
    s["episode_id"] = ep_ids
    ep = pd.DataFrame(ep)
    ep["still_in"] = ep["ep_end"] == pd.Timestamp.max
    ep.loc[ep["still_in"], "ep_end"] = pd.NaT
    return s, ep


def build_event_log(hist: pd.DataFrame, obs_end=None, same_day_rule="case_based",
                    log_start=None, transfer_gap=pd.Timedelta(days=1), eps=EPS):
    """Таблица 2: журнал событий для сети Петри.

    Активности: arrest, charge, jail, prison (по одной на каждую запись базы),
    free (выход из эпизода заключения), end (конец наблюдения, у всех людей).

    same_day_rule — порядок в день выхода на свободу (подробно — apply_same_day_rule):
      "case_based" — по номеру дела (правила A1–A3, согласованы 03.11.2026);
      "strict"     — строго по времени (arrest 00:00 окажется раньше выхода в 02:27).
    """
    if same_day_rule not in SAME_DAY_RULES:
        raise ValueError(f"same_day_rule должен быть одним из {SAME_DAY_RULES}")
    obs_end = observation_end(hist) if obs_end is None else pd.Timestamp(obs_end)
    report = {"obs_end": obs_end}

    # --- события из записей базы
    ev = hist[["person_id", "event", "source", "source_id", "case_number"]].copy()
    ev["case_number"] = ev["case_number"].where(ev["case_number"] != STUB)
    ev["ts_raw"] = event_time(hist)
    # вторичный ключ порядка внутри одинакового времени: номер ареста / обвинения
    sub = pd.Series(np.nan, index=hist.index)
    m = hist["event"] == "arrest"
    sub[m] = pd.to_numeric(hist.loc[m, "arrest_id"], errors="coerce")
    m = hist["event"] == "charge"
    sub[m] = pd.to_numeric(hist.loc[m, "charge_number"], errors="coerce")
    ev["sub_key"] = sub
    bad = ev["ts_raw"].isna() | (ev["ts_raw"] > obs_end)
    report["dropped_bad_or_after_end"] = ev[bad].groupby("event").size().to_dict()
    ev = ev[~bad].rename(columns={"event": "activity"})

    # --- эпизоды заключения и события free
    stays, eps_df = custody_episodes(hist, obs_end, transfer_gap)
    ev = ev.merge(stays[["source", "source_id", "episode_id"]], on=["source", "source_id"], how="left")
    fr = eps_df[~eps_df["still_in"]]
    free = pd.DataFrame({"person_id": fr["person_id"].values, "activity": "free",
                         "source": fr["end_source"].values, "source_id": fr["end_source_id"].values,
                         "case_number": np.nan, "ts_raw": fr["ep_end"].values, "sub_key": np.nan,
                         "episode_id": fr["episode_id"].values})
    report["episodes"] = len(eps_df)
    report["episodes_still_in_at_end"] = int(eps_df["still_in"].sum())
    report["stays_merged"] = int(eps_df["n_stays"].sum() - len(eps_df))

    # --- конец наблюдения
    people = np.sort(hist["person_id"].unique())
    end = pd.DataFrame({"person_id": people, "activity": "end", "source": "observation_end",
                        "source_id": -1, "case_number": np.nan, "ts_raw": obs_end, "sub_key": np.nan,
                        "episode_id": np.nan})
    log = pd.concat([ev, free, end], ignore_index=True)
    log["ts_raw"] = pd.to_datetime(log["ts_raw"])
    log["priority"] = log["activity"].map(PRIORITY)

    # --- правило одного дня (выход на свободу и аресты/обвинения с датой того же дня)
    log["ts_sort"] = log["ts_raw"]
    log["shifted_same_day"] = False
    log["same_day_note"] = ""
    if same_day_rule == "case_based":
        log, rep_sd = apply_same_day_rule(log, eps_df)
        report.update(rep_sd)
    report["free_shifted_same_day"] = int(log["shifted_same_day"].sum())

    # free не может стоять раньше входа в своё же заключение (пребывания нулевой длины: выход = вход)
    last_in = stays.groupby("episode_id")["in_t"].max()
    fmask = log["activity"] == "free"
    li = log.loc[fmask, "episode_id"].map(last_in)
    late = li.notna() & (log.loc[fmask, "ts_sort"] <= li)
    idx = li[late].index
    log.loc[idx, "ts_sort"] = li[late]
    log.loc[idx, "priority"] = max(PRIORITY.values()) - 1          # после jail/prison того же момента, до end
    report["free_zero_length_fixed"] = int(late.sum())

    # --- журнал с началом окна (по желанию)
    if log_start is not None:
        n0 = len(log)
        log = log[log["ts_sort"] >= pd.Timestamp(log_start)]
        report["dropped_before_log_start"] = n0 - len(log)
        # люди без событий в окне: трасса была бы из одного end -> убираем
        n_ev = log.groupby("person_id")["activity"].transform("size")
        only_end = n_ev.eq(1) & log["activity"].eq("end")
        report["people_without_events_in_window"] = int(only_end.sum())
        log = log[~only_end]
        # кто на начало окна уже сидел: первое событие трассы — free
        first = log.sort_values(["person_id", "ts_sort", "priority"]).groupby("person_id")["activity"].first()
        report["traces_starting_with_free"] = int((first == "free").sum())

    # --- порядок и почти нулевые интервалы
    log = log.sort_values(["person_id", "ts_sort", "priority", "ts_raw", "case_number", "sub_key", "source_id"],
                          na_position="last", kind="mergesort").reset_index(drop=True)
    pid = log["person_id"].to_numpy()
    t = log["ts_sort"].to_numpy("datetime64[ns]").astype("int64")
    out = t.copy()
    e = int(eps.value)
    for i in range(1, len(t)):
        if pid[i] == pid[i - 1] and out[i] <= out[i - 1]:
            out[i] = out[i - 1] + e
    log["ts"] = pd.to_datetime(out)
    log["tie_with_prev"] = (log["person_id"].eq(log["person_id"].shift())
                            & log["ts_sort"].eq(log["ts_sort"].shift()))
    log["event_idx"] = log.groupby("person_id").cumcount() + 1
    nxt = log.groupby("person_id")["ts"].shift(-1)
    log["next_activity"] = log.groupby("person_id")["activity"].shift(-1)
    log["dt_next_days"] = (nxt - log["ts"]).dt.total_seconds() / 86400
    log = log[["person_id", "event_idx", "activity", "ts", "ts_raw", "tie_with_prev", "shifted_same_day",
               "same_day_note", "next_activity", "dt_next_days", "source", "source_id", "case_number", "episode_id"]]
    report["events"] = len(log)
    report["people"] = log["person_id"].nunique()
    report["tied_events"] = int(log["tie_with_prev"].sum())
    return log, eps_df, report


def apply_same_day_rule(log: pd.DataFrame, eps_df: pd.DataFrame):
    """Правило дня выхода на свободу (case_based).

    Рассматривается день выхода D эпизода, начавшегося в более ранний день, если выход не в 00:00
    и на день D есть аресты/обвинения (у них известна только дата, поэтому по времени они 00:00,
    то есть раньше выхода). Дела эпизода — номера дел арестов с датой от (дня входа − 1)
    до дня выхода (не включая); по обвинениям дело эпизода не определяется (offense_date бывает раньше).

    * Есть дело НЕ из эпизода (номер дела сменился; A1 с новым заключением, A2 без него):
      free переносится на 00:00 дня D -> free -> arrest -> charge (-> jail).
      Записи дел эпизода того же дня ставятся перед free (они относятся к заключению).
    * Только дела эпизода и в тот же день после выхода начинается новое заключение (A1, новые
      обвинения в рамках одного дела): free не ставится, заключения идут подряд.
    * Только дела эпизода и нового заключения нет (A3, оформление внутри срока):
      аресты переносятся прямо перед входом в это заключение (почти нулевой переход arrest -> jail),
      обвинения остаются на своём месте внутри срока.
    Равное время (выход из prison в 00:00 и арест в 00:00, A4) решает приоритет: free -> arrest.
    """
    rep = {}
    E = eps_df[~eps_df["still_in"]].copy()
    E["day"] = E["ep_end"].dt.normalize()
    E = E[(E["ep_start"].dt.normalize() < E["day"]) & (E["ep_end"] > E["day"])]
    ac = log[log["activity"].isin(["arrest", "charge"]) & log["case_number"].notna()]
    ac = ac.assign(day=ac["ts_raw"].dt.normalize(), row=ac.index)

    # события дня выхода
    dd = E[["episode_id", "person_id", "day", "ep_start", "ep_end"]].merge(
        ac[["person_id", "day", "case_number", "activity", "row"]], on=["person_id", "day"])
    if dd.empty:
        return log, {"same_day_episodes": 0}
    # дела эпизода
    arr = ac[ac["activity"] == "arrest"]
    w = E[["episode_id", "person_id", "ep_start", "day"]].merge(
        arr[["person_id", "ts_raw", "case_number"]], on="person_id")
    w = w[(w["ts_raw"] >= w["ep_start"].dt.normalize() - pd.Timedelta(days=1)) & (w["ts_raw"] < w["day"])]
    ep_cases = set(zip(w["episode_id"], w["case_number"]))
    dd["same_case"] = [k in ep_cases for k in zip(dd["episode_id"], dd["case_number"])]
    # новое заключение в тот же день после выхода
    cu = log[log["activity"].isin(["jail", "prison"])]
    cu = cu.assign(day=cu["ts_raw"].dt.normalize())[["person_id", "day", "ts_raw"]]
    nc = E[["episode_id", "person_id", "day", "ep_end"]].merge(cu, on=["person_id", "day"])
    new_custody = set(nc.loc[nc["ts_raw"] > nc["ep_end"], "episode_id"])

    g = dd.groupby("episode_id").agg(other=("same_case", lambda s: (~s).any()), ep_start=("ep_start", "first"),
                                     day=("day", "first"))
    g["new_custody"] = g.index.isin(new_custody)
    free_rows = log.index[log["activity"] == "free"]
    free_by_ep = pd.Series(free_rows, index=log.loc[free_rows, "episode_id"].values)

    # 1) номер дела сменился: free -> 00:00, записи дел эпизода — перед free
    ep_other = g.index[g["other"]]
    fr = free_by_ep.reindex(ep_other).dropna().astype(int)
    log.loc[fr.values, "ts_sort"] = log.loc[fr.values, "ts_raw"].dt.normalize()
    log.loc[fr.values, "shifted_same_day"] = True
    log.loc[fr.values, "same_day_note"] = np.where(g.loc[fr.index, "new_custody"], "A1_free_before_new_case",
                                                   "A2_free_before_new_case")
    same_rows = dd.loc[dd["episode_id"].isin(ep_other) & dd["same_case"], "row"]
    log.loc[same_rows, "priority"] = -1
    log.loc[same_rows, "same_day_note"] = "same_case_before_free"
    rep["A1_A2_free_before_new_case"] = len(fr)

    # 2) только дела эпизода + новое заключение в тот же день: free не ставится
    ep_merge = g.index[~g["other"] & g["new_custody"]]
    drop = free_by_ep.reindex(ep_merge).dropna().astype(int)
    log = log.drop(index=drop.values)
    rep["A1_free_removed_same_case"] = len(drop)

    # 3) только дела эпизода, нового заключения нет: аресты -> прямо перед входом
    ep_move = g.index[~g["other"] & ~g["new_custody"]]
    mv = dd[dd["episode_id"].isin(ep_move) & (dd["activity"] == "arrest")]
    log.loc[mv["row"].values, "ts_sort"] = mv["ep_start"].values
    log.loc[mv["row"].values, "same_day_note"] = "A3_arrest_moved_before_entry"
    rep["A3_arrests_moved_before_entry"] = len(mv)
    rep["A3_episodes"] = int(dd[dd["episode_id"].isin(ep_move)]["episode_id"].nunique())
    rep["same_day_episodes"] = len(g)
    return log, rep


def check_event_log(log: pd.DataFrame):
    """Проверки журнала. Возвращает список строк-результатов; при ошибке — AssertionError."""
    res = []
    g = log.groupby("person_id")
    assert (g["ts"].diff().dropna() > pd.Timedelta(0)).all(), "время внутри трассы не строго возрастает"
    res.append("время внутри каждой трассы строго возрастает")
    last = g["activity"].last()
    assert (last == "end").all(), "не у всех трасс последнее событие end"
    assert (g["activity"].apply(lambda s: (s == "end").sum()) == 1).all()
    res.append("у каждой трассы ровно одно end, и оно последнее")
    lag = (log["ts"] - log["ts_raw"].where(~log["shifted_same_day"], log["ts_raw"].dt.normalize()))
    res.append(f"максимальный сдвиг из-за почти нулевых интервалов: {lag.max()}")
    ff = log[log["activity"] == "free"]
    prev = g["activity"].shift(1)[ff.index]
    res.append(f"free после free подряд: {int((prev == 'free').sum())}")
    return res


# ----------------------------------------------------------------------------
# Описание журнала
# ----------------------------------------------------------------------------
def directly_follows(log: pd.DataFrame):
    """Матрица directly-follows: число пар, вероятность, медиана времени (дни)."""
    d = log[log["next_activity"].notna()]
    cnt = pd.crosstab(d["activity"], d["next_activity"])
    prob = cnt.div(cnt.sum(axis=1), axis=0)
    med = d.pivot_table(index="activity", columns="next_activity", values="dt_next_days", aggfunc="median")
    known = list(PRIORITY) + REFINED_ORDER
    labels = set(cnt.index) | set(cnt.columns)
    order = list(dict.fromkeys([a for a in known if a in labels] + sorted(labels - set(known))))
    rows = [a for a in order if a in cnt.index]
    cols = [a for a in order if a in cnt.columns]
    return cnt.reindex(index=rows, columns=cols, fill_value=0), prob.reindex(index=rows, columns=cols).fillna(0), \
        med.reindex(index=rows, columns=cols)


def collapse_repeats(seq):
    """arrest, arrest, charge -> arrest+, charge (для обзора вариантов)."""
    out = []
    for a in seq:
        if out and out[-1].rstrip("+") == a:
            out[-1] = a + "+"
        else:
            out.append(a)
    return out


# ----------------------------------------------------------------------------
# Уточнение активностей по контексту (label refinement)
# ----------------------------------------------------------------------------
REFINED_ORDER = ["free_jail", "free_prison", "arrest_free", "arrest_in", "charge_pre", "charge_free",
                 "charge_in", "jail", "prison", "end"]


def refine_labels(log: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """Делит активности по контексту; события и порядок не меняются, меняется только имя.

    Статус «в заключении» идёт по трассе: после jail/prison — внутри, после free — на свободе.
    Если первое из событий jail/prison/free в трассе — free (человек сидел на начало окна),
    трасса начинается «внутри».
      arrest -> arrest_in (внутри срока) / arrest_free (на свободе);
      charge -> charge_in (внутри срока) /
                charge_pre (на свободе, раньше первого ареста по своему делу) /
                charge_free (на свободе, после ареста по делу или дело без ареста);
      free   -> free_jail / free_prison (по источнику последнего пребывания эпизода).
    Исходная активность сохраняется в activity_base.
    """
    lg = log.copy()
    lg["activity_base"] = lg["activity"]
    a = lg["activity"]
    mark = pd.Series(np.nan, index=lg.index)
    mark[a.isin(["jail", "prison"])] = 1.0
    mark[a.eq("free")] = 0.0
    first = mark.groupby(lg["person_id"]).transform(lambda s: s.dropna().iloc[0] if s.notna().any() else 0.0)
    start_in = (first == 0.0).astype(float)                      # первое из jail/prison/free — free
    # статус ДО события: состояние после предыдущих событий
    prev = mark.groupby(lg["person_id"]).shift(1)
    prev = prev.groupby(lg["person_id"]).ffill()
    inside = prev.fillna(start_in).astype(bool)

    # charge_pre: обвинение раньше ПЕРВОГО ареста по своему делу (вся история базы)
    ar = hist[hist["event"] == "arrest"]
    first_arr = (pd.DataFrame({"person_id": ar["person_id"], "case_number": ar["case_number"],
                               "t": as_dt(ar["arrest_date"])})
                 .dropna().query("t <= @DATA_CAP").groupby(["person_id", "case_number"])["t"].min()
                 .rename("first_arrest").reset_index())
    ch = lg[a.eq("charge")][["person_id", "case_number", "ts_raw"]].reset_index()
    ch = ch.merge(first_arr, on=["person_id", "case_number"], how="left").set_index("index")
    later = ch["first_arrest"] > ch["ts_raw"]

    new = lg["activity"].copy()
    m = a.eq("arrest")
    new[m] = np.where(inside[m], "arrest_in", "arrest_free")
    m = a.eq("charge")
    lat = later.reindex(lg.index[m]).fillna(False).to_numpy()
    new[m] = np.where(inside[m], "charge_in", np.where(lat, "charge_pre", "charge_free"))
    m = a.eq("free")
    new[m] = np.where(lg.loc[m, "source"].eq("prisonhistory"), "free_prison", "free_jail")
    lg["activity"] = new
    nxt = lg.groupby("person_id")["activity"].shift(-1)
    lg["next_activity"] = nxt
    return lg


# ----------------------------------------------------------------------------
# Этап 3. Сети Петри и базовые метрики PM4Py
# ----------------------------------------------------------------------------
def split_people(log: pd.DataFrame, test_share=0.2, seed=42):
    """Разбиение train/test по людям (а не по событиям)."""
    people = np.sort(log["person_id"].unique())
    rng = np.random.default_rng(seed)
    test = set(rng.choice(people, size=int(round(len(people) * test_share)), replace=False))
    is_test = log["person_id"].isin(test)
    return log[~is_test].copy(), log[is_test].copy()


def to_pm4py(log: pd.DataFrame):
    import pm4py
    df = log[["person_id", "activity", "ts"]].copy()
    df["person_id"] = df["person_id"].astype(str)
    return pm4py.format_dataframe(df, case_id="person_id", activity_key="activity", timestamp_key="ts")


def flower_net(activities):
    """«Цветочная» сеть: разрешает любые последовательности активностей (precision-база)."""
    from pm4py.objects.petri_net.obj import PetriNet, Marking
    from pm4py.objects.petri_net.utils import petri_utils as pu
    net = PetriNet("flower")
    src, hub, sink = PetriNet.Place("source"), PetriNet.Place("hub"), PetriNet.Place("sink")
    for p in (src, hub, sink):
        net.places.add(p)
    t0, t1 = PetriNet.Transition("tau_start", None), PetriNet.Transition("tau_end", None)
    net.transitions.update([t0, t1])
    pu.add_arc_from_to(src, t0, net)
    pu.add_arc_from_to(t0, hub, net)
    pu.add_arc_from_to(hub, t1, net)
    pu.add_arc_from_to(t1, sink, net)
    for a in sorted(activities):
        t = PetriNet.Transition("t_" + a, a)
        net.transitions.add(t)
        pu.add_arc_from_to(hub, t, net)
        pu.add_arc_from_to(t, hub, net)
    return net, Marking({src: 1}), Marking({sink: 1})


def discover_models(train_pm, im_noise=(0.0, 0.2), hm_dep=(0.9,)):
    import pm4py
    models, info = {}, []
    for nz in im_noise:
        t0 = time.time()
        models[f"IM_noise{nz}"] = pm4py.discover_petri_net_inductive(train_pm, noise_threshold=nz)
        info.append((f"IM_noise{nz}", time.time() - t0))
    for dp in hm_dep:
        t0 = time.time()
        models[f"HM_dep{dp}"] = pm4py.discover_petri_net_heuristics(train_pm, dependency_threshold=dp)
        info.append((f"HM_dep{dp}", time.time() - t0))
    models["flower"] = flower_net(train_pm["concept:name"].unique())
    return models, pd.DataFrame(info, columns=["model", "discover_sec"])


def dfg_petri_net(train_pm, min_prob=0.0):
    """Сеть из графа directly-follows (стандартное преобразование PM4Py: одна активность — одно место).
    min_prob — оставить переход a -> b, если P(b | a) >= min_prob (0 = все переходы train).
    Это сеть «первого порядка»: разрешает ровно наблюдавшиеся пары соседних активностей,
    поэтому её precision — ориентир-предел для сетей, которые помнят только последнюю активность."""
    import pm4py
    dfg, sa, ea = pm4py.discover_dfg(train_pm)
    if min_prob > 0:
        out = {}
        for (a, b), n in dfg.items():
            out[a] = out.get(a, 0) + n
        dfg = {k: n for k, n in dfg.items() if n / out[k[0]] >= min_prob}
    return pm4py.convert_to_petri_net(dfg, sa, ea)


def discover_extra_models(train_pm, ilp_alpha=(1.0,), split_miner=True, powl=True, dfg_min_prob=(0.0, 0.05)):
    """Дополнительные майнеры: ILP, Split Miner (BPMN -> сеть), POWL (-> сеть), сеть из DFG.
    Если майнера нет в установленной версии PM4Py, он пропускается (причина — в таблице info)."""
    import pm4py
    models, info = {}, []

    def run(name, fn):
        t0 = time.time()
        try:
            models[name] = fn()
            info.append((name, time.time() - t0, "ok"))
        except Exception as e:                       # noqa: BLE001 — показать причину, не падать
            info.append((name, time.time() - t0, f"пропущен: {type(e).__name__}: {str(e)[:120]}"))

    for a in ilp_alpha:
        run(f"ILP_alpha{a}", lambda a=a: pm4py.discover_petri_net_ilp(train_pm, alpha=a))
    if split_miner:
        if hasattr(pm4py, "discover_bpmn_split_miner"):
            run("SplitMiner", lambda: pm4py.convert_to_petri_net(pm4py.discover_bpmn_split_miner(train_pm)))
        else:
            info.append(("SplitMiner", 0.0, f"пропущен: нет в pm4py {pm4py.__version__}"))
    if powl:
        if hasattr(pm4py, "discover_powl"):
            run("POWL", lambda: pm4py.convert_to_petri_net(pm4py.discover_powl(train_pm)))
        else:
            info.append(("POWL", 0.0, f"пропущен: нет в pm4py {pm4py.__version__}"))
    for mp in dfg_min_prob:
        run(f"DFG_p{mp}", lambda mp=mp: dfg_petri_net(train_pm, mp))
    return models, pd.DataFrame(info, columns=["model", "discover_sec", "status"])


def alignment_fitness(log_pm, net, im, fm, max_sec_trace=5):
    """Alignment fitness стандартными функциями PM4Py, но с лимитом времени на вариант трассы.
    Без лимита A* на длинных трассах (сотни событий) и сетях с тихими переходами считается часами.
    Варианты, не уложившиеся в лимит, в метрику не входят; их число возвращается (n_timeout)."""
    import pm4py
    from pm4py.algo.conformance.alignments.petri_net import algorithm as ali
    from pm4py.algo.evaluation.replay_fitness.variants import alignment_based as ab
    el = pm4py.convert_to_event_log(log_pm)
    res = ali.apply(el, net, im, fm, parameters={ali.Parameters.PARAM_MAX_ALIGN_TIME_TRACE: max_sec_trace})
    ev = ab.evaluate(res)
    ev["n_timeout"] = sum(r is None for r in res)
    ev["n_traces"] = len(res)
    return ev


def evaluate_models(models: dict, logs: dict, align_sample=200, seed=42, precision_splits=("test",),
                    align_splits=("test",), align_models=None, align_max_sec_trace=3, verbose=True):
    """Базовые метрики PM4Py для каждой модели и каждого журнала.

    fitness_tbr    — token-based replay (pm4py.fitness_token_based_replay), весь журнал;
    fitness_align  — alignments (PM4Py, alignment-based fitness) на случайной выборке align_sample трасс
                     с лимитом align_max_sec_trace секунд на вариант (align_timeout — сколько не уложились);
    precision_etc  — ETConformance на token replay (pm4py.precision_token_based_replay);
    precision_norm — (P − P_flower) / (1 − P_flower): насколько модель точнее «цветочной»;
    f1             — гармоническое среднее fitness_tbr и precision_etc.
    precision_splits / align_splits — на каких журналах считать precision и alignments
    (на train они в 4 раза дольше; по умолчанию только test). TBR считается везде.
    """
    import pm4py
    rows = []
    rng = np.random.default_rng(seed)
    samples = {}
    for split, lg in logs.items():
        cases = lg["case:concept:name"].unique()
        if align_sample and len(cases) > align_sample:
            cases = rng.choice(cases, size=align_sample, replace=False)
        samples[split] = lg[lg["case:concept:name"].isin(set(cases))]
    for name, (net, im, fm) in models.items():
        for split, lg in logs.items():
            r = dict(model=name, split=split, traces=lg["case:concept:name"].nunique(),
                     places=len(net.places), transitions=len(net.transitions),
                     silent=sum(t.label is None for t in net.transitions), arcs=len(net.arcs))
            t0 = time.time()
            tbr = pm4py.fitness_token_based_replay(lg, net, im, fm)
            r["fitness_tbr"] = tbr["log_fitness"]
            r["pct_fit_traces_tbr"] = tbr["percentage_of_fitting_traces"]
            r["sec_tbr"] = time.time() - t0
            # цветочная сеть: alignment fitness = 1 по построению (PM4Py её не выравнивает — не easy sound)
            if (align_sample and split in align_splits and name != "flower"
                    and (align_models is None or name in align_models)):
                t0 = time.time()
                al = alignment_fitness(samples[split], net, im, fm, align_max_sec_trace)
                r["fitness_align"] = al["log_fitness"]
                r["pct_fit_traces_align"] = al["percentage_of_fitting_traces"]
                r["align_traces"] = al["n_traces"]
                r["align_timeout"] = al["n_timeout"]
                r["sec_align"] = time.time() - t0
            if split in precision_splits:
                t0 = time.time()
                r["precision_etc"] = pm4py.precision_token_based_replay(lg, net, im, fm)
                r["sec_prec"] = time.time() - t0
            rows.append(r)
            if verbose:
                print(f"{name:12s} {split:5s} fitness_tbr={r['fitness_tbr']:.4f} "
                      f"precision={r.get('precision_etc', float('nan')):.4f} "
                      f"align={r.get('fitness_align', float('nan')):.4f}", flush=True)
    q = pd.DataFrame(rows)
    if "precision_etc" in q:
        q["f1"] = 2 * q["fitness_tbr"] * q["precision_etc"] / (q["fitness_tbr"] + q["precision_etc"])
        if "flower" in models:
            pf = q[q["model"] == "flower"].set_index("split")["precision_etc"].dropna()
            q["precision_flower"] = q["split"].map(pf)
            q["precision_norm"] = (q["precision_etc"] - q["precision_flower"]) / (1 - q["precision_flower"])
    return q


def save_net_png(net, im, fm, path):
    import pm4py
    pm4py.save_vis_petri_net(net, im, fm, str(path))
    return Path(path)


def save_dfg_png(log_pm, path, performance=False):
    import pm4py
    if performance:
        dfg, sa, ea = pm4py.discover_performance_dfg(log_pm)
        pm4py.save_vis_performance_dfg(dfg, sa, ea, str(path))
    else:
        dfg, sa, ea = pm4py.discover_dfg(log_pm)
        pm4py.save_vis_dfg(dfg, sa, ea, str(path))
    return Path(path)


# ----------------------------------------------------------------------------
# Запуск целиком
# ----------------------------------------------------------------------------
def main(db=None, out="out", same_day_rule="case_based", align_sample=200):
    here = Path(__file__).resolve().parent
    db = find_db([db] if db else [here / ".." / "stage0 - data" / "compas.db", r"D:\diploma\compas.db",
                                   here / "compas.db"])
    out = here / out
    out.mkdir(exist_ok=True)
    cat = import_categorizer([here, here / ".." / "stage0 ver2 from charge"])
    tables, dups = drop_full_duplicates(load_tables(db))
    tables, bad_stays, _ = drop_invalid_stays(tables)
    print(bad_stays.to_string(index=False))
    hist = build_history_table(tables, cat)
    hist.to_csv(out / "history_table.csv", index=False)
    log, episodes, rep = build_event_log(hist, same_day_rule=same_day_rule)
    log.to_csv(out / "petri_log.csv", index=False)
    print(dups.to_string(index=False))
    print(rep)
    print("\n".join(check_event_log(log)))
    train, test = split_people(log)
    tr, te = to_pm4py(train), to_pm4py(test)
    models, _ = discover_models(tr)
    q = evaluate_models(models, {"train": tr, "test": te}, align_sample=align_sample)
    q.to_csv(out / "model_quality.csv", index=False)
    print(q.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
