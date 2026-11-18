"""Этап 2 (папка 1711). Эпизоды на свободе и признаки к моменту выхода (расширение RFM на основе графа заключений).

Вход:
  - данные/event_table_stage2.csv  - узлы графа заключений, собранного заново по решениям 17.11 (build_graph.py),
                                     + начальные prison (шли на 01.01.2013, start_state): выход из них - первый эпизод
  - данные/prison_before_2013.csv  - prison, закончившиеся до 2013 (в граф не входят, только история)
  - compas.db: casearrest (аресты, вся история), charge (обвинения: типы, степень, дела без ареста), people
Выход: таблица «эпизод на свободе» x (цели, причина возврата, признаки), см. build_dataset().
Правки 17.11 (номера правок - рабочие заметки 17.11): аресты считаются по делам (8); обвинения - отдельное
семейство по дате правонарушения, включая дела без ареста (1, 5, 8); число арестов за всю жизнь - в семействе F (10);
ступень тяжести и полная причина текущего узла, аресты без заключения, причина прошлого возврата (11, 14);
без балла COMPAS (17).
Все признаки считаются только по информации с датой <= дата выхода (t0).
"""
import sqlite3, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                                  # корень репозитория compas-process-mining
DB = REPO / "stage0-data" / "compas.db"              # та же база, что в build_graph.py (копия ProPublica)
EVENT_TABLE = HERE / "данные" / "event_table_stage2.csv"     # граф заключений 17.11 + начальные prison
PRISON_BEFORE = HERE / "данные" / "prison_before_2013.csv"
sys.path.insert(0, str(HERE / "vendor"))
import crime_categories as cc                        # те же 15 типов, что в графе

WINDOW_START = pd.Timestamp("2013-01-01")   # начало jailhistory = начало графа
STUDY_END = pd.Timestamp("2016-04-01")      # конец наблюдения: март 2016 - последний полный месяц арестов и jail
MERGE_GAP_DAYS = 1                          # заключения с разрывом <= 1 дня - одно пребывание
ENTRY_GUARD_DAYS = 1                        # арест в пределах 1 дня от входа относится к этому же заключению
HALF_LIVES = (180, 365, 1095)               # периоды полураспада для затухающих счётчиков арестов, дней
OLD_REASONS = {"rearrest_old_case", "revocation_old_case"}
H_TYPES = ["Traffic", "Drug", "Property", "Violent", "ResistPolice", "Fraud", "PublicOrder", "Sex", "Weapon"]
REASONS = ["new_offense", "rearrest_old_case", "revocation_old_case", "prison_by_arrest", "transfer", "procedural", "не привязано"]
CAUSE_OF_REASON = {"new_offense": 1, "prison_by_arrest": 1, "rearrest_old_case": 2, "revocation_old_case": 2}   # иначе 3
DEG_LEVELS = {1: "F1", 2: "F2", 3: "F3", 4: "Fx", 5: "M1", 6: "M2", 7: "MO3", 8: "other", 9: "none"}   # deg_rank узла


# ----------------------------------------------------------------------------- загрузка
def load_nodes(path=EVENT_TABLE):
    """Узлы графа этапа 1 -> одна строка на заключение (шаг Justice возвращаем в множество типов узла)."""
    t = pd.read_csv(path, parse_dates=["in_custody", "out_custody", "dob"])
    if "in_custody_true" in t.columns:     # начальное состояние (prison на 01.01.2013): в графе вход = 01.01.2013,
        t["in_custody"] = pd.to_datetime(t["in_custody_true"])   # для признаков нужен настоящий вход
    else:
        t["start_state"] = False
    g = t.sort_values(["ep_row", "step"]).groupby("ep_row")
    e = g.first()[["id", "in_custody", "out_custody", "ongoing", "sources", "reason", "charge_class", "sex", "dob",
                   "decile_score", "start_state", "deg_rank"]].copy()
    e["types"] = g["types"].agg(lambda s: frozenset(x for v in s for x in str(v).split("+")))
    return e.reset_index().rename(columns={"id": "person_id"})


def build_spells(nodes):
    """Пребывание в заключении = узлы одного человека, пересекающиеся или идущие с разрывом <= MERGE_GAP_DAYS."""
    rows = []
    for pid, g in nodes.sort_values(["person_id", "in_custody"]).groupby("person_id", sort=False):
        cur = None
        for r in g.itertuples(index=False):
            if cur is not None and r.in_custody <= cur["cust_out"] + pd.Timedelta(days=MERGE_GAP_DAYS):
                cur["cust_out"] = max(cur["cust_out"], r.out_custody); cur["types"] |= r.types; cur["n_nodes"] += 1
                cur["ongoing"] |= r.ongoing; cur["prison"] |= "prison" in r.sources
                cur["felony"] |= r.charge_class == "Felony"; cur["start"] |= bool(r.start_state)
                cur["deg_rank"] = min(cur["deg_rank"], int(r.deg_rank))
            else:
                if cur: rows.append(cur)
                cur = dict(person_id=pid, cust_in=r.in_custody, cust_out=r.out_custody, types=set(r.types), n_nodes=1,
                           ongoing=bool(r.ongoing), prison="prison" in r.sources, felony=r.charge_class == "Felony",
                           reason=r.reason, sex=r.sex, dob=r.dob, decile_score=r.decile_score, start=bool(r.start_state),
                           deg_rank=int(r.deg_rank))
        rows.append(cur)
    s = pd.DataFrame(rows)
    s["in_date"] = s["cust_in"].dt.normalize(); s["out_date"] = s["cust_out"].dt.normalize()
    s["k"] = s.groupby("person_id").cumcount() + 1
    return s


def load_arrests(db=DB):
    """Строка = день ареста человека (единица для целей: «первый арест после выхода»). В один день бывает несколько
    арестов по разным делам (21% дней), они НЕ теряются: n_cases - число дел (арестов) за день, n_old - из них
    повторных по старому делу, n_felony - felony, case_types - типы каждого дела. Признаки-счётчики считают аресты
    по делам (человек + дата + дело), интервалы между арестами - по дням.
    Типы ареста = типы обвинений его дел (charge), как в графе.
    old_case = все дела этого дня уже встречались у человека в более ранних арестах (повторный арест по старому делу)."""
    c = sqlite3.connect(db)
    ar = pd.read_sql("select person_id, case_number, arrest_date, charge_degree from casearrest", c,
                     parse_dates=["arrest_date"]).dropna(subset=["arrest_date"])
    ch = pd.read_sql("select person_id, case_number, charge from charge", c)
    ar["date"] = ar["arrest_date"].dt.normalize()
    ar = ar[ar["date"] < STUDY_END]                      # отсекает и мусорные даты (2017, 2020, 2049)
    ar["felony"] = ar["charge_degree"].fillna("").str.startswith("(F")
    ar["criminal"] = ar["charge_degree"].fillna("").str.startswith(("(F", "(M"))     # уголовное: felony или misdemeanor
    ch["type"] = ch["charge"].map(cc.categorize_fine)
    case_types = ch.groupby(["person_id", "case_number"])["type"].agg(frozenset)
    pc = ar.groupby(["person_id", "date", "case_number"], as_index=False)[["felony", "criminal"]].max()
    pc = pc.join(case_types.rename("types"), on=["person_id", "case_number"])
    pc["types"] = pc["types"].apply(lambda v: v if isinstance(v, frozenset) else frozenset())
    first_seen = pc.groupby(["person_id", "case_number"])["date"].transform("min")
    pc["new_case"] = pc["date"] == first_seen
    pc["new_crim"] = pc["new_case"] & pc["criminal"]
    a = pc.groupby(["person_id", "date"]).agg(felony=("felony", "max"), old_case=("new_case", lambda s: not s.any()),
                                              new_crim=("new_crim", "max"),
                                              types=("types", lambda s: frozenset().union(*s)),
                                              n_cases=("case_number", "size"), n_old=("new_case", lambda s: int((~s).sum())),
                                              n_felony=("felony", "sum"),
                                              case_types=("types", tuple)).reset_index()
    return a


def load_charges(db=DB):
    """Обвинения (charge) для семейства «обвинения»: строка = обвинение (полные дубли убраны), дата - дата правонарушения
    (offense_date; другой даты в базе нет, date_charge_filed = её копия). no_arrest - по делу не было ни одного ареста
    (штрафы, повестки). Степень: F, M или прочее ((0) - гражданские нарушения, CO, TC...)."""
    c = sqlite3.connect(db)
    ch = pd.read_sql("select * from charge", c)
    ch = ch[~ch.drop(columns="id").duplicated()]
    ch["date"] = pd.to_datetime(ch["offense_date"], errors="coerce").dt.normalize()
    ch = ch[ch["date"].notna() & (ch["date"] < STUDY_END)].copy()
    ch["type"] = ch["charge"].map(cc.categorize_fine)
    d = ch["charge_degree"].fillna("").str.strip("()").str[:1]
    ch["deg"] = np.where(d == "F", "F", np.where(d == "M", "M", "0"))
    ar = pd.read_sql("select distinct person_id, case_number from casearrest", c)
    arrested = set(zip(ar["person_id"], ar["case_number"]))
    ch["no_arrest"] = [(p, k) not in arrested for p, k in zip(ch["person_id"], ch["case_number"])]
    return ch[["person_id", "case_number", "date", "type", "deg", "no_arrest"]].sort_values(["person_id", "date"]).reset_index(drop=True)


def load_prison_before(path=PRISON_BEFORE):
    p = pd.read_csv(path, parse_dates=["in_custody", "out_custody"], encoding="utf-8-sig")
    return p[["person_id", "in_custody", "out_custody"]]


# ----------------------------------------------------------------------------- цели
def add_targets(spells, arrests):
    """Эпизод на свободе: t0 = дата выхода. Три цели:
       arrest    - первый арест после выхода; вход в заключение без ареста цензурирует эпизод;
       newarrest - первый арест по новому уголовному делу (дело впервые у человека, степень F или M);
                   вход в заключение раньше такого ареста (по старому делу, без ареста) цензурирует эпизод;
       custody   - следующий узел графа (вход в заключение), аресты без заключения игнорируются."""
    s = spells.copy()
    s["next_in"] = s.groupby("person_id")["in_date"].shift(-1)
    fe = s[(~s["ongoing"]) & (s["out_date"] < STUDY_END)].copy()
    fe["t0"] = fe["out_date"]
    guard = np.maximum(fe["out_date"], fe["in_date"] + pd.Timedelta(days=ENTRY_GUARD_DAYS))
    fe["guard"] = guard
    a = arrests[["person_id", "date"]].sort_values("date").rename(columns={"date": "t_arrest"})
    fe = fe.sort_values("guard")
    fe = pd.merge_asof(fe, a, left_on="guard", right_on="t_arrest", by="person_id", direction="forward",
                       allow_exact_matches=False)
    an = arrests.loc[arrests["new_crim"], ["person_id", "date"]].sort_values("date").rename(columns={"date": "t_newarrest"})
    fe = pd.merge_asof(fe, an, left_on="guard", right_on="t_newarrest", by="person_id", direction="forward",
                       allow_exact_matches=False)
    fe = fe.merge(arrests[["person_id", "date", "old_case"]].rename(columns={"date": "t_arrest", "old_case": "first_arrest_old"}),
                  on=["person_id", "t_arrest"], how="left")
    ta, tc, t0 = fe["t_arrest"], fe["next_in"], fe["t0"]
    has_c = tc.notna() & (tc < STUDY_END)
    ev_a = ta.notna() & (~has_c | (ta <= tc + pd.Timedelta(days=1)))
    end_c = tc.where(has_c, STUDY_END)                   # вход в заключение или конец наблюдения
    end_a = ta.where(ev_a, end_c)
    fe["event_arrest"] = ev_a.astype(int)
    fe["time_arrest"] = (end_a - t0).dt.days
    fe["cens_by_custody"] = (~ev_a & has_c).astype(int)
    tn = fe["t_newarrest"]
    ev_n = tn.notna() & (~has_c | (tn <= tc + pd.Timedelta(days=1)))
    fe["event_newarrest"] = ev_n.astype(int)
    fe["time_newarrest"] = (tn.where(ev_n, end_c) - t0).dt.days
    fe["cens_by_custody_new"] = (~ev_n & has_c).astype(int)
    fe["event_custody"] = has_c.astype(int)
    fe["time_custody"] = (end_c - t0).dt.days
    return fe.sort_values(["person_id", "k"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- признаки
def _entropy(counts):
    c = np.asarray([v for v in counts if v > 0], float)
    if c.size == 0: return 0.0
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def build_features(fe, spells, arrests, prison_before, charges=None):
    if charges is None:
        charges = load_charges()
    types_all = sorted({t for ts in spells["types"] for t in ts})
    sp_by = {pid: g for pid, g in spells.groupby("person_id")}
    ar_by = {pid: g for pid, g in arrests.sort_values("date").groupby("person_id")}
    pb_by = {pid: g for pid, g in prison_before.groupby("person_id")}
    ch_by = {pid: g for pid, g in charges.groupby("person_id")}
    out = []
    for r in fe.itertuples(index=False):
        t0, tin = r.t0, r.in_date
        f = {"person_id": r.person_id, "k": r.k}
        # --- A. база
        f["a_age"] = (t0 - r.dob).days / 365.25
        f["a_male"] = int(r.sex == "Male")
        # --- история человека до t0
        sp = sp_by[r.person_id]; prev = sp[sp["k"] < r.k]
        ar = ar_by.get(r.person_id)
        ad = ar["date"].values if ar is not None else np.array([], "datetime64[ns]")
        nc = ar["n_cases"].values if ar is not None else np.array([], int)     # арестов (дел) в этот день
        cut = np.datetime64(tin - pd.Timedelta(days=ENTRY_GUARD_DAYS))
        prior = ad < cut                                   # аресты до текущего заключения
        upto = ad <= np.datetime64(t0)                     # все аресты до выхода (включая текущий)
        # --- B. RFM (как в прошлой работе)
        f["r_prev_free_gap"] = (tin - prev["out_date"].iloc[-1]).days if len(prev) else np.nan
        f["r_days_since_prev_arrest"] = (t0 - pd.Timestamp(ad[prior][-1])).days if prior.any() else np.nan
        f["f_custody_num"] = r.k
        f["h_n_prior_arrests"] = int(nc[prior].sum())                # аресты по делам, не дни; правка 10: в семействе F
        cur_dur = (r.cust_out - r.cust_in).total_seconds() / 86400
        pd_dur = ((prev["cust_out"] - prev["cust_in"]).dt.total_seconds() / 86400) if len(prev) else pd.Series(dtype=float)
        f["m_cur_dur"] = cur_dur
        f["m_sum_dur_prev"] = float(pd_dur.sum())
        f["m_avg_dur_prev"] = float(pd_dur.mean()) if len(prev) else np.nan
        # --- C. время
        au = ad[upto]; nu = nc[upto]
        if au.size >= 2:
            iai = np.diff(au).astype("timedelta64[D]").astype(float)
            f["t_iai_last"], f["t_iai_mean"], f["t_iai_min"] = iai[-1], iai.mean(), iai.min()
            f["t_iai_trend"] = iai[-1] / iai.mean() if iai.mean() > 0 else np.nan
        else:
            f["t_iai_last"] = f["t_iai_mean"] = f["t_iai_min"] = f["t_iai_trend"] = np.nan
        age_days = (np.datetime64(t0) - au).astype("timedelta64[D]").astype(float) if au.size else np.array([])
        for h in HALF_LIVES:
            f[f"t_arr_decay_{h}"] = float((nu * np.exp(-np.log(2) * age_days / h)).sum()) if au.size else 0.0
        f["t_arr_last_1y"] = int(nu[age_days < 365].sum()) if au.size else 0
        f["t_arr_last_3y"] = int(nu[age_days < 1095].sum()) if au.size else 0
        if au.size:
            first = pd.Timestamp(au[0])
            f["t_age_first_arrest"] = (first - r.dob).days / 365.25
            career = (t0 - first).days / 365.25
            f["t_career_years"] = career
            f["t_arrest_rate"] = nu.sum() / max(career, 1.0)
        else:
            f["t_age_first_arrest"] = f["t_career_years"] = f["t_arrest_rate"] = np.nan
        win = lambda a, b: (b - max(a, WINDOW_START)).total_seconds() / 86400   # только часть срока внутри окна
        in_win = sum(win(a, b) for a, b in zip(prev["cust_in"], prev["cust_out"])) + win(r.cust_in, r.cust_out)
        f["t_share_in_custody"] = in_win / max((t0 - WINDOW_START).days, 30)
        pb = pb_by.get(r.person_id)
        if pb is not None:
            pb = pb[pb["out_custody"] <= t0]
        if pb is not None and len(pb):
            f["t_prison_before_n"] = len(pb)
            f["t_prison_before_days"] = float((pb["out_custody"] - pb["in_custody"]).dt.days.sum())
            f["t_years_since_prison"] = (t0 - pb["out_custody"].max()).days / 365.25
        else:
            f["t_prison_before_n"] = 0; f["t_prison_before_days"] = 0.0; f["t_years_since_prison"] = np.nan
        f["t_prison_in_window_n"] = int(prev["prison"].sum())
        # --- D. текущий узел графа
        for t in types_all: f[f"g_cur_{t}"] = int(t in r.types)
        f["g_cur_n_types"] = len(r.types)
        for rk, nm in DEG_LEVELS.items(): f[f"g_cur_deg_{nm}"] = int(r.deg_rank == rk)      # самая тяжёлая степень узла
        for rs in REASONS: f[f"g_cur_reason_{rs if rs != 'не привязано' else 'unlinked'}"] = int(r.reason == rs)
        f["g_cur_prison"] = int(r.prison)
        f["g_cur_n_nodes"] = r.n_nodes
        f["g_cur_start_state"] = int(r.start)      # выход из prison, который шёл на 01.01.2013 (начало трассы)
        # --- E. история по графу (узлы до текущего, окно 2013+)
        hist = {t: 0 for t in types_all}
        for ts in prev["types"]:
            for t in ts: hist[t] += 1
        for t in types_all: f[f"gh_n_{t}"] = hist[t]
        tot = {t: hist[t] + int(t in r.types) for t in types_all}
        f["gh_distinct_types"] = sum(v > 0 for v in tot.values())
        f["gh_entropy"] = _entropy(tot.values())
        seq = list(prev["types"]) + [set(r.types)]
        reps = [int(bool(set(seq[i]) & set(seq[i - 1]))) for i in range(1, len(seq))]
        f["gh_repeat_prev"] = reps[-1] if reps else 0
        f["gh_n_repeats"] = int(sum(reps))
        f["gh_share_repeats"] = float(np.mean(reps)) if reps else np.nan
        f["gh_share_oldcase"] = hist.get("OldCase", 0) / len(prev) if len(prev) else np.nan
        f["gh_n_felony_prev"] = int(prev["felony"].sum())
        # причина прошлого возврата = причина входа в текущее заключение (только для повторных эпизодов, k >= 2)
        pc = CAUSE_OF_REASON.get(r.reason, 3) if len(prev) else None
        for cz, nm in ((1, "new"), (2, "old"), (3, "unknown")):
            f[f"gh_prev_cause_{nm}"] = (int(pc == cz) if pc is not None else np.nan)
        # аресты без заключения: между выходом из прошлого заключения (или 01.01.2013) и входом в текущее
        lo = np.datetime64(prev["out_date"].iloc[-1]) if len(prev) else np.datetime64(WINDOW_START)
        btw = (ad > lo) & (ad < cut)
        f["gh_n_arrests_nocust_prev"] = int(nc[btw].sum())
        f["gh_n_arrests_nocust_window"] = int(f["gh_n_arrests_nocust_prev"] + sum(
            nc[(ad > np.datetime64(a)) & (ad < np.datetime64(b - pd.Timedelta(days=ENTRY_GUARD_DAYS)))].sum()
            for a, b in zip([WINDOW_START] + list(prev["out_date"].iloc[:-1]), list(prev["in_date"])) if len(prev)))
        # --- F. история арестов по типам, вся жизнь (charge)
        pa = ar[prior] if ar is not None else None
        n_pa = 0 if pa is None else int(pa["n_cases"].sum())             # аресты по делам
        cnt = {t: 0 for t in H_TYPES}; other = 0; same = 0; allc = {}
        cur_types = set(r.types) - {"OldCase", "Unknown"}
        if n_pa:
            for cts in pa["case_types"]:                                  # каждое дело дня отдельно
                for ts in cts:
                    for t in ts:
                        allc[t] = allc.get(t, 0) + 1
                        if t in cnt: cnt[t] += 1
                        else: other += 1
                    same += int(bool(cur_types & ts))
        for t in H_TYPES: f[f"h_n_{t}"] = cnt[t]
        f["h_n_other"] = other
        f["h_distinct_types"] = len(allc)
        f["h_entropy"] = _entropy(allc.values())
        f["h_share_same_as_cur"] = same / n_pa if (n_pa and cur_types) else np.nan
        f["h_felony_share"] = float(pa["n_felony"].sum() / n_pa) if n_pa else np.nan
        f["h_n_oldcase_arrests"] = int(pa["n_old"].sum()) if n_pa else 0
        # --- обвинения (charge), на дату правонарушения <= t0, включая дела без ареста (правки 1, 5, 8)
        chp = ch_by.get(r.person_id)
        if chp is not None:
            chp = chp[chp["date"].values <= np.datetime64(t0)]
        if chp is not None and len(chp):
            age = (np.datetime64(t0) - chp["date"].values).astype("timedelta64[D]").astype(float)
            na = chp["no_arrest"].values
            f["c_n_all"] = len(chp); f["c_n_1y"] = int((age < 365).sum()); f["c_n_3y"] = int((age < 1095).sum())
            f["c_n_cases"] = int(chp["case_number"].nunique())
            f["c_n_noarrest_all"] = int(na.sum()); f["c_n_noarrest_1y"] = int((na & (age < 365)).sum())
            f["c_n_noarrest_3y"] = int((na & (age < 1095)).sum())
            f["c_n_noarrest_cases"] = int(chp.loc[na, "case_number"].nunique())
            for dg in ("F", "M", "0"): f[f"c_n_deg_{dg}"] = int((chp["deg"].values == dg).sum())
            tv = chp["type"].values
            for t in H_TYPES: f[f"c_n_{t}"] = int((tv == t).sum())
            f["c_n_other"] = int((~np.isin(tv, H_TYPES)).sum())
        else:
            for k_ in ["c_n_all", "c_n_1y", "c_n_3y", "c_n_cases", "c_n_noarrest_all", "c_n_noarrest_1y", "c_n_noarrest_3y",
                       "c_n_noarrest_cases", "c_n_deg_F", "c_n_deg_M", "c_n_deg_0"] + [f"c_n_{t}" for t in H_TYPES] + ["c_n_other"]:
                f[k_] = 0
        out.append(f)
    return pd.DataFrame(out)


FAMILIES = {"A база": "a_", "B RFM": ("r_", "f_", "m_"), "C время": "t_", "D узел графа": "g_",
            "E история по графу": "gh_", "F типы арестов (вся история)": "h_", "F2 обвинения (вся история)": "c_",
            "G раса": "race_"}

def family_columns(df):
    return {name: [c for c in df.columns if c.startswith(pref)] for name, pref in FAMILIES.items()}


def build_dataset(save=True):
    nodes = load_nodes(); spells = build_spells(nodes); arrests = load_arrests()
    fe = add_targets(spells, arrests)
    X = build_features(fe, spells, arrests, load_prison_before(), load_charges())
    keep = ["person_id", "k", "t0", "cust_in", "cust_out", "time_arrest", "event_arrest", "cens_by_custody", "time_newarrest", "event_newarrest",
            "cens_by_custody_new", "first_arrest_old", "time_custody", "event_custody", "start"]
    ds = fe[keep].merge(X, on=["person_id", "k"])
    ds["types"] = fe["types"].map(lambda s: "+".join(sorted(s))).values
    # причина возврата (конкурирующие риски) и раса (семейство G) - функции в stage2_extra
    import stage2_extra as sx
    ds = sx.add_causes(ds, spells)
    R, race4 = sx.race_features(ds)
    ds = pd.concat([ds, R], axis=1); ds["race4"] = race4
    if save:
        (HERE / "out").mkdir(exist_ok=True)
        ds.to_csv(HERE / "out" / "free_episodes_features.csv", index=False)
    return ds, dict(nodes=nodes, spells=spells, arrests=arrests, fe=fe)


if __name__ == "__main__":
    ds, parts = build_dataset()
    print(ds.shape, ds["person_id"].nunique())
    print(ds[["event_arrest", "event_custody", "cens_by_custody"]].mean().round(3).to_dict())
    print(ds[["time_arrest", "time_custody"]].describe().round(1).to_string())
