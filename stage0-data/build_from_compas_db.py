"""Таблица заключений с обвинениями напрямую из ProPublica compas.db.

Одна строка = одно обвинение внутри одного заключения одного человека.
Столбцы как в individual_custody_timeline_rfm.csv Даши, но вместо name — id:

  id, age, sex, race, c_charge_degree, c_charge_desc, in_custody, out_custody, custody_num,
  time_to_last_custody, time_since_last_arrest, time_to_future_arrest, event, event_of_future_arrest,
  curr_dur_custody, sum_dur_custody, average_dur_custody, all_future_arrest_times, all_future_events
  + в конце: charge_num, n_charges, charge_category, charge_source, case_number, custody_source

Как строится:
  1. Люди: все person_id из casearrest; age, sex, race — из people (people.id = person_id).
  2. Заключения: jailhistory + prisonhistory этих людей с in_custody >= 2013-01-01
     (jailhistory в базе есть только с 2013 года; более ранние записи prisonhistory дали бы
     ложные многолетние промежутки). Точные дубли (тот же человек, те же даты) убраны.
     Даты округлены до дня, порядок — по времени входа.
  3. Обвинения заключения:
     a) arrest: аресты из casearrest с arrest_date в [in_custody − 1, in_custody + 1] дней,
        все обвинения этих дел из charge (по person_id + case_number);
     b) offense: если ареста нет — обвинения из charge с offense_date в [in_custody − 7, in_custody + 1];
     c) prior_case: если и так нет (приговор, нарушение условий) — обвинения последнего ареста
        за 365 дней до заключения;
     d) none: ничего не найдено — одна строка с пустым обвинением.
  4. Производные столбцы (формулы как у Даши, считаются по id):
     custody_num            — номер заключения у человека;
     time_to_last_custody   — in_custody − предыдущий in_custody, дни;
     time_since_last_arrest — in_custody − предыдущий out_custody, дни;
     time_to_future_arrest  — следующий in_custody − in_custody, дни (пусто для последнего);
     event, event_of_future_arrest — 1, если есть следующее заключение, иначе 0 (цензура);
     curr_dur_custody       — out_custody − in_custody, дни;
     sum_dur_custody        — сумма curr_dur_custody предыдущих заключений;
     average_dur_custody    — sum_dur_custody / (custody_num − 1);
     all_future_arrest_times — дни от in_custody до каждого будущего in_custody;
     all_future_events      — [1]*число будущих заключений + [0].

Запуск:  python build_from_compas_db.py D:\\diploma\\compas.db [папка_для_результата]
База открывается только на чтение и не меняется.
Результат: compas_custody_charges.csv и compas_custody_charges.xlsx.
"""
import os
import re
import sqlite3
import sys

import numpy as np
import pandas as pd

START = pd.Timestamp("2013-01-01")

# категории преступлений — те же правила, что в stage1/crime_categories.py
RULES = [
    ("NoCharge", r"no charge"),
    ("Sex", r"lewd|lasciv|sex|prostitu|indecent expos|voyeur|porn|obscen|molest|travel.*minor|child exploit|computer for child"),
    ("Traffic", r"\bdui\b|d\.u\.i|\bdwi\b|\bdwls\b|driv|licen|\bdl\b|\blic\b|motorcycle|vehicle insurance|motor veh ins"
                r"|insur.*card|proof of invalid insur|flee|elud|leav.*(scen|acc)|lve/scen|reckless driv|register vehicle"
                r"|tag|sticker|boater|load|tri-rail|taxi"
                # нарушения ПДД (часто идут сопутствующими обвинениями в том же деле)
                r"|safety belt|stbelt|seat ?belt|child restraint|pers/inj/prot|submit insurance"
                r"|stop sign|signal|red light|flashing|yield|ftyrow|\bspeed|too fast|following too closely"
                r"|\blanes?\b|\bturn|backing|stop/stand|stop stand|stand/park|imp park|improp park|park handicap|park/obstruct|hov|sunscr|wndow|windshield|window"
                r"|due care|unsafe/improp|unsafe/improper|improp equip|lights|lamps|mcycle|bike|ped fail|x-walk"
                r"|refuse submit blood|breath test|render aid|rept accd|road surface|sounds>1000|display regist"),
    ("Violent", r"murder|manslaughter|homicide|home invasion|kidnap|carjack|robbery|battery|\bbatt\b|batt/|assault|false imprison|stalk"
                r"|child abuse|cruelty|neglect|abuse|shoot|throw.*missile|throw in occup|harm public servant"
                r"|injunc|dom vi|domestic|dating viol|threat|interference with custody|culpable negligence"
                r"|exposes culpable"),
    ("Drug", r"cocaine|cannabis|heroin|meth|oxycod|hydrocod|hydromorph|alprazolam|clonazepam|diazepam|lorazepam"
             r"|amphet|morphine|codeine|fentanyl|methadone|bupren|carisoprodol|phentermine|oxymorph|mdma|lsd"
             r"|ecstasy|ethylone|butylone|xlr11|pyrrolid|piperaz|pipe\b|controlled? subst|contr sub|control subst"
             r"|cont subs|paraphernalia|tetrahydro|\brx\b|prescript|drug|traff|jwh|fluoro|gamma-hydroxy|butanediol"
             r"|steroid|trenbolon|amobarbital|methox|contrft schd|countrft|cntrft|canbs|fubinaca|synthetic"),
    ("Weapon", r"firearm|weapon|\bwep\b|f/arm|concealed|ammun|destruct device|bomb"),
    ("Fraud", r"forg|utter|fraud|credit card|\bcc\b|personal id|identity|false name|false own|counterfeit|worthless"
              r"|defraud|launder|structuring|scanning device|innkeeper|ID card|unlaw.*\bid\b|unlaw use false"
              r"|imperson|police badge|simulation|draft|trade secret|intellectual|bribery|telemarket"
              r"|white collar|bad check|false info|false 911|false crime|giving false|posses.*altered bills|unlaw pos of prson|hire|lease"),
    ("Property", r"theft|burgl|stolen|mischief|shoplift|arson|trespass|tresspass|retail|cart|tools|fire extin|\bburg\b|burg/"
                 r"|altered id|alt vin|railroad|fuel|deprive"),
]
COMPILED = [(n, re.compile(rx, re.I)) for n, rx in RULES]


def categorize(desc):
    if pd.isna(desc) or not str(desc).strip():
        return "Unknown"
    for name, rx in COMPILED:
        if rx.search(str(desc)):
            return name
    return "Other"


def lst(v):
    return "[" + ", ".join(str(int(x)) for x in v) + "]"


def timeline(g):
    i, o, n = g["in_custody"], g["out_custody"], len(g)
    r = pd.DataFrame(index=g.index)
    r["custody_num"] = np.arange(1, n + 1)
    r["time_to_last_custody"] = (i - i.shift()).dt.days
    r["time_since_last_arrest"] = (i - o.shift()).dt.days
    r["time_to_future_arrest"] = (i.shift(-1) - i).dt.days
    r["event"] = (np.arange(n) < n - 1).astype(int)
    r["event_of_future_arrest"] = r["event"]
    dur = (o - i).dt.days
    r["curr_dur_custody"] = dur
    r["sum_dur_custody"] = dur.fillna(0).cumsum().shift(fill_value=0).astype(int)
    avg = r["sum_dur_custody"] / (r["custody_num"] - 1).replace(0, np.nan)
    r["average_dur_custody"] = avg.round(2)
    days = [(i.iloc[k + 1:] - i.iloc[k]).dt.days.tolist() for k in range(n)]
    r["all_future_arrest_times"] = [lst(d) if d else np.nan for d in days]
    r["all_future_events"] = [lst([1] * len(d) + [0]) for d in days]
    return r


def day(s):
    return pd.to_datetime(s, format="mixed", errors="coerce").dt.normalize()


def main(db_path, out_dir):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    arrests = pd.read_sql("select person_id, case_number, arrest_date from casearrest", con)
    people = pd.read_sql("select id as person_id, age, sex, race from people", con)
    charges = pd.read_sql("select person_id, case_number, offense_date, charge_number, "
                          "charge_degree, charge from charge", con)
    hist = pd.concat([pd.read_sql(f"select person_id, in_custody, out_custody, '{src}' as custody_source "
                                  f"from {tbl}", con)
                      for tbl, src in [("jailhistory", "jail"), ("prisonhistory", "prison")]])

    arrests["arrest_date"] = day(arrests["arrest_date"])
    charges["offense_date"] = day(charges["offense_date"])
    charges = charges.dropna(subset=["charge"])
    arr_ch = arrests.drop_duplicates(["person_id", "case_number", "arrest_date"]).merge(
        charges, on=["person_id", "case_number"])

    # 1–2. люди из casearrest, их заключения с 2013 года
    ids = set(arrests["person_id"])
    hist["in_t"] = pd.to_datetime(hist["in_custody"], format="mixed")
    hist["in_custody"] = hist["in_t"].dt.normalize()
    hist["out_custody"] = day(hist["out_custody"])
    cust = (hist[hist["person_id"].isin(ids) & (hist["in_custody"] >= START)]
            .sort_values(["person_id", "in_t", "out_custody"])
            .drop_duplicates(["person_id", "in_custody", "out_custody"])
            .reset_index(drop=True))
    cust = cust.merge(people, on="person_id", how="left")
    cust = cust.rename(columns={"person_id": "id"})
    cust = pd.concat([cust, pd.concat([timeline(g) for _, g in cust.groupby("id", sort=False)])], axis=1)
    cust["ep"] = np.arange(len(cust))
    print("людей:", cust["id"].nunique(), " заключений:", len(cust))

    # 3. обвинения каждого заключения
    a_by = dict(tuple(arr_ch.groupby("person_id")))
    c_by = dict(tuple(charges.groupby("person_id")))
    one = pd.Timedelta(days=1)
    rows = []
    for ep, pid, d in zip(cust["ep"], cust["id"], cust["in_custody"]):
        a = a_by.get(pid)
        found, src = None, "none"
        if a is not None:
            x = a[(a["arrest_date"] >= d - one) & (a["arrest_date"] <= d + one)]
            if len(x):
                found, src = x, "arrest"
        if found is None and pid in c_by:
            c = c_by[pid]
            x = c[(c["offense_date"] >= d - 7 * one) & (c["offense_date"] <= d + one)]
            if len(x):
                found, src = x, "offense"
        if found is None and a is not None:
            x = a[(a["arrest_date"] < d - one) & (a["arrest_date"] >= d - 365 * one)]
            if len(x):
                found, src = x[x["arrest_date"] == x["arrest_date"].max()], "prior_case"
        if found is not None:
            f = (found.drop_duplicates(["case_number", "charge_number"])
                 .sort_values(["case_number", "charge_number"]))
            rows.append(pd.DataFrame({"ep": ep, "c_charge_degree": f["charge_degree"].values,
                                      "c_charge_desc": f["charge"].values,
                                      "case_number": f["case_number"].values, "charge_source": src}))
        else:
            rows.append(pd.DataFrame({"ep": [ep], "charge_source": ["none"]}))
    ch = pd.concat(rows, ignore_index=True)
    ch["n_charges"] = ch.groupby("ep")["c_charge_desc"].transform("count")
    ch["charge_num"] = np.where(ch["n_charges"] > 0, ch.groupby("ep").cumcount() + 1, 0)
    ch["charge_category"] = ch["c_charge_desc"].map(categorize)

    out = cust.merge(ch, on="ep").sort_values(["id", "custody_num", "charge_num"])
    for c in ["in_custody", "out_custody"]:
        out[c] = out[c].dt.strftime("%Y-%m-%d")
    cols = ["id", "age", "sex", "race", "c_charge_degree", "c_charge_desc", "in_custody", "out_custody",
            "custody_num", "time_to_last_custody", "time_since_last_arrest", "time_to_future_arrest", "event",
            "event_of_future_arrest", "curr_dur_custody", "sum_dur_custody", "average_dur_custody",
            "all_future_arrest_times", "all_future_events",
            "charge_num", "n_charges", "charge_category", "charge_source", "case_number", "custody_source"]
    out = out[cols]

    os.makedirs(out_dir, exist_ok=True)
    csv = os.path.join(out_dir, "compas_custody_charges.csv")
    out.to_csv(csv, index=False)
    print("строк (обвинений):", len(out))
    print(out.drop_duplicates(["id", "custody_num"])["charge_source"].value_counts().to_string())
    try:
        with pd.ExcelWriter(os.path.join(out_dir, "compas_custody_charges.xlsx"), engine="openpyxl") as w:
            out.to_excel(w, sheet_name="data", index=False)
            w.sheets["data"].freeze_panes = "B2"
    except ImportError:
        print("openpyxl не установлен — xlsx не записан, есть только csv")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "compas.db",
         sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__)))
