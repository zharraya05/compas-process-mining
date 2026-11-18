"""Сборка графа заключений для этапа 2 (папка stage2_custody). Логика и числа сборки 18.11.2026 - build_graph_описание.md.

Код ячеек `все по новой/30.10.ipynb` (таблицы, очистка, переводы, причины, типы, prison до 2013) и функций
`этап 1/make_notebook.py` (эпизоды, состояния узлов, Justice) исполняется с правками; исходные ноутбуки не меняются.

  1   база - `stage0-data/compas.db` репозитория (копия ProPublica).
  2   очистка - только 4 правила: jail с выходом раньше входа (в тот же день -> выход + 12 ч, иначе удалить);
      полные дубликаты charge; аресты с датой в будущем; фильтр people не применяется. Остальные фильтры 30.10 сняты.
  3   jail + prison -> одна таблица заключений.
  4.1 арест в [день входа - 1; день входа], все дела таких арестов; к заключению - обвинения дела с датой не позже
      дня входа; одно обвинение может быть привязано к нескольким заключениям. Дело, по которому у человека уже было
      заключение в окне (вход с 01.01.2013; prison до 2013 не считается), - старое: rearrest_case_ids,
      rearrest_prev_custody_ids, rearrest_new_charges_n (обвинения дела после прошлого заключения), типы его обвинений -
      rearrest_case_types. Только старое дело -> rearrest_old_case, узел OldCase; есть новое дело -> new_offense,
      состояние - типы нового дела, старое дело - теми же полями.
  4.2 перевод jail -> prison или prison -> jail (разрыв меньше суток, без новых арестов и обвинений между ними) -
      наследует дело первого заключения цепочки и склеивается с ним в один узел.
  4.4 непривязанные jail внутри срока prison (включая сроки с заглушкой 2020-01-01) удаляются.
  4.5 остальное - не привязано (Unknown); этапы 2-4, 6, 7 из 30.10 не используются.
  7   prison на 01.01.2013 - начальные (start_prisons.csv, не узлы графа); остальные prison до 2013 - prison_before_2013.csv.
  8   чистка трасс: конец наблюдения 01.04.2016; начальный prison без выхода - трасса убирается; трассы «только prison» убираются.
  9   узлы: переводы склеены; состояние - 15 типов обвинений (без обвинений старого дела); OldCase - узлы только со старым
      делом; тяжесть - по всем обвинениям узла; Justice отдельным шагом; последний шаг цензурирован.

Выход (папка данные/):
  custody_linked.csv, custody_charges.csv   - заключения после привязки и чистки (без начальных prison);
                                              в custody_charges - old_case, new_in_old_case
  start_prisons.csv                         - начальные prison (шли на 01.01.2013) с причиной и обвинениями
  prison_before_2013.csv                    - prison, закончившиеся до 2013 (только признаки истории)
  event_table.csv                           - граф этапа 1 (узлы и шаги), без начальных prison
  event_table_stage2.csv                    - то же + начальные prison (start_state = True) для эпизодов этапа 2
  cleaning_report.csv, log_info.csv, build_log.txt

Запуск: python build_graph.py   (несколько минут; с --checkpoint расчёт 30.10 сохраняется после долгих ячеек
        и продолжается повторным запуском)
"""
import contextlib, io, json, os, re, sys, types
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent                                   # корень репозитория compas-process-mining
VENDOR = HERE / "vendor"                             # копии кода, на который опирается сборка (см. README)
NB_3010 = VENDOR / "30.10.ipynb"                     # ноутбук «все по новой/30.10.ipynb»
MAKE_NB1 = VENDOR / "stage1_make_notebook.py"        # «этап 1/make_notebook.py»
DB = REPO / "stage0-data" / "compas.db"              # база ProPublica (та же, MD5 32d9cba8…)
OUT = HERE / "данные"
WINDOW_START = pd.Timestamp("2013-01-01")
STUDY_END = pd.Timestamp("2016-04-01")               # правка 6: конец наблюдения везде
PLACEHOLDER = pd.Timestamp("2020-01-01")
CHECKPOINT_SEC = 30 if "--checkpoint" in sys.argv else 0


def patch(src, old, new, count=1):
    assert src.count(old) == count, f"патч не применился: {old[:70]!r} (найдено {src.count(old)})"
    return src.replace(old, new)


# ----------------------------------------------------------------------------- 1. код 30.10 с правками
A2_FIX = '''
# --- правка А2: выход раньше входа в тот же день -> выход + 12 ч (время без AM/PM) ---
j = dfs["jailhistory"].copy()
_ti, _to = to_dt(j.in_custody), to_dt(j.out_custody)
_bad = _ti > _to
_same = _bad & (_ti.dt.normalize() == _to.dt.normalize())
j.loc[_same, "out_custody"] = (_to[_same] + pd.Timedelta(hours=12)).dt.strftime("%Y-%m-%d %H:%M:%S.%f")
REPORT["А2: выход раньше входа"] = int(_bad.sum()); REPORT["А2: исправлено +12 ч"] = int(_same.sum())
REPORT["А2: удалено"] = int((_bad & ~_same).sum())
'''

PEOPLE_NOFILTER = '''# --- people: правка 9 - фильтры по полям анализа ProPublica не применяются ---
p = dfs["people"]
REPORT["people: людей (фильтр не применяется)"] = len(p)
clean["people"] = p.reset_index(drop=True)

# --- compas ---'''

START_FIX = '''DATA_START = pd.Timestamp("2013-01-01")

# правка 3: prison, покрывающий 01.01.2013, - не узел графа (отдельный файл), трасса начинается с выхода из него
custody["start_state"] = (custody["source"] == "prison") & (custody["in_dt"] < DATA_START) & (custody["out_dt"] >= DATA_START)
REPORT["prison на 01.01.2013 (начальные)"] = int(custody["start_state"].sum())
pre2013 = (custody["source"] == "prison") & (custody["in_dt"] < DATA_START) & ~custody["start_state"]'''


LINK_1711 = r'''
# ======== привязка по заметке «Логика build_graph» (18.11.2026), вместо этапов 1-7 из 30.10 ========
# 4.1 арест в [день входа - 1; день входа]: все дела таких арестов. К заключению привязываются обвинения дела
#     с датой не позже дня входа; одно обвинение может быть привязано к нескольким заключениям.
#     Дело, по которому у человека уже было заключение в окне (вход с 01.01.2013), - старое дело:
#     номер (rearrest_case_ids), прошлые заключения (rearrest_prev_custody_ids), число новых обвинений по делу
#     после прошлого заключения (rearrest_new_charges_n). Только старое дело -> rearrest_old_case (узел OldCase);
#     есть новое дело -> new_offense (состояние - типы нового дела, старое - пометкой).
# 4.2 перевод jail -> prison или prison -> jail (разрыв меньше суток, без новых арестов и обвинений между ними) -
#     наследует дело первого заключения цепочки и склеивается с ним в один узел.
# 4.5 остальные - не привязаны (Unknown).
WEEK = pd.Timedelta("7D")
DAY  = pd.Timedelta("1D")
WINDOW_START = pd.Timestamp("2013-01-01")

custody["in_dt"]   = pd.to_datetime(custody["in_custody"])
custody["out_dt"]  = pd.to_datetime(custody["out_custody"])
custody["in_date"] = custody["in_dt"].dt.normalize()
ch = charge.assign(charge_date=pd.to_datetime(charge["date_charge_filed"]).dt.normalize())
ca = casearrest.assign(arrest_date=pd.to_datetime(casearrest["arrest_date"]).dt.normalize())
ca["first_arrest_date"] = ca.groupby(["person_id", "case_number"])["arrest_date"].transform("min")
ca["is_rearrest"] = ca["arrest_date"] > ca["first_arrest_date"]        # нужно ячейке причин 30.10, в правилах не участвует
case_degrees = (ch.groupby(["person_id", "case_number"])["charge_degree"]
                  .agg(lambda s: ", ".join(sorted(s.dropna().unique()))).to_dict())

charge_by_person = dict(tuple(ch.groupby("person_id")))
arrest_by_person = dict(tuple(ca.groupby("person_id")))
cust_by_person   = dict(tuple(custody.groupby("person_id")))
NO_CH, NO_AR = ch.iloc[0:0], ca.iloc[0:0]

links  = []            # (custody_id, charge_id, link_stage); обвинение может повторяться у разных заключений
result = {}
case_custodies = {}    # person_id -> {дело: [(custody_id, день входа), ...]} - заключения по делу в окне

def link(cid_custody, rows, stage):
    for cid in rows["id"]:
        links.append((cid_custody, cid, stage))

# ---------------- 4.1: арест в [n - 1; n] ----------------
for r in custody.sort_values(["person_id", "in_dt"]).itertuples():
    c = charge_by_person.get(r.person_id, NO_CH)
    c = c[c["charge_date"] <= r.in_date]                       # только обвинения с датой не позже дня входа
    a = arrest_by_person.get(r.person_id, NO_AR)
    res = dict(link_stage=None, reason_type=None, case_ids=None, link_status=None,
               rearrest_case_ids=None, rearrest_first_arrest_date=None, original_case_degrees=None,
               rearrest_prev_custody_ids=None, rearrest_new_charges_n=0,
               possible_oldcase=False, oldcase_custody_ids=None)
    arr = a[(a["arrest_date"] >= r.in_date - DAY) & (a["arrest_date"] <= r.in_date)]
    if arr.empty:
        res["link_status"] = "4.1: нет ареста в [вход - 1 день; вход]"
        result[r.custody_id] = res
        continue
    seen = case_custodies.get(r.person_id, {})
    cases = sorted(arr["case_number"].unique())
    rep = [cn for cn in cases if cn in seen]                   # по делу уже было заключение в окне
    new = [cn for cn in cases if cn not in seen]
    status = []
    s_new = c[c["case_number"].isin(new)]
    if len(s_new):
        link(r.custody_id, s_new, 1)
    elif new:
        status.append(f"4.1: новое дело {', '.join(new)} без обвинений с датой не позже входа")
    if rep:
        s_old = c[c["case_number"].isin(rep)]
        if len(s_old):
            link(r.custody_id, s_old, 1)
        last_in = {cn: max(d for _, d in seen[cn]) for cn in rep}             # вход в последнее прошлое заключение по делу
        n_new = int(sum((s_old["case_number"] == cn).values & (s_old["charge_date"] > last_in[cn]).values for cn in rep).sum()) \
            if len(s_old) else 0
        res.update(rearrest_case_ids=", ".join(rep),
                   rearrest_first_arrest_date=", ".join(sorted(arr.loc[arr["case_number"].isin(rep), "first_arrest_date"]
                                                               .dt.strftime("%Y-%m-%d").unique())),
                   original_case_degrees="; ".join(f"{cn}: {case_degrees.get((r.person_id, cn), '?')}" for cn in rep),
                   rearrest_prev_custody_ids="; ".join(f"{cn}: {', '.join(str(i) for i, _ in seen[cn])}" for cn in rep),
                   rearrest_new_charges_n=n_new)
        status.append(f"4.1: повторное заключение по делу {res['rearrest_case_ids']} "
                      f"(прошлые заключения {res['rearrest_prev_custody_ids']}; новых обвинений по делу {n_new})")
    linked_new = sorted(s_new["case_number"].unique())
    if linked_new:
        res.update(link_stage=1, reason_type="new_offense", case_ids=", ".join(linked_new + rep))
    elif rep:
        res.update(link_stage=1, reason_type="rearrest_old_case", case_ids=", ".join(rep))
    res["link_status"] = "; ".join(status) or None
    if res["link_stage"] is not None and r.in_dt >= WINDOW_START:           # prison до 2013 не считается
        for cn in linked_new + rep:
            case_custodies.setdefault(r.person_id, {}).setdefault(cn, []).append((r.custody_id, r.in_date))
    result[r.custody_id] = res

res = pd.DataFrame.from_dict(result, orient="index").rename_axis("custody_id").reset_index()
custody = (custody.drop(columns=[c for c in res.columns if c != "custody_id" and c in custody.columns])
                  .merge(res, on="custody_id", how="left"))
for col in ["stage4_arrest_date", "stage4_days_from_arrest", "inherited_from_custody_id"]:
    custody[col] = pd.NA

# ---------------- 4.2: перевод jail -> prison или prison -> jail ----------------
def new_events(person_id, start_date, end_date):
    c = charge_by_person.get(person_id, NO_CH)
    a = arrest_by_person.get(person_id, NO_AR)
    return (((c["charge_date"] > start_date) & (c["charge_date"] <= end_date)).sum(),
            ((a["arrest_date"] > start_date) & (a["arrest_date"] <= end_date)).sum())

custody["transfer_note"] = pd.NA
root = {}                                   # custody_id перевода -> первое заключение цепочки
n_dir = {"jail -> prison": 0, "prison -> jail": 0}
for r in custody[custody["link_stage"].isna()].sort_values(["person_id", "in_dt"]).itertuples():
    pc = cust_by_person[r.person_id]
    prev = pc[(pc["source"] != r.source) & (pc["in_dt"] < r.in_dt) & (pc["custody_id"] != r.custody_id)]
    if prev.empty:
        continue
    j = prev.loc[(r.in_dt - prev["out_dt"]).abs().idxmin()]          # заключение с выходом ближе всего ко входу
    gap = r.in_dt - j.out_dt
    if abs(gap) >= DAY:
        continue
    start = j.in_dt.normalize() if j.source == "jail" else max(j.in_dt.normalize(), j.out_dt.normalize() - DAY)
    n_ch, n_ar = new_events(r.person_id, start, r.in_date)
    if n_ch or n_ar:
        custody.loc[custody["custody_id"] == r.custody_id, "transfer_note"] = (
            f"не перевод {j.source} -> {r.source} (custody_id={j.custody_id}): новые обвинения {n_ch}, аресты {n_ar}")
        continue
    src = root.get(j.custody_id, j.custody_id)
    root[r.custody_id] = src
    n_dir[f"{j.source} -> {r.source}"] += 1
    custody.loc[custody["custody_id"] == r.custody_id, "transfer_note"] = (
        f"перевод {j.source} -> {r.source}: из custody_id={src} (непосредственно из {j.custody_id}, разрыв {gap})")

custody_charges = (custody
    .merge(pd.DataFrame(links, columns=["custody_id", "charge_id", "charge_link_stage"]), on="custody_id", how="left")
    .merge(ch[["id", "charge_date", "case_number", "charge_number", "charge_degree", "charge", "statute"]]
             .rename(columns={"id": "charge_id"}), on="charge_id", how="left"))
custody_charges["days_before_custody"] = (custody_charges["in_date"] - custody_charges["charge_date"]).dt.days

rt = custody["reason_type"]
tr = custody["transfer_note"].str.startswith("перевод", na=False)
print(f"Всего записей о заключении: {len(custody)}")
print(f"  4.1 новое дело (new_offense):                        {(rt == 'new_offense').sum()}")
print(f"      из них со старым делом в тот же день (смешанные):  {((rt == 'new_offense') & custody['rearrest_case_ids'].notna()).sum()}")
print(f"  4.1 только старое дело (rearrest_old_case -> OldCase): {(rt == 'rearrest_old_case').sum()}")
print(f"      с новыми обвинениями по старому делу:              {((custody['rearrest_new_charges_n'] > 0) & custody['rearrest_case_ids'].notna()).sum()}")
print(f"  4.2 переводы: {n_dir}")
print(f"  4.5 не привязано: {(custody['link_stage'].isna() & ~tr).sum()}")
print(f"Привязок обвинение -> заключение: {len(links)} (обвинений: {len({x[1] for x in links})})")
'''


CLEAN_PATCHES = [   # заметка «Логика build_graph», п. 2: очистка - только 4 правила
    ('    "in_custody раньше даты рождения": to_dt(j.in_custody) < to_dt(j.dob),\n', ''),
    ('clean["prisonhistory"] = apply_filters(pr, "prisonhistory", {\n'
     '    "in_custody > out_custody": to_dt(pr.in_custody) > to_dt(pr.out_custody),\n'
     '    "in_custody раньше даты рождения": to_dt(pr.in_custody) < to_dt(pr.dob),\n'
     '})\n', 'clean["prisonhistory"] = pr.reset_index(drop=True)          # без фильтров\n'),
    ('clean["compas"] = apply_filters(cp, "compas", {\n'
     '    "decile_score = -1 (нет оценки)": cp.decile_score < 1,\n'
     '    "полные дубликаты": cp.drop(columns="id").duplicated(),\n'
     '})\n', 'clean["compas"] = cp.reset_index(drop=True)                 # без фильтров\n'),
    ('    "offense_date > date_charge_filed": to_dt(ch.offense_date) > to_dt(ch.date_charge_filed),\n'
     '    "offense_date в будущем": to_dt(ch.offense_date) > CUTOFF,\n', ''),
]


def run_3010(log):
    nb = json.loads(NB_3010.read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    cells[0] = patch(cells[0], r'DB_PATH = r"D:\diploma\jupyter lab\stage0 - data\compas.db"', f"DB_PATH = {str(DB)!r}")
    cells[1] = patch(cells[1], '# --- jailhistory ---\nj = dfs["jailhistory"]\n', "# --- jailhistory ---\n" + A2_FIX)
    cells[1] = re.sub(r'# --- people ---\np = dfs\["people"\]\nclean\["people"\] = apply_filters\(.*?\}\)\n\n# --- compas ---',
                      lambda m: PEOPLE_NOFILTER, cells[1], count=1, flags=re.S)
    assert "PEOPLE_NOFILTER" not in cells[1] and "правка 9" in cells[1], "патч people не применился"
    for old, new in CLEAN_PATCHES:
        cells[1] = patch(cells[1], old, new)
    cells[3] = LINK_1711                              # привязка по заметке: арест [n-1; n] + переводы, остальное Unknown
    cells[4] = cells[6] = cells[7] = "pass"            # этапы 4, 6, 7 из 30.10 не используются
    cells[5] = patch(cells[5], 'assert not links_df["charge_id"].duplicated().any(), "обвинение привязано к нескольким заключениям"',
                     '# правило «одно обвинение - одно заключение» отменено (заметка «Логика build_graph», 4.1)')
    cells[5] = patch(cells[5], 'custody.loc[(custody["source"] == "prison") & (custody["out_dt"] != PLACEHOLDER),',
                     'custody.loc[(custody["source"] == "prison"),                  # правка 7: и сроки с заглушкой\n                   ')
    cells[-1] = patch(cells[-1], 'DATA_START = pd.Timestamp("2013-01-01")\n\npre2013 = (custody["source"] == "prison") & (custody["in_dt"] < DATA_START)',
                      START_FIX)
    import ast, pickle, time, types as _t
    state = OUT / "_state_3010.pkl"
    saved = pickle.loads(state.read_bytes()) if state.exists() else {"next": 0}
    ns, start = {"REPORT": {}}, 0
    if saved["next"] > 0:
        start, ns = saved["next"], saved["values"]
        log.write(saved["log"])
        for src in cells[:start]:                     # функции и импорты уже выполненных ячеек
            tree = ast.parse(src)
            tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))]
            exec(compile(tree, "defs", "exec"), ns)
    t0 = time.time()
    for i in range(start, len(cells)):
        log.write(f"\n===== 30.10, ячейка {i}\n")
        with contextlib.redirect_stdout(log):
            exec(compile(cells[i], f"30.10[{i}]", "exec"), ns)
        if CHECKPOINT_SEC and time.time() - t0 > CHECKPOINT_SEC and i < len(cells) - 1:
            keep = {}
            for k, v in ns.items():
                if k.startswith("__") or isinstance(v, (_t.FunctionType, _t.ModuleType, type)):
                    continue
                try:
                    pickle.dumps(v); keep[k] = v
                except Exception:
                    pass
            state.write_bytes(pickle.dumps(dict(next=i + 1, values=keep, log=log.getvalue())))
            print(f"30.10: выполнены ячейки до {i}, состояние сохранено — запустите скрипт ещё раз")
            sys.exit(3)
    if state.exists():
        state.write_bytes(pickle.dumps(dict(next=0)))
    return ns


# ----------------------------------------------------------------------------- 2. чистка трасс
def clean_traces(cust, cc, report):
    """Правки 3, 4, 6. cust - custody_linked после 30.10 (с колонкой start_state)."""
    c = cust.copy()
    n0, p0 = len(c), c["person_id"].nunique()
    # правка 6: конец наблюдения
    late = c["in_dt"] >= STUDY_END
    report["конец наблюдения: заключений с входом после 01.04.2016 (убраны)"] = int(late.sum())
    c = c[~late].copy()
    after = c["out_dt"] >= STUDY_END
    report["конец наблюдения: выход после 01.04.2016 -> «ещё сидит»"] = int((after & (c["out_dt"] < PLACEHOLDER)).sum())
    c.loc[after, "out_dt"] = PLACEHOLDER
    # правка 3: начальные prison без выхода до конца наблюдения -> трасса не рассматривается
    st = c[c["start_state"]]
    no_exit = set(st.loc[st["out_dt"] >= STUDY_END, "person_id"])
    report["начальные prison без выхода до 01.04.2016: людей (трасса убрана)"] = len(no_exit)
    report["  их заключений"] = int(c["person_id"].isin(no_exit).sum())
    c = c[~c["person_id"].isin(no_exit)]
    # правка 4: трассы «только prison» (в окне нет ни одного jail)
    with_jail = set(c.loc[c["source"] == "jail", "person_id"])
    only_p = set(c["person_id"]) - with_jail
    report["трассы «только prison»: людей (убраны)"] = len(only_p)
    report["  их сроков prison"] = int(c["person_id"].isin(only_p).sum())
    report["  из них начальных prison"] = int((c["person_id"].isin(only_p) & c["start_state"]).sum())
    c = c[~c["person_id"].isin(only_p)].reset_index(drop=True)
    report["заключений: до чистки / после"] = f"{n0} / {len(c)}"
    report["людей: до чистки / после"] = f"{p0} / {c['person_id'].nunique()}"
    report["из них начальных prison (не узлы графа)"] = int(c["start_state"].sum())
    cc = cc[cc["custody_id"].isin(c["custody_id"])]
    return c, cc


# ----------------------------------------------------------------------------- 2a. поля старого дела
def mark_old_case(cust, ccf):
    """Заметка «Логика build_graph», 4.1: для заключений со старым делом - пометки по обвинениям и полям заключения.
    ccf.old_case        - обвинение относится к старому делу этого заключения (дело было в прошлом заключении);
    ccf.new_in_old_case - обвинение старого дела с датой позже входа в последнее прошлое заключение по делу;
    cust.rearrest_case_types - типы обвинений старого дела (отдельное поле, не состояние узла)."""
    cin = cust.set_index("custody_id")["in_dt"].dt.normalize()
    pairs = []                                          # (custody_id, дело, вход в последнее прошлое заключение по делу)
    for r in cust[cust["rearrest_prev_custody_ids"].notna()].itertuples():
        for part in str(r.rearrest_prev_custody_ids).split("; "):
            cn, ids = part.split(": ")
            prev = [cin.get(int(float(i))) for i in ids.split(", ")]
            prev = [d for d in prev if pd.notna(d)]
            pairs.append((r.custody_id, cn, max(prev) if prev else pd.NaT))
    old = pd.DataFrame(pairs, columns=["custody_id", "case_number", "prev_case_in"])
    ccf = ccf.drop(columns=[c for c in ("old_case", "new_in_old_case", "prev_case_in") if c in ccf.columns])
    ccf = ccf.merge(old, on=["custody_id", "case_number"], how="left")
    ccf["old_case"] = ccf["prev_case_in"].notna() & ccf["charge_id"].notna() & ~ccf["inherited"].eq(True)
    cd = pd.to_datetime(ccf["charge_date"], errors="coerce")
    ccf["new_in_old_case"] = ccf["old_case"] & (cd > ccf["prev_case_in"])
    ccf = ccf.drop(columns="prev_case_in")
    types = (ccf[ccf["old_case"]].groupby("custody_id")["charge_category"]
             .agg(lambda s: ", ".join(sorted(s.dropna().unique()))))
    cust = cust.copy()
    cust["rearrest_case_types"] = cust["custody_id"].map(types)
    return cust, ccf


# ----------------------------------------------------------------------------- 3. эпизоды и состояния (код этапа 1)
def nb1_blocks():
    src = MAKE_NB1.read_text(encoding="utf-8")
    blocks = re.findall(r"code\(r'''\n?(.*?)'''\)", src, flags=re.S)
    pick = lambda key: next(b for b in blocks if key in b)
    build = pick("def build_log")
    build = patch(build, "    obs_end = real.max()", "    obs_end = OBS_END                                   # правка 6: 01.04.2016")
    build = patch(build, 'source="все по новой/custody_linked.csv + custody_charges.csv"', 'source="stage2_custody/данные/custody_linked.csv + custody_charges.csv"')
    build = build.split("\nt0 = time.time()")[0]          # только функции, вызов - ниже
    return [pick("PLACEHOLDER_OUT = pd.Timestamp"), pick("def episodes_from_custody").split("\ncust = episodes_from_custody")[0], build]


def stage1_namespace():
    import sqlite3, time
    ns = dict(pd=pd, np=np, sqlite3=sqlite3, time=time, DB=DB, OUT=OUT, OLDCASE_BY_TYPE=False, OBS_END=STUDY_END,
              base=types.SimpleNamespace(SEED=42, START="START"))
    sys.path.insert(0, str(VENDOR))
    from crime_categories import categorize_fine, UNKNOWN
    ns.update(categorize=categorize_fine, UNKNOWN=UNKNOWN)
    for b in nb1_blocks():
        exec(compile(b, "stage1", "exec"), ns)
    return ns


def build_event_table(ns, cust, chg, chg_all):
    """chg - обвинения для состояния узла (без обвинений старого дела: состояние смешанного узла - типы нового дела,
    узел только со старым делом - OldCase); chg_all - все привязанные обвинения (для тяжести узла)."""
    c = ns["episodes_from_custody"](cust)
    df, ev, info = ns["build_log"](c, chg)
    # самая тяжёлая степень обвинения эпизода (для признака «ступень тяжести», правка 11), по всем обвинениям узла
    ch = chg_all.merge(c[["custody_id", "ep_id"]], on="custody_id")
    ch["rank"] = ch["charge_degree"].map(ns["degree_rank"])
    best = ch.groupby("ep_id")["rank"].min()
    df["deg_rank"] = df["ep_row"].map(best).fillna(9).astype(int)       # 1 F1 … 7 MO3, 8 прочее / (0), 9 нет обвинений
    flags = c.groupby("ep_id").agg(start_state=("start_state", "max"))
    df = df.join(flags, on="ep_row")
    # поля старого дела узла (заметка «Логика build_graph», 4.1)
    join = lambda s: "; ".join(s.dropna().astype(str)) or None
    old = c.groupby("ep_id").agg(rearrest_case_ids=("rearrest_case_ids", join),
                                 rearrest_prev_custody_ids=("rearrest_prev_custody_ids", join),
                                 rearrest_case_types=("rearrest_case_types", join),
                                 rearrest_new_charges_n=("rearrest_new_charges_n", "sum"))
    df = df.join(old, on="ep_row")
    df["old_case_flag"] = df["rearrest_case_ids"].notna()
    df["start_state"] = df["start_state"].fillna(False).astype(bool)
    df["in_custody_true"] = df["in_custody"]
    out = df.drop(columns=["types", "next_types"]).assign(types=df["combo"])
    return out, info


def main():
    OUT.mkdir(exist_ok=True)
    buf = io.StringIO()
    cwd = os.getcwd(); os.chdir(OUT)                  # 30.10 пишет csv в текущую папку
    try:
        ns = run_3010(buf)
    finally:
        os.chdir(cwd)
    report = dict(ns["REPORT"])
    cust = pd.read_csv(OUT / "custody_linked.csv", parse_dates=["in_dt", "out_dt"], low_memory=False)
    ccf = pd.read_csv(OUT / "custody_charges.csv", low_memory=False)
    cust, ccf = clean_traces(cust, ccf, report)
    cust, ccf = mark_old_case(cust, ccf)
    report["4.1: заключений со старым делом (OldCase + смешанные)"] = int(cust["rearrest_case_ids"].notna().sum())
    report["4.1:   из них только старое дело (OldCase)"] = int((cust["reason_type"] == "rearrest_old_case").sum())
    report["4.1:   с новыми обвинениями по старому делу"] = int((cust["rearrest_new_charges_n"].fillna(0) > 0).sum())
    starts = cust[cust["start_state"]]
    graph = cust[~cust["start_state"]]
    graph.to_csv(OUT / "custody_linked.csv", index=False, encoding="utf-8-sig")
    ccf[ccf["custody_id"].isin(graph["custody_id"])].to_csv(OUT / "custody_charges.csv", index=False, encoding="utf-8-sig")
    starts.to_csv(OUT / "start_prisons.csv", index=False, encoding="utf-8-sig")

    cc = ccf[["custody_id", "charge_id", "inherited", "case_number", "charge", "charge_degree", "old_case"]]
    chg_all = cc[cc["charge_id"].notna() & ~cc["inherited"].eq(True)].copy()
    chg = chg_all[~chg_all["old_case"]]                                 # для состояния - без обвинений старого дела
    s1 = stage1_namespace()
    with contextlib.redirect_stdout(buf):
        et, info = build_event_table(s1, graph, chg, chg_all)            # граф этапа 1
        et2, _ = build_event_table(s1, cust, chg, chg_all)               # + начальные prison для этапа 2
    et.to_csv(OUT / "event_table.csv", index=False)
    et2.to_csv(OUT / "event_table_stage2.csv", index=False)
    report["граф: узлов (эпизодов заключения)"] = int(info["episodes"])
    report["граф: людей"] = int(info["people"])
    report["граф: конец наблюдения"] = info["obs_end"]
    report["граф: OldCase / Unknown эпизодов"] = f"{info['episodes_oldcase']} / {info['episodes_unknown']}"
    pd.Series(report, name="значение").to_csv(OUT / "cleaning_report.csv", encoding="utf-8-sig")
    pd.Series(info).to_csv(OUT / "log_info.csv", header=["value"])
    (OUT / "build_log.txt").write_text(buf.getvalue(), encoding="utf-8")
    print(pd.Series(report).to_string())


if __name__ == "__main__":
    main()
