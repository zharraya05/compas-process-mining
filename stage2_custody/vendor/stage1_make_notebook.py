"""Генератор ноутбука этапа 1 под таблицы ноутбука «все по новой/30.10.ipynb»."""
import json
import sys
import uuid

CELLS = []
# вариант: "common" — один общий OldCase; "by_type" — OldCase свой для каждого типа преступления
# "with_charges" — общий OldCase + события-обвинения, не привязанные ни к одному заключению;
#                  качество — стандартными функциями PM4Py
VARIANT = sys.argv[2] if len(sys.argv) > 2 else "common"
BY_TYPE = VARIANT == "by_type"
WITH_CH = VARIANT == "with_charges"
OUTDIR = {"common": "notebook_out", "by_type": "notebook_out_oldcase_by_type",
          "with_charges": "notebook_out_with_charges"}[VARIANT]
SUBST = {"{{OLDCASE_BY_TYPE}}": str(BY_TYPE), "{{ADD_UNLINKED}}": str(WITH_CH), "{{OUTDIR}}": OUTDIR,
         "{{EVAL_IM}}": str(WITH_CH)}   # стандартный token replay для IM быстрый, новая функция — нет


def sub(s):
    for k, v in SUBST.items():
        s = s.replace(k, v)
    return s


def md(s):
    CELLS.append(("markdown", sub(s.strip("\n"))))


def code(s):
    CELLS.append(("code", sub(s.strip("\n"))))


if BY_TYPE:
    md(r'''
# Этап 1 (п. 1.1 плана). Граф переходов по привязке 30.10: OldCase свой для каждого типа

Вариант ноутбука `stage1_30_10.ipynb` (там один общий `OldCase`). Здесь заключение по **повторному аресту по старому делу** (`reason_type = rearrest_old_case`, в 30.10 обвинения к нему не привязаны) получает состояние **`OldCase: <тип>`**, где тип — тип преступления **исходного дела**: по номерам дел из `rearrest_case_ids` берутся обвинения этих дел из `compas.db` и переводятся в 15 типов. Если в старом деле несколько типов, у эпизода несколько состояний (`OldCase: Drug`, `OldCase: Property`) — это развилка, как у обычных эпизодов.

| Заключение | Состояние эпизода |
|---|---|
| есть привязанные обвинения | множество 15 типов этих обвинений |
| `rearrest_old_case` | `OldCase: <тип исходного дела>` (по каждому типу дела) |
| не привязано | `Unknown` |

Так граф различает «вернулся по старому делу о наркотиках» и «вернулся по старому делу о краже», а тяжесть старого дела по-прежнему не выдаётся за новое преступление. Остальное как в `stage1_30_10.ipynb`; результаты пишутся в `notebook_out_oldcase_by_type/`, в шаге 9 — сравнение с вариантом с общим `OldCase`.
''')

    md(r'''
**Как запустить.** Сначала `все по новой/30.10.ipynb`, затем этот ноутбук, *Run → Run All Cells*. Для сравнения в шаге 9 нужен уже выполненный `stage1_30_10.ipynb` (его `notebook_out/`).
''')

if WITH_CH:
    md(r'''
# Этап 1 (п. 1.1 плана). Граф переходов: заключения 30.10 + неучтённые обвинения

Вариант ноутбука `stage1_30_10.ipynb` (общий `OldCase`). К таблице заключений из `все по новой` добавляются **обвинения, которые не привязаны ни к одному заключению** (в `charge` они есть, а в `custody_charges.csv` — нет: человек был обвинён, но в jail / prison по этому обвинению не попал, или заключение не нашлось). Каждое такое обвинение становится событием в трассе человека:

- берутся только люди, у которых **уже есть заключения** в таблице 30.10, и только обвинения внутри периода наблюдения (с 01.01.2013 до конца наблюдения);
- обвинения одного человека с одной датой — один **эпизод без заключения** (`source = charge`, `reason = charge_only`, длительность 0 дней), состояние — множество 15 типов его обвинений;
- флаг `DROP_INFRACTIONS` убирает гражданские нарушения (тяжесть `(0)`: штрафы ПДД и т.п.), по умолчанию они остаются.

| Эпизод | Состояние |
|---|---|
| заключение с привязанными обвинениями | множество 15 типов этих обвинений |
| заключение `rearrest_old_case` | `OldCase` |
| заключение без привязки | `Unknown` |
| **обвинение без заключения (новое)** | множество 15 типов обвинений этого дня |

Качество моделей считается **стандартными функциями PM4Py**: fitness — token replay (`fitness_token_based_replay`), precision — ETConformance (`precision_token_based_replay`), как в ноутбуках `stage1/`. Результаты — `notebook_out_with_charges/`; в шаге 9 — сравнение с вариантом без добавленных обвинений.

**Как запустить.** Сначала `все по новой/30.10.ipynb`, затем этот ноутбук, *Run → Run All Cells*. Для сравнения в шаге 9 нужен выполненный `stage1_30_10.ipynb`.
''')

if not BY_TYPE and not WITH_CH:
    md(r'''
# Этап 1 (п. 1.1 плана). Граф переходов по привязке 30.10

Ноутбук строит граф переходов между типами преступлений по таблицам ноутбука `все по новой/30.10.ipynb` шаг за шагом: эпизоды → журнал событий PM4Py → матрица переходов → графы (DFG, Heuristics Miner, Inductive Miner) → от чего зависит следующее состояние → качество моделей. Структура та же, что у `stage1/all_new/stage1_all_new.ipynb`; качество считается **новой функцией** `time_aware_quality` (папка `meeting17_fixes`), рядом для сравнения — стандартные метрики PM4Py.

**Чем эти данные отличаются от этапа 0 (`compas_custody_charges.csv`).** Привязка обвинений к заключениям сделана заново (этапы 1–7 в 30.10):
- одно обвинение привязано только к одному заключению;
- заключение, начавшееся в день **повторного ареста по старому делу** (ордер, нарушение условий, неявка), обвинения старого дела **не получает** (`reason_type = rearrest_old_case`);
- переводы jail → prison помечены (`transfer_note`), prison до 2013 года вынесены в отдельный файл.

**Правило состояний этого ноутбука.**
| Заключение | Состояние эпизода |
|---|---|
| есть привязанные обвинения | множество 15 типов этих обвинений (`categorize_fine`) |
| `rearrest_old_case` (повторный арест, обвинения не привязаны) | `OldCase` |
| не привязано | `Unknown` |

Перевод jail → prison склеивается с исходным jail в один эпизод; Justice идёт отдельным шагом после остальных типов эпизода (как в `all_new`); последний эпизод человека цензурируется концом наблюдения; модели учатся на 80% людей и проверяются на 20% (seed 42).

**Как запустить.** `D:\Git\diploma\.venv\Scripts\activate`, `jupyter lab` в `D:\diploma\jupyter lab`, открыть `этап 1/stage1_30_10.ipynb`, *Run → Run All Cells*. Перед этим должен быть запущен `все по новой/30.10.ipynb` (он пишет `custody_linked.csv` и `custody_charges.csv`). Всё, что считает ноутбук, пишется в `notebook_out/`.
''')

md(r'''
## Шаг 0. Настройки

- `IN` — папка с таблицами ноутбука 30.10 (`custody_linked.csv` — заключения, `custody_charges.csv` — заключение × обвинение);
- `DB` — `compas.db` (пол, раса, дата рождения, балл COMPAS);
- `OUT` — папка, куда ноутбук пишет результаты (`{{OUTDIR}}/`);
- `OLDCASE_BY_TYPE` — `False`: один общий `OldCase`; `True`: `OldCase: <тип исходного дела>`;
- `RUN_MODELS` — строить сети Петри (IM, HM, DFG) на train, меньше минуты;
- `RUN_EVAL` — считать качество моделей из `EVAL_MODELS` (стандарт PM4Py + новая функция, секунды); `EVAL_IM = True` добавляет Inductive Miner 0.2 (около 20 минут). Модели, которые не пересчитывались, берутся из сохранённого `notebook_out/model_quality.csv`.

Окружение: `D:\Git\diploma\.venv` (pm4py 2.7, pandas, scikit-learn, graphviz). Если `dot` не найден, но есть `D:\Graphviz\bin`, папка добавляется в PATH.
''')

code(r'''
import os, sys, time, shutil, warnings
from pathlib import Path

HERE = Path.cwd()                                     # папка ноутбука
IN = HERE / ".." / "все по новой"                     # выход ноутбука 30.10
DB = HERE / ".." / "stage0 - data" / "compas.db"
OUT = HERE / "{{OUTDIR}}"
OUT.mkdir(exist_ok=True)

OLDCASE_BY_TYPE = {{OLDCASE_BY_TYPE}}   # True — OldCase свой для каждого типа исходного дела
ADD_UNLINKED = {{ADD_UNLINKED}}         # True — добавить в трассы обвинения без заключения
DROP_INFRACTIONS = False      # True — не добавлять гражданские нарушения (тяжесть "(0)")
RUN_MODELS = True
RUN_EVAL = True
EVAL_MODELS = ["HM_dep0.9", "DFG_p>=0.1"]   # секунды
EVAL_IM = {{EVAL_IM}}                              # + IM_noise0.2 (новой функцией около 20 минут)

if shutil.which("dot") is None and Path(r"D:\Graphviz\bin").exists():
    os.environ["PATH"] = r"D:\Graphviz\bin" + os.pathsep + os.environ["PATH"]

import sqlite3
import numpy as np
import pandas as pd
import pm4py
from IPython.display import Image, display

warnings.filterwarnings("ignore")
sys.path.insert(0, str(HERE / "code"))
import stage1_common as base            # рисунок матрицы, V Крамера
from crime_categories import categorize_fine as categorize, UNKNOWN
from time_aware_quality import blocks_from_log, time_aware_fitness, time_aware_precision, time_aware_quality

pd.set_option("display.max_columns", 40, "display.width", 200)
print("pm4py", pm4py.__version__, "| pandas", pd.__version__)
for f in ["custody_linked.csv", "custody_charges.csv"]:
    print(f, "| есть:", (IN / f).exists())
print("Graphviz:", "есть" if shutil.which("dot") else "НЕТ (рисунки графов не строятся)")


def show(path, width=None, max_px=1400):
    # показать рисунок; большие уменьшаются до max_px по длинной стороне (полный размер — в файле)
    from io import BytesIO
    from PIL import Image as PILImage
    im = PILImage.open(path)
    if max(im.size) > max_px:
        im.thumbnail((max_px, max_px))
    buf = BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=85)
    display(Image(data=buf.getvalue(), format="jpeg", width=width))


def graphviz_ok():
    return shutil.which("dot") is not None
''')

md(r'''
Константы (те же, что в `all_new`):
- `PLACEHOLDER_OUT = 2020-01-01` — выход «ещё в заключении», заменяется на конец наблюдения;
- `DEGREE_RANK` — тяжесть обвинения (F1 самое тяжёлое … MO3), для класса Felony / Misdemeanor;
- `DUR_BINS` — классы длительности заключения.
''')

code(r'''
SEED = base.SEED          # 42
START = base.START        # "START" — фиктивное начальное состояние
CASE = "id"               # объект процесса = человек (person_id)
OLDCASE = "OldCase"
JUSTICE = "Justice"
PLACEHOLDER_OUT = pd.Timestamp("2020-01-01")
DEGREE_RANK = {"F1": 1, "F2": 2, "F3": 3, "F5": 4, "F6": 4, "F7": 4, "M1": 5, "M2": 6, "MO3": 7}
DEG = {1: "F", 2: "F", 3: "F", 4: "F", 5: "M", 6: "M", 7: "M"}
DUR_BINS = [-np.inf, 1, 30, 180, np.inf]
DUR_LABELS = ["S1_<=1d", "S2_2-30d", "S3_31-180d", "S4_>180d"]


def degree_rank(d):
    return DEGREE_RANK.get(str(d).strip("()"), 8)
''')

md(r'''
## Шаг 1. Входные данные и 15 типов преступлений

- `custody_linked.csv` — одна строка = одно заключение (jail или prison с 2013 года): `link_stage` (этап привязки 1–7), `reason_type` (тип причины), `transfer_note` (перевод jail → prison);
- `custody_charges.csv` — одна строка = заключение × привязанное обвинение (`charge_id`, текст `charge`, тяжесть `charge_degree`, номер дела). Строки с `inherited = True` — обвинения исходного jail, показанные у перевода в prison; здесь они не нужны, потому что перевод склеивается с jail.

`crime_categories.categorize_fine` переводит текст обвинения в один из 15 типов по регулярным выражениям (срабатывает первое совпавшее правило): Violent, Property, Drug, Traffic, Weapon, Fraud, Sex, NoCharge, ResistPolice, PublicOrder, Tampering, Justice, Conspiracy, Minors, Other; пустой текст — Unknown.
''')

code(r'''
cust = pd.read_csv(IN / "custody_linked.csv", parse_dates=["in_dt", "out_dt"], low_memory=False)
cc = pd.read_csv(IN / "custody_charges.csv", low_memory=False,
                 usecols=["custody_id", "charge_id", "inherited", "case_number", "charge", "charge_degree"])
chg = cc[cc["charge_id"].notna() & ~cc["inherited"].fillna(False).astype(bool)].copy()
print(f"заключений: {len(cust):,}; людей: {cust['person_id'].nunique():,}; привязанных обвинений: {len(chg):,}")
pd.crosstab(cust["reason_type"].fillna("не привязано"), cust["source"], margins=True)
''')

code(r'''
desc = chg["charge"].value_counts()
mapping = pd.DataFrame({"rows": desc, "type": [categorize(d) for d in desc.index]})
by_type = mapping.groupby("type")["rows"].agg(charges="sum", wordings="size").sort_values("charges", ascending=False)
by_type["share"] = (by_type["charges"] / by_type["charges"].sum()).round(3)
print("разных формулировок:", len(mapping))
by_type
''')

md(r'''
## Шаг 2. От заключений к эпизодам

### 2.1. Склейка переводов jail → prison
Перевод из jail в prison (в 30.10 найден по правилу «разрыв меньше суток и нет новых арестов и обвинений», номер исходного jail — в `transfer_note`) — продолжение того же заключения. Такой prison склеивается с исходным jail в один **эпизод**: вход — из jail, выход — из prison.
''')

code(r'''
def episodes_from_custody(cust):
    c = cust.copy()
    src = c["transfer_note"].where(c["transfer_note"].str.startswith("перевод", na=False)) \
                            .str.extract(r"custody_id=(\d+)")[0].astype(float)
    c["ep_id"] = c["custody_id"]
    ok = src.notna() & src.isin(c["custody_id"])            # исходный jail есть в таблице
    c.loc[ok, "ep_id"] = src[ok].astype(int)
    c["merged_into_prev"] = ok
    return c


cust = episodes_from_custody(cust)
print("склеено переводов jail -> prison:", int(cust["merged_into_prev"].sum()),
      "| эпизодов:", cust["ep_id"].nunique())
''')

if WITH_CH:
    md(r'''
### 2.1a. Неучтённые обвинения → эпизоды без заключения

1. Все обвинения из `compas.db` (`charge`), без точных дубликатов (как в 30.10).
2. Убираются обвинения, уже привязанные к какому-нибудь заключению: `charge_id` из `custody_charges.csv` и из `prison_before_2013_charges.csv`.
3. Остаются люди, у которых есть заключения в таблице 30.10, и даты обвинения от 01.01.2013 до последней даты заключений.
4. Обвинения человека с одной датой — один эпизод без заключения: строка в `cust` (`source = charge`, вход = выход = дата обвинения) и его обвинения в `chg`. Дальше `build_log` обрабатывает их так же, как заключения.
''')
    code(r'''
def unlinked_charge_episodes(cust, cc):
    with sqlite3.connect(DB) as con:
        allch = pd.read_sql("select id, person_id, case_number, charge, charge_degree, date_charge_filed from charge", con)
    allch = allch[~allch.drop(columns="id").duplicated()]
    allch["charge_date"] = pd.to_datetime(allch["date_charge_filed"], errors="coerce").dt.normalize()
    used = set(cc["charge_id"].dropna().astype(int))
    pre = IN / "prison_before_2013_charges.csv"
    if pre.exists():
        used |= set(pd.read_csv(pre, usecols=["charge_id"])["charge_id"].dropna().astype(int))
    period_end = cust.loc[cust["out_dt"] < PLACEHOLDER_OUT, "out_dt"].max().normalize()
    stats = {"обвинений в charge (без дублей)": len(allch),
             "из них уже привязаны к заключениям": int(allch["id"].isin(used).sum())}
    u = allch[~allch["id"].isin(used)]
    stats["не привязаны"] = len(u)
    u = u[u["person_id"].isin(set(cust["person_id"]))]
    stats["  у людей с заключениями 30.10"] = len(u)
    u = u[(u["charge_date"] >= pd.Timestamp("2013-01-01")) & (u["charge_date"] <= period_end)]
    stats["  и в периоде 2013 — " + str(period_end.date())] = len(u)
    infr = u["charge_degree"].astype(str).eq("(0)")
    stats["    из них гражданские нарушения (0)"] = int(infr.sum())
    if DROP_INFRACTIONS:
        u = u[~infr]
    stats["добавляется обвинений"] = len(u)
    # эпизод = человек + дата обвинения
    ep = (u.groupby(["person_id", "charge_date"], as_index=False).size()
            .rename(columns={"charge_date": "in_dt"}))
    ep["custody_id"] = np.arange(len(ep)) + int(cust["custody_id"].max()) + 1
    ep = ep.assign(out_dt=ep["in_dt"], source="charge", reason_type="charge_only", link_stage=np.nan,
                   transfer_note=np.nan, rearrest_case_ids=np.nan, merged_into_prev=False)
    ep["ep_id"] = ep["custody_id"]
    stats["добавляется эпизодов"] = len(ep)
    stats["  людей с добавленными эпизодами"] = ep["person_id"].nunique()
    new_chg = (u.merge(ep[["person_id", "in_dt", "custody_id"]], left_on=["person_id", "charge_date"],
                       right_on=["person_id", "in_dt"])
                 .rename(columns={"id": "charge_id"})[["custody_id", "charge_id", "case_number", "charge", "charge_degree"]]
                 .assign(inherited=False))
    return ep.drop(columns="size"), new_chg, stats


if ADD_UNLINKED:
    extra_ep, extra_chg, add_stats = unlinked_charge_episodes(cust, cc)
    # сколько добавленных эпизодов попадает внутрь заключения того же человека
    iv = cust[["person_id", "in_dt", "out_dt"]].merge(extra_ep[["person_id", "in_dt"]].rename(columns={"in_dt": "d"}), on="person_id")
    add_stats["  эпизодов внутри срока заключения"] = int(((iv["d"] > iv["in_dt"].dt.normalize()) & (iv["d"] < iv["out_dt"])).groupby([iv["person_id"], iv["d"]]).any().sum())
    cust = pd.concat([cust, extra_ep], ignore_index=True)
    chg = pd.concat([chg, extra_chg], ignore_index=True)
    print(pd.Series(add_stats).to_string())
''')
    code(r'''
# типы добавленных обвинений
if ADD_UNLINKED:
    t_add = extra_chg["charge"].map(lambda d: categorize(d) if pd.notna(d) else UNKNOWN).value_counts()
    display(pd.DataFrame({"обвинений": t_add, "доля": (t_add / t_add.sum()).round(3)}))
''')

md(r'''
### 2.2. Состояние эпизода

Главная функция `build_log`. По шагам:
1. **Конец наблюдения** `obs_end` — последняя реальная дата в данных; у тех, кто ещё сидит (`out = 2020-01-01`), выход = `obs_end`.
2. **Признаки человека** из `compas.db`: пол, раса, возраст на дату входа, балл COMPAS.
3. **Состояние эпизода** = множество типов его обвинений; Unknown убирается, если в эпизоде есть хоть один известный тип. Эпизод без обвинений: если это повторный арест по старому делу (`rearrest_old_case`) — `OldCase` (при `OLDCASE_BY_TYPE = True` — `OldCase: <тип>` по каждому типу обвинений исходного дела из `compas.db`), иначе `Unknown`.
4. **Justice** (побег, неявка в суд, нарушение условий…) — следствие заключения, поэтому эпизод `{A, B, Justice}` делится на два шага `{A, B} → {Justice}` с интервалом 0 дней.
5. Для каждого шага: номер события у человека, интервал до следующего эпизода `gap_next`, следующее состояние `next_types`, цензурирование (последний шаг человека).

`ev` — таблица «одна строка = один тип в шаге» (из неё делается журнал PM4Py), `info` — сводка.
''')

code(r'''
def people_features(db):
    with sqlite3.connect(db) as con:
        p = pd.read_sql("select id, sex, race, dob, decile_score from people", con)
    p["dob"] = pd.to_datetime(p["dob"], errors="coerce")
    return p.rename(columns={"id": CASE})


def oldcase_types(cust):
    """ep_id -> ["OldCase: <тип>", ...]: типы обвинений исходного дела повторного ареста
    (номера дел — rearrest_case_ids, обвинения дел — таблица charge в compas.db)."""
    r = cust.loc[cust["reason_type"] == "rearrest_old_case", ["ep_id", "person_id", "rearrest_case_ids"]].dropna()
    r = r.assign(case_number=r["rearrest_case_ids"].str.split(", ")).explode("case_number")
    with sqlite3.connect(DB) as con:
        allch = pd.read_sql("select person_id, case_number, charge from charge", con)
    m = r.merge(allch, on=["person_id", "case_number"], how="left")
    m["type"] = m["charge"].map(lambda d: categorize(d) if pd.notna(d) else UNKNOWN)
    known = m[m["type"] != UNKNOWN].groupby("ep_id").size()
    m = m[~((m["type"] == UNKNOWN) & m["ep_id"].isin(known.index))]
    return m.groupby("ep_id")["type"].apply(lambda s: sorted({f"{OLDCASE}: {t}" for t in s})).to_dict()


def is_oldcase(s):
    return str(s).startswith(OLDCASE)


def build_log(cust, chg):
    real = pd.concat([cust["in_dt"], cust.loc[cust["out_dt"] < PLACEHOLDER_OUT, "out_dt"]])
    obs_end = real.max()
    c = cust.assign(ongoing=cust["out_dt"] >= PLACEHOLDER_OUT)
    c.loc[c["ongoing"], "out_dt"] = obs_end
    c = c.sort_values(["person_id", "in_dt"])
    df = (c.groupby("ep_id", as_index=False)
           .agg(id=("person_id", "first"), in_custody=("in_dt", "min"), out_custody=("out_dt", "max"),
                ongoing=("ongoing", "max"), custodies=("custody_id", "size"),
                sources=("source", lambda s: "+".join(s)),
                reason=("reason_type", lambda s: s.dropna().iloc[0] if s.notna().any() else "не привязано"),
                link_stage=("link_stage", "first")))
    df = df.rename(columns={"ep_id": "ep_row"})
    pf = people_features(DB)
    df = df.merge(pf, on=CASE, how="left")
    df["age"] = ((df["in_custody"] - df["dob"]).dt.days / 365.25).round()
    # обвинения эпизода: тип и тяжесть
    ch = chg.merge(c[["custody_id", "ep_id"]], on="custody_id").rename(columns={"ep_id": "ep_row"})
    ch["type"] = ch["charge"].map(lambda d: categorize(d) if pd.notna(d) else UNKNOWN)
    ch["rank"] = ch["charge_degree"].map(degree_rank)
    t = ch.groupby(["ep_row", "type"], as_index=False).agg(rank=("rank", "min"), n_charges=("type", "size"))
    t["deg"] = t["rank"].map(DEG).fillna("NA")
    known = t[t["type"] != UNKNOWN].groupby("ep_row").size()
    t = t[~((t["type"] == UNKNOWN) & t["ep_row"].isin(known.index))]
    sets = t.groupby("ep_row")["type"].apply(sorted)
    # эпизод без обвинений: OldCase (повторный арест по старому делу) или Unknown
    is_old = cust.groupby("ep_id")["reason_type"].apply(lambda s: (s == "rearrest_old_case").any())
    old_types = oldcase_types(cust) if OLDCASE_BY_TYPE else {}

    def no_charge_state(e):
        if not is_old.get(e, False):
            return [UNKNOWN]
        return old_types.get(e, [f"{OLDCASE}: {UNKNOWN}"]) if OLDCASE_BY_TYPE else [OLDCASE]

    df["types"] = [sets[e] if e in sets.index else no_charge_state(e) for e in df["ep_row"]]
    df["n_types"] = df["types"].str.len()
    df["combo"] = df["types"].str.join("+")
    best = t.groupby("ep_row")["rank"].min()
    df["charge_class"] = df["ep_row"].map(best).map(lambda r: "Felony" if r <= 4 else "Misdemeanor" if r <= 7 else "Other/NA")
    df["dur"] = (df["out_custody"] - df["in_custody"]).dt.days.clip(lower=0)
    df["dur_class"] = pd.cut(df["dur"], DUR_BINS, labels=DUR_LABELS).astype(str)
    df["age_group"] = pd.cut(df["age"], [0, 25, 35, 45, 120], labels=["<25", "25-34", "35-44", "45+"]).astype(str)
    # Justice отдельным шагом после остальных типов
    df["step"] = 0
    split = df["types"].apply(lambda x: JUSTICE in x and len(x) > 1)
    jr = df[split].assign(types=[[JUSTICE]] * int(split.sum()), step=1)
    df.loc[split, "types"] = df.loc[split, "types"].apply(lambda x: [y for y in x if y != JUSTICE])
    df = pd.concat([df, jr], ignore_index=True)
    df["n_types"] = df["types"].str.len()
    df["combo"] = df["types"].str.join("+")
    # порядок, интервалы, следующее состояние, цензурирование
    df = df.sort_values([CASE, "in_custody", "ep_row", "step"]).reset_index(drop=True)
    df["event_idx"] = df.groupby(CASE).cumcount() + 1
    df["n_events"] = df.groupby(CASE)[CASE].transform("size")
    df["gap_prev"] = (df["in_custody"] - df.groupby(CASE)["out_custody"].shift()).dt.days
    df["gap_next"] = (df.groupby(CASE)["in_custody"].shift(-1) - df["out_custody"]).dt.days
    df["next_types"] = df.groupby(CASE)["types"].shift(-1)
    same_ep = df.groupby(CASE)["ep_row"].shift(-1) == df["ep_row"]
    df.loc[same_ep, "gap_next"] = 0
    df.loc[df["step"] == 1, "gap_prev"] = 0
    df["prev_combo"] = df.groupby(CASE)["combo"].shift(1).fillna("first")
    df["censored"] = df["event_idx"] == df["n_events"]
    df["followup_days"] = np.where(df["censored"], (obs_end - df["out_custody"]).dt.days, np.nan)
    ev = df[[CASE, "ep_row", "step", "event_idx", "in_custody", "types"]].explode("types").rename(columns={"types": "type"})
    ev = ev.merge(t[["ep_row", "type", "deg", "n_charges"]], on=["ep_row", "type"], how="left")
    ev["deg"] = ev["deg"].fillna("NA")
    ep0 = df[df["step"] == 0]
    info = dict(source="все по новой/custody_linked.csv + custody_charges.csv",
                charge_rows=len(ch), custodies_raw=len(cust),
                custodies_merged=int(cust["merged_into_prev"].sum()), episodes=len(ep0),
                justice_split_steps=int(df["step"].eq(1).sum()), steps=len(df),
                people=df[CASE].nunique(), case_id="id (person_id)", obs_end=str(obs_end.date()),
                ongoing_custody=int(ep0["ongoing"].sum()),
                people_repeat=int((df.groupby(CASE).size() > 1).sum()),
                transitions_observed=int(df["next_types"].notna().sum()),
                cases_censored=int(df["censored"].sum()),
                episodes_multi_type=int((df["n_types"] > 1).sum()),
                share_multi_type=round(float((df["n_types"] > 1).mean()), 3),
                type_events=len(ev), negative_gaps=int((df["gap_next"] < 0).sum()),
                episodes_oldcase=int(ep0["types"].apply(lambda x: any(is_oldcase(s) for s in x)).sum()),
                episodes_unknown=int((ep0["combo"] == UNKNOWN).sum()))
    return df, ev, info


t0 = time.time()
df, ev, info = build_log(cust, chg)
print(f"готово за {time.time() - t0:.0f} с")
pd.Series(info).to_csv(OUT / "log_info.csv", header=["value"])
pd.Series(info).to_frame("value")
''')

md(r'''
**Как читать.** `custodies_raw` — заключений в таблице 30.10, `custodies_merged` — склеено переводов jail → prison, `episodes` — эпизодов после склеек, `steps` — шагов с учётом отдельного Justice, `transitions_observed` — пар «шаг → следующий шаг» у одного человека, `share_multi_type` — доля развилок, `episodes_oldcase` / `episodes_unknown` — эпизоды без обвинений (повторный арест по старому делу / не привязано).
''')

if BY_TYPE:
    code(r'''
# эпизоды-повторные аресты по типу исходного дела (эпизод с делом из нескольких типов считается в каждом)
ep_old = df[(df["step"] == 0) & df["types"].apply(lambda x: any(is_oldcase(s) for s in x))]
oc = ep_old["types"].explode().value_counts()
print(f"эпизодов OldCase: {len(ep_old):,}; из них с несколькими типами исходного дела: {(ep_old['n_types'] > 1).sum():,}")
pd.DataFrame({"эпизодов": oc, "доля": (oc / len(ep_old)).round(3)})
''')

code(r'''
# таблица шагов (одна строка = один шаг человека)
df[[CASE, "event_idx", "n_events", "in_custody", "out_custody", "reason", "combo", "n_types", "dur", "gap_next", "next_types", "censored"]].head(12)
''')

code(r'''
# сколько шагов у человека
df.groupby(CASE).size().value_counts().sort_index().rename("людей").to_frame().head(10)
''')

md(r'''
## Шаг 3. Журнал событий для PM4Py

PM4Py работает с журналом «одна строка = одно событие» (`case:concept:name` — человек, `concept:name` — тип, `time:timestamp`). Развилка `{Drug, Property}` записывается двумя событиями одного эпизода; их порядок внутри эпизода случайный (seed 42), к времени добавляются секунды. Для новой функции качества это один **блок** (события с разницей меньше часа), порядок внутри которого не штрафуется.

Журнал и таблица шагов сохраняются в `notebook_out/` (`event_log.csv`, `event_table.csv`).
''')

code(r'''
def to_pm4py_log(ev):
    rng = np.random.default_rng(SEED)
    ev = ev.copy()
    ev["r"] = rng.random(len(ev))
    ev = ev.sort_values([CASE, "in_custody", "ep_row", "step", "r"])
    ev["ts"] = ev["in_custody"] + pd.to_timedelta(ev.groupby("ep_row").cumcount(), unit="s")
    ev = ev.rename(columns={"type": "activity"})[[CASE, "activity", "ts", "ep_row"]]
    ev[CASE] = ev[CASE].astype(str)
    return pm4py.format_dataframe(ev, case_id=CASE, activity_key="activity", timestamp_key="ts")


log = to_pm4py_log(ev)
log.to_csv(OUT / "event_log.csv", index=False)
df.drop(columns=["types", "next_types"]).assign(types=df["combo"]).to_csv(OUT / "event_table.csv", index=False)
print(f"событий: {len(log):,}; людей: {log['case:concept:name'].nunique():,}")

labels = ev["type"].value_counts().index.tolist()        # состояния по убыванию частоты
st = ev["type"].value_counts()
pd.DataFrame({"шагов с типом": st, "доля шагов": (st / len(df)).round(3)})
''')

code(r'''
# самые частые траектории людей (варианты процесса)
var = df.groupby(CASE)["combo"].agg(" > ".join).value_counts()
pd.DataFrame({"людей": var, "доля": (var / var.sum()).round(4)}).head(15)
''')

md(r'''
## Шаг 4. Матрица переходов

Переход = пара соседних шагов одного человека $S_k \to S_{k+1}$. Если состояния — множества (развилки), каждая пара типов $(a \in S_k,\ b \in S_{k+1})$ получает вес $1/(|S_k|\,|S_{k+1}|)$, чтобы один переход в сумме весил 1. Вероятность $p(a \to b)$ = сумма весов / сумма весов из $a$. Строка START — распределение первого эпизода. Последний шаг человека цензурирован и в вероятности не входит.

Ещё две таблицы: `contains` — доля переходов из $a$, у которых следующий эпизод *содержит* $b$ (без весов); медиана, 25% и 75% дней от выхода до следующего заключения по каждой дуге.
''')

code(r'''
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


wcounts, raw_pairs, probs, contains, arcs = transition_tables(df, labels)
probs.round(4).to_csv(OUT / "transition_probs.csv")
arcs.to_csv(OUT / "transition_arcs.csv", index=False)
raw_pairs.to_csv(OUT / "transition_pair_counts.csv")
contains.round(4).to_csv(OUT / "transition_contains.csv")
probs.round(2)
''')

code(r'''
med = arcs.set_index(["from", "to"])["median_days"].to_dict()
base.plot_matrix(probs, raw_pairs.drop(columns="CENSORED"), OUT / "transition_matrix.png", med)
show(OUT / "transition_matrix.png")
''')

md(r'''
**Как читать матрицу.** Строка — текущее состояние, столбец — следующее; в клетке вероятность, число пар `n` и медиана дней до следующего заключения. Диагональ — повтор того же типа (самопереход, петля на графе).
''')

code(r'''
# самопереходы и три самых вероятных перехода из каждого состояния
crime = [s for s in labels if not is_oldcase(s) and s != UNKNOWN]
loops = pd.Series(np.diag(probs.loc[labels, labels]), index=labels)
print(f"средний самопереход по типам преступлений: {loops[crime].mean():.3f}; "
      f"доля самопереходов среди всех переходов (по весу): {np.trace(wcounts.loc[labels, labels].values) / wcounts.loc[labels].values.sum():.3f}")
top3 = (arcs[arcs["from"] != START].sort_values("p", ascending=False).groupby("from").head(3)
        .assign(arc=lambda a: a["to"] + " " + a["p"].map("{:.2f}".format) + " (" + a["median_days"].map("{:.0f}".format) + " д)")
        .groupby("from")["arc"].agg(" | ".join))
pd.DataFrame({"самопереход": loops.round(3), "три самых вероятных": top3}).loc[labels]
''')

code(r'''
# матрица «следующий эпизод содержит тип» (без весов)
base.plot_matrix(contains.clip(upper=1), raw_pairs.drop(columns="CENSORED"), OUT / "transition_contains.png")
show(OUT / "transition_contains.png")
''')

md(r'''
## Шаг 5. Граф переходов и DFG

### 5.1. Граф переходов
Узлы — состояния, дуги с $p \ge$ порога, подпись — вероятность и медиана дней, толщина дуги пропорциональна $p$, самопереходы красные.
''')

code(r'''
def save_main_graph(arcs, path, min_p=0.05):
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
    g.render(str(path), cleanup=True)
    return Path(str(path) + ".png")


print("дуг с p >= 0.10:", int((arcs["p"] >= 0.10).sum()), "| с p >= 0.05:", int((arcs["p"] >= 0.05).sum()))
if graphviz_ok():
    show(save_main_graph(arcs, OUT / "transition_graph_p0.10", 0.10))
''')

md(r'''
С порогом 0.05 граф подробнее:
''')

code(r'''
if graphviz_ok():
    show(save_main_graph(arcs, OUT / "transition_graph_p0.05", 0.05))
''')

code(r'''
# самые частые дуги
arcs.sort_values("n_pairs", ascending=False).head(20)
''')

md(r'''
### 5.2. DFG PM4Py (directly-follows graph)
Стандартная картинка PM4Py по тем же парам эпизодов: на дугах число пар (frequency) или медиана времени (performance). Входящие стрелки — начало (первый эпизод), исходящие — цензурирование (последний эпизод).
''')

code(r'''
def save_episode_dfg(arcs, df, path, perf_path):
    a = arcs[arcs["from"] != START]
    dfg = {(r["from"], r["to"]): int(r["n_pairs"]) for _, r in a.iterrows()}
    sa = {k: int(v) for k, v in df[df["event_idx"] == 1].explode("types")["types"].value_counts().items()}
    ea = {k: int(v) for k, v in df[df["censored"]].explode("types")["types"].value_counts().items()}
    pm4py.save_vis_dfg(dfg, sa, ea, str(path))
    perf = {(r["from"], r["to"]): float(r["median_days"]) * 86400 for _, r in a.iterrows() if pd.notna(r["median_days"])}
    pm4py.save_vis_performance_dfg(perf, sa, ea, str(perf_path))


if graphviz_ok():
    save_episode_dfg(arcs, df, OUT / "dfg_frequency.png", OUT / "dfg_performance.png")
    show(OUT / "dfg_frequency.png", width=1100)
''')

code(r'''
if graphviz_ok():
    show(OUT / "dfg_performance.png", width=1100)
''')

md(r'''
## Шаг 6. Развилки (несколько типов в одном эпизоде)

Какие типы встречаются вместе. В матрице совместной встречаемости строка $a$, столбец $b$ — доля эпизодов с $a$, в которых есть и $b$. `share_alone` — доля эпизодов с типом, где он единственный.
''')

code(r'''
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


n_types, combos, co, co_p, solo = fork_tables(df, labels)
display(n_types.rename("шагов").to_frame().T)
display(combos.head(12))
solo
''')

code(r'''
base.plot_matrix(co_p.fillna(0), co, OUT / "fork_cooccurrence.png")
show(OUT / "fork_cooccurrence.png")
''')

md(r'''
## Шаг 7. От чего зависит следующее состояние

Для каждой пары (текущий тип → следующий тип) берутся признаки текущего эпизода: пол, возраст, раса, класс обвинения (Felony / Misdemeanor), длительность заключения, интервал до предыдущего, номер события, число типов, предыдущая комбинация и **тип причины задержания** из 30.10 (`reason`: новое обвинение, повторный арест по старому делу, перевод…).
- **V Крамера** — сила связи признака со следующим состоянием, отдельно для каждого текущего состояния;
- **дерево решений** (глубина 4) предсказывает следующий тип по всем признакам сразу; важность признака = вклад в разбиения.
''')

code(r'''
from sklearn.tree import DecisionTreeClassifier, export_text


def transition_drivers(df, labels):
    d, _ = pair_rows(df)
    feats = ["sex", "age_group", "race", "charge_class", "dur_class", "gap_prev_bin", "event_idx_bin",
             "n_types_bin", "prev_combo", "reason"]
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
    X = pd.get_dummies(d[["src"] + feats].astype(str), drop_first=False)
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=100, random_state=SEED,
                                  class_weight="balanced").fit(X, d["dst"], sample_weight=d["w"])
    imp = pd.Series(tree.feature_importances_, index=X.columns)
    imp = imp.groupby(lambda c: next(f for f in ["src"] + feats if c.startswith(f + "_"))).sum()
    return assoc, imp.sort_values(ascending=False).round(3), export_text(tree, feature_names=list(X.columns))


assoc, imp, tree_txt = transition_drivers(df, labels)
assoc.to_csv(OUT / "drivers_cramers_v.csv", index=False)
imp.to_csv(OUT / "drivers_tree_importance.csv", header=["importance"])
imp.rename("важность").to_frame()
''')

code(r'''
# самые сильные связи (V Крамера >= 0.15 при n >= 100)
assoc[(assoc["n"] >= 100) & (assoc["cramers_v"] >= 0.15)].sort_values("cramers_v", ascending=False).head(15)
''')

code(r'''
print(tree_txt[:3000])
''')

md(r'''
## Шаг 8. Модели процесса: Inductive Miner, Heuristics Miner, DFG

Модели учатся на журнале 80% людей (`train`), проверяются на 20% (`test`). Разбиение по людям — `split_people` (seed 42), то же, что в `meeting17_fixes/02_time_aware_quality.ipynb`.
- **Inductive Miner (IM)**, порог шума 0 / 0.2 / 0.5;
- **Heuristics Miner (HM)**, порог зависимости 0.5 / 0.9;
- **DFG с фильтром** — дуги с долей ≥ 5 / 10 / 20% от исходящих из узла, переведённые в сеть Петри.
''')

code(r'''
CASE_K, ACT_K, TIME_K = "case:concept:name", "concept:name", "time:timestamp"


def split_people(df, seed=SEED, test_share=0.2):
    ids = df[CASE_K].unique()
    test_ids = set(np.random.default_rng(seed).choice(ids, size=int(test_share * len(ids)), replace=False))
    return df[~df[CASE_K].isin(test_ids)], df[df[CASE_K].isin(test_ids)]


train_log, test_log = split_people(log)
print(f"train: {train_log[CASE_K].nunique():,} людей, test: {test_log[CASE_K].nunique():,} людей")

models = {}
if RUN_MODELS:
    t0 = time.time()
    for nt in [0.0, 0.2, 0.5]:
        models[f"IM_noise{nt}"] = pm4py.discover_petri_net_inductive(train_log, noise_threshold=nt)
    for dep in [0.5, 0.9]:
        models[f"HM_dep{dep}"] = pm4py.discover_petri_net_heuristics(train_log, dependency_threshold=dep)
    dfg_tr, sa_tr, ea_tr = pm4py.discover_dfg(train_log)
    out_tot = pd.Series(dfg_tr).groupby(level=0).sum()
    sa_tot = sum(sa_tr.values())
    for thr in [0.05, 0.10, 0.20]:
        dfg_f = {k: v for k, v in dfg_tr.items() if v / out_tot[k[0]] >= thr}
        sa_f = {k: v for k, v in sa_tr.items() if v / sa_tot >= thr}
        models[f"DFG_p>={thr}"] = pm4py.convert_to_petri_net(dfg_f, sa_f, ea_tr)
    print(f"моделей: {len(models)}, {time.time() - t0:.0f} с")
    display(pd.DataFrame({k: dict(places=len(n.places), transitions=len(n.transitions),
                                  silent=sum(t.label is None for t in n.transitions)) for k, (n, i, f) in models.items()}).T)
''')

if WITH_CH:
    md(r'''
### 8.1. Качество моделей на тесте: стандартные функции PM4Py

Базовые метрики PM4Py (как в ноутбуках `stage1/`):
- `fitness_tbr` — token replay (`pm4py.fitness_token_based_replay`, `log_fitness`): насколько трассы теста воспроизводятся сетью; дробный штраф за досозданные и оставшиеся фишки;
- `pct_fit_traces` — доля трасс, воспроизведённых без штрафа;
- `precision_etc` — ETConformance (`pm4py.precision_token_based_replay`): насколько модель не разрешает лишнего после каждого префикса;
- `f1` — гармоническое среднее fitness и precision.

Время PM4Py не учитывает: события одного эпизода (разница в секунды) и разных эпизодов для него одинаковы.
''')
    code(r'''
QUALITY_FILE = OUT / "model_quality.csv"

if RUN_EVAL and RUN_MODELS:
    rows = []
    for name in EVAL_MODELS + (["IM_noise0.2"] if EVAL_IM else []):
        net, im, fm = models[name]
        t0 = time.time()
        fit = pm4py.fitness_token_based_replay(test_log, net, im, fm)
        prec = pm4py.precision_token_based_replay(test_log, net, im, fm)
        f, p = fit["log_fitness"], prec
        rows.append(dict(model=name, fitness_tbr=round(f, 4),
                         pct_fit_traces=round(fit["percentage_of_fitting_traces"], 2),
                         precision_etc=round(p, 4), f1=round(2 * f * p / (f + p), 4) if f + p else 0.0,
                         places=len(net.places), transitions=len(net.transitions),
                         silent=sum(t.label is None for t in net.transitions),
                         sec=round(time.time() - t0, 1)))
        print(name, "готово за", rows[-1]["sec"], "с", flush=True)
    quality = pd.DataFrame(rows)
    quality.to_csv(QUALITY_FILE, index=False)
else:
    quality = pd.read_csv(QUALITY_FILE) if QUALITY_FILE.exists() else pd.DataFrame(columns=["model"])
    print("RUN_EVAL = False: таблица взята из", QUALITY_FILE.name if QUALITY_FILE.exists() else "— (её нет)")
quality
''')

if not WITH_CH:
    md(r'''
### 8.1. Качество моделей на тесте: новая функция `time_aware_quality`

Функция из `meeting17_fixes/time_aware_quality.py` (копия в `code/`). Отличия от стандарта PM4Py:

| | Стандарт PM4Py | Новая функция |
|---|---|---|
| Время | не используется | события с разницей < 1 часа = один **блок** (один эпизод) |
| Порядок внутри эпизода | штраф | **без штрафа** (перебор порядков блока) |
| Невыполнимое событие | дробный штраф через досозданные фишки | штраф **1 за событие**; fitness = 1 − невыполнимые / все |
| Активность, которой нет в модели | молча пропускается | штраф 1 |
| Конец трассы | штраф за оставшиеся фишки | без штрафа (цензурирование) |
| Precision | после каждого события | только между блоками; после последнего блока не считается |

Колонки таблицы:
- `fitness_tbr`, `precision_etc` — стандарт PM4Py (token replay, ETConformance);
- `fitness_ta`, `precision_ta`, `f1_ta` — новая функция, блоки 1 час, люди объединяются по всей истории (`prefix_key="full"`, как в классическом ETConformance);
- `precision_ta_marking` — новая precision с объединением по текущему состоянию сети (`prefix_key="marking"`);
- `fitness_noblocks`, `precision_noblocks` — новая функция без блоков (`gap_seconds=0.5`): сколько даёт само правило «нулевое время не штрафуем»;
- `missed_events` — сколько событий теста модель не может выполнить; `truncated` — сработало ограничение на число разметок (результат приближённый).
''')

    code(r'''
QUALITY_FILE = OUT / "model_quality.csv"

saved_q = pd.read_csv(QUALITY_FILE) if QUALITY_FILE.exists() else pd.DataFrame(columns=["model"])
if RUN_EVAL and RUN_MODELS:
    rows = []
    for name in EVAL_MODELS + (["IM_noise0.2"] if EVAL_IM else []):
        net, im, fm = models[name]
        t0 = time.time()
        q = time_aware_quality(test_log, net, im, fm)
        qm = time_aware_quality(test_log, net, im, fm, prefix_key="marking")
        q0 = time_aware_quality(test_log, net, im, fm, gap_seconds=0.5)
        rows.append(dict(model=name,
                         fitness_tbr=round(pm4py.fitness_token_based_replay(test_log, net, im, fm)["log_fitness"], 4),
                         precision_etc=round(pm4py.precision_token_based_replay(test_log, net, im, fm), 4),
                         fitness_ta=q["fitness_ta"], precision_ta=q["precision_ta"], f1_ta=q["f1_ta"],
                         precision_ta_marking=qm["precision_ta"],
                         fitness_noblocks=q0["fitness_ta"], precision_noblocks=q0["precision_ta"],
                         pct_fit_traces_ta=q["pct_fit_traces_ta"], missed_events=q["missed_events"],
                         events_not_in_model=q["events_not_in_model"],
                         truncated=q["truncated"] or qm["truncated"] or q0["truncated"],
                         places=len(net.places), transitions=len(net.transitions),
                         sec=round(time.time() - t0, 1)))
        print(name, "готово за", rows[-1]["sec"], "с", flush=True)
    new_q = pd.DataFrame(rows)
    # модели, которые сейчас не пересчитывались (например IM), остаются из сохранённой таблицы
    quality = pd.concat([saved_q[~saved_q["model"].isin(new_q["model"])], new_q], ignore_index=True)
    quality.to_csv(QUALITY_FILE, index=False)
else:
    quality = saved_q
    print("RUN_EVAL = False: таблица взята из", QUALITY_FILE.name if QUALITY_FILE.exists() else "— (её нет)")
quality
''')

    md(r'''
**Как читать.** Сравнение `precision_noblocks` и `precision_ta` (та же функция, разница только в блоках) показывает, сколько precision терялось из-за порядка типов внутри одного эпизода. `precision_ta_marking` — precision, когда «что наблюдалось дальше» собирается по состоянию сети, а не по всей уникальной истории человека.
''')

    code(r'''
# пример: какие трассы теста HM 0.9 не может выполнить целиком
if RUN_MODELS:
    lb = blocks_from_log(test_log)
    fit, per_trace = time_aware_fitness(lb, *models["HM_dep0.9"])
    print(fit)
    display(per_trace[per_trace["missed"] > 0].head())
''')

md(r'''
### 8.2. Heuristics Miner (dep 0.9): сеть Петри и heuristics net
''')

code(r'''
def petri_png(name):
    path = OUT / f"petri_{name}.png"
    pm4py.save_vis_petri_net(*models[name], str(path))
    return path


if RUN_MODELS and graphviz_ok():
    show(petri_png("HM_dep0.9"), width=1200)
''')

code(r'''
# на дугах heuristics net — число directly-follows и зависимость
if RUN_MODELS and graphviz_ok():
    pm4py.save_vis_heuristics_net(pm4py.discover_heuristics_net(train_log, dependency_threshold=0.9),
                                  str(OUT / "heuristics_net_dep0.9.png"))
    show(OUT / "heuristics_net_dep0.9.png", width=1200)
''')

md(r'''
### 8.3. Inductive Miner (noise 0.2): дерево процесса и сеть Петри
''')

code(r'''
if RUN_MODELS and graphviz_ok():
    pm4py.save_vis_process_tree(pm4py.discover_process_tree_inductive(train_log, noise_threshold=0.2),
                                str(OUT / "process_tree_IM_noise0.2.png"))
    show(OUT / "process_tree_IM_noise0.2.png", width=1100)
''')

code(r'''
if RUN_MODELS and graphviz_ok():
    show(petri_png("IM_noise0.2"), width=1200)
''')

md(r'''
## Шаг 9. Сравнение с прошлыми вариантами

Сохранённые результаты прошлых вариантов этапа 1 (данные этапа 0):
- число эпизодов, переходов, доля развилок, средний самопереход — из `stage1/<вариант>/results/tables`;
- качество новой функцией — из `meeting17_fixes/out/quality_time_aware.csv` (те же колонки, то же разбиение `split_people`).
''')

code(r'''
OTHERS = {"все новые (all_new)": "all_new/results", "OldCase": "oldcase/results", "без OldCase": "no_oldcase/results"}
rows = {}
for name, rel in OTHERS.items():
    r = HERE / ".." / "stage1" / rel
    if not (r / "tables" / "log_info.csv").exists():
        continue
    li = pd.read_csv(r / "tables" / "log_info.csv", index_col=0)["value"]
    pr = pd.read_csv(r / "tables" / "transition_probs.csv", index_col=0)
    tps = [c for c in pr.columns if c not in (OLDCASE, UNKNOWN) and c in pr.index]
    rows[name] = {"эпизодов": int(li["episodes"]), "шагов": int(li["steps"]),
                  "переходов": int(li["transitions_observed"]), "людей с повторами": int(li["people_repeat"]),
                  "доля развилок": float(li["share_multi_type"]),
                  "средний самопереход": round(float(np.mean([pr.loc[s, s] for s in tps])), 3)}
THIS = ("30.10 + неучтённые обвинения" if ADD_UNLINKED else
        "30.10, OldCase по типам" if OLDCASE_BY_TYPE else "30.10, общий OldCase")
COMMON_OUT = HERE / "notebook_out"                   # результаты stage1_30_10.ipynb (общий OldCase)
if (OLDCASE_BY_TYPE or ADD_UNLINKED) and (COMMON_OUT / "log_info.csv").exists():
    li = pd.read_csv(COMMON_OUT / "log_info.csv", index_col=0)["value"]
    pr = pd.read_csv(COMMON_OUT / "transition_probs.csv", index_col=0)
    tps = [c for c in pr.columns if not is_oldcase(c) and c != UNKNOWN and c in pr.index]
    rows["30.10, общий OldCase"] = {"эпизодов": int(li["episodes"]), "шагов": int(li["steps"]),
                                    "переходов": int(li["transitions_observed"]), "людей с повторами": int(li["people_repeat"]),
                                    "доля развилок": float(li["share_multi_type"]),
                                    "средний самопереход": round(float(np.mean([pr.loc[s, s] for s in tps])), 3)}
rows[THIS] = {"эпизодов": info["episodes"], "шагов": info["steps"],
              "переходов": info["transitions_observed"], "людей с повторами": info["people_repeat"],
              "доля развилок": info["share_multi_type"],
              "средний самопереход": round(float(loops[crime].mean()), 3)}
cmp = pd.DataFrame.from_dict(rows, orient="index")
cmp.to_csv(OUT / "compare_variants_log.csv")
cmp
''')

code(r'''
prev_q = HERE / ".." / "meeting17_fixes" / "out" / "quality_time_aware.csv"
if ADD_UNLINKED:     # здесь только стандартные метрики PM4Py
    cols = ["fitness_tbr", "precision_etc", "f1"]
else:
    cols = ["fitness_tbr", "precision_etc", "fitness_ta", "precision_ta", "f1_ta", "precision_ta_marking",
            "precision_noblocks", "missed_events", "truncated"]


def with_f1(q):
    q = q.copy()
    if "f1" not in q.columns:
        q["f1"] = (2 * q["fitness_tbr"] * q["precision_etc"] / (q["fitness_tbr"] + q["precision_etc"])).round(4)
    return q


parts = []
if prev_q.exists():
    parts.append(with_f1(pd.read_csv(prev_q)).set_index(["variant", "model"])[cols])
if (OLDCASE_BY_TYPE or ADD_UNLINKED) and (COMMON_OUT / "model_quality.csv").exists():
    parts.append(with_f1(pd.read_csv(COMMON_OUT / "model_quality.csv")).assign(variant="30.10, общий OldCase")
                   .set_index(["variant", "model"])[cols])
if len(quality):
    parts.append(with_f1(quality).assign(variant=THIS).set_index(["variant", "model"])[cols])
qcmp = pd.concat(parts).sort_index(level="model") if parts else pd.DataFrame()
qcmp.to_csv(OUT / "compare_variants_quality.csv")
qcmp
''')

md(r'''
## Шаг 10. Итог и файлы

Сводка журнала и список того, что ноутбук записал в `{{OUTDIR}}/`.
''')

code(r'''
print(pd.Series(info).to_string())
sorted(p.name for p in OUT.iterdir())
''')


def build(path):
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for kind, src in CELLS:
        cell = {"cell_type": kind, "id": uuid.uuid4().hex[:8], "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    json.dump(nb, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    build(sys.argv[1])
