"""Генератор ноутбуков petri_states_eval_<эксперимент>.ipynb: лучшие майнеры журнала состояний
на разбиении людей train/test с alignments на всём test. Запуск: python make_notebook_states_eval.py"""
import json
from pathlib import Path

EXPERIMENTS = {"2013": ('"2013-01-01"', "окно с 2013-01-01", 120), "all": ("None", "вся история", 20)}


def build(exp, log_start, title, limit):
    cells = []

    def md(s):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})

    def code(s):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

    md(rf"""
# Журнал состояний, {title}: лучшие майнеры на train/test с alignments

Продолжение `petri_states.ipynb`. Там сети строились и проверялись на всём журнале; здесь — **честная проверка**:

- люди делятся на train (80%) и test (20%), seed 42 — трассы одного человека не попадают в обе части;
- сеть строится на train, метрики считаются на test;
- **alignment fitness — на всех трассах test** (а не на выборке), лимит `ALIGN_MAX_SEC_TRACE` = {limit} с на вариант трассы{" (по всей истории трассы длинные: при 120 с расчёт занял бы несколько суток, решение 04.11)" if limit != 120 else ""}.

Модели (лучшие по F1 без разбиения): сеть из DFG p ≥ 0.05, Heuristics Miner 0.95, Split Miner, Inductive Miner 0.2; цветочная — для нормировки precision.

> ⚠️ **Время.** Alignments на всём test долгие. Тест на 150 трассах окна 2013 с лимитом 2 с: DFG — 35 с, HM — 48 с, Split Miner — 89 с, IM 0.2 — 147 с (у IM 71 из 150 трасс не уложились в 2 с). С лимитом 120 с IM 0.2 может считаться **десятки часов**. Поэтому:
> - каждая модель — в своей ячейке; готовая модель дописывается в `out_states\{exp}\split_eval.jsonl` и при повторном запуске **пропускается**;
> - alignments **сохраняются по каждому варианту трассы** (`out_states\{exp}\align_ckpt\<модель>_align.jsonl`): ячейку можно прервать (*Kernel → Interrupt*) и запустить снова — расчёт продолжится с места;
> - IM 0.2 стоит последним; `N_CORES > 1` — параллельные alignments (быстрее, но без продолжения с места);
> - `ALIGN_MAX_SEC_TRACE` можно уменьшить (например, 30 с) — трассы, не уложившиеся в лимит, в fitness не входят, их число — `align_timeout`.
""")

    md(r"""
## Шаг 0. Настройки
""")
    code(rf"""
from pathlib import Path

UP = Path("..")
DB_CANDIDATES = [UP / "stage0 - data" / "compas.db", Path(r"D:\diploma\compas.db"), Path("compas.db")]
CATEGORIES_DIRS = [Path("."), UP / "stage0 ver2 from charge"]

EXPERIMENT = "{exp}"
LOG_START = {log_start}               # None = вся история; "2013-01-01" = окно
SHORT_CUSTODY_DAYS = 15                # те же флаги, что в petri_states.ipynb
LONG_CUSTODY_DAYS = 365

TEST_SHARE = 0.2
SEED = 42
MODELS = ["DFG_p0.05", "HM_dep0.95", "SplitMiner", "IM_noise0.2"]   # IM 0.2 — последним (самый долгий)
ALIGN_MAX_SEC_TRACE = {limit}              # лимит alignments на вариант трассы, секунд
N_CORES = 1                            # >1 — параллельные alignments
RERUN = False                          # True — пересчитать модели, уже записанные в результаты

OUT = Path("out_states") / EXPERIMENT
RESULTS = OUT / "split_eval.jsonl"
""")
    code(r"""
import sys, time, json, warnings
import numpy as np
import pandas as pd
from IPython.display import display

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd()))
import petri_pipeline as pp
import petri_states as ps
import pm4py

OUT.mkdir(parents=True, exist_ok=True)
DB = pp.find_db(DB_CANDIDATES)
categorize = pp.import_categorizer(CATEGORIES_DIRS)
print("база:", DB, "| pm4py", pm4py.__version__, "| результаты:", RESULTS.resolve())


def done_models():
    if not RESULTS.exists():
        return {}
    rows = [json.loads(l) for l in RESULTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["model"]: r for r in rows}


def run(name):
    # посчитать одну модель и дописать строку в RESULTS (если уже есть и RERUN = False — пропустить)
    if name in done_models() and not RERUN:
        print(f"{name}: уже посчитано, пропуск (RERUN = True — пересчитать)")
        return done_models()[name]
    t0 = time.time()
    ck = OUT / ("align_ckpt" if ALIGN_MAX_SEC_TRACE == 120 else f"align_ckpt_{ALIGN_MAX_SEC_TRACE}s")
    r = ps.evaluate_split_one(name, MODELS_FIT[name], tr, te, max_sec_trace=ALIGN_MAX_SEC_TRACE, cores=N_CORES,
                              ckpt_dir=ck)
    if name != "flower" and not r.get("align_complete", True):
        print(f"{name}: alignments не закончены — запустить ячейку ещё раз")
        return r
    r.update(experiment=EXPERIMENT, align_max_sec_trace=ALIGN_MAX_SEC_TRACE, total_sec=round(time.time() - t0, 1),
             finished=time.strftime("%Y-%m-%d %H:%M"))
    r = {k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()}
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"{name}: {r['total_sec'] / 60:.1f} мин")
    return r
""")

    md(r"""
## Шаг 1. Журнал состояний и разбиение train/test

Журнал строится заново теми же функциями, что в `petri_states.ipynb` (около 2 минут).
""")
    code(r"""
tables, _ = pp.drop_full_duplicates(pp.load_tables(DB))
tables, _, _ = pp.drop_invalid_stays(tables)
hist = pp.build_history_table(tables, categorize)
log, episodes, report = ps.build_state_log(hist, log_start=LOG_START,
                                           short_days=SHORT_CUSTODY_DAYS, long_days=LONG_CUSTODY_DAYS)
for line in ps.check_state_log(log):
    print("OK:", line)
train, test = pp.split_people(log, TEST_SHARE, SEED)
tr, te = pp.to_pm4py(train), pp.to_pm4py(test)
print(f"train: {train.person_id.nunique():,} людей / {len(train):,} состояний; "
      f"test: {test.person_id.nunique():,} людей / {len(test):,} состояний; "
      f"вариантов трасс в test: {len(pm4py.get_variants(te)):,}")
""")
    code(r"""
MODELS_FIT = ps.discover_named(tr, MODELS)
pd.DataFrame([dict(model=m, places=len(n.places), transitions=len(n.transitions),
                   silent=sum(t.label is None for t in n.transitions)) for m, (n, im, fm) in MODELS_FIT.items()])
""")

    md(r"""
## Шаг 2. Метрики по моделям

Каждая ячейка: TBR fitness на train и test, ETC precision на test, alignment fitness на всех трассах test (прогресс печатается каждые 100 вариантов). Ячейку можно прервать (*Kernel → Interrupt*) и запустить снова: посчитанное сохранено.
""")
    code(r"""
run("flower")        # цветочная: только TBR и precision (секунды) — нужна для нормировки
""")
    for m in ["DFG_p0.05", "HM_dep0.95", "SplitMiner", "IM_noise0.2"]:
        code(f'run("{m}")')

    md(r"""
## Шаг 3. Итог
""")
    code(r"""
res = pd.DataFrame(done_models().values())
q = ps.summarize_split(res)
cols = ["model", "fitness_tbr_train", "fitness_tbr_test", "pct_fit_traces_tbr_test", "fitness_align_test",
        "pct_fit_traces_align_test", "easy_sound", "align_timeout", "align_traces", "precision_etc_test", "precision_norm_test",
        "f1_tbr", "f1_align", "places", "transitions", "silent", "total_sec", "finished"]
q[[c for c in cols if c in q]].round(3)
""")
    code(r"""
# сравнение с расчётом без разбиения (petri_states.ipynb, весь журнал)
f = OUT / "model_quality.csv"
if f.exists():
    full = pd.read_csv(f).set_index("model")[["fitness_tbr", "precision_etc", "precision_norm", "f1"]]
    full.columns = [c + " | весь журнал" for c in full.columns]
    split = q.set_index("model")[["fitness_tbr_test", "fitness_align_test", "precision_etc_test", "precision_norm_test", "f1_tbr", "f1_align"]]
    split.columns = [c + " | test" for c in split.columns]
    display(split.join(full, how="left").round(3))
else:
    print("нет", f, "— сначала выполнить petri_states.ipynb")
""")

    md(r"""
## ✅ Напоминание: проверить результаты

1. **Все ли модели посчитаны** — в итоговой таблице 5 строк (4 модели + `flower`), у всех заполнено `finished`.
2. **Тайм-ауты alignments**: доля `align_timeout / align_traces`. Если больше ~10% — `fitness_align_test` ненадёжна (в неё входят только «быстрые» трассы); указать это в тексте или пересчитать с большим лимитом / на `N_CORES > 1`.
3. **Переобучение**: `fitness_tbr_train` и `fitness_tbr_test` должны быть близки (разница > 0.02 — сеть подогнана под train).
4. **Сравнение с расчётом на всём журнале** (последняя таблица): precision на test обычно ниже — это нормально; если порядок моделей по F1 поменялся, отметить.
5. **TBR и alignments**: если `fitness_align_test` сильно ниже `fitness_tbr_test` (как у Split Miner), TBR завышает fitness из-за тихих переходов — в выводах опираться на alignments.
6. **Сеть без финальной разметки** (`easy_sound = False`): alignments для неё не определены (`fitness_align_test` пустой) — так вышло у HM 0.95 на всей истории; сравнивать её можно только по TBR и precision.
7. Записать итоговые числа и выбор лучшей модели в отчёт / `Требования_к_архитектуре.md`.
""")

    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"e{i:02d}"
        src = c["source"]
        c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
    path = Path(__file__).with_name(f"petri_states_eval_{exp}.ipynb")
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(path.name, len(cells), "ячеек")


for exp, (ls, title, limit) in EXPERIMENTS.items():
    build(exp, ls, title, limit)
