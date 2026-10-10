"""Собирает ноутбуки экспериментов по precision для варианта «категории × 2».
Запуск: python make_experiments.py <папка stage1>  -> <stage1>/<эксперимент>/<эксперимент>.ipynb"""
import shutil
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
ROOT = Path(sys.argv[1])

COMMON_HEAD = r"""
**Как запустить.** Окружение `D:\Git\diploma\.venv`, открыть ноутбук из этой папки, *Run All*. Вход — таблица этапа 0
(`jupyter lab\stage0 - data\out\compas_custody_charges.csv`), общий код журнала — `x2_common.py` рядом (тот же, что в
`oldcase_x2`). Результаты пишутся в `out\` этой папки.

**Как сравниваются модели.** Как в `oldcase_x2`: люди делятся на train 80% / test 20% (seed 42), модели строятся на
train, fitness и precision считаются token-based replay на первых 2 000 людях train и test. Базовая линия — вариант
`oldcase_x2` (29 состояний, порядок по тяжести): HM 0.9 — fitness 0.982, precision 0.543; IM 0.2 — 0.994 / 0.383.
"""

EXPERIMENTS = {
    "x2_exp1_one_event": dict(
        title="Эксперимент 1. Одно событие на заключение",
        why=r"""
**Идея.** В основном варианте заключение с несколькими категориями даёт цепочку событий (по тяжести, время 0).
Майнеры видят в этих цепочках параллельность и циклы и строят модель, которая разрешает много лишних путей.
Здесь от каждого заключения остаётся **одна категория — самая тяжёлая** (при равной тяжести — случайная, seed 42;
`Justice` — только если в заключении нет других категорий). Трасса человека становится простой
последовательностью заключений.

**Зачем.** Проверить, насколько низкая precision вызвана развилками внутри заключения. В Этапе 1 вариант
«одна категория на эпизод» давал precision около 0.87 против 0.65–0.75 с развилками.

**Цена.** Теряются менее тяжёлые категории заключения (в основном варианте 8 484 шага из 21 866 с развилкой).
""",
        build=dict(one_event=True), merge_rare=False,
        models=["HM_0.5", "HM_0.9", "HM_0.95", "IM_0.2", "IM_0.5"]),
    "x2_exp2_merge_rare": dict(
        title="Эксперимент 2. Слить редкие OldCase: X",
        why=r"""
**Идея.** У части состояний `OldCase: X` очень мало событий (от 1 до ~200). Они почти не влияют на граф, но каждое
добавляет узел, вокруг которого модель разрешает лишние переходы. Здесь все `OldCase: X` с числом событий
меньше `MIN_EVENTS = 300` сливаются в одно состояние **`OldCase: Other`** (новые категории не трогаются).

**Зачем.** Проверить, сколько precision теряется из-за редких состояний. Ожидаемо остаётся около 22 состояний.

**Цена.** Для редких категорий пропадает различие «старое дело Sex / Weapon / …».
""",
        build=dict(), merge_rare=True,
        models=["HM_0.5", "HM_0.9", "HM_0.95", "IM_0.2", "IM_0.5"]),
    "x2_exp3_one_event_merge": dict(
        title="Эксперимент 3. Одно событие на заключение + слить редкие OldCase: X",
        why=r"""
**Идея.** Оба упрощения журнала вместе: одна самая тяжёлая категория на заключение (эксп. 1) и слияние редких
`OldCase: X` в `OldCase: Other` (эксп. 2).

**Зачем.** Посмотреть верхнюю границу precision, которую можно получить упрощением журнала без потери частых состояний.
""",
        build=dict(one_event=True), merge_rare=True,
        models=["HM_0.5", "HM_0.9", "HM_0.95", "IM_0.2", "IM_0.5"]),
    "x2_exp4_miners": dict(
        title="Эксперимент 4. Параметры майнеров и фильтр редких трасс",
        why=r"""
**Идея.** Журнал тот же, что в основном варианте `oldcase_x2` (29 состояний). Меняются только алгоритм и его параметры:
- **Heuristics Miner** с более строгим порогом зависимости (0.95, 0.99) и с отсечением редких дуг
  (`min_dfg_occurrences = 30`): в модель попадают только устойчивые связи; и с более строгим порогом параллельности
  (`and_threshold = 0.95`), чтобы категории реже считались параллельными;
- **Inductive Miner** с порогом шума 0.3 и 0.5;
- **обучение только на частых трассах**: модель строится по людям, чья трасса встречается хотя бы у двух человек
  (редкие, уникальные трассы не тянут в модель лишние пути); проверка — на всех людях;
- **ILP Miner** — обычно даёт самую высокую precision, но здесь не уложился даже в 6 минут на 300 людях, поэтому
  выключен флагом `RUN_ILP = False` (можно включить и оставить считаться на ночь);
- **precision через выравнивания** (alignments) для HM 0.9 (флаг `ALIGN_CASES`, считается во всех экспериментах):
  token replay может искажать precision на сетях со многими скрытыми переходами, выравнивания точнее.

**Зачем.** Понять, сколько precision можно получить без изменения журнала.
""",
        build=dict(), merge_rare=False,
        models=["HM_0.9", "HM_0.95", "HM_0.99", "HM_0.9_min30", "HM_0.9_and0.95", "IM_0.2", "IM_0.3", "IM_0.5",
                "HM_0.9_freq", "IM_0.2_freq", "ILP"], align=300),
}


def notebook(key, e):
    cells = []
    md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
    code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))
    md(f"# {e['title']}\n\nВариант «категории × 2», поиск способов поднять precision моделей.\n" + e["why"] + COMMON_HEAD)
    md("## Шаг 0. Настройки")
    ilp, align = False, 2000
    code(rf'''
import os, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Image, display

warnings.filterwarnings("ignore")
os.environ["PM4PY_SHOW_PROGRESS_BAR"] = "False"
for gv in [Path(r"D:\Graphviz\bin")]:
    if gv.exists() and str(gv) not in os.environ["PATH"]:
        os.environ["PATH"] = str(gv) + os.pathsep + os.environ["PATH"]
import pm4py
import x2_common as X

HERE = Path.cwd()
ROOT = HERE.parent.parent                       # jupyter lab (на компьютере) или корень репозитория
CANDIDATES = [ROOT / "stage0 - data" / "out" / "compas_custody_charges.csv",
              ROOT / "stage0-data" / "compas_custody_charges.csv"]
DATA = next((p for p in CANDIDATES if p.exists()), CANDIDATES[0])
OUT = HERE / "out"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

EXP = "{key}"
BUILD = {e['build']!r}       # параметры журнала
MERGE_RARE = {e['merge_rare']!r}      # слить редкие OldCase: X
MIN_EVENTS = 300            # порог «редкого» состояния
MODELS = {e['models']!r}
EVAL_CASES = 2000           # людей в выборках train/test для fitness/precision
RUN_ILP = {ilp!r}             # ILP Miner (долго)
ALIGN_CASES = {align!r}          # precision через выравнивания для HM 0.9 на стольких людях из test (0 — не считать);
                            # чем меньше выборка, тем ниже precision (часть префиксов не встречается), поэтому 2000
print("pm4py", pm4py.__version__, "| вход:", DATA, DATA.exists())


def show(path, max_px=1400):
    from io import BytesIO
    from PIL import Image as PILImage
    PILImage.MAX_IMAGE_PIXELS = None
    im = PILImage.open(path)
    im.thumbnail((max_px, max_px))
    buf = BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=80)
    display(Image(data=buf.getvalue(), format="jpeg"))
''')
    md("## Шаг 1. Журнал событий")
    code(r'''
ch, ep = X.load(DATA)
merge_old = ()
if MERGE_RARE:
    base_ev, _, _ = X.build("x2", ch, ep, **{k: v for k, v in BUILD.items() if k != "merge_old"})
    vc = base_ev["concept:name"].value_counts()
    merge_old = sorted(a for a, n in vc.items() if a.startswith(X.OLD_PREFIX) and n < MIN_EVENTS)
    print("сливаются в «OldCase: Other»:", ", ".join(f"{a} ({vc[a]})" for a in merge_old))
ev, st, info = X.build("x2", ch, ep, merge_old=merge_old, **BUILD)
info
''')
    code(r'''
log = pm4py.format_dataframe(ev[["case:concept:name", "concept:name", "time:timestamp"]].copy(),
                             case_id="case:concept:name", activity_key="concept:name", timestamp_key="time:timestamp")
dfg, sa, ea = pm4py.discover_dfg(log)
traces = st.groupby("id")["state"].agg(" → ".join).value_counts()
stats = pd.Series({"состояний": log["concept:name"].nunique(), "событий": len(log), "шагов": len(st),
                   "дуг DFG": len(dfg), "петель DFG (X→X)": sum(a == b for a, b in dfg),
                   "вариантов трасс": len(traces), "людей с уникальной трассой": int((traces == 1).sum())})
ev.to_csv(OUT / f"event_log_{EXP}.csv", index=False)
stats.to_csv(OUT / "log_stats.csv", header=["value"])
print(log["concept:name"].value_counts().to_string())
stats
''')
    md("## Шаг 2. Граф: DFG (30% путей) и сеть Петри Heuristics Miner 0.9")
    code(r'''
dfg_f, sa_f, ea_f = pm4py.filter_dfg_paths_percentage(dfg, sa, ea, 0.3)
# у pm4py 2.7.23.9 рисунок падает, если начальная/конечная активность не осталась ни на одной дуге — убираем такие
acts_f = {a for k in dfg_f for a in k}
sa_f = {a: n for a, n in sa_f.items() if a in acts_f}
ea_f = {a: n for a, n in ea_f.items() if a in acts_f}
pm4py.save_vis_dfg(dfg_f, sa_f, ea_f, str(FIG / "dfg_frequency_paths30.png"))
show(FIG / "dfg_frequency_paths30.png")
''')
    code(r'''
net, im, fm = pm4py.discover_petri_net_heuristics(log, dependency_threshold=0.9)
pm4py.save_vis_petri_net(net, im, fm, str(FIG / "petri_HM_dep0.9.png"))
print(f"HM 0.9 на всём журнале: мест {len(net.places)}, переходов {len(net.transitions)}")
show(FIG / "petri_HM_dep0.9.png")
''')
    md(r"""
## Шаг 3. Модели и качество

Названия моделей: `HM_d` — Heuristics Miner с порогом зависимости d; `HM_0.9_min30` — то же, дуги реже 30 раз
отбрасываются; `HM_0.9_and0.95` — то же, параллельность (AND) объявляется только при мере ≥ 0.95; `IM_n` — Inductive Miner с порогом шума n; `*_freq` — модель построена только по людям с
трассой, которая встречается хотя бы у двух человек (проверка — на всех); `ILP` — ILP Miner.
""")
    code(r'''
train, test = X.split_people(log)
train_s, test_s = X.sample_cases(train, EVAL_CASES), X.sample_cases(test, EVAL_CASES)


def frequent_train():
    # люди из train, чья трасса (последовательность событий) встречается хотя бы у двух человек
    seq = train.groupby("case:concept:name")["concept:name"].agg(tuple)
    n = seq.map(seq.value_counts())
    return train[train["case:concept:name"].isin(n[n >= 2].index)]


def discover(name):
    if name.startswith("HM_"):
        parts = name.split("_")
        lg = frequent_train() if name.endswith("_freq") else train
        if "min30" in name:
            # параметр есть только в низкоуровневом API pm4py
            from pm4py.algo.discovery.heuristics import algorithm as hm
            P = hm.Variants.CLASSIC.value.Parameters
            return hm.apply(lg, parameters={P.DEPENDENCY_THRESH: float(parts[1]), P.MIN_DFG_OCCURRENCES: 30})
        kw = dict(dependency_threshold=float(parts[1]))
        if "and0.95" in name:
            kw["and_threshold"] = 0.95      # реже объявлять параллельность (AND) между категориями
        return pm4py.discover_petri_net_heuristics(lg, **kw)
    if name.startswith("IM_"):
        lg = frequent_train() if name.endswith("_freq") else train
        return pm4py.discover_petri_net_inductive(lg, noise_threshold=float(name.split("_")[1]))
    if name == "ILP":
        return pm4py.discover_petri_net_ilp(train)
    raise ValueError(name)


models, times = {}, {}
for name in MODELS:
    if name == "ILP" and not RUN_ILP:
        continue
    t0 = time.time()
    models[name] = discover(name)
    times[name] = round(time.time() - t0)
    print(f"{name}: построена за {times[name]} с")
''')
    code(r'''
quality = X.evaluate(models, train_s, test_s)
quality["discover_sec"] = quality["model"].map(times)
if ALIGN_CASES and "HM_0.9" in models:
    t0 = time.time()
    pa = pm4py.precision_alignments(X.sample_cases(test, ALIGN_CASES), *models["HM_0.9"])
    quality.loc[(quality["model"] == "HM_0.9") & (quality["split"] == "test"), "precision_align"] = round(pa, 4)
    print(f"HM 0.9: precision через выравнивания на {ALIGN_CASES} людях = {pa:.4f} ({time.time() - t0:.0f} с)")
quality.to_csv(OUT / "model_quality.csv", index=False)
quality[quality["split"] == "test"]
''')
    md("## Шаг 4. Сравнение с основным вариантом oldcase_x2")
    code(r'''
BASE = pd.DataFrame({"model": ["HM_0.5", "HM_0.9", "IM_0.0", "IM_0.2", "IM_0.5"],
                     "fitness_base": [0.9415, 0.9821, 1.0, 0.9938, 0.9716],
                     "precision_base": [0.6136, 0.5430, 0.3688, 0.3828, 0.4563]})
cmp = quality[quality["split"] == "test"][["model", "fitness_tbr", "precision_tbr", "f1"]].merge(BASE, on="model", how="left")
cmp["Δ precision"] = (cmp["precision_tbr"] - cmp["precision_base"]).round(4)
cmp.to_csv(OUT / "compare_with_oldcase_x2.csv", index=False)
cmp
''')
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


for key, e in EXPERIMENTS.items():
    d = ROOT / key
    (d / "out").mkdir(parents=True, exist_ok=True)
    nbf.write(notebook(key, e), d / f"{key}.ipynb")
    for f in ("x2_common.py", "crime_categories.py"):
        shutil.copy(HERE / f, d / f)
    print(d)
