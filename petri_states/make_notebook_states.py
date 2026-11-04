"""Генератор ноутбука petri_states.ipynb (журнал «состояний», правила 04.11.2026). Запуск: python make_notebook_states.py"""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md(r"""
# Сети Петри на журнале состояний (04.11.2026)

Отличие от `petri_nets.ipynb`: запись базы — не отдельное событие, а часть **состояния** человека. Человек всегда в одном из состояний: на свободе (`free`), в заключении (`jail` / `prison`), под арестом или обвинением по делу (`arrest`, `charge`, `arrest+charge`) или в составном состоянии «дело + заключение» (`arrest+charge+jail` и т. п.).

**Эксперименты** (без уточнённой разметки активностей):

| Эксперимент | Окно | Папка |
|---|---|---|
| `all` | вся история базы | `out_states\all\` |
| `2013` | с 2013-01-01 (есть `jailhistory`) | `out_states\2013\` |

**Метрики пока без разбиения train/test и без alignments**: сеть строится на всём журнале, fitness (token replay) и precision (ETC) считаются на нём же. Следующий шаг — лучшие майнеры с разбиением людей 80/20 и alignments (в задачах).

Логика — в `petri_states.py` (журнал состояний) и `petri_pipeline.py` (таблица 1, эпизоды заключения, майнеры).
""")

md(r"""
## Правила журнала состояний

| № | Ситуация | Правило |
|---|---|---|
| 1 | Несколько записей **одного дела за один день** | одно состояние: все аресты → `arrest`, все обвинения → `charge`, оба → `arrest+charge` |
| 2 | Несколько дел в один день | по состоянию на дело, порядок по номеру дела, между ними `free` нет |
| 3 | Вход в заключение в день D, **одно** дело в этот день, заключение короче `LONG_CUSTODY_DAYS` | составное состояние: `arrest+jail`, `charge+jail`, `arrest+charge+jail` (то же с `prison`); время — вход в заключение |
| 4 | Вход в заключение, **несколько** дел в этот день | состояния дел (по номеру), затем отдельно `jail` / `prison` |
| 5 | Заключение **длинное** (≥ `LONG_CUSTODY_DAYS`, флаг, 365 дней) и аресты/обвинения в день входа | аресты/обвинения — отдельными состояниями **перед** заключением |
| 6 | Аресты/обвинения **внутри срока**, заключение **короткое** (< `SHORT_CUSTODY_DAYS`, флаг, 15 дней) | склеиваются с состоянием заключения (его имя дополняется: `jail` → `charge+jail`) |
| 7 | Аресты/обвинения внутри срока, заключение не короткое | `jail → arrest/charge → jail`: возврат в заключение в D+1 00:00, если на следующий день внутри срока событий нет |
| 8 | Перевод jail → prison внутри эпизода | отдельное состояние `prison` |
| 9 | Выход из заключения | `free` во время `out_custody`; **не ставится**, если следующее событие в тот же или на следующий день (переход сразу в новое состояние) |
| 10 | Арест/обвинение без заключения | если на следующий день событий нет — `free` в D+1 00:00, иначе сразу следующее состояние |
| 11 | Начало трассы | без окна — первое состояние; в окне — состояние на 2013-01-01: `free` или `jail`/`prison`, если человек тогда сидел |
| 12 | Конец трассы | `end` (конец наблюдения 2016-04-19 07:19:36) — только из `free`, `jail` или `prison` |

Время состояния дела без заключения — дата (00:00); одинаковое время разводится на 1 мс в порядке состояний.
""")

md(r"""
## Шаг 0. Настройки

| Параметр | Значение | Смысл |
|---|---|---|
| `EXPERIMENTS` | `all`: без окна, `2013`: с 2013-01-01 | эксперименты |
| `SHORT_CUSTODY_DAYS` | 15 | флаг: заключение короче — события внутри срока склеиваются с ним (правило 6). Медиана длительности — в шаге 1.1 |
| `LONG_CUSTODY_DAYS` | 365 | флаг: заключение длиннее — аресты/обвинения дня входа ставятся перед ним (правило 5) |
| `IM_NOISE`, `HM_DEP`, `DFG_MIN_PROB` | (0, 0.2), (0.9, 0.95, 0.99), (0, 0.05) | пороги майнеров |
| `RUN_EVAL` | `False` | `True` — пересчитать метрики (окно 2013 — около 15 минут, вся история — около 1,5 часа; нужно ~3 ГБ памяти); `False` — взять `out_states\<эксперимент>\model_quality.csv` |
""")

code(r"""
from pathlib import Path

UP = Path("..")
DB_CANDIDATES = [UP / "stage0 - data" / "compas.db", Path(r"D:\diploma\compas.db"), Path("compas.db")]
CATEGORIES_DIRS = [Path("."), UP / "stage0 ver2 from charge"]
OUT = Path("out_states")

EXPERIMENTS = {"all": None, "2013": "2013-01-01"}
SHORT_CUSTODY_DAYS = 15          # флаг: короткое заключение (правило 6) — потом скорректировать по медиане
LONG_CUSTODY_DAYS = 365          # флаг: длинное заключение (правило 5) — потом скорректировать

IM_NOISE = (0.0, 0.2)
HM_DEP = (0.9, 0.95, 0.99)
DFG_MIN_PROB = (0.0, 0.05)
RUN_EVAL = False                 # True — пересчитать метрики
""")

code(r"""
import sys, time, warnings
import numpy as np
import pandas as pd
from IPython.display import display, Image

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd()))
import petri_pipeline as pp
import petri_states as ps
import pm4py

OUT.mkdir(exist_ok=True)
DB = pp.find_db(DB_CANDIDATES)
categorize = pp.import_categorizer(CATEGORIES_DIRS)
HAS_DOT = pp.setup_graphviz()
pd.set_option("display.max_columns", 30, "display.width", 220)
print("база:", DB)
print("pm4py", pm4py.__version__, "| pandas", pd.__version__, "| Graphviz:", "есть" if HAS_DOT else "НЕТ")
""")

md(r"""
## Шаг 1. Таблица истории (этап 1, как в `petri_nets.ipynb`)

Полные дубли удаляются, пребывания со входом позже выхода — тоже.
""")

code(r"""
tables, dup_report = pp.drop_full_duplicates(pp.load_tables(DB))
tables, bad_stays_report, _ = pp.drop_invalid_stays(tables)
hist = pp.build_history_table(tables, categorize)
display(dup_report)
display(bad_stays_report)
print(f"таблица 1: {len(hist):,} строк, {hist.person_id.nunique():,} человек")
""")

md(r"""
### 1.1. Длительность заключений (для флагов `SHORT_CUSTODY_DAYS` / `LONG_CUSTODY_DAYS`)

Эпизод заключения — склеенные пребывания jail/prison (пересечение или перевод jail → prison в течение суток). Кто сидит на конец наблюдения — длительность до конца наблюдения.
""")

code(r"""
obs_end = pp.observation_end(hist)
_, episodes = pp.custody_episodes(hist, obs_end)
dur = (episodes["ep_end"].fillna(obs_end) - episodes["ep_start"]).dt.total_seconds() / 86400
display(dur.describe(percentiles=[.1, .25, .5, .75, .9, .95]).round(1).to_frame("дней").T)
print(f"короче {SHORT_CUSTODY_DAYS} дн.: {(dur < SHORT_CUSTODY_DAYS).mean():.1%};  "
      f"{LONG_CUSTODY_DAYS} дн. и дольше: {(dur >= LONG_CUSTODY_DAYS).mean():.1%}")
pd.cut(dur, [-1, 1, 7, 15, 30, 90, 180, 365, 1e6],
       labels=["≤1", "1–7", "7–15", "15–30", "30–90", "90–180", "180–365", ">365"]).value_counts(sort=False).to_frame("эпизодов").T
""")

md(r"""
## Шаг 2. Журналы состояний
""")

code(r"""
LOGS, REPORTS = {}, {}
for key, start in EXPERIMENTS.items():
    t0 = time.time()
    log, eps, rep = ps.build_state_log(hist, obs_end=obs_end, log_start=start,
                                       short_days=SHORT_CUSTODY_DAYS, long_days=LONG_CUSTODY_DAYS)
    LOGS[key], REPORTS[key] = log, rep
    d = OUT / key
    d.mkdir(exist_ok=True)
    log.to_csv(d / "state_log.csv", index=False)
    print(f"{key}: {time.time() - t0:.0f} с, сохранено {d / 'state_log.csv'}")
    for line in ps.check_state_log(log):
        print("   OK:", line)
pd.DataFrame(REPORTS).rename_axis("показатель")
""")

md(r"""
Показатели: `composite` — составных состояний «дело + заключение»; `absorbed_units` — дел внутри коротких сроков, склеенных с заключением; `inside_units` / `returns` — состояний внутри срока и возвратов в заключение; `free_after_case` / `free_after_custody` — выходов в `free` после ареста/обвинения и после заключения; `no_free_next_day` — выходов из заключения без `free` (следующее событие в тот же или на следующий день); `started_inside` — трасс окна, начатых в заключении.
""")

code(r"""
# состояния: сколько и доля
pd.concat({k: l["activity"].value_counts() for k, l in LOGS.items()}, axis=1).fillna(0).astype(int)
""")

code(r"""
# длина трасс
pd.concat({k: l.groupby("person_id").size().describe(percentiles=[.5, .9]).round(1) for k, l in LOGS.items()}, axis=1).T
""")

md(r"""
### 2.1. Примеры трасс (окно 2013)

- 8399: вышел из jail 13.05.2014 в 02:09 и в тот же день сел по новому делу — переход сразу в `arrest+charge+jail`, без `free` (правило 9);
- 7433: 6 записей одного дела за день → одно `arrest+charge`, на следующий день событий нет → `free` 28.05 00:00;
- 6266: два дела в один день → по номеру дела, затем `free`;
- 4781: обвинение 09.10 → `free` 10.10 → `arrest+jail` 15.10;
- 10897: два дела и jail в один день → дела, затем `jail`.
""")

code(r"""
cols = ["person_id", "activity", "ts", "ts_raw", "kind", "cases", "n_records"]
ex = [(8399, "2014-04-20", "2014-06-01"), (7433, "2013-01-01", "2014-09-01"), (6266, "2013-01-01", "2016-05-01"),
      (4781, "2013-10-01", "2014-07-01"), (10897, "2015-08-01", "2015-12-31")]
l13 = LOGS.get("2013", next(iter(LOGS.values())))
pd.concat([l13[(l13.person_id == p) & l13.ts.between(a, b)][cols] for p, a, b in ex])
""")

md(r"""
### 2.2. Матрица directly-follows (вероятность перехода и медиана времени, дни)
""")

code(r"""
for key, log in LOGS.items():
    cnt, prob, med = pp.directly_follows(log)
    print(f"=== {key}: вероятность перехода")
    display(prob.round(2))
    print(f"=== {key}: медиана времени, дни")
    display(med.round(1))
""")

code(r"""
# самые частые варианты (повторы подряд свёрнуты: X+ = несколько X подряд)
for key, log in LOGS.items():
    var = log.groupby("person_id")["activity"].agg(lambda s: " → ".join(pp.collapse_repeats(s)))
    vc = var.value_counts()
    print(f"=== {key}: {len(vc):,} вариантов на {len(var):,} трасс")
    display(pd.DataFrame({"людей": vc.head(12), "доля": (vc.head(12) / len(var)).round(4)}))
""")

md(r"""
## Шаг 3. Сети Петри: все майнеры, весь журнал

| Модель | Функция PM4Py |
|---|---|
| Inductive Miner (шум 0 / 0.2) | `discover_petri_net_inductive` |
| Heuristics Miner (0.9 / 0.95 / 0.99) | `discover_petri_net_heuristics` |
| Split Miner | `discover_bpmn_split_miner` → `convert_to_petri_net` (только свежие версии PM4Py, иначе пропускается) |
| POWL | `discover_powl` → `convert_to_petri_net` |
| Сеть из DFG (p0 / p0.05) | `discover_dfg` → `convert_to_petri_net` |
| Alpha / Alpha+ | `discover_petri_net_alpha` / `discover_petri_net_alpha_plus` |
| цветочная | разрешает любую последовательность — базовая линия precision |

ILP Miner и Genetic Miner не используются (ILP на наших трассах не помещается в память, Genetic считается часами).

Метрики: `fitness_tbr` (`fitness_token_based_replay`), `pct_fit_traces_tbr`, `precision_etc` (`precision_token_based_replay`), `precision_norm = (P − P_flower)/(1 − P_flower)`, `f1`. **Сеть строится и проверяется на одном и том же журнале** (без разбиения train/test) — это оценка того, насколько хорошо модель описывает данные, а не прогноза.
""")

code(r"""
LPM, MODELS, infos = {}, {}, []
for key, log in LOGS.items():
    LPM[key] = pp.to_pm4py(log)
    MODELS[key], info = ps.discover_all(LPM[key], im_noise=IM_NOISE, hm_dep=HM_DEP, dfg_min_prob=DFG_MIN_PROB)
    infos.append(info.assign(experiment=key))
pd.concat(infos, ignore_index=True).set_index(["experiment", "model"]).round(2)
""")

code(r"""
# рисунки сетей: out_states/<эксперимент>/net_<модель>.png
if HAS_DOT:
    for key, models in MODELS.items():
        for name, (net, im, fm) in models.items():
            if name == "flower":
                continue
            p = pp.save_net_png(net, im, fm, OUT / key / f"net_{name}.png")
            print(f"{key} | {name}")
            display(Image(filename=str(p), width=1000))
""")

code(r"""
# DFG: частоты и среднее время переходов
if HAS_DOT:
    for key, L in LPM.items():
        print(key)
        display(Image(filename=str(pp.save_dfg_png(L, OUT / key / "dfg_frequency.png")), width=900))
        display(Image(filename=str(pp.save_dfg_png(L, OUT / key / "dfg_performance.png", performance=True)), width=900))
""")

md(r"""
### 3.1. Fitness и precision (весь журнал)
""")

code(r"""
qs = []
for key in LOGS:
    f = OUT / key / "model_quality.csv"
    if RUN_EVAL or not f.exists():
        t0 = time.time()
        q = ps.evaluate_full(MODELS[key], LPM[key])
        q.to_csv(f, index=False)
        print(f"{key}: {(time.time() - t0) / 60:.1f} мин")
    else:
        q = pd.read_csv(f)
    qs.append(q.assign(experiment=key))
quality = pd.concat(qs, ignore_index=True)
cols = ["experiment", "model", "fitness_tbr", "pct_fit_traces_tbr", "precision_etc", "precision_flower",
        "precision_norm", "f1", "places", "transitions", "silent"]
quality[[c for c in cols if c in quality]].round(3).set_index(["experiment", "model"])
""")

code(r"""
# сравнение экспериментов
order = list(dict.fromkeys(quality["model"]))
cmp = quality.pivot_table(index="model", columns="experiment",
                          values=["fitness_tbr", "precision_etc", "precision_norm", "f1"]).reindex(order).round(3)
cmp.columns = [f"{m} | {e}" for m, e in cmp.columns]
cmp
""")

code(r"""
# три лучшие модели каждого эксперимента по F1 (без цветочной)
(quality[quality.model != "flower"].sort_values("f1", ascending=False).groupby("experiment").head(3)
   .sort_values(["experiment", "f1"], ascending=[True, False])
   [["experiment", "model", "fitness_tbr", "pct_fit_traces_tbr", "precision_etc", "precision_norm", "f1"]]
   .round(3).set_index(["experiment", "model"]))
""")

md(r"""
## Выводы

Сеть строится и проверяется на одном журнале (без train/test), alignments не считались.

| Модель | окно 2013: fitness | precision | precision норм. | F1 | вся история: fitness | precision | precision норм. | F1 |
|---|---|---|---|---|---|---|---|---|
| **DFG p0.05** | 0.966 | 0.815 | 0.681 | **0.884** | 0.957 | 0.702 | 0.615 | **0.810** |
| HM 0.95 | 0.982 | 0.675 | 0.439 | 0.800 | 0.984 | 0.417 | 0.246 | 0.585 |
| Split Miner | 0.772 | 0.813 | 0.678 | 0.792 | 0.726 | **0.887** | **0.854** | 0.798 |
| IM 0.2 | 1.000 | 0.442 | 0.036 | 0.613 | 0.984 | 0.524 | 0.385 | 0.684 |
| DFG p0 | 1.000 | 0.527 | 0.183 | 0.690 | 1.000 | 0.302 | 0.098 | 0.464 |
| IM 0, POWL | 1.000 | ≈ цветочная | ≈ 0 | 0.59–0.61 | 1.000 | 0.23–0.32 | 0–0.12 | 0.37–0.49 |
| Alpha / Alpha+ | 0.47 / 0.57 | 0.54 / 0.41 | 0.20 / −0.01 | 0.50 / 0.48 | 0.26 / 0.74 | 0.51 / 0.17 | 0.37 / −0.08 | 0.34 / 0.27 |

- **Лучшая по F1 в обоих экспериментах — сеть из DFG с фильтром p ≥ 0.05**: 12–15 мест, одна активность — одно место. На окне 2013 без ошибок проходят 70% трасс, на всей истории — 50%.
- **Heuristics Miner** на журнале состояний на окне 2013 стал заметно «чище»: без ошибок проходят 61% трасс (на журнале событий — 2–5%), но precision ниже, чем у DFG p0.05. На всей истории ни одна трасса не проходит без ошибок.
- **Split Miner** — самая высокая precision (на всей истории нормированная 0.85), но fitness падает до 0.73–0.77: он отбрасывает много поведения, почти ни одна трасса не проходит целиком.
- **Inductive Miner и POWL**: на окне 2013 — по-прежнему почти цветочная сеть; на всей истории IM 0.2 находит часть структуры (нормированная precision 0.39).
- **Alpha / Alpha+** не годятся: fitness 0.26–0.74, у Alpha+ precision ниже цветочной.
- Сравнивать с прежним журналом событий напрямую нельзя: там метрики считались на test (80/20), здесь — на всём журнале. Для честного сравнения — следующий шаг.

**Кандидаты на расчёт с train/test и alignments:** DFG p0.05, HM 0.95, Split Miner (оба эксперимента), IM 0.2 (вся история).
""")

md(r"""
## Что сделано дальше

### 1. Лучшие майнеры: train/test 80/20 и alignments

Модели: DFG p0.05, HM 0.95, Split Miner, IM 0.2 (+ цветочная как база precision).

Параметры поменялись против плана:
- alignments считаются на **всём** test, а не на 200 трассах;
- лимит на трассу: окно 2013 — 120 с, вся история — 20 с;
- тайм-ауты досчитываются с повышенным лимитом: окно 2013 — 600 с, вся история — 120 с (идёт);
- перед alignments проверяется easy soundness: HM 0.95 на всей истории — не sound
  (финальная разметка недостижима), alignments для неё не определены.

Итог по окну 2013: лучшая — DFG p0.05
(alignment fitness 0.964, precision 0.759, F1 0.849).
Вся история — после досчёта тайм-аутов.

| Что | Файл |
|---|---|
| Ноутбук, окно 2013 (выполнен) | `petri_states_eval_2013.ipynb` |
| Ноутбук, вся история (запуск после досчёта) | `petri_states_eval_all.ipynb` |
| Генератор этих ноутбуков | `make_notebook_states_eval.py` |
| Разбиение людей 80/20 | `petri_pipeline.py` → `split_people` |
| Оценка модели, alignments с контрольными точками, сводка | `petri_states.py` → `evaluate_split_one`, `alignment_fitness_resumable`, `summarize_split` |
| Скрипты запуска расчёта и досчёта тайм-аутов | `run_split.py`, `retry_timeouts.py` |
| Итоги по моделям | `out_states\<эксперимент>\split_eval.jsonl` |
| Alignments по вариантам трасс | `out_states\2013\align_ckpt\`, `out_states\all\align_ckpt_20s\` |
| Досчёт тайм-аутов | `split_eval_retry600s.jsonl` + `align_ckpt_retry600s\` (2013),<br>`split_eval_retry120s.jsonl` + `align_ckpt_retry120s\` (вся история) |

### 2. Флаги `SHORT_CUSTODY_DAYS` и `LONG_CUSTODY_DAYS`

Шаги: длительности заключений по зонам правил, медианы на переходах,
границы по данным, перебор флагов по качеству сетей (train/test, без alignments).

Вывод:
- `LONG_CUSTODY_DAYS`: **365 → 180** (с 180 дней эпизод — уже «срок»: больше половины с prison);
  по метрикам 180–365 — плато;
- `SHORT_CUSTODY_DAYS`: **оставить 15** (метрики к нему почти не чувствительны).

Рекомендация **ещё не применена**: этот ноутбук и расчёты п. 1 — с флагами 15 / 365.

| Что | Файл |
|---|---|
| Ноутбук | `petri_states_flags.ipynb` |
| Генератор ноутбука | `make_notebook_flags.py` |
| Расчёты (зоны, границы, перебор) | `flags_analysis.py` |
| Результаты | `out_states\flags\` |

### Осталось

1. Досчитать тайм-ауты alignments и выполнить `petri_states_eval_all.ipynb`.
2. Применить `LONG_CUSTODY_DAYS = 180` и пересчитать журнал, сети и метрики.
3. Решить по правилам журнала: повтор `jail → jail` / `prison → prison`
   и вложенный `jail` внутри `prison` (сейчас после него состояние остаётся `jail`).
4. Сеть DFG p0.05: убрать мёртвые места (состояния, все входы в которые отрезал фильтр)
   или сохранять для каждого состояния хотя бы один вход.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"s{i:02d}"
    src = c["source"]
    c["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
Path(__file__).with_name("petri_states.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("petri_states.ipynb:", len(cells), "ячеек")
