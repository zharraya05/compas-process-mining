"""Два ноутбука с результатами журнала состояний, версия 2 (новое правило end, флаги 15 / 180):
    petri_states_v2_results_2013.ipynb — окно с 2013-01-01
    petri_states_v2_results_all.ipynb  — вся история, без окна
Расчёт — run_states_v2.py (результаты в out_states_v2\\); ноутбуки только читают готовые файлы.
Для сравнения читаются результаты версии 1 из out_states\\ (их не меняют).
Запуск из папки «петри»:  python make_notebook_results_v2.py"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = {"2013": dict(title="окно с 2013-01-01"), "all": dict(title="вся история (без окна)")}
CONCL_FILE = HERE / "out_states_v2" / "conclusions.json"   # выводы пишутся после расчёта


def md(src):
    lines = src.strip("\n").split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def code(src):
    lines = src.strip("\n").split("\n")
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def build(key):
    c = CFG[key]
    concl = json.loads(CONCL_FILE.read_text(encoding="utf-8")).get(key, "") if CONCL_FILE.exists() else ""
    cells = [
        md(f"""
# Журнал состояний, версия 2: результаты — {c['title']}

Чем отличается от версии 1 (`petri_states_results_{key}.ipynb`, папка `out_states\\`; её результаты не менялись):

| | Версия 1 | Версия 2 |
|---|---|---|
| Флаги | `SHORT` = 15, `LONG` = 365 | `SHORT` = 15, `LONG` = **180** (подобраны в `petri_states_flags.ipynb`) |
| Переход в `end` | только из `free`, `jail`, `prison`; иначе перед `end` ставится возврат в заключение | из `free`, из **любого состояния с заключением** (`charge+prison` и т. п.), из ареста/обвинения — **если оно в день конца наблюдения или накануне** (тогда `free` не ставится) |
| Оценка | все майнеры на всём журнале; лучшие — train/test + alignments | **все майнеры на train/test**, TBR fitness и ETC precision, без alignments |

| Что | Файл |
|---|---|
| Расчёт (журнал, майнеры, метрики) | `run_states_v2.py` |
| Правило `end` (параметр `end_rule="v2"`) | `petri_states.py` → `build_state_log`, `check_state_log` |
| Этот ноутбук (генератор) | `make_notebook_results_v2.py` |
| Результаты | `out_states_v2\\{key}\\`: `state_log.csv`, `report.json`, `split_quality.csv`, `discovery_info.csv`, `net_*.png`, `dfg_*.png` |
"""),
        code(f"""
import json
from pathlib import Path
import pandas as pd
from IPython.display import display, Image

EXP = "{key}"
V2 = Path("out_states_v2") / EXP
V1 = Path("out_states") / EXP
BEST = ["flower", "DFG_p0.05", "HM_dep0.95", "SplitMiner", "IM_noise0.2"]
pd.set_option("display.precision", 3)
pd.set_option("display.max_columns", 40)

def state_log(d):
    p = d / "state_log.csv"
    return pd.read_csv(p if p.exists() else d / "state_log.csv.gz", parse_dates=["ts", "ts_raw"])


log2 = state_log(V2)
log1 = state_log(V1)
rep = json.loads((V2 / "report.json").read_text(encoding="utf-8"))
q2 = pd.read_csv(V2 / "split_quality.csv")
print("журнал v2:", f"{{len(log2):,}} событий, {{log2.person_id.nunique():,}} трасс")
print("журнал v1:", f"{{len(log1):,}} событий, {{log1.person_id.nunique():,}} трасс")
"""),
        md("""
## 1. Правила, которые изменились

**Правило 5 — `LONG_CUSTODY_DAYS` = 180.** Одно дело в день входа и эпизод заключения от 180 дней —
дело отдельным состоянием перед заключением (`arrest → jail`); короче 180 дней — составное (`arrest+jail`, правило 3).

**Правило 12 — переход в `end`:**
- из `free`;
- из любого состояния, в названии которого есть заключение: `jail`, `prison`, `arrest+jail`, `charge+prison`, …;
- из ареста/обвинения без заключения (`arrest`, `charge`, `arrest+charge`) — только если оно в день
  конца наблюдения или накануне: `free` в D+1 тогда не ставится, следующего дня в данных нет;
- дело внутри срока у того, кто ещё сидит, в последние два дня — тоже сразу в `end`, без возврата в заключение.

Конец наблюдения и «последние два дня»:
"""),
        code("""
obs_end = pd.Timestamp(rep["obs_end"])
print("конец наблюдения:", obs_end, "| дела с", (obs_end.normalize() - pd.Timedelta(days=1)).date(), "идут сразу в end")
print("переходов в end из дела (без free):", rep["end_from_case"])
print("переходов в end из составного состояния:", rep["end_from_composite"])
for line in rep["checks"]:
    print("проверка:", line)
"""),
        md("""
### 1.1. Что стоит перед `end`: версия 1 и версия 2
"""),
        code("""
before = pd.concat({"v1": log1.loc[log1.next_activity == "end", "activity"].value_counts(),
                    "v2": log2.loc[log2.next_activity == "end", "activity"].value_counts()}, axis=1).fillna(0).astype(int)
before.sort_values("v2", ascending=False)
"""),
        md("""
### 1.2. Примеры трасс: конец трассы в версии 1 и версии 2

Люди, у которых в версии 2 трасса заканчивается не из `free` / `jail` / `prison`. Показаны последние 4 состояния.
"""),
        code("""
cols = ["person_id", "activity", "ts_raw", "kind"]
pre2 = log2[log2.next_activity == "end"]
pick = pd.concat([pre2[pre2.kind == "case"].head(2), pre2[pre2.kind == "inside"].head(1),
                  pre2[pre2.kind == "custody"].loc[lambda x: ~x.activity.isin(["jail", "prison"])].head(2)])
for p in pick.person_id:
    t1 = log1[log1.person_id == p].tail(4)[cols].assign(версия="v1")
    t2 = log2[log2.person_id == p].tail(4)[cols].assign(версия="v2")
    print(f"человек {p}:")
    print("  v1:", " → ".join(f"{a} ({t:%d.%m.%Y})" for a, t in zip(t1.activity, t1.ts_raw)))
    print("  v2:", " → ".join(f"{a} ({t:%d.%m.%Y})" for a, t in zip(t2.activity, t2.ts_raw)))
"""),
        md("""
## 2. Журнал: показатели и состояния (версия 1 и версия 2)

- `composite` — составных состояний «дело + заключение» (правило 3);
- `absorbed_units` — дел внутри коротких сроков, склеенных с заключением (правило 6);
- `inside_units` / `returns` — дел внутри срока и возвратов в заключение (правило 7);
- `free_after_case` / `free_after_custody` — выходов в `free` после дела и после заключения;
- `end_from_case` / `end_from_composite` — переходов в `end` из дела и из составного состояния (только v2).
"""),
        code("""
keys = ["composite", "absorbed_units", "inside_units", "returns", "free_after_case", "free_after_custody",
        "no_free_next_day", "started_inside", "end_from_case", "end_from_composite", "events", "people", "activities"]
pd.DataFrame({"v2": [rep.get(k) for k in keys]}, index=keys)
"""),
        code("""
acts = pd.concat({"v1": log1.activity.value_counts(), "v2": log2.activity.value_counts()}, axis=1).fillna(0).astype(int)
acts["разница"] = acts.v2 - acts.v1
acts.sort_values("v2", ascending=False)
"""),
        md("""
## 3. Все майнеры: train/test 80/20, token replay

- люди делятся 80/20 (seed 42): сеть строится на train, метрики — на train и test;
- `fitness_tbr_train` / `fitness_tbr_test` — fitness token-based replay; `pct_fit_traces_tbr_test` — % трасс test без ошибок;
- `precision_etc_test` — precision ETC на test;
- `precision_norm_test = (P − P_flower) / (1 − P_flower)`: 0 — не лучше цветочной сети, 1 — максимум;
- `f1_tbr = 2·fitness_test·precision / (fitness_test + precision)`.
"""),
        code("""
cols = [k for k in ["model", "places", "transitions", "silent", "fitness_tbr_train", "fitness_tbr_test",
                    "pct_fit_traces_tbr_test", "precision_etc_test", "precision_norm_test", "f1_tbr", "error"]
        if k in q2.columns]
q2[cols].sort_values("f1_tbr", ascending=False).set_index("model")
"""),
        md("""
## 4. Сравнение с версией 1 (лучшие модели, train/test)

Версия 1 — `out_states\\<эксперимент>\\split_eval.jsonl` (флаги 15 / 365, старое правило `end`).
Разбиение то же (seed 42), но набор людей может немного отличаться: в окне 2013 трассы без событий
в окне отбрасываются, а это зависит от правил журнала.
"""),
        code("""
se = pd.DataFrame([json.loads(l) for l in (V1 / "split_eval.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()])


def prep(q, fa="fitness_tbr_test"):
    pf = float(q.loc[q.model == "flower", "precision_etc_test"].iloc[0])
    q = q.set_index("model").reindex(BEST)
    out = pd.DataFrame({"fitness TBR test": q[fa], "precision test": q["precision_etc_test"]})
    out["precision норм."] = (out["precision test"] - pf) / (1 - pf)
    out["F1 TBR"] = 2 * out["fitness TBR test"] * out["precision test"] / (out["fitness TBR test"] + out["precision test"])
    out.loc["flower", ["precision норм.", "F1 TBR"]] = None
    return out


cmp = pd.concat({"v1": prep(se), "v2": prep(q2)}, axis=1)
for m in ["fitness TBR test", "precision test", "precision норм.", "F1 TBR"]:
    cmp[("v2 − v1", m)] = cmp[("v2", m)] - cmp[("v1", m)]
cmp.round(3)
"""),
        md("""
## 5. Сети лучших моделей (построены на train) и DFG (весь журнал)
"""),
        code("""
for m in BEST[1:]:
    p = V2 / f"net_{m}.png"
    if p.exists():
        print(m)
        display(Image(filename=str(p), width=1000))
p = V2 / "dfg_frequency.png"
if p.exists():
    print("DFG, частоты (весь журнал v2)")
    display(Image(filename=str(p), width=1000))
"""),
    ]
    if concl:
        cells.append(md(concl))
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    path = HERE / f"petri_states_v2_results_{key}.ipynb"
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("сохранено", path)


if __name__ == "__main__":
    for k in CFG:
        build(k)
