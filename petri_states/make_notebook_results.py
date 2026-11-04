"""Собирает два ноутбука со всеми результатами журнала состояний:
    petri_states_results_2013.ipynb — окно с 2013-01-01
    petri_states_results_all.ipynb  — вся история, без окна
Запуск из папки «петри»:  python make_notebook_results.py
Ноутбуки только читают готовые файлы из out_states\\ (ничего не пересчитывают)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

CFG = {
    "2013": dict(title="окно с 2013-01-01", limit=120, retry=600, sweep="2013-01-01", base="align_ckpt",
                 eval_nb="petri_states_eval_2013.ipynb"),
    "all": dict(title="вся история (без окна)", limit=20, retry=120, sweep="all", base="align_ckpt_20s",
                eval_nb="petri_states_eval_all.ipynb"),
}

CONCL = {
    "2013": r"""
## 7. Выводы (окно 2013)

**Лучшая модель — сеть из DFG с фильтром p ≥ 0.05.**
- alignment fitness на test — 0.964, precision ETC — 0.759, нормированная — 0.625;
- F1 по alignments — 0.849, лучший среди моделей;
- не уложилась в лимит одна трасса (99 событий) даже при 600 с — на итог не влияет.

**Остальные модели:**
- **HM 0.95** — fitness 0.944, но precision 0.602 (нормированная 0.381): разрешает много лишнего;
- **Split Miner** — precision 0.773 (нормированная 0.647, выше всех), но fitness 0.633:
  отбрасывает много поведения, почти ни одна трасса не проходит без ошибок;
- **IM 0.2** — fitness 1.0, но precision почти как у цветочной сети (нормированная 0.024):
  модель почти ничего не запрещает.

**Переобучения нет:** TBR fitness на train и test различается не больше чем на 0.003.

**Флаги:** по перебору на окне 2013 метрики почти не зависят от `SHORT_CUSTODY_DAYS`;
по `LONG_CUSTODY_DAYS` от 180 до 365 дней — плато. Рекомендация: `LONG` = 180, `SHORT` = 15
(подробно — `petri_states_flags.ipynb`). Все таблицы выше посчитаны с флагами 15 / 365.

*Досчёт тайм-аутов остановлен 04.11 (раздел 3): у DFG p0.05, HM 0.95 и Split Miner не досчитано
не больше одной трассы, у IM 0.2 — 117 (5%); вывод по IM 0.2 от этого не изменится (его fitness и так 1.0).*
""",
    "all": r"""
## 7. Выводы (вся история) — промежуточные

*Досчёт тайм-аутов с лимитом 120 с остановлен 04.11 (раздел 3): числа alignments ниже — с лимитом 20 с;
без учёта остаются 13% трасс test у DFG p0.05, 19% у Split Miner и 36% у IM 0.2.*

**DFG p0.05** — по-прежнему лучший баланс:
- alignment fitness 0.936 (без 13% трасс с тайм-аутом), precision ETC 0.620, нормированная 0.531;
- TBR fitness на train и test совпадает (0.957 / 0.958) — переобучения нет.

**Split Miner** — самая высокая precision (0.884, нормированная 0.857), но сеть почти не воспроизводит
трассы: TBR fitness 0.720, alignment fitness 0.154. Выровненные трассы проходят сеть только
по тихим переходам — видимая часть сети почти не доходит до конца. Для alignments сеть непригодна.

**HM 0.95** — сеть не easy sound (финальная разметка недостижима): alignments не определены;
TBR fitness 0.984, но precision 0.358 (нормированная 0.209).

**IM 0.2** — alignment fitness 0.959 (с частично досчитанными тайм-аутами — 0.956), precision 0.453 (нормированная 0.326);
тайм-аутов 39% (после частичного досчёта — 36%) — на alignment fitness опираться нельзя.

**Вся история сложнее окна 2013:** precision цветочной сети 0.189 против 0.358 —
состояний больше и они разнообразнее, поэтому все модели здесь слабее.
""",
}


def md(src):
    lines = src.strip("\n").split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def code(src):
    lines = src.strip("\n").split("\n")
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}


def build(key):
    c = CFG[key]
    cells = [
        md(f"""
# Журнал состояний: все результаты — {c['title']}

Ноутбук собирает готовые результаты из `out_states\\{key}\\` и `out_states\\flags\\`.
Ничего не пересчитывает: после досчёта тайм-аутов достаточно выполнить его заново.

**Флаги журнала** во всех расчётах, кроме перебора флагов:
`SHORT_CUSTODY_DAYS` = 15, `LONG_CUSTODY_DAYS` = 365.

| Раздел | Что | Где посчитано |
|---|---|---|
| 1 | Все майнеры, сеть строится и проверяется на всём журнале | `petri_states.ipynb`, шаг 3 |
| 2 | Лучшие майнеры, люди train/test 80/20, alignments | `{c['eval_nb']}`, `run_split.py` |
| 3 | Досчёт тайм-аутов alignments с большим лимитом | `retry_timeouts.py` |
| 4 | Сводка по лучшим моделям | собирается здесь из 1–3 |
| 5 | Рисунки сетей лучших моделей и DFG | `petri_states.ipynb`, шаг 3 |
| 6 | Флаги `SHORT` / `LONG` | `petri_states_flags.ipynb` |
| 7 | Выводы | |
"""),
        code(f"""
import json
from pathlib import Path
import pandas as pd
from IPython.display import display, Image

EXP = "{key}"
OUT = Path("out_states") / EXP
FLAGS = Path("out_states") / "flags"
BEST = ["flower", "DFG_p0.05", "HM_dep0.95", "SplitMiner", "IM_noise0.2"]
pd.set_option("display.precision", 3)
pd.set_option("display.max_columns", 40)


def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()])


mq = pd.read_csv(OUT / "model_quality.csv")
se = read_jsonl(OUT / "split_eval.jsonl")
rt = pd.concat([read_jsonl(p) for p in sorted(OUT.glob("split_eval_retry*s.jsonl"))] or [pd.DataFrame()],
               ignore_index=True)
print("файлы:", OUT / "model_quality.csv", OUT / "split_eval.jsonl",
      *[p.name for p in sorted(OUT.glob("split_eval_retry*s.jsonl"))], sep="\\n  ")
"""),
        md("""
## 1. Все майнеры: сеть строится и проверяется на всём журнале

Без разбиения train/test и без alignments: насколько модель описывает данные, а не прогноз.

- `fitness_tbr` — fitness token-based replay; `pct_fit_traces_tbr` — % трасс, прошедших без ошибок;
- `precision_etc` — precision ETC (escaping edges);
- `precision_norm = (P − P_flower) / (1 − P_flower)`: 0 — не лучше цветочной сети, 1 — максимум;
- `f1 = 2·fitness·precision / (fitness + precision)`.
"""),
        code("""
cols = ["model", "places", "transitions", "silent", "fitness_tbr", "pct_fit_traces_tbr",
        "precision_etc", "precision_flower", "precision_norm", "f1"]
mq[cols].sort_values("f1", ascending=False).set_index("model")
"""),
        md(f"""
## 2. Лучшие майнеры: train/test 80/20 и alignments

- люди делятся 80/20 (seed 42): сеть строится на train, метрики считаются на test;
- alignments — на **всём** test, лимит **{c['limit']} с** на вариант трассы;
  варианты, не уложившиеся в лимит, в alignment fitness не входят (столбец `align_timeout`);
- перед alignments проверяется easy soundness: если финальная разметка недостижима, alignments не определены.
"""),
        code("""
cols = [k for k in ["model", "places", "transitions", "silent", "train_traces", "test_traces",
                    "fitness_tbr_train", "fitness_tbr_test", "pct_fit_traces_tbr_test", "precision_etc_test",
                    "easy_sound", "fitness_align_test", "pct_fit_traces_align_test", "align_traces",
                    "align_timeout", "align_max_sec_trace", "note"] if k in se.columns]
se[cols].set_index("model")
"""),
        md(f"""
## 3. Попытка досчёта alignments с большим лимитом

**Как шёл расчёт:**
1. Alignments на всём test с лимитом **{c['limit']} с** на вариант трассы (раздел 2); варианты, не уложившиеся в лимит, —
   тайм-ауты, в alignment fitness они не входят.
2. Тайм-ауты пересчитывались с лимитом **{c['retry']} с** (`retry_timeouts.py`, папка `{c['base']}` → `align_ckpt_retry{c['retry']}s`);
   готовые варианты не пересчитывались, порядок моделей: DFG p0.05 → Split Miner → IM 0.2.
3. **Досчёт остановлен 04.11.2026 в 20:42** — ресурсы отданы пересчёту журнала версии 2
   (`petri_states_v2_results_{{2013,all}}.ipynb`). Контрольные точки сохранены: при запуске `retry_timeouts.py`
   досчёт продолжится с места.

Таблица ниже собирается по контрольным точкам:
- `тайм-аутов при лимите` — трасс test, не уложившихся в исходный лимит;
- `перепроверено вариантов` / `выровнено трасс` — сколько из них пересчитано с большим лимитом и сколько выровнялось;
- `не досчитано трасс`, `доля test` — сколько трасс так и не вошло в alignment fitness;
- `alignment fitness сейчас` — с учётом всего, что досчитано (в том числе для моделей, досчёт которых не закончен).

HM 0.95 на всей истории — не тайм-ауты: сеть не easy sound, alignments для неё не определены.
"""),
        code(f"""
BASE = OUT / "{c['base']}"
RETRY = OUT / "align_ckpt_retry{c['retry']}s"
rows = []
for m in BEST[1:]:
    b = read_jsonl(BASE / f"{{m}}_align.jsonl")
    if b.empty:
        continue
    r = read_jsonl(RETRY / f"{{m}}_align.jsonl")
    to = b[b.timeout]
    rr = r.set_index("i") if not r.empty else pd.DataFrame(columns=["timeout", "n"])
    ok_retry = rr[~rr.timeout.astype(bool)] if len(rr) else rr
    allr = pd.concat([b[~b.timeout].set_index("i"), ok_retry])
    sound = se.set_index("model").get("easy_sound", pd.Series(dtype=object)).get(m)
    fit = None if allr.empty or sound is False else 1 - (allr.cost * allr.n).sum() / (allr.bwc * allr.n).sum()
    not_done = int(to.n.sum() - ok_retry.n.sum()) if len(rr) else int(to.n.sum())
    rows.append({{"модель": m, "трасс test": int(b.n.sum()),
                 "тайм-аутов при лимите": int(to.n.sum()), "вариантов с тайм-аутом": len(to),
                 "перепроверено вариантов": len(rr), "выровнено трасс": int(ok_retry.n.sum()) if len(rr) else 0,
                 "не досчитано трасс": not_done, "доля test, %": round(100 * not_done / b.n.sum(), 1),
                 "alignment fitness сейчас": fit,
                 "примечание": "не easy sound — alignments не определены" if sound is False else "тайм-аутов нет" if to.empty else
                               ("досчёт закончен" if m in set(rt.get("model", [])) else
                                ("досчёт остановлен" if len(rr) else "досчёт не начинался"))}})
pd.DataFrame(rows).set_index("модель").round(3)
"""),
        code("""
# модели, для которых досчёт закончен: итог (лимит, сколько тайм-аутов было и осталось)
if not rt.empty:
    display(rt[["model", "retry_of", "align_max_sec_trace", "align_timeout_before", "align_timeout",
                "align_traces", "fitness_align_test", "pct_fit_traces_align_test", "finished"]].set_index("model"))
"""),
        md("""
## 4. Сводка по лучшим моделям

Весь журнал (раздел 1) + test (раздел 2) + итоговые alignments: если для модели есть досчёт
с большим лимитом (раздел 3), берётся он. Precision нормируется на цветочную сеть **на test**.
"""),
        code("""
P_FLOWER = se.loc[se.model == "flower", "precision_etc_test"].iloc[0]
rows = []
for m in BEST:
    q = mq.set_index("model").loc[m] if m in set(mq.model) else None
    s = se.set_index("model").loc[m]
    r = rt[rt.model == m].sort_values("align_max_sec_trace").iloc[-1] if (not rt.empty and m in set(rt.model)) else None
    a = r if r is not None else s
    fa = a.get("fitness_align_test")
    p = s["precision_etc_test"]
    rows.append({
        "модель": m,
        "F1 весь журнал": None if q is None else q["f1"],
        "fitness TBR train": s["fitness_tbr_train"],
        "fitness TBR test": s["fitness_tbr_test"],
        "precision test": p,
        "precision норм. test": None if m == "flower" else (p - P_FLOWER) / (1 - P_FLOWER),
        "F1 TBR test": None if m == "flower" else 2 * s["fitness_tbr_test"] * p / (s["fitness_tbr_test"] + p),
        "easy sound": s.get("easy_sound"),
        "alignment fitness test": None if m == "flower" else fa,
        "F1 alignments test": None if (m == "flower" or pd.isna(fa)) else 2 * fa * p / (fa + p),
        "лимит, с": None if m == "flower" else a.get("align_max_sec_trace"),
        "доля тайм-аутов": None if (m == "flower" or pd.isna(a.get("align_timeout"))) else a["align_timeout"] / s["test_traces"],
    })
summary = pd.DataFrame(rows).set_index("модель")
summary["доля тайм-аутов"] = (summary["доля тайм-аутов"] * 100).round(1)
summary = summary.rename(columns={"доля тайм-аутов": "тайм-аутов, %"})
summary.round(3)
"""),
        md("""
## 5. Сети лучших моделей и DFG

Сети построены на всём журнале (раздел 1). DFG — все пары соседних состояний с частотами, без фильтра.
"""),
        code("""
for m in BEST[1:]:
    p = OUT / f"net_{m}.png"
    if p.exists():
        print(m)
        display(Image(filename=str(p), width=1000))
p = OUT / "dfg_frequency.png"
if p.exists():
    print("DFG, частоты")
    display(Image(filename=str(p), width=1000))
"""),
        md(f"""
## 6. Флаги `SHORT_CUSTODY_DAYS` и `LONG_CUSTODY_DAYS`

Перебор флагов (`petri_states_flags.ipynb`, шаг 5): сеть на train, на test — TBR fitness и ETC precision,
без alignments; `long = 100000` — правило 5 выключено. Модели: DFG p0.05, HM 0.95, Split Miner.
"""),
        code(f"""
sw = pd.read_csv(FLAGS / "flags_sweep_table.csv")
sw = sw[sw["exp"].astype(str) == "{c['sweep']}"].drop(columns="exp").sort_values(["seed", "short", "long"])
sw.set_index(["seed", "short", "long"])
"""),
    ]
    if key == "2013":
        cells += [
            md("Медианы времени между состояниями журнала (окно 2013, флаги 15 / 365), дни:"),
            code('pd.read_csv(FLAGS / "transition_medians_current.csv").set_index("переход")'),
        ]
    else:
        cells += [
            md("""
Длительность эпизодов заключения по зонам правил (вся база), дни: при текущих флагах (15 / 365)
и рекомендованных (15 / 180).
"""),
            code("""
display(pd.read_csv(FLAGS / "domain_medians_current.csv").set_index(pd.read_csv(FLAGS / "domain_medians_current.csv").columns[0]))
display(pd.read_csv(FLAGS / "domain_medians_recommended.csv").set_index(pd.read_csv(FLAGS / "domain_medians_recommended.csv").columns[0]))
"""),
            md("""
Профиль по длительности эпизода (вся база):
- **LONG** — доля эпизодов с prison и с делом в день входа: где задержание по делу становится «сроком»;
- **SHORT** — события внутри срока: доля на следующий день после входа и медиана дней от входа.
"""),
            code("""
display(pd.read_csv(FLAGS / "long_profile.csv").set_index("dur"))
display(pd.read_csv(FLAGS / "short_profile.csv").set_index("dur"))
"""),
        ]
    cells.append(md(CONCL[key]))
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    path = HERE / f"petri_states_results_{key}.ipynb"
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("сохранено", path)


if __name__ == "__main__":
    for k in CFG:
        build(k)
