"""Генератор petri_states_flags.ipynb: медианы на переходах и подбор флагов SHORT/LONG_CUSTODY_DAYS.
Запуск: python make_notebook_flags.py"""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md(r"""
# Журнал состояний: подбор флагов `SHORT_CUSTODY_DAYS` и `LONG_CUSTODY_DAYS`

К `petri_states.ipynb`, `petri_states_eval_2013.ipynb`, `petri_states_eval_all.ipynb` (04.11.2026).

| Флаг | Было | Правило |
|---|---|---|
| `SHORT_CUSTODY_DAYS` | 15 | 6: заключение короче — аресты/обвинения внутри срока склеиваются с ним (`jail` → `charge+jail`); иначе правило 7: `jail → дело → jail` |
| `LONG_CUSTODY_DAYS` | 365 | 5: заключение не короче — аресты/обвинения дня входа отдельным состоянием перед ним; иначе правило 3: составное `arrest+jail` |

Шаги:
1. медианы длительности заключения по зонам правил 3–7;
2. медианы времени на переходах журнала состояний (окно 2013, текущие флаги);
3. граница `LONG` по данным — где эпизод из «задержания по делу дня входа» становится «сроком»;
4. граница `SHORT` по данным — до какой длительности события внутри срока — продолжение оформления задержания;
5. перебор флагов по качеству сетей (train/test 80/20, TBR fitness, ETC precision; без alignments).
""")

md("## Шаг 0. Настройки")
code(r"""
from pathlib import Path

UP = Path("..")
DB_CANDIDATES = [UP / "stage0 - data" / "compas.db", Path(r"D:\diploma\compas.db"), Path("compas.db")]
CATEGORIES_DIRS = [Path("."), UP / "stage0 ver2 from charge"]

SHORT_CUSTODY_DAYS = 15          # текущие значения флагов
LONG_CUSTODY_DAYS = 365
LOG_START = "2013-01-01"         # окно для шага 2 и перебора

SWEEP_SHORT = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 90, 120, 180]     # при LONG = 365
SWEEP_LONG = [7, 15, 30, 60, 90, 120, 180, 270, 365, 545, 730, 100000]     # при SHORT = 15
RUN_SWEEP = False                # True — досчитать перебор (≈ 2 мин на пару флагов); False — взять из файла
OUT = Path("out_states") / "flags"
SWEEP_FILE = OUT / "flags_sweep.jsonl"
""")
code(r"""
import sys, warnings
import numpy as np
import pandas as pd
from IPython.display import display

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path.cwd()))
import petri_pipeline as pp
import petri_states as ps
import flags_analysis as fa

pd.set_option("display.width", 250)
OUT.mkdir(parents=True, exist_ok=True)
DB = pp.find_db(DB_CANDIDATES)
categorize = pp.import_categorizer(CATEGORIES_DIRS)
tables, _ = pp.drop_full_duplicates(pp.load_tables(DB))
tables, _, _ = pp.drop_invalid_stays(tables)
hist = pp.build_history_table(tables, categorize)
obs_end = pp.observation_end(hist)
eps, inside = fa.episode_profile(hist, obs_end)
print(f"эпизодов заключения: {len(eps):,}; единиц дел внутри срока: {len(inside):,} в {inside.episode_id.nunique():,} эпизодах")
""")

md(r"""
## Шаг 1. Медианы длительности заключения по зонам правил

Эпизод — склеенные пребывания jail/prison (как в `petri_states.build_state_log`); кто сидит на конец наблюдения — до конца наблюдения. Дни.
""")
code(r"""
t1 = fa.domain_medians(eps, SHORT_CUSTODY_DAYS, LONG_CUSTODY_DAYS)
t1.to_csv(OUT / "domain_medians_current.csv")
display(t1)
""")

md(r"""
## Шаг 2. Медианы на переходах журнала состояний (окно 2013, текущие флаги)

Время от состояния до следующего (`dt_next_days`), дни.
""")
code(r"""
log, _, rep = ps.build_state_log(hist, obs_end=obs_end, log_start=LOG_START,
                                 short_days=SHORT_CUSTODY_DAYS, long_days=LONG_CUSTODY_DAYS)
t2 = fa.transition_medians(log)
t2.to_csv(OUT / "transition_medians_current.csv")
display(t2)
""")

md(r"""
## Шаг 3. Граница `LONG_CUSTODY_DAYS` по данным

По длительности эпизода: доля эпизодов с prison и доля с делом в день входа. Короткие эпизоды — задержание по делу дня входа (jail), длинные — срок (prison), дела в день входа обычно нет. Правило 5 нужно там, где начинается «срок».
""")
code(r"""
t3 = fa.long_profile(eps)
t3.to_csv(OUT / "long_profile.csv")
display(t3)
print("prison > 50% — с", fa.first_crossing(t3, "доля_prison", above=True), "дней;",
      "дело в день входа < 50% — с", fa.first_crossing(t3, "доля_с_делом_в_день_входа", above=False), "дней")
""")

md(r"""
## Шаг 4. Граница `SHORT_CUSTODY_DAYS` по данным

События внутри срока по длительности заключения: доля на следующий день после входа (продолжение оформления задержания — склейка уместна) и медиана дней от входа.
""")
code(r"""
t4 = fa.short_profile(inside)
t4.to_csv(OUT / "short_profile.csv")
display(t4)
b = t4[t4["доля_на_след_день"] < 0.5]
print("на следующий день < 50% — с", b.index[0].left if len(b) else None, "дней")
""")

md(r"""
## Шаг 5. Перебор флагов по качеству сетей

Окно 2013, люди train/test 80/20 (seed 42; часть пар — ещё seed 7 и вся история). Сеть на train, на test — TBR fitness и ETC precision; `F1 = 2·f·p/(f+p)`, `Pnorm` — precision, нормированная на цветочную сеть. Alignments не считаются. Модели: DFG p0.05, HM 0.95, Split Miner.
""")
code(r"""
if RUN_SWEEP:
    cfg = [(s, 365) for s in SWEEP_SHORT] + [(15, l) for l in SWEEP_LONG]
    fa.sweep(hist, cfg, SWEEP_FILE, log_start=LOG_START)
t5 = fa.sweep_table(SWEEP_FILE)
t5.to_csv(OUT / "flags_sweep_table.csv")
idx = pd.IndexSlice
print("SHORT при LONG = 365 (окно 2013, seed 42)")
display(t5.loc[idx["2013-01-01", 42, :, 365], :].droplevel([0, 1]))
print("LONG при SHORT = 15 (окно 2013, seed 42)")
display(t5.loc[idx["2013-01-01", 42, 15, :], :].droplevel([0, 1]))
print("проверка: seed 7 и вся история")
display(t5.loc[idx[:, :, :, :], :].query("seed == 7 or exp == 'all'"))
""")

md(r"""
## Шаг 6. Медианы при рекомендованных флагах
""")
code(r"""
NEW_SHORT, NEW_LONG = 15, 180
t6 = fa.domain_medians(eps, NEW_SHORT, NEW_LONG)
t6.to_csv(OUT / "domain_medians_recommended.csv")
display(t6)
""")

md(r"""
## Выводы

**`LONG_CUSTODY_DAYS`: 365 → 180.**
- По данным граница между «задержанием по делу дня входа» и «сроком» — 180 дней: с 180 дней доля эпизодов с prison больше 50%, а доля эпизодов с делом в день входа меньше 50% (шаг 3).
- При 365 правило 5 срабатывает в 327 эпизодах, медиана их длительности — 610 дней (87% с prison). При 180 — в 749 эпизодах, медиана 331 день (53% с prison). Остальные эпизоды с одним делом в день входа (составное состояние, правило 3): медиана 2 дня. На переходах (шаг 2): дело → заключение в тот же день — медиана 0.3 дня (7 ч от даты дела до входа), составное состояние → следующее — 2 дня.
- Метрики (шаг 5): от 180 до 365 дней — плато (средний F1 0.791–0.793), меньше 90 дней — F1 DFG и HM ниже, 545 дней и больше — F1 Split Miner падает с 0.78 до 0.72. По всей истории 180 немного лучше 365 (средний F1 0.692 против 0.690), с seed 7 — на уровне шума (0.7906 против 0.7916). Получается, метрики допускают весь диапазон 180–365, а выбор внутри него делается по данным.

**`SHORT_CUSTODY_DAYS`: оставить 15.**
- События внутри срока при заключении до 15 дней — в основном на следующий день после входа (56–94%) с медианой 1 день от входа: это продолжение оформления задержания, склейка уместна. С 15 дней доля на следующий день меньше 50%, медиана — 3 дня и больше (шаг 4).
- В зоне правила 6 (короче 15 дней) — 154 эпизода, медиана 6 дней; в зоне правила 7 — 1 141 эпизод, медиана 157 дней. На переходах: дело внутри срока → возврат в заключение — медиана 1 день, возврат → следующее — 53 дня (шаг 2).
- Метрики к `SHORT` почти не чувствительны: от 0 до 60 дней средний F1 меняется меньше чем на 0.005, и лучшая точка разная у разных seed (затронуто до 549 единиц из ~117 тыс. состояний). То есть метрики выбор не определяют, решает профиль данных.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
Path("petri_states_flags.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("petri_states_flags.ipynb")
