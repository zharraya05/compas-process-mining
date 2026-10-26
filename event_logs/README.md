# Таблицы логов для PM4Py (из compas_custody_charges)

Собраны скриптом `build_event_logs.py` из `compas_custody_charges.csv` по тем же правилам, что Этап 1:
склейка jail→prison в тот же день и заключений одного дня, 15 типов преступлений (`crime_categories.py`),
шаг Justice после остальных типов того же заключения, без состояния END (последний шаг человека
цензурирован концом наблюдения 2016-04-19). Проверено: пары (человек, тип, время) совпадают
один в один с `event_log.csv` Этапа 1 во всех трёх вариантах.

## Файлы

| Файл | Что внутри |
|---|---|
| `event_log_oldcase.csv` / `.xlsx` | **основной вариант**: заключение только из старых дел = состояние `OldCase` |
| `event_log_no_oldcase.csv` / `.xlsx` | такие заключения убраны из последовательности человека |
| `event_log_all_new.csv` / `.xlsx` | все обвинения считаются новыми (старые дела не выделяются) |
| `steps_<вариант>.csv` | та же информация, одна строка на шаг (заключение), удобно для анализа выживаемости |
| `log_summary.csv` | число людей, эпизодов, событий, переходов по вариантам |
| `build_event_logs.py`, `crime_categories.py` | пересборка: `python build_event_logs.py compas_custody_charges.csv` (флаг `--xes` дописывает XES, ~50 МБ на вариант) |
| `load_example.py` | пример загрузки в PM4Py |

В `.xlsx` два листа: `events` (= csv) и `steps`.

| Вариант | Людей | Эпизодов | Шагов | Событий | Переходов | Активностей |
|---|---|---|---|---|---|---|
| oldcase | 10 580 | 21 716 | 21 817 | 29 371 | 11 237 | 16 (15 типов + OldCase) |
| no_oldcase | 10 580 | 17 646 | 17 747 | 25 301 | 7 167 | 15 |
| all_new | 10 580 | 21 716 | 21 866 | 33 364 | 11 286 | 15 |

## Строка = событие

Событие = один тип преступления внутри одного шага. Если в заключении несколько типов, событий несколько,
порядок между ними случайный (seed 42), к времени прибавлены 0, 1, 2… секунды, чтобы PM4Py этот порядок сохранил.

| Колонка | Смысл |
|---|---|
| `case:concept:name` | id человека (person_id из compas.db), строкой — case для PM4Py |
| `concept:name` | активность = тип преступления (или `OldCase`) |
| `time:timestamp` | дата заключения (in_custody) + секунды порядка, UTC |
| `id` | id человека числом |
| `episode_id` | номер заключения после склейки (общий для всех событий и шага Justice этого заключения) |
| `step` | 0 — основной шаг, 1 — отдельный шаг Justice после остальных типов |
| `step_idx`, `n_steps` | номер шага у человека (1, 2, …) и сколько шагов всего |
| `order_in_episode` | порядок события внутри заключения (0, 1, …) |
| `in_custody`, `out_custody` | даты начала и конца заключения; у ещё не вышедших out = 2016-04-19 |
| `ongoing` | на конец наблюдения человек ещё в заключении |
| `dur_days` | длительность заключения, дней |
| `custodies`, `sources` | сколько исходных записей склеено и откуда они (`jail`, `prison`, `jail+prison`) |
| `state` | состояние шага = множество типов через `+` (например `Drug+Traffic`) |
| `n_types` | число типов в шаге |
| `prev_state`, `next_state` | состояние предыдущего и следующего шага (пусто у первого и последнего) |
| `gap_prev_days`, `gap_next_days` | дней от выхода из предыдущего заключения до этого / от выхода до следующего (0 для шага Justice) |
| `censored` | последний шаг человека: следующее событие не наблюдалось |
| `followup_days` | для цензурированных: дней от выхода до 2016-04-19 (время до цензуры) |
| `type_degree_class` | F (felony) / M (misdemeanor) по самому тяжёлому обвинению этого типа; пусто — степень неизвестна |
| `type_degrees` | все степени обвинений этого типа (F3; M1 …) |
| `type_n_charges` | число обвинений этого типа в заключении |
| `type_case_numbers` | номера дел этих обвинений (у `OldCase` — старые дела) |
| `type_charges` | формулировки обвинений (c_charge_desc) |
| `charge_class` | Felony / Misdemeanor / Other/NA по самому тяжёлому обвинению шага |
| `n_charges_all`, `n_old_charges` | всего обвинений в заключении и сколько из них по старым делам |
| `has_old_case`, `only_old_case` | есть ли в заключении старые дела / только старые |
| `case_numbers_all`, `old_case_numbers` | все номера дел заключения / только старые |
| `age`, `sex`, `race` | из compas.db people |

`steps_<вариант>.csv` содержит те же колонки уровня шага (без `case:concept:name`, `concept:name`,
`time:timestamp`, `order_in_episode` и колонок `type_*`).

## Загрузка в PM4Py (Jupyter)

```python
import pandas as pd
import pm4py

df = pd.read_csv("event_log_oldcase.csv", parse_dates=["time:timestamp", "in_custody", "out_custody"])
df["case:concept:name"] = df["case:concept:name"].astype(str)
log = pm4py.format_dataframe(df, case_id="case:concept:name", activity_key="concept:name",
                             timestamp_key="time:timestamp")

dfg, sa, ea = pm4py.discover_dfg(log)
pm4py.view_dfg(dfg, sa, ea)

net, im, fm = pm4py.discover_petri_net_heuristics(log, dependency_threshold=0.9)
pm4py.view_petri_net(net, im, fm)

# только одноэлементные шаги, без развилок:
# log1 = log[log["n_types"] == 1]
# XES, если нужен:
# pm4py.write_xes(log, "event_log_oldcase.xes")
```

Проверено здесь на pm4py 2.7.23.9: загружается, DFG 207 дуг, Heuristics Miner 0.9 строится.
