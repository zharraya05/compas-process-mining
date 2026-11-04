# Сети Петри на журнале состояний (COMPAS DB), 03–04.11.2026

Структура данных базы через сети Петри. Трасса человека — последовательность **состояний**:
на свободе (`free`), в заключении (`jail` / `prison`), под арестом / обвинением по делу
(`arrest`, `charge`, `arrest+charge`) или в составном состоянии «дело + заключение» (`arrest+jail`, …).
Два эксперимента: вся история базы (`all`) и окно с 2013-01-01 (`2013`).

## Ноутбуки

| Ноутбук | Что в нём |
|---|---|
| [`petri_states.ipynb`](petri_states.ipynb) | **основной**: правила журнала состояний, построение журналов, все майнеры на всём журнале (версия 1, флаги 15 / 365) |
| [`petri_states_flags.ipynb`](petri_states_flags.ipynb) | подбор флагов `SHORT_CUSTODY_DAYS` / `LONG_CUSTODY_DAYS` (итог: 15 / 180) |
| [`petri_states_results_2013.ipynb`](petri_states_results_2013.ipynb), [`petri_states_results_all.ipynb`](petri_states_results_all.ipynb) | все результаты версии 1: майнеры на всём журнале, лучшие майнеры на train/test 80/20 с alignments, попытка досчёта тайм-аутов, сводка, сети, флаги, выводы |
| [`petri_states_v2_results_2013.ipynb`](petri_states_v2_results_2013.ipynb), [`petri_states_v2_results_all.ipynb`](petri_states_v2_results_all.ipynb) | **версия 2**: новое правило перехода в `end`, флаги 15 / 180; все майнеры на train/test (token replay, без alignments), сравнение с версией 1, выводы |

Ноутбуки сохранены с результатами. Ноутбуки `*_results_*` ничего не пересчитывают — читают файлы из `out_states/` и `out_states_v2/`.

## Версии журнала

| | Версия 1 | Версия 2 |
|---|---|---|
| Флаги | `SHORT` = 15, `LONG` = 365 | `SHORT` = 15, `LONG` = 180 |
| Переход в `end` | только из `free`, `jail`, `prison` | из `free`, из любого состояния с заключением, из ареста/обвинения — если оно в день конца наблюдения или накануне |
| Оценка | все майнеры на всём журнале; лучшие — train/test + alignments | все майнеры на train/test, TBR fitness и ETC precision |

## Скрипты

| Файл | Что делает |
|---|---|
| `petri_pipeline.py` | таблица истории из базы, эпизоды заключения, майнеры, метрики, рисунки |
| `petri_states.py` | журнал состояний (`build_state_log`, параметр `end_rule="v1"/"v2"`), проверки, оценка на train/test и alignments с контрольными точками |
| `make_notebook_states.py` | генератор `petri_states.ipynb` |
| `make_notebook_states_eval.py` | генератор ноутбуков расчёта train/test + alignments (их результаты — в `*_results_*`) |
| `run_split.py` | расчёт лучших майнеров на train/test с alignments (`python run_split.py <2013\|all> [лимит, с]`) |
| `retry_timeouts.py` | досчёт тайм-аутов alignments с большим лимитом (`python retry_timeouts.py <exp> <старый> <новый> <модели…>`) |
| `flags_analysis.py`, `make_notebook_flags.py` | расчёты и генератор `petri_states_flags.ipynb` |
| `run_states_v2.py` | расчёт версии 2: журналы, все майнеры на train/test |
| `make_notebook_results.py`, `make_notebook_results_v2.py` | генераторы ноутбуков `*_results_*` и `*_v2_results_*` |
| `make_results_tables.py` | те же результаты версии 1 в Excel (`results_states_2013.xlsx`, `results_states_all.xlsx`) |

`run_split.py` и `retry_timeouts.py` запускались в облачной среде: пути к журналам (`/home/claude/states_*.pkl`)
и к папке результатов в них нужно поправить под свою машину.

## Результаты

| Папка | Что там |
|---|---|
| `out_states/<exp>/` | версия 1: `state_log.csv.gz` (журнал), `model_quality.csv` (весь журнал), `split_eval.jsonl` (train/test + alignments), `split_eval_retry*s.jsonl` (досчёт), `align_ckpt*/` (alignments по вариантам трасс), `net_*.png`, `dfg_*.png` |
| `out_states/flags/` | таблицы подбора флагов |
| `out_states_v2/<exp>/` | версия 2: `state_log.csv.gz`, `report.json` (показатели журнала и проверки), `split_quality.csv` (все майнеры на train/test), `discovery_info.csv`, `net_*.png`, `dfg_*.png` |

Журналы сжаты (`state_log.csv.gz`); ноутбуки версии 2 читают их напрямую.

## Как запустить

Нужны `compas.db` (`../stage0-data/compas.db`) и `crime_categories.py` (`../event_logs/crime_categories.py` —
скопировать в эту папку), PM4Py 2.7+ (Split Miner — `pm4py.discover_bpmn_split_miner`) и Graphviz для рисунков.
Пути к базе задаются в `DB_CANDIDATES` в начале ноутбуков и скриптов.
