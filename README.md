# compas-process-mining

Граф переходов между типами преступлений (PM4Py) и модели повторных событий на данных ProPublica COMPAS (округ Броуард, Флорида).
Каждый этап лежит в своей папке: код, ноутбук с пояснениями и результат.

## Структура

| папка | этап |
|---|---|
| [`stage0-data/`](stage0-data/) | Этап 0. Таблица заключений с обвинениями `compas_custody_charges` из `compas.db` |
| [`event_logs/`](event_logs/) | Скрипт построения журналов событий PM4Py (all_new / oldcase / no_oldcase) и их сводка; сами журналы не выложены, скрипт их пересобирает |
| [`stage1/all_new/`](stage1/all_new/) | Этап 1, 15 категорий, все обвинения новые (+ этапы 2–3) |
| [`stage1/oldcase/`](stage1/oldcase/) | Этап 1, 15 категорий + общее состояние OldCase (+ этапы 2–3) |
| [`stage1/no_oldcase/`](stage1/no_oldcase/) | Этап 1, без заключений только по старым делам |
| [`stage1/glue7/`](stage1/glue7/) | Этап 1, склейка заключений ±7 дней по общей статье (glue_all и glue_noold) |
| [`stage1/oldcase_x2/`](stage1/oldcase_x2/) | Этап 1, вариант «категории × 2»: журнал событий, статистика, граф PM4Py; `explained/` — разбор шага 4 на игрушечном журнале |
| [`stage1/x2_exp1_one_event/`](stage1/x2_exp1_one_event/) … [`x2_exp4_miners/`](stage1/x2_exp4_miners/) | Эксперименты по повышению precision для «× 2» (эксп. 1 и 3 — с этапами 2–3) |
| [`stage1/x2_exp_summary/`](stage1/x2_exp_summary/) | Сводка fitness / precision по всем вариантам |
| [`stage1/dasha_severity/`](stage1/dasha_severity/) | Данные Даши, состояние = тяжесть обвинения (+ этапы 2–3) |
| [`stage1/stage23_tools/`](stage1/stage23_tools/) | Общий модуль этапов 2–3 и выводы по вариантам |
| [`reports/`](reports/) | Отчёты и презентации по датам (08.10, 09.10, 16–17.10) |
| [`docs/teacher/`](docs/teacher/) | Пояснения по теории и коду (графы PM4Py, HM / IM, выравнивания, Каплан–Мейер, Кокс, деревья и лес выживаемости) |

## Этап 0 в двух словах

Одна строка = одно обвинение внутри одного заключения одного человека (65 363 строки, 22 552 заключения, 10 580 человек).

- `id` — `person_id` из `casearrest` (он же `people.id`); `age`, `sex`, `race` — из `people`.
- Заключения — `jailhistory` + `prisonhistory` с датой входа с 2013-01-01.
- Обвинения заключения — дела из `casearrest`, арестованные в ±1 день от входа (иначе по дате преступления, иначе последнее дело за 365 дней), все их обвинения из `charge`.
- Производные столбцы (`custody_num`, `time_*`, `event`, `*_dur_custody`, `all_future_*`) считаются по `id`.

Подробно — в [`stage0-data/README.md`](stage0-data/README.md).
