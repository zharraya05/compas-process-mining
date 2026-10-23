# stage1/glue7

Этап 1, вариант «склейка ±7 дней по общей статье». Дата: 10.10.2026.

## Как открыть

```
D:\Git\diploma\.venv\Scripts\activate
cd "D:\diploma\jupyter lab"
jupyter lab
```

Открыть `stage1/glue7/stage1_glue7.ipynb`, *Run → Run All Cells* (около минуты). Входные данные — таблица этапа 0 `../../stage0 - data/out/compas_custody_charges.csv` и `../../stage0 - data/compas.db` (статья `statute`). Ноутбук проверен: пересчёт совпадает с сохранёнными `log_info.csv`, `event_log.csv` и `transition_probs.csv` (разница 0).

Флаги в шаге 0: `RUN_MODELS` (сети Петри IM / HM / DFG, меньше минуты), `RUN_EVAL` (fitness/precision HM dep 0.9, несколько минут; для всех моделей часы, поэтому таблица качества берётся из сохранённого файла); `GLUE_RULE` = `"all_new"` (glue_all) или `"no_oldcase"` (glue_noold).

## Что в папке

| Файл / папка | Что там |
|---|---|
| `stage1_glue7.ipynb` | ноутбук: подготовка эпизодов, журнал PM4Py, матрица переходов, граф, DFG, развилки, причины переходов, IM / HM / DFG, сравнение вариантов |
| `code/` | `stage1_common.py` (рисунок матрицы, V Крамера, оценка моделей), `crime_categories.py` (15 типов), `stage1_glue.py` (исходный скрипт варианта, для справки) |
| `results_glue_all/` | сохранённые результаты: `event_log.csv`, `tables/`, `figures/` (с ними сверяется пересчёт) |
| `results_glue_noold/` | сохранённые результаты: `event_log.csv`, `tables/`, `figures/` (с ними сверяется пересчёт) |
| `notebook_out/` | появляется после запуска: журнал событий, таблицы и рисунки, пересчитанные ноутбуком |
