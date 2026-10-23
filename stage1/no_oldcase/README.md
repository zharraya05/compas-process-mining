# stage1/no_oldcase

Этап 1, вариант «без OldCase». Дата: 10.10.2026.

## Как открыть

```
D:\Git\diploma\.venv\Scripts\activate
cd "D:\diploma\jupyter lab"
jupyter lab
```

Открыть `stage1/no_oldcase/stage1_no_oldcase.ipynb`, *Run → Run All Cells* (около минуты). Входные данные — таблица этапа 0 `../../stage0 - data/out/compas_custody_charges.csv`. Ноутбук проверен: пересчёт совпадает с сохранёнными `log_info.csv`, `event_log.csv` и `transition_probs.csv` (разница 0).

Флаги в шаге 0: `RUN_MODELS` (сети Петри IM / HM / DFG, меньше минуты), `RUN_EVAL` (fitness/precision HM dep 0.9, несколько минут; для всех моделей часы, поэтому таблица качества берётся из сохранённого файла).

## Что в папке

| Файл / папка | Что там |
|---|---|
| `stage1_no_oldcase.ipynb` | ноутбук: подготовка эпизодов, журнал PM4Py, матрица переходов, граф, DFG, развилки, причины переходов, IM / HM / DFG, сравнение вариантов |
| `code/` | `stage1_common.py` (рисунок матрицы, V Крамера, оценка моделей), `crime_categories.py` (15 типов), `stage1_no_oldcase.py` (исходный скрипт варианта, для справки) |
| `results/` | сохранённые результаты: `event_log.csv`, `tables/`, `figures/` (с ними сверяется пересчёт) |
| `notebook_out/` | появляется после запуска: журнал событий, таблицы и рисунки, пересчитанные ноутбуком |
