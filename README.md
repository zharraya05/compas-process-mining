# compas-process-mining

Таблица заключений и обвинений из базы ProPublica COMPAS (округ Броуард, Флорида) для построения графа переходов между типами преступлений (PM4Py) и моделей повторных событий.

## Файлы

| файл | что это |
|---|---|
| `build_from_compas_db.py` | код: строит таблицу из `compas.db` (база только читается) |
| `compas_custody_charges.csv` / `.xlsx` | результат: 65 363 строки, 22 552 заключения, 10 580 человек |
| `COMPAS_DB_TABLE.md` | описание таблицы: откуда берётся каждый столбец |
| `COMPARE_WITH_DASHA.md` | сверка с прежней таблицей `individual_custody_timeline_rfm.csv` и причины расхождений |

## Данные

`compas.db` в репозиторий не входит. Это открытая база ProPublica: https://github.com/propublica/compas-analysis (файл `compas.db`).

## Запуск

```bash
pip install pandas openpyxl
python build_from_compas_db.py path/to/compas.db [папка_для_результата]
```

## Таблица в двух словах

Одна строка = одно обвинение внутри одного заключения одного человека.

- `id` — `person_id` из `casearrest` (он же `people.id`); `age`, `sex`, `race` — из `people`.
- Заключения — `jailhistory` + `prisonhistory` с датой входа с 2013-01-01.
- Обвинения заключения — дела из `casearrest`, арестованные в ±1 день от входа (иначе по дате преступления, иначе последнее дело за 365 дней), все их обвинения из `charge`.
- Производные столбцы (`custody_num`, `time_*`, `event`, `*_dur_custody`, `all_future_*`) считаются по `id`.
- В конце: `charge_num`, `n_charges`, `charge_category`, `charge_source`, `case_number`, `custody_source`.

Подробно — в `COMPAS_DB_TABLE.md`.
