# Этап 0. Таблица `compas_custody_charges` из `compas.db`

| файл | что это |
|---|---|
| `build_compas_custody_charges.ipynb` | ноутбук: тот же код, что в скрипте, по шагам с пояснениями; в конце — дальнейший план работ |
| `build_from_compas_db.py` | тот же код одним скриптом |
| `compas.db` | исходная база ProPublica (только читается) |
| `compas_custody_charges.csv` / `.xlsx` | результат: 65 363 строки, 25 столбцов, 22 552 заключения, 10 580 человек |
| `COMPAS_DB_TABLE.md` | описание таблицы: откуда берётся каждый столбец |

## Запуск

```bash
pip install pandas numpy openpyxl
python build_from_compas_db.py compas.db .
```

или открыть ноутбук из этой папки и выполнить все ячейки (около 3 минут). Ноутбук пишет результат в `out/`.

Ноутбук и скрипт дают одну и ту же таблицу: проверено 2026-10-10, 0 отличающихся ячеек
(в облаке на pandas 3.0.5 файл совпадает побайтно, MD5 `f29e6a6b957fb3c320c85c0e8aa409b0`; на Windows с pandas 2.3.3 значения те же, отличаются только переводы строк CRLF).

## `compas.db`

Файл взят из репозитория ProPublica https://github.com/propublica/compas-analysis (последний коммит `master` — 2017-06-13, после него база не менялась).
Проверено 2026-10-10: git-объект `compas.db` здесь и в `propublica/compas-analysis@master` один и тот же (`5413d293`), MD5 `32d9cba86f697609ad672c5c73c71c0e`, 62 816 256 байт.
