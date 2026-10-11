# Данные Даши: состояние = тяжесть обвинения

Отдельный эксперимент Этапа 1. Журнал по таблице Даши `individual_custody_timeline_rfm.csv`, состояние = степень тяжести обвинения (F1, F2, F3, F_other, M1, M2, Unknown), одно событие на заключение, последнее заключение цензурировано.

Тяжесть берётся для каждого заключения из `compas.db` (самое тяжёлое обвинение ареста), потому что колонка Даши `c_charge_degree` почти у всех одна на человека. Колонка Даши посчитана рядом для сравнения.

Запуск: открыть `dasha_severity.ipynb`, Run All. Входы: `data/individual_custody_timeline_rfm.csv` (или `D:\Git\diploma\recurrent_event\data\...`) и `../../stage0 - data/compas.db`. Код: `dasha_severity.py`; генератор ноутбука `make_dasha.py`.

Итог (test, TBR): по заключению HM 0.9 fitness 0.998, precision 0.937; IM 0.2 0.994 / 0.962. Колонка Даши: 100% переходов петли X → X, HM fitness = precision = 1.
