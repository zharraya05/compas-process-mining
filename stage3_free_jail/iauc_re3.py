"""IAUC_RE3 - интегральный AUC для повторных событий (подход 3).

Копия класса IAUCRE3 из репозитория прошлой работы, чтобы ноутбук не зависел от внешней папки:
    репозиторий: https://github.com/DKoleda6/recurrent_event
    файл:        metrics/iauc_re3.py
    коммит:      e96b65e (20.03.2026, "UPD metrics & experiments"); копия снята 20.11.2026 с клона на коммите 8a6bb21
    лицензии в репозитории нет - при использовании ссылаться на источник.

Отличия от оригинала (расчёт не меняют):
    - класс не наследует BaseMetric (abstract-класс с одним методом compute), чтобы файл был самодостаточным;
    - np.trapz -> np.trapezoid, если np.trapz нет (numpy 2.x).

Что считает: на каждом t сетки люди теста сравниваются попарно - верно, если у кого к моменту t наблюдалось
больше событий (строки с event = 1 и stop <= t), у того больше прогноз (estimate[t] первой строки человека).
Результат - площадь под AUC(t), делённая на длину сетки. Цензура не учитывается.

survival_test - DataFrame со столбцами name (человек), event (0/1), stop (время события на оси человека);
строки человека по порядку, первая строка - первый эпизод.
estimate - прогноз (строки = строки survival_test, столбцы = times), например накопленный риск H = -log S.
survival_train не используется.
"""
import numpy as np
import pandas as pd

_trapz = getattr(np, "trapz", None) or np.trapezoid


class IAUCRE3:
    def compute(self, survival_train, survival_test, estimate, times):
        test_df = survival_test.copy()
        test_df.reset_index(drop=True, inplace=True)
        if not isinstance(estimate, pd.DataFrame):
            estimate = pd.DataFrame(estimate, index=test_df.index, columns=times)

        auc_re_per_time = []
        person_names = test_df["name"].unique()
        pred_by_person = []

        n_people = len(person_names)
        name_to_ind = {name: i for i, name in enumerate(person_names)}
        person_ind = test_df["name"].map(name_to_ind).values
        obs_matrix = np.zeros((len(times), n_people))

        for row_i, row in test_df.iterrows():
            t_mask = times >= row["stop"]
            if row["event"] == 1:
                obs_matrix[t_mask, person_ind[row_i]] += 1
            # else:
            #     obs_matrix[t_mask, person_ind[row_i]] = np.nan

        first_indices = test_df.groupby("name").head(1).index
        estimate_first = estimate.loc[first_indices]

        for t in times:
            pred_sum = pd.Series(
                estimate_first[t].values,
                index=test_df.loc[first_indices, "name"]
            )
            pred_counts = pred_sum.reindex(person_names).values
            pred_by_person.append(pred_counts)

        pred_by_person = np.array(pred_by_person)

        for t_ind in range(len(times)):
            obs_counts = obs_matrix[t_ind]
            pred_counts = pred_by_person[t_ind]
            obs_i = np.tile(obs_counts[:, np.newaxis], (1, len(obs_counts)))
            obs_j = np.tile(obs_counts[np.newaxis, :], (len(obs_counts), 1))
            pred_i = np.tile(pred_counts[:, np.newaxis], (1, len(pred_counts)))
            pred_j = np.tile(pred_counts[np.newaxis, :], (len(pred_counts), 1))

            valid_mask = obs_i > obs_j
            correct_mask = pred_i > pred_j
            denominator = np.sum(valid_mask)
            numerator = np.sum(valid_mask & correct_mask)

            auc_re = 0.5 if denominator == 0 else numerator / denominator
            auc_re_per_time.append(auc_re)

        integral = _trapz(auc_re_per_time, times)
        iauc_re = integral / (times[-1] - times[0]) if len(times) > 1 else 0.5

        return iauc_re
