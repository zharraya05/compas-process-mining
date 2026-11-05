"""Метрики на всём журнале с продолжением с места: ps.evaluate_full по одной модели.

Самый тяжёлый по памяти шаг пересчёта. После каждой модели строка пишется на диск, поэтому сбой или нехватка
памяти теряют только текущую модель: повторный запуск ячейки продолжит со следующей. Итог тот же, что у
ps.evaluate_full: precision_norm и f1 считаются по готовой таблице.
"""
import gc
import json
from pathlib import Path

import pandas as pd

import petri_states as ps


def evaluate_full_resumable(models: dict, log_pm, checkpoint, verbose=True) -> pd.DataFrame:
    ck = Path(checkpoint)
    done = {}
    if ck.exists():
        for line in ck.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["model"]] = r
    for name, model in models.items():
        if name in done:
            if verbose:
                print(name, ": уже посчитано", flush=True)
            continue
        try:
            r = ps.evaluate_full({name: model}, log_pm, verbose=verbose).iloc[0].to_dict()
        except KeyError:                     # ошибка одной модели (как в evaluate_full: строка с error, расчёт идёт дальше)
            r = dict(model=name, split="full", error="метрики не посчитаны")
            print(name, "ошибка: метрики не посчитаны", flush=True)
        r = {k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()}
        done[name] = r
        with ck.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")
        gc.collect()
    q = pd.DataFrame([done[m] for m in models])
    pf = q.loc[q["model"] == "flower", "precision_etc"]
    if len(pf) and pd.notna(pf.iloc[0]):
        pf = float(pf.iloc[0])
        q["precision_flower"] = pf
        q["precision_norm"] = (q["precision_etc"] - pf) / (1 - pf)
    q["f1"] = 2 * q["fitness_tbr"] * q["precision_etc"] / (q["fitness_tbr"] + q["precision_etc"])
    return q
