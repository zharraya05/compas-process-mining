"""Сравнение пересчёта с прежними результатами папки «петри» (развитие run_states_v2_recheck.py).

Запуск:  python compare_with_old.py [2013|all ...]
Сравнивает журналы состояний, размер сетей и метрики по моделям для всех трёх конфигураций;
то, чего ещё нет, пропускается. Таблица: <часть>\\out\\compare_with_old.csv
Если прежней папки на компьютере нет, поправить OLD_ROOT в settings.py.

Не всё должно совпасть точно. TBR fitness сети DFG p0.05 плавает от запуска к запуску даже на одних данных
(замечено 0.9673–0.9689 в окне 2013): порядок обхода сети в PM4Py не фиксирован, поэтому отличие около 0.001 — норма.
Alignment fitness зависит от числа тайм-аутов (лимит по настенным часам), а его отличие — от того, досчитаны ли они.
Журнала long365_v1 в прежней папке нет, его сравнивать не с чем.
"""
import json
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent))
import settings as cfg  # noqa: E402

TOL = 1e-6
INT_COLS = ["places", "transitions", "silent", "train_traces", "test_traces"]
M_FULL = ["fitness_tbr", "precision_etc", "precision_norm", "f1"]
M_SPLIT = ["fitness_tbr_train", "fitness_tbr_test", "pct_fit_traces_tbr_test", "precision_etc_test"]
M_ALIGN = M_SPLIT + ["fitness_align_test", "pct_fit_traces_align_test", "align_timeout"]
M_V2 = M_SPLIT + ["precision_norm_test", "f1_tbr"]


def table(p):
    p = Path(p)
    if not p.exists():
        return None
    if p.suffix == ".jsonl":
        t = pd.DataFrame([json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()])
    else:
        t = pd.read_csv(p)
    return t.drop_duplicates("model", keep="last").set_index("model")


def sources(part):
    o = cfg.OLD_ROOT
    return [("long365_v1", "model_quality.csv", o / "out_states" / part, M_FULL, INT_COLS),
            ("long365_v1", "split_eval.jsonl", o / "out_states" / part, M_ALIGN, INT_COLS),
            ("long180_v1", "split_eval.jsonl", o / "out_states" / f"{part}_long180", M_ALIGN, INT_COLS),
            ("long180_v2", "split_quality.csv", o / "out_states_v2" / part, M_V2, INT_COLS)]


def compare_logs(part):
    for c, old in [("long365_v1", cfg.OLD_ROOT / "out_states" / part), ("long180_v2", cfg.OLD_ROOT / "out_states_v2" / part)]:
        po, pn = old / "state_log.csv", cfg.out_dir(part, c) / "state_log.csv"
        if not (po.exists() and pn.exists()):
            print(f"{part} {c}: журнала нет — пропуск", flush=True)
            continue
        lo, ln = pd.read_csv(po), pd.read_csv(pn)
        cols = list(lo.columns)
        same = (list(ln.columns) == cols and len(lo) == len(ln)
                and lo.sort_values(cols).reset_index(drop=True).equals(ln.sort_values(cols).reset_index(drop=True)))
        print(f"{part} {c}: журнал прежний {len(lo):,}, новый {len(ln):,}: {'СОВПАДАЕТ' if same else 'ОТЛИЧАЕТСЯ'}", flush=True)


def compare(part):
    compare_logs(part)
    rows = []
    for c, f, old_dir, metrics, ints in sources(part):
        qo, qn = table(old_dir / f), table(cfg.out_dir(part, c) / f)
        if qo is None or qn is None:
            print(f"{part} {c} {f}: нет {'прежнего' if qo is None else 'нового'} файла — пропуск", flush=True)
            continue
        for m in sorted(set(qo.index) | set(qn.index)):
            r = dict(part=part, config=c, file=f, model=m)
            if m not in qo.index or m not in qn.index:
                r["status"] = "нет в прежнем" if m not in qo.index else "нет в новом"
                rows.append(r)
                continue
            diffs = []
            for k in ints:
                if k in qo and k in qn and qo.at[m, k] != qn.at[m, k]:
                    diffs.append(f"{k} {qo.at[m, k]}→{qn.at[m, k]}")
            for k in metrics:
                if k in qo and k in qn:
                    a, b = pd.to_numeric(qo.at[m, k], errors="coerce"), pd.to_numeric(qn.at[m, k], errors="coerce")
                    r[k + "_old"], r[k + "_new"] = a, b
                    if pd.isna(a) != pd.isna(b) or (not pd.isna(a) and abs(a - b) > TOL):
                        diffs.append(f"{k} {a:.4f}→{b:.4f}")
            r["status"] = "совпадает" if not diffs else "ОТЛИЧАЕТСЯ: " + "; ".join(diffs)
            rows.append(r)
    if not rows:
        return None
    q = pd.DataFrame(rows)
    p = cfg.part_dir(part) / "out" / "compare_with_old.csv"
    q.to_csv(p, index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 220, "display.max_colwidth", 120):
        print(q[["config", "file", "model", "status"]].to_string(index=False))
    bad = q[~q["status"].eq("совпадает")]
    print(f"\nИтого: строк {len(q)}, совпадает {len(q) - len(bad)}, отличается / нет {len(bad)} (допуск {TOL}). "
          f"Таблица: {p}", flush=True)
    return q


if __name__ == "__main__":
    for key in sys.argv[1:] or list(cfg.PARTS):
        compare(key)
