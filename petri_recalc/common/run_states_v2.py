"""Журнал состояний, версия 2 (04.11.2026): новое правило end и флаги SHORT = 15, LONG = 180.
Все майнеры: сеть строится на train (люди 80/20, seed 42), на train/test — TBR fitness, на test — ETC precision.
Alignments не считаются.

Запуск: python run_states_v2.py [2013|all ...] или 05_v2_tbr.ipynb.
Результаты: <часть>\\out\\long180_v2\\ (повторный запуск продолжает с места: метрики моделей в tbr_ckpt\\).
"""
import json
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import petri_pipeline as pp  # noqa: E402
import petri_states as ps  # noqa: E402

import settings as cfg  # noqa: E402

DB_CANDIDATES = cfg.DB_CANDIDATES
CATEGORIES_DIRS = cfg.CATEGORIES_DIRS
CONFIG = "long180_v2"
EXPERIMENTS = {k: v["log_start"] for k, v in cfg.PARTS.items()}
SHORT_CUSTODY_DAYS = cfg.CONFIGS[CONFIG]["short"]
LONG_CUSTODY_DAYS = cfg.CONFIGS[CONFIG]["long"]
END_RULE = cfg.CONFIGS[CONFIG]["end_rule"]
SEED, TEST_SHARE = cfg.SEED, cfg.TEST_SHARE


def main(keys):
    t00 = time.time()
    tables, _ = pp.drop_full_duplicates(pp.load_tables(pp.find_db(DB_CANDIDATES)))
    tables, _, _ = pp.drop_invalid_stays(tables)
    hist = pp.build_history_table(tables, pp.import_categorizer(CATEGORIES_DIRS))
    obs_end = pp.observation_end(hist)
    has_dot = pp.setup_graphviz()
    print(f"таблица истории: {len(hist):,} строк, конец наблюдения {obs_end}", flush=True)

    for key in keys:
        d = cfg.out_dir(key, CONFIG)
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        log, eps, rep = ps.build_state_log(hist, obs_end=obs_end, log_start=EXPERIMENTS[key],
                                           short_days=SHORT_CUSTODY_DAYS, long_days=LONG_CUSTODY_DAYS,
                                           end_rule=END_RULE)
        checks = ps.check_state_log(log, END_RULE)
        log.to_csv(d / "state_log.csv", index=False)
        rep = {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in rep.items()}
        rep.update(short_days=SHORT_CUSTODY_DAYS, long_days=LONG_CUSTODY_DAYS, end_rule=END_RULE,
                   log_start=EXPERIMENTS[key], checks=checks)
        (d / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{key}: журнал {len(log):,} событий, {time.time() - t0:.0f} с", *checks, sep="\n  ", flush=True)

        train, test = pp.split_people(log, TEST_SHARE, SEED)
        tr, te = pp.to_pm4py(train), pp.to_pm4py(test)
        full = pp.to_pm4py(log)
        if has_dot and not (d / "dfg_frequency.png").exists():
            pp.save_dfg_png(full, d / "dfg_frequency.png")
            pp.save_dfg_png(full, d / "dfg_performance.png", performance=True)
        del full

        t0 = time.time()
        models, info = ps.discover_all(tr)
        info.to_csv(d / "discovery_info.csv", index=False)
        print(f"{key}: майнеры на train, {time.time() - t0:.0f} с; моделей {len(models)}", flush=True)
        rows = []
        for name, m in models.items():
            t0 = time.time()
            try:
                r = ps.evaluate_split_one(name, m, tr, te, do_align=False, ckpt_dir=d / "tbr_ckpt")
            except Exception as e:  # noqa: BLE001
                r = dict(model=name, error=f"{type(e).__name__}: {str(e)[:150]}")
            rows.append(r)
            print(f"  {name}: {time.time() - t0:.0f} с", {k: round(v, 3) for k, v in r.items()
                                                          if k in ("fitness_tbr_test", "precision_etc_test")},
                  flush=True)
            if has_dot and name != "flower" and not (d / f"net_{name}.png").exists():
                try:
                    pp.save_net_png(*m, d / f"net_{name}.png")
                except Exception as e:  # noqa: BLE001
                    print("   рисунок не сохранён:", e, flush=True)
        q = ps.summarize_split(pd.DataFrame(rows))
        q.to_csv(d / "split_quality.csv", index=False)
        print(f"{key}: готово, {(time.time() - t00) / 60:.0f} мин с начала", flush=True)
    print("V2_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(EXPERIMENTS))
