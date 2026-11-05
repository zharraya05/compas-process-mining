"""Подбор флагов SHORT_CUSTODY_DAYS / LONG_CUSTODY_DAYS журнала состояний (04.11.2026).

Функции для petri_states_flags.ipynb:
  * episode_profile   — эпизоды заключения: длительность, число дел в день входа, события внутри срока;
  * domain_medians    — медианы длительности по «зонам» правил 3–7 при заданных флагах;
  * transition_medians— медианы времени на переходах журнала состояний (dt до следующего состояния);
  * long_profile / short_profile — профили по длительности, по которым выбираются границы;
  * sweep             — перебор флагов: журнал -> train/test 80/20 -> TBR fitness / ETC precision.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import petri_pipeline as pp
import petri_states as ps


def episode_profile(hist: pd.DataFrame, obs_end=None):
    """Эпизоды (как в petri_states.build_state_log) + единицы дел в день входа и внутри срока."""
    obs_end = pp.observation_end(hist) if obs_end is None else pd.Timestamp(obs_end)
    units = ps._units(hist, obs_end)
    stays, eps = pp.custody_episodes(hist, obs_end)
    eps = eps.copy()
    eps["end_eff"] = eps["ep_end"].fillna(obs_end)
    eps["dur"] = (eps["end_eff"] - eps["ep_start"]).dt.total_seconds() / 86400
    st = stays.sort_values(["episode_id", "in_t"]).groupby("episode_id")["event"]
    eps["first"] = eps["episode_id"].map(st.first())
    eps["has_prison"] = eps["episode_id"].map(st.apply(lambda s: (s == "prison").any()))
    eps["De"] = eps["ep_start"].dt.normalize()
    eps["Dr"] = eps["end_eff"].dt.normalize()
    m = units.merge(eps[["episode_id", "person_id", "De", "Dr", "still_in", "dur", "has_prison"]], on="person_id")
    entry = m[m["day"] == m["De"]]
    inside = m[(m["day"] > m["De"]) & ((m["day"] < m["Dr"]) | m["still_in"])].copy()
    ec = entry.groupby("episode_id")["case"].agg(n_cases="nunique", cases=lambda s: set(s))
    eps = eps.merge(ec, left_on="episode_id", right_index=True, how="left")
    eps["n_cases"] = eps["n_cases"].fillna(0).astype(int)
    inside = inside.merge(eps[["episode_id", "cases"]], on="episode_id", how="left")
    inside["same_case"] = [isinstance(s, set) and c in s for c, s in zip(inside["case"], inside["cases"])]
    inside["t_from_entry"] = (inside["day"] - inside["De"]).dt.days
    eps["n_inside"] = eps["episode_id"].map(inside.groupby("episode_id").size()).fillna(0).astype(int)
    return eps, inside


def _q(s: pd.Series) -> dict:
    return dict(n=len(s), q25=s.quantile(.25), median=s.median(), q75=s.quantile(.75))


def domain_medians(eps: pd.DataFrame, short_days: float, long_days: float) -> pd.DataFrame:
    """Длительность заключения (дней) по зонам правил при заданных флагах."""
    zones = {
        "все эпизоды": eps,
        "3: одно дело в день входа, < LONG (составное)": eps[(eps.n_cases == 1) & (eps.dur < long_days)],
        "4: несколько дел в день входа": eps[eps.n_cases > 1],
        "5: одно дело в день входа, ≥ LONG (дело → заключение)": eps[(eps.n_cases == 1) & (eps.dur >= long_days)],
        "без дела в день входа": eps[eps.n_cases == 0],
        "6: события внутри срока, < SHORT (склейка)": eps[(eps.n_inside > 0) & (eps.dur < short_days)],
        "7: события внутри срока, ≥ SHORT (jail → дело → jail)": eps[(eps.n_inside > 0) & (eps.dur >= short_days)],
    }
    t = pd.DataFrame({k: _q(v["dur"]) for k, v in zones.items()}).T
    t["доля prison"] = [v["has_prison"].mean() if len(v) else np.nan for v in zones.values()]
    t["n"] = t["n"].astype(int)
    return t.round(2).rename_axis(f"зона (SHORT={short_days}, LONG={long_days})")


def transition_medians(log: pd.DataFrame) -> pd.DataFrame:
    """Медиана времени (дней) от состояния до следующего — по типам переходов журнала состояний."""
    lg = log[log["next_activity"].notna()].copy()
    cust = lg["activity"].str.contains("jail|prison")
    case_like = lg["activity"].isin(["arrest", "charge", "arrest+charge"])
    nxt_cust = lg["next_activity"].isin(["jail", "prison"])
    groups = {
        "составное дело+заключение → следующее (правила 3, 6)": lg[(lg.kind == "custody") & lg.activity.str.contains(r"\+")],
        "дело → jail/prison в тот же день (правила 4, 5)": lg[case_like & (lg.kind == "case") & nxt_cust
                                                            & (lg.dt_next_days < 1)],
        "jail/prison без дела → следующее": lg[(lg.kind == "custody") & lg.activity.isin(["jail", "prison"])],
        "заключение → дело внутри срока (правило 7)": lg[cust & (lg.next_activity.isin(["arrest", "charge", "arrest+charge"]))
                                                        & lg["inside"]],
        "дело внутри срока → возврат (правило 7)": lg[lg.kind == "inside"],
        "возврат → следующее (правило 7)": lg[lg.kind == "return"],
    }
    t = pd.DataFrame({k: _q(v["dt_next_days"]) for k, v in groups.items()}).T
    t["n"] = t["n"].astype(int)
    return t.round(2).rename_axis("переход")


def long_profile(eps: pd.DataFrame, bins=(0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 365, 545, 730, 1e5)):
    """По длительности: доля эпизодов с prison и с делом в день входа. Граница LONG — где эпизод
    из «задержания по делу дня входа» (jail) становится «сроком» (prison, дела в день входа нет)."""
    g = eps.groupby(pd.cut(eps.dur, list(bins), right=False), observed=True).agg(
        эпизодов=("dur", "size"), доля_prison=("has_prison", "mean"),
        доля_с_делом_в_день_входа=("n_cases", lambda s: (s >= 1).mean()),
        доля_одно_дело=("n_cases", lambda s: (s == 1).mean()))
    return g.round(3)


def short_profile(inside: pd.DataFrame, bins=(0, 3, 7, 10, 15, 20, 30, 45, 60, 90, 180, 365, 1e5)):
    """По длительности заключения: события внутри срока — доля на следующий день после входа
    (продолжение оформления задержания) и медиана дней от входа. Граница SHORT — где доля < 50%."""
    x = inside.assign(day1=inside.t_from_entry <= 1)
    g = x.groupby(pd.cut(x.dur, list(bins), right=False), observed=True).agg(
        единиц=("day1", "size"), доля_на_след_день=("day1", "mean"),
        медиана_дней_от_входа=("t_from_entry", "median"), доля_то_же_дело=("same_case", "mean"))
    return g.round(3)


def first_crossing(profile: pd.DataFrame, col: str, above=True, level=0.5):
    """Левая граница первого интервала, где col > level (above) или < level."""
    s = profile[col]
    hit = s[s > level] if above else s[s < level]
    return float(hit.index[0].left) if len(hit) else None


# ----------------------------------------------------------------------------
def _run_one(hist, obs_end, short_days, long_days, log_start, models, test_share, seed):
    import pm4py
    t0 = time.time()
    log, _, rep = ps.build_state_log(hist, obs_end=obs_end, log_start=log_start,
                                     short_days=short_days, long_days=long_days)
    train, test = pp.split_people(log, test_share, seed)
    tr, te = pp.to_pm4py(train), pp.to_pm4py(test)
    nets = ps.discover_named(tr, models)
    base = dict(exp=log_start or "all", seed=seed, short=short_days, long=long_days,
                variants_test=len(pm4py.get_variants(te)), composite=rep["composite"],
                absorbed=rep["absorbed_units"], inside=rep["inside_units"], returns=rep["returns"])
    rows = []
    for name, (net, im, fm) in nets.items():
        r = dict(base, model=name)
        r["fit"] = pm4py.fitness_token_based_replay(te, net, im, fm)["log_fitness"]
        r["prec"] = pm4py.precision_token_based_replay(te, net, im, fm)
        rows.append(r)
    for r in rows:
        r["sec"] = round(time.time() - t0)
    return rows


def sweep(hist, configs, results: Path, log_start="2013-01-01", models=("DFG_p0.05", "HM_dep0.95", "SplitMiner"),
          test_share=0.2, seed=42, obs_end=None):
    """configs — список (short, long). Каждая пара дописывается в results (.jsonl); готовые пропускаются.
    Одна пара ≈ 2 минуты (журнал ~1.5 мин + TBR/ETC), без alignments."""
    obs_end = pp.observation_end(hist) if obs_end is None else obs_end
    results = Path(results)
    done = set()
    if results.exists():
        for l in results.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            done.add((r["exp"], r["seed"], r["short"], r["long"]))
    for sd, ld in configs:
        if ((log_start or "all"), seed, sd, ld) in done:
            continue
        rows = _run_one(hist, obs_end, sd, ld, log_start, list(models), test_share, seed)
        with results.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
        print(f"SHORT={sd}, LONG={ld}: {rows[0]['sec']} с", flush=True)


def sweep_table(results: Path) -> pd.DataFrame:
    d = pd.DataFrame([json.loads(l) for l in Path(results).read_text(encoding="utf-8").splitlines()])
    d["f1"] = 2 * d.fit * d.prec / (d.fit + d.prec)
    key = ["exp", "seed", "short", "long"]
    pf = d[d.model == "flower"].set_index(key)["prec"]
    d["prec_norm"] = (d["prec"] - d.set_index(key).index.map(pf)) / (1 - d.set_index(key).index.map(pf))
    m = d[d.model != "flower"]
    f1 = m.pivot_table(index=key, columns="model", values="f1").add_prefix("F1 ")
    f1["F1 среднее"] = f1.mean(axis=1)
    pn = m.pivot_table(index=key, columns="model", values="prec_norm").add_prefix("Pnorm ")
    info = d[d.model == "flower"].set_index(key)[["variants_test", "composite", "absorbed", "inside"]]
    return f1.join(pn).join(info).round(4).sort_index()
