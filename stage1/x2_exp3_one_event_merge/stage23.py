"""Этапы 2 и 3 плана (п. 1.1): параметры переходов и модели выживаемости между состояниями.

Общий код для всех кейсов (x2 эксп. 1 и 3, данные Даши, 15 категорий: all_new / oldcase / no_oldcase).
Ноутбук кейса строит свой журнал (этап 1), затем приводит его к общей таблице эпизодов функцией
episodes_x2 / episodes_variant / episodes_dasha, дальше всё одинаково.

Таблица эпизодов E: одна строка = одно заключение человека.
  state       — состояние заключения (для заключений с несколькими категориями — главная: самая тяжёлая;
                Justice — только если других нет; при равной тяжести случайная, seed 42);
  next_state  — состояние следующего заключения ("" у последнего);
  T           — время на свободе, дни: выход из этого заключения -> вход в следующее (< 0 -> 0);
                у последнего — до конца наблюдения (цензурирование);
  event       — 1, если следующее заключение наблюдалось, 0 — цензурировано (терминального END нет);
  at_risk     — False у тех, кто на конец наблюдения ещё в заключении (времени на свободе нет).
Признаки объекта (не меняются во времени): age10 (возраст / 10, на дату оценки COMPAS), male, race_afam,
race_other (Hispanic, Asian, Native American, Other; база — Caucasian), score (decile_score COMPAS, 1–10).
Признаки события (текущего заключения): felony (самое тяжёлое обвинение — фелония), n_charges (число обвинений,
до 10), log_dur (log(1 + дней в заключении)), prior_eps (сколько заключений было раньше, до 5),
old_case (в заключении есть обвинение по делу из прежнего заключения; нет у данных Даши).
"""
import sqlite3
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42
JUSTICE = "Justice"
OBJECT_FEATURES = ["age10", "male", "race_afam", "race_other", "score"]
EVENT_FEATURES = ["felony", "n_charges", "log_dur", "prior_eps", "old_case"]
FEATURE_RU = {"age10": "возраст (на 10 лет)", "male": "мужчина", "race_afam": "афроамериканец",
              "race_other": "др. раса (не белый)", "score": "балл COMPAS (decile)",
              "felony": "фелония", "n_charges": "число обвинений", "log_dur": "log длит. заключения",
              "prior_eps": "заключений до этого", "old_case": "старое дело в заключении"}
HORIZONS = [90, 180, 365, 730]
BINARY = {"male", "race_afam", "race_other", "felony", "old_case"}


# ---------------------------------------------------------------- таблица эпизодов

def compas_people(db_path):
    """Признаки объекта из compas.db: возраст, пол, раса, исходный балл COMPAS (decile_score)."""
    con = sqlite3.connect(str(db_path))
    p = pd.read_sql("select id, age, sex, race, decile_score from people", con)
    con.close()
    p["decile_score"] = p["decile_score"].where(p["decile_score"] >= 1)     # -1 = нет балла
    return p.rename(columns={"id": "person_id"})


def _object_features(E):
    E["age10"] = E["age"] / 10
    E["male"] = (E["sex"] == "Male").astype(int)
    E["race_afam"] = (E["race"] == "African-American").astype(int)
    E["race_other"] = (~E["race"].isin(["African-American", "Caucasian"])).astype(int)
    E["score"] = E["decile_score"]
    return E


def _finish(E, obs_end):
    """Порядок, следующее состояние, время на свободе, цензурирование, признаки."""
    E = E.sort_values(["id", "in_custody", "ep"]).reset_index(drop=True)
    g = E.groupby("id")
    E["k"] = g.cumcount() + 1
    E["n_eps"] = g["id"].transform("size")
    E["next_state"] = g["state"].shift(-1).fillna("")
    nxt_in = g["in_custody"].shift(-1)
    E["event"] = nxt_in.notna().astype(int)
    T = np.where(E["event"] == 1, (nxt_in - E["out_custody"]).dt.days, (obs_end - E["out_custody"]).dt.days)
    E["T_raw"] = T
    E["T"] = np.clip(T, 0, None).astype(float)
    E["at_risk"] = ~((E["event"] == 0) & E["ongoing"])                   # ещё сидит на конец наблюдения
    E["dur_days"] = (E["out_custody"] - E["in_custody"]).dt.days.clip(lower=0)
    E["log_dur"] = np.log1p(E["dur_days"])
    E["prior_eps"] = (E["k"] - 1).clip(upper=5)
    E["n_charges"] = E["n_charges"].clip(upper=10)
    E = _object_features(E)
    return E


def _main_type(types_rank, rng):
    """Главная категория заключения: самая тяжёлая; Justice — только если других нет."""
    items = [(t, r) for t, r in types_rank if t != JUSTICE and not t.endswith(": " + JUSTICE)] or list(types_rank)
    best = min(r for _, r in items)
    cand = sorted(t for t, r in items if r == best)
    return cand[0] if len(cand) == 1 else cand[rng.integers(len(cand))]


def episodes_x2(st, ch, people, obs_end):
    """Кейсы «категории × 2» (x2_common.build): st — таблица шагов, ch — обвинения из X.load."""
    st = st.copy()
    first = st.sort_values(["episode_id", "step"]).groupby("episode_id", as_index=False).first()
    rk = ch.groupby("ep_row")["rank"].min()
    E = pd.DataFrame({"id": first["id"], "ep": first["episode_id"], "in_custody": first["in_custody"],
                      "out_custody": first["out_custody"], "ongoing": first["ongoing"].astype(bool),
                      "state": first["state"], "n_charges": first["n_charges_all"],
                      "old_case": first["has_old_case"].astype(int)})
    E["felony"] = (E["ep"].map(rk) <= 4).astype(int)
    E = E.merge(people, left_on="id", right_on="person_id", how="left")
    return _finish(E, pd.Timestamp(obs_end))


def episodes_variant(df, ch, people, obs_end):
    """Кейсы 15 категорий (build_log ноутбуков stage1\\<вариант>): df — таблица шагов (типы — множества),
    ch — обвинения с ep_row и rank (load() + merge_continuations()). Шаги одного заключения (отдельный Justice)
    собираются обратно в одно заключение, состояние = главная категория."""
    rank_t = ch.groupby(["ep_row", "type"])["rank"].min().to_dict()
    rk = ch.groupby("ep_row")["rank"].min()
    first_ep = ch.dropna(subset=["case_number"]).groupby(["id", "case_number"])["ep_row"].min()
    fe = pd.MultiIndex.from_frame(ch[["id", "case_number"]]).map(first_ep.to_dict().get)
    old = ch["case_number"].notna().values & (pd.Series(fe, index=ch.index).fillna(np.inf) < ch["ep_row"])
    has_old = pd.Series(old, index=ch.index).groupby(ch["ep_row"]).any()
    n_ch = ch.groupby("ep_row").size()
    d = df.sort_values(["ep_row", "step"])
    e = d.groupby("ep_row", as_index=False).agg(id=("id", "first"), in_custody=("in_custody", "first"),
                                                out_custody=("out_custody", "first"), ongoing=("ongoing", "first"),
                                                types=("types", lambda s: sorted({t for x in s for t in x})))
    rng = np.random.default_rng(SEED)
    e["state"] = [_main_type([(t, rank_t.get((r, t), rk.get(r, 8))) for t in ts], rng)
                  for r, ts in zip(e["ep_row"], e["types"])]
    E = pd.DataFrame({"id": e["id"], "ep": e["ep_row"], "in_custody": e["in_custody"],
                      "out_custody": e["out_custody"], "ongoing": e["ongoing"].astype(bool),
                      "state": e["state"], "n_types": e["types"].str.len(),
                      "n_charges": e["ep_row"].map(n_ch), "old_case": e["ep_row"].map(has_old).astype(int),
                      "felony": (e["ep_row"].map(rk) <= 4).astype(int)})
    E = E.merge(people, left_on="id", right_on="person_id", how="left")
    return _finish(E, pd.Timestamp(obs_end))


def episodes_dasha(log, ep, people, obs_end, degree_col="ep_charge_degree"):
    """Данные Даши (dasha_severity.build_log): log — журнал (одно событие = заключение), ep — таблица Даши
    с тяжестью. Возраст, пол, раса — из таблицы Даши, балл COMPAS — из compas.db по person_id."""
    x = log.merge(ep[["ep_row", "age", "sex", "race", "ep_n_charges"]], on="ep_row", how="left")
    out = pd.to_datetime(x["out_custody"])
    E = pd.DataFrame({"id": x["case:concept:name"], "ep": x["ep_row"],
                      "in_custody": pd.to_datetime(x["time:timestamp"]).dt.normalize(),
                      "out_custody": out.fillna(pd.Timestamp(obs_end)), "ongoing": out.isna(),
                      "state": x["concept:name"], "n_charges": x["ep_n_charges"].fillna(0),
                      "felony": x[degree_col].astype(str).str.contains("F").astype(int),
                      "age": x["age"], "sex": x["sex"], "race": x["race"], "person_id": x["person_id"]})
    E = E.merge(people[["person_id", "decile_score"]], on="person_id", how="left")
    return _finish(E, pd.Timestamp(obs_end))


def features_of(E):
    """Признаки, которые есть у кейса (old_case нет у данных Даши)."""
    return [f for f in OBJECT_FEATURES + EVENT_FEATURES if f in E.columns and E[f].notna().any()]


def split_people(E, seed=SEED, test_share=0.2):
    """train/test по людям (как в этапе 1: 80/20, seed 42)."""
    rng = np.random.default_rng(seed)
    ids = E["id"].unique()
    test_ids = set(rng.choice(ids, size=int(test_share * len(ids)), replace=False))
    is_test = E["id"].isin(test_ids)
    return E[~is_test].copy(), E[is_test].copy()


def episodes_summary(E):
    return pd.Series({
        "людей": E["id"].nunique(), "заключений": len(E), "состояний": E["state"].nunique(),
        "переходов (наблюдались)": int(E["event"].sum()),
        "цензурировано (последние)": int((E["event"] == 0).sum()),
        "  из них ещё в заключении": int((~E["at_risk"]).sum()),
        "доля петель X→X": round(float((E.loc[E["event"] == 1, "state"] == E.loc[E["event"] == 1, "next_state"]).mean()), 3),
        "медиана дней на свободе до след.": float(E.loc[E["event"] == 1, "T"].median()),
        "переходов за 0 дней": int(((E["event"] == 1) & (E["T"] == 0)).sum()),
        "нет балла COMPAS": int(E["score"].isna().sum()),
    })


# ---------------------------------------------------------------- этап 2: параметры переходов

def state_order(E):
    return list(E["state"].value_counts().index)


def transition_matrix(E, order=None):
    """Вероятности перехода i -> j (вложенная цепь: среди наблюдавшихся переходов) и доля цензурирования."""
    order = order or state_order(E)
    cnt = pd.crosstab(E["state"], E["next_state"].where(E["event"] == 1, "цензура"))
    cnt = cnt.reindex(index=order, columns=order + ["цензура"], fill_value=0)
    trans = cnt[order]
    P = trans.div(trans.sum(axis=1).replace(0, np.nan), axis=0)
    summ = pd.DataFrame({"заключений": cnt.sum(axis=1), "переходов": trans.sum(axis=1),
                         "доля цензуры": (cnt["цензура"] / cnt.sum(axis=1)).round(3),
                         "петля p(i→i)": pd.Series(np.diag(P.fillna(0)), index=order).round(3)})
    return cnt, P, summ


def plot_matrix(P, path, title, n=None, annotate_min=0.0):
    k = len(P)
    size = max(6, 0.42 * k + 3)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    im = ax.imshow(P.fillna(0).values, cmap="Blues", vmin=0, vmax=max(0.5, float(np.nanmax(P.values))))
    ax.set_xticks(range(k), P.columns, rotation=90, fontsize=8 if k > 12 else 9)
    ax.set_yticks(range(k), [f"{s} ({n[s]})" if n is not None else s for s in P.index], fontsize=8 if k > 12 else 9)
    for i in range(k):
        for j in range(k):
            v = P.values[i, j]
            if not np.isnan(v) and v >= max(annotate_min, 0.005):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6 if k > 15 else 8,
                        color="white" if v > 0.45 else "black")
    ax.set_xlabel("следующее состояние"); ax.set_ylabel("текущее состояние (число заключений)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def arc_times(E, min_n=1):
    """Время на свободе по дугам i -> j (только наблюдавшиеся переходы): n, p, среднее, квартили, дни."""
    tr = E[E["event"] == 1]
    g = tr.groupby(["state", "next_state"])["T"]
    a = pd.DataFrame({"n": g.size(), "mean": g.mean(), "q25": g.quantile(0.25), "median": g.median(),
                      "q75": g.quantile(0.75), "q90": g.quantile(0.9)})
    a["p"] = a["n"] / tr.groupby("state").size().reindex(a.index.get_level_values(0)).values
    a = a[a["n"] >= min_n].reset_index().rename(columns={"state": "from", "next_state": "to"})
    return a[["from", "to", "n", "p", "mean", "q25", "median", "q75", "q90"]].round(
        {"p": 3, "mean": 1, "q25": 1, "median": 1, "q75": 1, "q90": 1}).sort_values(["from", "n"], ascending=[True, False])


def _pos(T):
    return np.maximum(np.asarray(T, float), 0.5)       # параметрическим моделям нужно T > 0


def fit_state_distributions(E, min_n=100):
    """Время до следующего заключения (любого типа) из состояния i с учётом цензурирования:
    Каплан–Мейер (медиана, доля без нового заключения за 1 и 2 года) и параметрические модели
    (экспонента, Вейбулл, логнормальное, лог-логистическое) — выбирается минимальный AIC."""
    from lifelines import (KaplanMeierFitter, ExponentialFitter, WeibullFitter, LogNormalFitter,
                           LogLogisticFitter)
    rows, fits = [], {}
    R = E[E["at_risk"]]
    for s in state_order(R):
        d = R[R["state"] == s]
        if len(d) < min_n:
            continue
        T, ev = _pos(d["T"]), d["event"].values
        km = KaplanMeierFitter().fit(T, ev)
        row = {"state": s, "n": len(d), "events": int(ev.sum()), "km_median": km.median_survival_time_,
               "S(365)": float(km.predict(365)), "S(730)": float(km.predict(730))}
        best, best_aic = None, np.inf
        for name, F in [("exponential", ExponentialFitter), ("weibull", WeibullFitter),
                        ("lognormal", LogNormalFitter), ("loglogistic", LogLogisticFitter)]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    f = F().fit(T, ev)
                except Exception:
                    continue
            row[f"AIC_{name}"] = round(f.AIC_, 1)
            if f.AIC_ < best_aic:
                best, best_aic = (name, f), f.AIC_
            if name == "weibull":
                row["weibull_lambda"], row["weibull_rho"] = round(f.lambda_, 1), round(f.rho_, 3)
        row["best"] = best[0]
        row["best_median"] = round(float(best[1].median_survival_time_), 1)
        fits[s] = (km, best)
        rows.append(row)
    tab = pd.DataFrame(rows).set_index("state")
    return tab.round({"km_median": 1, "S(365)": 3, "S(730)": 3}), fits


def plot_state_fits(fits, path, states, title):
    n = len(states)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.6 * rows), squeeze=False)
    t = np.linspace(1, 1500, 300)
    for ax, s in zip(axes.flat, states):
        km, (name, f) = fits[s]
        km.plot_survival_function(ax=ax, ci_show=False, label="Каплан–Мейер")
        ax.plot(t, f.survival_function_at_times(t).values, "--", label=name)
        ax.set_title(s, fontsize=10); ax.set_xlim(0, 1500); ax.set_ylim(0, 1)
        ax.set_xlabel("дней на свободе"); ax.legend(fontsize=8)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def fit_arc_distributions(E, min_n=30):
    """Полумарковская модель: для дуги i -> j распределение времени на свободе при условии, что переход
    i -> j произошёл (только наблюдавшиеся переходы, без цензуры). Лучшее по AIC из Вейбулла,
    логнормального, лог-логистического; показаны параметры Вейбулла (rho < 1 — риск со временем падает)."""
    from lifelines import WeibullFitter, LogNormalFitter, LogLogisticFitter
    rows = []
    tr = E[E["event"] == 1]
    for (a, b), d in tr.groupby(["state", "next_state"]):
        if len(d) < min_n:
            continue
        T = _pos(d["T"])
        row = {"from": a, "to": b, "n": len(d), "median_days": float(np.median(d["T"]))}
        best, best_aic = None, np.inf
        for name, F in [("weibull", WeibullFitter), ("lognormal", LogNormalFitter), ("loglogistic", LogLogisticFitter)]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    f = F().fit(T)
                except Exception:
                    continue
            if name == "weibull":
                row["weibull_lambda"], row["weibull_rho"] = round(f.lambda_, 1), round(f.rho_, 3)
            if f.AIC_ < best_aic:
                best, best_aic = name, f.AIC_
                row["best_median"] = round(float(f.median_survival_time_), 1)
        row["best"] = best
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def cumulative_incidence(E, min_n=200, top=6, horizons=HORIZONS):
    """Конкурирующие риски: оценка Аалена–Йохансена вероятности перейти из i в j за t дней
    (с учётом того, что другой переход мог случиться раньше, и цензурирования).
    Возвращает таблицу CIF на горизонтах и кривые для рисунков."""
    from sksurv.nonparametric import cumulative_incidence_competing_risks
    rows, curves = [], {}
    R = E[E["at_risk"]]
    for s in state_order(R):
        d = R[R["state"] == s]
        if len(d) < min_n:
            continue
        dest = d.loc[d["event"] == 1, "next_state"].value_counts()
        code = {k: i + 1 for i, k in enumerate(dest.index)}
        ev = np.where(d["event"] == 1, d["next_state"].map(code).fillna(0), 0).astype(int)
        t, ci = cumulative_incidence_competing_risks(ev, d["T"].values.astype(float))
        idx = [np.searchsorted(t, h, side="right") - 1 for h in horizons]
        for k, i in code.items():
            rows.append({"from": s, "to": k, "n_from": len(d), "events": int(dest[k]),
                         **{f"CIF_{h}d": round(float(ci[i][j]), 4) if j >= 0 else 0.0 for h, j in zip(horizons, idx)}})
        rows.append({"from": s, "to": "любое", "n_from": len(d), "events": int(dest.sum()),
                     **{f"CIF_{h}d": round(float(ci[0][j]), 4) if j >= 0 else 0.0 for h, j in zip(horizons, idx)}})
        curves[s] = (t, ci, list(dest.index[:top]), code)
    return pd.DataFrame(rows), curves


def plot_cif(curves, path, states, title, tmax=1095):
    n = len(states)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.8 * rows), squeeze=False)
    for ax, s in zip(axes.flat, states):
        t, ci, top, code = curves[s]
        m = t <= tmax
        for k in top:
            ax.step(t[m], ci[code[k]][m], where="post", label=k)
        other = ci[0] - sum(ci[code[k]] for k in top)
        ax.step(t[m], other[m], where="post", color="grey", ls=":", label="другие")
        ax.step(t[m], ci[0][m], where="post", color="black", lw=1.5, label="любое")
        ax.set_title(f"из {s}", fontsize=10); ax.set_xlim(0, tmax); ax.set_ylim(0, None)
        ax.set_xlabel("дней на свободе"); ax.set_ylabel("CIF")
        ax.legend(fontsize=7)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- этап 3: модели выживаемости

def model_frame(E, feats):
    """Строки, пригодные для моделей: есть время на свободе и все признаки."""
    R = E[E["at_risk"]].dropna(subset=feats).copy()
    R["T"] = _pos(R["T"])
    return R


def _usable(d, feats, min_var=1e-8):
    return [f for f in feats if d[f].var() > min_var]


def cox_any(train, test, feats, penalizer=0.01):
    """CoxPH для времени до следующего заключения (любого типа).
    (а) стратифицированная по текущему состоянию — у каждого состояния свой базовый риск, коэффициенты общие;
    (б) C-index на test для сравнения: только состояние / только признаки / состояние + признаки."""
    from lifelines import CoxPHFitter
    from sksurv.metrics import concordance_index_censored
    # признак, постоянный внутри каждого состояния (у данных Даши — фелония), в стратифицированной модели не оценивается
    fs = [f for f in feats if train.groupby("state")[f].nunique().max() > 1]
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(train[["T", "event", "state"] + fs], "T", "event", strata=["state"])
    s = cph.summary
    hr = pd.DataFrame({"признак": [FEATURE_RU.get(f, f) for f in s.index], "HR": s["exp(coef)"],
                       "HR 95% низ": s["exp(coef) lower 95%"], "HR 95% верх": s["exp(coef) upper 95%"],
                       "p": s["p"]}, index=s.index).round({"HR": 3, "HR 95% низ": 3, "HR 95% верх": 3, "p": 4})

    def cidx(model_cols, use_state):
        X_tr = train[model_cols].copy()
        X_te = test[model_cols].copy()
        if use_state:
            vc = train["state"].value_counts()
            lv = sorted(vc[vc >= 30].index)              # редкие состояния остаются в базе вместе с первым
            for v in lv[1:]:
                X_tr["st_" + v] = (train["state"] == v).astype(int)
                X_te["st_" + v] = (test["state"] == v).astype(int)
        m = CoxPHFitter(penalizer=penalizer).fit(pd.concat([X_tr, train[["T", "event"]]], axis=1), "T", "event")
        risk = m.predict_partial_hazard(X_te).values
        return concordance_index_censored(test["event"].astype(bool).values, test["T"].values, risk)[0], m

    out = {}
    out["только состояние"], _ = cidx([], True)
    out["только признаки"], _ = cidx(feats, False)
    out["состояние + признаки"], m_full = cidx(feats, True)
    st_hr = m_full.summary.loc[[c for c in m_full.summary.index if c.startswith("st_")], ["exp(coef)", "p"]]
    st_hr.index = [c[3:] for c in st_hr.index]
    return cph, hr, pd.Series(out).round(4), st_hr.rename(columns={"exp(coef)": "HR"}).round(4)


def cox_arcs(R, feats, min_events=50, penalizer=0.05, alpha=0.05):
    """Причинно-специфичные модели Кокса для каждой дуги i -> j с не менее min_events переходами:
    люди в состоянии i, событие = следующее заключение типа j, переходы в другие типы — цензура.
    HR > 1 — признак ускоряет именно этот переход."""
    from lifelines import CoxPHFitter
    rows = []
    for s in state_order(R):
        d = R[R["state"] == s]
        dest = d.loc[d["event"] == 1, "next_state"].value_counts()
        for j, n in dest.items():
            if n < min_events:
                continue
            y = ((d["event"] == 1) & (d["next_state"] == j)).astype(int)
            fs = _usable(d, feats)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    m = CoxPHFitter(penalizer=penalizer).fit(d[fs].assign(T=d["T"], y=y), "T", "y")
                except Exception as ex:
                    print(f"{s} -> {j}: {type(ex).__name__}")
                    continue
            sm = m.summary
            for f in fs:
                rows.append({"from": s, "to": j, "arc": f"{s} → {j}", "n_from": len(d), "events": int(n),
                             "feature": f, "HR": sm.loc[f, "exp(coef)"], "lo": sm.loc[f, "exp(coef) lower 95%"],
                             "hi": sm.loc[f, "exp(coef) upper 95%"], "p": sm.loc[f, "p"]})
    res = pd.DataFrame(rows)
    if len(res):
        res["signif"] = res["p"] < alpha
        res = res.round({"HR": 3, "lo": 3, "hi": 3, "p": 4})
    return res


def arc_effects_summary(res, feats):
    """Сводка по признакам: на скольких дугах HR значимо > 1 и < 1, медиана HR."""
    if not len(res):
        return pd.DataFrame()
    g = res.groupby("feature")
    s = pd.DataFrame({"дуг": g.size(), "HR>1 значимо": g.apply(lambda x: int(((x["HR"] > 1) & x["signif"]).sum())),
                      "HR<1 значимо": g.apply(lambda x: int(((x["HR"] < 1) & x["signif"]).sum())),
                      "медиана HR": g["HR"].median().round(3), "мин HR": g["HR"].min(), "макс HR": g["HR"].max()})
    s = s.reindex([f for f in feats if f in s.index])
    s.index = [FEATURE_RU.get(f, f) for f in s.index]
    return s


def plot_arc_heatmap(res, feats, path, title, max_arcs=40):
    if not len(res):
        return None
    arcs = (res.drop_duplicates("arc").sort_values("events", ascending=False)["arc"].head(max_arcs).tolist())
    fs = [f for f in feats if f in set(res["feature"])]
    M = res.pivot(index="arc", columns="feature", values="HR").reindex(index=arcs, columns=fs)
    S = res.pivot(index="arc", columns="feature", values="signif").reindex(index=arcs, columns=fs)
    L = np.log2(M.astype(float))
    fig, ax = plt.subplots(figsize=(1.1 * len(fs) + 3.5, 0.3 * len(arcs) + 2))
    im = ax.imshow(L.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(fs)), [FEATURE_RU.get(f, f) for f in fs], rotation=45, ha="right", fontsize=8)
    ev = res.drop_duplicates("arc").set_index("arc")["events"]
    ax.set_yticks(range(len(arcs)), [f"{a} ({ev[a]})" for a in arcs], fontsize=7)
    for i in range(len(arcs)):
        for j in range(len(fs)):
            v = M.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}{'*' if S.values[i, j] else ''}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.6, label="log2 HR")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _xy(d, feats, states):
    X = d[feats].astype(float).copy()
    for v in states:
        X["st_" + v] = (d["state"] == v).astype(float)
    from sksurv.util import Surv
    return X, Surv.from_arrays(d["event"].astype(bool).values, d["T"].values)


def survival_tree(train, test, feats, max_depth=3, min_samples_leaf=300):
    """Дерево выживаемости (scikit-survival, log-rank разбиения) для времени до следующего заключения;
    признаки + текущее состояние (one-hot). Листья описываются медианой Каплана–Мейера."""
    from sksurv.tree import SurvivalTree
    from lifelines import KaplanMeierFitter
    states = sorted(train["state"].unique())
    X_tr, y_tr = _xy(train, feats, states)
    X_te, y_te = _xy(test, feats, states)
    tree = SurvivalTree(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=SEED).fit(X_tr.values, y_tr)
    c = tree.score(X_te.values, y_te)
    t = tree.tree_
    names = list(X_tr.columns)
    leaf = tree.apply(X_tr.values.astype(np.float32))
    lines = []

    def label(f, thr):
        n = names[f]
        if n.startswith("st_"):
            return f"состояние = {n[3:]}", f"состояние ≠ {n[3:]}"
        if n in BINARY:
            return f"{FEATURE_RU[n]}: да", f"{FEATURE_RU[n]}: нет"
        return f"{FEATURE_RU.get(n, n)} > {thr:.2f}", f"{FEATURE_RU.get(n, n)} ≤ {thr:.2f}"

    def walk(node, depth, cond):
        if t.children_left[node] == -1:
            d = train[leaf == node]
            km = KaplanMeierFitter().fit(_pos(d["T"]), d["event"])
            lines.append(dict(leaf=node, path=" и ".join(cond), n=len(d), events=int(d["event"].sum()),
                              km_median=km.median_survival_time_, S365=round(float(km.predict(365)), 3)))
            return
        yes, no = label(t.feature[node], t.threshold[node])
        walk(t.children_left[node], depth + 1, cond + [no])
        walk(t.children_right[node], depth + 1, cond + [yes])

    walk(0, 0, [])
    leaves = pd.DataFrame(lines).sort_values("S365").reset_index(drop=True)
    return tree, leaves, c


def plot_tree_leaves(train, tree, feats, leaves, path, title):
    from lifelines import KaplanMeierFitter
    states = sorted(train["state"].unique())
    X_tr, _ = _xy(train, feats, states)
    leaf = tree.apply(X_tr.values.astype(np.float32))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for _, r in leaves.iterrows():
        d = train[leaf == r["leaf"]]
        KaplanMeierFitter().fit(_pos(d["T"]), d["event"], label=f"лист {r['leaf']} (n={r['n']})").plot_survival_function(ax=ax, ci_show=False)
    ax.set_xlim(0, 1200); ax.set_ylim(0, 1); ax.set_xlabel("дней на свободе"); ax.set_ylabel("доля без нового заключения")
    ax.set_title(title); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return path


def random_forest(train, test, feats, n_estimators=100, min_samples_leaf=50, repeats=3):
    """Случайный лес выживаемости: C-index на test и важность признаков перестановкой
    (падение C-index, если перемешать признак; состояние перемешивается целиком)."""
    from sksurv.ensemble import RandomSurvivalForest
    states = sorted(train["state"].unique())
    X_tr, y_tr = _xy(train, feats, states)
    X_te, y_te = _xy(test, feats, states)
    t0 = time.time()
    rsf = RandomSurvivalForest(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features="sqrt",
                               n_jobs=-1, random_state=SEED, low_memory=True).fit(X_tr.values, y_tr)
    base = rsf.score(X_te.values, y_te)
    rng = np.random.default_rng(SEED)
    groups = {f: [f] for f in feats}
    groups["state"] = [c for c in X_te.columns if c.startswith("st_")]
    imp = {}
    for g, cols in groups.items():
        drops = []
        for _ in range(repeats):
            Xp = X_te.copy()
            perm = rng.permutation(len(Xp))
            Xp[cols] = X_te[cols].values[perm]
            drops.append(base - rsf.score(Xp.values, y_te))
        imp["текущее состояние" if g == "state" else FEATURE_RU.get(g, g)] = np.mean(drops)
    return base, pd.Series(imp).sort_values(ascending=False).round(4), round(time.time() - t0)


def save_excel(path, sheets):
    """Все таблицы этапов 2–3 в один .xlsx (лист = таблица). Без openpyxl — пропускается."""
    try:
        with pd.ExcelWriter(path) as xw:
            for name, t in sheets.items():
                t.to_excel(xw, sheet_name=name[:31])
        print("таблицы:", path)
    except ImportError:
        print("openpyxl не установлен — Excel не записан, CSV есть в той же папке")
