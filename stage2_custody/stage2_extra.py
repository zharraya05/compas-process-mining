"""Этап 2, событие = заключение. Дополнительные признаки (раса, взаимодействия, марковские) и конкурирующие риски.
Используется в ноутбуках 03 и 04."""
import sqlite3
import numpy as np, pandas as pd
import stage2_features as sf, stage2_models as sm

# ----------------------------------------------------------------------------- общие данные
def load_parts(cache=True):
    """Узлы, пребывания, аресты. Кэш out/_cache_parts.pkl пересобирается, если входные файлы новее него."""
    import os, pickle
    path = sf.HERE / "out" / "_cache_parts.pkl"
    src = [sf.DB, sf.EVENT_TABLE, sf.HERE / "stage2_features.py"]
    if cache and path.exists() and all(os.path.getmtime(path) > os.path.getmtime(p) for p in src):
        return pickle.load(open(path, "rb"))
    nodes = sf.load_nodes(); spells = sf.build_spells(nodes); arrests = sf.load_arrests()
    if cache: pickle.dump((nodes, spells, arrests), open(path, "wb"))
    return nodes, spells, arrests

GROUPS = ["Violent", "Property", "Drug", "Traffic", "OldCase", "Other"]       # укрупнённые состояния узла
SHORT = dict(Violent="V", Property="P", Drug="D", Traffic="T", OldCase="O", Other="X")

def groups_of(types):
    """Множество типов узла -> множество из 6 укрупнённых групп (всё прочее, включая Unknown, -> Other)."""
    g = {t if t in GROUPS[:5] else "Other" for t in types}
    return g or {"Other"}


# ----------------------------------------------------------------------------- раса
RACE_MAP = {"African-American": "AfricanAmerican", "Caucasian": "Caucasian", "Hispanic": "Hispanic",
            "Other": "Other", "Asian": "Other", "Native American": "Other"}
RACE_LEVELS = ["AfricanAmerican", "Hispanic", "Other"]                         # опорная категория - Caucasian

def race_table(db=sf.DB):
    r = pd.read_sql("select id as person_id, race from people", sqlite3.connect(db))
    r["race4"] = r["race"].map(RACE_MAP)
    return r[["person_id", "race", "race4"]]

def race_features(ds, db=sf.DB):
    r = ds[["person_id"]].merge(race_table(db), on="person_id", how="left")
    out = pd.DataFrame({f"race_{lv}": (r["race4"] == lv).astype(int).values for lv in RACE_LEVELS}, index=ds.index)
    return out, r["race4"].values


# ----------------------------------------------------------------------------- причина возврата (конкурирующие риски)
CAUSE_OF_REASON = {"new_offense": 1, "prison_by_arrest": 1,                 # новое дело
                   "rearrest_old_case": 2, "revocation_old_case": 2,         # старое дело
                   "не привязано": 3, "procedural": 3}                       # причина не определена
CAUSE_NAMES = {1: "new", 2: "old", 3: "unknown"}
CAUSE_RU = {1: "новое дело", 2: "старое дело", 3: "не определено"}

def add_causes(ds, spells):
    nxt = spells.assign(next_reason=spells.groupby("person_id")["reason"].shift(-1))[["person_id", "k", "next_reason"]]
    x = ds[["person_id", "k", "event_custody"]].merge(nxt, on=["person_id", "k"], how="left")
    cause = np.where(x["event_custody"].values == 1, x["next_reason"].map(CAUSE_OF_REASON).fillna(3).values, 0).astype(int)
    out = ds.copy(); out["cause"] = cause; out["next_reason"] = np.where(cause > 0, x["next_reason"].values, None)
    for c, nm in CAUSE_NAMES.items():
        out[f"time_{nm}"] = out["time_custody"]; out[f"event_{nm}"] = (cause == c).astype(int)
    return out


# ----------------------------------------------------------------------------- взаимодействия
def _pair_frame(prev_sets, cur_sets, prefix, index, min_count=30):
    cols = {}
    for a in GROUPS:
        pa = np.array([a in s if s is not None else False for s in prev_sets])
        for b in GROUPS:
            cb = np.array([b in s for s in cur_sets])
            cols[f"{prefix}{SHORT[a]}_{SHORT[b]}"] = (pa & cb).astype(int)
    df = pd.DataFrame(cols, index=index)
    return df.loc[:, df.sum() >= min_count]

def interaction_features(ds, spells, arrests):
    """ix_A_B - предыдущий узел графа содержит группу A, текущий - группу B (только k >= 2).
       ia_A_B - последний арест до текущего заключения (вся история, casearrest) имеет группу A, текущий узел - B.
       Для ia: O = последний арест был повторным по старому делу."""
    sp = spells.set_index(["person_id", "k"])["types"]
    cur = [groups_of(sp.loc[(p, k)]) for p, k in zip(ds.person_id, ds.k)]
    prev = [groups_of(sp.loc[(p, k - 1)]) if k > 1 else None for p, k in zip(ds.person_id, ds.k)]
    ix = _pair_frame(prev, cur, "ix_", ds.index)
    # последний арест до текущего заключения
    ins = spells.set_index(["person_id", "k"])["in_date"]
    q = pd.DataFrame({"person_id": ds.person_id.values, "cut": [ins.loc[(p, k)] - pd.Timedelta(days=sf.ENTRY_GUARD_DAYS) for p, k in zip(ds.person_id, ds.k)]})
    q["row"] = np.arange(len(q))
    a = arrests.sort_values("date")[["person_id", "date", "types", "old_case"]]
    m = pd.merge_asof(q.sort_values("cut"), a, left_on="cut", right_on="date", by="person_id",
                      direction="backward", allow_exact_matches=False).sort_values("row")
    last = []
    for ts, old in zip(m["types"], m["old_case"]):
        if not isinstance(ts, frozenset): last.append(None); continue
        g = {t for t in ts if t in GROUPS[:4]} or {"Other"}
        if old is True: g = {"OldCase"}
        last.append(g)
    ia = _pair_frame(last, cur, "ia_", ds.index)
    return ix, ia


# ----------------------------------------------------------------------------- марковские признаки (по обучающей части)
class Markov:
    """Матрица переходов между укрупнёнными группами узлов по людям из обучающей части.
    P(b | a) = доля переходов из узла, содержащего a, в узел, содержащий b (сглаживание к общей частоте b)."""
    def __init__(self, spells, alpha=5.0):
        self.alpha = alpha
        g = spells[["person_id", "k", "types"]].copy(); g["G"] = g["types"].map(lambda t: frozenset(groups_of(t)))
        g["G_next"] = g.groupby("person_id")["G"].shift(-1)
        self.g = g

    def fit(self, persons):
        t = self.g[self.g.person_id.isin(persons) & self.g.G_next.notna()]
        gi = {g: i for i, g in enumerate(GROUPS)}
        C = np.zeros((6, 6)); n_a = np.zeros(6)
        for (A, B), n in t.groupby(["G", "G_next"]).size().items():      # уникальных пар множеств немного
            ia = [gi[a] for a in A]; ib = [gi[b] for b in B]
            n_a[ia] += n; C[np.ix_(ia, ib)] += n
        base = C.sum(0) / C.sum()
        self.P = pd.DataFrame((C + self.alpha * base) / (n_a + self.alpha)[:, None], index=GROUPS, columns=GROUPS)
        self.counts = pd.DataFrame(C, index=GROUPS, columns=GROUPS)
        return self

    def trans_logp(self, A, B):
        key = (A, B)
        if key not in self._cache:
            self._cache[key] = float(np.log(np.mean([self.P.loc[a, b] for a in A for b in B])))
        return self._cache[key]

    def features(self, ds, spells_by_person):
        """spells_by_person: {person_id: [frozenset групп узла 1, узла 2, ...]}"""
        self._cache = {}
        nxt_of = {}
        lp_of = {p: [self.trans_logp(seq[i - 1], seq[i]) for i in range(1, len(seq))] for p, seq in spells_by_person.items()}
        rows = []
        for p, k in zip(ds.person_id.values, ds.k.values):
            cur = spells_by_person[p][k - 1]
            if cur not in nxt_of:
                nxt = self.P.loc[list(cur)].mean(); q = nxt.values / nxt.values.sum()
                f0 = {f"mk_next_{SHORT[b]}": nxt[b] for b in GROUPS}; f0["mk_next_entropy"] = float(-(q * np.log(q)).sum())
                nxt_of[cur] = f0
            f = dict(nxt_of[cur]); lp = lp_of[p][:k - 1]
            if lp:
                f["mk_path_mean_logp"], f["mk_path_min_logp"], f["mk_path_last_logp"] = float(np.mean(lp)), float(np.min(lp)), lp[-1]
                f["mk_path_n_rare"] = int(sum(v < np.log(0.05) for v in lp))
            else:
                f["mk_path_mean_logp"] = f["mk_path_min_logp"] = f["mk_path_last_logp"] = np.nan; f["mk_path_n_rare"] = 0
            rows.append(f)
        return pd.DataFrame(rows, index=ds.index)

def make_markov_fold_fn(ds, spells, alpha=5.0):
    mk = Markov(spells, alpha)
    by = {p: [frozenset(groups_of(t)) for t in g.sort_values("k")["types"]] for p, g in spells.groupby("person_id")}
    pid = ds.person_id.values
    def fold_fn(tr_mask):
        mk.fit(set(pid[tr_mask])); return mk.features(ds, by)
    return fold_fn, mk, by


# ----------------------------------------------------------------------------- конкурирующие риски
def aalen_johansen(time, cause, causes=(1, 2, 3)):
    """Непараметрическая кумулятивная функция частоты (CIF) каждой причины."""
    d = pd.DataFrame({"t": time, "c": cause})
    tab = pd.crosstab(d["t"], d["c"]).sort_index()
    for c in (0,) + tuple(causes):
        if c not in tab: tab[c] = 0
    n_at = tab.sum(axis=1)[::-1].cumsum()[::-1]
    d_all = tab[list(causes)].sum(axis=1)
    S = (1 - d_all / n_at).cumprod(); S_prev = S.shift(1, fill_value=1.0)
    out = {c: (S_prev * tab[c] / n_at).cumsum().values for c in causes}
    return tab.index.values.astype(float), out, S.values

def cif_from_cox(models, Xs, grid):
    """CIF по причинам из причинно-специфических моделей Кокса. models, Xs - списки по причинам (порядок причин)."""
    n = Xs[0].shape[0]; Hc = np.zeros((n, len(grid))); hs = []
    for m, X in zip(models, Xs):
        idx = np.searchsorted(m.t_base, grid, side="right") - 1
        H0 = np.where(idx >= 0, m.H0[np.clip(idx, 0, None)], 0.0)
        dH0 = np.diff(np.concatenate([[0.0], H0]))
        h = np.exp(m.risk(X))[:, None] * dH0[None, :]
        hs.append(h); Hc += np.cumsum(h, axis=1)
    S_prev = np.exp(-np.concatenate([np.zeros((n, 1)), Hc[:, :-1]], axis=1))
    return [np.cumsum(S_prev * h, axis=1) for h in hs]

def brier_cr(time, cause, k, F_t, t, cens_km):
    """Оценка Брайера для CIF причины k на горизонте t, веса IPCW (цензурирование = cause 0)."""
    time = np.asarray(time, float); tk, gk = cens_km
    G_T = np.clip(sm.km_at(tk, gk, time, left=True), 1e-3, None); G_t = max(float(sm.km_at(tk, gk, t)), 1e-3)
    ev = (time <= t) & (cause > 0); y = ((time <= t) & (cause == k)).astype(float)
    w = np.where(ev, 1 / G_T, np.where(time > t, 1 / G_t, 0.0))
    return float(np.mean(w * (y - F_t) ** 2))

def c_wolbers(time, cause, k, risk, chunk=2000):
    """C-индекс для конкурирующих рисков (Wolbers et al., 2009): пара сравнима, если у i событие причины k раньше,
    чем у j любое событие или цензура, ЛИБО у j раньше было конкурирующее событие (j остаётся «под риском» причины k)."""
    time, cause, risk = map(np.asarray, (time, cause, risk)); num = den = 0.0
    idx = np.where(cause == k)[0]
    for s in range(0, len(idx), chunk):
        i = idx[s:s + chunk]
        comp = (time[i][:, None] < time[None, :]) | ((cause[None, :] > 0) & (cause[None, :] != k) & (time[None, :] <= time[i][:, None]))
        d = risk[i][:, None] - risk[None, :]
        num += ((d > 0) & comp).sum() + 0.5 * ((d == 0) & comp).sum(); den += comp.sum()
    return num / den

def cv_competing(ds, cols, causes=(1, 2, 3), l2=0.01, n_folds=5, seed=0, horizons=(180, 365, 730), fold_fn=None):
    """Причинно-специфические модели Кокса для всех причин; метрики на отложенных фолдах:
    C Харрелла по причине (конкурирующие события = цензура), C Вольберса по CIF(365), Брайер для CIF и skill к Аалену-Йохансену."""
    T = ds["time_custody"].values.astype(float); Cz = ds["cause"].values
    folds = sm.person_folds(ds["person_id"].values, n_folds, seed); rows = []
    oof = {c: np.full(len(ds), np.nan) for c in causes}
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        dsf, colsf = ds, cols
        if fold_fn is not None:
            extra = fold_fn(tr); dsf = pd.concat([ds, extra], axis=1); colsf = list(cols) + list(extra.columns)
        prep = sm.Prep().fit(dsf[tr], colsf); Xtr, Xte = prep.transform(dsf[tr]), prep.transform(dsf[te])
        models = [sm.Cox(l2=l2).fit(Xtr, T[tr], (Cz[tr] == c).astype(int)) for c in causes]
        grid = np.arange(1, T.max() + 1)
        F = cif_from_cox(models, [Xte] * len(causes), grid)
        ck = sm.km(T[tr], (Cz[tr] == 0).astype(int))
        tg, aj, _ = aalen_johansen(T[tr], Cz[tr], causes)
        row = {"fold": f}
        for j, c in enumerate(causes):
            nm = CAUSE_NAMES[c]
            row[f"C_{nm}"] = sm.c_index(T[te], (Cz[te] == c).astype(int), models[j].risk(Xte))
            row[f"Cw_{nm}"] = c_wolbers(T[te], Cz[te], c, F[j][:, 364])
            for h in horizons:
                ref = float(sm.km_at(tg, aj[c], h))
                bs = brier_cr(T[te], Cz[te], c, F[j][:, h - 1], h, ck)
                bref = brier_cr(T[te], Cz[te], c, np.full(te.sum(), ref), h, ck)
                row[f"skill_{nm}_{h}"] = 1 - bs / bref
            oof[c][te] = F[j][:, 364]
        rows.append(row)
    return pd.DataFrame(rows), oof


# ----------------------------------------------------------------------------- кэш результатов кросс-валидации
def make_memo(ds, use_cache=True, folder=None):
    """memo(fn, *args, **kw): результат fn(ds, *args, **kw) сохраняется в out/_cv_cache.
    Ключ - имя функции, аргументы и отпечаток данных (размер и сумма времени/событий/признаков), так что при
    изменении данных или набора признаков кэш не используется. use_cache=False - всегда пересчитывать."""
    import hashlib, pickle
    folder = folder or (sf.HERE / "out" / "_cv_cache"); folder.mkdir(parents=True, exist_ok=True)
    num = ds.select_dtypes("number")
    fp = hashlib.md5(f"{ds.shape}|{np.nansum(num.values):.6f}|{list(num.columns)}".encode()).hexdigest()[:10]
    def memo(fn, *args, **kw):
        key = hashlib.md5(repr((fn.__module__, fn.__name__, fp, args, sorted(kw.items()))).encode()).hexdigest()
        path = folder / f"{fn.__name__}_{key[:16]}.pkl"
        if use_cache and path.exists():
            return pickle.load(open(path, "rb"))
        res = fn(ds, *args, **kw); pickle.dump(res, open(path, "wb")); return res
    return memo
