"""Этап 2. Модель Кокса и метрики на чистом numpy (чтобы ноутбук считался в любом окружении).
В venv диплома рядом можно проверить lifelines / scikit-survival - см. ноутбук 03, раздел 8."""
import numpy as np, pandas as pd

NOLOG = {"t_iai_trend", "t_share_in_custody", "t_age_first_arrest"}
LOG_PREFIX = ("r_", "f_", "m_", "t_", "gh_n_", "h_n_", "g_cur_n_", "c_n_")   # mk_ (марковские) и race_, ix_, ia_ - без log


# ----------------------------------------------------------------------------- подготовка признаков
class Prep:
    """log1p для счётчиков и длительностей, индикатор пропуска + медиана, стандартизация. Всё оценивается на train.
    Правка 13 (17.11): индикатор пропуска заводится один на шаблон пропусков - признаки с одинаковыми пропусками
    (например, t_age_first_arrest, t_career_years, t_arrest_rate - нет арестов) дают один индикатор, а не несколько
    одинаковых колонок, которые раздувают наивные SE и искажают z."""
    def fit(self, df, cols):
        self.cols = list(cols)
        self.log = [c for c in self.cols if c.startswith(LOG_PREFIX) and c not in NOLOG]
        Z = self._raw(df)
        self.na = [c for c in self.cols if Z[c].isna().any()]
        seen, self.na_ind = {}, []
        for c in self.na:
            key = Z[c].isna().values.tobytes()
            if key not in seen: seen[key] = c; self.na_ind.append(c)
        self.na_alias = {c: seen[Z[c].isna().values.tobytes()] for c in self.na}
        self.med = Z[self.cols].median()
        Z = self._fill(Z)
        self.mu, self.sd = Z.mean(), Z.std(ddof=0)
        self.keep = [c for c in Z.columns if self.sd[c] > 1e-9]
        return self
    def _raw(self, df):
        Z = df[self.cols].astype(float).copy()
        Z[self.log] = np.log1p(Z[self.log].clip(lower=0))
        return Z
    def _fill(self, Z):
        for c in self.na_ind: Z[c + "_na"] = Z[c].isna().astype(float)
        return Z.fillna(self.med)
    def transform(self, df):
        Z = self._fill(self._raw(df))
        return ((Z[self.keep] - self.mu[self.keep]) / self.sd[self.keep]).values


# ----------------------------------------------------------------------------- модель Кокса
class Cox:
    """Пропорциональные риски Кокса, связи по Бреслоу, гребневый штраф l2 (на стандартизованных признаках)."""
    def __init__(self, l2=0.01, max_iter=50, tol=1e-7):
        self.l2, self.max_iter, self.tol = l2, max_iter, tol

    def _parts(self, X, time, event, beta):
        eta = X @ beta
        w = np.exp(eta - eta.max())
        S0 = np.cumsum(w)[self.last]
        S1 = np.cumsum(w[:, None] * X, axis=0)[self.last]
        ll = float((event * (eta - eta.max() - np.log(S0))).sum())
        return eta, w, S0, S1, ll

    def fit(self, X, time, event):
        o = np.argsort(-time, kind="stable")
        X, time, event = X[o], time[o], event[o].astype(float)
        n, p = X.shape
        _, first, cnt = np.unique(-time, return_index=True, return_counts=True)
        grp = np.repeat(np.arange(len(first)), cnt)
        self.last = (first + cnt - 1)[grp]; first_i = first[grp]
        ne = event.sum(); beta = np.zeros(p)
        obj = lambda b: self._parts(X, time, event, b)[4] / ne - 0.5 * self.l2 * b @ b
        cur = obj(beta)
        for it in range(self.max_iter):
            eta, w, S0, S1, ll = self._parts(X, time, event, beta)
            M = S1 / S0[:, None]
            g = (event[:, None] * (X - M)).sum(0) / ne - self.l2 * beta
            A = np.cumsum((event / S0)[::-1])[::-1][first_i]
            H = (X * (w * A)[:, None]).T @ X - (M * event[:, None]).T @ M
            Hn = H / ne + self.l2 * np.eye(p)
            step = np.linalg.solve(Hn, g); s = 1.0
            while s > 1e-4:
                new = obj(beta + s * step)
                if new >= cur - 1e-12: break
                s /= 2
            beta = beta + s * step
            if abs(new - cur) < self.tol: cur = new; break
            cur = new
        self.beta, self.n_iter = beta, it + 1
        eta, w, S0, S1, ll = self._parts(X, time, event, beta)
        self.se = np.sqrt(np.diag(np.linalg.inv(Hn * ne)))            # наивные SE (без поправки на кластеры)
        S0_true = S0 * np.exp(eta.max())
        ev = event > 0
        h = pd.Series(1.0 / S0_true[ev]).groupby(time[ev]).sum().sort_index()
        self.t_base, self.H0 = h.index.values.astype(float), h.cumsum().values
        return self

    def risk(self, X):
        return X @ self.beta

    def survival(self, X, t):
        """S(t | x) для вектора горизонтов t -> массив (n, len(t))."""
        t = np.atleast_1d(t)
        idx = np.searchsorted(self.t_base, t, side="right") - 1
        H = np.where(idx >= 0, self.H0[np.clip(idx, 0, None)], 0.0)
        return np.exp(-np.exp(self.risk(X))[:, None] * H[None, :])


# ----------------------------------------------------------------------------- метрики
def c_index(time, event, risk, chunk=2000):
    """C-индекс Харрелла: доля сравнимых пар, где у раньше «упавшего» риск выше (равные риски = 0.5)."""
    time, event, risk = map(np.asarray, (time, event, risk))
    num = den = 0.0
    idx = np.where(event == 1)[0]
    for s in range(0, len(idx), chunk):
        i = idx[s:s + chunk]
        comp = time[i][:, None] < time[None, :]
        d = risk[i][:, None] - risk[None, :]
        num += ((d > 0) & comp).sum() + 0.5 * ((d == 0) & comp).sum(); den += comp.sum()
    return num / den


def km(time, event):
    """Каплан-Мейер -> (времена, S(t))."""
    d = pd.DataFrame({"t": time, "e": event}).groupby("t")["e"].agg(["sum", "count"]).sort_index()
    at_risk = d["count"][::-1].cumsum()[::-1]
    return d.index.values.astype(float), (1 - d["sum"] / at_risk).cumprod().values


def km_at(tk, sk, t, left=False):
    idx = np.searchsorted(tk, t, side="left" if left else "right") - 1
    return np.where(idx >= 0, sk[np.clip(idx, 0, None)], 1.0)


def brier(time, event, surv_t, t, cens_km):
    """Оценка Брайера на горизонте t с весами IPCW (Graf et al., 1999). cens_km = KM цензурирования."""
    time, event = np.asarray(time, float), np.asarray(event)
    tk, gk = cens_km
    G_T = np.clip(km_at(tk, gk, time, left=True), 1e-3, None); G_t = max(float(km_at(tk, gk, t)), 1e-3)
    died = (time <= t) & (event == 1); alive = time > t
    return float(np.mean(died * surv_t ** 2 / G_T + alive * (1 - surv_t) ** 2 / G_t))


# ----------------------------------------------------------------------------- кросс-валидация по людям
def person_folds(person_id, n_folds=5, seed=0):
    u = np.unique(person_id); rng = np.random.default_rng(seed); rng.shuffle(u)
    fold_of = {p: i % n_folds for i, p in enumerate(u)}
    return np.array([fold_of[p] for p in person_id])


def cv_cox(ds, cols, target="custody", l2=0.01, n_folds=5, seed=0, horizons=(180, 365, 730), subsets=None,
           fold_fn=None):
    """Возвращает (таблица метрик по фолдам, out-of-fold прогноз S(365)).
    subsets = {имя: булева маска} - дополнительно C-индекс на части тестового фолда (обучение на всех)."""
    time = ds[f"time_{target}"].values.astype(float); event = ds[f"event_{target}"].values.astype(int)
    folds = person_folds(ds["person_id"].values, n_folds, seed)
    rows, oof = [], np.full(len(ds), np.nan)
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        dsf, colsf = ds, cols
        if fold_fn is not None:                      # признаки, которые оцениваются только на обучающей части
            extra = fold_fn(tr); dsf = pd.concat([ds, extra], axis=1); colsf = list(cols) + list(extra.columns)
        prep = Prep().fit(dsf[tr], colsf)
        m = Cox(l2=l2).fit(prep.transform(dsf[tr]), time[tr], event[tr])
        Xte = prep.transform(dsf[te]); S = m.survival(Xte, horizons)
        row = {"fold": f, "c_index": c_index(time[te], event[te], m.risk(Xte))}
        ck = km(time[tr], 1 - event[tr]); tk, sk = km(time[tr], event[tr])
        for j, h in enumerate(horizons):
            bs = brier(time[te], event[te], S[:, j], h, ck)
            ref = brier(time[te], event[te], np.full(te.sum(), float(km_at(tk, sk, h))), h, ck)
            row[f"brier_{h}"] = bs; row[f"skill_{h}"] = 1 - bs / ref
        for name, mask in (subsets or {}).items():
            mm = np.asarray(mask)[te]
            row[f"c_{name}"] = c_index(time[te][mm], event[te][mm], m.risk(Xte)[mm])
        rows.append(row); oof[te] = S[:, list(horizons).index(365)] if 365 in horizons else S[:, 0]
    return pd.DataFrame(rows), oof


def summarize_cv(res):
    m, s = res.drop(columns="fold").mean(), res.drop(columns="fold").std(ddof=1)
    return pd.Series({k: f"{m[k]:.3f} ± {s[k]:.3f}" for k in m.index})


# ----------------------------------------------------------------------------- метрики как у Коледы (библиотека survivors)
def cv_survivors(ds, cols, target="custody", l2=0.01, n_folds=5, seed=0, n_grid=200, q=0.95):
    """IBS и AUPRC функциями survivors.metrics (ibs_remain, auprc), как в DKoleda6/recurrent_event (metrics/ibs.py,
    metrics/auprc.py): сетка из n_grid точек от 0 до q-квантиля длительностей эпизодов train, функция выживания
    эпизода на сетке. Возвращает таблицу по фолдам; если survivors не установлен - ImportError."""
    from survivors import metrics as svm
    from survivors.constants import get_y
    if not hasattr(np, "trapz"):                 # survivors вызывает np.trapz, которого нет в новых версиях numpy
        np.trapz = np.trapezoid
    time = ds[f"time_{target}"].values.astype(float); event = ds[f"event_{target}"].values.astype(int)
    folds = person_folds(ds["person_id"].values, n_folds, seed); rows = []
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        prep = Prep().fit(ds[tr], cols)
        m = Cox(l2=l2).fit(prep.transform(ds[tr]), time[tr], event[tr])
        grid = np.linspace(0, np.quantile(time[tr], q), n_grid)
        S = m.survival(prep.transform(ds[te]), grid).astype(np.float64)
        y_tr = get_y(time=time[tr], cens=event[tr].astype(bool)); y_te = get_y(time=time[te], cens=event[te].astype(bool))
        ibs = svm.ibs_remain(survival_train=y_tr, survival_test=y_te, estimate=S, times=grid, axis=-1)
        auprc = svm.auprc(survival_train=y_tr, survival_test=y_te, estimate=S, times=grid)
        rows.append({"fold": f, "IBS": float(ibs), "AUPRC": float(auprc), "horizon": float(grid[-1])})
    return pd.DataFrame(rows)
