"""Журнал «состояний» для сети Петри (правила 04.11.2026).

Отличие от petri_pipeline.build_event_log: запись базы — не событие, а часть состояния.

Состояния
  * arrest / charge / arrest+charge — записи ОДНОГО дела за ОДИН день склеиваются в одно состояние
    (несколько арестов дела за день -> одно arrest, несколько обвинений -> одно charge);
  * jail / prison — заключение;
  * составные: arrest+jail, charge+jail, arrest+charge+jail (и то же с prison) — дело и вход
    в заключение в один день;
  * free — на свободе (нет заключения, нет ареста/обвинения в этот день);
  * end — искусственное состояние: конец наблюдения базы.

Правила
  1. Несколько дел в один день — по состоянию на дело, порядок по номеру дела, между ними free нет.
  2. Вход в заключение в день D:
       – одно дело в этот день и заключение не длинное (< LONG дней) -> составное состояние дело+заключение;
       – несколько дел или заключение длинное (>= LONG) -> состояния дел, затем отдельно jail/prison.
  3. Аресты/обвинения ВНУТРИ срока (дни строго между днём входа и днём выхода):
       – заключение короткое (< SHORT дней) -> поглощаются состоянием заключения
         (его имя дополняется типами записей, например jail -> charge+jail);
       – иначе — отдельные состояния: заключение -> арест/обвинение -> заключение (возврат в D+1 00:00,
         если на следующий день внутри срока событий нет).
     Перевод jail -> prison внутри эпизода — отдельное состояние prison.
  4. Выход из заключения — по out_custody последнего пребывания эпизода. free не ставится, если следующее
     событие в тот же или на следующий день (переход сразу в новое состояние).
  5. После ареста/обвинения без заключения: если на следующий день событий нет -> free в D+1 00:00,
     иначе сразу следующее состояние.
  6. Окно (log_start): трасса начинается состоянием на начало окна — free или jail/prison, если человек
     тогда сидел. Без окна — с первого склеенного состояния.
  7. В end входят только free или заключение: кто сидит на конец наблюдения — из заключения
     (после событий внутри срока — возврат в заключение).
Время составного состояния — время входа в заключение; состояния дела без заключения — дата 00:00.
Одинаковое время разводится на 1 мс (порядок задаётся последовательностью состояний).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import petri_pipeline as pp

DAY = pd.Timedelta(days=1)
EPS = pd.Timedelta(milliseconds=1)
TYPE_ORDER = {"arrest": 0, "charge": 1, "jail": 2, "prison": 3}


def label(types) -> str:
    return "+".join(sorted(set(types), key=TYPE_ORDER.get))


def _units(hist: pd.DataFrame, obs_end) -> pd.DataFrame:
    """Записи arrest/charge -> единицы (человек, день, дело) с набором типов."""
    t = pp.event_time(hist)
    r = hist[hist["event"].isin(["arrest", "charge"])][["person_id", "event", "case_number"]].copy()
    r["t"] = t[r.index]
    r = r[r["t"].notna() & (r["t"] <= obs_end)]
    r["day"] = r["t"].dt.normalize()
    r["case"] = r["case_number"].where(r["case_number"].notna() & (r["case_number"] != pp.STUB), "NA")
    u = (r.groupby(["person_id", "day", "case"])
           .agg(types=("event", lambda s: tuple(sorted(set(s), key=TYPE_ORDER.get))),
                n_records=("event", "size"))
           .reset_index())
    return u


def build_state_log(hist: pd.DataFrame, obs_end=None, log_start=None, short_days=15, long_days=365,
                    transfer_gap=pd.Timedelta(days=1), end_rule="v1"):
    """Журнал состояний. Возвращает (log, episodes, report).

    end_rule — из каких состояний трасса может перейти в end:
      "v1" — только из free, jail, prison (иначе перед end ставится возврат в заключение);
      "v2" (04.11.2026) — из free, из любого состояния с заключением (в т. ч. составного: charge+prison),
            а из ареста/обвинения — если оно в день конца наблюдения или накануне (тогда free не ставится).
    """
    assert end_rule in ("v1", "v2")
    V2 = end_rule == "v2"
    obs_end = pp.observation_end(hist) if obs_end is None else pd.Timestamp(obs_end)
    last_days = obs_end.normalize() - DAY          # дела с этого дня (v2) переходят в end без free
    W = None if log_start is None else pd.Timestamp(log_start)
    units = _units(hist, obs_end)
    stays, eps = pp.custody_episodes(hist, obs_end, transfer_gap)
    eps = eps.copy()
    eps["end_eff"] = eps["ep_end"].fillna(obs_end)
    eps["dur_days"] = (eps["end_eff"] - eps["ep_start"]).dt.total_seconds() / 86400
    eps["short"] = eps["dur_days"] < short_days
    eps["long"] = eps["dur_days"] >= long_days
    stays = stays.sort_values(["episode_id", "in_t"])

    U = {p: g for p, g in units.groupby("person_id")}
    E = {p: g.sort_values("ep_start") for p, g in eps.groupby("person_id")}
    S = {e: g for e, g in stays.groupby("episode_id")}
    rows = []
    rep = dict(obs_end=obs_end, composite=0, absorbed_units=0, inside_units=0, returns=0, free_after_case=0,
               free_after_custody=0, no_free_next_day=0, started_inside=0, dropped_people_no_events=0,
               end_from_case=0, end_from_composite=0)

    people = np.sort(hist["person_id"].unique())
    for pid in people:
        u = U.get(pid)
        e = E.get(pid)
        u = u if u is not None else units.iloc[0:0]
        e = e if e is not None else eps.iloc[0:0]
        # ---- блоки: эпизод заключения (дела дня входа -> заключение -> внутри срока -> выход)
        #      и день вне заключений (дела по номеру); блоки идут по дню начала
        blocks = []
        used = np.zeros(len(u), dtype=bool)
        uday = u["day"].to_numpy()
        ucase = u["case"].to_numpy()
        for ep in e.itertuples(index=False):
            De = ep.ep_start.normalize()
            Dr = ep.end_eff.normalize()
            st = S[ep.episode_id]
            first_type = st["event"].iloc[0]
            blk = []
            m_entry = (uday == De) & ~used
            m_in = ((uday > De) & ~used) if ep.still_in else ((uday > De) & (uday < Dr) & ~used)
            entry_cases = sorted(set(ucase[m_entry]))
            comp_types = [first_type]
            if len(entry_cases) == 1 and not ep.long:
                comp_types += list(u.loc[m_entry, "types"].iloc[0])
                rep["composite"] += 1
            else:
                for c in entry_cases:
                    k = m_entry & (ucase == c)
                    blk.append(dict(kind="case", t=De, day=De, types=list(u.loc[k, "types"].iloc[0]),
                                    cases=c, n=int(u.loc[k, "n_records"].sum()), ep=np.nan))
            used |= m_entry
            inside_items = []
            if ep.short:
                for tp in u.loc[m_in, "types"]:
                    comp_types += list(tp)
                rep["absorbed_units"] += int(m_in.sum())
            else:
                for (d, c), gg in u[m_in].groupby(["day", "case"]):
                    inside_items.append(dict(kind="inside", t=d, day=d, types=list(gg["types"].iloc[0]),
                                             cases=c, n=int(gg["n_records"].sum()), ep=ep.episode_id))
                rep["inside_units"] += int(m_in.sum())
            used |= m_in
            blk.append(dict(kind="custody", t=ep.ep_start, day=De, types=comp_types, ep=ep.episode_id,
                            cur=first_type))
            extra = [dict(kind="transfer", t=r.in_t, day=r.in_t.normalize(), types=[r.event], ep=ep.episode_id)
                     for r in st.iloc[1:].itertuples(index=False) if r.in_t > ep.ep_start]
            blk += sorted(inside_items + extra, key=lambda x: (x["t"], x["kind"] != "inside", x.get("cases", "")))
            blk.append(dict(kind="release", t=ep.ep_end, day=Dr, ep=ep.episode_id, still_in=bool(ep.still_in)))
            blocks.append(((De, 0), blk))
        for d, g in u[~used].groupby("day"):
            blk = [dict(kind="case", t=d, day=d, types=list(gg["types"].iloc[0]), cases=c,
                        n=int(gg["n_records"].sum()), ep=np.nan) for c, gg in g.groupby("case")]
            blocks.append(((d, 1), blk))
        blocks.sort(key=lambda b: b[0])
        seq = [it for _, blk in blocks for it in blk]

        # ---- окно
        start_state = None
        if W is not None:
            cur_ep = e[(e["ep_start"] < W) & (e["end_eff"] > W)]
            if len(cur_ep):
                ep = cur_ep.iloc[0]
                st = S[ep.episode_id]
                active = st[st["in_t"] < W]["event"].iloc[-1]
                start_state = dict(kind="start", t=W, label=active, inside=True, ep=ep.episode_id)
                rep["started_inside"] += 1
            else:
                start_state = dict(kind="start", t=W, label="free", inside=False, ep=np.nan)
            seq = [it for it in seq if _item_time(it) >= W and not (it["kind"] == "release" and _item_time(it) <= W)]
            # события эпизода, начатого до окна: внутри срока остаются, выход тоже
            if not seq and start_state["label"] == "free":
                rep["dropped_people_no_events"] += 1
                continue

        # ---- превращаем последовательность в состояния с free / возвратами
        out = []
        if start_state is not None:
            out.append(dict(activity=start_state["label"], ts_raw=W, kind="start", inside=start_state["inside"],
                            cases="", n_records=0, episode_id=start_state["ep"]))
        cur_custody = start_state["label"] if (start_state and start_state["inside"]) else None
        for i, it in enumerate(seq):
            nxt = seq[i + 1] if i + 1 < len(seq) else None
            nxt_day = nxt["day"] if nxt is not None else None
            k = it["kind"]
            if k == "custody":
                out.append(dict(activity=label(it["types"]), ts_raw=it["t"], kind="custody", inside=True,
                                cases="", n_records=0, episode_id=it["ep"]))
                cur_custody = it["cur"]
            elif k == "transfer":
                out.append(dict(activity=it["types"][0], ts_raw=it["t"], kind="transfer", inside=True,
                                cases="", n_records=0, episode_id=it["ep"]))
                cur_custody = it["types"][0]
            elif k == "inside":
                out.append(dict(activity=label(it["types"]), ts_raw=it["t"], kind="inside", inside=True,
                                cases=it["cases"], n_records=it["n"], episode_id=it["ep"]))
                # возврат в заключение, если на следующий день внутри срока нет событий
                nxt_inside = nxt is not None and nxt["kind"] in ("inside", "transfer") and nxt["day"] <= it["day"] + DAY
                nxt_none = nxt is None or (nxt["kind"] == "release" and nxt.get("still_in"))
                if V2 and not nxt_inside and nxt_none and it["day"] >= last_days:
                    pass                                   # ещё сидит, дело в последние дни -> сразу end
                elif not nxt_inside:
                    rt = it["day"] + DAY
                    if nxt is not None and nxt["kind"] == "release" and pd.notna(nxt["t"]):
                        rt = min(rt, nxt["t"])
                    out.append(dict(activity=cur_custody, ts_raw=min(rt, obs_end), kind="return", inside=True,
                                    cases="", n_records=0, episode_id=it["ep"]))
                    rep["returns"] += 1
            elif k == "release":
                if it["still_in"]:
                    continue
                cur_custody = None
                if nxt is not None and nxt_day <= it["day"] + DAY:
                    rep["no_free_next_day"] += 1          # переход сразу в новое состояние
                    continue
                out.append(dict(activity="free", ts_raw=it["t"], kind="free_release", inside=False,
                                cases="", n_records=0, episode_id=it["ep"]))
                rep["free_after_custody"] += 1
            elif k == "case":
                out.append(dict(activity=label(it["types"]), ts_raw=it["t"], kind="case", inside=False,
                                cases=it["cases"], n_records=it["n"], episode_id=np.nan))
                if nxt is not None and nxt_day <= it["day"] + DAY:
                    continue
                if V2 and nxt is None and it["day"] >= last_days:
                    continue                               # дело в последние дни наблюдения -> сразу end
                out.append(dict(activity="free", ts_raw=min(it["day"] + DAY, obs_end), kind="free_case",
                                inside=False, cases="", n_records=0, episode_id=np.nan))
                rep["free_after_case"] += 1
        if not out:
            continue
        # конец: в end только из free или заключения
        last = out[-1]
        act = last["activity"]
        if V2:
            ok = act == "free" or "jail" in act or "prison" in act
            if not ok and last["kind"] in ("case", "inside") and pd.Timestamp(last["ts_raw"]).normalize() >= last_days:
                ok = True
                rep["end_from_case"] += 1
            elif ok and act not in ("free", "jail", "prison"):
                rep["end_from_composite"] += 1
        else:
            ok = act in ("free", "jail", "prison")
        if not ok:
            if cur_custody is not None:
                out.append(dict(activity=cur_custody, ts_raw=obs_end, kind="return", inside=True, cases="",
                                n_records=0, episode_id=last["episode_id"]))
        out.append(dict(activity="end", ts_raw=obs_end, kind="end", inside=False, cases="", n_records=0,
                        episode_id=np.nan))
        for o in out:
            o["person_id"] = pid
        rows += out

    log = pd.DataFrame(rows)
    log["ts_raw"] = pd.to_datetime(log["ts_raw"])
    # время строго растёт внутри трассы: одинаковое/обратное время -> +1 мс к предыдущему
    pid = log["person_id"].to_numpy()
    t = log["ts_raw"].to_numpy("datetime64[ns]").astype("int64")
    out_t = t.copy()
    for i in range(1, len(t)):
        if pid[i] == pid[i - 1] and out_t[i] <= out_t[i - 1]:
            out_t[i] = out_t[i - 1] + int(EPS.value)
    log["ts"] = pd.to_datetime(out_t)
    log["event_idx"] = log.groupby("person_id").cumcount() + 1
    log["next_activity"] = log.groupby("person_id")["activity"].shift(-1)
    log["dt_next_days"] = (log.groupby("person_id")["ts"].shift(-1) - log["ts"]).dt.total_seconds() / 86400
    log = log[["person_id", "event_idx", "activity", "ts", "ts_raw", "kind", "inside", "cases", "n_records",
               "episode_id", "next_activity", "dt_next_days"]]
    rep.update(events=len(log), people=log["person_id"].nunique(), activities=log["activity"].nunique())
    return log, eps, rep


def _item_time(it):
    if it["kind"] == "release":
        return it["t"] if pd.notna(it["t"]) else pd.Timestamp.max
    return it["t"]


def check_state_log(log: pd.DataFrame, end_rule="v1"):
    res = []
    g = log.groupby("person_id")
    assert (g["ts"].diff().dropna() > pd.Timedelta(0)).all(), "время в трассе не строго растёт"
    res.append("время внутри каждой трассы строго возрастает")
    assert (g["activity"].last() == "end").all()
    before_end = log[log["next_activity"] == "end"]["activity"]
    if end_rule == "v1":
        bad = before_end[~before_end.isin(["free", "jail", "prison"])]
        res.append(f"перед end: {before_end.value_counts().to_dict()}; не free/jail/prison: {len(bad)}")
    else:
        pre = log[log["next_activity"] == "end"]
        no_cust = pre[(pre["activity"] != "free") & ~pre["activity"].str.contains("jail|prison")]
        obs_day = log.loc[log["activity"] == "end", "ts_raw"].max().normalize()
        late = no_cust["ts_raw"].dt.normalize() >= obs_day - DAY
        res.append(f"перед end: {before_end.value_counts().to_dict()}")
        res.append(f"перед end без заключения и не free: {len(no_cust)}, из них не в последние 2 дня: {int((~late).sum())}")
        assert late.all(), "дело перед end не в последние дни наблюдения"
    ff = log[(log["activity"] == "free") & (log["next_activity"] == "free")]
    res.append(f"free → free подряд: {len(ff)}")
    return res


# ----------------------------------------------------------------------------
# Модели и метрики без разбиения train/test (сеть строится и проверяется на всём журнале)
# ----------------------------------------------------------------------------
def discover_all(log_pm, im_noise=(0.0, 0.2), hm_dep=(0.9, 0.95, 0.99), dfg_min_prob=(0.0, 0.05), alpha=True):
    """Все майнеры: IM, HM, Split Miner, POWL, сеть из DFG, Alpha, Alpha+ и цветочная сеть.
    ILP и Genetic не используются (ILP не помещается в память, Genetic — часы)."""
    import time
    import pm4py
    models, info = pp.discover_models(log_pm, im_noise=im_noise, hm_dep=hm_dep)
    info["status"] = "ok"
    flower = models.pop("flower")
    extra, extra_info = pp.discover_extra_models(log_pm, ilp_alpha=(), dfg_min_prob=dfg_min_prob)
    models.update(extra)
    info = pd.concat([info, extra_info], ignore_index=True)
    if alpha:
        for name, fn in [("Alpha", pm4py.discover_petri_net_alpha), ("AlphaPlus", pm4py.discover_petri_net_alpha_plus)]:
            t0 = time.time()
            try:
                models[name] = fn(log_pm)
                info.loc[len(info)] = [name, time.time() - t0, "ok"]
            except Exception as e:                       # noqa: BLE001
                info.loc[len(info)] = [name, time.time() - t0, f"пропущен: {type(e).__name__}: {str(e)[:100]}"]
    models["flower"] = flower
    info.loc[len(info)] = ["flower", 0.0, "ok"]
    for col, f in [("places", lambda n: len(n.places)), ("transitions", lambda n: len(n.transitions)),
                   ("silent", lambda n: sum(t.label is None for t in n.transitions))]:
        info[col] = [f(models[m][0]) if m in models else None for m in info["model"]]
    return models, info


def evaluate_full(models: dict, log_pm, verbose=True):
    """TBR fitness и ETC precision на всём журнале, без alignments. Ошибка одной модели не останавливает расчёт."""
    rows = []
    for name, m in models.items():
        try:
            q = pp.evaluate_models({name: m}, {"full": log_pm}, align_sample=0, precision_splits=("full",),
                                   align_splits=(), verbose=verbose)
            rows.append(q.iloc[0].to_dict())
        except Exception as e:                           # noqa: BLE001
            rows.append(dict(model=name, split="full", error=f"{type(e).__name__}: {str(e)[:120]}"))
            if verbose:
                print(name, "ошибка:", rows[-1]["error"], flush=True)
    q = pd.DataFrame(rows)
    pf = q.loc[q["model"] == "flower", "precision_etc"]
    if len(pf) and pd.notna(pf.iloc[0]):
        pf = float(pf.iloc[0])
        q["precision_flower"] = pf
        q["precision_norm"] = (q["precision_etc"] - pf) / (1 - pf)
    q["f1"] = 2 * q["fitness_tbr"] * q["precision_etc"] / (q["fitness_tbr"] + q["precision_etc"])
    return q


# ----------------------------------------------------------------------------
# Лучшие майнеры с разбиением train/test и alignments на всём test
# ----------------------------------------------------------------------------
BEST_MODELS = ("DFG_p0.05", "HM_dep0.95", "SplitMiner", "IM_noise0.2")


def discover_named(train_pm, names=BEST_MODELS):
    """Строит на train только перечисленные модели (+ цветочную для нормировки precision)."""
    import pm4py
    out = {}
    for m in names:
        if m.startswith("IM_noise"):
            out[m] = pm4py.discover_petri_net_inductive(train_pm, noise_threshold=float(m[len("IM_noise"):]))
        elif m.startswith("HM_dep"):
            out[m] = pm4py.discover_petri_net_heuristics(train_pm, dependency_threshold=float(m[len("HM_dep"):]))
        elif m == "SplitMiner":
            out[m] = pm4py.convert_to_petri_net(pm4py.discover_bpmn_split_miner(train_pm))
        elif m == "POWL":
            out[m] = pm4py.convert_to_petri_net(pm4py.discover_powl(train_pm))
        elif m.startswith("DFG_p"):
            out[m] = pp.dfg_petri_net(train_pm, float(m[len("DFG_p"):]))
        else:
            raise ValueError(f"неизвестная модель {m}")
    out["flower"] = pp.flower_net(train_pm["concept:name"].unique())
    return out


def alignment_fitness_full(log_pm, net, im, fm, max_sec_trace=120, cores=1):
    """Alignment fitness на ВСЕХ трассах журнала, лимит max_sec_trace секунд на вариант трассы.
    cores > 1 — параллельно (alignments.apply_multiprocessing). Варианты, не уложившиеся в лимит,
    в метрику не входят; их число и число трасс в них возвращаются."""
    import pm4py
    from pm4py.algo.conformance.alignments.petri_net import algorithm as ali
    from pm4py.algo.evaluation.replay_fitness.variants import alignment_based as ab
    el = pm4py.convert_to_event_log(log_pm)
    params = {ali.Parameters.PARAM_MAX_ALIGN_TIME_TRACE: max_sec_trace}
    if cores and cores > 1:
        params[ali.Parameters.CORES] = cores
        res = ali.apply_multiprocessing(el, net, im, fm, parameters=params)
    else:
        res = ali.apply(el, net, im, fm, parameters=params)
    ev = ab.evaluate(res)
    ev["n_traces"] = len(res)
    ev["n_timeout"] = sum(r is None for r in res)
    return ev


def alignment_fitness_resumable(log_pm, net, im, fm, max_sec_trace=120, checkpoint=None, progress_every=100):
    """Alignment fitness на всех трассах журнала, вариант за вариантом, с сохранением каждого результата.
    checkpoint — файл .jsonl: при повторном запуске уже выровненные варианты пропускаются.
    Лимит max_sec_trace секунд на вариант; вариант, не уложившийся в лимит, в fitness не входит.
    Итог считается как в PM4Py (alignment-based fitness): 1 − Σcost / Σbwc по трассам."""
    import json, time
    from pathlib import Path
    import pm4py
    from pm4py.algo.conformance.alignments.petri_net import algorithm as ali
    el = pm4py.convert_to_event_log(log_pm)
    variants = {}
    for tr_ in el:
        k = tuple(e["concept:name"] for e in tr_)
        if k in variants:
            variants[k][1] += 1
        else:
            variants[k] = [tr_, 1]
    keys = list(variants)
    done = {}
    ck = Path(checkpoint) if checkpoint else None
    if ck is not None and ck.exists():
        for l in ck.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                done[r["i"]] = r
    bwc = getattr(ali, "__get_best_worst_cost")(net, im, fm, ali.DEFAULT_VARIANT, {})
    params = {ali.Parameters.PARAM_MAX_ALIGN_TIME_TRACE: max_sec_trace,
              ali.Parameters.BEST_WORST_COST_INTERNAL: bwc}
    t_start, n_new = time.time(), 0
    f = ck.open("a", encoding="utf-8") if ck is not None else None
    try:
        for i, k in enumerate(keys):
            if i in done:
                continue
            trace, cnt = variants[k]
            t0 = time.time()
            res = ali.apply_trace(trace, net, im, fm, parameters=params)
            r = dict(i=i, n=cnt, len=len(k), sec=round(time.time() - t0, 2))
            if res is None or "fitness" not in res:
                r["timeout"] = True
            else:
                r.update(timeout=False, cost=res["cost"], bwc=res["bwc"], fitness=res["fitness"])
            done[i] = r
            n_new += 1
            if f is not None:
                f.write(json.dumps(r) + "\n")
                f.flush()
            if progress_every and n_new % progress_every == 0:
                print(f"  выровнено {len(done):,} / {len(keys):,} вариантов, "
                      f"{(time.time() - t_start) / 60:.1f} мин", flush=True)
    finally:
        if f is not None:
            f.close()
    ok = [r for r in done.values() if not r["timeout"]]
    n_tr = sum(r["n"] for r in done.values())
    n_ok = sum(r["n"] for r in ok)
    s_cost = sum(r["cost"] * r["n"] for r in ok)
    s_bwc = sum(r["bwc"] * r["n"] for r in ok)
    return dict(log_fitness=1 - s_cost / s_bwc if s_bwc else float("nan"),
                percentage_of_fitting_traces=100 * sum(r["n"] for r in ok if r["fitness"] == 1.0) / n_ok if n_ok else float("nan"),
                n_traces=n_tr, n_timeout=n_tr - n_ok, n_variants=len(keys),
                n_variants_timeout=sum(1 for r in done.values() if r["timeout"]),
                complete=len(done) == len(keys))


def evaluate_split_one(name, model, tr, te, max_sec_trace=120, cores=1, do_align=True, ckpt_dir=None):
    """Одна модель: TBR fitness train/test, ETC precision test, alignment fitness на всём test.
    ckpt_dir — папка контрольных точек: TBR/precision и каждый выровненный вариант сохраняются,
    повторный запуск продолжает с места (при cores > 1 — параллельно, без контрольных точек)."""
    import json, time
    from pathlib import Path
    import pm4py
    net, im, fm = model
    meta = Path(ckpt_dir) / f"{name}_tbr.json" if ckpt_dir else None
    if meta is not None and meta.exists():
        r = json.loads(meta.read_text(encoding="utf-8"))
    else:
        r = dict(model=name, places=len(net.places), transitions=len(net.transitions),
                 silent=sum(t.label is None for t in net.transitions),
                 train_traces=tr["case:concept:name"].nunique(), test_traces=te["case:concept:name"].nunique())
        t0 = time.time()
        r["fitness_tbr_train"] = pm4py.fitness_token_based_replay(tr, net, im, fm)["log_fitness"]
        tb = pm4py.fitness_token_based_replay(te, net, im, fm)
        r["fitness_tbr_test"] = tb["log_fitness"]
        r["pct_fit_traces_tbr_test"] = tb["percentage_of_fitting_traces"]
        r["sec_tbr"] = time.time() - t0
        t0 = time.time()
        r["precision_etc_test"] = pm4py.precision_token_based_replay(te, net, im, fm)
        r["sec_prec"] = time.time() - t0
        if meta is not None:
            meta.parent.mkdir(parents=True, exist_ok=True)
            meta.write_text(json.dumps(r), encoding="utf-8")
    if do_align and name != "flower":
        # alignments определены, только если из начальной разметки достижима финальная (easy soundness);
        # иначе ни одна трасса не выравнивается — это свойство сети, а не тайм-аут
        from pm4py.objects.petri_net.utils import check_soundness
        r["easy_sound"] = bool(check_soundness.check_easy_soundness_net_in_fin_marking(net, im, fm))
        if not r["easy_sound"]:
            r.update(fitness_align_test=None, pct_fit_traces_align_test=None,
                     align_traces=te["case:concept:name"].nunique(), align_timeout=None, align_complete=True)
            return r
    if do_align and name != "flower":                 # цветочная: alignment fitness = 1 по построению
        t0 = time.time()
        if cores and cores > 1:
            al = alignment_fitness_full(te, net, im, fm, max_sec_trace, cores)
            al["complete"] = True
        else:
            ck = Path(ckpt_dir) / f"{name}_align.jsonl" if ckpt_dir else None
            al = alignment_fitness_resumable(te, net, im, fm, max_sec_trace, ck)
        r["fitness_align_test"] = al["log_fitness"]
        r["pct_fit_traces_align_test"] = al["percentage_of_fitting_traces"]
        r["align_traces"] = al["n_traces"]
        r["align_timeout"] = al["n_timeout"]
        r["align_complete"] = al["complete"]
        r["sec_align_session"] = time.time() - t0
    return r


def summarize_split(rows: pd.DataFrame) -> pd.DataFrame:
    q = rows.copy()
    if "flower" in set(q["model"]):
        pf = float(q.loc[q["model"] == "flower", "precision_etc_test"].iloc[0])
        q["precision_flower_test"] = pf
        q["precision_norm_test"] = (q["precision_etc_test"] - pf) / (1 - pf)
    q["f1_tbr"] = 2 * q["fitness_tbr_test"] * q["precision_etc_test"] / (q["fitness_tbr_test"] + q["precision_etc_test"])
    if "fitness_align_test" in q:
        q["f1_align"] = 2 * q["fitness_align_test"] * q["precision_etc_test"] / (q["fitness_align_test"] + q["precision_etc_test"])
    return q
