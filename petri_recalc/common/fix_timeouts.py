"""Досчёт трасс, не уложившихся в лимит alignments (LONG = 180, eval-ноутбуки), 05.11.2026.

Запуск (обычно из 04_alignments.ipynb):
    python fix_timeouts.py <2013|all> [dfg split im check summary] [--workers=N] [--config=long180_v2]
    dfg      DFG p >= 0.05: точный расчёт стоимости выравнивания (минуты)
    split    Split Miner, только окно 2013: alignments с лимитом 1800 с
    im       IM 0.2: TBR + alignments 600 с (долго; можно прерывать)
    check    сверка: заново выровнять 50 уже посчитанных вариантов и сравнить cost и время
    summary  только итоговая таблица
Без шагов: шаги части из settings.PARTS[...]["fix_steps"]. После каждого шага пишется итог:
    <часть>\\out\\<конфигурация>\\split_eval_timeouts_fixed.csv
Прерывание безопасно: каждый досчитанный вариант сразу пишется в
    <часть>\\out\\<конфигурация>\\align_ckpt\\<модель>_timeouts_fixed.jsonl, повторный запуск продолжает с места.
Исходные контрольные точки (<модель>_align.jsonl) и split_eval.jsonl не изменяются.

Отличия от fix_timeouts_051126.py: части разделены; сети и варианты берутся из prepared\\ (журнал не строится);
alignments с большим лимитом идут параллельно (align_parallel.py); IM 0.2 окна 2013 считается здесь же
(раньше — отдельный im_timeouts.py с тем же методом).

Методы:
- dfg_exact. Сеть из DFG (pm4py.convert_to_petri_net(dfg)) — автомат: у каждого перехода одно входное и одно
  выходное место, в разметке всегда одна фишка. Оптимальное выравнивание = кратчайший путь (Дейкстра) по
  состояниям (позиция в трассе, место) с ценами PM4Py: синхронный ход 0, ход по журналу 10000, видимый ход
  по модели 10000, тихий 1. Это та же задача, которую решает A* в PM4Py, поэтому стоимость совпадает точно.
  Перед записью стоимость сверяется со ВСЕМИ уже выровненными PM4Py вариантами; при любом расхождении — стоп.
- tbr_fit. Вариант, который token replay проигрывает идеально, — допустимая последовательность срабатываний
  до финальной разметки: выравнивание без отклонений, fitness трассы = 1 точно (cost в пределах 0..число тихих
  переходов — отсюда нижняя и верхняя оценка fitness журнала, разница ~1e-5). Как в im_timeouts.py.
- align_<N>s. Обычные alignments PM4Py с увеличенным лимитом на вариант.
"""
import heapq
import json
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import align_parallel as ap  # noqa: E402
import settings as cfg  # noqa: E402

CONFIG = "long180_v2"
MODELS = cfg.BEST_180
MOVE, TAU = 10000, 1
SPLIT_LIMIT, IM_LIMIT = cfg.SPLIT_LIMIT, cfg.IM_LIMIT
WORKERS = cfg.N_WORKERS
EXPS = {}                                    # часть задаётся в командной строке


def now():
    return time.strftime("%H:%M:%S")


def jl_read(f):
    f = Path(f)
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()] if f.exists() else []


# ---------------------------------------------------------------- сети и варианты из prepared\\
_CACHE = {}


def experiment(exp):
    """Варианты test в том же порядке, что в alignment_fitness_resumable."""
    if exp in _CACHE:
        return _CACHE[exp]
    keys, counts = ap.load_variants(exp, CONFIG)
    out = cfg.out_dir(exp, CONFIG)
    variants = {k: [ap.make_trace(k), n] for k, n in zip(keys, counts)}
    print(now(), f"{exp}: вариантов в test {len(keys):,}", flush=True)
    _CACHE[exp] = dict(variants=variants, keys=keys, out=out, ck=out / "align_ckpt")
    return _CACHE[exp]


def net_file(exp, name):
    return ap.prepared_dir(exp, CONFIG) / "nets" / f"{name}.pkl"


def model(exp, name):
    E = experiment(exp)
    key = ("net", name)
    if key not in E:
        E[key] = ap.load_net(net_file(exp, name))
    return E[key]


def checkpoint(exp, name):
    E = experiment(exp)
    done = {r["i"]: r for r in jl_read(E["ck"] / f"{name}_align.jsonl")}
    if len(done) != len(E["keys"]):
        raise SystemExit(f"{exp} {name}: в контрольной точке {len(done)} вариантов, в test {len(E['keys'])} — стоп")
    for i, k in enumerate(E["keys"]):
        if done[i]["len"] != len(k):
            raise SystemExit(f"{exp} {name}: порядок вариантов не совпал (вариант {i}) — стоп")
    return done


def bwc_model(net, im, fm):
    from pm4py.algo.conformance.alignments.petri_net import algorithm as ali
    return getattr(ali, "__get_best_worst_cost")(net, im, fm, ali.DEFAULT_VARIANT, {})


def fitness_of(cost, bwc):
    return 1 - (cost // MOVE) / (bwc // MOVE) if bwc > 0 else 0.0


# ---------------------------------------------------------------- точное выравнивание для сети-автомата
def state_machine(net, im, fm):
    places = list(net.places)
    idx = {p: j for j, p in enumerate(places)}
    out = [[] for _ in places]
    for t in net.transitions:
        if len(t.in_arcs) != 1 or len(t.out_arcs) != 1:
            raise SystemExit(f"переход {t} не автоматный (входов {len(t.in_arcs)}, выходов {len(t.out_arcs)}) — стоп")
        a_in, a_out = next(iter(t.in_arcs)), next(iter(t.out_arcs))
        if a_in.weight != 1 or a_out.weight != 1:
            raise SystemExit("вес дуги не 1 — стоп")
        out[idx[a_in.source]].append((t.label, idx[a_out.target]))
    if sum(im.values()) != 1 or sum(fm.values()) != 1:
        raise SystemExit("в начальной / финальной разметке не одна фишка — стоп")
    return out, idx[next(iter(im))], idx[next(iter(fm))]


def sm_cost(sm, trace):
    """Стоимость оптимального выравнивания (цены PM4Py) — Дейкстра по (позиция в трассе, место)."""
    out, p0, pf = sm
    n = len(trace)
    dist = {(0, p0): 0}
    heap = [(0, 0, p0)]
    while heap:
        c, i, p = heapq.heappop(heap)
        if c > dist.get((i, p), float("inf")):
            continue
        if i == n and p == pf:
            return c
        steps = [(i + 1, p, MOVE)] if i < n else []                       # ход по журналу
        for lab, q in out[p]:
            if lab is None:
                steps.append((i, q, TAU))                                  # тихий переход
            else:
                steps.append((i, q, MOVE))                                 # ход по модели
                if i < n and lab == trace[i]:
                    steps.append((i + 1, q, 0))                            # синхронный ход
        for j, q, w in steps:
            nc = c + w
            if nc < dist.get((j, q), float("inf")):
                dist[(j, q)] = nc
                heapq.heappush(heap, (nc, j, q))
    return None


def fix_dfg(exp, name="DFG_p0.05"):
    E = experiment(exp)
    done = checkpoint(exp, name)
    to = [i for i in range(len(E["keys"])) if done[i]["timeout"]]
    print(now(), f"{exp} {name}: тайм-аутов {len(to)} вариантов ({sum(done[i]['n'] for i in to)} трасс)", flush=True)
    if not to:
        return
    net, im, fm = model(exp, name)
    sm = state_machine(net, im, fm)
    bm = bwc_model(net, im, fm)
    if sm_cost(sm, ()) != bm:
        raise SystemExit(f"{exp} {name}: стоимость пустой трассы {sm_cost(sm, ())} != bwc PM4Py {bm} — стоп")
    ok = [i for i in range(len(E["keys"])) if not done[i]["timeout"]]
    t0, bad = time.time(), []
    for i in ok:
        c = sm_cost(sm, E["keys"][i])
        b = MOVE * len(E["keys"][i]) + bm
        if c != done[i]["cost"] or b != done[i]["bwc"] or abs(fitness_of(c, b) - done[i]["fitness"]) > 1e-12:
            bad.append((i, c, done[i]["cost"]))
    print(now(), f"  сверка с PM4Py: {len(ok)} вариантов, расхождений {len(bad)} ({time.time() - t0:.1f} с)", flush=True)
    if bad:
        raise SystemExit(f"  расхождения (вариант, наш cost, cost PM4Py): {bad[:10]} — результат НЕ записан")
    recs = []
    for i in to:
        k = E["keys"][i]
        t1 = time.time()
        c = sm_cost(sm, k)
        b = MOVE * len(k) + bm
        recs.append(dict(i=i, n=done[i]["n"], len=len(k), sec=round(time.time() - t1, 3), timeout=False,
                         cost=c, bwc=b, fitness=fitness_of(c, b), how="dfg_exact"))
    f = E["ck"] / f"{name}_timeouts_fixed.jsonl"
    f.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    print(now(), f"  досчитано {len(recs)} вариантов -> {f.name}; fitness трасс: "
                 f"мин {min(r['fitness'] for r in recs):.3f}, медиана "
                 f"{sorted(r['fitness'] for r in recs)[len(recs) // 2]:.3f}", flush=True)


# ---------------------------------------------------------------- TBR + alignments с большим лимитом
def fix_align(exp, name, limit, use_tbr):
    import pm4py
    from pm4py.algo.conformance.alignments.petri_net import algorithm as ali
    from pm4py.algo.conformance.tokenreplay import algorithm as tbr
    from pm4py.objects.log.obj import EventLog
    E = experiment(exp)
    done = checkpoint(exp, name)
    to = [i for i in range(len(E["keys"])) if done[i]["timeout"]]
    f = E["ck"] / f"{name}_timeouts_fixed.jsonl"
    fixed = {r["i"]: r for r in jl_read(f)}
    how = f"align_{limit}s"
    todo = [i for i in to if not (i in fixed and (not fixed[i]["timeout"] or fixed[i].get("how") == how))]
    print(now(), f"{exp} {name}: тайм-аутов {len(to)} вариантов ({sum(done[i]['n'] for i in to)} трасс), "
                 f"уже досчитано {len(to) - len(todo)}, осталось {len(todo)}", flush=True)
    if not todo:
        return
    net, im, fm = model(exp, name)
    bm = bwc_model(net, im, fm)
    with f.open("a", encoding="utf-8") as fh:
        if use_tbr:
            rep = tbr.apply(EventLog([E["variants"][E["keys"][i]][0] for i in todo]), net, im, fm)
            rest = []
            for i, r in zip(todo, rep):
                if r["trace_is_fit"] and r["trace_fitness"] == 1.0:
                    k = E["keys"][i]
                    taus = sum(1 for t in r["activated_transitions"] if t.label is None)
                    rec = dict(i=i, n=done[i]["n"], len=len(k), timeout=False, cost=0, cost_max=taus,
                               bwc=MOVE * len(k) + bm, fitness=1.0, how="tbr_fit")
                    fixed[i] = rec
                    fh.write(json.dumps(rec) + "\n")
                else:
                    rest.append(i)
            fh.flush()
            print(now(), f"  идеально по TBR: {len(todo) - len(rest)} вариантов; alignments с лимитом {limit} с: "
                         f"{len(rest)} (худший случай {len(rest) * limit / 3600:.0f} ч)", flush=True)
            todo = rest
        todo.sort(key=lambda i: len(E["keys"][i]))                   # короткие — первыми
        n_ok = 0
        jobs = {i: E["keys"][i] for i in todo}
        for j, (i, res) in enumerate(ap.align_many(net_file(exp, name), jobs, limit, bm, WORKERS), 1):
            rec = dict(i=i, n=done[i]["n"], len=len(E["keys"][i]), how=how, **res)
            n_ok += not rec["timeout"]
            fixed[i] = rec
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(now(), f"  {j}/{len(todo)}: длина {rec['len']}, {rec['sec']} с, "
                         f"{'тайм-аут' if rec['timeout'] else 'fitness %.3f' % rec['fitness']} "
                         f"(выровнено {n_ok})", flush=True)


# ---------------------------------------------------------------- сверка с прежним расчётом
def check(exp, name, n=50, seed=0):
    """Заново выровнять n случайных вариантов, уже выровненных в eval-ноутбуке, и сравнить cost и время."""
    import random
    from pm4py.algo.conformance.alignments.petri_net import algorithm as ali
    E = experiment(exp)
    done = checkpoint(exp, name)
    ok = [i for i in range(len(E["keys"])) if not done[i]["timeout"]]
    ok_long = sorted(ok, key=lambda i: -done[i]["sec"])[:n // 2]          # самые долгие в прежнем расчёте
    rest = [i for i in ok if i not in set(ok_long)]
    sample = ok_long + random.Random(seed).sample(rest, min(n - len(ok_long), len(rest)))
    net, im, fm = model(exp, name)
    bm = bwc_model(net, im, fm)
    params = {ali.Parameters.PARAM_MAX_ALIGN_TIME_TRACE: 600, ali.Parameters.BEST_WORST_COST_INTERNAL: bm}
    bad, t_old, t_new = [], 0.0, 0.0
    for i in sample:
        t1 = time.time()
        res = ali.apply_trace(E["variants"][E["keys"][i]][0], net, im, fm, parameters=params)
        t_new += time.time() - t1
        t_old += done[i]["sec"]
        if res is None or res["cost"] != done[i]["cost"]:
            bad.append((i, None if res is None else res["cost"], done[i]["cost"]))
    print(now(), f"{exp} {name}: сверено {len(sample)} вариантов, cost не совпал у {len(bad)} {bad[:5]}; "
                 f"время: прежний расчёт {t_old:.0f} с, сейчас {t_new:.0f} с "
                 f"(в {t_old / max(t_new, 1e-9):.0f} раз быстрее)", flush=True)


# ---------------------------------------------------------------- итог
def summary():
    rows = []
    for exp in EXPS:
        lim = cfg.PARTS[exp]["align_limit"]
        out = cfg.out_dir(exp, CONFIG)
        ckdir = "align_ckpt"
        ev = {r["model"]: r for r in jl_read(out / "split_eval.jsonl")}
        pf = ev.get("flower", {}).get("precision_etc_test")
        for m in MODELS:
            base = {r["i"]: r for r in jl_read(out / ckdir / f"{m}_align.jsonl")}
            if not base or m not in ev:
                continue
            fx = {r["i"]: r for r in jl_read(out / ckdir / f"{m}_timeouts_fixed.jsonl")}
            allr = {**base, **{i: r for i, r in fx.items() if not r["timeout"]}}

            def agg(rr):
                ok = [r for r in rr.values() if not r["timeout"]]
                n = sum(r["n"] for r in rr.values())
                n_ok = sum(r["n"] for r in ok)
                sb = sum(r["bwc"] * r["n"] for r in ok)
                lo = 1 - sum(r.get("cost_max", r["cost"]) * r["n"] for r in ok) / sb if sb else None
                hi = 1 - sum(r["cost"] * r["n"] for r in ok) / sb if sb else None
                pct = 100 * sum(r["n"] for r in ok if r["fitness"] == 1.0) / n_ok if n_ok else None
                return n, n - n_ok, lo, hi, pct

            n, to0, _, f0, pct0 = agg(base)
            _, to1, f1lo, f1hi, pct1 = agg(allr)
            p = ev[m].get("precision_etc_test")
            f1 = lambda a: 2 * a * p / (a + p) if a is not None and p else None  # noqa: E731
            hows = {}
            for r in fx.values():
                key = r.get("how", "?") + (" (тайм-аут)" if r["timeout"] else "")
                hows[key] = hows.get(key, 0) + r["n"]
            rows.append(dict(experiment=exp, model=m, align_limit_s=lim, traces=n,
                             timeout_before=to0, timeout_after=to1,
                             fitness_align_before=f0, fitness_align_after=f1lo, fitness_align_after_upper=f1hi,
                             pct_fit_traces_before=pct0, pct_fit_traces_after=pct1,
                             precision_etc_test=p,
                             precision_norm_test=(p - pf) / (1 - pf) if p is not None and pf else None,
                             f1_align_before=f1(f0), f1_align_after=f1(f1lo),
                             fixed_how="; ".join(f"{k}: {v}" for k, v in sorted(hows.items()))))
    q = pd.DataFrame(rows)
    if q.empty:
        print("нет данных для итога")
        return q
    for exp in EXPS:
        q[q.experiment == exp].to_csv(cfg.out_dir(exp, CONFIG) / "split_eval_timeouts_fixed.csv", index=False,
                                      encoding="utf-8-sig")
    show = ["experiment", "model", "timeout_before", "timeout_after", "fitness_align_before",
            "fitness_align_after", "f1_align_before", "f1_align_after"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(q[show].round(4).to_string(index=False))
    print(now(), "итог -> split_eval_timeouts_fixed.csv", flush=True)
    return q


def main(exp, steps):
    EXPS[exp] = None
    try:
        for s in steps:
            t0 = time.time()
            if s == "dfg":
                fix_dfg(exp)
            elif s == "split":
                fix_align(exp, "SplitMiner", SPLIT_LIMIT, use_tbr=False)
            elif s == "im":
                fix_align(exp, "IM_noise0.2", IM_LIMIT, use_tbr=True)
            elif s == "check":
                check(exp, "SplitMiner" if exp == "2013" else "IM_noise0.2")
                continue
            elif s != "summary":
                raise SystemExit(f"неизвестный шаг {s}: dfg | split | im | check | summary")
            if s != "summary":
                print(now(), f"шаг {s}: {(time.time() - t0) / 60:.1f} мин", flush=True)
                summary()
        if "summary" in steps:
            summary()
    except KeyboardInterrupt:
        print("\nпрервано — досчитанное сохранено, повторный запуск продолжит с места", flush=True)
        summary()
    print("FIX_DONE", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for a in sys.argv[1:]:
        if a.startswith("--workers="):
            WORKERS = int(a.split("=")[1])
        if a.startswith("--config="):
            CONFIG = a.split("=")[1]
    main(args[0], args[1:] or cfg.PARTS[args[0]]["fix_steps"])
