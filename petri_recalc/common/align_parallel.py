"""Параллельные alignments: заменяет последовательный ps.alignment_fitness_resumable, run_split.py и retry_timeouts.py.

Как устроено и почему это быстрее:
  * 03_tbr_precision.ipynb один раз строит журнал и сети и кладёт их в out\\<конфигурация>\\prepared\\
    (nets\\<модель>.pkl и variants.pkl). Здесь журнал не строится: процессы загружают готовую сеть,
    и все считают ОДНУ И ТУ ЖЕ сеть, а не заново построенную.
  * Единица работы — один вариант трассы, а не модель: свободный процесс берёт следующий вариант любой модели
    и любой конфигурации. Долгие варианты не задерживают остальные.
  * Записывает в контрольные точки только главный процесс, формат тот же, что в ps.alignment_fitness_resumable.
    Прерывание безопасно: повторный запуск продолжит с места.

Метрика считается как раньше: 1 − Σcost / Σbwc по трассам; вариант, не уложившийся в лимит, в неё не входит.

Запуск (обычно из 04_alignments.ipynb):
    python align_parallel.py 2013 --configs long180_v2 --workers 2
    python align_parallel.py 2013 --configs long180_v2 --retry            # тайм-ауты с большим лимитом
"""
import argparse
import contextlib
import io
import json
import os
import pickle
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import settings as cfg  # noqa: E402

_NETS = {}


def now():
    return time.strftime("%H:%M:%S")


def jl_read(f):
    f = Path(f)
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()] if f.exists() else []


# ---------------------------------------------------------------- подготовка (из 03_tbr_precision.ipynb)
def prepared_dir(part, config) -> Path:
    return cfg.out_dir(part, config) / "prepared"


def variants_of(te):
    """Варианты test: тот же порядок, что в ps.alignment_fitness_resumable. -> (keys, counts)."""
    import pm4py
    counts = {}
    for tr_ in pm4py.convert_to_event_log(te):
        k = tuple(e["concept:name"] for e in tr_)
        counts[k] = counts.get(k, 0) + 1
    keys = list(counts)
    return keys, [counts[k] for k in keys]


def save_prepared(part, config, models, te):
    d = prepared_dir(part, config)
    (d / "nets").mkdir(parents=True, exist_ok=True)
    keys, counts = variants_of(te)
    (d / "variants.pkl").write_bytes(pickle.dumps(dict(keys=keys, counts=counts), protocol=5))
    for name, m in models.items():
        (d / "nets" / f"{name}.pkl").write_bytes(pickle.dumps(m, protocol=5))
    print(f"{part} {config}: сохранено {len(models)} сетей и {len(keys):,} вариантов -> {d}", flush=True)
    return d


def load_variants(part, config):
    v = pickle.loads((prepared_dir(part, config) / "variants.pkl").read_bytes())
    return v["keys"], v["counts"]


def load_net(net_file):
    return pickle.loads(Path(net_file).read_bytes())


def make_trace(key):
    from pm4py.objects.log.obj import Event, Trace
    return Trace([Event({"concept:name": a}) for a in key])


# ---------------------------------------------------------------- работа процесса
def _init():
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    with contextlib.redirect_stdout(io.StringIO()):       # баннер pm4py
        import pm4py  # noqa: F401
        from pm4py.algo.conformance.alignments.petri_net import algorithm  # noqa: F401


def _net(net_file):
    if net_file not in _NETS:
        _NETS[net_file] = load_net(net_file)
    return _NETS[net_file]


def task_sound(net_file):
    """Easy soundness с лимитом и стоимость пути по модели (bwc) — как в ps.evaluate_split_one."""
    import petri_states as ps
    from pm4py.algo.conformance.alignments.petri_net import algorithm as ali
    net, im, fm = _net(net_file)
    sound = ps.easy_sound_timed(net, im, fm, ps.EASY_SOUND_SEC)
    bwc = getattr(ali, "__get_best_worst_cost")(net, im, fm, ali.DEFAULT_VARIANT, {}) if sound else None
    return sound, bwc


def task_align(net_file, key, limit, bwc):
    from pm4py.algo.conformance.alignments.petri_net import algorithm as ali
    net, im, fm = _net(net_file)
    params = {ali.Parameters.PARAM_MAX_ALIGN_TIME_TRACE: limit, ali.Parameters.BEST_WORST_COST_INTERNAL: bwc}
    t0 = time.time()
    res = ali.apply_trace(make_trace(key), net, im, fm, parameters=params)
    r = dict(sec=round(time.time() - t0, 2))
    if res is None or "fitness" not in res:
        r["timeout"] = True
    else:
        r.update(timeout=False, cost=res["cost"], bwc=res["bwc"], fitness=res["fitness"])
    return r


def pool_map(tasks, workers):
    """tasks — список (метка, функция, аргументы). Выдаёт (метка, результат) по мере готовности."""
    ex = ProcessPoolExecutor(max_workers=workers, initializer=_init)
    futures = {ex.submit(fn, *args): tag for tag, fn, args in tasks}
    try:
        for f in as_completed(futures):
            yield futures[f], f.result()
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def align_many(net_file, keys_by_i, limit, bwc, workers):
    """Для fix_timeouts.py: {i: вариант} -> выдаёт (i, результат)."""
    yield from pool_map([(i, task_align, (str(net_file), k, limit, bwc)) for i, k in keys_by_i.items()], workers)


# ---------------------------------------------------------------- задания
class Job:
    """Одна модель одной конфигурации: основной расчёт (limit) или досчёт тайм-аутов (retry_of)."""

    def __init__(self, part, config, name, limit, retry_of=None):
        self.part, self.config, self.name, self.limit, self.retry_of = part, config, name, limit, retry_of
        self.out = cfg.out_dir(part, config)
        self.base_ck = self.out / "align_ckpt"
        self.net_file = str(prepared_dir(part, config) / "nets" / f"{name}.pkl")
        self.meta_file = self.base_ck / f"{name}_tbr.json"
        if not self.meta_file.exists():
            raise SystemExit(f"нет {self.meta_file} — сначала 03_tbr_precision.ipynb")
        self.meta = json.loads(self.meta_file.read_text(encoding="utf-8"))
        if retry_of is None:
            self.ck, self.res = self.base_ck / f"{name}_align.jsonl", self.out / "split_eval.jsonl"
        else:
            dst = self.out / f"align_ckpt_retry{limit}s"
            dst.mkdir(exist_ok=True)
            self.ck, self.res = dst / f"{name}_align.jsonl", self.out / f"split_eval_retry{limit}s.jsonl"
        self.tag = f"{config}/{name}"

    def finished(self):
        return any(r.get("model") == self.name for r in jl_read(self.res))

    def write_meta(self):
        self.meta_file.write_text(json.dumps(self.meta), encoding="utf-8")

    def append_result(self, r):
        r = {k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()}
        r.pop("bwc_model", None)
        with self.res.open("a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    def base_record(self):
        r = dict(self.meta)
        r.update(experiment=self.part, align_max_sec_trace=self.limit, finished=time.strftime("%Y-%m-%d %H:%M"))
        return r

    def finish_unsound(self):
        r = self.base_record()
        r.update(fitness_align_test=None, pct_fit_traces_align_test=None, align_traces=r["test_traces"],
                 align_timeout=None, align_complete=True)
        self.append_result(r)

    def finish(self, keys, counts):
        done = {r["i"]: r for r in jl_read(self.ck)}
        if self.retry_of is None:
            allr = done
            source_timeouts = None
        else:
            src = {r["i"]: r for r in jl_read(self.base_ck / f"{self.name}_align.jsonl")}
            source_timeouts = sum(r["n"] for r in src.values() if r["timeout"])
            allr = {**{i: r for i, r in src.items() if not r["timeout"]}, **done}
        ok = [r for r in allr.values() if not r["timeout"]]
        n_tr = sum(r["n"] for r in allr.values())
        n_ok = sum(r["n"] for r in ok)
        s_bwc = sum(r["bwc"] * r["n"] for r in ok)
        fit = 1 - sum(r["cost"] * r["n"] for r in ok) / s_bwc if s_bwc else float("nan")
        pct = 100 * sum(r["n"] for r in ok if r["fitness"] == 1.0) / n_ok if n_ok else float("nan")
        if self.retry_of is None:
            r = self.base_record()
            r.update(fitness_align_test=fit, pct_fit_traces_align_test=pct, align_traces=n_tr,
                     align_timeout=n_tr - n_ok, align_complete=len(done) == len(keys),
                     sec_align_session=sum(x["sec"] for x in done.values()))
        else:
            r = dict(model=self.name, experiment=self.part, align_max_sec_trace=self.limit, retry_of=self.retry_of,
                     fitness_align_test=fit, pct_fit_traces_align_test=pct, align_traces=n_tr,
                     align_timeout=n_tr - n_ok, align_timeout_before=source_timeouts,
                     finished=time.strftime("%Y-%m-%d %H:%M"))
        self.append_result(r)
        print(now(), self.tag, "готово: fitness", round(fit, 4), "| тайм-аут трасс", n_tr - n_ok, "из", n_tr, flush=True)


def plan(part, configs, models, retry):
    jobs = []
    p = cfg.PARTS[part]
    for c in configs:
        names = cfg.EVAL_MODELS[c] if models is None else [m for m in models if m in cfg.EVAL_MODELS[c] + ["flower"]]
        if retry:
            jobs += [Job(part, c, m, p["retry_limit"], retry_of=p["align_limit"]) for m in names
                     if m in p["retry_models"]]
        else:
            names = ["flower"] + [m for m in names if m != "flower"]   # нужна для нормировки precision
            jobs += [Job(part, c, m, p["align_limit"]) for m in names]
    return jobs


def run(part, configs, models=None, workers=cfg.N_WORKERS, retry=False):
    t00 = time.time()
    jobs = [j for j in plan(part, configs, models, retry) if not j.finished()]
    for j in [j for j in jobs if j.name == "flower"]:      # цветочная: alignment fitness = 1 по построению
        j.append_result(j.base_record())
    jobs = [j for j in jobs if j.name != "flower"]
    if not jobs:
        print("нечего считать: всё уже записано в результаты", flush=True)
        return
    print(now(), f"часть {part}: заданий {len(jobs)}, процессов {workers}", flush=True)

    # 1. easy soundness: у неограниченной сети проверка идёт до лимита, поэтому тоже в процессах
    need = [j for j in jobs if "bwc_model" not in j.meta and "easy_sound" not in j.meta
            or (j.meta.get("easy_sound") and "bwc_model" not in j.meta)]
    if need:
        print(now(), f"проверка easy soundness: {len(need)} сетей (лимит 600 с на сеть)", flush=True)
        for j, (sound, bwc) in pool_map([(j, task_sound, (j.net_file,)) for j in need], workers):
            j.meta["easy_sound"] = sound
            if sound:
                j.meta["bwc_model"] = bwc
            j.write_meta()
            print(now(), f"  {j.tag}: easy_sound = {sound}", flush=True)
    live = []
    for j in jobs:
        if j.meta.get("easy_sound"):
            live.append(j)
        elif j.retry_of is None:
            print(now(), f"{j.tag}: сеть не easy sound (или проверка не уложилась в лимит) — alignments не считаются",
                  flush=True)
            j.finish_unsound()

    # 2. варианты: короткие первыми, чтобы прогресс был виден сразу
    variants = {}
    tasks, files, left = [], {}, {}
    for j in live:
        keys, counts = variants.setdefault(j.config, load_variants(part, j.config))
        done = {r["i"] for r in jl_read(j.ck)}
        if j.retry_of is None:
            todo = [i for i in range(len(keys)) if i not in done]
        else:
            src = {r["i"]: r for r in jl_read(j.base_ck / f"{j.name}_align.jsonl")}
            todo = [i for i, r in src.items() if r["timeout"] and i not in done]
        left[j] = len(todo)
        files[j] = j.ck.open("a", encoding="utf-8")
        tasks += [((j, i), task_align, (j.net_file, keys[i], j.limit, j.meta["bwc_model"])) for i in todo]
        print(now(), f"  {j.tag}: вариантов {len(keys):,}, уже готово {len(done):,}, осталось {len(todo):,}, "
                     f"лимит {j.limit} с", flush=True)
    tasks.sort(key=lambda t: len(t[2][1]))
    total, n = len(tasks), 0
    try:
        for (j, i), r in pool_map(tasks, workers):
            keys, counts = variants[j.config]
            r = dict(i=i, n=counts[i], len=len(keys[i]), **r)
            files[j].write(json.dumps(r) + "\n")
            files[j].flush()
            left[j] -= 1
            n += 1
            if n % 100 == 0 or n == total:
                print(now(), f"выровнено {n:,} / {total:,}, {(time.time() - t00) / 60:.1f} мин", flush=True)
    finally:
        for f in files.values():
            f.close()
    for j in live:
        if left[j] == 0:
            j.finish(*variants[j.config])
    print(now(), f"ALIGN_DONE, {(time.time() - t00) / 60:.1f} мин", flush=True)


# ---------------------------------------------------------------- запуск из ноутбука
def launch(args):
    """Запустить скрипт из common отдельным процессом, печатая вывод в ячейку.
    Прерывание ячейки (Interrupt) останавливает все процессы; посчитанное сохранено."""
    cmd = [sys.executable, "-u", *map(str, args)]
    print(" ".join(cmd), flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace", env=env)
    try:
        for line in p.stdout:
            print(line, end="", flush=True)
        if p.wait() != 0:
            raise RuntimeError(f"процесс завершился с кодом {p.returncode}")
    except KeyboardInterrupt:
        if os.name == "nt":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(p.pid)], capture_output=True)
        else:
            p.terminate()
        print("\nостановлено: посчитанное сохранено, повторный запуск продолжит с места", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=list(cfg.PARTS))
    ap.add_argument("--configs", nargs="+", default=cfg.EVAL_CONFIGS, choices=list(cfg.EVAL_MODELS))
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=cfg.N_WORKERS)
    ap.add_argument("--retry", action="store_true", help="досчёт тайм-аутов с большим лимитом")
    a = ap.parse_args()
    try:
        run(a.part, a.configs, a.models, a.workers, a.retry)
    except KeyboardInterrupt:
        print("\nпрервано — посчитанное сохранено, повторный запуск продолжит с места", flush=True)
