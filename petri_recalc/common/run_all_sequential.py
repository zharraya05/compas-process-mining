"""Последовательный запуск ноутбуков на этом компьютере, по одному, со сторожем памяти.

    python common/run_all_sequential.py [2013 all]

Порядок в части: 02, 03, 04, 06, 05, 08; при RUN_V1 = True ещё 01 в начале и 07 перед 08. Выполненные пропускаются (запись в out\\run_done.txt части).
Прогресс: run_log.txt в корне папки. Прерывание безопасно: повторный запуск продолжит с незаконченного ноутбука,
а 02–05 внутри него — с места.

Сторож: ноутбук снимается, если он занял больше MAX_TREE_GB или в системе кончается память вместе с подкачкой.
"""
import re
import subprocess
import sys
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
import settings as cfg  # noqa: E402

ORDER = ["02_flags", "03_tbr_precision", "04_alignments", "06_results_long180", "05_v2_tbr", "08_results_v2"]
if cfg.RUN_V1:
    ORDER = ["01_states_all_miners", *ORDER[:5], "07_results_long365", "08_results_v2"]
MAX_TREE_GB = 7.5
MIN_FREE_GB = 0.25
MAX_SWAP = 85
LOG = cfg.ROOT / "run_log.txt"
KERN = cfg.ROOT / ".kernel"


def say(text):
    line = f"{time.strftime('%d.%m %H:%M:%S')} {text}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def tree_gb(proc):
    total = 0
    try:
        for p in [proc] + proc.children(recursive=True):
            try:
                total += p.memory_info().rss
            except psutil.Error:
                pass
    except psutil.Error:
        pass
    return total / 2**30


def run_nb(part, name, env):
    d = cfg.part_dir(part)
    logf = d / "out" / f"run_{name}.log"
    t0, peak, reason = time.time(), 0.0, None
    say(f"{part} {name}: старт, свободно {psutil.virtual_memory().available / 2**30:.1f} ГБ")
    with logf.open("w", encoding="utf-8") as f:
        pr = subprocess.Popen([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                               "--ExecutePreprocessor.timeout=-1", "--ExecutePreprocessor.kernel_name=python3",
                               name + ".ipynb"], cwd=d, env=env, stdout=f, stderr=subprocess.STDOUT)
        P = psutil.Process(pr.pid)
        last = time.time()
        while pr.poll() is None:
            g = tree_gb(P)
            peak = max(peak, g)
            vm, sw = psutil.virtual_memory(), psutil.swap_memory()
            if g > MAX_TREE_GB:
                reason = f"занял {g:.1f} ГБ"
            elif vm.available / 2**30 < MIN_FREE_GB and sw.percent > MAX_SWAP:
                reason = f"в системе кончается память: свободно {vm.available / 2**30:.2f} ГБ, подкачка {sw.percent:.0f}%"
            if reason:
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(pr.pid)], capture_output=True)
                break
            if time.time() - last > 1800:
                last = time.time()
                say(f"{part} {name}: идёт {(last - t0) / 3600:.1f} ч, память {g:.2f} ГБ")
            time.sleep(2)
        pr.wait()
    ok = pr.returncode == 0 and reason is None
    tail = ""
    if not ok:
        hits = re.findall(r"\w*Error: .*", logf.read_text(encoding="utf-8", errors="replace"))
        tail = (hits[-1][:160] if hits else "") if reason is None else "снят сторожем: " + reason
    say(f"{part} {name}: {'готово' if ok else 'ОШИБКА'}, {(time.time() - t0) / 60:.1f} мин, пик {peak:.2f} ГБ {tail}")
    return ok


def main(parts):
    if not (KERN / "share/jupyter/kernels/python3/kernel.json").exists():
        subprocess.run([sys.executable, "-m", "ipykernel", "install", "--prefix", str(KERN), "--name", "python3"], check=True)
    import os
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", JUPYTER_PATH=str(KERN / "share/jupyter"))
    for part in parts:
        out = cfg.part_dir(part) / "out"
        out.mkdir(exist_ok=True)
        done_f = out / "run_done.txt"
        done = set(done_f.read_text(encoding="utf-8").split()) if done_f.exists() else set()
        for name in ORDER:
            if name in done:
                continue
            if not run_nb(part, name, env):
                say(f"{part}: остановлено на {name}")
                if name in ("03_tbr_precision", "04_alignments", "06_results_long180"):
                    break                    # от них зависят следующие; часть v2 всё равно попробовать нельзя без них
                continue
            with done_f.open("a", encoding="utf-8") as f:
                f.write(name + "\n")
    say("ВСЁ: очередь закончена")


if __name__ == "__main__":
    main(sys.argv[1:] or list(cfg.PARTS))
