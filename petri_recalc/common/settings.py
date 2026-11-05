"""Единые настройки пересчёта журнала состояний. Все ноутбуки и скрипты берут пути и флаги отсюда.

Части:         "2013" — с окном (журнал с 2013-01-01), "all" — без окна (вся история).
Конфигурации журнала (раньше различались только папками out_states*):
    long365_v1 — SHORT 15, LONG 365, правило end v1 (было out_states\\<часть>)
    long180_v1 — SHORT 15, LONG 180, правило end v1 (было out_states\\<часть>_long180)
    long180_v2 — SHORT 15, LONG 180, правило end v2 (было out_states_v2\\<часть>)
Результаты каждой части лежат в <папка части>\\out\\<конфигурация>\\ и не пересекаются с другой частью.
Одновременно можно запускать 04_alignments.ipynb двух частей; остальное считающее — по одному (нужна память, см. README).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "common"
DB_CANDIDATES = [ROOT / "data" / "compas.db", ROOT.parent / "stage0-data" / "compas.db"]
CATEGORIES_DIRS = [COMMON]
OLD_ROOT = ROOT.parent / "петри"           # прежняя папка — только для сравнения (compare_with_old.py)

SEED, TEST_SHARE = 42, 0.2

# Сколько процессов считает alignments в ОДНОМ запуске 04_alignments.ipynb.
# Лимит на вариант трассы — по настенным часам: при перегрузке процессора тайм-аутов станет больше.
# На этом компьютере 4 ядра / 8 потоков, 7.8 ГБ: обе части одновременно — по 2, одна часть — 3–4.
N_WORKERS = 3

CONFIGS = {
    "long365_v1": dict(short=15, long=365, end_rule="v1"),
    "long180_v1": dict(short=15, long=180, end_rule="v1"),
    "long180_v2": dict(short=15, long=180, end_rule="v2"),
}

BEST_365 = ["DFG_p0.05", "HM_dep0.95", "SplitMiner", "IM_noise0.2"]
BEST_180 = ["DFG_p0.05", "DFG_p0.0", "HM_dep0.95", "SplitMiner", "IM_noise0.2"]
EVAL_MODELS = {"long365_v1": BEST_365, "long180_v1": BEST_180, "long180_v2": BEST_180}

# Вариант 1 (правило end v1): False — его метрики не считаются, код не удалён.
# Выключает: 01_states_all_miners, 07_results_long365, перебор флагов в 02, сравнения с v1 в 08.
# True — вернуть как было.
RUN_V1 = True
# Конфигурации с разбиением train/test и alignments (03 и 04).
EVAL_CONFIGS = ["long180_v2"] + (["long180_v1", "long365_v1"] if RUN_V1 else [])

PARTS = {
    "2013": dict(folder="с_окном_2013", title="окно с 2013-01-01", log_start="2013-01-01", align_limit=120,
                 retry_limit=600, retry_models=["DFG_p0.05", "SplitMiner", "IM_noise0.2"],
                 fix_steps=["dfg", "split", "im"], sweep_exp="2013-01-01", results_file="results_states_2013.xlsx",
                 sweep={42: [(0, 365), (1, 365), (2, 365), (3, 365), (5, 180), (5, 270), (5, 365), (7, 180), (7, 365), (10, 365), (15, 7), (15, 15), (15, 30), (15, 60), (15, 90), (15, 120), (15, 180), (15, 270), (15, 365), (15, 545), (15, 730), (15, 100000), (20, 365), (30, 365), (45, 365), (60, 365), (90, 365), (120, 365), (180, 365)],
                        7: [(5, 180), (5, 365), (15, 90), (15, 180), (15, 365), (15, 730)]}),
    "all": dict(folder="без_окна", title="вся история (без окна)", log_start=None, align_limit=20,
                retry_limit=120, retry_models=["IM_noise0.2"],
                fix_steps=["dfg", "im"], sweep_exp="all", results_file="results_states_all.xlsx",
                sweep={42: [(5, 180), (15, 180), (15, 365)]}),
}

# досчёт тайм-аутов конфигураций LONG = 180 (fix_timeouts.py)
SPLIT_LIMIT, IM_LIMIT = 1800, 600


def part_dir(part) -> Path:
    return ROOT / PARTS[part]["folder"]


def out_dir(part, sub) -> Path:
    """sub — конфигурация журнала или "flags"."""
    return part_dir(part) / "out" / sub
