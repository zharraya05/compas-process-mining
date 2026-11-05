"""Графы связей файлов (Graphviz). Запуск: python build_graphs.py
  1_старая_папка_петри — как связаны файлы petri_states_* в папке «петри»;
  2_новая_структура    — порядок расчёта в одной части новой папки.
"""
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
FONT = 'fontname="Arial"'
COL = dict(imp="#8a8f98", gen="#9b59b6", wr="#2a78d6", rd="#1a9e6e", logic="#c0392b")
LEGEND = f'''
  subgraph cluster_legend {{ label="Обозначения"; style="rounded"; color="#bbbbbb"; fontsize=12; {FONT};
    node [shape=plaintext, style=solid, color=white, width=.3, fontsize=11, {FONT}];
    l1 [label="импорт кода"]; l1b [label=""]; l1 -> l1b [color="{COL['imp']}", arrowsize=.7];
    l2 [label="пишет результат"]; l2b [label=""]; l2 -> l2b [color="{COL['wr']}", penwidth=1.6];
    l3 [label="читает результат"]; l3b [label=""]; l3 -> l3b [color="{COL['rd']}", penwidth=1.6];
    %s
  }}
'''


def dot_file(name, body):
    p = HERE / f"{name}.dot"
    p.write_text(body, encoding="utf-8")
    for fmt in ("svg", "png"):
        subprocess.run(["dot", f"-T{fmt}", "-Gdpi=130", str(p), "-o", str(HERE / f"{name}.{fmt}")], check=True)
    print(name)


def node(i, label, kind):
    st = {"in": 'shape=cylinder, fillcolor="#f3ead7"', "core": 'shape=box, fillcolor="#dfe3ea"',
          "nb": 'shape=box, style="rounded,filled", fillcolor="#d6e6f7"', "py": 'shape=box, fillcolor="#fde3cf"',
          "gen": 'shape=box, style="filled,dashed", fillcolor="#efe4f5"', "out": 'shape=folder, fillcolor="#fff6c9"',
          "rep": 'shape=box, style="rounded,filled", fillcolor="#d5f0e3"', "xl": 'shape=note, fillcolor="#d5f0e3"',
          "bad": 'shape=box, style="filled,dashed", fillcolor="#fbd9d5"'}[kind]
    return f'  {i} [label="{label}", {st}];\n'


def edges(pairs, kind):
    extra = {"imp": f'color="{COL["imp"]}", arrowsize=.6', "gen": f'color="{COL["gen"]}", style=dashed, arrowsize=.7',
             "wr": f'color="{COL["wr"]}", penwidth=1.6', "rd": f'color="{COL["rd"]}", penwidth=1.6',
             "logic": f'color="{COL["logic"]}", style=dotted, penwidth=1.5, constraint=false'}[kind]
    return "".join(f"  {a} -> {b} [{extra}];\n" for a, b in pairs)


def cluster(i, title, nodes, color="#cccccc"):
    return (f'  subgraph cluster_{i} {{ label="{title}"; style="rounded"; color="{color}"; fontsize=13; {FONT};\n'
            + "".join(node(*n) for n in nodes) + "  }\n")


def old_graph():
    b = f'''digraph G {{ rankdir=LR; ranksep=1.0; nodesep=.16; newrank=true; labelloc=t; fontsize=18; {FONT};
  label="Папка «петри»: файлы журнала состояний petri_states_*";
  node [style=filled, fontsize=11, {FONT}, margin="0.08,0.04"]; edge [{FONT}, fontsize=9];
'''
    b += cluster("in", "Данные", [("db", "compas.db\\n(stage0 - data)", "in"), ("cc", "crime_categories.py\\n(stage0 ver2 from charge)", "core")])
    b += cluster("core", "Ядро", [("pp", "petri_pipeline.py", "core"), ("ps", "petri_states.py", "core"), ("fa", "flags_analysis.py", "core")])
    b += cluster("gen", "Генераторы ноутбуков (в расчёте не участвуют)", [
        ("g1", "make_notebook_states.py", "gen"), ("g2", "make_notebook_flags.py", "gen"), ("g3", "make_notebook_states_eval.py", "gen"),
        ("g4", "make_notebook_results.py", "gen"), ("g5", "make_notebook_results_v2.py", "gen")])
    b += cluster("calc", "Расчёт", [
        ("n1", "petri_states.ipynb\\nвсе майнеры, весь журнал", "nb"), ("n2", "petri_states_flags.ipynb\\nподбор SHORT / LONG", "nb"),
        ("n3", "petri_states_eval_2013.ipynb\\npetri_states_eval_all.ipynb\\ntrain/test + alignments", "nb"),
        ("s1", "run_split.py\\n(пути /home/claude)", "bad"), ("s2", "retry_timeouts.py\\n(пути /home/claude)", "bad"), ("s3", "im_timeouts.py\\n(пути /home/claude)", "bad"),
        ("s4", "fix_timeouts_051126.py", "py"), ("s5", "run_states_v2.py", "py"), ("s6", "run_states_v2_recheck.py", "py")], "#7aa6d6")
    b += node("pk", "states_*.pkl\\nтолько в облаке, на диске нет", "bad")
    b += cluster("out", "Результаты", [
        ("o1", "out_states\\\\&lt;2013|all&gt;\\\\\\n15 / 365 / end v1", "out"), ("o2", "out_states\\\\flags\\\\", "out"),
        ("o3", "out_states\\\\&lt;2013|all&gt;_long180\\\\\\n15 / 180 / end v1", "out"), ("o4", "out_states_v2\\\\&lt;2013|all&gt;\\\\\\n15 / 180 / end v2", "out"),
        ("o5", "out_states_v2_recheck\\\\", "out")], "#d6b656")
    b += cluster("rep", "Отчёты (только читают)", [
        ("r1", "petri_states_results_2013.ipynb\\npetri_states_results_all.ipynb", "rep"),
        ("r2", "petri_states_v2_results_2013.ipynb\\npetri_states_v2_results_all.ipynb", "rep"),
        ("r3", "make_results_tables.py", "rep"), ("x1", "results_states_2013.xlsx\\nresults_states_all.xlsx", "xl")], "#6fbf95")
    b += cluster("orphan", "Ни с чем не связаны", [("f1", "flags_plots.py\\nникто не импортирует", "bad"), ("f2", "out_states\\\\long180_vs_365.xlsx\\nскрипта-источника нет", "bad")], "#d98880")
    b += edges([("db", "pp"), ("cc", "pp"), ("pp", "ps"), ("ps", "fa"), ("pp", "fa"), ("ps", "n1"), ("ps", "n3"), ("fa", "n2"), ("ps", "n2"),
                ("ps", "s1"), ("ps", "s2"), ("ps", "s3"), ("ps", "s4"), ("ps", "s5"), ("s5", "s6")], "imp")
    b += edges([("g1", "n1"), ("g2", "n2"), ("g3", "n3"), ("g4", "r1"), ("g5", "r2")], "gen")
    b += edges([("n1", "o1"), ("n2", "o2"), ("n3", "o3"), ("s1", "o1"), ("s2", "o1"), ("s3", "o1"), ("s3", "o3"), ("s4", "o3"), ("s5", "o4"), ("s6", "o5"), ("r3", "x1")], "wr")
    b += edges([("pk", "s1"), ("pk", "s2"), ("o1", "n3"), ("o1", "r1"), ("o2", "r1"), ("o1", "r2"), ("o4", "r2"), ("o1", "r3"), ("o2", "r3"), ("o4", "s6"), ("o4", "g5")], "rd")
    b += edges([("n2", "n3"), ("n2", "s5")], "logic")
    b += LEGEND % (f'l4 [label="генерирует ноутбук"]; l4b [label=""]; l4 -> l4b [color="{COL["gen"]}", style=dashed];'
                   f'l5 [label="по смыслу: выбор LONG = 180"]; l5b [label=""]; l5 -> l5b [color="{COL["logic"]}", style=dotted, penwidth=1.5];')
    dot_file("1_старая_папка_петри", b + "}\n")


def new_graph():
    b = f'''digraph G {{ rankdir=LR; ranksep=.9; nodesep=.2; newrank=true; labelloc=t; fontsize=18; {FONT};
  label="Новая папка: порядок расчёта. Красное считается только при RUN_V1 = True. Схема одна для с_окном_2013 и без_окна; результаты частей не зависят друг от друга";
  node [style=filled, fontsize=11, {FONT}, margin="0.08,0.05"]; edge [{FONT}, fontsize=9];
'''
    b += cluster("in", "common + data", [("db", "data\\\\compas.db", "in"), ("cm", "settings.py, petri_pipeline.py,\\npetri_states.py, flags_analysis.py,\\neval_resumable.py, crime_categories.py", "core"),
                                         ("ap", "align_parallel.py\\nfix_timeouts.py", "core")])
    b += cluster("par", "Шаг 1: по одному ноутбуку, в любом порядке (память)", [
        ("n1", "01_states_all_miners\\nвариант 1", "bad"), ("n2", "02_flags\\nподбор флагов", "nb"),
        ("n3", "03_tbr_precision\\nсети, TBR, precision", "nb"), ("n5", "05_v2_tbr\\nверсия 2, все майнеры", "nb")], "#7aa6d6")
    b += cluster("o", "out\\\\", [("o1", "long365_v1\\\\\\nвариант 1", "bad"), ("o2", "flags\\\\\\nперебор при RUN_V1", "out"), ("o3", "long180_v2\\\\\\nprepared\\\\ (сети, варианты)\\nalign_ckpt\\\\*_tbr.json", "out"), ("o5", "long180_v2\\\\\\nstate_log, split_quality", "out")], "#d6b656")
    b += cluster("al", "Шаг 2: долгий расчёт, можно обе части сразу", [("n4", "04_alignments\\nпроцессы берут варианты трасс", "nb"), ("o4", "long180_v2\\\\split_eval.jsonl\\nsplit_eval_timeouts_fixed.csv", "out")], "#7aa6d6")
    b += cluster("rep", "Шаг 3: только чтение, минуты", [("r6", "06_results_long180", "rep"), ("r7", "07_results_long365\\nвариант 1", "bad"), ("r8", "08_results_v2", "rep")], "#6fbf95")
    b += edges([("db", "n1"), ("db", "n2"), ("db", "n3"), ("db", "n5"), ("cm", "n1"), ("cm", "n2"), ("cm", "n3"), ("cm", "n5"), ("ap", "n4")], "imp")
    b += edges([("n1", "o1"), ("n2", "o2"), ("n3", "o3"), ("n5", "o5"), ("n4", "o4")], "wr")
    b += edges([("o3", "n4"), ("o4", "r6"), ("o1", "r7"), ("o5", "r8")], "rd")
    b += LEGEND % ""
    dot_file("2_новая_структура", b + "}\n")


old_graph()
new_graph()
