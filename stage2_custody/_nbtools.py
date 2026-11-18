"""Сборка и прогон ноутбуков без Jupyter: ячейки исполняются по порядку, вывод (текст, таблицы, рисунки)
записывается в .ipynb. В Jupyter Lab ноутбук перезапускается обычным Run All."""
import ast, base64, contextlib, io, json, os, sys, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd


class NB:
    def __init__(self): self.cells = []
    def md(self, s): self.cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
    def code(self, s): self.cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                                          "source": s.strip("\n").splitlines(True)})

    def run(self, path, execute=True):
        ns = {"__name__": "__main__"}; outs = []
        def display(*objs):
            for o in objs:
                if hasattr(o, "to_html"):
                    outs.append({"output_type": "display_data", "metadata": {},
                                 "data": {"text/html": o.to_html(), "text/plain": repr(o)}})
                else:
                    outs.append({"output_type": "display_data", "metadata": {}, "data": {"text/plain": repr(o)}})
        def show(*a, **k):
            for n in plt.get_fignums():
                b = io.BytesIO(); plt.figure(n).savefig(b, format="png", dpi=110, bbox_inches="tight")
                outs.append({"output_type": "display_data", "metadata": {},
                             "data": {"image/png": base64.b64encode(b.getvalue()).decode(), "text/plain": "<Figure>"}})
            plt.close("all")
        ns["display"] = display; plt.show = show
        n = 0; t_all = time.time()
        for c in self.cells:
            if c["cell_type"] != "code" or not execute: continue
            n += 1; outs.clear(); buf = io.StringIO(); src = "".join(c["source"])
            tree = ast.parse(src); last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr): last = ast.Expression(tree.body.pop().value)
            with contextlib.redirect_stdout(buf):
                exec(compile(tree, f"<cell {n}>", "exec"), ns)
                if last is not None:
                    v = eval(compile(last, f"<cell {n}>", "eval"), ns)
                    if v is not None: display(v)
                show()
            res = []
            if buf.getvalue(): res.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue().splitlines(True)})
            c["outputs"] = res + list(outs); c["execution_count"] = n
        nb = {"cells": self.cells, "nbformat": 4, "nbformat_minor": 5,
              "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                           "language_info": {"name": "python"}}}
        json.dump(nb, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"{path}: {n} ячеек, {time.time() - t_all:.0f} с")
        return ns
