"""Сводные таблицы результатов журнала состояний: два файла Excel.

    results_states_2013.xlsx — окно с 2013-01-01
    results_states_all.xlsx  — вся история, без окна

Запуск из папки «петри»:  python make_results_tables.py
Берёт файлы из out_states\\<эксперимент>\\ и out_states\\flags\\; перезапускать после досчёта тайм-аутов.
"""
import json
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HERE = Path(__file__).resolve().parent
OUT = HERE / "out_states"
FONT = "Arial"
HEAD_FILL = PatternFill("solid", start_color="DDE5F0")
THIN = Side(style="thin", color="B0B0B0")
BEST = ["flower", "DFG_p0.05", "HM_dep0.95", "SplitMiner", "IM_noise0.2"]

EXP = {
    "2013": dict(title="окно с 2013-01-01", file="results_states_2013.xlsx", sweep_exp="2013-01-01",
                 limit=120, align_dir="align_ckpt"),
    "all": dict(title="вся история (без окна)", file="results_states_all.xlsx", sweep_exp="all",
                limit=20, align_dir="align_ckpt_20s"),
}

RU = {  # названия столбцов
    "model": "модель", "split": "выборка", "traces": "трасс", "places": "мест", "transitions": "переходов",
    "silent": "тихих переходов", "arcs": "дуг", "fitness_tbr": "fitness TBR", "pct_fit_traces_tbr": "% трасс без ошибок (TBR)",
    "sec_tbr": "сек TBR", "precision_etc": "precision ETC", "sec_prec": "сек precision", "f1": "F1",
    "sec": "сек всего", "precision_flower": "precision цветочной", "precision_norm": "precision норм.",
    "train_traces": "трасс train", "test_traces": "трасс test", "fitness_tbr_train": "fitness TBR train",
    "fitness_tbr_test": "fitness TBR test", "pct_fit_traces_tbr_test": "% трасс без ошибок test (TBR)",
    "precision_etc_test": "precision ETC test", "easy_sound": "easy sound",
    "fitness_align_test": "alignment fitness test", "pct_fit_traces_align_test": "% трасс без ошибок test (alignments)",
    "align_traces": "трасс в alignments", "align_timeout": "трасс с тайм-аутом", "align_complete": "alignments досчитаны",
    "sec_align_session": "сек alignments", "experiment": "эксперимент", "align_max_sec_trace": "лимит, с на трассу",
    "finished": "закончено", "note": "примечание", "retry_of": "исходный лимит, с",
    "align_timeout_before": "тайм-аутов до досчёта",
}


def read_jsonl(p):
    return pd.DataFrame([json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]) \
        if p.exists() else pd.DataFrame()


def style_header(ws, row, ncol):
    for c in range(1, ncol + 1):
        cell = ws.cell(row, c)
        cell.font = Font(name=FONT, bold=True)
        cell.fill = HEAD_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        cell.border = Border(bottom=THIN)


def put_df(ws, df, row=1, note=None):
    """Таблица со строки row; возвращает номер строки после неё."""
    if note:
        ws.cell(row, 1, note).font = Font(name=FONT, italic=True, color="555555")
        row += 1
    df = df.rename(columns=RU)
    for j, col in enumerate(df.columns, 1):
        ws.cell(row, j, str(col))
    style_header(ws, row, len(df.columns))
    for i, rec in enumerate(df.itertuples(index=False), row + 1):
        for j, v in enumerate(rec, 1):
            if isinstance(v, float) and pd.isna(v):
                v = None
            elif hasattr(v, "item"):
                v = v.item()
            cell = ws.cell(i, j, v)
            cell.font = Font(name=FONT)
            if isinstance(v, float):
                cell.number_format = "0.000"
    return row + len(df) + 2


def finish(ws, widths=None, freeze="B2"):
    for j in range(1, ws.max_column + 1):
        letter = get_column_letter(j)
        ws.column_dimensions[letter].width = (widths or {}).get(j, 16)
    ws.row_dimensions[1].height = 45
    if freeze:
        ws.freeze_panes = freeze


def summary_sheet(ws, key, mq, se, rt):
    """Лучшие модели: метрики со ссылками-формулами (precision норм., F1, доля тайм-аутов)."""
    lim = EXP[key]["limit"]
    hdr = ["модель", "fitness TBR (весь журнал)", "precision ETC (весь журнал)", "F1 (весь журнал)",
           "fitness TBR train", "fitness TBR test", "precision ETC test", "precision цветочной test",
           "precision норм. test", "F1 TBR test", "easy sound", "alignment fitness test",
           "F1 alignments test", "лимит alignments, с", "трасс test", "трасс с тайм-аутом", "доля тайм-аутов",
           "откуда alignments"]
    for j, h in enumerate(hdr, 1):
        ws.cell(1, j, h)
    style_header(ws, 1, len(hdr))
    flower_row = 2  # цветочная — первая строка, на неё ссылается precision норм.
    for i, m in enumerate(BEST, 2):
        q = mq[mq.model == m].iloc[0] if (mq.model == m).any() else None
        s = se[se.model == m].iloc[0] if (not se.empty and (se.model == m).any()) else None
        r = rt[rt.model == m].iloc[-1] if (not rt.empty and (rt.model == m).any()) else None
        src = s
        lim_m, origin = lim, "split_eval.jsonl"
        if r is not None:
            src, lim_m, origin = r, int(r["align_max_sec_trace"]), f"досчёт, лимит {int(r['align_max_sec_trace'])} с"

        def g(row, k):
            if row is None or k not in row.index or pd.isna(row[k]):
                return None
            v = row[k]
            return v.item() if hasattr(v, "item") else v

        vals = {
            1: m, 2: g(q, "fitness_tbr"), 3: g(q, "precision_etc"), 4: g(q, "f1"),
            5: g(s, "fitness_tbr_train"), 6: g(s, "fitness_tbr_test"), 7: g(s, "precision_etc_test"),
            8: f"=$G${flower_row}",
            9: f'=IF(A{i}="flower","",(G{i}-H{i})/(1-H{i}))',
            10: f'=IF(A{i}="flower","",2*F{i}*G{i}/(F{i}+G{i}))',
            11: None if s is None or "easy_sound" not in s.index or pd.isna(s.get("easy_sound")) else bool(s["easy_sound"]),
            12: g(src, "fitness_align_test"),
            13: f'=IF(ISNUMBER(L{i}),2*L{i}*G{i}/(L{i}+G{i}),"")',
            14: None if m == "flower" else lim_m,
            15: g(s, "test_traces"),
            16: g(src, "align_timeout"),
            17: f'=IF(ISNUMBER(P{i}),P{i}/O{i},"")',
            18: "" if m == "flower" else origin,
        }
        if m == "flower":
            vals[16] = None
        for j, v in vals.items():
            c = ws.cell(i, j, v)
            c.font = Font(name=FONT, bold=(j == 1))
            if j in (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13):
                c.number_format = "0.000"
            if j == 17:
                c.number_format = "0.0%"
    n = 2 + len(BEST) + 1
    notes = [
        "Как читать:",
        "весь журнал — сеть строится и проверяется на одном журнале (лист «Все майнеры»);",
        "train/test — люди 80/20, seed 42; сеть на train, метрики на test (лист «train-test»);",
        "precision норм. = (P − P цветочной) / (1 − P цветочной): 0 — не лучше цветочной сети, 1 — максимум;",
        "F1 = 2·fitness·precision / (fitness + precision);",
        "alignment fitness считается по выровненным трассам; трассы с тайм-аутом в неё не входят;",
        "если для модели есть досчёт тайм-аутов с большим лимитом — берётся он (лист «Досчёт тайм-аутов»);",
        "easy sound = FALSE — финальная разметка недостижима, alignments не определены.",
    ]
    for k, t in enumerate(notes):
        ws.cell(n + k, 1, t).font = Font(name=FONT, italic=(k > 0), bold=(k == 0))
    finish(ws, {1: 14, 18: 22})


def build(key):
    cfg = EXP[key]
    d = OUT / key
    mq = pd.read_csv(d / "model_quality.csv")
    se = read_jsonl(d / "split_eval.jsonl")
    rt = pd.concat([read_jsonl(p).assign(_f=p.name) for p in sorted(d.glob("split_eval_retry*s.jsonl"))],
                   ignore_index=True) if list(d.glob("split_eval_retry*s.jsonl")) else pd.DataFrame()
    if not rt.empty:
        rt = rt.sort_values(["model", "align_max_sec_trace"]).drop(columns="_f")
    fl = OUT / "flags"

    wb = Workbook()
    ws = wb.active
    ws.title = "Описание"
    lines = [
        (f"Журнал состояний — результаты: {cfg['title']}", True),
        ("", False),
        ("Флаги журнала во всех расчётах, кроме перебора флагов: SHORT_CUSTODY_DAYS = 15, LONG_CUSTODY_DAYS = 365.", False),
        ("", False),
        ("Лист — что в нём — откуда", True),
        ("Сводка — лучшие модели: весь журнал, train/test, итоговые alignments — собрано из листов ниже", False),
        ("Все майнеры — все майнеры, сеть строится и проверяется на всём журнале — "
         f"out_states\\{key}\\model_quality.csv (petri_states.ipynb, шаг 3)", False),
        ("train-test — лучшие майнеры, люди 80/20, alignments на всём test, лимит "
         f"{cfg['limit']} с на трассу — out_states\\{key}\\split_eval.jsonl (petri_states_eval_{key}.ipynb, run_split.py)", False),
        ("Досчёт тайм-аутов — трассы, не уложившиеся в лимит, пересчитаны с большим лимитом — "
         f"out_states\\{key}\\split_eval_retry*s.jsonl (retry_timeouts.py)", False),
        ("Флаги: перебор — качество сетей при разных SHORT/LONG (train/test, без alignments) — "
         "out_states\\flags\\flags_sweep_table.csv (petri_states_flags.ipynb, шаг 5)", False),
    ]
    if key == "2013":
        lines.append(("Флаги: медианы переходов — время между состояниями журнала (окно 2013, флаги 15/365) — "
                      "out_states\\flags\\transition_medians_current.csv (petri_states_flags.ipynb, шаг 2)", False))
    else:
        lines += [
            ("Флаги: зоны правил — длительность эпизодов заключения по зонам правил 3–7 (вся база) — "
             "out_states\\flags\\domain_medians_*.csv (petri_states_flags.ipynb, шаги 1 и 6)", False),
            ("Флаги: профиль LONG / SHORT — доли по длительности эпизода (вся база) — "
             "out_states\\flags\\long_profile.csv, short_profile.csv (шаги 3 и 4)", False),
        ]
    lines += [("", False),
              ("Таблица собирается скриптом make_results_tables.py; после досчёта тайм-аутов — запустить заново.", False),
              (f"Собрано: {pd.Timestamp.now():%Y-%m-%d %H:%M}", False)]
    for i, (t, b) in enumerate(lines, 1):
        ws.cell(i, 1, t).font = Font(name=FONT, bold=b, size=12 if i == 1 else 10)
    ws.column_dimensions["A"].width = 150

    summary_sheet(wb.create_sheet("Сводка"), key, mq, se, rt)

    ws = wb.create_sheet("Все майнеры")
    cols = ["model", "places", "transitions", "silent", "arcs", "fitness_tbr", "pct_fit_traces_tbr",
            "precision_etc", "precision_flower", "precision_norm", "f1", "sec"]
    put_df(ws, mq[cols].sort_values("f1", ascending=False))
    finish(ws, {1: 14})

    ws = wb.create_sheet("train-test")
    cols = [c for c in ["model", "places", "transitions", "silent", "train_traces", "test_traces",
                        "fitness_tbr_train", "fitness_tbr_test", "pct_fit_traces_tbr_test", "precision_etc_test",
                        "easy_sound", "fitness_align_test", "pct_fit_traces_align_test", "align_traces",
                        "align_timeout", "align_complete", "align_max_sec_trace", "sec_align_session",
                        "finished", "note"] if c in se.columns]
    put_df(ws, se[cols])
    finish(ws, {1: 14, len(cols): 50 if "note" in cols else 16})

    ws = wb.create_sheet("Досчёт тайм-аутов")
    if rt.empty:
        ws.cell(1, 1, "Досчёт ещё не закончен ни для одной модели.").font = Font(name=FONT)
    else:
        cols = ["model", "retry_of", "align_max_sec_trace", "align_timeout_before", "align_timeout",
                "align_traces", "fitness_align_test", "pct_fit_traces_align_test", "finished"]
        put_df(ws, rt[cols])
        finish(ws, {1: 14})

    sw = pd.read_csv(fl / "flags_sweep_table.csv")
    sw = sw[sw["exp"].astype(str) == cfg["sweep_exp"]].drop(columns="exp")
    ws = wb.create_sheet("Флаги перебор")
    put_df(ws, sw.sort_values(["seed", "short", "long"]).rename(columns={"long": "long (100000 = без правила 5)"}))
    finish(ws, {}, freeze="A2")

    if key == "2013":
        ws = wb.create_sheet("Флаги медианы переходов")
        put_df(ws, pd.read_csv(fl / "transition_medians_current.csv"), note="Дни; флаги SHORT = 15, LONG = 365.")
        finish(ws, {1: 55}, freeze=None)
    else:
        ws = wb.create_sheet("Флаги зоны правил")
        r = put_df(ws, pd.read_csv(fl / "domain_medians_current.csv"), note="Длительность эпизода, дни. Текущие флаги:")
        put_df(ws, pd.read_csv(fl / "domain_medians_recommended.csv"), row=r, note="Рекомендованные флаги:")
        finish(ws, {1: 55}, freeze=None)
        ws = wb.create_sheet("Флаги профиль LONG SHORT")
        r = put_df(ws, pd.read_csv(fl / "long_profile.csv"), note="LONG: эпизоды по длительности (дни)")
        put_df(ws, pd.read_csv(fl / "short_profile.csv"), row=r, note="SHORT: события внутри срока по длительности заключения (дни)")
        finish(ws, {1: 20}, freeze=None)

    path = HERE / cfg["file"]
    wb.save(path)
    print("сохранено", path)
    return path


if __name__ == "__main__":
    for k in EXP:
        build(k)
