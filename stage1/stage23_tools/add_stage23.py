"""Дописывает в ноутбуки этапа 1 ячейки этапов 2–3 (код + пояснения) и кладёт stage23.py рядом.

python add_stage23.py "<jupyter lab>"  — для всех кейсов; ячейки, добавленные раньше (metadata stage23),
заменяются, остальные ячейки ноутбука не трогаются.
"""
import shutil
import sys
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1])

CASES = {
    "x2_exp1_one_event": dict(nb="x2_exp1_one_event.ipynb", kind="x2", mod_dir=".",
                              title="эксп. 1 «категории × 2, одно событие на заключение»"),
    "x2_exp3_one_event_merge": dict(nb="x2_exp3_one_event_merge.ipynb", kind="x2", mod_dir=".",
                                    title="эксп. 3 «категории × 2, одно событие + редкие OldCase слиты»"),
    "dasha_severity": dict(nb="dasha_severity.ipynb", kind="dasha", mod_dir=".",
                           title="данные Даши, состояние = тяжесть обвинения"),
    "all_new": dict(nb="stage1_all_new.ipynb", kind="variant", mod_dir="code",
                    title="15 категорий, все обвинения новые"),
    "oldcase": dict(nb="stage1_oldcase.ipynb", kind="variant", mod_dir="code",
                    title="15 категорий + OldCase"),
    "no_oldcase": dict(nb="stage1_no_oldcase.ipynb", kind="variant", mod_dir="code",
                       title="15 категорий, без OldCase"),
}

STATE_NOTE = {
    "x2": "Состояние заключения — его единственное событие из журнала этапа 1 (самая тяжёлая категория, "
          "`X` или `OldCase: X`), поэтому граф этапа 1 и переходы здесь совпадают.",
    "dasha": "Состояние заключения — тяжесть самого тяжёлого обвинения (F1, F2, F3, F_other, M1, M2, Unknown), "
             "как в журнале «по заключению» этапа 1. Тяжесть здесь и есть состояние, поэтому признак события "
             "«фелония» внутри одного состояния постоянен и в моделях по состояниям выпадает сам.",
    "variant": "В журнале этапа 1 заключение с несколькими категориями — развилка из нескольких событий. Для "
               "переходов между заключениями нужно одно состояние, поэтому здесь берётся **главная категория** "
               "заключения: самая тяжёлая по обвинению (F1 > F2 > F3 > прочие F > M1 > M2 > MO3), `Justice` — "
               "только если других нет, при равной тяжести случайная (seed 42); это то же правило, что в "
               "эксперименте 1 варианта «категории × 2». Матрица шага 4 выше считает развилки с весами, "
               "поэтому вероятности немного отличаются.",
}

ADAPTER = {
    "x2": '''import stage23 as ST

DB = next((p for p in [ROOT / "stage0 - data" / "compas.db", ROOT / "stage0-data" / "compas.db"] if p.exists()),
          ROOT / "stage0 - data" / "compas.db")
OUT23 = OUT / "stage23"
FIG23 = OUT23 / "figures"
FIG23.mkdir(parents=True, exist_ok=True)
RUN_RSF = True              # случайный лес выживаемости (около минуты)
MIN_ARC_EVENTS = 50         # модели Кокса по дугам — только для дуг с таким числом переходов

people = ST.compas_people(DB)                       # возраст, пол, раса, балл COMPAS из compas.db
E = ST.episodes_x2(st, ch, people, info["obs_end"])
E.drop(columns=["person_id"]).to_csv(OUT23 / "episodes.csv", index=False)
ST.episodes_summary(E)''',
    "dasha": '''import stage23 as ST

OUT23 = OUT / "stage23"
FIG23 = OUT23 / "figures"
FIG23.mkdir(parents=True, exist_ok=True)
RUN_RSF = True              # случайный лес выживаемости (около минуты)
MIN_ARC_EVENTS = 50         # модели Кокса по дугам — только для дуг с таким числом переходов

people = ST.compas_people(DB)                       # балл COMPAS из compas.db (по person_id)
E = ST.episodes_dasha(log, ep, people, S.OBS_END)   # log — журнал «по заключению» (шаг 2)
E.to_csv(OUT23 / "episodes.csv", index=False)
ST.episodes_summary(E)''',
    "variant": '''import stage23 as ST

DB = HERE / ".." / ".." / "stage0 - data" / "compas.db"
OUT23 = OUT / "stage23"
FIG23 = OUT23 / "figures"
FIG23.mkdir(parents=True, exist_ok=True)
RUN_RSF = True              # случайный лес выживаемости (около минуты)
MIN_ARC_EVENTS = 50         # модели Кокса по дугам — только для дуг с таким числом переходов

# обвинения с номером эпизода и тяжестью — для выбора главной категории заключения
ch23, ep23 = load()
ep23 = merge_continuations(ep23)
ch23 = ch23.merge(ep23[["ep_raw", "ep_row"]], on="ep_raw")
people = ST.compas_people(DB)                       # возраст, пол, раса, балл COMPAS из compas.db
E = ST.episodes_variant(df, ch23, people, info["obs_end"])
E.drop(columns=["person_id"]).to_csv(OUT23 / "episodes.csv", index=False)
ST.episodes_summary(E)''',
}


def md(s):
    c = nbformat.v4.new_markdown_cell(s.strip("\n"))
    c.metadata["stage23"] = True
    return c


def code(s):
    c = nbformat.v4.new_code_cell(s.strip("\n"))
    c.metadata["stage23"] = True
    return c


def cells(case, cfg):
    k = cfg["kind"]
    return [
        md(f"""
# Этапы 2–3 (п. 1.1 плана): параметры переходов и модели выживаемости

Кейс: {cfg["title"]}. Журнал и граф этапа 1 построены выше; здесь к ним добавляются

- **Этап 2.** Параметры переходов: вероятности $p_{{ij}}$ и распределения времени между заключениями.
- **Этап 3.** Признаки объекта (возраст, пол, раса, исходный балл COMPAS) и признаки события (тяжесть,
  число обвинений, длительность заключения, номер заключения, старое дело) и модели выживаемости между
  состояниями: CoxPH, дерево и случайный лес выживаемости; влияние признаков на переходы.

**Одна строка = одно заключение человека** (таблица `E`). Из каждого заключения $A$ человек выходит на свободу
и либо попадает в следующее заключение $B$ (переход $A \\to B$), либо наблюдение кончается (цензура).

- **Время перехода** $T$ — время на свободе: от выхода из $A$ до входа в $B$, дни (пересечения заключений
  в данных → 0). Это время риска для конкурирующих рисков.
- **Тип следующего события** $K$ — состояние $B$. Пара $(T, K)$ — то, что моделируется в п. 1.2 плана.
- **Цензура.** Последнее заключение человека цензурировано: $T$ — от выхода до конца наблюдения
  (`obs_end`), терминального состояния END нет. Кто на конец наблюдения ещё в заключении, времени на свободе
  не имеет и в моделях выживаемости не участвует (`at_risk = False`), но учтён в долях цензуры.

{STATE_NOTE[k]}

Общий код этапов 2–3 — `stage23.py` (тот же файл во всех кейсах). Результаты — в `{"notebook_out" if k == "variant" else "out"}\\stage23\\`:
CSV-таблицы, `stage23_tables.xlsx` (все таблицы листами) и рисунки в `figures\\`.
"""),
        code(ADAPTER[k]),
        md("""
**Как читать.** `переходов (наблюдались)` — пары соседних заключений; `цензурировано` — по одному последнему
заключению на человека; `переходов за 0 дней` — следующее заключение началось в день выхода или раньше
(пересечения в данных). `нет балла COMPAS` — у этих людей нет `decile_score`, в моделях этапа 3 они выпадают.
"""),
        md("""
## Этап 2. Параметры переходов

### 2.1. Вероятности переходов

Вложенная цепь Маркова: $p_{ij} = n_{ij} / \\sum_k n_{ik}$ — доля переходов в $j$ среди всех **наблюдавшихся**
переходов из $i$. Доля цензуры — доля заключений $i$, после которых следующего заключения не было до конца
наблюдения. $p_{ij}$ не учитывает, когда случился переход и сколько людей не дожило до него под наблюдением;
это делают кумулятивные функции инцидентности в 2.5.
"""),
        code('''cnt, P, trans_summary = ST.transition_matrix(E)
cnt.to_csv(OUT23 / "transition_counts.csv")
P.round(4).to_csv(OUT23 / "transition_probs.csv")
trans_summary.to_csv(OUT23 / "transition_summary.csv")
ST.plot_matrix(P, FIG23 / "transition_matrix.png", "Вероятности переходов между заключениями",
               n=trans_summary["заключений"], annotate_min=0.02)
show(FIG23 / "transition_matrix.png")
trans_summary'''),
        md("""
**Как читать матрицу.** Строка — состояние текущего заключения (в скобках — число заключений), столбец —
следующего; в клетке $p_{ij}$ (подписаны $\\ge 0.02$). Диагональ — повтор того же состояния (петля графа).
В таблице: сколько заключений в состоянии, сколько из них закончились переходом, доля цензуры и петля $p_{ii}$.
"""),
        md("""
### 2.2. Время на свободе по дугам

Для каждой дуги $i \\to j$ — описательная статистика времени на свободе среди тех, кто совершил этот переход
(дни): число переходов, $p_{ij}$, среднее, квартили, 90-й перцентиль. Показаны дуги с не менее 30 переходами.
"""),
        code('''arcs23 = ST.arc_times(E)
arcs23.to_csv(OUT23 / "arc_times.csv", index=False)
arcs23[arcs23["n"] >= 30].sort_values("n", ascending=False).head(25)'''),
        md("""
### 2.3. Время до следующего заключения из каждого состояния (с цензурой)

Для состояния $i$ — время до следующего заключения любого типа. Цензурированные (последние) заключения
учитываются: кривая Каплана–Мейера $S(t)$ — доля тех, кто ещё не вернулся в заключение к дню $t$.
Подбираются параметрические распределения (экспоненциальное, Вейбулла, логнормальное, лог-логистическое),
лучшее — по минимальному AIC. Параметр формы Вейбулла $\\rho < 1$ означает, что риск нового заключения
со временем на свободе падает (чаще всего возвращаются в первые месяцы).
"""),
        code('''state_dist, state_fits = ST.fit_state_distributions(E)
state_dist.to_csv(OUT23 / "state_time_distributions.csv")
top_states = list(state_dist.index[:6])
ST.plot_state_fits(state_fits, FIG23 / "state_time_fits.png", top_states,
                   "Время до следующего заключения: Каплан–Мейер и лучшее распределение")
show(FIG23 / "state_time_fits.png")
state_dist[["n", "events", "km_median", "S(365)", "S(730)", "weibull_rho", "best", "best_median"]]'''),
        md("""
**Как читать.** `km_median` — медиана Каплана–Мейера (дни; `inf` — больше половины не вернулись до конца
наблюдения), `S(365)` и `S(730)` — доля не вернувшихся за 1 и 2 года, `weibull_rho` — форма Вейбулла,
`best` — распределение с минимальным AIC и его медиана. Значения AIC всех распределений — в CSV.
"""),
        md("""
### 2.4. Полумарковская модель: распределение времени на дуге

В полумарковской модели процесс задаётся двумя частями: вероятностями $p_{ij}$ (2.1) и распределением
времени $F_{ij}(t)$ при условии, что произошёл именно переход $i \\to j$. Для дуг с не менее 30 переходами
подбираются Вейбулл, логнормальное и лог-логистическое распределения (только наблюдавшиеся переходы, без
цензуры), лучшее — по AIC. Условные распределения на дугах обычно с $\\rho \\approx 1$: падающий риск
в 2.3 получается в основном из-за смеси людей, которые вернутся, и тех, кто не вернётся.
"""),
        code('''arc_dist = ST.fit_arc_distributions(E)
arc_dist.to_csv(OUT23 / "arc_time_distributions.csv", index=False)
print("лучшее распределение на дугах:", arc_dist["best"].value_counts().to_dict())
print("доля дуг с формой Вейбулла rho < 1:", round(float((arc_dist["weibull_rho"] < 1).mean()), 3))
arc_dist.head(20)'''),
        md("""
### 2.5. Конкурирующие риски: вероятность перейти в $j$ за $t$ дней

Из состояния $i$ человек может попасть в любое из состояний $j$ — это конкурирующие риски; цензура —
конец наблюдения. Оценка Аалена–Йохансена даёт кумулятивную функцию инцидентности
$\\mathrm{CIF}_{ij}(t) = P(T \\le t,\\ K = j \\mid i)$ — вероятность, что следующим будет заключение типа $j$ и
случится не позже $t$. В отличие от $p_{ij}$, она учитывает время и цензуру; сумма по $j$ — вероятность
любого нового заключения за $t$ (линия «любое» $= 1 - S(t)$ из 2.3). Таблица — для состояний с не менее
200 заключениями на горизонтах 90, 180, 365, 730 дней.
"""),
        code('''cif, cif_curves = ST.cumulative_incidence(E)
cif.to_csv(OUT23 / "cif_competing_risks.csv", index=False)
cif_states = list(cif_curves)[:6]
ST.plot_cif(cif_curves, FIG23 / "cif_top_states.png", cif_states,
            "Кумулятивная инцидентность следующего заключения по типу (Аален–Йохансен)")
show(FIG23 / "cif_top_states.png")
# вероятность за 1 год: строки — откуда, столбцы — куда (самые частые направления)
cif_1y = cif.pivot(index="from", columns="to", values="CIF_365d")
cols = ["любое"] + [c for c in cif[cif["to"] != "любое"].groupby("to")["events"].sum().sort_values(ascending=False).index[:8]]
cif_1y.reindex(index=[s for s in ST.state_order(E) if s in cif_1y.index], columns=cols).round(3)'''),
        md("""
## Этап 3. Признаки объекта и события, модели выживаемости между состояниями

Признаки (в моделях берутся строки с известным временем на свободе и баллом COMPAS):

| признак | что это | тип |
|---|---|---|
| `age10` | возраст на дату оценки COMPAS, на 10 лет (HR — на каждые 10 лет) | объект |
| `male` | мужчина | объект |
| `race_afam`, `race_other` | афроамериканец; другая раса (Hispanic, Asian, Native American, Other); база — Caucasian | объект |
| `score` | исходный балл COMPAS `decile_score` (1–10, риск рецидива) | объект |
| `felony` | самое тяжёлое обвинение заключения — фелония | событие |
| `n_charges` | число обвинений в заключении (до 10) | событие |
| `log_dur` | $\\log(1 + \\text{дней в заключении})$ | событие |
| `prior_eps` | сколько заключений было у человека раньше (до 5) | событие |
| `old_case` | в заключении есть обвинение по делу из прежнего заключения (нет у данных Даши) | событие |

Признаки объекта не меняются во времени. Балл COMPAS и возраст взяты на дату оценки COMPAS, которая для
части заключений позже их начала, — это свойство данных ProPublica (одна оценка на человека).
Люди делятся на train 80% / test 20% (seed 42), C-index (согласованность прогноза риска и времени;
0.5 — случайно, 1 — идеально) считается на test.
"""),
        code('''feats = ST.features_of(E)
R = ST.model_frame(E, feats)
train23, test23 = ST.split_people(R)
print("признаки:", feats)
print(f"строк для моделей: {len(R):,} (train {len(train23):,}, test {len(test23):,}); "
      f"событий в train: {int(train23['event'].sum()):,}")
R[feats].describe().T[["mean", "std", "min", "max"]].round(3)'''),
        md("""
### 3.1. CoxPH: время до следующего заключения любого типа

Модель пропорциональных рисков Кокса, **стратифицированная по текущему состоянию**: у каждого состояния свой
базовый риск $h_{0,i}(t)$, а признаки действуют одинаково, $h(t \\mid i, x) = h_{0,i}(t)\\, e^{\\beta^\\top x}$.
$\\mathrm{HR} = e^\\beta$ — во сколько раз признак меняет риск нового заключения (HR > 1 — возвращаются
быстрее). Ниже — сравнение C-index на test для моделей «только состояние», «только признаки»,
«состояние + признаки» (состояние — индикаторами, база — первое по алфавиту; состояния реже 30 заключений
в train остаются в базе) и HR состояний из последней модели.
"""),
        code('''cph, cox_hr, cox_cindex, cox_state_hr = ST.cox_any(train23, test23, feats)
cox_hr.to_csv(OUT23 / "cox_any_hr.csv")
cox_cindex.to_csv(OUT23 / "cox_cindex_test.csv", header=["c_index"])
cox_state_hr.to_csv(OUT23 / "cox_state_hr.csv")
print("C-index на test:")
print(cox_cindex.to_string())
cox_hr'''),
        code('''# HR текущего состояния (модель «состояние + признаки»; база — первое по алфавиту состояние)
cox_state_hr.sort_values("HR")'''),
        md("""
### 3.2. Модели Кокса по дугам (причинно-специфичные риски)

Влияние признаков на **конкретный переход** $i \\to j$: для людей в состоянии $i$ событие — следующее
заключение типа $j$, переходы в другие типы и конец наблюдения — цензура. Такая модель
причинно-специфичного риска $h_{ij}(t \\mid x) = h_{0,ij}(t)\\, e^{\\beta_{ij}^\\top x}$ — стандартный способ
оценивать признаки при конкурирующих рисках. Модели строятся на всех людях для дуг с не менее
`MIN_ARC_EVENTS` переходами (лёгкая L2-регуляризация, penalizer = 0.05, иначе на редких дугах коэффициенты
расходятся). Признак, постоянный внутри состояния, выпадает.
"""),
        code('''arc_cox = ST.cox_arcs(R, feats, min_events=MIN_ARC_EVENTS)
arc_cox.to_csv(OUT23 / "cox_arcs_hr.csv", index=False)
arc_effects = ST.arc_effects_summary(arc_cox, feats)
arc_effects.to_csv(OUT23 / "cox_arcs_feature_summary.csv")
print(f"дуг с моделью: {arc_cox['arc'].nunique()}")
arc_effects'''),
        code('''ST.plot_arc_heatmap(arc_cox, feats, FIG23 / "cox_arcs_heatmap.png",
                    "HR признаков по дугам (цвет — log2 HR, * — p < 0.05)")
show(FIG23 / "cox_arcs_heatmap.png")'''),
        md("""
**Как читать.** Таблица: на скольких дугах признак значимо (p < 0.05) ускоряет (HR > 1) или замедляет
(HR < 1) переход, медиана и размах HR. Тепловая карта — самые частые дуги (в скобках число переходов):
красное — признак ускоряет именно этот переход, синее — замедляет. Признак, который ускоряет одни переходы
и замедляет другие, меняет не только «скорость возвращения», но и **тип** следующего заключения.
Все HR и доверительные интервалы — в `cox_arcs_hr.csv`.
"""),
        md("""
### 3.3. Дерево выживаемости

Дерево выживаемости (scikit-survival, разбиения по log-rank) для времени до следующего заключения: признаки
+ текущее состояние. Глубина 3, в листе не меньше 300 заключений, чтобы дерево читалось. Листья
отсортированы от самого быстрого возвращения: `km_median` — медиана Каплана–Мейера в листе, `S365` — доля
не вернувшихся за год. Ниже — кривые Каплана–Мейера листьев.
"""),
        code('''tree23, leaves23, tree_c = ST.survival_tree(train23, test23, feats)
leaves23.to_csv(OUT23 / "survival_tree_leaves.csv", index=False)
print(f"C-index дерева на test: {tree_c:.4f}")
ST.plot_tree_leaves(train23, tree23, feats, leaves23, FIG23 / "survival_tree_leaves.png",
                    "Дерево выживаемости: Каплан–Мейер по листьям (train)")
show(FIG23 / "survival_tree_leaves.png")
leaves23'''),
        md("""
### 3.4. Случайный лес выживаемости и важность признаков

Random Survival Forest (100 деревьев, лист от 50 заключений) — нелинейная модель для сравнения с Коксом.
Важность признака — насколько падает C-index на test, если перемешать этот признак (текущее состояние
перемешивается целиком). Флаг `RUN_RSF = False` пропускает этот шаг.
"""),
        code('''if RUN_RSF:
    rsf_c, rsf_imp, rsf_sec = ST.random_forest(train23, test23, feats)
    rsf_imp.to_csv(OUT23 / "rsf_permutation_importance.csv", header=["drop_in_c_index"])
    print(f"RSF: C-index на test {rsf_c:.4f} ({rsf_sec} с)")
    display(rsf_imp.to_frame("падение C-index"))
else:
    rsf_c, rsf_imp = float("nan"), pd.Series(dtype=float)

cindex_all = pd.concat([cox_cindex.rename(lambda s: "Кокс: " + s),
                        pd.Series({"дерево выживаемости": round(tree_c, 4), "случайный лес": round(rsf_c, 4)})])
cindex_all.to_csv(OUT23 / "cindex_all_models.csv", header=["c_index"])
ST.save_excel(OUT23 / "stage23_tables.xlsx", {
    "сводка": ST.episodes_summary(E).to_frame("значение"), "переходы_число": cnt, "переходы_p": P.round(4),
    "переходы_сводка": trans_summary, "время_дуги": arcs23, "время_состояния": state_dist,
    "распр_дуги": arc_dist, "CIF": cif, "cox_hr": cox_hr, "cox_состояния": cox_state_hr,
    "cox_дуги": arc_cox, "cox_дуги_сводка": arc_effects, "дерево_листья": leaves23,
    "rsf_важность": rsf_imp.to_frame("падение C-index"), "c_index": cindex_all.to_frame("c_index")})
cindex_all.to_frame("C-index (test)")'''),
    ]


def main():
    for case, cfg in CASES.items():
        d = ROOT / "stage1" / case
        nbp = d / cfg["nb"]
        if not nbp.exists():
            print("нет", nbp)
            continue
        shutil.copy(HERE / "stage23.py", d / cfg["mod_dir"] / "stage23.py")
        nb = nbformat.read(nbp, as_version=4)
        nb.cells = [c for c in nb.cells if not c.metadata.get("stage23")]
        nb.cells += cells(case, cfg)
        concl = HERE / "conclusions" / f"{case}.md"
        if concl.exists():
            nb.cells.append(md(concl.read_text(encoding="utf-8")))
        nbformat.write(nb, nbp)
        print("ok", nbp, len(nb.cells))


if __name__ == "__main__":
    main()
