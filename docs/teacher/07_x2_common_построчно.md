# `x2_common.py` построчно

Это общий модуль всех экспериментов «категории × 2» (exp1–exp4). Он делает три вещи:
1. `load` читает таблицу `compas_custody_charges.csv` и готовит заключения;
2. `build` собирает из них журнал событий для PM4Py по правилам выбранного варианта;
3. `split_people`, `sample_cases`, `evaluate` делят людей на train и test и считают fitness и precision.

Входная таблица: одна строка на одно обвинение. Колонки, которые здесь используются: `id` (человек), `custody_num` (номер заключения у человека), `in_custody` / `out_custody`, `custody_source` (jail или prison), `c_charge_desc` (текст обвинения), `c_charge_degree` (тяжесть, например `(F3)`), `charge_num`, `case_number`, `age`, `sex`, `race`.

---

## 1. Шапка и константы

```python
"""Общий код журнала событий ... """
```
Строка документации (docstring). Описывает модуль и два параметра экспериментов:
- `one_event`: оставить в заключении одну, самую тяжёлую категорию;
- `merge_old`: какие `OldCase: X` слить в `OldCase: Other`.

```python
import numpy as np
import pandas as pd
from crime_categories import categorize_fine as categorize, UNKNOWN
```
`crime_categories.py` лежит рядом. `categorize_fine` переводит текст обвинения в одну из 15 категорий (Drug, Property, Violent, Justice, ...). `UNKNOWN` — название категории для случаев, когда обвинение не найдено. `as categorize` — короткое имя.

```python
SEED = 42
```
Seed генератора случайных чисел. Используется везде, где есть случайность: при равной тяжести и при делении на train и test. С одним и тем же seed результат всегда одинаковый.

```python
OLDCASE, JUSTICE = "OldCase", "Justice"
OLD_PREFIX = "OldCase: "
```
Названия особых состояний. В варианте ×2 к категории старого дела добавляется префикс: `Drug` превращается в `OldCase: Drug`.

```python
VARIANTS = ["all_new", "oldcase", "no_oldcase", "x2"]
VNAME = {...}
```
Четыре варианта журнала и их русские названия для таблиц:
- `all_new`: все обвинения считаются новыми;
- `oldcase`: старые обвинения образуют одно общее состояние `OldCase`;
- `no_oldcase`: заключения только по старым делам выбрасываются;
- `x2`: каждая категория в двух видах, `X` и `OldCase: X`.

```python
PLACEHOLDER_OUT = pd.Timestamp("2020-01-01")
```
Если человек на момент выгрузки ещё сидел, дата выхода в таблице заполнена условной датой из будущего. Все даты от 2020-01-01 и позже считаются заглушкой «ещё в заключении».

```python
DEGREE_RANK = {"F1": 1, "F2": 2, "F3": 3, "F5": 4, "F6": 4, "F7": 4, "M1": 5, "M2": 6, "MO3": 7}
DEG = {1: "F", 2: "F", 3: "F", 4: "F", 5: "M", 6: "M", 7: "M"}
```
Шкала тяжести, твоё правило от 10.10: F1 > F2 > F3 > прочие F > M1 > M2 > MO3 > неизвестно. Меньшее число означает более тяжёлое обвинение. `DEG` переводит ранг в класс: F (felony, тяжкое) или M (misdemeanor, мелкое).

## 2. Маленькие вспомогательные функции

```python
def degree_rank(d):
    return DEGREE_RANK.get(str(d).strip("()"), 8)
```
Из `(F3)` делает `F3` (срезает скобки) и возвращает ранг 3. Если степени нет или она неизвестна, возвращает 8, самую низкую тяжесть.

```python
def uniq(s):
    return "; ".join(dict.fromkeys(x for x in s if isinstance(x, str) and x))
```
Склеивает значения колонки в строку через `; `, без повторов и пустых значений. `dict.fromkeys` убирает дубликаты и сохраняет порядок. Нужна для колонок вроде «все номера дел заключения».

```python
def base_type(a):
    return a[len(OLD_PREFIX):] if a.startswith(OLD_PREFIX) else a
```
`OldCase: Drug` → `Drug`. Срезает префикс, если он есть.

## 3. `load(data)`: чтение и подготовка заключений

```python
ch = pd.read_csv(data, parse_dates=["in_custody", "out_custody"], low_memory=False)
```
Читает таблицу обвинений (`ch` = charges). Даты сразу превращаются в тип datetime. `low_memory=False` нужен, чтобы pandas не угадывал типы колонок по кускам файла и не выдавал предупреждений.

```python
ch["type"] = ch["c_charge_desc"].map(categorize)
ch.loc[ch["charge_num"] == 0, "type"] = UNKNOWN
```
Каждому обвинению присваивается категория. `charge_num == 0` означает, что у заключения в compas.db не нашлось ни одного обвинения. Такие строки получают категорию Unknown.

```python
ch["rank"] = ch["c_charge_degree"].map(degree_rank)
ch["degree"] = ch["c_charge_degree"].astype(str).str.strip("()").replace("nan", "")
```
Ранг тяжести (1–8) и степень в виде текста без скобок (`F3`). Пустое значение превращается в пустую строку, а не в `nan`.

```python
ep = (ch.sort_values(["id", "custody_num"])
        .groupby(["id", "custody_num"], as_index=False)
        .agg(in_custody=("in_custody", "first"), ...))
```
Таблица заключений (`ep` = episodes): одна строка на пару (человек, номер заключения). Даты, источник, возраст, пол и раса одинаковы у всех обвинений заключения, поэтому берётся первое значение (`first`).

```python
ep["ep_raw"] = np.arange(len(ep))
ch = ch.merge(ep[["id", "custody_num", "ep_raw"]], on=["id", "custody_num"])
```
Каждому заключению присваивается сквозной номер `ep_raw` (0, 1, 2, …), и этот номер приписывается обратно к обвинениям.

### Склейка заключений
```python
ep = ep.sort_values(["id", "in_custody", "custody_num"]).reset_index(drop=True)
prev_out = ep.groupby("id")["out_custody"].shift()
prev_in = ep.groupby("id")["in_custody"].shift()
```
Заключения сортируются по человеку и дате входа. `groupby("id").shift()` даёт значение **предыдущего** заключения того же человека (у первого заключения получается NaN).

```python
cont = ((ep["custody_source"] == "prison") & (ep["in_custody"] == prev_out)) | (ep["in_custody"] == prev_in)
```
Заключение считается продолжением предыдущего, если:
- это prison и оно началось в день выхода из предыдущего (перевод из jail в prison, человек на свободу не выходил), **или**
- оно началось в тот же день, что и предыдущее (две записи об одном аресте).

```python
ep["merged_into_prev"] = cont
ep["ep_row"] = (~cont).cumsum() - 1
```
`ep_row` — номер **склеенного** эпизода. Приём с `cumsum`: каждое «не продолжение» (`~cont` = True) увеличивает счётчик на 1, а продолжение получает тот же номер, что и предыдущее. `- 1`, чтобы нумерация начиналась с 0. У первого заключения человека `cont` = False, поэтому эпизоды разных людей никогда не склеиваются.

```python
ch = ch.merge(ep[["ep_raw", "ep_row"]], on="ep_raw")
```
Номер склеенного эпизода переносится на обвинения.

### Старые дела
```python
first_ep = ch.dropna(subset=["case_number"]).groupby(["id", "case_number"])["ep_row"].min()
```
Для каждой пары (человек, номер дела) находим самый ранний эпизод, в котором это дело встретилось.

```python
fe = pd.MultiIndex.from_frame(ch[["id", "case_number"]]).map(first_ep.to_dict().get)
```
Для каждой строки обвинения смотрим, в каком эпизоде её дело появилось впервые. `.get` возвращает None, если номера дела нет.

```python
ch["old_case"] = ch["case_number"].notna().values & (pd.Series(fe, index=ch.index).fillna(np.inf) < ch["ep_row"])
```
Обвинение **старое**, если номер дела есть и дело впервые встретилось в **более раннем** эпизоде, чем текущий. Пустые значения заменяются на бесконечность, тогда сравнение даёт False и обвинение считается новым.

```python
return ch, ep
```
Возвращает обвинения (с категорией, рангом, номером эпизода, флагом старого дела) и заключения.

## 4. `build(variant, ch, ep, one_event=False, merge_old=())`: сборка журнала

```python
def is_justice(a):
    return a in (JUSTICE, OLD_PREFIX + JUSTICE)
```
Проверка, является ли категория Justice (новое или старое дело).

```python
ch = ch.copy()
```
Работаем с копией, чтобы не испортить исходную таблицу. Тогда `build` можно вызывать много раз с разными параметрами.

### Шаг 1. Конец наблюдения
```python
real = pd.concat([ep["in_custody"], ep.loc[ep["out_custody"] < PLACEHOLDER_OUT, "out_custody"]])
obs_end = real.max()
```
Дата конца наблюдения — самая поздняя **настоящая** дата в данных (вход в заключение или реальный выход, без заглушек). Ей цензурируются последние эпизоды.

```python
ep = ep.assign(ongoing=ep["out_custody"] >= PLACEHOLDER_OUT)
ep.loc[ep["ongoing"], "out_custody"] = obs_end
```
`ongoing` = человек ещё сидит. Для таких заключений дата-заглушка заменяется на `obs_end`.

```python
df = (ep.groupby("ep_row", as_index=False).agg(id=..., in_custody=("in_custody", "first"),
         out_custody=("out_custody", "max"), ongoing=("ongoing", "max"), ...,
         custodies=("custody_num", "size"), sources=("custody_source", lambda s: "+".join(s))))
```
Одна строка на склеенный эпизод: вход берётся по первому заключению, выход по самому позднему (`max`). `custodies` — сколько записей склеено, `sources` — их источники, например `jail+prison`.

```python
epi = ch.groupby("ep_row").agg(n_charges_all=("type", "size"), n_old_charges=("old_case", "sum"),
                               case_numbers_all=("case_number", uniq))
epi["old_case_numbers"] = ch[ch["old_case"]].groupby("ep_row")["case_number"].agg(uniq)
epi["old_case_numbers"] = epi["old_case_numbers"].fillna("")
df = df.merge(epi, left_on="ep_row", right_index=True)
```
Сводка по обвинениям эпизода: сколько всего обвинений, сколько из них старых, все номера дел и номера старых дел. Сводка приклеивается к эпизодам.

```python
df["has_old_case"] = df["n_old_charges"] > 0
df["only_old_case"] = df["n_old_charges"] == df["n_charges_all"]
```
Флаги: в эпизоде есть хотя бы одно старое дело; в эпизоде только старые дела.

### Шаг 2. Как назвать старые обвинения
```python
if variant in ("oldcase", "no_oldcase"):
    ch.loc[ch["old_case"], "type"] = OLDCASE
elif variant == "x2":
    ch.loc[ch["old_case"], "type"] = OLD_PREFIX + ch.loc[ch["old_case"], "type"]
```
В вариантах oldcase и no_oldcase все старые обвинения получают категорию `OldCase`. В варианте ×2 категория сохраняется, к ней только добавляется префикс: `OldCase: Drug`.

```python
if merge_old:
    ch.loc[ch["type"].isin(list(merge_old)), "type"] = OLD_PREFIX + "Other"
```
Эксперименты exp2 и exp3: перечисленные редкие `OldCase: X` превращаются в `OldCase: Other`. Список приходит из ноутбука: это состояния, у которых меньше `MIN_EVENTS = 300` событий.

### Шаг 3. Набор категорий эпизода
```python
t = ch.groupby(["ep_row", "type"], as_index=False).agg(
    rank=("rank", "min"), n_charges=("type", "size"), case_numbers=("case_number", uniq),
    charges=("c_charge_desc", uniq), degrees=("degree", uniq))
```
`t` — одна строка на пару (эпизод, категория). Тяжесть категории в эпизоде равна тяжести её **самого тяжёлого** обвинения (`min`, потому что 1 = самое тяжёлое). Сохраняется число обвинений, номера дел, тексты обвинений и степени.

```python
t["deg"] = t["rank"].map(DEG).fillna("NA")
```
Класс тяжести: F, M или NA (неизвестно).

```python
unk = t["type"].isin([UNKNOWN, OLD_PREFIX + UNKNOWN])
known = t[~unk].groupby("ep_row").size()
t = t[~(unk & t["ep_row"].isin(known.index))]
```
Unknown выбрасывается, если в эпизоде есть хоть одна известная категория. Если эпизод целиком неизвестный, Unknown остаётся единственным состоянием.

```python
if variant in ("oldcase", "no_oldcase"):
    new = t[t["type"] != OLDCASE].groupby("ep_row").size()
    t = t[~((t["type"] == OLDCASE) & t["ep_row"].isin(new.index))]
```
Твоё правило от 08.10 для вариантов OldCase: эпизод получает состояние `OldCase`, только если **все** обвинения в нём старые. Если есть новые, старые просто выбрасываются.

```python
# ВЫБОР ДЛЯ СМЕШАННЫХ (x2): ...
```
В варианте ×2 по умолчанию эпизод «старое + новое» сохраняет обе категории (`Drug` и `OldCase: Property`). В закомментированных строках лежит альтернатива: учитывать только новые.

```python
df["types"] = df["ep_row"].map(t.groupby("ep_row")["type"].apply(sorted))
```
Каждому эпизоду ставится в соответствие отсортированный список его категорий, например `['Drug', 'Traffic']`.

### Эксперимент «одно событие на заключение»
```python
if one_event:
    rng1 = np.random.default_rng(SEED)
    t = t.assign(r=rng1.random(len(t))).sort_values(["ep_row", "rank", "r"])
```
Каждой строке `t` присваивается случайное число `r`. Сортировка идёт по эпизоду, затем по тяжести (самая тяжёлая первая), затем по `r`, чтобы при равной тяжести порядок был случайным, но воспроизводимым.

```python
    nonj = t.assign(nj=~t["type"].map(is_justice)).groupby("ep_row")["nj"].transform("any")
    t = t[~(t["type"].map(is_justice) & nonj)]
```
`nonj` = в эпизоде есть хотя бы одна категория, кроме Justice. Тогда Justice из такого эпизода удаляется. Justice остаётся, только если он в эпизоде один.

```python
    t = t.groupby("ep_row", as_index=False).head(1).drop(columns="r")
    df["types"] = df["ep_row"].map(t.groupby("ep_row")["type"].apply(sorted))
```
Из каждого эпизода остаётся первая строка, то есть самая тяжёлая категория. Список категорий эпизода пересчитывается и состоит теперь из одного элемента.

### Шаг 4. Вариант без OldCase
```python
if variant == "no_oldcase":
    df = df[df["types"].apply(lambda x: x != [OLDCASE])].reset_index(drop=True)
    t = t[t["type"] != OLDCASE]
```
Эпизоды, у которых единственное состояние `OldCase`, удаляются целиком.

```python
best = t.groupby("ep_row")["rank"].min()
df["charge_class"] = df["ep_row"].map(best).map(lambda r: "Felony" if r <= 4 else "Misdemeanor" if r <= 7 else "Other/NA")
df["dur_days"] = (df["out_custody"] - df["in_custody"]).dt.days.clip(lower=0)
```
Класс эпизода по самому тяжёлому обвинению (Felony, Misdemeanor или Other/NA) и длительность заключения в днях. `clip(lower=0)` нужен на случай ошибочных дат, когда выход раньше входа.

### Шаг 5. Justice отдельным шагом
```python
df["step"] = 0
split = df["types"].apply(lambda x: any(map(is_justice, x)) and not all(map(is_justice, x)))
```
`split` = в эпизоде есть Justice и есть что-то ещё.

```python
jr = df[split].copy()
jr["types"] = jr["types"].apply(lambda x: [y for y in x if is_justice(y)])
jr["step"] = 1
df.loc[split, "types"] = df.loc[split, "types"].apply(lambda x: [y for y in x if not is_justice(y)])
df = pd.concat([df, jr], ignore_index=True)
```
Такие эпизоды делятся на два шага. Шаг 0 содержит всё, кроме Justice, а шаг 1 (копия эпизода) — только Justice. Твоё правило от 09.10: Justice (нарушение условий, неявка в суд) идёт после основного обвинения. При `one_event=True` деление не срабатывает, потому что Justice уже удалён.

```python
df["n_types"] = df["types"].str.len()
df["state"] = df["types"].str.join("+")
```
Число категорий в шаге и состояние шага строкой, например `Drug+Traffic`.

### Шаг 6. Порядок шагов, интервалы, цензурирование
```python
df = df.sort_values(["id", "in_custody", "ep_row", "step"]).reset_index(drop=True)
g = df.groupby("id")
df["step_idx"] = g.cumcount() + 1
df["n_steps"] = g["id"].transform("size")
```
Шаги сортируются по человеку и времени. `step_idx` — номер шага у человека (1, 2, …), `n_steps` — сколько шагов у человека всего.

```python
df["gap_prev_days"] = (df["in_custody"] - g["out_custody"].shift()).dt.days
df["gap_next_days"] = (g["in_custody"].shift(-1) - df["out_custody"]).dt.days
```
Время на свободе в днях: от выхода из предыдущего заключения до входа в текущее и от выхода из текущего до входа в следующее. Это и есть время перехода (твоё правило от 10.10: out → in, а не in → in).

```python
df.loc[g["ep_row"].shift(-1) == df["ep_row"], "gap_next_days"] = 0
df.loc[df["step"] == 1, "gap_prev_days"] = 0
```
Между шагом 0 и шагом 1 (Justice) одного эпизода человек на свободу не выходил, поэтому интервал равен 0.

```python
df["prev_state"] = g["state"].shift(1).fillna("")
df["next_state"] = g["state"].shift(-1).fillna("")
```
Предыдущее и следующее состояние человека. Из этих колонок потом строятся матрица переходов и модели Cox по дугам.

```python
df["censored"] = df["step_idx"] == df["n_steps"]
df["followup_days"] = np.where(df["censored"], (obs_end - df["out_custody"]).dt.days, np.nan)
df["episode_id"] = df["ep_row"]
```
Последний шаг человека **цензурирован**: следующего заключения мы не видим, но это не значит, что его не будет (решение после встречи с Васильевым: терминального END нет). `followup_days` — сколько дней мы наблюдали человека на свободе после последнего выхода. Для остальных шагов значение пустое.

### Шаг 7. События журнала
```python
ev = df.explode("types").rename(columns={"types": "type"})
```
`explode` разворачивает список: шаг `['Drug', 'Traffic']` становится двумя строками-событиями. Событие = (шаг, категория).

```python
ev = ev.merge(t[["ep_row", "type", "rank", "deg", ...]], on=["ep_row", "type"], how="left")
ev["deg"] = ev["deg"].fillna("NA")
```
К событию приклеиваются данные категории из `t`: тяжесть, класс, номера дел, тексты обвинений.

```python
rng = np.random.default_rng(SEED)
ev["r"] = rng.random(len(ev))
ev = ev.sort_values(["id", "in_custody", "ep_row", "step", "rank", "r"]).reset_index(drop=True)
ev["order_in_episode"] = ev.groupby("ep_row").cumcount()
```
Порядок событий внутри эпизода: сначала по шагу, затем по тяжести (F1 первым), при равной тяжести случайно. `order_in_episode` = 0, 1, 2, … внутри эпизода.

```python
ev["time:timestamp"] = (ev["in_custody"] + pd.to_timedelta(ev["order_in_episode"], unit="s")).dt.tz_localize("UTC")
```
У всех событий эпизода одна дата входа. Чтобы PM4Py сохранил нужный порядок, к времени добавляется 0, 1, 2… **секунды**. Часовой пояс UTC нужен, потому что PM4Py ожидает время с часовым поясом.

```python
ev["case:concept:name"] = ev["id"].astype(str)
ev["concept:name"] = ev["type"]
```
Две обязательные колонки PM4Py: кейс (человек, обязательно строкой, иначе PM4Py выдаёт ошибку) и активность (категория).

```python
ev["base_type"] = ev["type"].map(base_type)
ev["is_old"] = ev["type"].str.startswith(OLD_PREFIX) | (ev["type"] == OLDCASE)
```
Категория без префикса (`Drug`) и флаг «событие из старого дела».

```python
ev = ev.rename(columns={"deg": "type_degree_class", "rank": "type_rank", ...})
ev_cols = [...]
st_cols = [...]
```
Колонки переименовываются в понятные названия, и задаются списки колонок двух выходных таблиц: событий (`ev_cols`) и шагов (`st_cols`).

```python
info = dict(variant=..., people=..., episodes=..., justice_steps=..., steps=..., events=...,
            transitions=..., people_repeat=..., multi_type_steps=..., activities=..., obs_end=...)
return ev[ev_cols], df[st_cols], info
```
Сводка журнала: число людей, эпизодов, шагов Justice, шагов, событий, переходов (нецензурированных шагов), людей с повторными заключениями, шагов с несколькими категориями, число состояний и дата конца наблюдения.
Функция возвращает журнал событий, таблицу шагов и сводку.

## 5. Деление и оценка качества

```python
def split_people(log, seed=SEED, test_share=0.2):
    rng = np.random.default_rng(seed)
    ids = log["case:concept:name"].unique()
    test_ids = set(rng.choice(ids, size=int(test_share * len(ids)), replace=False))
    return log[~log["case:concept:name"].isin(test_ids)], log[log["case:concept:name"].isin(test_ids)]
```
Деление на train и test **по людям**: 20% людей случайно (без повторов, `replace=False`) уходят в test. Возвращает две части журнала. Подробнее в [[03_train_test]].

```python
def sample_cases(log, n):
    keep = sorted(set(log["case:concept:name"]))[:n]
    return log[log["case:concept:name"].isin(keep)]
```
Первые `n` людей по номеру. Нужна, чтобы считать метрики не на всех людях, а на 2000. Номера сортируются как **строки** (`"10" < "9"`), но на результат это не влияет.

```python
def evaluate(models, train_s, test_s, align_cases=0):
```
Принимает словарь моделей `{имя: (сеть Петри, начальная разметка, конечная разметка)}` и две выборки.

```python
    for name, (net, im, fm) in models.items():
        for split, lg in (("train", train_s), ("test", test_s)):
            t0 = time.time()
            fit = pm4py.fitness_token_based_replay(lg, net, im, fm)
            prec = pm4py.precision_token_based_replay(lg, net, im, fm)
```
Для каждой модели и каждой выборки считаются fitness и precision методом token replay (см. [[01_этап1_графы_PM4Py]]). `t0` засекает время.

```python
            row = dict(model=name, split=split, cases=...,
                       fitness_tbr=round(fit["log_fitness"], 4),
                       pct_fit_traces=round(fit["percentage_of_fitting_traces"], 2),
                       precision_tbr=round(prec, 4),
                       f1=round(2 * fit["log_fitness"] * prec / (fit["log_fitness"] + prec), 4),
                       places=len(net.places), transitions=len(net.transitions),
                       silent=sum(t.label is None for t in net.transitions))
```
Строка таблицы качества: fitness, доля полностью подходящих трасс, precision, F1 (гармоническое среднее fitness и precision), размер сети (позиции, переходы, немые переходы). Немой переход — это переход без метки (`label is None`).

```python
            if align_cases and split == "test":
                try:
                    row["precision_align"] = round(pm4py.precision_alignments(sample_cases(lg, align_cases), net, im, fm), 4)
                except Exception as ex:
                    row["precision_align"] = float("nan")
                    print(name, "alignments:", type(ex).__name__)
```
Если задано `align_cases`, на test дополнительно считается precision по выравниваниям. `try/except` нужен потому, что выравнивания могут не уложиться в память. Тогда в таблицу пишется NaN, а ноутбук не падает (см. [[04_выравнивания]]). В ноутбуке exp3 это вызывается только для HM 0.9.

```python
            row["sec"] = round(time.time() - t0)
            rows.append(row)
            print(row)
    return pd.DataFrame(rows)
```
Сколько секунд заняла оценка. Строка печатается сразу (видно прогресс) и в конце все строки собираются в таблицу `model_quality.csv`.
