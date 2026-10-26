"""Пример: загрузка таблицы логов в PM4Py (запускать из папки с event_log_*.csv)."""
import pandas as pd
import pm4py

df = pd.read_csv("event_log_oldcase.csv", parse_dates=["time:timestamp", "in_custody", "out_custody"])
df["case:concept:name"] = df["case:concept:name"].astype(str)
log = pm4py.format_dataframe(df, case_id="case:concept:name", activity_key="concept:name",
                             timestamp_key="time:timestamp")
print(pm4py.get_event_attribute_values(log, "concept:name"))
print("людей:", log["case:concept:name"].nunique(), "событий:", len(log))

dfg, sa, ea = pm4py.discover_dfg(log)
print("дуг DFG:", len(dfg))
net, im, fm = pm4py.discover_petri_net_heuristics(log, dependency_threshold=0.9)
print("HM 0.9: мест", len(net.places), "переходов", len(net.transitions))
# pm4py.view_dfg(dfg, sa, ea)          # в Jupyter: картинка графа
# pm4py.view_petri_net(net, im, fm)
# pm4py.write_xes(log, "event_log_oldcase.xes")   # если нужен XES (~50 МБ)
