import pandas as pd
J='jupyter_lab/stage1/'; V='jupyter_lab/stage1_variants/stage1/'
cases=[
('A. Данные Даши','9 типов, развилки (multi)','stage1/results_multi/tables/model_quality.csv','9','все типы заключения, порядок случайный'),
('A. Данные Даши','9 типов, самое тяжёлое (epcharge)','stage1/results_epcharge/tables/model_quality.csv','9','одно событие на заключение'),
('A. Данные Даши','Тяжесть по заключению','jupyter_lab/stage1/dasha_severity/out/model_quality.csv|по заключению','7','F1…M2, одно событие'),
('A. Данные Даши','Тяжесть, колонка Даши','jupyter_lab/stage1/dasha_severity/out/model_quality.csv|колонка Даши','7','100% петель, модель тривиальна'),
('B. Этап 0, 15 категорий','all_new (все новые)',V+'all_new/results/tables/model_quality.csv','15','развилки, порядок случайный'),
('B. Этап 0, 15 категорий','OldCase',V+'oldcase/results/tables/model_quality.csv','16','развилки, порядок случайный'),
('B. Этап 0, 15 категорий','без OldCase',V+'no_oldcase/results/tables/model_quality.csv','15','развилки, порядок случайный'),
('B. Этап 0, 15 категорий','glue7, все новые',V+'glue7/results_glue_all/tables/model_quality.csv','15','склейка заключений ≤ 7 дней'),
('B. Этап 0, 15 категорий','glue7, без OldCase',V+'glue7/results_glue_noold/tables/model_quality.csv','15','склейка ≤ 7 дней'),
('C. Категории × 2','База oldcase_x2',J+'oldcase_x2/out/model_quality_x2.csv','29','цепочки по тяжести'),
('C. Категории × 2','Эксп 1. одно событие',J+'x2_exp1_one_event/out/model_quality.csv','28','самая тяжёлая категория'),
('C. Категории × 2','Эксп 2. слить редкие',J+'x2_exp2_merge_rare/out/model_quality.csv','22','OldCase < 300 → OldCase: Other'),
('C. Категории × 2','Эксп 3. одно событие + слить',J+'x2_exp3_one_event_merge/out/model_quality.csv','20','эксп 1 + эксп 2'),
('C. Категории × 2','Эксп 4. настройки майнеров',J+'x2_exp4_miners/out/model_quality.csv','29','журнал как в базе'),
]
rows=[]
for g,n,p,st,note in cases:
    path,_,var=p.partition('|')
    q=pd.read_csv('/mnt/project-files/'+path)
    if var: q=q[q.variant==var]
    q=q[q.split=='test'].copy()
    q['model']=q.model.str.replace('HM_dep','HM_').str.replace('IM_noise','IM_')
    prec=q['precision_tbr'] if 'precision_tbr' in q else q['precision_etc']
    for m,f,pr in zip(q.model,q.fitness_tbr,prec):
        rows.append(dict(группа=g,кейс=n,состояний=st,журнал=note,модель=m,fitness=round(f,3),precision=round(pr,3),f1=round(2*f*pr/(f+pr),3)))
a=pd.DataFrame(rows)
a.to_csv('/tmp/claude-0/w/all_quality.csv',index=False)
def get(c,m):
    x=a[(a.кейс==c)&(a.модель==m)]
    return f"{x.fitness.iloc[0]:.3f} / {x.precision.iloc[0]:.3f}" if len(x) else "—"
lines=[]
for g,n,p,st,note in cases:
    sub=a[(a.кейс==n)&(~a.модель.str.startswith('DFG'))]
    sub2=sub[sub.fitness>=0.95]
    b=(sub2 if len(sub2) else sub).sort_values('precision',ascending=False).iloc[0]
    lines.append((g,n,st,note,get(n,'HM_0.9'),get(n,'IM_0.2'),f"{b.модель}: {b.fitness:.3f} / {b.precision:.3f}"))
s=pd.DataFrame(lines,columns=['группа','кейс','состояний','журнал','HM 0.9','IM 0.2','лучшая'])
s.to_csv('/tmp/claude-0/w/summary_quality.csv',index=False)
print(s.to_string()); print(sorted(a.модель.unique()))
