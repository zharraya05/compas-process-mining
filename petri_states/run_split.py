"""python run_split.py <2013|all> — лучшие майнеры, train/test, alignments на всём test (лимит 120 с), с продолжением."""
import sys, json, time
sys.path[:0] = ['/home/claude/petri']
import pandas as pd
import petri_states as ps, petri_pipeline as pp
from pathlib import Path
exp = sys.argv[1]
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 120
pkl = {'2013': 'states_2013-01-01.pkl', 'all': 'states_all.pkl'}[exp]
OUT = Path('/home/claude/nbtest/jupyter lab/петри/out_states') / exp
RES = OUT / 'split_eval.jsonl'
CK = OUT / ('align_ckpt' if LIMIT == 120 else f'align_ckpt_{LIMIT}s')
CK.mkdir(parents=True, exist_ok=True)
done = {json.loads(l)['model'] for l in RES.read_text().splitlines() if l.strip()} if RES.exists() else set()
log = pd.read_pickle('/home/claude/' + pkl)
train, test = pp.split_people(log, 0.2, 42)
tr, te = pp.to_pm4py(train), pp.to_pm4py(test)
del log
order = ['flower', 'DFG_p0.05', 'HM_dep0.95', 'SplitMiner', 'IM_noise0.2']
models = ps.discover_named(tr, [m for m in order if m != 'flower'])
for m in order:
    if m in done:
        continue
    t0 = time.time()
    print(time.strftime('%X'), exp, m, 'старт', flush=True)
    r = ps.evaluate_split_one(m, models[m], tr, te, max_sec_trace=LIMIT, ckpt_dir=CK)
    if m == 'flower' or r.get('align_complete'):
        r.update(experiment=exp, align_max_sec_trace=LIMIT, finished=time.strftime('%Y-%m-%d %H:%M'))
        r = {k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()}
        with RES.open('a') as f:
            f.write(json.dumps(r, default=str) + '\n')
        print(time.strftime('%X'), exp, m, 'готово', {k: r.get(k) for k in ['fitness_tbr_test', 'fitness_align_test', 'align_timeout', 'precision_etc_test']}, flush=True)
print('EXP_DONE', exp, flush=True)
