"""Досчёт вариантов трасс, не уложившихся в лимит alignments, с повышенным лимитом.
python retry_timeouts.py <2013|all> <старый_лимит> <новый_лимит> <модель> [<модель> ...]
Берёт контрольные точки align_ckpt[_<старый>s]/<модель>_align.jsonl, копирует готовые варианты
и пересчитывает тайм-ауты в align_ckpt_retry<новый>s/<модель>_align.jsonl (с продолжением с места).
Итог по модели дописывается в split_eval_retry<новый>s.jsonl."""
import sys, json, time
sys.path[:0] = ['/home/claude/petri']
import pandas as pd, pm4py
import petri_states as ps, petri_pipeline as pp
from pathlib import Path
from pm4py.algo.conformance.alignments.petri_net import algorithm as ali

exp, old, new, models = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
OUT = Path('/home/claude/nbtest/jupyter lab/петри/out_states') / exp
SRC = OUT / ('align_ckpt' if old == 120 else f'align_ckpt_{old}s')
DST = OUT / f'align_ckpt_retry{new}s'
DST.mkdir(exist_ok=True)
RES = OUT / f'split_eval_retry{new}s.jsonl'
done_models = {json.loads(l)['model'] for l in RES.read_text().splitlines() if l.strip()} if RES.exists() else set()
pkl = {'2013': 'states_2013-01-01.pkl', 'all': 'states_all.pkl'}[exp]
log = pd.read_pickle('/home/claude/' + pkl)
train, test = pp.split_people(log, 0.2, 42)
tr, te = pp.to_pm4py(train), pp.to_pm4py(test)
del log
fits = ps.discover_named(tr, models)
el = pm4py.convert_to_event_log(te)
variants = {}
for t in el:
    variants.setdefault(tuple(e["concept:name"] for e in t), t)
keys = list(variants)

for m in models:
    if m in done_models:
        print(m, 'уже досчитано', flush=True)
        continue
    net, im, fm = fits[m]
    src = {r['i']: r for r in map(json.loads, (SRC / f'{m}_align.jsonl').read_text().splitlines()) if True}
    assert len(src) == len(keys), (m, len(src), len(keys))
    dst_f = DST / f'{m}_align.jsonl'
    dst = {r['i']: r for r in map(json.loads, dst_f.read_text().splitlines())} if dst_f.exists() else {}
    todo = [i for i in sorted(src) if src[i]['timeout'] and i not in dst]
    print(time.strftime('%X'), exp, m, f'тайм-аутов было {sum(r["timeout"] for r in src.values())}, осталось досчитать {len(todo)}, лимит {new} с', flush=True)
    bwc = getattr(ali, "__get_best_worst_cost")(net, im, fm, ali.DEFAULT_VARIANT, {})
    P = {ali.Parameters.PARAM_MAX_ALIGN_TIME_TRACE: new, ali.Parameters.BEST_WORST_COST_INTERNAL: bwc}
    with dst_f.open('a') as f:
        for j, i in enumerate(todo, 1):
            t0 = time.time()
            res = ali.apply_trace(variants[keys[i]], net, im, fm, parameters=P)
            r = dict(i=i, n=src[i]['n'], len=src[i]['len'], sec=round(time.time() - t0, 2), retried=True)
            if res is None or 'fitness' not in res:
                r['timeout'] = True
            else:
                r.update(timeout=False, cost=res['cost'], bwc=res['bwc'], fitness=res['fitness'])
            dst[i] = r
            f.write(json.dumps(r) + '\n'); f.flush()
            if j % 10 == 0:
                print(f'  {j}/{len(todo)}, выровнено из них {sum(not dst[k]["timeout"] for k in todo[:j])}', flush=True)
    allr = {**{i: r for i, r in src.items() if not r['timeout']}, **dst}
    ok = [r for r in allr.values() if not r['timeout']]
    n_tr = sum(r['n'] for r in allr.values()); n_ok = sum(r['n'] for r in ok)
    out = dict(model=m, experiment=exp, align_max_sec_trace=new, retry_of=old,
               fitness_align_test=1 - sum(r['cost'] * r['n'] for r in ok) / sum(r['bwc'] * r['n'] for r in ok),
               pct_fit_traces_align_test=100 * sum(r['n'] for r in ok if r['fitness'] == 1.0) / n_ok,
               align_traces=n_tr, align_timeout=n_tr - n_ok,
               align_timeout_before=sum(r['n'] for r in src.values() if r['timeout']),
               finished=time.strftime('%Y-%m-%d %H:%M'))
    with RES.open('a') as f:
        f.write(json.dumps(out) + '\n')
    print(time.strftime('%X'), exp, m, 'готово', out, flush=True)
print('RETRY_DONE', exp, flush=True)
