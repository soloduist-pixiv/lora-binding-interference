"""Single-process, strictly serial training. Every completed run is resumable."""
import argparse
import json
import math
import random
import time
import torch
from core import ROOT, OUT, TASKS, MODES, Engine, calibration, optimizer, write
from checks import check


VARIANTS = ('original','paraphrase','reorder','permutation','format')


def stage_evaluate(e, data):
    return {t: {'test': e.evaluate(data[t]['test'], VARIANTS),
                'validation': e.evaluate(data[t]['validation']),
                'control': e.evaluate(data[t]['control'])} for t in TASKS}


def run(e, data, mode, seed, order, raw=False, steps=32):
    key = f'{mode}{"_raw" if raw else ""}_s{seed}_{"-".join(order)}'
    directory = OUT/'runs'/key
    if (directory/'result.json').exists():
        print('SKIP', key, flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    # Identical samples and random seed for every mode.
    rng = random.Random(seed)
    schedules = {t: [rng.sample(data[t]['train'], 2) for _ in range(steps)] for t in order}
    cal = calibration(e, schedules[order[0]][0], seed, mode, matched=not raw)
    write(directory/'calibration.json', cal)
    print('RUN', key, 'lr', cal['lr'], 'scale ratio', cal['ratio'], flush=True)
    base = json.loads((OUT/'baseline.json').read_text(encoding='utf-8'))
    stages = [{'name':'initial','evaluation':base}]
    logs = []
    for stage, task in enumerate(order):
        opt = optimizer(e.ps, cal['lr'])
        for step, rows in enumerate(schedules[task]):
            st = time.perf_counter()
            loss, grad = e.step(rows, opt)
            torch.cuda.synchronize()
            log = {'task':task,'step':step+1,'loss':loss,'gradient_norm':grad,'seconds':time.perf_counter()-st,
                   'ids':[r['id'] for r in rows], 'input_tokens':sum(len(e.encode(r['variants']['original']['text'])) for r in rows)}
            logs.append(log)
            if (step+1) % 8 == 0:
                print('TRAIN',key,task,step+1,'loss',round(loss,4),'sec',round(log['seconds'],3),flush=True)
        e.save(directory/f'checkpoint_{stage+1}_{task}.pt')
        torch.save(opt.state_dict(), directory/f'optimizer_{stage+1}_{task}.pt')
        print('EVALUATE',key,task,flush=True)
        ev = stage_evaluate(e, data)
        stages.append({'name':task,'evaluation':ev,'weight_stats':e.weight_stats(),
                       'train_evaluation':e.evaluate(data[task]['train'][:16])})
        write(directory/'partial.json', {'stages':stages,'training':logs})
        print('STAGE',key,task,{t:round(ev[t]['test']['original']['correct'],3) for t in TASKS},flush=True)
    result = {'key':key,'seed':seed,'mode':mode,'raw_lr':raw,'order':order,'steps_per_task':steps,
              'calibration':cal,'stages':stages,'training':logs,'seconds':time.perf_counter()-start,
              'peak_cuda_allocated_gb':torch.cuda.max_memory_allocated()/1e9,'peak_cuda_reserved_gb':torch.cuda.max_memory_reserved()/1e9,
              'trainable_parameters':sum(p.numel() for p in e.ps),'optimizer':'AdamW; new state at task boundary; no weight decay',
              'precision':'BF16 frozen transformer; FP32 dense adapter residual; FP32 CPU tied embedding and head; TF32 disabled; batch=1 accumulation=2'}
    assert all(x['clipped'] == 0 for s in stages[1:] for x in s['weight_stats']), 'exp clamp activated'
    write(directory/'result.json', result)
    print('DONE',key,round(result['seconds'],1),'peak GB',round(result['peak_cuda_allocated_gb'],3),flush=True)


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--pilot', action='store_true')
    a.add_argument('--all', action='store_true')
    a.add_argument('--steps', type=int, default=32)
    args = a.parse_args()
    assert args.pilot or args.all
    check()
    data = json.loads((OUT/'data/dataset.json').read_text(encoding='utf-8'))
    e = Engine()
    e.reset(17,'add')
    if not (OUT/'baseline.json').exists():
        # Repeat identical row and snapshot restore: exact deterministic score path.
        row = data['gsm']['control'][0]
        z1 = e.logits(row).detach()
        st = e.snapshot()
        with torch.no_grad():
            e.ps[1].add_(.01)
        e.restore(st)
        z2 = e.logits(row).detach()
        assert torch.equal(z1,z2)
        write(OUT/'checks/model.json', {'exact_restore_and_zero_repeat':True,'max_logit_error':float((z1-z2).abs().max()),'n_adapters':len(e.adapters),'trainable_parameters':sum(p.numel() for p in e.ps)})
        print('BASELINE EVALUATION',flush=True)
        write(OUT/'baseline.json',stage_evaluate(e,data))
    if args.pilot:
        run(e,data,'add',17,['law','gsm'],steps=args.steps)
    else:
        for seed in [17,29,43]:
            for order in [['law','gsm'],['gsm','law']]:
                for mode in MODES:
                    run(e,data,mode,seed,order,steps=args.steps)
        for mode in ['hadamard','exp']:
            run(e,data,mode,17,['law','gsm'],raw=True,steps=args.steps)
    write(OUT/'training_completed.json', {'all':args.all,'pilot':args.pilot,'time':time.strftime('%Y-%m-%d %H:%M:%S')})


if __name__ == '__main__':
    main()
