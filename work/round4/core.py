"""Round 4/4B: explicit, shared numerical path for all adapter modes."""
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/round4_20261107'
OUT.mkdir(parents=True, exist_ok=True)
TASKS = ['gsm', 'law']
MODES = ['add', 'hadamard', 'exp', 'mix05', 'mix1']
LETTERS = 'ABCDE'
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def hash_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


class Adapter(nn.Module):
    def __init__(self, base, rank=4, mode='add'):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = 2.
        self.mode = mode
        self.gate = 1.
        self.register_buffer('rms', base.weight.detach().float().square().mean().sqrt())
        self.z_min = 0.
        self.z_max = 0.
        self.clipped = 0

    def delta(self):
        z = self.scale * (self.B @ self.A)
        w = self.base.weight.float()
        if self.mode == 'add':
            return z
        if self.mode == 'hadamard':
            return w * z
        if self.mode == 'exp':
            # expm1 prevents cancellation at tiny Z; clamp is audited, never silent.
            with torch.no_grad():
                self.z_min = min(self.z_min, z.min().item())
                self.z_max = max(self.z_max, z.max().item())
                self.clipped += int(((z < -15) | (z > 15)).sum().item())
            return w * torch.expm1(z.clamp(-15, 15))
        lam = {'mix05': .5, 'mix1': 1., 'mix0': 0.}[self.mode]
        return ((1-lam) + lam*w/self.rms) * z

    def forward(self, x):
        # Same dense residual in ALL modes: isolates geometry, not kernel speed.
        y = self.base(x)
        if self.gate == 0:
            return y
        return y + (self.gate * F.linear(x.float(), self.delta())).to(y.dtype)


class Engine:
    def __init__(self, model_name='Qwen3-1.7B', target_layers=4):
        self.path = ROOT / 'Models' / model_name
        self.tok = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
        self.letters = [self.tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]
        assert all(len(self.tok.encode(c, add_special_tokens=False)) == 1 for c in LETTERS)
        print('LOADING', model_name, flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.path, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa')
        self.model.requires_grad_(False)
        # Move layers individually: tied embedding/head stay CPU, avoiding transient OOM.
        self.model.model.embed_tokens.to(dtype=torch.float32, device='cpu')
        self.model.lm_head.to(dtype=torch.float32, device='cpu')
        self.model.model.rotary_emb.to('cuda')
        self.model.model.norm.to('cuda')
        for layer in self.model.model.layers:
            layer.to('cuda')
        self.adapters = []
        n = len(self.model.model.layers)
        for idx in range(n-target_layers, n):
            att = self.model.model.layers[idx].self_attn
            for name in ['q_proj', 'v_proj']:
                a = Adapter(getattr(att, name))
                setattr(att, name, a)
                self.adapters.append((f'layers.{idx}.self_attn.{name}', a))
        self.model.eval()
        self.ps = [p for _, a in self.adapters for p in [a.A, a.B]]
        self.cache = {}
        print('LOADED', sum(p.numel() for p in self.ps), 'adapter params; CUDA GB', torch.cuda.memory_allocated()/1e9, flush=True)

    def reset(self, seed, mode):
        torch.manual_seed(seed)
        for _, a in self.adapters:
            nn.init.kaiming_uniform_(a.A, a=math.sqrt(5))
            with torch.no_grad():
                a.B.zero_()
            a.mode = mode
            a.gate = 1.
            a.z_min = a.z_max = 0.
            a.clipped = 0
        self.model.zero_grad(set_to_none=True)

    def encode(self, text):
        if text not in self.cache:
            encoded = self.tok.apply_chat_template([{'role': 'user', 'content': text}], tokenize=True, add_generation_prompt=True, enable_thinking=False)
            self.cache[text] = encoded['input_ids'] if hasattr(encoded, 'keys') else encoded
            assert isinstance(self.cache[text], list) and isinstance(self.cache[text][0], int)
        return self.cache[text]

    def logits(self, row, variant='original', return_hidden=False):
        ids = self.encode(row['variants'][variant]['text'])
        ids = torch.tensor([ids], dtype=torch.long)
        emb = self.model.model.embed_tokens(ids).to('cuda', dtype=torch.bfloat16)
        h = self.model.model(inputs_embeds=emb, use_cache=False).last_hidden_state[:, -1]
        z = self.model.lm_head(h.to(device='cpu', dtype=torch.float32))[0]
        return (z, h) if return_hidden else z

    def loss(self, rows, variant='original'):
        ls = []
        for r in rows:
            z = self.logits(r, variant)
            y = self.letters[r['variants'][variant]['answer']]
            ls.append(F.cross_entropy(z[None], torch.tensor([y])))
        return torch.stack(ls).mean()

    def gradient(self, rows, variant='original'):
        # Autograd per sample avoids retaining a batch's full attention graphs.
        total = [torch.zeros_like(p) for p in self.ps]
        val = 0.
        for r in rows:
            loss = self.loss([r], variant)
            gs = torch.autograd.grad(loss, self.ps)
            val += loss.item()/len(rows)
            for g, gg in zip(total, gs):
                g.add_(gg.detach(), alpha=1/len(rows))
        return val, total

    def step(self, rows, opt):
        opt.zero_grad(set_to_none=True)
        loss, gs = self.gradient(rows)
        assert math.isfinite(loss) and all(torch.isfinite(g).all() for g in gs)
        for p, g in zip(self.ps, gs):
            p.grad = g
        norm = math.sqrt(dot(gs, gs))
        opt.step()
        return loss, norm

    @torch.no_grad()
    def evaluate(self, rows, variants=('original',), hidden=False):
        res = {}
        for variant in variants:
            vals = []
            for row in rows:
                if variant not in row['variants']:
                    continue
                z, h = self.logits(row, variant, True)
                n = len(row['variants'][variant]['options'])
                answer = row['variants'][variant]['answer']
                # Float64 score reductions avoid cancellation of large common logit offsets.
                z = z.double()
                q = z[self.letters[:n]]
                cond = F.cross_entropy(q[None], torch.tensor([answer])).item()
                full = F.cross_entropy(z[None], torch.tensor([self.letters[answer]])).item()
                logmass = (torch.logsumexp(q, 0)-torch.logsumexp(z, 0)).item()
                incorrect = q.clone()
                incorrect[answer] = -float('inf')
                pred = int(q.argmax())
                item = {'id': row['id'], 'answer': answer, 'prediction': pred,
                        'semantic_prediction': row['variants'][variant]['options'][pred],
                        'correct': int(pred == answer), 'full_ce': full, 'conditional_ce': cond,
                        'log_answer_mass': logmass, 'answer_mass': math.exp(logmass),
                        'margin': float(q[answer]-incorrect.max()), 'full_vocab_prediction': int(z.argmax()),
                        'format_letter': int(int(z.argmax()) in self.letters[:n]), 'length': len(self.encode(row['variants'][variant]['text']))}
                if hidden:
                    item['hidden'] = h[0].float().cpu().tolist()
                assert abs(full-(cond-logmass)) < 2e-5, (full, cond, logmass)
                vals.append(item)
            res[variant] = {'n': len(vals), 'examples': vals}
            for metric in ['correct', 'full_ce', 'conditional_ce', 'log_answer_mass', 'answer_mass', 'margin', 'format_letter']:
                res[variant][metric] = float(np.mean([x[metric] for x in vals])) if vals else 0.
        equivalent = [v for v in ['original', 'paraphrase', 'reorder', 'permutation'] if v in res]
        if len(equivalent) > 1:
            res['robust_accuracy'] = float(np.mean([all(res[v]['examples'][i]['correct'] for v in equivalent) for i in range(len(rows))]))
            res['semantic_consistency'] = float(np.mean([len({res[v]['examples'][i]['semantic_prediction'] for v in equivalent}) == 1 for i in range(len(rows))]))
            res['variant_mean_accuracy'] = float(np.mean([res[v]['correct'] for v in equivalent if v != 'original']))
        return res

    def snapshot(self):
        return [p.detach().clone() for p in self.ps]

    @torch.no_grad()
    def restore(self, state, delta=None, scale=0):
        for i, p in enumerate(self.ps):
            p.copy_(state[i] if delta is None else state[i]+scale*delta[i])

    @torch.no_grad()
    def weights(self):
        return [a.delta().detach().clone() for _, a in self.adapters]

    @torch.no_grad()
    def weight_stats(self):
        vals = []
        for name, a in self.adapters:
            d = a.delta()
            vals.append({'name': name, 'norm': float(d.norm()), 'relative_norm': float(d.norm()/a.base.weight.float().norm()), 'z_min': a.z_min, 'z_max': a.z_max, 'clipped': a.clipped})
        return vals

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'parameters': [p.detach().cpu() for p in self.ps], 'modes': [a.mode for _, a in self.adapters], 'names': [n for n, _ in self.adapters], 'rank': 4, 'scale': 2.}, path)


def dot(xs, ys):
    return sum((x.double()*y.double()).sum().item() for x, y in zip(xs, ys))


def cosine(xs, ys):
    return dot(xs, ys)/max(1e-30, math.sqrt(dot(xs, xs)*dot(ys, ys)))


def optimizer(ps, lr):
    return torch.optim.AdamW(ps, lr=lr, weight_decay=0., foreach=False)


def calibration(engine, rows, seed, mode, base_lr=.0003, matched=True):
    result = {'mode': mode, 'base_lr': base_lr, 'matched': matched}
    norms = {}
    for m in ['add', mode] if mode != 'add' else ['add']:
        engine.reset(seed, m)
        st = engine.snapshot()
        opt = optimizer(engine.ps, base_lr)
        engine.step(rows, opt)
        norms[m] = math.sqrt(sum(w.square().sum().item() for w in engine.weights()))
        engine.restore(st)
    lr = base_lr * (norms['add']/max(1e-30, norms[mode])) if matched else base_lr
    # Two corrective iterations also handle exp's non-linear effective weight path.
    for _ in range(2):
        engine.reset(seed, mode)
        opt = optimizer(engine.ps, lr)
        engine.step(rows, opt)
        got = math.sqrt(sum(w.square().sum().item() for w in engine.weights()))
        if matched and abs(got/norms['add']-1) > .002:
            lr *= norms['add']/got
        else:
            break
    result.update(raw_norms=norms, lr=lr, matched_norm=got, target_norm=norms['add'], ratio=got/norms['add'])
    engine.reset(seed, mode)
    return result
