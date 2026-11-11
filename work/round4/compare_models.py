import sys,json,time,traceback,subprocess,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import core,train
import torch
from types import SimpleNamespace
from transformers import AutoTokenizer,AutoModelForCausalLM
ROOT=core.ROOT
OUT=ROOT/'outputs/model_comparison_20261107'
OUT.mkdir(exist_ok=True,parents=True)
class Engine(core.Engine):
 def __init__(self,name):
  self.path=ROOT/'Models'/name
  self.tok=AutoTokenizer.from_pretrained(self.path,local_files_only=True)
  self.letters=[self.tok.encode(c,add_special_tokens=False)[0] for c in core.LETTERS]
  assert all(len(self.tok.encode(c,add_special_tokens=False))==1 for c in core.LETTERS)
  cls=AutoModelForCausalLM
  if name.startswith('Qwen3.5'):
   from transformers import Qwen3_5ForConditionalGeneration
   cls=Qwen3_5ForConditionalGeneration
  loaded=cls.from_pretrained(self.path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':0},low_cpu_mem_usage=True)
  loaded.requires_grad_(False);loaded.eval()
  text=loaded.model.language_model if name.startswith('Qwen3.5') else loaded.model
  self.model=SimpleNamespace(model=text,lm_head=loaded.lm_head)
  text.embed_tokens.to(device='cpu',dtype=torch.float32);loaded.lm_head.to(device='cpu',dtype=torch.float32)
  text.rotary_emb.to('cuda');text.norm.to('cuda')
  for layer in text.layers:layer.to('cuda')
  targets=[(i,l) for i,l in enumerate(text.layers) if hasattr(l,'self_attn')][-4:]
  assert len(targets)==4
  self.adapters=[]
  for i,l in targets:
   for n in ['q_proj','v_proj']:
    a=core.Adapter(getattr(l.self_attn,n));setattr(l.self_attn,n,a);self.adapters.append((f'layers.{i}.self_attn.{n}',a))
  self.ps=[p for _,a in self.adapters for p in [a.A,a.B]];self.cache={}
  self.loaded=loaded
 def reset(self,seed,mode):
  torch.manual_seed(seed)
  for _,a in self.adapters:
   torch.nn.init.kaiming_uniform_(a.A,a=5**.5)
   with torch.no_grad():a.B.zero_()
   a.mode=mode;a.gate=1.;a.z_min=a.z_max=0.;a.clipped=0
  self.loaded.zero_grad(set_to_none=True)
def child(name):
 dest=OUT/name;dest.mkdir(exist_ok=True)
 core.OUT=dest;train.OUT=dest
 data=json.loads((ROOT/'outputs/round4_20261107/data/dataset.json').read_text(encoding='utf-8'))
 try:
  e=Engine(name);e.reset(17,'add')
  lengths=[len(e.encode(r['variants']['original']['text'])) for t in data.values() for rows in t.values() for r in rows]
  core.write(dest/'audit.json',{'max_original_tokens':max(lengths),'target_modules':[n for n,a in e.adapters],'parameters':sum(p.numel() for p in e.loaded.parameters()),'trainable':sum(p.numel() for p in e.ps)})
  assert max(lengths)<=1536,'token budget exceeded'
  core.write(dest/'baseline.json',train.stage_evaluate(e,data))
  for order in [['law','gsm'],['gsm','law']]:train.run(e,data,'add',17,order,steps=32)
  core.write(dest/'status.json',{'success':True})
 except Exception as ex:core.write(dest/'status.json',{'success':False,'error':str(ex),'traceback':traceback.format_exc()})
def summarize():
 rows=[]
 for p in OUT.glob('*/runs/*/result.json'):
  r=json.loads(p.read_text());old,new=r['order'];st=r['stages'];initial=st[0]['evaluation'];own=st[1]['evaluation'];final=st[2]['evaluation']
  rows.append({'model':p.parents[2].name,'order':' → '.join(r['order']),'old_initial':initial[old]['test']['original']['correct'],'old_own':own[old]['test']['original']['correct'],'old_final':final[old]['test']['original']['correct'],'new_final':final[new]['test']['original']['correct'],'robust_mean':sum(final[t]['test']['robust_accuracy'] for t in core.TASKS)/2,'seconds':r['seconds'],'peak_gb':r['peak_cuda_allocated_gb']})
 core.write(OUT/'summary.json',rows)
 report='# 新增小模型串行对比实验\n\n单种子17筛查；CaseHOLD/GSM8K-MC，沿用 Round4 固定样本与五种输入变体。每任务32次AdamW更新、batch2，最后四个注意力层q/v、rank4、scale2、BF16基座/FP32适配器；各模型自己的聊天模板。普通加法LoRA；训练题目相同但分词长度和可训练参数量可能不同，因此不是纯参数量因果对比。GSM8K-MC不是官方自由生成指标。所有运行严格串行，失败单列。\n\n|模型|顺序|旧任务初始|自身训练后|后续保留|新任务最终|最终严格稳健性均值|峰值GB|\n|---|---|---:|---:|---:|---:|---:|---:|\n'
 for r in rows:report+='|'+r['model']+'|'+r['order']+'|'+ '|'.join(f'{100*r[k]:.2f}%' for k in ['old_initial','old_own','old_final','new_final','robust_mean'])+f"|{r['peak_gb']:.2f}|\n"
 report+='\n## 加载/执行状态\n\n'
 for p in OUT.glob('*/status.json'):
  s=json.loads(p.read_text());report+=f"- {p.parent.name}: {'完成' if s['success'] else s.get('error','原生进程失败')}\n"
 report+='\n仅一个种子，不能据此断言模型优劣或普遍遗忘机制。旧任务保留下降只有在自身先获得可靠学习增益时才有清晰解释；请结合每组验证集、逐题CE与稳健性记录。\n'
 if rows:
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,ax=plt.subplots(figsize=(11,5));labels=[r['model']+'\n'+r['order'] for r in rows];x=list(range(len(rows)))
  ax.bar([i-.18 for i in x],[100*r['old_own'] for r in rows],.36,label='Old task after own training');ax.bar([i+.18 for i in x],[100*r['old_final'] for r in rows],.36,label='Old task after next task');ax.set_xticks(x,labels,rotation=35,ha='right');ax.set_ylabel('Restricted accuracy (%)');ax.legend();fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=220);fig.savefig(OUT/'comparison.svg');plt.close(fig)
  report+='\n![旧任务保留对比](comparison.png)\n'
 (OUT/'实验报告.md').write_text(report,encoding='utf-8')
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='--summary':summarize()
 elif len(sys.argv)>1:child(sys.argv[1])
 else:
  import ctypes
  ctypes.windll.kernel32.SetThreadExecutionState(0x80000003)
  try:
   for name in ['Qwen3-0.6B','MiniCPM5-1B','Qwen3.5-0.8B','MiniCPM5-2B','Qwen3.5-2B','Qwen3-1.7B','Qwen3-4B',
                'SmolLM2-135M-Instruct','SmolLM2-360M-Instruct','SmolLM2-1.7B-Instruct']:
    status=OUT/name/'status.json'
    if status.exists():continue
    with (OUT/(name+'.log')).open('w',encoding='utf-8') as log:
     p=subprocess.run([sys.executable,'-u',__file__,name],stdout=log,stderr=subprocess.STDOUT)
    if not status.exists():core.write(status,{'success':False,'error':f'process exit {p.returncode}; see log'})
    summarize()
  finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
  summarize()
