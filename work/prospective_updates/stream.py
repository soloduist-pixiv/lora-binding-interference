"""Buffered local Qwen3 loader for Windows hosts with unreliable mapped reads."""
import hashlib
import json
import struct
from types import SimpleNamespace

from accelerate import init_empty_weights
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer,GenerationConfig

import diagnostics as d


class BufferedEngine(d.p.Engine):
    def __init__(self,name):
        assert name=="Qwen3-0.6B"
        t=d.p.torch
        self.path=d.ROOT/"Models"/name
        self.tok=AutoTokenizer.from_pretrained(self.path,local_files_only=True)
        self.letters=[self.tok.encode(ch,add_special_tokens=False)[0] for ch in "ABCDE"]
        config=AutoConfig.from_pretrained(self.path,local_files_only=True)
        assert config.tie_word_embeddings
        with init_empty_weights():
            model=AutoModelForCausalLM.from_config(config,dtype=t.bfloat16,attn_implementation="sdpa")
        model.generation_config=GenerationConfig.from_pretrained(self.path,local_files_only=True)
        path=self.path/"model.safetensors"
        with path.open("rb",buffering=0) as f:
            header_len=struct.unpack("<Q",f.read(8))[0]
            header=json.loads(f.read(header_len))
            start=8+header_len
            weights={}
            shared="model.embed_tokens.weight"
            head="lm_head.weight"
            def digest(item):
                a,b=header[item]["data_offsets"]
                f.seek(start+a)
                h=hashlib.sha256()
                while b>a:
                    block=f.read(min(b-a,4*1024**2))
                    assert block
                    h.update(block)
                    a+=len(block)
                return h.digest()
            assert digest(head)==digest(shared)
            for key,info in header.items():
                if key in ("__metadata__",head):
                    continue
                assert info["dtype"]=="BF16"
                a,b=info["data_offsets"]
                f.seek(start+a)
                if key==shared:
                    target=t.empty((b-a)//2,dtype=t.float32)
                    pos=0
                    while pos<target.numel():
                        chunk=min(target.numel()-pos,4*1024**2)
                        buf=bytearray(chunk*2)
                        view=memoryview(buf)
                        filled=0
                        while filled<len(buf):
                            n=f.readinto(view[filled:])
                            assert n
                            filled+=n
                        target[pos:pos+chunk].copy_(t.frombuffer(buf,dtype=t.bfloat16))
                        pos+=chunk
                    weights[key]=target.reshape(info["shape"])
                    continue
                buf=bytearray(b-a)
                view=memoryview(buf)
                pos=0
                while pos<len(view):
                    n=f.readinto(view[pos:])
                    assert n
                    pos+=n
                local=t.frombuffer(buf,dtype=t.bfloat16).reshape(info["shape"])
                weights[key]=local.to("cuda")
                del local,view,buf
        weights[head]=weights[shared]
        model.load_state_dict(weights,strict=True,assign=True)
        model.tie_weights()
        assert model.lm_head.weight.data_ptr()==model.model.embed_tokens.weight.data_ptr()
        model.requires_grad_(False)
        model.eval()
        model.model.rotary_emb.to("cuda")
        self.model=SimpleNamespace(model=model.model,lm_head=model.lm_head)
        self.loaded=model
        self.adapters=[]
        # Reuse the LoRA implementation and exact layer target convention.
        from core import Adapter
        for idx,layer in list(enumerate(model.model.layers))[-4:]:
            for key in ("q_proj","v_proj"):
                adapter=Adapter(getattr(layer.self_attn,key))
                setattr(layer.self_attn,key,adapter)
                self.adapters.append((f"layers.{idx}.self_attn.{key}",adapter))
        self.ps=[p for _,adapter in self.adapters for p in (adapter.A,adapter.B)]
        self.cache={}
        print("Buffered safetensors load finished",flush=True)
