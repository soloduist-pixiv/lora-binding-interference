"""Same FP32 sequence scores with one promotion per outer evaluation block."""
from contextlib import contextmanager
import datetime
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"work/binding_review"))
from cached_score import CandidateScorer
import torch
from resource_guard import host_memory,wait_for_scoring

class MemoryStableScorer(CandidateScorer):
    allocation_policy="one FP32 promotion per evaluation block; unchanged batches and logits; release idle cache after restore"

    def __init__(self,*args,diagnostic_directory=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.diagnostic_directory=Path(diagnostic_directory) if diagnostic_directory else None
        if self.diagnostic_directory:self.diagnostic_directory.mkdir(parents=True,exist_ok=True)
        self._precision_depth=0
        self.phase="unspecified"

    @contextmanager
    def evaluation_precision(self):
        if self._precision_depth:
            yield
            return
        folder=self.diagnostic_directory
        wait_for_scoring(folder/"scoring_resource_wait.json" if folder else None)
        frozen=[(v,v.dtype) for v in self.engine.model.model.parameters()
                if v.device.type=="cuda" and not v.requires_grad]
        self._precision_depth=1
        try:
            if self.precision=="float32":
                for v,_ in frozen:v.data=v.data.float()
            yield
            for v,dtype in frozen:v.data=v.data.to(dtype=dtype)
            torch.cuda.empty_cache()
        except BaseException as error:
            # The child exits after an evaluation failure; do not allocate again in an OOM handler.
            if folder:
                try:
                    stats=torch.cuda.memory_stats()
                    value=dict(time=datetime.datetime.now().astimezone().isoformat(),phase=self.phase,
                        error_type=type(error).__name__,error=str(error),host=host_memory(),
                        allocator={k:int(stats.get(k,0)) for k in ("allocated_bytes.all.current",
                            "reserved_bytes.all.current","reserved_bytes.all.peak","inactive_split_bytes.all.current")})
                    (folder/"scoring_failure.json").write_text(json.dumps(value,indent=2))
                except Exception:pass
            raise
        finally:
            self._precision_depth=0
