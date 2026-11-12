"""CPU lifecycle checks only; GPU numerical equivalence is a separate required gate."""
import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import json
import resource_guard
from resource_guard import ready

def fixture():
    events=[]
    class Storage:
        def __init__(self,name,dtype):self.name=name;self.dtype=dtype
        def float(self):events.append((self.name,"promote"));return Storage(self.name,"float32")
        def to(self,dtype):events.append((self.name,"restore"));return Storage(self.name,dtype)
    class Parameter:
        def __init__(self,name,device="cuda",grad=False):
            self.data=Storage(name,"bfloat16");self.device=SimpleNamespace(type=device);self.requires_grad=grad
        @property
        def dtype(self):return self.data.dtype
    parameters=[Parameter("base1"),Parameter("base2"),Parameter("cpu_embedding","cpu"),Parameter("adapter",grad=True)]
    class Base:
        def __init__(self):
            self.engine=SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(parameters=lambda:parameters)))
            self.precision="float32"
    source=ast.parse(Path(__file__).with_name("stable_score.py").read_text())
    node=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=="MemoryStableScorer")
    namespace=dict(CandidateScorer=Base,contextmanager=contextmanager,Path=Path,
        wait_for_scoring=lambda _:None,
        torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:events.append(("allocator","release")))))
    exec(compile(ast.Module(body=[node],type_ignores=[]),"actual_scoring_context","exec"),namespace)
    return namespace["MemoryStableScorer"](),parameters,events

class ResourcePolicy(unittest.TestCase):
    def test_user_offline_choice_blocks_before_even_querying_gpu_resources(self):
        path=Path(__file__).resolve().parents[2]/"outputs/binding_revision2_20261111/local_system/test_offline_mode.json"
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(dict(mode="offline_only")))
        try:
            with patch.object(resource_guard,"MODE_FILE",path),patch.object(resource_guard,"snapshot",side_effect=AssertionError("must not query GPU")):
                with self.assertRaisesRegex(RuntimeError,"offline-only"):
                    resource_guard.wait_for_model()
        finally:path.unlink()

    def test_observed_gpu_free_but_host_shortage_does_not_start_model(self):
        observed=dict(free_gpu_mib=7947,available_commit_mib=4005612/1024,available_physical_mib=1477884/1024)
        self.assertGreater(observed["free_gpu_mib"],6144)
        self.assertFalse(ready(observed))
        self.assertTrue(ready(dict(free_gpu_mib=7947,available_commit_mib=12000,available_physical_mib=4000)))

    def test_nested_calls_keep_one_promotion_and_restore_only_frozen_cuda_parameters(self):
        scorer,parameters,events=fixture()
        with scorer.evaluation_precision():
            with scorer.evaluation_precision():
                self.assertEqual([p.dtype for p in parameters],["float32","float32","bfloat16","bfloat16"])
            self.assertEqual(events,[("base1","promote"),("base2","promote")])
        self.assertEqual([p.dtype for p in parameters],["bfloat16"]*4)
        self.assertEqual(events,[("base1","promote"),("base2","promote"),("base1","restore"),("base2","restore"),("allocator","release")])
        self.assertEqual(scorer._precision_depth,0)

    def test_failure_does_not_attempt_more_gpu_allocations_during_unwind(self):
        scorer,_,events=fixture()
        with self.assertRaisesRegex(RuntimeError,"simulated allocation failure"):
            with scorer.evaluation_precision():
                with scorer.evaluation_precision():raise RuntimeError("simulated allocation failure")
        self.assertEqual(events,[("base1","promote"),("base2","promote")])
        self.assertEqual(scorer._precision_depth,0)

if __name__=="__main__":unittest.main()
