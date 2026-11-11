"""Pre-fresh-anchor operational plateau criterion; validation data only."""
import math
import statistics

def plateau(trace,step):
    left=[r for r in trace if step-128<r["step"]<=step-64]
    right=[r for r in trace if step-64<r["step"]<=step]
    if len(left)<4 or len(right)<4:return False
    for key in ("maple","cedar_validation_logp"):
        if any(key not in r for r in left+right):return False
        a=[r[key] for r in left]; b=[r[key] for r in right]
        if not all(math.isfinite(v) for v in a+b):return False
        if abs(statistics.mean(a)-statistics.mean(b))>.01:return False
        if statistics.pstdev(a+b)>.02:return False
    return True
