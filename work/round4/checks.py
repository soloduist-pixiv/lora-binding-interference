"""Numerical checks which must pass before any model training."""
import io
import torch
import torch.nn.functional as F
from core import Adapter, OUT, write


def check():
    torch.manual_seed(17)
    x = torch.randn(3, 7)
    base = torch.nn.Linear(7, 5, bias=False)
    base.weight.data[0, 0] = 0
    a = Adapter(base, rank=2)
    result = []
    for mode in ['add','hadamard','exp','mix0','mix05','mix1']:
        a.mode = mode
        with torch.no_grad():
            a.B.zero_()
        zero_error = float((a(x)-base(x)).abs().max())
        assert zero_error == 0
        a(x).square().mean().backward()
        assert a.B.grad is not None and a.B.grad.norm() > 0
        assert a.A.grad is not None and a.A.grad.norm() == 0  # B=0 => dL/dA=0 is expected.
        assert base.weight.grad is None and not base.weight.requires_grad
        a.zero_grad(set_to_none=True)
        with torch.no_grad():
            a.B.normal_(0, .02)
        z = 2*a.B@a.A
        w = base.weight.float()
        ref = {'add': z, 'hadamard': w*z, 'exp': w*torch.expm1(z),
               'mix0': z, 'mix05': (.5+.5*w/a.rms)*z, 'mix1': w/a.rms*z}[mode]
        assert torch.allclose(a.delta(), ref, atol=1e-7)
        dense_error = float((a(x)-F.linear(x, w+ref)).abs().max().detach())
        assert dense_error < 1e-6
        loss = a(x).square().mean()
        loss.backward()
        assert a.A.grad.norm() > 0 and a.B.grad.norm() > 0
        # Central finite difference for one B coordinate, away from clamps.
        analytic = float(a.B.grad[0, 0])
        old = float(a.B[0, 0].detach())
        with torch.no_grad():
            a.B[0, 0] = old+.001
        plus = a(x).square().mean().item()
        with torch.no_grad():
            a.B[0, 0] = old-.001
        minus = a(x).square().mean().item()
        with torch.no_grad():
            a.B[0, 0] = old
        numerical = (plus-minus)/.002
        assert abs(analytic-numerical) < 2e-4, (mode, analytic, numerical)
        before = a(x).detach().clone()
        buf = io.BytesIO()
        torch.save(a.state_dict(), buf)
        saved = {k: v.clone() for k, v in a.state_dict().items()}
        with torch.no_grad():
            a.B.add_(.1)
        a.load_state_dict(saved)
        assert torch.equal(before, a(x))
        buf.seek(0)
        a.load_state_dict(torch.load(buf, weights_only=True))
        assert torch.equal(before, a(x))
        if mode in ['hadamard','exp','mix1']:
            assert a.delta()[0, 0] == 0
        result.append({'mode':mode,'zero_output_error':zero_error,'dense_equivalence_error':dense_error,
                       'analytic_gradient':analytic,'finite_difference_gradient':numerical,'serialization_exact':True,'restore_exact':True})
        a.zero_grad(set_to_none=True)
    write(OUT/'checks/unit.json', {'passed':True,'checks':result,'note':'FP32 CPU single layer; real-model BF16 checks separately logged'})
    print('ALL UNIT CHECKS PASSED', flush=True)


if __name__ == '__main__':
    check()
