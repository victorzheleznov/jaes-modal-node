"""Tiny synthetic regressions; no checkpoint, dataset, or audio is needed."""

import copy
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from src.models.grad_net import BaseCGradNet, BaseMGradNet
from src.ode.adjoint import AutoDiffAdjoint, CheckpointAdjoint
from src.ode.methods.sav import SAV
from src.ode.problem import InitialValueProblem
from src.utils.activation import Exp, Tanh


class SeparateLaw(nn.Module):
    """Exercise the unchanged two-method protocol using the same learned law."""

    def __init__(self, law):
        super().__init__()
        self.law = law

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.law(x)

    @torch.jit.export
    def potential(self, x: torch.Tensor) -> torch.Tensor:
        return self.law.potential(x)


class PolynomialLaw(nn.Module):
    """An independent nonlinear law with no joint-evaluation method."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -x.pow(3)

    @torch.jit.export
    def potential(self, x: torch.Tensor) -> torch.Tensor:
        return 0.25 * x.pow(4).sum(-1)


def make_law(kind, bias=True, dtype=torch.float64):
    torch.manual_seed(17)
    if kind == "cgrad":
        law = BaseCGradNet(3, 7, Tanh(), bias=bias, log_scale_std=0.1)
    else:
        law = BaseMGradNet(
            3, [7, 4], Tanh(), transform=Exp() if kind == "mgrad_transform" else None, bias=bias, log_coefs_std=0.1
        )
    return law.to(dtype=dtype)


def problem(dtype=torch.float64):
    batch, modes, samples = 2, 3, 12
    return InitialValueProblem(
        8000,
        samples,
        torch.linspace(-0.01, 0.02, 2 * modes, dtype=dtype).repeat(batch, 1),
        torch.tensor([[2.0, 3.0, 5.0]], dtype=dtype).repeat(batch, 1),
        torch.full((batch, modes), 0.1, dtype=dtype),
        torch.full((batch,), 0.2, dtype=dtype),
        torch.full((batch, modes), 0.2, dtype=dtype),
        torch.full((batch, samples), 0.01, dtype=dtype),
    )


class JointPotentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_values_input_and_parameter_gradients(self):
        for kind in ["cgrad", "mgrad", "mgrad_transform"]:
            for bias in [False, True]:
                for dtype in [torch.float32, torch.float64]:
                    with self.subTest(kind=kind, bias=bias, dtype=dtype):
                        a = make_law(kind, bias, dtype)
                        b = copy.deepcopy(a)
                        x = torch.linspace(-0.2, 0.3, 12, dtype=dtype).reshape(4, 3).requires_grad_()
                        z = x.detach().clone().requires_grad_()
                        force, potential = a(x), a.potential(x)
                        joint_force, joint_potential = b.forward_and_potential(z)
                        tol = 3e-5 if dtype == torch.float32 else 2e-11
                        torch.testing.assert_close(joint_force, force, rtol=tol, atol=tol)
                        torch.testing.assert_close(joint_potential, potential, rtol=tol, atol=tol)
                        old_grads = torch.autograd.grad(
                            force.square().sum() + 0.3 * potential.square().sum(), (x, *a.parameters())
                        )
                        new_grads = torch.autograd.grad(
                            joint_force.square().sum() + 0.3 * joint_potential.square().sum(), (z, *b.parameters())
                        )
                        for old, new in zip(old_grads, new_grads):
                            self.assertTrue(torch.isfinite(new).all().item())
                            torch.testing.assert_close(new, old, rtol=tol, atol=tol)
                        self.assertEqual(list(a.state_dict()), list(b.state_dict()))

    def test_joint_calls_remove_one_projection_per_module(self):
        for kind, modules in [("cgrad", 1), ("mgrad", 2), ("mgrad_transform", 2)]:
            with self.subTest(kind=kind):
                law, x = make_law(kind), torch.ones((2, 3), dtype=torch.float64) * 0.1
                linear = F.linear
                with patch("torch.nn.functional.linear", wraps=linear) as count:
                    law(x), law.potential(x)
                    self.assertEqual(count.call_count, 3 * modules)
                    count.reset_mock()
                    law.forward_and_potential(x)
                    self.assertEqual(count.call_count, 2 * modules)

    def test_torchscript_joint_api(self):
        x = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float64)
        for kind in ["cgrad", "mgrad", "mgrad_transform"]:
            for bias in [False, True]:
                with self.subTest(kind=kind, bias=bias):
                    law = make_law(kind, bias)
                    scripted = torch.jit.script(copy.deepcopy(law))
                    for actual, expected in zip(scripted.forward_and_potential(x), law.forward_and_potential(x)):
                        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)

    def test_generic_fallback_and_nonsequential_steps(self):
        law = PolynomialLaw()
        method = SAV(law, lambda0=0)
        scripted = torch.jit.script(SAV(PolynomialLaw(), lambda0=0))
        ivp = problem()
        y, params = method.init(ivp)
        for n in [0, 5, 1, 0]:
            expected = method.step(n, y, params)
            actual = scripted.step(n, y, params)
            torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-11)

    def test_solver_values_and_parameter_gradients_both_adjoints(self):
        for kind in ["cgrad", "mgrad", "mgrad_transform"]:
            for adjoint in [AutoDiffAdjoint, CheckpointAdjoint]:
                for lambda0 in [0, 1000]:
                    with self.subTest(kind=kind, adjoint=adjoint.__name__, lambda0=lambda0):
                        law = make_law(kind)
                        baseline = adjoint(SAV(SeparateLaw(copy.deepcopy(law)), lambda0=lambda0))
                        candidate = adjoint(SAV(copy.deepcopy(law), lambda0=lambda0))
                        old, new = baseline(problem()), candidate(problem())
                        torch.testing.assert_close(new, old, rtol=2e-10, atol=2e-11)
                        old.square().sum().backward()
                        new.square().sum().backward()
                        for a, b in zip(baseline.parameters(), candidate.parameters()):
                            self.assertIsNotNone(a.grad)
                            self.assertIsNotNone(b.grad)
                            self.assertTrue(torch.isfinite(b.grad).all().item())
                            torch.testing.assert_close(b.grad, a.grad, rtol=2e-9, atol=2e-10)

    def test_torchscript_solver_joint_and_fallback(self):
        for kind in ["cgrad", "mgrad", "mgrad_transform"]:
            law = make_law(kind)
            for separate in [False, True]:
                with self.subTest(kind=kind, separate=separate):
                    selected = SeparateLaw(copy.deepcopy(law)) if separate else copy.deepcopy(law)
                    eager = AutoDiffAdjoint(SAV(selected, lambda0=0))
                    scripted = torch.jit.script(copy.deepcopy(eager))
                    torch.testing.assert_close(scripted(problem()), eager(problem()), rtol=2e-10, atol=2e-11)


if __name__ == "__main__":
    unittest.main()
