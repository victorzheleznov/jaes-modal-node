"""
Adjoints for backpropagation through the numerical solver.
"""

import torch
from torch import nn

from .methods import BaseMethod
from .problem import InitialValueProblem


class AutoDiffAdjoint(nn.Module):
    """Automatic differentiation adjoint (i.e., the "discretise-then-optimise" method)."""
    def __init__(self, method: BaseMethod, use_last: bool = False):
        super().__init__()
        self._method = method
        self._use_last = use_last

    def __str__(self):
        return "auto_diff"

    def forward(self, ivp: InitialValueProblem) -> torch.Tensor:
        # initialise
        y0, params = self._method.init(ivp)
        y = torch.zeros(
            (y0.shape[0], y0.shape[1], ivp.num_samples) if not self._use_last else (0, 0, 0),
            dtype=y0.dtype,
            device=y0.device
        )
        if not self._use_last:
            y[..., 0] = y0

        # main loop
        for n in range(ivp.num_samples - 1):
            y0 = self._method.step(n, y0, params)
            if not self._use_last:
                y[..., n + 1] = y0

        return y0 if self._use_last else y


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, solver: AutoDiffAdjoint, ivp: InitialValueProblem, use_last: bool, *trainable_params):
        with torch.no_grad():
            y = solver(ivp)
        ctx.solver = solver
        ctx.ivp = ivp
        ctx.use_last = use_last
        ctx.save_for_backward(y.detach(), *trainable_params)
        return y[..., -1] if use_last else y

    @staticmethod
    def backward(ctx, partial_grad_y: torch.Tensor):
        # parse forward call
        solver = ctx.solver
        ivp = ctx.ivp
        use_last = ctx.use_last
        y, *trainable_params, = ctx.saved_tensors

        # initialise
        _, method_params = solver._method.init(ivp)
        grad_trainable_params = [torch.zeros(p.shape, dtype=p.dtype, device=p.device) for p in trainable_params]
        grad_outputs = (
            partial_grad_y.unsqueeze(0)
            if use_last
            else partial_grad_y.movedim(-1, 0)
        )

        # main loop
        for n in range(ivp.num_samples - 2, -1, -1):
            idx = 0 if use_last else slice(n + 1, None)

            # calculate partial gradient
            with torch.enable_grad():
                y0 = y[..., n].requires_grad_(True)
                y1 = solver._method.step(n, y0, method_params)

                *partial_grad_trainable_params, grad_outputs[idx, ...] = torch.autograd.grad(
                    y1,
                    trainable_params + [y0],
                    grad_outputs=grad_outputs[idx, ...],
                    is_grads_batched=(not use_last),
                    retain_graph=False
                )

            # accumulate total gradient
            for i in range(len(grad_trainable_params)):
                grad_trainable_params[i] += (
                    partial_grad_trainable_params[i]
                    if use_last
                    else partial_grad_trainable_params[i].sum(0)
                )

        return (
            None,
            None,
            None,
            *grad_trainable_params
        )


class CheckpointAdjoint(nn.Module):
    """Checkpoint adjoint.
    
    Accumulates gradients by backpropagating through each time step iteratively without storing the whole computational
    graph for the trajectory. Used to reduce memory footprint.
    """
    # TODO: Calculate gradients with respect to initial conditions.
    def __init__(self, method: BaseMethod, use_last: bool = False):
        super().__init__()
        self._solver = AutoDiffAdjoint(method, use_last=False)
        self._use_last = use_last

    def __str__(self):
        return "checkpoint"

    def forward(self, ivp: InitialValueProblem) -> torch.Tensor:
        y = CheckpointFunction.apply(
            self._solver,
            ivp,
            self._use_last,
            *list(self._solver.parameters(recurse=True))
        )
        return y
