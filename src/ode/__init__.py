"""
Suite of differentiable numerical solvers for modal systems.

Implementation here is inspired by the `torchode` package:
https://github.com/martenlienen/torchode
"""


from .adjoint import AutoDiffAdjoint, CheckpointAdjoint
from .methods import SAV, Euler, Verlet
from .problem import InitialValueProblem

METHODS = dict()
IS_INTERLEAVED = dict()
ADJOINTS = dict()


def _register_method(name, constructor, is_interleaved):
    METHODS[name] = constructor
    IS_INTERLEAVED[name] = is_interleaved


def _register_adjoint(name, constructor):
    ADJOINTS[name] = constructor


_register_method("euler", Euler, False)
_register_method("verlet", Verlet, True)
_register_method("sav", SAV, True)

_register_adjoint("auto_diff", AutoDiffAdjoint)
_register_adjoint("checkpoint", CheckpointAdjoint)


__all__ = [
    "InitialValueProblem",
    "METHODS",
    "IS_INTERLEAVED",
    "ADJOINTS"
]
