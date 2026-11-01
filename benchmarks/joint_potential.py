"""Callable, CPU-only synthetic microbenchmark; no checkpoint or data download.

Run from the repository root: python3 -m benchmarks.joint_potential
This times joint force/potential evaluation, not an audio renderer.
"""

import argparse
import json
import statistics
import time

import torch

from src.models.grad_net import BaseCGradNet
from src.utils.activation import Tanh


def benchmark(dim=8, hidden=32, batch=4, iterations=500, repeats=6, scripted=False):
    if min(dim, hidden, batch, iterations, repeats) < 1:
        raise ValueError("All dimensions and counts must be positive")
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(17)
        law = BaseCGradNet(dim, hidden, Tanh(), log_scale_std=0.1).double().eval()
        if scripted:
            law = torch.jit.script(law)
        x = torch.linspace(-0.2, 0.3, batch * dim, dtype=torch.float64).reshape(batch, dim)

        def separate():
            return law(x), law.potential(x)

        def joint():
            return law.forward_and_potential(x)

        functions = {"separate": separate, "joint": joint}
        rows = []
        with torch.inference_mode():
            for a, b in zip(separate(), joint()):
                torch.testing.assert_close(a, b, rtol=2e-11, atol=2e-11)
            for _ in range(50):
                separate()
                joint()
            for repeat in range(repeats):
                order = ["separate", "joint"] if repeat % 2 == 0 else ["joint", "separate"]
                for name in order:
                    started = time.perf_counter()
                    for _ in range(iterations):
                        functions[name]()
                    rows.append({"repeat": repeat, "method": name, "seconds": time.perf_counter() - started})
        medians = {name: statistics.median(r["seconds"] for r in rows if r["method"] == name) for name in functions}
        return {
            "torch": torch.__version__,
            "device": "cpu",
            "dtype": "float64",
            "threads": 1,
            "scripted": scripted,
            "dimensions": {"input": dim, "hidden": hidden, "batch": batch},
            "iterations_per_measurement": iterations,
            "repeats": repeats,
            "rows": rows,
            "median_seconds": medians,
            "separate_over_joint": medians["separate"] / medians["joint"],
            "scope": "Synthetic joint-evaluation microbenchmark; no renderer speedup claim.",
        }
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scripted", action="store_true")
    args = parser.parse_args()
    print(json.dumps(benchmark(scripted=args.scripted), indent=2))
