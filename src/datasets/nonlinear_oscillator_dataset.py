import random
from collections.abc import Callable
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src.datasets import BaseDataset
from src.generators import NonlinearOscillator
from src.models import ModalSystem
from src.utils.excitation import Excitation
from src.utils.init import log_git
from src.utils.io import ROOT_PATH, write_wav
from src.utils.misc import calc_md5, gen_from_range, get_dtype
from src.utils.writer import WandBWriter

MAX_NUM_PLOTS = 10
NUM_POINTS = 1000
NUM_PERIODS = 5


class NonlinearOscillatorDataset(BaseDataset):
    """Class for nonlinear oscillator dataset."""
    def __init__(
            self,
            fs: int,
            dur: float,
            method_name: str,
            omega_range: list[float],
            sigma_range: list[float],
            mu_range: list[float],
            exc_amp_range: list[float],
            exc_dur_range: list[float],
            exc_type: int,
            pos_ic_range: list[float],
            vel_ic_range: list[float],
            nl_name: str = "pow",
            num_variations: int = 10,
            seed: int = 0,
            float_dtype: str = "double",
            device: str = "cpu",
            save: bool = True,
            method_kwargs: dict = {},
            nl_kwargs: dict = {}
        ) -> None:
        """Parameters
        ----------
        fs : int
            Sampling rate [Hz].
        dur : float
            Simulation duration [sec].
        method_name : str
            Numerical method (e.g., "verlet" or "sav").
        omega_range : list[float]
            Range for randomised angular frequencies.
        sigma_range : list[float]
            Range for randomised damping parameters.
        mu_range : list[float]
            Range for randomised scaling factors of the nonlinearity (denoted as "nu" in the JAES paper).
        exc_amp_range : list[float]
            Range for randomised excitation amplitudes (if [0, 0] then there is no excitation).
        exc_dur_range : list[float]
            Range for randomised excitation durations.
        exc_type : int
            Excitation type: 1 - pluck, 2 - strike.
        pos_ic_range : list[float]
            Range for randomised initial displacements.
        vel_ic_range : list[float]
            Range for randomised initial velocities.
        nl_name : str
            Nonlinearity type: "pow", "tanh", "sinh" or "zero".
        num_variations : int
            Number of variations (i.e., number of different oscillators) to generate.
        seed : int
            Seed for random numbers.
        float_dtype : str
            Floating number type: "float" or "double".
        device : str
            Device (e.g., "cpu" or "cuda") used to generate data.
        save : bool
            Flag to save dataset.
        method_kwargs : dict
            Keyword arguments for the numerical method.
        nl_kwargs : dict
            Keyword arguments for the nonlinearity (e.g., "exponent" for "pow" function).

        Notes
        -----
        The dataset is generated upon initialisation of this class. It is saved to a folder with an unique name based on
        specified generation parameters. If the path exists, it is assumed that the data is already generated and
        the dataset is loaded without generation.
        """
        super().__init__()

        # parse parameters
        self._omega_range = tuple(omega_range)
        self._sigma_range = tuple(sigma_range)
        self._mu_range = tuple(mu_range)
        self._exc_amp_range = tuple(exc_amp_range)
        self._exc_dur_range = tuple(exc_dur_range)
        self._exc_st = 0.0
        self._exc_type = exc_type
        self._pos_ic_range = tuple(pos_ic_range)
        self._vel_ic_range = tuple(vel_ic_range)
        self._num_variations = num_variations
        self._seed = seed
        self._float_dtype = get_dtype(float_dtype)
        self._device = torch.device(device)
        self._save = save
        self._nl_name = nl_name
        self._method_kwargs = method_kwargs
        self._nl_kwargs = nl_kwargs

        # create generators
        self._gen = NonlinearOscillator(
            fs,
            dur,
            method_name,
            nl_name=nl_name,
            method_kwargs=method_kwargs,
            nl_kwargs=nl_kwargs
        ).to(self._float_dtype).to(self._device)

        self._gen_lin = NonlinearOscillator(
            fs,
            dur,
            method_name,
            nl_name="zero",
            method_kwargs=method_kwargs
        ).to(self._float_dtype).to(self._device)

        if self._save:
            # specify data paths
            self._dataset_dir = (
                ROOT_PATH
                / "data"
                / "nonlinear_oscillator"
                / str(self)
            )
            self._audio_dir = self._dataset_dir / "audio"
            self._data_file = self._dataset_dir / "data.pt"
            self._data_lin_file = self._dataset_dir / "data_lin.pt"
            self._meta_file = self._dataset_dir / "meta.pt"
            self._params_file = self._dataset_dir / "params.txt"

            # synthesise or load dataset
            if not self._dataset_dir.exists():
                self._dataset_dir.mkdir(parents=True)
                self._audio_dir.mkdir(parents=True)
                self._log_params()
                log_git(self._dataset_dir)
                self._synthesise_dataset()
                self._save_dataset()
                print(f"Dataset saved to {self._dataset_dir}")
            else:
                self._load_dataset()
                print(f"Loaded dataset from {self._dataset_dir}")
        else:
            self._synthesise_dataset()

    def __str__(self):
        return f"{self._gen.method_name}_{self._gen.fs:d}Hz_{self._gen.dur:g}sec_{self.md5}"

    def __len__(self):
        return self._num_variations

    def __getitem__(self, idx):
        # get metadata
        instance = self._meta[idx]

        # get nonlinear oscillator output
        for key in self._data.keys():
            instance["target_" + key] = self._data[key][idx, ...]

        # get linear oscillator output
        for key in self._data_lin.keys():
            instance["linear_" + key] = self._data_lin[key][idx, ...]

        instance["idx"] = idx
        return instance

    def _synthesise_dataset(self):
        g = torch.Generator(device=self._device)
        g.manual_seed(self._seed)

        # generate initial conditions
        q0 = gen_from_range(
            self._pos_ic_range,
            (self._num_variations, 1),
            dtype=self._float_dtype,
            device=self._device,
            generator=g
        )
        p0 = gen_from_range(
            self._vel_ic_range,
            (self._num_variations, 1),
            dtype=self._float_dtype,
            device=self._device,
            generator=g
        )
        y0 = torch.cat([q0, p0], dim=-1)

        # generate physical parameters
        omega = gen_from_range(
            self._omega_range,
            (self._num_variations, 1),
            dtype=self._float_dtype,
            device=self._device,
            generator=g
        )
        sigma = gen_from_range(
            self._sigma_range,
            (self._num_variations, 1),
            dtype=self._float_dtype,
            device=self._device,
            generator=g
        )
        mu = gen_from_range(
            self._mu_range,
            (self._num_variations,),
            dtype=self._float_dtype,
            device=self._device,
            generator=g
        )

        # generate excitation (if amplitude range is defined)
        use_excitation = any(self._exc_amp_range)
        if use_excitation:
            exc_amp = gen_from_range(
                self._exc_amp_range,
                (self._num_variations,),
                dtype=self._float_dtype,
                device=self._device,
                generator=g
            )
            exc_dur = gen_from_range(
                self._exc_dur_range,
                (self._num_variations,),
                dtype=self._float_dtype,
                device=self._device,
                generator=g
            )
            exc_st = self._exc_st * torch.ones((self._num_variations,), dtype=self._float_dtype, device=self._device)
            exc_type = self._exc_type * torch.ones((self._num_variations,), dtype=torch.int, device=self._device)

        # generate oscillator output
        if use_excitation:
            data = self._gen(y0, omega, sigma, mu, exc_amp=exc_amp, exc_dur=exc_dur, exc_st=exc_st, exc_type=exc_type)
            data_lin = self._gen_lin(y0, omega, sigma, mu, exc_amp=exc_amp, exc_dur=exc_dur, exc_st=exc_st, exc_type=exc_type)
        else:
            data = self._gen(y0, omega, sigma, mu)
            data_lin = self._gen_lin(y0, omega, sigma, mu)
        self._data = {key: value.detach().cpu() for key, value in data.items()}
        self._data_lin = {key: value.detach().cpu() for key, value in data_lin.items()}

        # store metadata
        self._meta = [{
            "fs": self._gen.fs,
            "dur": self._gen.dur,
            "method_name": self._gen.method_name,
            "y0": y0[idx, :].detach().cpu(),
            "omega": omega[idx, :].detach().cpu(),
            "sigma": sigma[idx, :].detach().cpu(),
            "mu": mu[idx].item(),
            "xe": self._gen.xe,
            "xo": self._gen.xo
        } for idx in range(self._num_variations)]

        if use_excitation:
            for idx in range(self._num_variations):
                self._meta[idx].update({
                    "exc_amp": exc_amp[idx].item(),
                    "exc_dur": exc_dur[idx].item(),
                    "exc_st": exc_st[idx].item(),
                    "exc_type": exc_type[idx].item(),
                })

    def _save_dataset(self):
        torch.save(self._data, self._data_file)
        torch.save(self._data_lin, self._data_lin_file)
        torch.save(self._meta, self._meta_file)
        for idx in range(len(self)):
            write_wav(self._audio_dir / f"{idx}.wav", self._data["w"][idx, :], self._gen.fs, normalise=True)
            write_wav(self._audio_dir / f"{idx}_lin.wav", self._data_lin["w"][idx, :], self._gen_lin.fs, normalise=True)

    def _load_dataset(self):
        self._meta = torch.load(self._meta_file, weights_only=True)
        self._data = torch.load(self._data_file, weights_only=True)
        self._data_lin = torch.load(self._data_lin_file, weights_only=True)

    def _log_params(self):
        with open(self._params_file, "w") as f:
            f.write(f"{str(self.params)}\n{self.md5}")

    def save_pred(self, test_dir: Path, output: torch.Tensor, w: torch.Tensor, idx: torch.Tensor, **batch):
        data_dir = test_dir / "data"
        audio_dir = test_dir / "audio"
        data_dir.mkdir(exist_ok=True, parents=True)
        audio_dir.mkdir(exist_ok=True, parents=True)
        batch_size = output.shape[0]
        for i in range(batch_size):
            d = {"output": output[i, ...].detach().cpu(), "w": w[i, ...].detach().cpu()}
            torch.save(d, data_dir / f"{idx[i]}_pred.pt")
            write_wav(audio_dir / f"{idx[i]}_pred.wav", d["w"], self._gen.fs, normalise=True)

    def plot(self):
        use_excitation = ("exc_amp" in self[0].keys())
        num_plots = 2 + int(use_excitation)

        fig, axs = plt.subplots(nrows=num_plots, ncols=1, layout="constrained")
        fig.suptitle("Nonlinear oscillator dataset", size="medium")

        axs[0].set_xlabel("Time [sec]")
        axs[0].set_ylabel("Displacement")

        axs[1].set_xlabel("Time [sec]")
        axs[1].set_ylabel("Velocity")

        if use_excitation:
            axs[2].set_xlabel("Time [sec]")
            axs[2].set_ylabel("Excitation")

        for idx in range(len(self)):
            instance = self[idx]
            self._plot_instance(axs, **instance)

    @torch.no_grad()
    def plot_batch(
            self,
            idx: torch.Tensor,
            target_output: torch.Tensor,
            output: torch.Tensor = None,
            linear_output: torch.Tensor = None,
            model: nn.Module = None,
            key: str = None,
            **batch
        ):
        batch_size = target_output.shape[0]
        indices = random.sample(range(batch_size), min(MAX_NUM_PLOTS, batch_size))

        for i in indices:
            title = f"Instance {idx[i].item()}"
            if key is not None:
                title = key.title() + " " + title.lower()

            fig, _ = self._plot_output(
                target_output[i, ...],
                output[i, ...] if output is not None else None,
                linear_output[i, ...] if linear_output is not None else None,
            )
            fig.suptitle(title, size="medium")

        if isinstance(model, ModalSystem):
            fig, _ = self._plot_nonlinearity(self._gen.nl, model.nl, target_output)

    @torch.no_grad()
    def log_batch(
            self,
            writer: WandBWriter,
            y0: torch.Tensor,
            omega: torch.Tensor,
            sigma: torch.Tensor,
            mu: torch.Tensor,
            xe: torch.Tensor,
            xo: torch.Tensor,
            idx: torch.Tensor,
            target_output: torch.Tensor,
            output: torch.Tensor = None,
            linear_output: torch.Tensor = None,
            exc_amp: torch.Tensor = None,
            exc_dur: torch.Tensor = None,
            exc_st: torch.Tensor = None,
            exc_type: torch.Tensor = None,
            model: nn.Module = None,
            **batch
        ):
        title = f"Instance {idx[0].item()}"

        fig, _ = self._plot_output(
            target_output[0, ...],
            output[0, ...] if output is not None else None,
            linear_output[0, ...] if linear_output is not None else None
        )
        fig.suptitle(title, size="medium")
        writer.add_fig("output_slice_dur", fig)

        model.eval()
        if isinstance(model, ModalSystem):
            pred = model(
                self._gen._fs,
                self._gen._dur,
                y0[0, ...].unsqueeze(0),
                omega[0, ...].unsqueeze(0),
                sigma[0, ...].unsqueeze(0),
                mu[0].unsqueeze(0),
                xe[0].unsqueeze(0),
                xo[0].unsqueeze(0),
                exc_amp[0].unsqueeze(0) if exc_amp is not None else None,
                exc_dur[0].unsqueeze(0) if exc_dur is not None else None,
                exc_st[0].unsqueeze(0) if exc_st is not None else None,
                exc_type[0].unsqueeze(0) if exc_type is not None else None
            )
            instance = self[idx[0].item()]
            xlim = [
                1,
                2.0 * np.pi * NUM_PERIODS / omega[0, 0].item() * self._gen.fs
            ]

            fig, _ = self._plot_output(
                instance["target_output"],
                pred["output"][0, ...],
                instance["linear_output"],
                xlim
            )
            fig.suptitle(title, size="medium")
            writer.add_fig("output_dataset_dur", fig)

            writer.add_audio("wave_dataset_dur", pred["w"][0, ...], fs=self._gen.fs)

            fig, _ = self._plot_nonlinearity(self._gen.nl, model.nl, target_output)
            writer.add_fig("nonlinearity", fig)

            self._log_activations(writer, model.nl, target_output)

            del pred
            del instance

    @staticmethod
    def _log_activations(writer: WandBWriter, nl: nn.Module, target_output: torch.Tensor):
        if not isinstance(nl, torch.jit.ScriptModule):
            q_points = target_output[:, 0, :].detach().reshape((-1, 1))

            hooks = dict()
            def activation_hook(module, input, output, name):
                histogram_name = name + "_" + repr(module)
                writer.add_histogram(histogram_name, output)
                hooks[name].remove()
                del hooks[name]

            for name, module in nl.named_modules(remove_duplicate=False):
                if len(module._modules) == 0:
                    hooks[name] = module.register_forward_hook(partial(activation_hook, name=name))

            nl(q_points)

    @staticmethod
    def _plot_instance(
            axs,
            fs: int,
            target_output: torch.Tensor,
            linear_output: torch.Tensor = None,
            exc_amp: float = None,
            exc_dur: float = None,
            exc_st: float = None,
            exc_type: int = None,
            **instance
        ):
        use_excitation = (exc_amp is not None)

        target_output = target_output.detach().cpu().numpy()
        if linear_output is not None:
            linear_output = linear_output.detach().cpu().numpy()

        num_samples = target_output.shape[-1]
        t_points = np.arange(start=0, stop=num_samples, step=1) / fs

        p = axs[0].plot(t_points, target_output[0, :])
        color = p[0].get_color()
        if linear_output is not None:
            axs[0].plot(t_points, linear_output[0, :], color=color, linestyle="dashed")

        axs[1].plot(t_points, target_output[1, :], color=color)
        if linear_output is not None:
            axs[1].plot(t_points, linear_output[1, :], color=color, linestyle="dashed")

        if use_excitation:
            fe = Excitation(
                torch.as_tensor(exc_amp),
                torch.as_tensor(exc_dur),
                torch.as_tensor(exc_st),
                torch.as_tensor(exc_type)
            )
            fe_points = fe(torch.from_numpy(t_points)).squeeze().numpy()
            axs[2].plot(t_points, fe_points, color=color)

    @staticmethod
    def _plot_output(
            target_output: torch.Tensor,
            output: torch.Tensor = None,
            linear_output: torch.Tensor = None,
            xlim: list[float] = None
        ):
        target_output = target_output.detach().cpu().numpy()
        if output is not None:
            output = output.detach().cpu().numpy()
        if linear_output is not None:
            linear_output = linear_output.detach().cpu().numpy()

        num_states = target_output.shape[-2]
        fig, axs = plt.subplots(nrows=num_states, ncols=1, layout="constrained")

        axs[0].set_ylabel("Displacement", fontsize="small")
        axs[1].set_ylabel("Velocity", fontsize="small")
        if num_states > 2:
            axs[2].set_ylabel(r"$\psi^2 - \psi_0^2$", fontsize="small")

        for n in range(num_states):
            if linear_output is not None:
                axs[n].plot(
                    (linear_output[n, :]**2 - linear_output[n, 0]**2)
                    if (n == 2)
                    else linear_output[n, :],
                    label="Linear",
                    color="tab:blue"
                )
            axs[n].plot(
                (target_output[n, :]**2 - target_output[n, 0]**2)
                if (n == 2)
                else target_output[n, :],
                label="Target",
                color="tab:green"
            )
            if output is not None:
                axs[n].plot(
                    (output[n, :]**2 - output[n, 0]**2)
                    if (n == 2)
                    else output[n, :],
                    label="Predicted",
                    color="tab:orange",
                    linestyle="dashed"
                )
            if xlim:
                axs[n].set_xlim(xlim)
            axs[n].ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
            axs[n].yaxis.get_offset_text().set_fontsize("small")
            axs[n].tick_params(axis="both", labelsize="small")
        axs[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3, fontsize="small")
        axs[-1].set_xlabel("Sample [n]", fontsize="small")

        return fig, axs

    @staticmethod
    def _plot_nonlinearity(target_nl: Callable, nl: nn.Module, target_output: torch.Tensor):
        dtype = next(nl.parameters()).dtype
        device = next(nl.parameters()).device

        # generate input points
        q_max = torch.amax(target_output[:, 0, :]).item()
        q_min = torch.amin(target_output[:, 0, :]).item()
        q_step = (q_max - q_min) / NUM_POINTS
        q_points = torch.arange(start=q_min, end=q_max, step=q_step, dtype=dtype, device=device)

        # initialise figure
        if hasattr(target_nl, "potential") and hasattr(nl, "potential"):
            plot_potential = (callable(target_nl.potential) and callable(nl.potential))
        else:
            plot_potential = False
        num_plots = 1 + int(plot_potential)
        fig, axs = plt.subplots(nrows=1, ncols=num_plots, layout="constrained", squeeze=False)

        # plot nonlinearity
        axs[0, 0].plot(
            q_points.cpu().numpy(),
            target_nl(q_points.unsqueeze(-1)).cpu().numpy(),
            label="Target",
            color="tab:green"
        )
        axs[0, 0].plot(
            q_points.cpu().numpy(),
            nl(q_points.unsqueeze(-1)).cpu().numpy(),
            label="Predicted",
            color="tab:orange",
            linestyle="dashed"
        )
        axs[0, 0].set_xlabel("Displacement", fontsize="small")
        axs[0, 0].set_ylabel("Nonlinearity", fontsize="small")
        axs[0, 0].ticklabel_format(style="sci", axis="both", scilimits=(0, 0), useMathText=True)
        axs[0, 0].xaxis.get_offset_text().set_fontsize("small")
        axs[0, 0].yaxis.get_offset_text().set_fontsize("small")
        axs[0, 0].tick_params(axis="both", labelsize="small")
        axs[0, 0].grid()
        fig.legend(loc="outside upper center", ncols=2, fontsize="small")

        # plot potential
        if plot_potential:
            axs[0, 1].plot(
                q_points.cpu().numpy(),
                (
                    target_nl.potential(q_points.unsqueeze(-1))
                    - target_nl.potential(torch.zeros_like(q_points.unsqueeze(-1)))
                ).cpu().numpy(),
                label="Target",
                color="tab:green"
            )
            axs[0, 1].plot(
                q_points.cpu().numpy(),
                (
                    nl.potential(q_points.unsqueeze(-1))
                    - nl.potential(torch.zeros_like(q_points.unsqueeze(-1)))
                ).cpu().numpy(),
                label="Predicted",
                color="tab:orange",
                linestyle="dashed"
            )
            axs[0, 1].set_xlabel("Displacement", fontsize="small")
            axs[0, 1].set_ylabel(r"$V - V_0$", fontsize="small")
            axs[0, 1].ticklabel_format(style="sci", axis="both", scilimits=(0, 0), useMathText=True)
            axs[0, 1].xaxis.get_offset_text().set_fontsize("small")
            axs[0, 1].yaxis.get_offset_text().set_fontsize("small")
            axs[0, 1].tick_params(axis="both", labelsize="small")
            axs[0, 1].grid()
            potential0 = nl.potential(torch.zeros((1, 1), dtype=dtype, device=device)).item()
            axs[0, 1].set_title(f"$V_0=${potential0:.1E}", size="small")

        return fig, axs

    @property
    def params(self) -> tuple:
        t = (
            self._gen.fs,
            self._gen.dur,
            self._gen.method_name,
            self._omega_range,
            self._sigma_range,
            self._mu_range,
            self._exc_amp_range,
            self._exc_dur_range,
            self._exc_type,
            self._pos_ic_range,
            self._vel_ic_range,
            self._nl_name,
            self._num_variations,
            self._seed,
            self._float_dtype,
            self._method_kwargs,
            self._nl_kwargs
        )
        return t

    @property
    def md5(self) -> str:
        return calc_md5(str(self.params))
