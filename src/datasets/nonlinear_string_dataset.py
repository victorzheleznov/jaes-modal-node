import copy
import random
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from src.datasets import BaseDataset
from src.generators import NonlinearString
from src.models import ModalSystem
from src.utils.excitation import Excitation
from src.utils.init import log_git
from src.utils.io import ROOT_PATH, write_wav
from src.utils.misc import calc_md5, gen_from_range, get_dtype
from src.utils.writer import WandBWriter

MAX_NUM_PLOTS = 10
NUM_POINTS = 1000
START_MODE = 0
NUM_MODES = 3
NUM_PERIODS = 5


class NonlinearStringDataset(BaseDataset):
    """Class for nonlinear string dataset."""
    def __init__(
            self,
            fs: int,
            dur: float,
            method_name: str,
            num_modes: int,
            gamma_range: list[float],
            kappa_range: list[float],
            mu_range: list[float],
            sigma0_range: list[float],
            sigma1_range: list[float],
            xe_range: list[float],
            xo_range: list[float],
            exc_amp_range: list[float],
            exc_dur_range: list[float],
            exc_type: int,
            nl_name: str = "spectral",
            num_variations: int = 10,
            seed: int = 0,
            float_dtype: str = "double",
            device: str = "cpu",
            batch_size: int = None,
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
        num_modes : int
            Number of modes.
        gamma_range : list[float]
            Range for randomised wavespeeds.
        kappa_range : list[float]
            Range for randomised stiffness parameters.
        mu_range : list[float]
            Range for randomised scaling factors of the nonlinearity (denoted as "nu" in the JAES paper).
        sigma0_range : list[float]
            Range for randomised frequency-independent damping parameters.
        sigma1_range : list[float]
            Range for randomised frequency-dependent damping parameters.
        xe_range : list[float]
            Range for randomised excitation positions.
        xo_range : list[float]
            Range for randomised output positions.
        exc_amp_range : list[float]
            Range for randomised excitation amplitudes.
        exc_dur_range : list[float]
            Range for randomised excitation durations.
        exc_type : int
            Excitation type: 1 - pluck, 2 - strike.
        nl_name : str
            Nonlinearity type: "spectral" (used in the JAES paper) or "tensor" (used in the DAFx25 paper).
        num_variations : int
            Number of variations (i.e., number of different strings) to generate.
        seed : int
            Seed for random numbers.
        float_dtype : str
            Floating number type: "float" or "double".
        device : str
            Device (e.g., "cpu" or "cuda") used to generate data.
        batch_size : int
            Batch size for dataset generation (if None then equal to number of variations).
        method_kwargs : dict
            Keyword arguments for the numerical method.
        nl_kwargs : dict
            Keyword arguments for the nonlinearity.

        Notes
        -----
        The dataset is generated upon initialisation of this class. It is saved to a folder with an unique name based on
        specified generation parameters. If the path exists, it is assumed that the data is already generated and
        the dataset is loaded without generation.
        """
        super().__init__()

        # parse parameters
        self._gamma_range = tuple(gamma_range)
        self._kappa_range = tuple(kappa_range)
        self._mu_range = tuple(mu_range)
        self._sigma0_range = tuple(sigma0_range)
        self._sigma1_range = tuple(sigma1_range)
        self._xe_range = tuple(xe_range)
        self._xo_range = tuple(xo_range)
        self._exc_amp_range = tuple(exc_amp_range)
        self._exc_dur_range = tuple(exc_dur_range)
        self._exc_st = 0.0
        self._exc_type = exc_type
        self._nl_name = nl_name
        self._num_variations = num_variations
        self._seed = seed
        self._float_dtype = get_dtype(float_dtype)
        self._device = torch.device(device)
        self._batch_size = min(batch_size, num_variations) if batch_size else num_variations
        self._method_kwargs = method_kwargs
        self._nl_kwargs = nl_kwargs

        # create generators
        self._gen = NonlinearString(
            fs,
            dur,
            method_name,
            num_modes,
            nl_name=nl_name,
            method_kwargs=method_kwargs,
            nl_kwargs=nl_kwargs
        ).to(self._float_dtype).to(self._device)

        self._gen_lin = NonlinearString(
            fs,
            dur,
            method_name,
            num_modes,
            nl_name="zero",
            method_kwargs=method_kwargs,
            nl_kwargs=nl_kwargs
        ).to(self._float_dtype).to(self._device)

        # specify data paths
        self._dataset_dir = (
            ROOT_PATH
            / "data"
            / "nonlinear_string"
            / str(self)
        )
        self._audio_dir = self._dataset_dir / "audio"
        self._data_dir = self._dataset_dir / "data"
        self._meta_file = self._dataset_dir / "meta.pt"
        self._params_file = self._dataset_dir / "params.txt"

        # synthesise or load dataset
        if not self._dataset_dir.exists():
            self._dataset_dir.mkdir(parents=True)
            self._audio_dir.mkdir(parents=True)
            self._data_dir.mkdir(parents=True)
            self._log_params()
            log_git(self._dataset_dir)
            self._synthesise_dataset()
            self._save_dataset()
            print(f"Dataset saved to {self._dataset_dir}")
        else:
            self._load_dataset()
            print(f"Loaded dataset from {self._dataset_dir}")

    def __str__(self):
        return f"{self._gen.method_name}_{self._gen.fs:d}Hz_{self._gen.dur:g}sec_{self._gen.num_modes:d}modes_{self.md5}"

    def __len__(self):
        return self._num_variations

    def __getitem__(self, idx):
        # get metadata
        instance = self._meta[idx]

        # get nonlinear string output
        data = torch.load(self._data_dir / f"{idx}.pt", weights_only=True)
        for key in data.keys():
            instance["target_" + key] = data[key]

        instance["idx"] = idx
        return instance

    def _getlin(self, idx):
        # get linear string output
        instance = dict()
        data_lin = torch.load(self._data_dir / f"{idx}_lin.pt", weights_only=True)
        for key in data_lin.keys():
            instance["linear_" + key] = data_lin[key]
        return instance

    def _synthesise_dataset(self):
        g = torch.Generator(device=self._device)
        g.manual_seed(self._seed)

        # generate initial conditions
        y0 = torch.zeros((self._num_variations, int(2 * self._gen.num_modes)), dtype=self._float_dtype, device=self._device)

        # generate physical parameters
        gamma = gen_from_range(
            self._gamma_range,
            (self._num_variations,),
            dtype=self._float_dtype,
            device=self._device,
            generator=g
        )
        kappa = gen_from_range(
            self._kappa_range,
            (self._num_variations,),
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
        sigma0 = gen_from_range(
            self._sigma0_range,
            (self._num_variations,),
            dtype=self._float_dtype,
            device=self._device,
            generator=g
        )
        sigma1 = gen_from_range(
            self._sigma1_range,
            (self._num_variations,),
            dtype=self._float_dtype,
            device=self._device,
            generator=g
        )
        xe = gen_from_range(
            self._xe_range,
            (self._num_variations,),
            dtype=self._float_dtype,
            device=self._device,
            generator=g
        )
        xo = gen_from_range(
            self._xo_range,
            (self._num_variations,),
            dtype=self._float_dtype,
            device=self._device,
            generator=g
        )

        # generate excitation
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

        # batched loop
        omega = torch.zeros((self._num_variations, self._gen.num_modes), dtype=self._float_dtype, device=self._device)
        sigma = torch.zeros((self._num_variations, self._gen.num_modes), dtype=self._float_dtype, device=self._device)
        indices = torch.arange(
            start=0,
            end=((self._num_variations // self._batch_size) * self._batch_size)
        ).reshape((-1, self._batch_size))
        indices = [r for r in indices]
        if (self._num_variations % self._batch_size) != 0:
            indices.append(
                torch.arange(
                    start=((self._num_variations // self._batch_size) * self._batch_size),
                    end=self._num_variations
                )
            )
        for r in tqdm(indices):
            # generate nonlinear string output
            data = self._gen(
                y0[r, ...],
                gamma[r, ...],
                kappa[r, ...],
                mu[r, ...],
                sigma0[r, ...],
                sigma1[r, ...],
                xe[r, ...],
                xo[r, ...],
                exc_amp=exc_amp[r, ...],
                exc_dur=exc_dur[r, ...],
                exc_st=exc_st[r, ...],
                exc_type=exc_type[r, ...]
            )
            self._save_batch(data, r, save_lin=False)
            del data

            # store modal parameters
            omega[r, ...] = self._gen.omega
            sigma[r, ...] = self._gen.sigma

            # generate linear string output
            data_lin = self._gen_lin(
                y0[r, ...],
                gamma[r, ...],
                kappa[r, ...],
                mu[r, ...],
                sigma0[r, ...],
                sigma1[r, ...],
                xe[r, ...],
                xo[r, ...],
                exc_amp=exc_amp[r, ...],
                exc_dur=exc_dur[r, ...],
                exc_st=exc_st[r, ...],
                exc_type=exc_type[r, ...]
            )
            self._save_batch(data_lin, r, save_lin=True)
            del data_lin

        # store metadata
        self._meta = [{
            "fs": self._gen.fs,
            "dur": self._gen.dur,
            "method_name": self._gen.method_name,
            "num_modes": self._gen.num_modes,
            "gamma": gamma[idx].item(),
            "kappa": kappa[idx].item(),
            "mu": mu[idx].item(),
            "sigma0": sigma0[idx].item(),
            "sigma1": sigma1[idx].item(),
            "xe": xe[idx].item(),
            "xo": xo[idx].item(),
            "exc_amp": exc_amp[idx].item(),
            "exc_dur": exc_dur[idx].item(),
            "exc_st": exc_st[idx].item(),
            "exc_type": exc_type[idx].item(),
            "y0": y0[idx, :].detach().cpu(),
            "omega": omega[idx, :].detach().cpu(),
            "sigma": sigma[idx, :].detach().cpu(),
        } for idx in range(self._num_variations)]

    def _save_batch(self, data: dict[str, torch.Tensor], r: torch.Tensor, save_lin: bool = False):
        for i in range(len(r)):
            d = {key: value[i, ...].detach().cpu() for key, value in data.items()}
            idx = r[i]
            torch.save(
                d,
                self._data_dir / (f"{idx}.pt" if not save_lin else f"{idx}_lin.pt")
            )
            write_wav(
                self._audio_dir / (f"{idx}.wav" if not save_lin else f"{idx}_lin.wav"),
                d["w"],
                self._gen.fs,
                normalise=True
            )

    def _save_dataset(self):
        torch.save(self._meta, self._meta_file)

    def _load_dataset(self):
        self._meta = torch.load(self._meta_file, weights_only=True)

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
        fig, axs = plt.subplots(nrows=2, ncols=1, layout="constrained")
        fig.suptitle("Nonlinear string dataset", size="medium")

        axs[0].set_xlabel("Time [sec]")
        axs[0].set_ylabel("Output wave")

        axs[1].set_xlabel("Time [sec]")
        axs[1].set_ylabel("Excitation")

        for idx in range(len(self)):
            instance = copy.deepcopy(self[idx])
            instance.update(self._getlin(idx))
            self._plot_instance(axs, **instance)

    @torch.no_grad()
    def plot_batch(
            self,
            idx: torch.Tensor,
            target_output: torch.Tensor,
            target_w: torch.Tensor,
            output: torch.Tensor = None,
            w: torch.Tensor = None,
            linear_output: torch.Tensor = None,
            linear_w: torch.Tensor = None,
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

            # plot displacement
            fig, _ = self._plot_displacement(
                target_output[i, ...],
                target_w[i, ...],
                self._gen.num_modes,
                output[i, ...] if output is not None else None,
                w[i, ...] if w is not None else None,
                linear_output[i, ...] if linear_output is not None else None,
                linear_w[i, ...] if linear_w is not None else None,
            )
            fig.suptitle(title, size="medium")

            # plot velocity
            fig, _ = self._plot_velocity(
                target_output[i, ...],
                self._gen.num_modes,
                output[i, ...] if output is not None else None,
                linear_output[i, ...] if linear_output is not None else None,
            )
            fig.suptitle(title, size="medium")

            # plot auxiliary variable
            fig, _ = self._plot_auxiliary_variable(
                target_output[0, ...],
                self._gen.num_modes,
                output[0, ...] if output is not None else None,
                linear_output[0, ...] if linear_output is not None else None,
            )
            if fig is not None:
                fig.suptitle(title, size="medium")

        if isinstance(model, ModalSystem):
            self._plot_nonlinearity(self._gen.nl, model.nl, target_output, self._gen.num_modes)

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
            target_w: torch.Tensor,
            output: torch.Tensor = None,
            w: torch.Tensor = None,
            linear_output: torch.Tensor = None,
            linear_w: torch.Tensor = None,
            exc_amp: torch.Tensor = None,
            exc_dur: torch.Tensor = None,
            exc_st: torch.Tensor = None,
            exc_type: torch.Tensor = None,
            model: nn.Module = None,
            **batch
        ):
        title = f"Instance {idx[0].item()}"

        # plot displacement
        fig, _ = self._plot_displacement(
            target_output[0, ...],
            target_w[0, ...],
            self._gen.num_modes,
            output[0, ...] if output is not None else None,
            w[0, ...] if w is not None else None,
            linear_output[0, ...] if linear_output is not None else None,
            linear_w[0, ...] if linear_w is not None else None
        )
        fig.suptitle(title, size="medium")
        writer.add_fig("displacement_slice_dur", fig)

        # plot velocity
        fig, _ = self._plot_velocity(
            target_output[0, ...],
            self._gen.num_modes,
            output[0, ...] if output is not None else None,
            linear_output[0, ...] if linear_output is not None else None,
        )
        fig.suptitle(title, size="medium")
        writer.add_fig("velocity_slice_dur", fig)

        # plot auxiliary variable
        fig, _ = self._plot_auxiliary_variable(
            target_output[0, ...],
            self._gen.num_modes,
            output[0, ...] if output is not None else None,
            linear_output[0, ...] if linear_output is not None else None,
        )
        if fig is not None:
            fig.suptitle(title, size="medium")
            writer.add_fig("psi_slice_dur", fig)

        # evaluate model
        model.eval()
        if isinstance(model, ModalSystem):
            # simulate the whole duration
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

            # plot displacement
            fig, _ = self._plot_displacement(
                instance["target_output"],
                instance["target_w"],
                self._gen.num_modes,
                output=pred["output"][0, ...],
                w=pred["w"][0, ...],
                xlim=xlim
            )
            fig.suptitle(title, size="medium")
            writer.add_fig("displacement_dataset_dur", fig)

            # plot velocity
            fig, _ = self._plot_velocity(
                instance["target_output"],
                self._gen.num_modes,
                output=pred["output"][0, ...],
                xlim=xlim
            )
            fig.suptitle(title, size="medium")
            writer.add_fig("velocity_dataset_dur", fig)

            # plot auxiliary variable
            fig, _ = self._plot_auxiliary_variable(
                instance["target_output"],
                self._gen.num_modes,
                output=pred["output"][0, ...],
                xlim=xlim
            )
            if fig is not None:
                fig.suptitle(title, size="medium")
                writer.add_fig("psi_dataset_dur", fig)

            # log audio
            writer.add_audio("wave_dataset_dur", pred["w"][0, ...], fs=self._gen.fs)

            # plot nonlinearity
            fig, _ = self._plot_nonlinearity(self._gen.nl, model.nl, target_output, self._gen.num_modes)
            writer.add_fig("nonlinearity", fig)

            # log activations
            self._log_activations(writer, model.nl, target_output, self._gen.num_modes)

            del pred
            del instance

    @staticmethod
    def _log_activations(writer: WandBWriter, nl: nn.Module, target_output: torch.Tensor, num_modes: int):
        if not isinstance(nl, torch.jit.ScriptModule):
            q_points = target_output[:, :num_modes, :].detach()
            q_points = torch.movedim(q_points, -1, 1)
            q_points = torch.reshape(q_points, (-1, *q_points.shape[2:]))

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
            target_w: torch.Tensor,
            fs: int,
            exc_amp: float = None,
            exc_dur: float = None,
            exc_st: float = None,
            exc_type: int = None,
            linear_w: torch.Tensor = None,
            **instance
        ):
        target_w = target_w.detach().cpu().numpy()
        if linear_w is not None:
            linear_w = linear_w.detach().cpu().numpy()
        
        num_samples = target_w.shape[-1]
        t_points = np.arange(start=0, stop=num_samples, step=1) / fs

        p = axs[0].plot(t_points, target_w)
        color = p[0].get_color()
        if linear_w is not None:
            axs[0].plot(t_points, linear_w, color=color, linestyle="dashed")

        fe = Excitation(
            torch.as_tensor(exc_amp),
            torch.as_tensor(exc_dur),
            torch.as_tensor(exc_st),
            torch.as_tensor(exc_type)
        )
        fe_points = fe(torch.from_numpy(t_points)).squeeze().numpy()
        axs[1].plot(t_points, fe_points, color=color)

    @staticmethod
    def _plot_displacement(
            target_output: torch.Tensor,
            target_w: torch.Tensor,
            num_modes: int,
            output: torch.Tensor = None,
            w: torch.Tensor = None,
            linear_output: torch.Tensor = None,
            linear_w: torch.Tensor = None,
            xlim: list[float] = None
        ):
        target_output = target_output.detach().cpu().numpy()
        target_w = target_w.detach().cpu().numpy()
        if output is not None:
            output = output.detach().cpu().numpy()
        if w is not None:
            w = w.detach().cpu().numpy()
        if linear_output is not None:
            linear_output = linear_output.detach().cpu().numpy()
        if linear_w is not None:
            linear_w = linear_w.detach().cpu().numpy()

        num_plots = min(NUM_MODES, num_modes)
        fig, axs = plt.subplots(nrows=(num_plots + 1), ncols=1, layout="constrained")

        axs[0].set_ylabel("Output wave", fontsize="small")
        if linear_w is not None:
            axs[0].plot(linear_w, label="Linear", color="tab:blue")
        axs[0].plot(target_w, label="Target", color="tab:green")
        if w is not None:
            axs[0].plot(w, label="Predicted", color="tab:orange", linestyle="dashed")
        axs[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3, fontsize="small")
        if xlim:
            axs[0].set_xlim(xlim)
        axs[0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
        axs[0].yaxis.get_offset_text().set_fontsize("small")
        axs[0].tick_params(axis="both", labelsize="small")

        for n in range(num_plots):
            m = START_MODE + n
            axs[n + 1].set_ylabel(f"Mode {m + 1}", fontsize="small")
            if linear_output is not None:
                axs[n + 1].plot(linear_output[m, :], label="Linear", color="tab:blue")
            axs[n + 1].plot(target_output[m, :], label="Target", color="tab:green")
            if output is not None:
                axs[n + 1].plot(output[m, :], label="Predicted", color="tab:orange", linestyle="dashed")
            if xlim:
                axs[n + 1].set_xlim(xlim)
            axs[n + 1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
            axs[n + 1].yaxis.get_offset_text().set_fontsize("small")
            axs[n + 1].tick_params(axis="both", labelsize="small")
        axs[n + 1].set_xlabel("Sample [n]", fontsize="small")

        return fig, axs

    @staticmethod
    def _plot_velocity(
            target_output: torch.Tensor,
            num_modes: int,
            output: torch.Tensor = None,
            linear_output: torch.Tensor = None,
            xlim: list[float] = None
        ):
        target_output = target_output.detach().cpu().numpy()
        if output is not None:
            output = output.detach().cpu().numpy()
        if linear_output is not None:
            linear_output = linear_output.detach().cpu().numpy()

        num_plots = min(NUM_MODES, num_modes)
        fig, axs = plt.subplots(nrows=num_plots, ncols=1, layout="constrained", squeeze=False)

        for n in range(num_plots):
            m = START_MODE + n
            axs[n, 0].set_ylabel(f"Velocity {m + 1}", fontsize="small")
            if linear_output is not None:
                axs[n, 0].plot(linear_output[num_modes + m, :], label="Linear", color="tab:blue")
            axs[n, 0].plot(target_output[num_modes + m, :], label="Target", color="tab:green")
            if output is not None:
                axs[n, 0].plot(output[num_modes + m, :], label="Predicted", color="tab:orange", linestyle="dashed")
            if xlim:
                axs[n, 0].set_xlim(xlim)
            axs[n, 0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
            axs[n, 0].yaxis.get_offset_text().set_fontsize("small")
            axs[n, 0].tick_params(axis="both", labelsize="small")
        axs[0, 0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3, fontsize="small")
        axs[n, 0].set_xlabel("Sample [n]", fontsize="small")

        return fig, axs

    @staticmethod
    def _plot_auxiliary_variable(
            target_output: torch.Tensor,
            num_modes: int,
            output: torch.Tensor = None,
            linear_output: torch.Tensor = None,
            xlim: list[float] = None
        ):
        num_states = target_output.shape[-2]
        if num_states > (2 * num_modes):
            target_psi = target_output[-1, :].detach().cpu().numpy()
            if output is not None:
                psi = output[-1, :].detach().cpu().numpy()
            if linear_output is not None:
                linear_psi = linear_output[-1, :].detach().cpu().numpy()

            fig, axs = plt.subplots(nrows=1, ncols=1, layout="constrained")
            axs.set_ylabel(r"$\psi^2 - \psi_0^2$", fontsize="small")
            axs.set_xlabel("Sample [n]", fontsize="small")

            if linear_output is not None:
                axs.plot((linear_psi**2 - linear_psi[0]**2), label="Linear", color="tab:blue")
            axs.plot((target_psi**2 - target_psi[0]**2), label="Target", color="tab:green")
            if output is not None:
                axs.plot((psi**2 - psi[0]**2), label="Predicted", color="tab:orange", linestyle="dashed")

            axs.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3, fontsize="small")
            if xlim:
                axs.set_xlim(xlim)
            axs.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
            axs.yaxis.get_offset_text().set_fontsize("small")
            axs.tick_params(axis="both", labelsize="small")
        else:
            fig = None
            axs = None

        return fig, axs

    @staticmethod
    def _plot_nonlinearity(target_nl: nn.Module, nl: nn.Module, target_output: torch.Tensor, num_modes: int):
        dtype = next(nl.parameters()).dtype
        device = next(nl.parameters()).device
        target_nl = target_nl.to(dtype).to(device)

        # generate input points
        q_max = torch.amax(target_output[:, :num_modes, :].abs(), dim=(0, -1)).unsqueeze(0)
        s_points = torch.arange(start=-1, end=1, step=(2 / NUM_POINTS), dtype=dtype, device=device).unsqueeze(-1)
        q_points = s_points * q_max

        # initialise figure
        if hasattr(target_nl, "potential") and hasattr(nl, "potential"):
            plot_potential = (callable(target_nl.potential) and callable(nl.potential))
        else:
            plot_potential = False
        num_plots = 1 + int(plot_potential)
        fig, axs = plt.subplots(nrows=1, ncols=num_plots, layout="constrained", squeeze=False)

        # plot nonlinearity
        axs[0, 0].plot(
            s_points.cpu().numpy(),
            target_nl(q_points)[:, 0].cpu().numpy(),
            label="Target",
            color="tab:green"
        )
        axs[0, 0].plot(
            s_points.cpu().numpy(),
            nl(q_points)[:, 0].cpu().numpy(),
            label="Predicted",
            color="tab:orange",
            linestyle="dashed"
        )
        axs[0, 0].set_xlabel("$s$", fontsize="small")
        axs[0, 0].set_ylabel("$f_1(q(s))$", fontsize="small")
        axs[0, 0].ticklabel_format(style="sci", axis="both", scilimits=(0, 0), useMathText=True)
        axs[0, 0].xaxis.get_offset_text().set_fontsize("small")
        axs[0, 0].yaxis.get_offset_text().set_fontsize("small")
        axs[0, 0].tick_params(axis="both", labelsize="small")
        axs[0, 0].grid()
        fig.legend(loc="outside upper center", ncols=2, fontsize="small")

        # plot potential
        if plot_potential:
            axs[0, 1].plot(
                s_points.cpu().numpy(),
                (
                    target_nl.potential(q_points)
                    - target_nl.potential(torch.zeros_like(q_points))
                ).cpu().numpy(),
                label="Target",
                color="tab:green"
            )
            axs[0, 1].plot(
                s_points.cpu().numpy(),
                (
                    nl.potential(q_points)
                    - nl.potential(torch.zeros_like(q_points))
                ).cpu().numpy(),
                label="Predicted",
                color="tab:orange",
                linestyle="dashed"
            )
            axs[0, 1].set_xlabel("$s$", fontsize="small")
            axs[0, 1].set_ylabel(r"$V(q(s)) - V_0$", fontsize="small")
            axs[0, 1].ticklabel_format(style="sci", axis="both", scilimits=(0, 0), useMathText=True)
            axs[0, 1].xaxis.get_offset_text().set_fontsize("small")
            axs[0, 1].yaxis.get_offset_text().set_fontsize("small")
            axs[0, 1].tick_params(axis="both", labelsize="small")
            axs[0, 1].grid()
            potential0 = nl.potential(torch.zeros((1, num_modes), dtype=dtype, device=device)).item()
            axs[0, 1].set_title(f"$V_0=${potential0:.1E}", size="small")

        return fig, axs

    @property
    def params(self) -> tuple:
        t = (
            self._gen.fs,
            self._gen.dur,
            self._gen.method_name,
            self._gen.num_modes,
            self._gamma_range,
            self._kappa_range,
            self._mu_range,
            self._sigma0_range,
            self._sigma1_range,
            self._xe_range,
            self._xo_range,
            self._exc_amp_range,
            self._exc_dur_range,
            self._exc_type,
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
