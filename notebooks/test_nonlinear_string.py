# %%
import sys
sys.path.append("../")

# %%
import random
from collections import defaultdict
from pathlib import Path
from glob import glob
from math import log, ceil, sqrt

import torch
import numpy as np
import pandas as pd
from numpy import inf

from src.utils.error import calc_mse, calc_mae, calc_mse_rel, calc_mae_rel
from src.utils.misc import get_window

# %%
%matplotlib ipympl
import matplotlib as mpl
import matplotlib.pyplot as plt
import IPython.display as ipd
import scienceplots
from librosa.display import specshow
from tabulate import tabulate
plt.style.use("default")

# %%
# set parameters
test_dir = "../test/2025_12_07_nonlinear_string_c_grad_net/sav_96000Hz_3sec_75modes_827174835397d508a1a39775b6959823"
slice_dur = 0.1
monitor_key = "mse_rel_w"
modes_to_plot = [0, 29, 59]
freq_range = 6e3

# %%
# get dataset directory
test_dir = Path(test_dir)
dataset_dir = "../data/nonlinear_string/" + test_dir.stem
dataset_dir = Path(dataset_dir)

# %%
# create media directory
media_dir = test_dir / "media"
media_dir.mkdir(exist_ok=True, parents=True)

# %%
# get partition
partition_file = test_dir / "partition.txt"
with open(partition_file, "r") as f:
    partition = f.read().rstrip()
print(f"Partition: {partition}")

# %%
# get files with predicted data
files = glob(test_dir.as_posix() + "/data/*.pt")
indices = [int(Path(f).stem.removesuffix("_pred")) for f in files]
pred_files = {idx: Path(f) for idx, f in zip(indices, files)}

# %%
# get metadata
meta = torch.load(dataset_dir / "meta.pt", weights_only=True)
fs = meta[0]["fs"]
num_modes = meta[0]["num_modes"]
slice_len = round(slice_dur * fs)

# %%
# define data loader
def load_data(idx: int):
    pred = torch.load(pred_files[idx], weights_only=True)
    data = torch.load(dataset_dir / "data" / f"{idx}.pt", weights_only=True)
    data_lin = torch.load(dataset_dir / "data" / f"{idx}_lin.pt", weights_only=True)
    return pred, data, data_lin

# %%
# define data parser
def parse_data(data: dict[str, torch.Tensor]):
    output = data["output"].detach().cpu()
    q, p, _ = torch.tensor_split(output, (num_modes, 2 * num_modes), dim=0)
    w = data["w"].detach().cpu()
    return output, q, p, w

# %%
# calculate metrics
metrics = defaultdict(list)
metrics_per_mode = defaultdict(list)

# define function for metrics calculation
def calc_metrics(
        pred: dict[str, torch.Tensor],
        data: dict[str, torch.Tensor],
        data_lin: dict[str, torch.Tensor],
        slice_len: int = None,
        suffix: str = ""
    ):
    # parse data
    output, q, p, w = parse_data(pred)
    target_output, target_q, target_p, target_w = parse_data(data)
    linear_output, linear_q, linear_p, linear_w = parse_data(data_lin)

    # calculate MSE metrics per mode
    metrics_per_mode["mse"     + suffix].append(calc_mse(       output[:, :slice_len], target_output[:, :slice_len], dim=-1))
    metrics_per_mode["mse_lin" + suffix].append(calc_mse(linear_output[:, :slice_len], target_output[:, :slice_len], dim=-1))

    # calculate scalar MSE metrics
    metrics["mse"     + suffix].append(calc_mse(       output[:, :slice_len], target_output[:, :slice_len]).item())
    metrics["mse_lin" + suffix].append(calc_mse(linear_output[:, :slice_len], target_output[:, :slice_len]).item())

    # calculate scalar MAE metrics
    metrics["mae"     + suffix].append(calc_mae(       output[:, :slice_len], target_output[:, :slice_len]).item())
    metrics["mae_lin" + suffix].append(calc_mae(linear_output[:, :slice_len], target_output[:, :slice_len]).item())

    # calculate scalar relative MSE metrics
    metrics["mse_rel"       + suffix].append(calc_mse_rel(       output[:, :slice_len], target_output[:, :slice_len]).item())
    metrics["mse_rel_lin"   + suffix].append(calc_mse_rel(linear_output[:, :slice_len], target_output[:, :slice_len]).item())
    metrics["mse_rel_q"     + suffix].append(calc_mse_rel(       q[:, :slice_len], target_q[:, :slice_len]).item())
    metrics["mse_rel_q_lin" + suffix].append(calc_mse_rel(linear_q[:, :slice_len], target_q[:, :slice_len]).item())
    metrics["mse_rel_p"     + suffix].append(calc_mse_rel(       p[:, :slice_len], target_p[:, :slice_len]).item())
    metrics["mse_rel_p_lin" + suffix].append(calc_mse_rel(linear_p[:, :slice_len], target_p[:, :slice_len]).item())
    metrics["mse_rel_w"     + suffix].append(calc_mse_rel(       w[:slice_len], target_w[:slice_len]).item())
    metrics["mse_rel_w_lin" + suffix].append(calc_mse_rel(linear_w[:slice_len], target_w[:slice_len]).item())

    # calculate scalar relative MAE metrics
    metrics["mae_rel"       + suffix].append(calc_mae_rel(       output[:, :slice_len], target_output[:, :slice_len]).item())
    metrics["mae_rel_lin"   + suffix].append(calc_mae_rel(linear_output[:, :slice_len], target_output[:, :slice_len]).item())
    metrics["mae_rel_q"     + suffix].append(calc_mae_rel(       q[:, :slice_len], target_q[:, :slice_len]).item())
    metrics["mae_rel_q_lin" + suffix].append(calc_mae_rel(linear_q[:, :slice_len], target_q[:, :slice_len]).item())
    metrics["mae_rel_p"     + suffix].append(calc_mae_rel(       p[:, :slice_len], target_p[:, :slice_len]).item())
    metrics["mae_rel_p_lin" + suffix].append(calc_mae_rel(linear_p[:, :slice_len], target_p[:, :slice_len]).item())
    metrics["mae_rel_w"     + suffix].append(calc_mae_rel(       w[:slice_len], target_w[:slice_len]).item())
    metrics["mae_rel_w_lin" + suffix].append(calc_mae_rel(linear_w[:slice_len], target_w[:slice_len]).item())

# main loop
for idx in indices:
    metrics["idx"].append(idx)
    pred, data, data_lin = load_data(idx)
    calc_metrics(pred, data, data_lin)
    calc_metrics(pred, data, data_lin, slice_len=slice_len, suffix="_slice")

# convert the `metrics` dictionary to a `DataFrame`
df = pd.DataFrame.from_dict(metrics, orient="columns")
df = df.set_index("idx")

# convert each metric in the `metrics_per_mode` dictionary to a tensor
for key in metrics_per_mode.keys():
    metrics_per_mode[key] = torch.stack(metrics_per_mode[key], dim=0)

# %%
# get metrics table
metrics_table = [
    [key, df[key].mean(), df[key].std(), df[key].median()]
    for key in df
]
headers = ["Metric", "Mean", "Std", "Median"]
metrics_str = tabulate(metrics_table, headers=headers, floatfmt=".2e")
metrics_str = metrics_str + f"\n\n* slice duration is {slice_dur:.2f} seconds"

# %%
# get monitor table
monitor_max_idx = int(df[monitor_key].idxmax())
monitor_min_idx = int(df[monitor_key].idxmin())
monitor_table = [[
    monitor_key,
    df[monitor_key].max(),
    monitor_max_idx,
    df[monitor_key].min(),
    monitor_min_idx
]]
headers = ["Monitor", "Max", "Max Idx", "Min", "Min Idx"]
monitor_str = tabulate(monitor_table, headers=headers, floatfmt=".2e")

# %%
# display metrics
combined_str = metrics_str + "\n\n" + monitor_str
print(combined_str)

# %%
# save metrics to file
metrics_file = media_dir / f"metrics_{partition}.txt"
with open(metrics_file, "w") as f:
    f.write(combined_str)

# %%
# define metric per mode plotter
def plot_metric_per_mode(metric: torch.Tensor, linear_metric: torch.Tensor, name: str, plot_velocity: bool = True):
    ncols = 1 + int(plot_velocity)
    fig, axs = plt.subplots(nrows=1, ncols=ncols, layout="constrained", squeeze=False)

    metric = metric.detach().cpu().numpy()
    linear_metric = linear_metric.detach().cpu().numpy()

    modes = np.arange(start=1, stop=(num_modes + 1), step=1)
    xticks = np.arange(start=0, stop=(num_modes + 1), step=(num_modes // 5))
    xticks[0] = 1

    axs[0, 0].semilogy(
        modes,
        linear_metric[:num_modes],
        "o",
        fillstyle="none",
        markeredgewidth=0.5,
        markersize=3,
        label="Linear",
        color="b"
    )
    axs[0, 0].semilogy(
        modes,
        metric[:num_modes],
        "x",
        markeredgewidth=0.5,
        markersize=3,
        label="Predicted",
        color="r"
    )
    axs[0, 0].set_xlabel("Mode")
    axs[0, 0].set_ylabel(f"{name} for displacement")
    axs[0, 0].set_xlim(1, num_modes)
    axs[0, 0].set_xticks(xticks)
    axs[0, 0].grid()

    fig.legend(loc="outside upper center", ncols=2)

    if plot_velocity:
        axs[0, 1].semilogy(
            modes,
            linear_metric[num_modes:2*num_modes],
            "o",
            fillstyle="none",
            markeredgewidth=0.5,
            markersize=3,
            label="Linear",
            color="b"
        )
        axs[0, 1].semilogy(
            modes,
            metric[num_modes:2*num_modes],
            "x",
            markeredgewidth=0.5,
            markersize=3,
            label="Predicted",
            color="r"
        )
        axs[0, 1].set_xlabel("Mode")
        axs[0, 1].set_ylabel(f"{name} for velocity")
        axs[0, 1].set_xlim(1, num_modes)
        axs[0, 1].set_xticks(xticks)
        axs[0, 1].grid()

    return fig, axs

# %%
# plot MSE per mode
fig, _ = plot_metric_per_mode(
    metrics_per_mode["mse_slice"].mean(dim=0),
    metrics_per_mode["mse_lin_slice"].mean(dim=0),
    name="MSE"
)

# %%
# save MSE per mode figure to file
fig.savefig(media_dir / f"mse_per_mode_slice_{partition}.png", format="png", bbox_inches="tight")

# %%
# plot median MSE per mode
fig, _ = plot_metric_per_mode(
    metrics_per_mode["mse_slice"].median(dim=0).values,
    metrics_per_mode["mse_lin_slice"].median(dim=0).values,
    name="Median MSE"
)

# %%
# save median MSE per mode figure to file
fig.savefig(media_dir / f"median_mse_per_mode_slice_{partition}.png", format="png", bbox_inches="tight")

# %%
# define waveform plotter
def plot_wave(
        target_w: torch.Tensor,
        w: torch.Tensor,
        linear_w: torch.Tensor,
        fs: int,
        plot_legend: bool = True,
        axs = None
    ):
    if axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=1, layout="constrained")
    else:
        fig = axs.get_figure()

    target_w = target_w.detach().cpu().numpy()
    w = w.detach().cpu().numpy()
    linear_w = linear_w.detach().cpu().numpy()

    num_samples = target_w.shape[0]
    t_points = np.arange(start=0, stop=num_samples, step=1) / fs

    axs.plot(t_points, linear_w, label="Linear", color="b", linestyle="dotted")
    axs.plot(t_points, target_w, label="Target", color="k")
    axs.plot(t_points, w, label="Predicted", color="r", linestyle="dashed")
    axs.set_xlabel("Time [sec]")
    axs.set_ylabel("Output")
    axs.ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
    axs.grid()
    if plot_legend:
        axs.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3)

    return fig, axs

# %%
# define spectrogram plotter
LOG_TICKS = [62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
LOG_LABELS = ["62.5", "125", "250", "500", "1k", "2k", "4k", "8k", "16k"]

def plot_spec(
        input: torch.Tensor,
        fs: int,
        win_name: str = "blackmanharris",
        win_len: int = 4096,
        overlap: float = 0.9,
        dynamic_range: float = 120,
        freq_range: float = 20e3,
        freq_axis: str = "linear",
        plot_bar: bool = False,
        axs = None
    ):
    if axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=1, layout="constrained")
    else:
        fig = axs.get_figure()

    win = get_window(win_name, win_len, sym=False)
    fft_len = 2**(ceil(log(win_len) / log(2)))
    hop_len = round(win_len * (1.0 - overlap))

    stft = torch.stft(
        input,
        n_fft=fft_len,
        hop_length=hop_len,
        win_length=win_len,
        window=win,
        center=False,
        return_complex=True
    )
    mag = torch.abs(stft)
    mag_db = 20.0 * torch.log10(mag / torch.max(mag))
    mag_db = mag_db.detach().cpu().numpy()

    num_frames = mag.shape[1]
    t_points = np.arange(start=0, stop=num_frames, step=1) / fs * hop_len
    freqs = np.arange(start=0, stop=((fft_len // 2) + 1), step=1) * fs / fft_len

    img = specshow(
        mag_db,
        x_coords=t_points,
        y_coords=freqs,
        x_axis="time",
        y_axis=freq_axis,
        cmap="magma",
        vmin=-dynamic_range,
        vmax=0,
        ax=axs
    )
    axs.set_ylim(0, freq_range)
    axs.set_xlabel("Time [sec]")
    axs.set_ylabel("Frequency [Hz]")
    if plot_bar:
        fig.colorbar(img, label="[dB]")
    if freq_axis == "log":
        axs.tick_params(left=False)
        axs.set_yticks(LOG_TICKS, LOG_LABELS)

    return fig, axs

# %%
# define displacement plotter
def plot_displacement(
        target_q: torch.Tensor,
        q: torch.Tensor,
        linear_q: torch.Tensor,
        fs: int,
        modes: list[int]
    ):
    fig, axs = plt.subplots(nrows=len(modes), ncols=1, layout="constrained", squeeze=False)

    target_q = target_q.detach().cpu().numpy()
    q = q.detach().cpu().numpy()
    linear_q = linear_q.detach().cpu().numpy()

    num_samples = target_q.shape[-1]
    t_points = np.arange(start=0, stop=num_samples, step=1) / fs

    for i, mode in enumerate(modes):
        axs[i, 0].plot(t_points, linear_q[mode, :], label="Linear", color="b", linestyle="dotted")
        axs[i, 0].plot(t_points, target_q[mode, :], label="Target", color="k")
        axs[i, 0].plot(t_points, q[mode, :], label="Predicted", color="r", linestyle="dashed")
        axs[i, 0].set_ylabel(f"Displacement {mode + 1}")
        axs[i, 0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
        axs[i, 0].grid()
    axs[0, 0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3)
    axs[i, 0].set_xlabel("Time [sec]")

    return fig, axs

# %%
# define velocity plotter
def plot_velocity(
        target_p: torch.Tensor,
        p: torch.Tensor,
        linear_p: torch.Tensor,
        fs: int,
        modes: list[int]
    ):
    fig, axs = plt.subplots(nrows=len(modes), ncols=1, layout="constrained", squeeze=False)

    target_p = target_p.detach().cpu().numpy()
    p = p.detach().cpu().numpy()
    linear_p = linear_p.detach().cpu().numpy()

    num_samples = target_p.shape[-1]
    t_points = np.arange(start=0, stop=num_samples, step=1) / fs

    for i, mode in enumerate(modes):
        axs[i, 0].plot(t_points, linear_p[mode, :], label="Linear", color="b", linestyle="dotted")
        axs[i, 0].plot(t_points, target_p[mode, :], label="Target", color="k")
        axs[i, 0].plot(t_points, p[mode, :], label="Predicted", color="r", linestyle="dashed")
        axs[i, 0].set_ylabel(f"Velocity {mode + 1}")
        axs[i, 0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0), useMathText=True)
        axs[i, 0].grid()
    axs[0, 0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3)
    axs[i, 0].set_xlabel("Time [sec]")

    return fig, axs

# %%
# define modal shape
def Phi(x: torch.Tensor) -> torch.Tensor:
    beta = torch.arange(start=1, end=(num_modes + 1), step=1, dtype=x.dtype) * torch.pi
    return sqrt(2) * torch.sin(torch.outer(x, beta))

# %%
# define displacement grid plotter
def plot_displacement_grid(
        q: torch.Tensor,
        fs: int,
        target_q: torch.Tensor = None,
        num_points: int = 100,
        plot_bar: bool = False,
        axs = None
    ):
    plot_diff = (target_q is not None)

    if axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=1, layout="constrained")
    else:
        fig = axs.get_figure()

    x = torch.arange(start=0, end=num_points, dtype=q.dtype) / num_points
    Phi_x = Phi(x)
    u = torch.matmul(Phi_x, q).detach().cpu().numpy().T
    if plot_diff:
        # copute the relative absolute difference
        target_u = torch.matmul(Phi_x, target_q).detach().cpu().numpy().T
        diff = np.abs(u - target_u) / np.max(np.abs(target_u))
        max_diff = np.max(diff)

        # create colormap normalisation
        def _forward(x):
            return np.tanh(x / max_diff)
        def _inverse(y):
            return max_diff * np.arctanh(y)
        norm = mpl.colors.FuncNorm((_forward, _inverse), vmin=0, vmax=max_diff)

    num_samples = q.shape[-1]
    dur = num_samples / fs

    img = axs.imshow(
        (diff if plot_diff else u),
        cmap=("magma" if plot_diff else "bwr"),
        aspect="auto",
        interpolation="none",
        origin="upper",
        extent=(x[0].item(), x[-1].item(), dur, 0),
        norm=(norm if plot_diff else None)
    )
    axs.set_xlabel("Position")
    axs.set_ylabel("Time [sec]")
    if plot_bar:
        fig.colorbar(img)

    return fig, axs

# %%
# define instance tester
def test_instance(idx: int, title: str = None):
    if title:
        print(f"------ {title} ------\n")

    # load data
    pred, data, data_lin = load_data(idx)

    # print metrics
    try:
        headers = ["Metric", "Value"]
        print(tabulate(df.loc[idx].to_frame(), headers=headers, floatfmt=".2e") + "\n")
    except:
        pass

    # parse data
    _, q, p, w = parse_data(pred)
    _, target_q, target_p, target_w = parse_data(data)
    _, linear_q, linear_p, linear_w = parse_data(data_lin)

    # display audio
    print("Linear waveform")
    ipd.display(ipd.Audio(linear_w, rate=fs))
    print("Target waveform")
    ipd.display(ipd.Audio(target_w, rate=fs))
    print("Predicted waveform")
    ipd.display(ipd.Audio(w, rate=fs))

    # plot displacement grid
    fig, axs = plt.subplots(1, 3, layout="constrained")
    plot_displacement_grid(target_q[:, :slice_len], fs=fs, axs=axs[0])
    plot_displacement_grid(q[:, :slice_len], fs=fs, axs=axs[1])
    plot_displacement_grid(q[:, :slice_len], target_q=target_q[:, :slice_len], fs=fs, plot_bar=True, axs=axs[2])
    axs[0].set_title("Target")
    axs[1].set_title("Predicted")
    axs[1].set_yticklabels([])
    axs[1].set_ylabel(None)
    axs[2].set_title("Rel. Abs. Error")
    axs[2].set_yticklabels([])
    axs[2].set_ylabel(None)
    fig.suptitle(title)
    plt.show()

    # plot time domain
    fig, _ = plot_wave(target_w, w, linear_w, fs=fs)
    fig.suptitle(title)
    plt.show()
    fig, _ = plot_displacement(target_q, q, linear_q, fs=fs, modes=modes_to_plot)
    fig.suptitle(title)
    plt.show()
    fig, _ = plot_velocity(target_p, p, linear_p, fs=fs, modes=modes_to_plot)
    fig.suptitle(title)
    plt.show()

    # plot spectrograms
    fig, axs = plt.subplots(1, 2, layout="constrained")
    plot_spec(target_w, fs=fs, freq_range=freq_range, axs=axs[0])
    plot_spec(w, fs=fs, freq_range=freq_range, axs=axs[1])
    axs[0].set_title("Target")
    axs[1].set_title("Predicted")
    axs[1].set_yticklabels([])
    axs[1].set_ylabel(None)
    fig.suptitle(title)
    plt.show()

# %%
# test instance with max montor
test_instance(monitor_max_idx, f"Instance {monitor_max_idx} (max monitor)")

# %%
# test instance with min montor
test_instance(monitor_min_idx, f"Instance {monitor_min_idx} (min monitor)")

# %%
# test random instance
random_idx = random.choice(indices)
test_instance(random_idx, f"Instance {random_idx}")

# %% [markdown]
# ## JAES Figures

# %%
# define figure saving
JAES_TEXT_WIDTH = 6.72
JAES_COLUMN_WIDTH = 3.23
JAES_TEXT_HEIGHT = 9.74
GOLDEN_RATIO = (5**0.5 - 1) / 2

def save_fig(fig, name: str, width: float, height: float = None, dpi: float = 600, format: str = "pdf"):
    if height is None:
        height = width * GOLDEN_RATIO
    fig.set_size_inches(width, height)
    fig.savefig(media_dir / (name + "." + format), format=format, bbox_inches="tight", dpi=dpi)

# %%
# set figure style for a paper
plt.style.use(["science", "ieee"])

# %%
# plot MSE per mode (displacement only)
fig, _ = plot_metric_per_mode(
    metrics_per_mode["mse_slice"].mean(dim=0),
    metrics_per_mode["mse_lin_slice"].mean(dim=0),
    name="MSE",
    plot_velocity=False
)
save_fig(fig, f"fig_2_mse_q_per_mode_slice_{partition}", JAES_COLUMN_WIDTH, 2.8, format="eps")

# %%
# get data with max monitor
pred, data, data_lin = load_data(monitor_max_idx)
output, q, p, w = parse_data(pred)
target_output, target_q, target_p, target_w = parse_data(data)
linear_output, linear_q, linear_p, linear_w = parse_data(data_lin)

# %%
# plot waveform
fig, axs = plt.subplots(nrows=1, ncols=2, layout="constrained")
plot_wave(target_w, w, linear_w, fs=fs, plot_legend=False, axs=axs[0])
fig.legend(loc="outside upper center", ncols=3)
plot_wave(target_w, w, linear_w, fs=fs, plot_legend=False, axs=axs[1])
axs[0].set_xlim(0, 25e-3)
xticks_labels = np.arange(0, 26, 5)
xticks = xticks_labels * 1e-3
axs[0].set_xticks(xticks, xticks_labels)
axs[0].set_xlabel("Time [ms]")
axs[1].set_xlim(505e-3, 505e-3 + 25e-3)
xticks_labels = np.arange(505, 531, 5)
xticks = xticks_labels * 1e-3
axs[1].set_xticks(xticks, xticks_labels)
axs[1].set_xlabel("Time [ms]")
axs[1].set_yticklabels([])
axs[1].set_ylabel(None)
save_fig(fig, f"fig_3_{monitor_max_idx}_wave", JAES_TEXT_WIDTH, 2.8, format="eps")

# %%
# plot displacement
NUM_PERIODS = 20
MODES = [0, 29, 59]
omega = meta[monitor_max_idx]["omega"]
exc_dur = meta[monitor_max_idx]["exc_dur"]
fig, axs = plot_displacement(target_q, q, linear_q, fs=fs, modes=MODES)
axs[0, 0].set_xlim(exc_dur, exc_dur + 2.0 * np.pi * NUM_PERIODS / omega[MODES[0]].item())
axs[0, 0].set_ylim(-1.5e-2, 1.5e-2)
xticks_labels = np.arange(10, 171, 20)
xticks = xticks_labels * 1e-3
axs[0, 0].set_xticks(xticks, xticks_labels)
axs[1, 0].set_yticks([-1e-2, 0, 1e-2])
axs[0, 0].set_ylabel("1st mode")
axs[1, 0].set_xlim(exc_dur, exc_dur + 2.0 * np.pi * NUM_PERIODS / omega[MODES[1]].item())
axs[1, 0].set_ylim(-2.5e-4, 2.5e-4)
axs[1, 0].set_xticks([1e-3, 1.5e-3, 2e-3, 2.5e-3, 3e-3, 3.5e-3, 4e-3, 4.5e-3, 5e-3, 5.5e-3], ["1", "1.5", "2", "2.5", "3", "3.5", "4", "4.5", "5", "5.5"])
axs[1, 0].set_yticks([-2e-4, 0, 2e-4])
axs[1, 0].set_ylabel("30th mode")
axs[2, 0].set_xlim(exc_dur, exc_dur + 2.0 * np.pi * NUM_PERIODS / omega[MODES[2]].item())
axs[2, 0].set_ylim(-2.5e-5, 2.5e-5)
axs[2, 0].set_xticks([0.75e-3, 1e-3, 1.25e-3, 1.5e-3, 1.75e-3, 2e-3, 2.25e-3, 2.5e-3, 2.75e-3], ["0.75", "1", "1.25", "1.5", "1.75", "2", "2.25", "2.5", "2.75"])
axs[2, 0].set_yticks([-2e-5, 0, 2e-5])
axs[2, 0].set_ylabel("60th mode")
axs[2, 0].set_xlabel("Time [ms]")
fig.align_ylabels(axs)
save_fig(fig, f"fig_4_{monitor_max_idx}_displacement", JAES_TEXT_WIDTH, 2.8, format="eps")

# %%
# plot displacement grid
fig, axs = plt.subplots(1, 3, layout="constrained")
plot_displacement_grid(target_q[:, :slice_len], fs=fs, axs=axs[0], num_points=1000)
plot_displacement_grid(q[:, :slice_len], fs=fs, axs=axs[1], num_points=1000)
plot_displacement_grid(q[:, :slice_len], target_q=target_q[:, :slice_len], fs=fs, plot_bar=True, axs=axs[2], num_points=1000)
axs[0].set_title("Target")
yticks_labels = np.arange(0, 101, 20)
yticks = yticks_labels * 1e-3
axs[0].set_yticks(yticks, yticks_labels)
axs[0].set_ylabel("Time [ms]")
axs[0].set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
axs[1].set_title("Predicted")
axs[1].set_yticklabels([])
axs[1].set_ylabel(None)
axs[1].set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
axs[2].set_title("Rel. Abs. Error")
axs[2].set_yticklabels([])
axs[2].set_ylabel(None)
axs[2].set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
cbar = axs[2].images[-1].colorbar
cbar.ax.set_yticks([0, 1e-2, 2e-2, 3e-2], ["0", "0.01", "0.02", "0.03"])
save_fig(fig, f"fig_1_{monitor_max_idx}_displacement_grid", JAES_COLUMN_WIDTH, 2.8, format="eps")

# %%
# plot spectrogram for the most "nonlinear" waveform
nl_max_idx = int(df["mse_rel_w_lin_slice"].idxmax())
pred, data, _ = load_data(nl_max_idx)
_, _, _, w = parse_data(pred)
_, _, _, target_w = parse_data(data)

fig, axs = plt.subplots(nrows=1, ncols=2, layout="constrained")
plot_spec(target_w, fs=fs, freq_range=1e4, axs=axs[0])
plot_spec(w, fs=fs, freq_range=1e4, plot_bar=True, axs=axs[1])
axs[0].set_yticks([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000], ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
axs[0].set_ylabel("Frequency [kHz]")
axs[0].set_title("Target output")
axs[1].set_yticklabels([])
axs[1].set_ylabel(None)
axs[1].set_title("Predicted output")
save_fig(fig, f"fig_5_{nl_max_idx}_spec", JAES_TEXT_WIDTH, 2.8, format="eps")

# %%
