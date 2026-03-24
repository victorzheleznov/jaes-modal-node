<h2 style="font-size: 1.5em" align="center">
  Stable Differentiable Modal Synthesis for Learning Nonlinear Dynamics
</h2>

<p style="font-size: 1.0em" align="center">
  Victor Zheleznov, Stefan Bilbao, Alec Wright and Simon King
</p>

<p style="font-size: 1.0em" align="center">
  Accompanying repository for the JAES paper
</p>

<div align="center">

  [![Sound Examples](https://img.shields.io/badge/Sound_Examples-blue)](https://victorzheleznov.github.io/jaes-modal-node/)
  [![arXiv](https://img.shields.io/badge/arXiv-2601.10453-b31b1b.svg)](https://arxiv.org/abs/2601.10453)
  
</div>



## Repository Contents

`audio/` includes all sound examples for the datasets used in the paper. Some of these sound examples are presented on the accompanying web-page.

`cfg/` includes configuration files for experiments.

`src/` includes source code for datasets, generators, models, solvers and other utils.

`notebooks/` includes notebooks for analysis.

`out/` includes configurations and checkpoints for trained models.

`generate.py` is the script for datasets generation.

`train.py` is the script for training the model.

`test.py` is the script for testing the trained model.



## Instructions

### Environment Setup

[Python 3.11.9](https://www.python.org/downloads/release/python-3119/) was used for simulations.
The required packages are provided in the `requirements.txt` file. To setup the environment, use:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data Generation (Optional)

The `cfg/datasets/dataset` folder includes configuration files for individual datasets. The `generate.py` scipt is provided to generate the data:
```
python3 -m generate --config-name=nonlinear_string_dataset_B1_F2
```

To generate training, validation and test datasets used in the paper, skip this step as all three datasets will be automatically generated on the first run of the `train.py` script. Otherwise, you will need to specify correct seeds and dataset sizes for the `generate.py` script as described in the `cfg/datasets/nonlinear_string_datasets.yaml` configuration file.

The generated datasets will be saved within the `data/nonlinear_string` folder. Please refer to the paper for more details on the chosen parameters for the nonlinear string datasets.

### Training

The `cfg` folder includes the `nonlinear_string.yaml` configuration file which matches the experimental
setup described in the paper. The `train.py` script is designed for a single-node multi-GPU training using the [`DistributedDataParallel` module ](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) in PyTorch. Thus, the [`torchrun` command](https://docs.pytorch.org/docs/stable/elastic/run.html) is needed to run this script and train the model:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=gpu train.py --config-name=nonlinear_string
```
It is still possible to train the model on a single GPU by specifying `CUDA_VISIBLE_DEVICES=0` in the above command.

If you have a [WandB](https://wandb.ai/) account, you can also enable online logging by appending:
```
writer.mode=online writer.project=project_name
```

The training run will be saved within the `out/nonlinear_string` folder by default. This folder name is specified in the `save_dir` parameter of the configuration file and can be changed if needed.
Checkpoint with the lowest validation loss `checkpoint_best.pt` and
checkpoint from the last epoch `checkpoint_*.pt` will be saved.

### Testing

 To test the trained model using the `test.py` script, you will need to:
- specify the folder with the saved training run as a configuration path;
- disable the trajectory slicing (implemented in the `src/datasets/slice_collator.py` file) which is used during training as a teacher forcing technique;
- increase the batch size since the model inference requires significantly less memory.

The `test.py` script also relies on the `DistributedDataParallel` module and requiers a similar procedure as the `train.py` script to run. As a result, the testing command is constructed as:
```
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=gpu test.py --config-path=out/nonlinear_string collate_func=null batch_size=5
```

Computed model predictions for training, validation and test datasets will be stored within the `test/nonlinear_string` folder. These predictions can be analysed in the provided `notebooks/test_nonlinear_string.py` notebook which was used to calculate the metrics and plot the figures for the paper.

### Results Reproduction

Even with fixed seeds, randomised values depend on the specific device. Thus, your generated datasets would probably be different from the ones used in the paper and showcased on the accompanying web-page. Due to the large size (around 77 GB), the datasets used in the paper can be obtained by a personal request to [v.zheleznov@ed.ac.uk](mailto:v.zheleznov@ed.ac.uk?subject=Datasets%20Request). Even though the training process also involves randomisation, it should be possible to closely reproduce results in the paper using these datasets.
