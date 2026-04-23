<h1>MMBench & M3PO</span></h1>

Official code repository for the paper

Vendored in this Isaac Lab workspace as **M3PO**, based on the upstream **NEWT** release by Nicklas Hansen et al.

[Learning Massively Multitask World Models for Continuous Control](https://www.nicklashansen.com/NewtWM)

[Nicklas Hansen](https://nicklashansen.github.io), [Hao Su](https://cseweb.ucsd.edu/~haosu)\*, [Xiaolong Wang](https://xiaolonw.github.io)\* (UC San Diego)</br>

<img src="assets/0.gif" width="12.5%"><img src="assets/1.gif" width="12.5%"><img src="assets/2.gif" width="12.5%"><img src="assets/3.gif" width="12.5%"><img src="assets/4.gif" width="12.5%"><img src="assets/5.gif" width="12.5%"><img src="assets/6.gif" width="12.5%"><img src="assets/7.gif" width="12.5%"></br>

[[Website]](https://www.nicklashansen.com/NewtWM) [[Paper]](https://www.nicklashansen.com/NewtWM/newt.pdf) [[Models]](https://huggingface.co/nicklashansen/newt) [[Dataset]](https://huggingface.co/datasets/nicklashansen/mmbench)

----

**Early access (Nov 2025):** This is an early code release; we will continue to add features and code improvements in the coming months, but wanted to make the code available to the public as soon as possible. Please let us know if you have any questions or issues by opening an issue on GitHub!

----


## MMBench

MMBench contains a total of **200** unique continuous control tasks for training of massively multitask RL policies. The task suite consists of 159 existing tasks proposed in previous work, 22 new tasks and task variants for these existing domains, as well as 19 entirely new arcade-style tasks that we dub *MiniArcade*. MMBench tasks span multiple domains and embodiments, and each task comes with language instructions, demonstrations, and optionally image observations, enabling research on both multitask pretraining, offline-to-online RL, and RL from scratch.

<img src="assets/0.png" width="100%" style="max-width: 640px"><br/>


## M3PO (Vendored NEWT)

M3PO in this workspace is a vendored adaptation of Newt, a language-conditioned multitask world model based on [TD-MPC2](https://www.tdmpc2.com). The upstream algorithm first pretrains on demonstrations to acquire task-aware representations and action priors, and then jointly optimizes with online interaction across all tasks. To extend TD-MPC2 to the massively multitask online setting, the paper proposes a refined architecture, model-based pretraining on the available demonstrations, additional action supervision in RL policy updates, and a drastically accelerated training pipeline.

<img src="assets/1.png" width="100%" style="max-width: 640px"><br/>

----

## Getting started

We provide three options for getting started with our codebase: (1) local installation using `conda`, (2) building a `docker` image using our provided `Dockerfile`, or (3) using our prebuilt `docker` image hosted on Docker Hub. Note however that option (3) may not always be available due to rate limits.

First, we recommend downloading required ManiSkill assets from huggingface by running

```
wget https://huggingface.co/datasets/nicklashansen/mmbench/resolve/main/maniskill.tar.gz
tar -xvf maniskill.tar.gz && mv .maniskill ~ && rm maniskill.tar.gz
```

which will create a `.maniskill` folder in your home directory. This is the default location where the ManiSkill environments look for assets. You can also specify a different location by setting the `MANISKILL_ASSET_DIR` environment variable.

Then, choose one of the following installation options:

### Option 1: Local installation with conda

Most dependencies can be installed via `conda`. We provide an `environment.yml` file for easy installation. You can create a new conda environment and install all dependencies by running

```
conda env create -f docker/environment.yaml
conda activate m3po
pip install --no-cache-dir 'ale_py==0.10'
```

Finally, we recommend setting the `MS_SKIP_ASSET_DOWNLOAD_PROMPT` environment variable to `1` to avoid prompts from ManiSkill about downloading assets during runtime (assuming you have already downloaded the assets as described above):

```
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
```


### Option 2: Building a docker image

We provide a `Dockerfile` for easy installation. You can build the docker image by first moving your downloaded `.maniskill` asset directory to `docker/.maniskill` and then running

```
cd docker && docker build . -t <user>/m3po:1.0.0
```

This docker image contains all dependencies needed for running MMBench and Newt.

### Option 3: Using a prebuilt docker image

We provide a prebuilt docker image on Docker Hub that you can use directly without having to build the image yourself. You can pull the image by running

```
docker pull nicklashansen/newt:1.0.0
```

This option may not always be available due to Docker Hub rate limits, but can be a convenient way to get started quickly or for debugging purposes.

----

## Example usage

### Training

Agents can be trained by running the `train.py` script. Below are some example commands:

```
$ python train.py    # <-- a 20M parameter agent trained on all 200 MMBench tasks
$ python train.py model_size=XL    # <-- a 80M parameter agent
$ python train.py model_size=B task=walker-walk   # <-- a 5M parameter single-task agent
$ python train.py obs=rgb    # <-- a 20M parameter agent trained with state+RGB observations
$ python train.py checkpoint=<path>/<to>/<checkpoint>.pt    # <-- resume training from checkpoint
```

We recommend using default hyperparameters, including the default model size of 20M parameters (`model_size=L`) for multitask experiments. For single-task experiments we recommend `model_size=B`. See `config.py` for a full list of arguments.


### Loading model checkpoints

A model checkpoint can be loaded from a local path specified by the `checkpoint` argument; this can be either a checkpoint saved during training or one of our provided model checkpoints.

Our provided checkpoints can be downloaded in two ways: (1) manually navigating to our [Hugging Face Models page](https://huggingface.co/nicklashansen/newt) and downloading the desired checkpoint file, or (2) using the provided `download_checkpoints.py` script. To use the script, first make sure you have `huggingface_hub` installed:

```
$ pip install -U huggingface_hub
```

Once installed, you can download any (or all) of our provided model checkpoints by running one of the following commands:

```
$ python download_checkpoints.py --filename "walker-walk" --cache-dir="./checkpoints"   # <-- download single checkpoint
$ python download_checkpoints.py --all --cache-dir="./checkpoints"   # <-- download all checkpoints
```

Multitask checkpoints use a `soup` prefix in the filename, and model size is also specified in the filename (`S=2M`, `B=5M`, `L=20M`, `XL=80M`). You will need to use `model_size=B` when loading single-task checkpoints. We are actively working on better support for model loading and finetuning, so check back soon for updates!

### Generating demonstrations

You can generate demonstrations using a trained agent by running the `generate_demos.py` script. You will need to specify your checkpoint directory (`CHECKPOINT_PATH`) directly in the script, as well as `data_dir` (where to save the demos), `+num_demos` (number of successful demos to collect), and `task` (task to generate demos for). Below is an example command:

```
$ python generate_demos.py task=walker-walk +num_demos=10 data_dir=<path>/<to>/<data>
```

The script assumes that the agent used for generating demos is a single-task agent trained with default hyperparameters (e.g., any of our provided checkpoints).

### Results

We provide precomputed results for Newt and baselines in the `csv/` directory. For your convenience, we report normalized scores at three different levels of aggregation: `avg` (average across all 200 tasks), `by_domain` (average within each domain), and `by_task` (individual task scores). You can easily load these results using `pandas`:

```python
import pandas as pd
results = pd.read_csv("./csv/newt/newt_avg.csv")  # <-- average score across all tasks
```

----

## Citation

If you find our work useful, please consider citing our paper as follows:

```
@misc{Hansen2025Newt,
	title={Learning Massively Multitask World Models for Continuous Control}, 
	author={Nicklas Hansen and Hao Su and Xiaolong Wang},
	year={2025},
	eprint={2511.19584},
	archivePrefix={arXiv},
	primaryClass={cs.LG},
	url={https://arxiv.org/abs/2511.19584}, 
}
```

----

## Contributing

You are very welcome to contribute to this project. Feel free to open an issue or pull request if you have any suggestions or bug reports, but please review our [guidelines](CONTRIBUTING.md) first. Our goal is to build a codebase that can easily be extended to new environments and tasks, and we would love to hear about your experience!

----

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. Note that the repository relies on third-party code, which is subject to their respective licenses.
