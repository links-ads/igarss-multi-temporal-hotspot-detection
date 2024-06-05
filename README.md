# Rapid Wildfire Hotspot Detection Using Self-Supervised Learning On Temporal Remote Sensing Data
Dataset and code for the paper *Rapid Wildfire Hotspot Detection Using Self-Supervised Learning On Temporal Remote Sensing Data* (IGARSS 2024).

[![arXiv](https://img.shields.io/badge/arXiv-2405.20093v1-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2405.20093v1)

https://arxiv.org/abs/2405.20093v1


> [!NOTE]  
> Dataset available at [hf.co/datasets/links-ads/multi-temporal-hotspot-dataset](https://huggingface.co/datasets/links-ads/multi-temporal-hotspot-dataset).

---------------

![Architecture](/resources/Presto_igarss.drawio.png)


## Installation

First, create a python environment. Here we used `python 3.9` and `torch 1.9`, with `CUDA 11.1`.
We suggest creating a python environment, using `venv` or `conda` first.

```
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

## Training
You can launch a training with the following commands:

```console
$ CUDA_VISIBLE_DEVICES=... python src/train.py  --catalog_file_train=... --catalog_file_val=....  --catalog_file_test=... <..args>
```
You can specify the following args:
- batch_size
- max_epochs
- lr
- gpus
- log_dir
- seed
- optimizer
- scheduler
- compute_loss_lc (False if not specified)
- positive_weight_loss_class (default 1)
- lc_loss_weight (default 2)
- mask_strategies (use "random_timesteps")
- mask_ratio (default 0.75)

## Inference

To produce inference maps, run something like the following:

```
$ CUDA_VISIBLE_DEVICES=... python src/test.py --model_checkpoint <args>
```

## Citation
```
@misc{barco2024rapid,
      title={Rapid Wildfire Hotspot Detection Using Self-Supervised Learning on Temporal Remote Sensing Data}, 
      author={Luca Barco and Angelica Urbanelli and Claudio Rossi},
      year={2024},
      eprint={2405.20093},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
