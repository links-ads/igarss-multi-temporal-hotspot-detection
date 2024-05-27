# RAPID WILDFIRE HOTSPOT DETECTION USING SELF-SUPERVISED LEARNING ON TEMPORAL REMOTE SENSING DATA
Dataset and code for the paper *RAPID WILDFIRE HOTSPOT DETECTION USING SELF-SUPERVISED LEARNING ON TEMPORAL REMOTE SENSING DATA* (IGARSS 2024).

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2306.16252-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2306.16252) -->

> [!NOTE]  
> Dataset available at [hf.co/datasets/links-ads/spada-dataset](https://huggingface.co/datasets/links-ads/multi-temporal-hotspot-dataset).

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
$ CUDA_VISIBLE_DEVICES=... python src/mthd/train.py  --catalog_file_train=... --catalog_file_val=....  --catalog_file_test=... <..args>
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
$ CUDA_VISIBLE_DEVICES=... python src/mthd/test.py --model_checkpoint <args>
```

## Citing this work
```bibtex
@inproceedings{galatola2023land,
  title={Land Cover Segmentation with Sparse Annotations from Sentinel-2 Imagery},
  author={Galatola, Marco and Arnaudo, Edoardo and Barco, Luca and Rossi, Claudio and Dominici, Fabrizio},
  booktitle={IGARSS 2023-2023 IEEE International Geoscience and Remote Sensing Symposium},
  pages={6952--6955},
  year={2023},
  organization={IEEE}
}
```
