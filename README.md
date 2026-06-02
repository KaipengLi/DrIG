<div align="center">

# DrIG

**Generative Universal Multimodal Retrieval with Dual-role Identifiers**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](drig_env.yml)
[![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c.svg)](drig_env.yml)
[![Dataset: M-BEIR](https://img.shields.io/badge/Dataset-M--BEIR-orange.svg)](https://huggingface.co/datasets/TIGER-Lab/M-BEIR)
[![CLIP-SF](https://img.shields.io/badge/Checkpoint-CLIP--SF-yellow.svg)](https://huggingface.co/TIGER-Lab/UniIR/resolve/main/checkpoint/CLIP_SF/clip_sf_large.pth)
![DrIG Checkpoints](https://img.shields.io/badge/DrIG_Checkpoints-uploading-lightgrey.svg)

</div>

DrIG is a generative universal multimodal retrieval framework that learns **dual-role identifiers** for unified retrieval across heterogeneous multimodal tasks.

## Overview

DrIG follows a three-stage pipeline:

1. **Stage 0: Multimodal feature extraction**
   Extract dense multimodal features using pretrained encoders such as LAMRA and CLIP-SF.

2. **Stage 1: Residual quantization**
   Train a residual quantizer that maps multimodal candidates into discrete identifiers.

3. **Stage 2: Generative retriever training**
   Train a sequence-to-sequence generator to produce retrieval identifiers from multimodal queries.

The repository contains code for M-BEIR experiments, MSCOCO/Flickr30k text-to-image retrieval, residual quantization, generator training, and evaluation.

## Installation

Clone this repository and create the Conda environment:

```bash
git clone https://github.com/ii-research/DrIG.git
cd DrIG
conda env create -f drig_env.yml
conda activate dig
```

The environment uses Python 3.10 and PyTorch 2.5.1. Git LFS is also included for downloading large model and dataset files.

## Dataset

### M-BEIR

We use [M-BEIR](https://huggingface.co/datasets/TIGER-Lab/M-BEIR), the Multimodal BEnchmark for Instructed Retrieval, for training and evaluating universal multimodal retrieval models.

Install Git LFS and clone the dataset:

```bash
git lfs install
git clone https://huggingface.co/datasets/TIGER-Lab/M-BEIR
```

After downloading, update the dataset paths in the corresponding YAML config files and shell scripts before training or evaluation.

### MSCOCO and Flickr30k

This repository also supports text-to-image retrieval experiments on MSCOCO and Flickr30k.

For Flickr30k, download the original dataset first. One available Kaggle mirror is:

```text
https://www.kaggle.com/datasets/eeshawn/flickr30k
```

Prepare MSCOCO and Flickr30k task0 training data:

```bash
cd src/data
bash preprocessing/coco_flickr/run_flickr_mscoco_task0_pipeline.sh
```

## Model Checkpoints

### CLIP-SF

We use UniIR's CLIP-SF checkpoint for encoder-side feature extraction.

```bash
mkdir -p checkpoint/CLIP_SF
wget https://huggingface.co/TIGER-Lab/UniIR/resolve/main/checkpoint/CLIP_SF/clip_sf_large.pth \
  -O checkpoint/CLIP_SF/clip_sf_large.pth
```

Checkpoint page:

```text
https://huggingface.co/TIGER-Lab/UniIR/blob/main/checkpoint/CLIP_SF/clip_sf_large.pth
```

### DrIG Checkpoints

DrIG checkpoints are being uploaded. After release, this section will include:

- Residual Quantization checkpoint
- Generator checkpoint
- Full download commands

## Usage

### Stage 0: Feature Extraction

#### LAMRA features for training data

```bash
cd src/feature_extraction/LAMRA
bash lamra_run_feature_extraction_train.sh
```

Outputs are saved under paths such as:

```text
embed/lamra/train
```

#### LAMRA features for candidate pools

```bash
cd src/feature_extraction/LAMRA
bash lamra_run_feature_extraction_cand.sh
```

Outputs are saved under paths such as:

```text
embed/lamra/cand
```

#### CLIP-SF features for candidate pools

```bash
cd src/feature_extraction/CLIP_SF
bash run_feature_extraction_cand.sh
```

Outputs are saved under paths such as:

```text
embed/CLIP_SF/cand
```

### Stage 1: Residual Quantization

Edit the config file first, including data paths, batch size, output paths, and codebook settings.

```bash
cd src/models/residual_quantization
vim configs_scripts/train_rq.yaml
bash configs_scripts/run_train.sh
```

For Flickr30k:

```bash
cd src/models/residual_quantization
bash configs_scripts/run_train_flickr.sh
```

### Stage 2: Generator Training

Edit the generator config before training.

```bash
cd src
vim models/generative_retriever/configs/train.yaml
bash models/generative_retriever/configs/run_train.sh
```

For Flickr30k:

```bash
cd src
bash models/generative_retriever/configs/run_train_flickr.sh
```

## Inference and Evaluation

### Compile the C++ trie

Compiling `trie_cpp` is recommended for faster inference.

```bash
cd src/models/generative_retriever
c++ -O3 -Wall -shared -std=c++17 -fPIC \
  $(python3 -m pybind11 --includes) \
  trie_cpp.cpp -o trie_cpp$(python3-config --extension-suffix)
```

DrIG supports three trie implementations during inference:

- `trie_cpp`: fastest C++ implementation
- `trie`: pure Python implementation
- `marisa`: alternative trie backend

### Run evaluation

```bash
cd src/eval/configs
bash run_eval.sh
```

For Flickr30k:

```bash
cd src/eval/configs
bash run_eval_flickr.sh
```

## Repository Structure

```text
DrIG/
├── drig_env.yml
├── LICENSE
├── README.md
└── src/
    ├── collators/
    ├── data/
    │   └── preprocessing/
    ├── eval/
    │   └── configs/
    ├── feature_extraction/
    │   ├── CLIP_SF/
    │   └── LAMRA/
    └── models/
        ├── generative_retriever/
        ├── lamra/
        ├── residual_quantization/
        └── uniir_clip/
```

## Performance

Results will be released with the paper and checkpoints.

The results in parentheses denote scores from reimplemented checkpoints, as the originals were lost during server migration. Slight variations may occur due to retraining randomness.

## Notes

Some shell scripts contain machine-specific absolute paths. Before running experiments, update paths such as dataset roots, checkpoint roots, output directories, and CUDA device settings according to your local environment.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
