<div align="center">

# DrIG

**Generative Universal Multimodal Retrieval with Dual-role Identifiers**

[![License: MIT](https://img.shields.io/badge/License-MIT-green?logo=opensourceinitiative&logoColor=white)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](drig_env.yml)
[![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1-ee4c2c?logo=pytorch&logoColor=white)](drig_env.yml)
[![Dataset: M-BEIR](https://img.shields.io/badge/Dataset-M--BEIR-yellow?logo=huggingface&logoColor=yellow)](https://huggingface.co/datasets/TIGER-Lab/M-BEIR)
[![Checkpoints: DrIG](https://img.shields.io/badge/Checkpoints-DrIG-yellow?logo=huggingface&logoColor=yellow)](https://huggingface.co/KaiPengLi/DrIG/tree/main/checkpoints)

</div>

DrIG is a generative universal multimodal retrieval framework that learns **dual-role identifiers** for unified retrieval across heterogeneous multimodal tasks.

## 🔎 Overview

DrIG follows a three-stage pipeline:

1. **Stage 0: Multimodal feature extraction**
   Encode queries and candidates into a shared embedding space using the external LamRA-Ret encoder.

2. **Stage 1: Dual-role identifier construction**
   Convert candidate embeddings into modality-aware residual-quantized identifiers.

3. **Stage 2: Generative retriever training**
   Train the set-based guidance module and T5 decoder, then retrieve candidates with Trie-constrained beam search and optional dense reranking.

The repository contains code for M-BEIR experiments, MSCOCO/Flickr30k text-to-image retrieval, residual quantization, generator training, and evaluation.

## ⚙️ Installation

Clone this repository and create the Conda environment:

```bash
git clone https://github.com/ii-research/DrIG.git
cd DrIG
conda env create -f drig_env.yml
conda activate dig
```

The environment uses Python 3.10 and PyTorch 2.5.1. Git LFS is also included for downloading large model and dataset files.

## 📚 Dataset

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

## 🤗 Model Checkpoints

### 🔗 LamRA-Ret

We use LamRA-Ret for encoder-side multimodal feature extraction.

```text
https://huggingface.co/code-kunkun/LamRA-Ret
```

### 📦 DrIG Checkpoints

We provide DrIG checkpoints on Hugging Face:

```text
https://huggingface.co/KaiPengLi/DrIG/tree/main/checkpoints
```

Download the residual quantization checkpoint and the available generative retriever checkpoints:

```bash
mkdir -p checkpoints/DrIG

wget https://huggingface.co/KaiPengLi/DrIG/resolve/main/checkpoints/rq_lamra.pt \
  -O checkpoints/DrIG/rq_lamra.pt

wget https://huggingface.co/KaiPengLi/DrIG/resolve/main/checkpoints/DrIG_t5small.pt \
  -O checkpoints/DrIG/DrIG_t5small.pt

wget https://huggingface.co/KaiPengLi/DrIG/resolve/main/checkpoints/DrIG_t5base.pt \
  -O checkpoints/DrIG/DrIG_t5base.pt

wget https://huggingface.co/KaiPengLi/DrIG/resolve/main/checkpoints/DrIG_t5large.pt \
  -O checkpoints/DrIG/DrIG_t5large.pt
```

Checkpoint links:

- **Feature encoder** (external): [LamRA-Ret](https://huggingface.co/code-kunkun/LamRA-Ret)
- **Residual quantization checkpoint**: [`rq_lamra.pt`](https://huggingface.co/KaiPengLi/DrIG/blob/main/checkpoints/rq_lamra.pt)
- **Generative retriever checkpoints**: [`DrIG_t5small.pt`](https://huggingface.co/KaiPengLi/DrIG/blob/main/checkpoints/DrIG_t5small.pt), [`DrIG_t5base.pt`](https://huggingface.co/KaiPengLi/DrIG/blob/main/checkpoints/DrIG_t5base.pt), [`DrIG_t5large.pt`](https://huggingface.co/KaiPengLi/DrIG/blob/main/checkpoints/DrIG_t5large.pt)

For full inference, use LamRA-Ret, `rq_lamra.pt`, and one generative retriever checkpoint.

## 🚀 Usage

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

### Stage 2: Generative Retriever Training

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

## 🔍 Inference and Evaluation

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

## 🗂️ Repository Structure

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
    │   └── LAMRA/
    └── models/
        ├── generative_retriever/
        ├── lamra/
        ├── residual_quantization/
        └── uniir_clip/
```

## 📈 Performance

Detailed experimental results are reported in the paper, including the main M-BEIR benchmark results and additional text-to-image retrieval results on Flickr30k and MSCOCO.

## 📝 Notes

Some shell scripts contain machine-specific absolute paths. Before running experiments, update paths such as dataset roots, checkpoint roots, output directories, and CUDA device settings according to your local environment.

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
