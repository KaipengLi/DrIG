# DrIG

### Introduction

---

We propose **DrIG**, a novel **Dual-identifier Generative framework for Universal Multimodal Retrieval**.


## Overview

---

## Installation

Clone the repository and create the Conda environment:

```bash
git clone <YOUR_REPO_URL>
cd DiG
conda env create -f drig_env.yml
```

## M-BEIR

---

We utilize **M-BEIR** (Multimodal BEnchmark for Instructed Retrieval) for training and evaluating our universal multimodal retrieval models. This large-scale benchmark enables comprehensive assessment of model performance across various retrieval tasks.

### Downloading M-BEIR

The M-BEIR dataset is available on Hugging Face. To download and prepare the dataset:

1. Set up Git LFS (Large File Storage):

```bash
git lfs install
```

2. Clone the dataset repository:

```bash
git clone <M-BEIR_DATASET_REPO>
```

The dataset will be used for both training and evaluation phases of **DrIG**.

---

## Usage

---

### Multimodal Encoder Feature Extraction (Stage 0)

We utilize UniIR's score fusion model as a replacement for the encoder pretraining stage.

### 1. Download Pretrained CLIP-SF

```bash
mkdir -p checkpoint/CLIP_SF
wget <CLIP_SF_CHECKPOINT_URL> -O checkpoint/CLIP_SF/clip_sf_large.pth
```

### 2. Feature Extraction

#### Training Data

Extracts Lamra features for training set → `embed/lamra/train`.

```bash
# Navigate to feature extraction directory
cd feature_extraction/LAMRA

# Run feature extraction for training data
bash lamra_run_feature_extraction_train.sh
```

#### Candidate Pool

Extracts Lamra features for the retrieval candidate pool → `embed/lamra/cand`.

```bash
# Run feature extraction for candidate pool
bash lamra_run_feature_extraction_cand.sh
```
or

Extracts CLIP-SF features for the retrieval candidate pool → `embed/CLIP_SF/cand`.

```bash
cd feature_extraction/CLIP_SF
# Run feature extraction for candidate pool
bash run_feature_extraction_cand.sh
```

---

### Residual Quantization (Stage 1)

```bash
cd models/residual_quantization
vim configs_scripts/train_rq.yaml  # Edit config like data path, batch size, etc.
bash configs_scripts/run_train.sh
```

---

### Generator Training (Stage 2)

```bash
cd src/
vim configs/train.yaml  # Edit config like data path, batch size, etc.
bash models/generative_retriever/configs/run_train.sh
```

### Inference

1. Extract Lamra features for candidate pool (if not already done):
```bash
cd feature_extraction
bash lamra_run_feature_extraction_cand.sh
```

2. Compile trie_cpp (recommended for faster inference):
```bash
cd models/generative_retriever
c++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    trie_cpp.cpp -o trie_cpp$(python3-config --extension-suffix)
```

3. Run evaluation:
```bash
cd eval/configs
bash run_eval.sh
```
> For inference, you can choose between three trie implementations: `trie_cpp` (fastest), `trie` (Python), `marisa` (alternative).



## Model Checkpoints

We provide model checkpoints for DrIG in the 🤗 [Hugging Face](https://huggingface.co/):

### How to Download
```bash
# Download the CLIP-SF model (Stage 0)
wget https://huggingface.co/TIGER-Lab/UniIR/resolve/main/checkpoint/CLIP_SF/clip_sf_large.pth -O checkpoint/CLIP_SF/clip_sf_large.pth

# Clone the DiG4UMR checkpoints (Stage 1 and 2)
git clone https://huggingface.co/
```
### Each Component Checkpoints
- **CLIP-SF Model** (Stage 0): [`clip_sf_large.pth`](https://huggingface.co/TIGER-Lab/UniIR/blob/main/checkpoint/CLIP_SF/clip_sf_large.pth)
- **Residual Quantization Model** (Stage 1): [`rq_clip_large.pth`](https://huggingface.co/)
- **Generator Model** (Stage 2): [`DrIG_t5small.pth`]([https://huggingface.co/](https://huggingface.co/KaiPengLi/DrIG))

> Note: All three models are required for full functionality. 

## 📈 Performance

> The results in parentheses denote scores from our reimplemented checkpoints, as the originals were lost during server migration. While close to the paper, slight variations may occur due to retraining randomness.


## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/) file for details.



## MSCOCO & Flickr30k

This project supports **text-to-image retrieval** on both **MSCOCO** and **Flickr30k**.

### 1. Prepare MSCOCO & Flickr30k task0 training data
### Flckr30k
about flckr30k, you should dowload the original dataset
https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset

```bash
cd src/data
bash preprocessing/coco_flickr/run_flickr_mscoco_task0_pipeline.sh 
```

and you can use them to feature extraction and training
```bash
cd src/
bash feature_extraction/LAMRA/lamra_run_feature_extraction_train_flickr.sh
bash feature_extraction/LAMRA/lamra_run_feature_extraction_cand_flickr.sh

bash feature_extraction/LAMRA/lamra_run_feature_extraction_cand_coco.sh
```

and use the generated features to train the RQ model and the generator model.
```bash
cd models/residual_quantization
bash configs_scripts/run_train_flickr.sh
cd src/
bash models/generative_retriever/configs/run_train_flickr.sh
```

and finally, you can evaluate the trained model on the test set of MSCOCO task0.
```bash 
eval/configs/run_eval_flickr.sh
```
