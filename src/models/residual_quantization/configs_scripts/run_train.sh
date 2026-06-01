#!/usr/bin/env bash
set -euo pipefail

genir_dir="/home/iiserver31/Workbench/likaipeng/dig"
SRC="$genir_dir/src"
mbeir_data_dir="/data/likaipeng/M-BEIR"

export PYTHONPATH="$SRC"
export CUDA_VISIBLE_DEVICES=0,1,2,3

NPROC=4
BASE_PORT=29501

RQ_DIR="$SRC/models/residual_quantization"
TRAIN_PY="$RQ_DIR/train.py"
CFG_DIR="$RQ_DIR/configs_scripts"
BASE_CONFIG_YAML="$CFG_DIR/train_rq.yaml"

# --------------------------------------------------
# Ablation settings: (codebook_vocab, codebook_level)
# --------------------------------------------------
SETTINGS=(
  "4096 8"
#  "4096 6"
#  "4096 4"
#  "1024 8"
#  "1024 6"
#  "1024 4"
#  "256 4"
#  "256 8"
#  "256 6"
#  "2048 4"
#  "2048 6"
#  "2048 8"
)

run_id=0

for item in "${SETTINGS[@]}"; do
  read -r vocab level <<< "$item"

  port=$((BASE_PORT + run_id))
  run_id=$((run_id + 1))

  tmp_yaml="$(mktemp /tmp/train_rq_${vocab}x${level}_XXXX.yaml)"

  python - <<PY
from omegaconf import OmegaConf

cfg = OmegaConf.load("${BASE_CONFIG_YAML}")

cfg.rq_config.codebook_vocab = int(${vocab})
cfg.rq_config.codebook_level = int(${level})

# Keep naming consistent and explicit
cfg.paths.output_dir = f"checkpoints/RQ/{cfg.rq_config.codebook_level}x{cfg.rq_config.codebook_vocab}/"
cfg.model.size = f"{cfg.rq_config.codebook_level}x{cfg.rq_config.codebook_vocab}"

OmegaConf.save(cfg, "${tmp_yaml}")
print("saved:", "${tmp_yaml}")
PY

  echo "============================================================"
  echo ">>> Start training: codebook_vocab=${vocab}, codebook_level=${level}"
  echo ">>> Config: ${tmp_yaml}"
  echo ">>> MASTER_PORT=${port}"
  echo "============================================================"

  torchrun \
    --nproc_per_node="$NPROC" \
    --master_port="$port" \
    "$TRAIN_PY" \
    --config "$tmp_yaml" \
    --genir_dir "/data/likaipeng/dig" \
    --mbeir_data_dir "$mbeir_data_dir"

  rm -f "$tmp_yaml"

  echo "============================================================"
  echo ">>> Finished: codebook_vocab=${vocab}, codebook_level=${level}"
  echo "============================================================"
  echo
done