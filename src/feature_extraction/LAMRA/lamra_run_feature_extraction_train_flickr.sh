#!/usr/bin/env bash
set -euo pipefail

genir_dir="/home/iiserver31/Workbench/likaipeng/dig"
SRC="$genir_dir/src"
MBEIR_DATA_DIR="/data/likaipeng/Flickr30k/"

export PYTHONPATH="$SRC"
export CUDA_VISIBLE_DEVICES=0,1,2,3
NPROC=4

MODEL_NAME_OR_PATH="/data/likaipeng/checkpoint/LamRA-Ret/"
ORIGINAL_MODEL_ID="$MODEL_NAME_OR_PATH"
DTYPE="bf16"

# ----------------------------
# Train split
# ----------------------------
QUERY_DATA_PATH="query/train/mbeir_flickr30k_task0_train.jsonl"
CAND_POOL_PATH="cand_pool/local/train/mbeir_flickr30k_task0_train_cand_pool.jsonl"

# ----------------------------
# Output controls
# ----------------------------
SAVE_DIR="/data/likaipeng/dig/embed/lamra/flickr/train/"
QUERY_OUT="mbeir_flickr30k_train_query_dict.pt"
POOL_OUT="mbeir_flickr30k_train_cand_pool_dict.pt"

EXTRACT_PY="$SRC/feature_extraction/LAMRA/lamra_feature_extraction_train.py"
CFG="$SRC/feature_extraction/LAMRA/lamra_extract_train.yaml"

mkdir -p "$SAVE_DIR"

MASTER_PORT=23334

echo "============================================================"
echo "[RUN] flickr30k_train"
echo "  cfg            = $CFG"
echo "  mbeir_dir      = $MBEIR_DATA_DIR"
echo "  query_path     = $QUERY_DATA_PATH"
echo "  cand_pool_path = $CAND_POOL_PATH"
echo "  out_dir        = $SAVE_DIR"
echo "  query_out      = $QUERY_OUT"
echo "  pool_out       = $POOL_OUT"
echo "  model          = $MODEL_NAME_OR_PATH"
echo "  dtype          = $DTYPE"
echo "  gpus           = $CUDA_VISIBLE_DEVICES (nproc=$NPROC)"
echo "  port           = $MASTER_PORT"
echo "============================================================"

python3 -m torch.distributed.run --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
  "$EXTRACT_PY" \
  --config_path "$CFG" \
  --mbeir_data_dir "$MBEIR_DATA_DIR" \
  --save_dir "$SAVE_DIR" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --original_model_id "$ORIGINAL_MODEL_ID" \
  --dtype "$DTYPE" \
  --do_query --do_pool \
  --query_data_path "$QUERY_DATA_PATH" \
  --cand_pool_path  "$CAND_POOL_PATH" \
  --query_out "$QUERY_OUT" \
  --pool_out  "$POOL_OUT"

echo "[DONE] flickr30k_train embeddings saved to: $SAVE_DIR"