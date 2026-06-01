#!/usr/bin/env bash
set -euo pipefail

genir_dir="/home/iiserver31/Workbench/likaipeng/dig"
SRC="$genir_dir/src"
MBEIR_DATA_DIR="/data/likaipeng/M-BEIR/"

export PYTHONPATH="$SRC"
export CUDA_VISIBLE_DEVICES=0,1,2,3
NPROC=4

MODEL_NAME_OR_PATH="/data/likaipeng/checkpoint/LamRA-Ret/"
ORIGINAL_MODEL_ID="$MODEL_NAME_OR_PATH"
DTYPE="bf16"

EXTRACT_PY="$SRC/feature_extraction/LAMRA/lamra_feature_extraction_train.py"
CFG="$SRC/feature_extraction/LAMRA/lamra_extract_train.yaml"   # 通用 yaml（不包含 query/cand 路径）

# 根输出目录
SAVE_ROOT="/data/likaipeng/dig/embed/lamra"
QUERY_DIR="$SAVE_ROOT/test"   # query 输出
POOL_DIR="$SAVE_ROOT/cand"    # pool 输出

mkdir -p "$QUERY_DIR" "$POOL_DIR"

# 任务清单：task_name | query_rel_path | pool_rel_path
DATASETS=(
  # 1) visualnews_task0
  "visualnews_task0_test|query/test/mbeir_visualnews_task0_test.jsonl|cand_pool/local/mbeir_visualnews_task0_cand_pool.jsonl"

  # 2) mscoco_task0_test
  "mscoco_task0_test|query/test/mbeir_mscoco_task0_test.jsonl|cand_pool/local/mbeir_mscoco_task0_test_cand_pool.jsonl"

  # 3) fashion200k_task0
  "fashion200k_task0_test|query/test/mbeir_fashion200k_task0_test.jsonl|cand_pool/local/mbeir_fashion200k_task0_cand_pool.jsonl"

  # 4) webqa_task1
  "webqa_task1_test|query/test/mbeir_webqa_task1_test.jsonl|cand_pool/local/mbeir_webqa_task1_cand_pool.jsonl"

  # 5) edis_task2
  "edis_task2_test|query/test/mbeir_edis_task2_test.jsonl|cand_pool/local/mbeir_edis_task2_cand_pool.jsonl"

  # 6) webqa_task2
  "webqa_task2_test|query/test/mbeir_webqa_task2_test.jsonl|cand_pool/local/mbeir_webqa_task2_cand_pool.jsonl"

  # 7) visualnews_task3
  "visualnews_task3_test|query/test/mbeir_visualnews_task3_test.jsonl|cand_pool/local/mbeir_visualnews_task3_cand_pool.jsonl"

  # 8) mscoco_task3_test
  "mscoco_task3_test|query/test/mbeir_mscoco_task3_test.jsonl|cand_pool/local/mbeir_mscoco_task3_test_cand_pool.jsonl"

  # 9) fashion200k_task3
  "fashion200k_task3_test|query/test/mbeir_fashion200k_task3_test.jsonl|cand_pool/local/mbeir_fashion200k_task3_cand_pool.jsonl"

  # 10) nights_task4
  "nights_task4_test|query/test/mbeir_nights_task4_test.jsonl|cand_pool/local/mbeir_nights_task4_cand_pool.jsonl"

  # 11) oven_task6
  "oven_task6_test|query/test/mbeir_oven_task6_test.jsonl|cand_pool/local/mbeir_oven_task6_cand_pool.jsonl"

  # 12) infoseek_task6
  "infoseek_task6_test|query/test/mbeir_infoseek_task6_test.jsonl|cand_pool/local/mbeir_infoseek_task6_cand_pool.jsonl"

  # 13) fashioniq_task7
  "fashioniq_task7_test|query/test/mbeir_fashioniq_task7_test.jsonl|cand_pool/local/mbeir_fashioniq_task7_cand_pool.jsonl"

  # 14) cirr_task7
  "cirr_task7_test|query/test/mbeir_cirr_task7_test.jsonl|cand_pool/local/mbeir_cirr_task7_cand_pool.jsonl"

  # 15) oven_task8
  "oven_task8_test|query/test/mbeir_oven_task8_test.jsonl|cand_pool/local/mbeir_oven_task8_cand_pool.jsonl"

  # 16) infoseek_task8
  "infoseek_task8_test|query/test/mbeir_infoseek_task8_test.jsonl|cand_pool/local/mbeir_infoseek_task8_cand_pool.jsonl"
)

MASTER_PORT_BASE=23334
i=0
last_index=$((${#DATASETS[@]} - 1))

for item in "${DATASETS[@]}"; do
  IFS="|" read -r TASK QUERY_PATH POOL_PATH <<< "$item"

  # 输出文件名：沿用你之前的命名风格
  query_base=$(basename "$QUERY_PATH" .jsonl)
  pool_base=$(basename "$POOL_PATH" .jsonl)
  pool_task=${pool_base%_cand_pool}

  OUT_QUERY_NAME="${query_base}_dict.pt"
  OUT_POOL_NAME="${pool_task}_cand_pool_dict.pt"

  MASTER_PORT=$((MASTER_PORT_BASE + i))

  #  union（ "--union_pool" or ""）
  UNION_FLAG=""
  if [[ $i -eq $last_index ]]; then
    UNION_FLAG="--union_pool"
  fi

  echo "============================================================"
  echo "[RUN] $TASK"
  echo "  cfg        =$CFG"
  echo "  mbeir_dir  =$MBEIR_DATA_DIR"
  echo "  query_path =$QUERY_PATH -> $QUERY_DIR/$OUT_QUERY_NAME"
  echo "  pool_path  =$POOL_PATH  -> $POOL_DIR/$OUT_POOL_NAME"
  echo "  union      =$([[ -n "$UNION_FLAG" ]] && echo "ON (at end)" || echo "OFF")"
  echo "  model      =$MODEL_NAME_OR_PATH"
  echo "  dtype      =$DTYPE"
  echo "  gpus       =$CUDA_VISIBLE_DEVICES (nproc=$NPROC)"
  echo "  port       =$MASTER_PORT"
  echo "============================================================"

  python3 -m torch.distributed.run --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    "$EXTRACT_PY" \
    --config_path "$CFG" \
    --mbeir_data_dir "$MBEIR_DATA_DIR" \
    --save_dir "$SAVE_ROOT" \
    --query_save_dir "$QUERY_DIR" \
    --pool_save_dir "$POOL_DIR" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --original_model_id "$ORIGINAL_MODEL_ID" \
    --dtype "$DTYPE" \
    --do_query --do_pool \
    --query_data_path "$QUERY_PATH" \
    --cand_pool_path  "$POOL_PATH" \
    --query_out "$OUT_QUERY_NAME" \
    --pool_out  "$OUT_POOL_NAME" \
    $UNION_FLAG

  i=$((i + 1))
done

echo "============================================================"
echo "[DONE ALL]"
echo "  query dir: $QUERY_DIR"
echo "  pool  dir: $POOL_DIR"
echo "  union    : $POOL_DIR/mbeir_union_cand_pool_dict.pt"
echo "============================================================"