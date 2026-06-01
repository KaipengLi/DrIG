#!/usr/bin/env bash
set -euo pipefail

DATASET_NAME="flickr30k"
RQ_DATASET_NAME="flickr"
# -------------------------
# Entry
# -------------------------
TRAIN_PY="models/generative_retriever/train.py"
BASE_YAML="models/generative_retriever/configs/train_flickr.yaml"
BASE_RQ_YAML="models/residual_quantization/configs_scripts/train_rq_flickr.yaml"

OUT_ROOT="/data/likaipeng/dig/checkpoints/DiG4UMR"
LOG_ROOT="${OUT_ROOT}/logs"

# Absolute roots (ONLY HERE)
DATA_ROOT="/data/likaipeng"
WORK_ROOT="/home/iiserver31/Workbench/likaipeng"

export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

NPROC=4
BASE_PORT=29500

T5_LIST=(
  "google-t5/t5-small"
# "google-t5/t5-base"
# "google-t5/t5-large"
)

# format: "VOCAB LEVEL_WO_MODALITY"
RQ_SETTINGS=(
  "4096 8"

)

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

# -------------------------
# Helpers
# -------------------------
tag_from_name() {
  local name="$1"
  if [[ "$name" == *"t5-small"* ]]; then echo "t5_small"; return; fi
  if [[ "$name" == *"t5-base"*  ]]; then echo "t5_base";  return; fi
  if [[ "$name" == *"t5-large"* ]]; then echo "t5_large"; return; fi
  if [[ "$name" == *"t5-3b"*    ]]; then echo "t5_3b";    return; fi
  if [[ "$name" == *"t5-11b"*   ]]; then echo "t5_11b";   return; fi
  echo "${name//\//_}"
}

TMP_YAML=""
TMP_RQ_YAML=""

cleanup_tmp() {
  if [[ -n "${TMP_YAML}" && -f "${TMP_YAML}" ]]; then
    rm -f "${TMP_YAML}"
  fi
  if [[ -n "${TMP_RQ_YAML}" && -f "${TMP_RQ_YAML}" ]]; then
    rm -f "${TMP_RQ_YAML}"
  fi
}
trap cleanup_tmp EXIT

# -------------------------
# Main loop
# -------------------------
idx=0

for T5_NAME in "${T5_LIST[@]}"; do
  TAG="$(tag_from_name "${T5_NAME}")"

  for item in "${RQ_SETTINGS[@]}"; do
    read -r CODEBOOK_VOCAB CODEBOOK_LEVEL_WO_MODALITY <<< "${item}"

    CODEBOOK_LEVEL_TOTAL=$((CODEBOOK_LEVEL_WO_MODALITY + 1))
    RQ_TAG="${CODEBOOK_LEVEL_WO_MODALITY}x${CODEBOOK_VOCAB}"

    RUN_OUT="${OUT_ROOT}/${DATASET_NAME}/${TAG}/${RQ_TAG}"
    RUN_LOG="${LOG_ROOT}/${DATASET_NAME}_${TAG}_${RQ_TAG}.log"
    PORT=$((BASE_PORT + idx))

    mkdir -p "${RUN_OUT}"

    TMP_YAML="$(mktemp "/tmp/train_${TAG}_${RQ_TAG}_XXXX.yaml")"
    TMP_RQ_YAML="$(mktemp "/tmp/train_rq_${RQ_TAG}_XXXX.yaml")"

    # --------------------------------------------------
    # 1) Build per-run RQ yaml
    # --------------------------------------------------
    python - <<PY
from omegaconf import OmegaConf

rq_cfg = OmegaConf.load('${BASE_RQ_YAML}')

CODEBOOK_VOCAB = int(${CODEBOOK_VOCAB})
CODEBOOK_LEVEL_WO_MODALITY = int(${CODEBOOK_LEVEL_WO_MODALITY})
RQ_DATASET_NAME = '${RQ_DATASET_NAME}'
rq_cfg.rq_config.codebook_vocab = CODEBOOK_VOCAB
rq_cfg.rq_config.codebook_level = CODEBOOK_LEVEL_WO_MODALITY

# Keep output dir consistent with this run
rq_cfg.paths.output_dir = f"checkpoints/RQ/{RQ_DATASET_NAME}/{CODEBOOK_LEVEL_WO_MODALITY}x{CODEBOOK_VOCAB}/"

if not hasattr(rq_cfg, 'model') or rq_cfg.model is None:
    rq_cfg.model = {}
rq_cfg.model.size = f"{CODEBOOK_LEVEL_WO_MODALITY}x{CODEBOOK_VOCAB}"

OmegaConf.save(rq_cfg, '${TMP_RQ_YAML}')

print('Wrote RQ yaml:', '${TMP_RQ_YAML}')
print('  rq.codebook_vocab:', rq_cfg.rq_config.codebook_vocab)
print('  rq.codebook_level:', rq_cfg.rq_config.codebook_level)
print('  rq.output_dir:', rq_cfg.paths.output_dir)
PY

    # --------------------------------------------------
    # 2) Build per-run T5 yaml
    # --------------------------------------------------
    python - <<PY
from omegaconf import OmegaConf

cfg = OmegaConf.load('${BASE_YAML}')

DATA_ROOT = '${DATA_ROOT}'
RUN_OUT = '${RUN_OUT}'
T5_NAME = '${T5_NAME}'
TAG = '${TAG}'
TMP_RQ_YAML = '${TMP_RQ_YAML}'
DATASET_NAME = '${DATASET_NAME}'

CODEBOOK_VOCAB = int(${CODEBOOK_VOCAB})
CODEBOOK_LEVEL_WO_MODALITY = int(${CODEBOOK_LEVEL_WO_MODALITY})
CODEBOOK_LEVEL_TOTAL = int(${CODEBOOK_LEVEL_TOTAL})
RQ_DATASET_NAME = '${RQ_DATASET_NAME}'

RQ_TAG = '${RQ_TAG}'

def abs_join(root, rel):
    rel = str(rel)
    return rel if rel.startswith('/') else root.rstrip('/') + '/' + rel.lstrip('/')

# -------------------------
# overrides per run
# -------------------------
cfg.t5_name = T5_NAME
cfg.out_dir = RUN_OUT

cfg.codebook_vocab = CODEBOOK_VOCAB
cfg.codebook_level_wo_modality = CODEBOOK_LEVEL_WO_MODALITY
cfg.codebook_level_total = CODEBOOK_LEVEL_TOTAL

# IMPORTANT:
# Use per-run RQ yaml + matching ckpt
cfg.rq_yaml = TMP_RQ_YAML
cfg.rq_ckpt = f"dig/checkpoints/RQ/{RQ_DATASET_NAME}/{RQ_TAG}/ckpt_last.pt"

# Better naming for model / wandb
if not hasattr(cfg, 'model') or cfg.model is None:
    cfg.model = {}

cfg.model.name = str(getattr(cfg.model, 'name', 'T5'))
cfg.model.size = f"{DATASET_NAME}_{TAG}_{RQ_TAG}"

# -------------------------
# absolutize paths
# -------------------------
cfg.train_jsonl = abs_join(DATA_ROOT, cfg.train_jsonl)
cfg.query_store = abs_join(DATA_ROOT, cfg.query_store)
cfg.cand_store  = abs_join(DATA_ROOT, cfg.cand_store)
cfg.rq_ckpt     = abs_join(DATA_ROOT, cfg.rq_ckpt)
cfg.out_dir     = abs_join(DATA_ROOT, cfg.out_dir)

OmegaConf.save(cfg, '${TMP_YAML}')

print('Wrote T5 yaml:', '${TMP_YAML}')
print('  t5_name:', cfg.t5_name)
print('  codebook_vocab:', cfg.codebook_vocab)
print('  codebook_level_wo_modality:', cfg.codebook_level_wo_modality)
print('  codebook_level_total:', cfg.codebook_level_total)
print('  rq_yaml:', cfg.rq_yaml)
print('  rq_ckpt:', cfg.rq_ckpt)
print('  out_dir:', cfg.out_dir)
print('  model.size:', cfg.model.size)
print('  wandb.name:', cfg.wandb_config.experiment_name if hasattr(cfg, 'wandb_config') else None)
PY

    # --------------------------------------------------
    # 3) Safety checks
    # --------------------------------------------------
    RQ_CKPT_ABS="${DATA_ROOT}/dig/checkpoints/RQ/${RQ_DATASET_NAME}/${RQ_TAG}/ckpt_last.pt"
    if [[ ! -f "${RQ_CKPT_ABS}" ]]; then
      echo "[SKIP] missing rq_ckpt: ${RQ_CKPT_ABS}"
      rm -f "${TMP_YAML}" "${TMP_RQ_YAML}"
      TMP_YAML=""
      TMP_RQ_YAML=""
      idx=$((idx + 1))
      continue
    fi

    echo "=================================================="
    echo "[RUN] ${TAG} + ${RQ_TAG}"
    echo "  t5_name   = ${T5_NAME}"
    echo "  rq_tag    = ${RQ_TAG}"
    echo "  rq_yaml   = ${TMP_RQ_YAML}"
    echo "  out_dir   = ${RUN_OUT}"
    echo "  log       = ${RUN_LOG}"
    echo "  port      = ${PORT}"
    echo "=================================================="

    torchrun \
      --nproc_per_node="${NPROC}" \
      --master_port="${PORT}" \
      "${TRAIN_PY}" \
      --config_path "${TMP_YAML}" 2>&1 | tee "${RUN_LOG}"

    rm -f "${TMP_YAML}" "${TMP_RQ_YAML}"
    TMP_YAML=""
    TMP_RQ_YAML=""

    idx=$((idx + 1))
  done
done

echo "[DONE] all runs finished."