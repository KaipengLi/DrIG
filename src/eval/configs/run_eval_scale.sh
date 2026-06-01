#!/usr/bin/env bash
set -euo pipefail

# =========================
# Absolute roots (ONLY HERE)
# =========================
REPO_ROOT="/home/iiserver31/Workbench/likaipeng/dig"
SRC="${REPO_ROOT}/src"

MBEIR_ROOT="/data/likaipeng/M-BEIR"
DIG_ROOT="/data/likaipeng/dig"

export PYTHONPATH="${SRC}:${PYTHONPATH:-}"

# =========================
# Config
# =========================
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CFG_PATH="${1:-${SCRIPT_DIR}/eval.yaml}"

BASE_RQ_YAML="${SRC}/models/residual_quantization/configs_scripts/train_rq.yaml"

# -------------------------
# Search space
# -------------------------
T5_LIST=(
  "google-t5/t5-small"
# "google-t5/t5-base"
# "google-t5/t5-large"
)

# format: "VOCAB LEVEL_WO_MODALITY"
RQ_SETTINGS=(
  "4096 8"
# "4096 6"
# "4096 4"
# "1024 8"
# "1024 6"
# "1024 4"
# "256 4"
# "256 8"
# "256 6"
# "2048 8"
# "2048 6"
# "2048 4"
)

GUIDANCE_SCALES=(
  "0.1"
  "0.2"
  "0.3"
  "0.4"
  "0.5"
  "0.6"
  "0.7"
  "0.8"
  "0.9"
  "1.0"
  "5.0"
  "10.0"
)

TMP_EVAL_YAML=""
TMP_RQ_YAML=""
BASE_TMP_YAML=""

cleanup_tmp() {
  [[ -n "${TMP_EVAL_YAML}" && -f "${TMP_EVAL_YAML}" ]] && rm -f "${TMP_EVAL_YAML}"
  [[ -n "${TMP_RQ_YAML}" && -f "${TMP_RQ_YAML}" ]] && rm -f "${TMP_RQ_YAML}"
  [[ -n "${BASE_TMP_YAML}" && -f "${BASE_TMP_YAML}" ]] && rm -f "${BASE_TMP_YAML}"
}
trap cleanup_tmp EXIT

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

float_tag() {
  local x="$1"
  echo "${x//./p}"
}

# -------------------------
# 1) Read base eval yaml once
# -------------------------
BASE_TMP_YAML="$(mktemp /tmp/eval_base_XXXX.yaml)"

python - <<PY
from omegaconf import OmegaConf

cfg = OmegaConf.load("${CFG_PATH}")

MBEIR="${MBEIR_ROOT}"
DIG="${DIG_ROOT}"

def abs_join(root, rel):
    if rel is None:
        return rel
    rel = str(rel)
    if rel.startswith("/"):
        return rel
    rel = rel.lstrip("./")
    return root.rstrip("/") + "/" + rel

# Roots
cfg.mbeir_data_dir = abs_join(MBEIR, cfg.mbeir_data_dir)
cfg.embed_root     = abs_join(DIG,   cfg.embed_root)
cfg.save_dir       = abs_join(DIG,   cfg.save_dir)

# Optional union cand store
if "union_cand_store" in cfg and cfg.union_cand_store is not None:
    cfg.union_cand_store = abs_join(DIG, cfg.union_cand_store)

# External rerank dir
if "external_rerank_dir" in cfg and cfg.external_rerank_dir is not None:
    cfg.external_rerank_dir = abs_join(DIG, cfg.external_rerank_dir)

OmegaConf.save(cfg, "${BASE_TMP_YAML}")
PY

# -------------------------
# 2) Export runtime vars + arrays
# -------------------------
eval "$(
python - <<'PY' "${BASE_TMP_YAML}"
import sys, shlex
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])

def q(s):
    return shlex.quote(str(s))

print(f'NPROC={q(cfg.nproc)}')
print(f'MASTER_PORT={q(cfg.master_port)}')
print(f'CUDA_VISIBLE_DEVICES={q(cfg.cuda_visible_devices)}')
print(f'TORCH_CPP_LOG_LEVEL={q(cfg.torch_cpp_log_level)}')
print(f'EVAL_MODULE={q(cfg.eval_module)}')

print(f'USE_UNION_CAND={"true" if cfg.use_union_cand else "false"}')
print(f'UNION_CAND_STORE={q(cfg.union_cand_store)}')

print('TASKS=(' + ' '.join(q(x) for x in cfg.tasks) + ')')

nb = cfg.num_beams
if isinstance(nb, str):
    items = [x.strip() for x in nb.split(",") if x.strip()]
elif isinstance(nb, (list, tuple)):
    items = [str(x) for x in nb]
else:
    items = [str(nb)]
print('NUM_BEAMS=(' + ' '.join(q(x) for x in items) + ')')
PY
)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
[[ -n "${TORCH_CPP_LOG_LEVEL}" ]] && export TORCH_CPP_LOG_LEVEL="${TORCH_CPP_LOG_LEVEL}"

# -------------------------
# Build CLI args from YAML
# -------------------------
yaml_to_args() {
  local yaml_path="$1"
  python - <<'PY' "${yaml_path}"
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])

SKIP = {
  "nproc", "master_port", "cuda_visible_devices", "torch_cpp_log_level", "eval_module",
  "t5_ckpts", "tasks",
  "use_union_cand", "union_cand_store",
  "num_beams",
}

def to_cli(k, v):
  if k in SKIP:
    return []
  if isinstance(v, bool):
    return [f"--{k}"] if v else []
  if v is None:
    return []
  if OmegaConf.is_list(v):
    out = []
    for x in v:
      out += [f"--{k}", str(x)]
    return out
  if isinstance(v, (list, tuple)):
    out = []
    for x in v:
      out += [f"--{k}", str(x)]
    return out
  return [f"--{k}", str(v)]

args = []
for k in cfg.keys():
  args += to_cli(k, cfg[k])

print(" ".join(args))
PY
}

# -------------------------
# Run
# -------------------------
idx=0

for T5_NAME in "${T5_LIST[@]}"; do
  TAG="$(tag_from_name "${T5_NAME}")"

  for item in "${RQ_SETTINGS[@]}"; do
    read -r CODEBOOK_VOCAB CODEBOOK_LEVEL_WO_MODALITY <<< "${item}"
    CODEBOOK_LEVEL_TOTAL=$((CODEBOOK_LEVEL_WO_MODALITY + 1))
    RQ_TAG="${CODEBOOK_LEVEL_WO_MODALITY}x${CODEBOOK_VOCAB}"

    T5_CKPT_REL="checkpoints/DiG4UMR/${TAG}/${RQ_TAG}/finetune_t5_epoch_030.pt"
    RQ_CKPT_REL="checkpoints/RQ/${RQ_TAG}/ckpt_last.pt"

    T5_CKPT_ABS="${DIG_ROOT}/${T5_CKPT_REL}"
    RQ_CKPT_ABS="${DIG_ROOT}/${RQ_CKPT_REL}"

    if [[ ! -f "${T5_CKPT_ABS}" ]]; then
      echo "[SKIP] missing t5_ckpt: ${T5_CKPT_ABS}"
      continue
    fi

    if [[ ! -f "${RQ_CKPT_ABS}" ]]; then
      echo "[SKIP] missing rq_ckpt: ${RQ_CKPT_ABS}"
      continue
    fi

    TMP_RQ_YAML="$(mktemp /tmp/eval_rq_${RQ_TAG}_XXXX.yaml)"

    # --------------------------------------------------
    # A) Build per-run RQ yaml
    # --------------------------------------------------
    python - <<PY
from omegaconf import OmegaConf

rq_cfg = OmegaConf.load("${BASE_RQ_YAML}")

CODEBOOK_VOCAB = int(${CODEBOOK_VOCAB})
CODEBOOK_LEVEL_WO_MODALITY = int(${CODEBOOK_LEVEL_WO_MODALITY})

rq_cfg.rq_config.codebook_vocab = CODEBOOK_VOCAB
rq_cfg.rq_config.codebook_level = CODEBOOK_LEVEL_WO_MODALITY
rq_cfg.paths.output_dir = f"checkpoints/RQ/{CODEBOOK_LEVEL_WO_MODALITY}x{CODEBOOK_VOCAB}/"

if not hasattr(rq_cfg, "model") or rq_cfg.model is None:
    rq_cfg.model = {}
rq_cfg.model.size = f"{CODEBOOK_LEVEL_WO_MODALITY}x{CODEBOOK_VOCAB}"

OmegaConf.save(rq_cfg, "${TMP_RQ_YAML}")

print("Wrote RQ yaml:", "${TMP_RQ_YAML}")
print("  rq.codebook_vocab:", rq_cfg.rq_config.codebook_vocab)
print("  rq.codebook_level:", rq_cfg.rq_config.codebook_level)
PY

    for GS in "${GUIDANCE_SCALES[@]}"; do
      GS_TAG="$(float_tag "${GS}")"
      TMP_EVAL_YAML="$(mktemp /tmp/eval_${TAG}_${RQ_TAG}_gs_${GS_TAG}_XXXX.yaml)"

      # --------------------------------------------------
      # B) Build per-run eval yaml
      #   save_dir:
      #   .../${TAG}/${RQ_TAG}/gs/gs_${GS_TAG}
      # --------------------------------------------------
      RUN_SAVE_DIR="${DIG_ROOT}/eval_outputs/${TAG}/${RQ_TAG}/gs/gs_${GS_TAG}"

      python - <<PY
from omegaconf import OmegaConf

cfg = OmegaConf.load("${BASE_TMP_YAML}")

T5_NAME = "${T5_NAME}"
T5_CKPT_ABS = "${T5_CKPT_ABS}"
RQ_CKPT_ABS = "${RQ_CKPT_ABS}"
TMP_RQ_YAML = "${TMP_RQ_YAML}"
RUN_SAVE_DIR = "${RUN_SAVE_DIR}"
GUIDANCE_SCALE = float("${GS}")
GS_TAG = "${GS_TAG}"

CODEBOOK_VOCAB = int(${CODEBOOK_VOCAB})
CODEBOOK_LEVEL_WO_MODALITY = int(${CODEBOOK_LEVEL_WO_MODALITY})
CODEBOOK_LEVEL_TOTAL = int(${CODEBOOK_LEVEL_TOTAL})

cfg.t5_name = T5_NAME
cfg.t5_ckpts = [T5_CKPT_ABS]
cfg.rq_ckpt = RQ_CKPT_ABS
cfg.rq_yaml = TMP_RQ_YAML
cfg.save_dir = RUN_SAVE_DIR
cfg.guidance_scale = GUIDANCE_SCALE
cfg.out_tsv = f"recall_results_gs_{GS_TAG}.tsv"

# Optional consistency/debug fields
cfg.codebook_vocab = CODEBOOK_VOCAB
cfg.total_levels = CODEBOOK_LEVEL_TOTAL

OmegaConf.save(cfg, "${TMP_EVAL_YAML}")

print("Wrote EVAL yaml:", "${TMP_EVAL_YAML}")
print("  t5_name:", cfg.t5_name)
print("  t5_ckpt:", cfg.t5_ckpts[0])
print("  rq_ckpt:", cfg.rq_ckpt)
print("  rq_yaml:", cfg.rq_yaml)
print("  guidance_scale:", cfg.guidance_scale)
print("  save_dir:", cfg.save_dir)
print("  out_tsv:", cfg.out_tsv)
PY

      ARGS="$(yaml_to_args "${TMP_EVAL_YAML}")"
      PORT=$((MASTER_PORT + idx))

      for task in "${TASKS[@]}"; do
        for nb in "${NUM_BEAMS[@]}"; do
          EXTRA=()
          if [[ "${USE_UNION_CAND}" == "true" ]]; then
            EXTRA+=(--cand_store "${UNION_CAND_STORE}")
          fi

          echo "============================================================"
          echo ">>> model=${TAG} rq=${RQ_TAG} gs=${GS} task=${task} num_beams=${nb}"
          echo ">>> t5_ckpt=$(basename "${T5_CKPT_ABS}")"
          echo ">>> rq_yaml=${TMP_RQ_YAML}"
          echo ">>> port=${PORT}"
          echo ">>> save_dir=${RUN_SAVE_DIR}"
          echo "============================================================"

          torchrun --nproc_per_node="${NPROC}" --master_port="${PORT}" \
            -m "${EVAL_MODULE}" \
            --task "${task}" \
            --t5_ckpt "${T5_CKPT_ABS}" \
            --num_beams "${nb}" \
            ${ARGS} "${EXTRA[@]}"

          PORT=$((PORT + 1))
        done
      done

      rm -f "${TMP_EVAL_YAML}"
      TMP_EVAL_YAML=""
      idx=$((idx + 10))
    done

    rm -f "${TMP_RQ_YAML}"
    TMP_RQ_YAML=""
  done
done

echo "[DONE] all eval runs finished."