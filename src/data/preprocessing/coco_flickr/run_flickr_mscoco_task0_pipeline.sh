#!/usr/bin/env bash
set -euo pipefail

# Go to project root based on the location of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# =========================
# Paths
# =========================
FLICKR_DIR="/data/likaipeng/Flickr30k"
IMG_DIR="mbeir_images/flickr30k/Images/"
MBEIR_DIR="/data/likaipeng/M-BEIR"

# =========================
# Flickr30k Step 1: Image preprocessing
# =========================
echo "========================================"
echo "Flickr30k Step 1: Image preprocessing"
echo "========================================"
python preprocessing/flickr_data_preprocessor.py \
  --mbeir_data_dir "${FLICKR_DIR}" \
  --flickr30k_images_dir "${IMG_DIR}" \
  --flickr30k_dir "" \
  --enable_image_processing

# =========================
# Flickr30k Step 2: Generate candidate pool
# =========================
echo
echo "========================================"
echo "Flickr30k Step 2: Generate candidate pool"
echo "========================================"
python preprocessing/flickr_data_preprocessor.py \
  --mbeir_data_dir "${FLICKR_DIR}" \
  --flickr30k_images_dir "${IMG_DIR}" \
  --flickr30k_dir "" \
  --enable_candidate_pool

# =========================
# Flickr30k Step 3: Convert to M-BEIR format
# =========================
echo
echo "========================================"
echo "Flickr30k Step 3: Convert to M-BEIR query format"
echo "========================================"
python preprocessing/flickr_data_preprocessor.py \
  --mbeir_data_dir "${FLICKR_DIR}" \
  --flickr30k_images_dir "${IMG_DIR}" \
  --flickr30k_dir "" \
  --enable_mbeir_conversion

# =========================
# Flickr30k Step 4-6: Prepare task0-only files
# =========================
echo
echo "========================================"
echo "Flickr30k Step 4-6: Prepare task0-only files"
echo "========================================"
python preprocessing/coco_flickr/prepare_flickr_task0.py \
  --flickr30k_dir "${FLICKR_DIR}"

# =========================
# MSCOCO Step 1-2: Prepare task0-only files
# =========================
echo
echo "========================================"
echo "MSCOCO Step 1-2: Prepare task0-only files"
echo "========================================"
python preprocessing/coco_flickr/prepare_mscoco_task0.py \
  --mbeir_root "${MBEIR_DIR}"

echo
echo "========================================"
echo "All steps completed successfully."
echo "========================================"