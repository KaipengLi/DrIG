#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse


def ensure_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def extract_mscoco_task0_train_candidate_pool(mbeir_root: str):
    """
    Extract MSCOCO task0 image training candidate pool from the global union candidate pool.

    Input:
        {mbeir_root}/cand_pool/global/mbeir_union_train_cand_pool.jsonl

    Output:
        {mbeir_root}/cand_pool/local/train/mbeir_mscoco_task0_train_cand_pool.jsonl
    """
    input_path = os.path.join(
        mbeir_root,
        "cand_pool",
        "global",
        "mbeir_union_train_cand_pool.jsonl",
    )
    output_path = os.path.join(
        mbeir_root,
        "cand_pool",
        "local",
        "train",
        "mbeir_mscoco_task0_train_cand_pool.jsonl",
    )

    if not os.path.exists(input_path):
        print(f"[SKIP] Candidate pool file not found: {input_path}")
        return

    ensure_parent(output_path)

    count = 0
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            entry = json.loads(line)
            if entry.get("did", "").startswith("9:") and entry.get("modality") == "image":
                fout.write(line)
                count += 1

    print("[OK] MSCOCO task0 train candidate pool saved to:")
    print(f"     {output_path}")
    print(f"     Total entries: {count}")


def extract_mscoco_task0_train_queries(mbeir_root: str):
    """
    Extract MSCOCO task0 text-to-image training queries.

    Input:
        {mbeir_root}/query/train/mbeir_mscoco_train.jsonl

    Output:
        {mbeir_root}/query/train/mscoco/mbeir_mscoco_task0_train.jsonl
    """
    input_path = os.path.join(
        mbeir_root,
        "query",
        "train",
        "mbeir_mscoco_train.jsonl",
    )
    output_path = os.path.join(
        mbeir_root,
        "query",
        "train",
        "mscoco",
        "mbeir_mscoco_task0_train.jsonl",
    )

    if not os.path.exists(input_path):
        print(f"[SKIP] Train query file not found: {input_path}")
        return

    ensure_parent(output_path)

    count = 0
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            entry = json.loads(line)
            if entry.get("query_modality") == "text":
                fout.write(line)
                count += 1

    print("[OK] MSCOCO task0 train queries saved to:")
    print(f"     {output_path}")
    print(f"     Total entries: {count}")


def main():
    parser = argparse.ArgumentParser(description="Prepare MSCOCO task0 training query and candidate pool files.")
    parser.add_argument(
        "--mbeir_root",
        type=str,
        default="/data/likaipeng/M-BEIR",
        help="Root directory of M-BEIR",
    )
    args = parser.parse_args()

    mbeir_root = args.mbeir_root

    print("Step 1: Extract MSCOCO task0 training candidate pool")
    extract_mscoco_task0_train_candidate_pool(mbeir_root)

    print("\nStep 2: Extract MSCOCO task0 training queries")
    extract_mscoco_task0_train_queries(mbeir_root)

    print("\nDone.")


if __name__ == "__main__":
    main()