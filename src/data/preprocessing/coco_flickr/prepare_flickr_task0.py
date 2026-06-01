#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import shutil


MBEIR_TASK_TEXT_TO_IMAGE = 0


def find_first_existing(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def ensure_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def extract_task0_train_candidate_pool(flickr30k_dir: str):
    """
    Read the mixed train candidate pool and keep image-only entries for task0.

    Output:
        {flickr30k_dir}/cand_pool/local/train/mbeir_flickr30k_task0_train_cand_pool.jsonl
    """
    input_path = find_first_existing([
        os.path.join(flickr30k_dir, "cand_pool", "train", "mbeir_flickr30k_train_cand_pool.jsonl"),
        os.path.join(flickr30k_dir, "mbeir_flickr30k_train_cand_pool.jsonl"),
    ])
    output_path = os.path.join(
        flickr30k_dir,
        "cand_pool",
        "local",
        "train",
        "mbeir_flickr30k_task0_train_cand_pool.jsonl",
    )

    if input_path is None:
        print("[SKIP] Train candidate pool file not found.")
        return

    ensure_parent(output_path)

    count = 0
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            entry = json.loads(line)
            if entry.get("modality") == "image":
                fout.write(line)
                count += 1

    print("[OK] Task0 train candidate pool saved to:")
    print(f"     {output_path}")
    print(f"     Total entries: {count}")


def extract_task0_train_queries(flickr30k_dir: str):
    """
    Read the mixed train query file and keep text-only queries for task0.

    Output:
        {flickr30k_dir}/query/train/mbeir_flickr30k_task0_train.jsonl
    """
    input_path = find_first_existing([
        os.path.join(flickr30k_dir, "query", "train", "mbeir_flickr30k_train.jsonl"),
        os.path.join(flickr30k_dir, "mbeir_flickr30k_train.jsonl"),
    ])
    output_path = os.path.join(
        flickr30k_dir,
        "query",
        "train",
        "mbeir_flickr30k_task0_train.jsonl",
    )

    if input_path is None:
        print("[SKIP] Train query file not found.")
        return

    ensure_parent(output_path)

    count = 0
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            entry = json.loads(line)
            if entry.get("query_modality") == "text":
                fout.write(line)
                count += 1

    print("[OK] Task0 train queries saved to:")
    print(f"     {output_path}")
    print(f"     Total entries: {count}")


def organize_task0_val_test_queries(flickr30k_dir: str):
    """
    Copy task0 val/test query files into query/val and query/test directories.

    Source candidates:
        {flickr30k_dir}/mbeir_flickr30k_task0_val.jsonl
        {flickr30k_dir}/mbeir_flickr30k_task0_test.jsonl

    Destination:
        {flickr30k_dir}/query/val/mbeir_flickr30k_task0_val.jsonl
        {flickr30k_dir}/query/test/mbeir_flickr30k_task0_test.jsonl
    """
    mapping = [
        (
            find_first_existing([
                os.path.join(flickr30k_dir, "query", "val", "mbeir_flickr30k_task0_val.jsonl"),
                os.path.join(flickr30k_dir, "mbeir_flickr30k_task0_val.jsonl"),
            ]),
            os.path.join(flickr30k_dir, "query", "val", "mbeir_flickr30k_task0_val.jsonl"),
        ),
        (
            find_first_existing([
                os.path.join(flickr30k_dir, "query", "test", "mbeir_flickr30k_task0_test.jsonl"),
                os.path.join(flickr30k_dir, "mbeir_flickr30k_task0_test.jsonl"),
            ]),
            os.path.join(flickr30k_dir, "query", "test", "mbeir_flickr30k_task0_test.jsonl"),
        ),
    ]

    for src_path, dst_path in mapping:
        if src_path is None:
            print(f"[SKIP] Source query file not found for: {dst_path}")
            continue

        if os.path.abspath(src_path) == os.path.abspath(dst_path):
            print("[OK] Query file already in place:")
            print(f"     {dst_path}")
            continue

        ensure_parent(dst_path)
        shutil.copy2(src_path, dst_path)

        print("[OK] Copied query file to:")
        print(f"     {dst_path}")


def organize_task0_test_candidate_pool(flickr30k_dir: str):
    """
    Copy task0 test candidate pool file into cand_pool directory.

    Source candidates:
        {flickr30k_dir}/mbeir_flickr30k_task0_test_cand_pool.jsonl
        {flickr30k_dir}/cand_pool/mbeir_flickr30k_task0_test_cand_pool.jsonl

    Destination:
        {flickr30k_dir}/cand_pool/mbeir_flickr30k_task0_test_cand_pool.jsonl
    """
    src_path = find_first_existing([
        os.path.join(flickr30k_dir, "cand_pool", "mbeir_flickr30k_task0_test_cand_pool.jsonl"),
        os.path.join(flickr30k_dir, "mbeir_flickr30k_task0_test_cand_pool.jsonl"),
    ])
    dst_path = os.path.join(
        flickr30k_dir,
        "cand_pool",
        "mbeir_flickr30k_task0_test_cand_pool.jsonl",
    )

    if src_path is None:
        print(f"[SKIP] Source test candidate pool file not found for: {dst_path}")
        return

    if os.path.abspath(src_path) == os.path.abspath(dst_path):
        print("[OK] Test candidate pool file already in place:")
        print(f"     {dst_path}")
        return

    ensure_parent(dst_path)
    shutil.copy2(src_path, dst_path)

    print("[OK] Copied test candidate pool file to:")
    print(f"     {dst_path}")


def generate_qrels_from_query(query_jsonl_path: str, output_qrels_path: str, task_id: int = 0):
    ensure_parent(output_qrels_path)

    count = 0
    with open(query_jsonl_path, "r") as fin, open(output_qrels_path, "w") as fout:
        for line in fin:
            entry = json.loads(line.strip())
            qid = entry["qid"]
            for did in entry.get("pos_cand_list", []):
                fout.write(f"{qid} 0 {did} 1 {task_id}\n")
                count += 1

    print("[OK] Qrels saved to:")
    print(f"     {output_qrels_path}")
    print(f"     Total entries: {count}")


def generate_task0_qrels(flickr30k_dir: str):
    """
    Generate qrels for task0 only.

    Outputs:
        {flickr30k_dir}/qrels/train/mbeir_flickr30k_task0_train_qrels.txt
        {flickr30k_dir}/qrels/val/mbeir_flickr30k_task0_val_qrels.txt
        {flickr30k_dir}/qrels/test/mbeir_flickr30k_task0_test_qrels.txt
    """
    mapping = [
        (
            [
                os.path.join(flickr30k_dir, "query", "train", "mbeir_flickr30k_task0_train.jsonl"),
                os.path.join(flickr30k_dir, "mbeir_flickr30k_task0_train.jsonl"),
            ],
            os.path.join(flickr30k_dir, "qrels", "train", "mbeir_flickr30k_task0_train_qrels.txt"),
        ),
        (
            [
                os.path.join(flickr30k_dir, "query", "val", "mbeir_flickr30k_task0_val.jsonl"),
                os.path.join(flickr30k_dir, "mbeir_flickr30k_task0_val.jsonl"),
            ],
            os.path.join(flickr30k_dir, "qrels", "val", "mbeir_flickr30k_task0_val_qrels.txt"),
        ),
        (
            [
                os.path.join(flickr30k_dir, "query", "test", "mbeir_flickr30k_task0_test.jsonl"),
                os.path.join(flickr30k_dir, "mbeir_flickr30k_task0_test.jsonl"),
            ],
            os.path.join(flickr30k_dir, "qrels", "test", "mbeir_flickr30k_task0_test_qrels.txt"),
        ),
    ]

    for query_candidates, qrels_path in mapping:
        query_path = find_first_existing(query_candidates)
        if query_path is None:
            print(f"[SKIP] Query file not found for qrels: {qrels_path}")
            continue

        generate_qrels_from_query(
            query_jsonl_path=query_path,
            output_qrels_path=qrels_path,
            task_id=MBEIR_TASK_TEXT_TO_IMAGE,
        )


def main():
    parser = argparse.ArgumentParser(description="Prepare Flickr30k task0-only query/candidate/qrels files.")
    parser.add_argument(
        "--flickr30k_dir",
        type=str,
        default="/data/likaipeng/Flickr30k",
        help="Root directory of Flickr30k",
    )
    args = parser.parse_args()

    flickr30k_dir = args.flickr30k_dir

    print("Step 4: Extract Flickr30k task0 train candidate pool")
    extract_task0_train_candidate_pool(flickr30k_dir)

    print("\nStep 5: Extract Flickr30k task0 train queries")
    extract_task0_train_queries(flickr30k_dir)

    print("\nStep 5.5: Organize Flickr30k task0 val/test queries")
    organize_task0_val_test_queries(flickr30k_dir)

    print("\nStep 5.6: Organize Flickr30k task0 test candidate pool")
    organize_task0_test_candidate_pool(flickr30k_dir)

    print("\nStep 6: Generate Flickr30k task0 qrels")
    generate_task0_qrels(flickr30k_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()