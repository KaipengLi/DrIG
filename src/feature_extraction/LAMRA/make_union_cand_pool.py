#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hard-coded union builder for M-BEIR cand pool embedding dicts.

Input dir : /data/likaipeng/gur/embed_inst/test/
Output    : /data/likaipeng/gur/embed_inst/test/mbeir_union_cand_pool_dict.pt

Keeps the same saving convention as your embedding extraction script:
- emb saved as float32 (like emb[i].cpu().float())
- masks saved as long
- id_to_index rebuilt from union ordering
"""

import os
import glob
from typing import Dict, List

import torch

# ---------------------------
# Hard-coded paths
# ---------------------------
INPUT_DIR = "/data/likaipeng/dig/embed/lamra/cand"
OUTPUT_PT = "/data/likaipeng/dig/embed/lamra/cand/mbeir_union_cand_pool_dict.pt"
PATTERN = "mbeir_*_cand_pool_dict.pt"  # matches: mbeir_cirr_task7_cand_pool_dict.pt, mbeir_edis_task2_cand_pool_dict.pt, ...


def load_pt(path: str) -> dict:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"{path}: expected dict, got {type(obj)}")
    required = ["emb", "img_mask", "txt_mask", "modality_id", "id_to_index"]
    for k in required:
        if k not in obj:
            raise KeyError(f"{path}: missing key '{k}'")
    if not torch.is_tensor(obj["emb"]) or obj["emb"].dim() != 2:
        raise TypeError(f"{path}: 'emb' must be a 2D Tensor [N, D]")
    return obj


def invert_id_to_index(id_to_index: Dict[int, int]) -> List[int]:
    """
    Build hids_by_index where hids_by_index[i] gives the hid for row i.
    Assumes id_to_index is a permutation of 0..N-1.
    """
    n = len(id_to_index)
    hids_by_index = [None] * n
    for hid, idx in id_to_index.items():
        if idx < 0 or idx >= n:
            raise ValueError(f"Bad index in id_to_index: hid={hid} idx={idx} n={n}")
        if hids_by_index[idx] is not None:
            raise ValueError(f"Duplicate index in id_to_index: idx={idx}")
        hids_by_index[idx] = hid
    if any(x is None for x in hids_by_index):
        raise ValueError("id_to_index is not a full permutation of [0..N-1]")
    return hids_by_index


def main():
    files = glob.glob(os.path.join(INPUT_DIR, PATTERN))
    files = sorted(files)  # deterministic order

    if not files:
        raise FileNotFoundError(f"No files matched: {os.path.join(INPUT_DIR, PATTERN)}")

    print(f"[INFO] input_dir = {INPUT_DIR}")
    print(f"[INFO] found {len(files)} files:")
    for fp in files:
        print(f"  - {fp}")

    # union containers
    union_embs = []
    union_img = []
    union_txt = []
    union_mid = []
    union_id_to_index: Dict[int, int] = {}

    modality_vocab = {0: "image", 1: "text", 2: "image,text"}
    D_ref = None

    dup_count = 0

    for fp in files:
        obj = load_pt(fp)
        emb = obj["emb"]              # [N, D]
        img_mask = obj["img_mask"]    # [N]
        txt_mask = obj["txt_mask"]    # [N]
        modality_id = obj["modality_id"]  # [N]
        id_to_index = obj["id_to_index"]

        if "modality_vocab" in obj and isinstance(obj["modality_vocab"], dict):
            modality_vocab = obj["modality_vocab"]

        N, D = emb.shape
        if D_ref is None:
            D_ref = D
        elif D != D_ref:
            raise ValueError(f"Embedding dim mismatch: {fp} has D={D}, expected D={D_ref}")

        # sanity checks
        for name, t in [("img_mask", img_mask), ("txt_mask", txt_mask), ("modality_id", modality_id)]:
            if not torch.is_tensor(t):
                raise TypeError(f"{fp}: {name} must be Tensor")
            if t.shape[0] != N:
                raise ValueError(f"{fp}: {name} length mismatch: {t.shape[0]} vs N={N}")

        hids_by_index = invert_id_to_index(id_to_index)

        for i in range(N):
            hid = int(hids_by_index[i])

            # dedup by hid (keep first, consistent & safe)
            if hid in union_id_to_index:
                dup_count += 1
                continue

            new_idx = len(union_embs)
            union_id_to_index[hid] = new_idx

            # keep same saving convention as your extractor: emb saved as float32
            union_embs.append(emb[i].detach().cpu().float())
            union_img.append(img_mask[i].detach().cpu().long())
            union_txt.append(txt_mask[i].detach().cpu().long())
            union_mid.append(modality_id[i].detach().cpu().long())

    # stack
    out = {
        "emb": torch.stack(union_embs, dim=0),                 # float32
        "img_mask": torch.stack(union_img, dim=0),             # long
        "txt_mask": torch.stack(union_txt, dim=0),             # long
        "modality_id": torch.stack(union_mid, dim=0),          # long
        "id_to_index": union_id_to_index,
        "modality_vocab": modality_vocab,
    }

    os.makedirs(os.path.dirname(OUTPUT_PT), exist_ok=True)
    torch.save(out, OUTPUT_PT)

    print(f"[OK] saved union cand pool dict to: {OUTPUT_PT}")
    print(f"[OK] union N={out['emb'].shape[0]} D={out['emb'].shape[1]}")
    print(f"[OK] dup_count={dup_count} (kept first occurrence)")


if __name__ == "__main__":
    main()
