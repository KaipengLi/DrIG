#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
utils.py

Shared utilities for embedding extraction scripts.

This module centralizes:
- M-BEIR hashing rules (qid/did)
- modality parsing -> masks / modality_id
- jsonl loading
- GENIUS-style query instruction TSV loading
- LamRA-style message building
- output filename derivation
- distributed helpers + variable-length gather
- LamRA eval-compatible <emb> token injection
- embedding extraction from model outputs
- union builder for cand_pool embedding dicts (optional post-processing)

Notes:
- Keep the behavior consistent with the original script.
- Do not normalize embeddings here (normalization is controlled by model.inference()).
"""

import os
import json
import glob
from typing import Dict, List, Tuple, Optional, Iterable

import torch
import torch.distributed as dist

# ---------------------------
# Hash ID (must match M-BEIR preprocess)
# ---------------------------
DATASET_QUERY_NUM_UPPER_BOUND = 500000
DATASET_CAN_NUM_UPPER_BOUND = 10000000


def hash_qid(qid: str) -> int:
    """qid format: 'dataset_id:within_id'."""
    dataset_id, data_within_id = map(int, qid.split(":"))
    return dataset_id * DATASET_QUERY_NUM_UPPER_BOUND + data_within_id


def hash_did(did: str) -> int:
    """did format: 'dataset_id:within_id'."""
    dataset_id, data_within_id = map(int, did.split(":"))
    return dataset_id * DATASET_CAN_NUM_UPPER_BOUND + data_within_id


# ---------------------------
# Modality -> masks / ids
# ---------------------------
# modality string examples: "text", "image", "image,text"
def modality_to_masks_and_id(modality: Optional[str]) -> Tuple[int, int, int]:
    """
    Return:
      img_mask: 1 if image exists else 0
      txt_mask: 1 if text exists else 0
      modality_id: {0:image, 1:text, 2:image,text, 255:unknown}
    """
    m = (modality or "").lower().replace(" ", "")
    has_img = 1 if "image" in m else 0
    has_txt = 1 if "text" in m else 0
    if has_img and has_txt:
        mid = 2
    elif has_txt:
        mid = 1
    elif has_img:
        mid = 0
    else:
        mid = 255
    return has_img, has_txt, mid


# ---------------------------
# Utils: jsonl + prompt table
# ---------------------------
def load_jsonl(path: str) -> List[dict]:
    """Load jsonl as list[dict]. Keep original behavior."""
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_string(s: str) -> str:
    """Lightweight string normalization (same as original)."""
    s = (s or "").replace("\r", "").strip().strip('"')
    if s:
        s = s[0].upper() + s[1:]
        s = s + "." if s[-1] not in [".", "?", "!"] else s
    return s


def load_query_instructions(tsv_path: str) -> Dict[str, List[str]]:
    """
    TSV columns in GENIUS:
      [query_modality, cand_modality, ?, dataset_id, prompt1, prompt2, ...]
    key = f"{dataset_id}, {query_modality}, {cand_modality}"
    """
    prompts_dict: Dict[str, List[str]] = {}
    with open(tsv_path, "r") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            key = f"{parts[3]}, {parts[0]}, {parts[1]}"
            prompts = [p for p in parts[4:] if p]
            if prompts:
                prompts_dict[key] = prompts
    return prompts_dict


def pick_prompt(prompts: List[str], stable_seed: int) -> str:
    """Deterministic prompt selection."""
    return prompts[stable_seed % len(prompts)]


# ---------------------------
# Message builder (LamRA style)
# ---------------------------
def build_chat_messages(txt: Optional[str], image_abs_path: Optional[str]) -> List[dict]:
    """
    LamRA style:
      - user: image + text (if exist)
      - assistant: "<emb>."
    """
    content = []
    if image_abs_path is not None:
        content.append({"type": "image", "image": image_abs_path})
    if txt is not None and txt != "":
        content.append({"type": "text", "text": f"{txt}\nSummarize above content in one word: "})
    else:
        content.append({"type": "text", "text": "\nSummarize above content in one word: "})

    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": "<emb>."}]},
    ]


# ---------------------------
# Output filename derivation
# ---------------------------
def _stem(p: str) -> str:
    """Return filename stem without extension. e.g. a/b/c.jsonl -> c"""
    base = os.path.basename(str(p))
    for ext in [".jsonl", ".json", ".txt"]:
        if base.endswith(ext):
            return base[:-len(ext)]
    return os.path.splitext(base)[0]


def derive_out_name(kind: str, cfg) -> str:
    """
    Auto-derive output filename from cfg input paths:
      - query: stem(query_data_path) + '_query_dict.pt'
      - pool : stem(cand_pool_path) + '_cand_pool_dict.pt'
    Also removes duplicate suffix like '_cand_pool' if it already exists in stem.
    """
    assert kind in ["query", "pool"]
    data_cfg = cfg.data_config

    if kind == "query":
        base = _stem(data_cfg.query_data_path)
        base = base.replace(" ", "_")
        return f"{base}_query_dict.pt"

    base = _stem(data_cfg.cand_pool_path)
    base = base.replace(" ", "_")
    base = base.replace("_cand_pool", "")
    return f"{base}_cand_pool_dict.pt"


# ---------------------------
# Distributed helpers (variable batch gather)
# ---------------------------
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def gather_tensor_varlen(t: torch.Tensor) -> torch.Tensor:
    """All-gather variable-length batch (pad -> gather -> trim)."""
    if not is_dist():
        return t

    device = t.device
    local_n = torch.tensor([t.size(0)], device=device, dtype=torch.long)
    sizes = [torch.zeros_like(local_n) for _ in range(get_world_size())]
    dist.all_gather(sizes, local_n)
    sizes = [int(s.item()) for s in sizes]
    max_n = max(sizes)

    if t.size(0) < max_n:
        pad_shape = (max_n - t.size(0),) + t.shape[1:]
        pad = torch.zeros(pad_shape, device=device, dtype=t.dtype)
        t_pad = torch.cat([t, pad], dim=0)
    else:
        t_pad = t

    gathered = [torch.zeros_like(t_pad) for _ in range(get_world_size())]
    dist.all_gather(gathered, t_pad)

    outs = []
    for g, n in zip(gathered, sizes):
        outs.append(g[:n])
    return torch.cat(outs, dim=0)


# ---------------------------
# Model output -> embedding
# ---------------------------
def extract_embedding(model_out) -> torch.Tensor:
    """Extract embedding tensor from LamRA-style model output."""
    if torch.is_tensor(model_out):
        return model_out
    if isinstance(model_out, (tuple, list)) and len(model_out) > 0 and torch.is_tensor(model_out[0]):
        return model_out[0]
    raise TypeError(f"Unexpected model_out type: {type(model_out)}")


# ---------------------------
# Distributed init / cleanup
# ---------------------------
def init_distributed():
    """
    Initialize torch.distributed using env:// (torchrun).
    Returns: (dist_on: bool, local_rank: int)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
        dist.barrier(device_ids=[local_rank])
        return True, local_rank
    return False, 0


def cleanup_distributed():
    if is_dist():
        dist.barrier(device_ids=[torch.cuda.current_device()])
        dist.destroy_process_group()


# ---------------------------
# LamRA eval-compatible helper
# ---------------------------
def add_embed_token(tokenizer, model, emb_token: str = "<emb>") -> int:
    """
    Ensure <emb> exists in tokenizer and model config is updated.
    Keep behavior consistent with your original script.
    """
    vocab = tokenizer.get_vocab()
    if emb_token not in vocab:
        tokenizer.add_tokens([emb_token])
        model.resize_token_embeddings(len(tokenizer))
    emb_id = tokenizer.convert_tokens_to_ids(emb_token)
    model.config.emb_token_ids = [emb_id]
    return emb_id


# ===========================
# Union builder for cand pool embedding dicts
# ===========================
def load_pt_embedding_dict(path: str) -> dict:
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
    hids_by_index: List[Optional[int]] = [None] * n
    for hid, idx in id_to_index.items():
        if idx < 0 or idx >= n:
            raise ValueError(f"Bad index in id_to_index: hid={hid} idx={idx} n={n}")
        if hids_by_index[idx] is not None:
            raise ValueError(f"Duplicate index in id_to_index: idx={idx}")
        hids_by_index[idx] = int(hid)
    if any(x is None for x in hids_by_index):
        raise ValueError("id_to_index is not a full permutation of [0..N-1]")
    return [int(x) for x in hids_by_index]


def build_union_cand_pool_dict(
    input_files: Iterable[str],
    output_pt: str,
    pattern_hint: str = "",
) -> Dict[str, object]:
    """
    Merge multiple '*_cand_pool_dict.pt' embedding dicts into one union dict.

    Behavior (consistent with your original union script):
    - Deterministic file order (sorted)
    - Deduplicate by hid (keep first occurrence)
    - emb saved as float32
    - masks/modality_id saved as long
    - id_to_index rebuilt in union ordering
    - embedding dim must match across files
    """
    files = sorted(list(input_files))
    if not files:
        raise FileNotFoundError(f"No input files for union. {pattern_hint}".strip())

    print(f"[INFO] union: merging {len(files)} files")
    for fp in files:
        print(f"  - {fp}")

    union_embs = []
    union_img = []
    union_txt = []
    union_mid = []
    union_id_to_index: Dict[int, int] = {}

    modality_vocab = {0: "image", 1: "text", 2: "image,text"}
    D_ref = None
    dup_count = 0

    for fp in files:
        obj = load_pt_embedding_dict(fp)
        emb = obj["emb"]                 # [N, D]
        img_mask = obj["img_mask"]       # [N]
        txt_mask = obj["txt_mask"]       # [N]
        modality_id = obj["modality_id"] # [N]
        id_to_index = obj["id_to_index"]

        if "modality_vocab" in obj and isinstance(obj["modality_vocab"], dict):
            modality_vocab = obj["modality_vocab"]

        N, D = emb.shape
        if D_ref is None:
            D_ref = D
        elif D != D_ref:
            raise ValueError(f"Embedding dim mismatch: {fp} has D={D}, expected D={D_ref}")

        for name, t in [("img_mask", img_mask), ("txt_mask", txt_mask), ("modality_id", modality_id)]:
            if not torch.is_tensor(t):
                raise TypeError(f"{fp}: {name} must be Tensor")
            if t.shape[0] != N:
                raise ValueError(f"{fp}: {name} length mismatch: {t.shape[0]} vs N={N}")

        hids_by_index = invert_id_to_index(id_to_index)

        for i in range(N):
            hid = int(hids_by_index[i])

            # dedup by hid (keep first)
            if hid in union_id_to_index:
                dup_count += 1
                continue

            new_idx = len(union_embs)
            union_id_to_index[hid] = new_idx

            union_embs.append(emb[i].detach().cpu().float())
            union_img.append(img_mask[i].detach().cpu().long())
            union_txt.append(txt_mask[i].detach().cpu().long())
            union_mid.append(modality_id[i].detach().cpu().long())

    out = {
        "emb": torch.stack(union_embs, dim=0),
        "img_mask": torch.stack(union_img, dim=0),
        "txt_mask": torch.stack(union_txt, dim=0),
        "modality_id": torch.stack(union_mid, dim=0),
        "id_to_index": union_id_to_index,
        "modality_vocab": modality_vocab,
    }

    os.makedirs(os.path.dirname(output_pt), exist_ok=True)
    torch.save(out, output_pt)

    print(f"[OK] union saved to: {output_pt}")
    print(f"[OK] union N={out['emb'].shape[0]} D={out['emb'].shape[1]}")
    print(f"[OK] union dup_count={dup_count} (kept first occurrence)")
    return out


def union_cand_pool(
    input_dir: str,
    output_pt: str,
    pattern: str = "mbeir_*_cand_pool_dict.pt",
) -> Dict[str, object]:
    """
    Merge all matching cand pool dicts under input_dir into output_pt,
    while excluding output_pt itself.
    """
    files = glob.glob(os.path.join(input_dir, pattern))
    output_abs = os.path.abspath(output_pt)
    files = [fp for fp in files if os.path.abspath(fp) != output_abs]
    return build_union_cand_pool_dict(
        input_files=files,
        output_pt=output_pt,
        pattern_hint=f"dir={input_dir} pattern={pattern} exclude={output_pt}",
    )