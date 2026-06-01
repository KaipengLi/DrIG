# utils.py
# -*- coding: utf-8 -*-

import os
from typing import Tuple
from typing import Dict, Optional
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import OmegaConf

import torch.nn.functional as F

# =========================
# Distributed helpers
# =========================
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def ddp_init(backend: str = "nccl", dist_url: str = "env://") -> Tuple[bool, int]:
    """
    Initialize DDP from env vars:
      - RANK, WORLD_SIZE, LOCAL_RANK
    Returns:
      (ddp_on, local_rank)

    NOTE:
      Use barrier(device_ids=[local_rank]) to avoid NCCL warnings about unknown devices.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        dist.init_process_group(
            backend=backend,
            init_method=dist_url,
            rank=rank,
            world_size=world_size,
        )

        # Avoid NCCL warning: "using GPU X to perform barrier as devices used by this process are currently unknown"
        if torch.cuda.is_available():
            dist.barrier(device_ids=[local_rank])
        else:
            dist.barrier()

        return True, local_rank

    return False, 0


def ddp_cleanup():
    """
    Safely finalize DDP.

    NOTE:
      Use barrier(device_ids=[current_device]) to avoid NCCL warnings.
    """
    if is_dist():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()
        dist.destroy_process_group()


# =========================
# Path resolving helpers
# =========================
def _resolve_path(p: str, base: str) -> str:
    """
    Resolve a path:
      - If p is absolute, return it unchanged.
      - If p is relative, join it with base.
    """
    if p is None:
        return p
    p = str(p)
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(str(base), p))


def resolve_paths_with_roots(cfg, genir_dir: str, mbeir_data_dir: str):
    """
    Resolve cfg.paths.* according to GENIUS-style roots:

    - query_emb_pt / pool_emb_pt / output_dir: relative to genir_dir
    - query_jsonl: relative to mbeir_data_dir

    Absolute paths remain unchanged.
    """
    if not hasattr(cfg, "paths"):
        raise AttributeError("Config must have 'paths' section.")

    cfg.paths.query_emb_pt = _resolve_path(cfg.paths.query_emb_pt, genir_dir)
    cfg.paths.pool_emb_pt = _resolve_path(cfg.paths.pool_emb_pt, genir_dir)
    cfg.paths.output_dir = _resolve_path(cfg.paths.output_dir, genir_dir)
    cfg.paths.query_jsonl = _resolve_path(cfg.paths.query_jsonl, mbeir_data_dir)

    return cfg


# =========================
# Checkpoint I/O
# =========================
def save_ckpt(path: str, model: nn.Module, optim: torch.optim.Optimizer, step: int, epoch: int, cfg):
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "step": int(step),
        "epoch": int(epoch),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }
    torch.save(state, path)


# =========================
# Debug / printing helpers
# =========================
MID2STR = {0: "image", 1: "text", 2: "image,text"}

_LEVEL_CHARS = "abcdefghijklmnopqrstuvwxyz"


def _level_prefix(i: int) -> str:
    """Support more than 26 levels by using aa, ab, ..."""
    s = ""
    while True:
        s = _LEVEL_CHARS[i % 26] + s
        i = i // 26 - 1
        if i < 0:
            break
    return s


def codes_to_token_str(code_row: torch.Tensor, modality_index: bool, codebook_level: int) -> str:
    """
    Convert one RQ code row into a readable token string.

    code_row: [L] long tensor, where
      L = codebook_level + (1 if modality_index else 0)
    """
    codes = code_row.tolist()
    out = []
    offset = 0

    if modality_index:
        out.append(f"<a{int(codes[0])}>")
        offset = 1
        level_start = 1
    else:
        level_start = 0

    for li, cid in enumerate(codes[offset : offset + codebook_level]):
        out.append(f"<{_level_prefix(li + level_start)}{int(cid)}>")

    return " ".join(out)


def update_confusion(cm: torch.Tensor, gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Update a 3x3 confusion matrix where classes are {0,1,2}."""
    gt = gt.clamp(0, 2).to(torch.long)
    pred = pred.clamp(0, 2).to(torch.long)
    idx = (gt * 3 + pred).view(-1)
    cm += torch.bincount(idx, minlength=9).view(3, 3)
    return cm


def best_perm_and_acc(cm_cpu: torch.Tensor) -> Tuple[Tuple[int, int, int], float]:
    """Find best column permutation aligning predicted clusters to GT labels."""
    import itertools

    best_perm, best_sum = (0, 1, 2), -1
    for perm in itertools.permutations([0, 1, 2]):
        s = int(cm_cpu[0, perm[0]] + cm_cpu[1, perm[1]] + cm_cpu[2, perm[2]])
        if s > best_sum:
            best_sum, best_perm = s, perm

    total = int(cm_cpu.sum())
    acc = best_sum / max(total, 1)
    return best_perm, acc


def print_cm(cm_np, title: str):
    """Pretty-print a 3x3 confusion matrix."""
    row_labels = [MID2STR[0], MID2STR[1], MID2STR[2]]
    col_labels = ["cluster0", "cluster1", "cluster2"]
    print(f"\n[Epoch DBG] {title} confusion (GT rows x Is0 cols):")
    print(" " * 14 + "  ".join([f"{c:>10}" for c in col_labels]))
    for r, rl in enumerate(row_labels):
        vals = "  ".join([f"{int(cm_np[r, c]):>10d}" for c in range(3)])
        print(f"{rl:>14}  {vals}")
    print(f"[Epoch DBG] GT counts:  {cm_np.sum(axis=1).tolist()}  (rows)")
    print(f"[Epoch DBG] Is0 counts: {cm_np.sum(axis=0).tolist()}  (cols)")
    perm, acc = best_perm_and_acc(torch.tensor(cm_np))
    print(f"[Epoch DBG] best col-perm (GT->Is0) = {perm}, best-align-acc = {acc:.4f}")


# =========================
# Codebook statistics
# =========================


@torch.no_grad()
def compute_rq_stats(Is: torch.Tensor, codebook_vocab: int) -> dict:
    """
    GENIUS-style RQ stats:
      - num_collision: number of duplicated full code rows within the batch
      - perplexity: average perplexity across codebook levels

    Args:
        Is: [N, L] long tensor, RQ codes (full rows).
        codebook_vocab: size of codebook vocabulary (e.g., 4096)

    Returns:
        dict with:
          rq/num_collision, rq/perplexity
    """
    if Is is None:
        return {}

    if Is.dim() != 2:
        raise ValueError(f"Is must be 2D [N,L], got {tuple(Is.shape)}")

    # ---- collision (simple batch-level duplicate rows) ----
    # num_collision = total_rows - unique_rows
    # (This matches "collision" intuition and is the simplest usable equivalent.)
    unique_rows = torch.unique(Is, dim=0).size(0)
    num_collision = int(Is.size(0) - unique_rows)

    # ---- perplexity (GENIUS-style) ----
    # perplexity per level = exp(-sum(avg_probs * log(avg_probs)))
    # average over levels
    L = Is.size(1)
    perplexity = 0.0
    for i in range(L):
        col = Is[:, i].long().clamp(min=0, max=codebook_vocab - 1)
        encodings = F.one_hot(col, num_classes=codebook_vocab).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity_i = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-6)))
        perplexity += float(perplexity_i.item())
    perplexity = perplexity / max(L, 1)

    return {
        "rq/num_collision": float(num_collision),
        "rq/perplexity": float(perplexity),
    }