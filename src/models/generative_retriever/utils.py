# -*- coding: utf-8 -*-
"""
utils.py

Shared utilities:
- DDP helpers
- Seeding
- Hashing (GENIUS-compatible)
- Optimizer / scheduler (GENIUS-style)
- JSONL offset reader
- Code-token helpers (build tokens / labels)
- Similarity helpers
- Checkpoint helpers
"""

from __future__ import annotations

import os
import re
import math
import json
import random
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from transformers import T5TokenizerFast

# -------------------------
# GENIUS-compatible hashing
# -------------------------
DATASET_QUERY_NUM_UPPER_BOUND = 500_000
DATASET_CAN_NUM_UPPER_BOUND = 10_000_000

IGNORE_INDEX = -100


def hash_qid(qid: str) -> int:
    """Hash qid like 'dataset_id:within_id' -> integer id (GENIUS-compatible)."""
    dataset_id, within_id = map(int, qid.split(":"))
    return dataset_id * DATASET_QUERY_NUM_UPPER_BOUND + within_id


def hash_did(did: str) -> int:
    """Hash did like 'dataset_id:within_id' -> integer id (GENIUS-compatible)."""
    dataset_id, within_id = map(int, did.split(":"))
    return dataset_id * DATASET_CAN_NUM_UPPER_BOUND + within_id


# -------------------------
# DDP helpers
# -------------------------
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main() -> bool:
    return get_rank() == 0


def ddp_barrier() -> None:
    if is_dist():
        dist.barrier()


def setup_distributed() -> Tuple[int, int, torch.device]:
    """
    DDP env://:
      RANK, WORLD_SIZE, LOCAL_RANK
    Returns: (local_rank, world_size, device)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        world = get_world_size()
    else:
        local_rank = 0
        world = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return local_rank, world, device


def cleanup_distributed() -> None:
    if is_dist():
        dist.destroy_process_group()


def unwrap_ddp(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_t5_size_tag(t5_name: str) -> str:
    name = t5_name.lower()
    for k in ["small", "base", "large", "3b", "11b", "xxl", "xl"]:
        if k in name:
            return k
    return re.split(r"[\/\-]", name)[-1] or "t5"


# -------------------------
# Optimizer & scheduler
# -------------------------
def _filter_parameters(model: nn.Module, condition_fn):
    return [p for n, p in model.named_parameters() if condition_fn(n, p) and p.requires_grad]


def create_optimizer_genius(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    exclude_condition = lambda n, p: (
        p.ndim < 2 or any(sub in n for sub in ["bn", "ln", "bias", "logit_scale"])
    )
    include_condition = lambda n, p: not exclude_condition(n, p)

    gain_or_bias_params = _filter_parameters(model, exclude_condition)
    rest_params = _filter_parameters(model, include_condition)

    return torch.optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": weight_decay},
        ],
        lr=lr,
        betas=(0.9, 0.98),
        eps=1.0e-6,
    )


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# -------------------------
# JSONL indexed reading
# -------------------------
def build_line_offsets(jsonl_path: str) -> List[int]:
    offsets: List[int] = []
    with open(jsonl_path, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            offsets.append(pos)
    return offsets


class JsonlOffsetReader:
    def __init__(self, path: str, offsets: List[int]):
        self.path = path
        self.offsets = offsets
        self._fp = None

    def _ensure_open(self) -> None:
        if self._fp is None:
            self._fp = open(self.path, "rb")

    def read(self, idx: int) -> Dict[str, Any]:
        self._ensure_open()
        self._fp.seek(self.offsets[idx])
        line = self._fp.readline()
        return json.loads(line.decode("utf-8"))


# -------------------------
# Code-token helpers
# -------------------------
def build_code_tokens(codebook_vocab: int, total_levels: int, modality_index: bool = True) -> Tuple[List[str], List[str]]:
    """
    Return:
      level_indicators: ["a","b","c",...]
      code_tokens: ["<a0>", "<a1>", ...]
    """
    assert total_levels >= 1
    letters = [chr(ord("a") + i) for i in range(total_levels)]
    code_tokens: List[str] = []
    for li, ch in enumerate(letters):
        if modality_index and li == 0:
            for v in range(3):
                code_tokens.append(f"<{ch}{v}>")
        else:
            for v in range(codebook_vocab):
                code_tokens.append(f"<{ch}{v}>")
    return letters, code_tokens


def modality_id_from_masks(img_mask: torch.Tensor, txt_mask: torch.Tensor) -> torch.Tensor:
    """
    Map (img_mask, txt_mask) -> modality id:
      image-only -> 0
      text-only  -> 1
      both       -> 2
    """
    im = img_mask.view(-1)
    tm = txt_mask.view(-1)
    im1 = (im > 0).long()
    tm1 = (tm > 0).long()

    text_only = ((1 - im1) & tm1)
    image_only = (im1 & (1 - tm1))

    out = torch.full_like(im1, 2)
    out[text_only.bool()] = 1
    out[image_only.bool()] = 0
    return out


def _codes_row_to_str(codes_row: torch.Tensor, level_indicators: List[str]) -> str:
    toks = [f"<{level_indicators[i]}{int(codes_row[i])}>" for i in range(len(level_indicators))]
    return "".join(toks)


def build_labels_genius_style(
    p_codes: torch.Tensor,
    tokenizer: T5TokenizerFast,
    level_indicators: List[str],
    device: torch.device,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    label_str_list = [_codes_row_to_str(p_codes[i], level_indicators) for i in range(p_codes.size(0))]
    tok = tokenizer(
        label_str_list,
        padding=True,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    labels = tok.input_ids.to(device)
    labels[labels == tokenizer.pad_token_id] = ignore_index
    return labels


# -------------------------
# Similarity helpers
# -------------------------
def cosine_sim_matrix(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    a = F.normalize(a.float(), dim=-1, eps=eps)
    b = F.normalize(b.float(), dim=-1, eps=eps)
    return a @ b.t()


# -------------------------
# RQ cfg helper (fallback)
# -------------------------
def build_rq_cfg_for_codegen(emb_dim: int, codebook_vocab: int, codebook_level_wo_modality: int) -> SimpleNamespace:
    cfg = SimpleNamespace(
        rq_config=SimpleNamespace(
            emb_dim=int(emb_dim),
            normalize_before_rq=True,
            rq_loss_weight=1.0,
            mse_loss_weight=1.0,
            use_emb_encoder=True,
            use_modality_cond=True,
            modality_scale=0.1,
            temperature=0.01,
            codebook_vocab=int(codebook_vocab),
            codebook_level=int(codebook_level_wo_modality),
            modality_index=True,
            return_codes=True,
        )
    )
    return cfg


# -------------------------
# Run naming & checkpointing
# -------------------------
def build_run_name(args, emb_dim: int) -> str:
    """
    NOTE:
    - Tag reflects invariant pretrain settings.
    """
    t5_tag = infer_t5_size_tag(args.t5_name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    use_inv = bool(getattr(args, "use_invariant_pretrain", False))
    inv_ep = int(getattr(args, "invariant_pretrain_epochs", 0))
    inv_tag = f"inv{inv_ep}" if use_inv else "inv0"

    mi_tag = "mi1" if args.modality_index else "mi0"
    rk_tag = f"rk{args.lambda_rank:g}"

    return (
        f"{ts}_t5{t5_tag}_emb{emb_dim}_L{args.codebook_level_total}_V{args.codebook_vocab}_"
        f"p{args.num_prefix}_bs{args.train_bs}_lr{args.lr:g}_"
        f"{inv_tag}_{rk_tag}_{mi_tag}_seed{args.seed}"
    )


def save_checkpoint(
    args,
    out_dir: str,
    run_name: str,
    stage: str,
    epoch_idx: int,
    stage_epoch_idx: int,
    global_step: int,
    model: nn.Module,
    optimizer,
    scheduler,
    scaler,
    tokenizer: Optional[T5TokenizerFast] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    stage_norm = str(stage).lower()
    if stage_norm in ["invariant", "order_invariant", "inv"]:
        ckpt_name = f"pretrain_invariant_epoch_{stage_epoch_idx:03d}.pt"
    elif stage_norm in ["t5", "finetune", "finetune_t5"]:
        ckpt_name = f"finetune_t5_epoch_{stage_epoch_idx:03d}.pt"
    else:
        # fallback: keep stage name in filename to avoid confusion
        ckpt_name = f"{stage_norm}_epoch_{stage_epoch_idx:03d}.pt"

    save_path = os.path.join(out_dir, ckpt_name)

    payload = {
        "epoch": epoch_idx,
        "stage_epoch": stage_epoch_idx,
        "global_step": global_step,
        "stage": stage,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
        "run_name": run_name,
    }
    torch.save(payload, save_path)

    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)

    if is_main():
        print(f"[ckpt] saved: {save_path}")