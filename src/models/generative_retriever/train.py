#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py

Two-stage training:
  Stage-1: invariant pretrain (train model.order_invariant only)
  Stage-2: T5 training (train projector + T5 only)

Paths:
- YAML should use relative paths.
- Shell script injects absolute paths (recommended).
"""

from __future__ import annotations

import os
import time
import argparse
import warnings
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

from transformers import T5TokenizerFast
from omegaconf import OmegaConf

from models.residual_quantization.model import ResidualQuantizerModel as RQModel

from models.generative_retriever.utils import (
    IGNORE_INDEX,
    is_dist,
    is_main,
    ddp_barrier,
    setup_distributed,
    cleanup_distributed,
    unwrap_ddp,
    set_seed,
    build_run_name,
    save_checkpoint,
    build_line_offsets,
    build_code_tokens,
    modality_id_from_masks,
    build_labels_genius_style,
    cosine_sim_matrix,
    create_optimizer_genius,
    get_cosine_schedule_with_warmup,
    build_rq_cfg_for_codegen,
)

from models.generative_retriever.datasets import Dataset, collate_fn
from models.generative_retriever.model import T5ForGenerativeRetrieval


# -------------------------
# warnings / env
# -------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=r".*torch\.load.*weights_only=False.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*torch\.cuda\.amp\.GradScaler.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*torch\.cuda\.amp\.autocast.*deprecated.*", category=FutureWarning)


# =========================
# W&B helpers (rank0 only)
# =========================
def _get_wandb_cfg(cfg) -> dict:
    """Read cfg.wandb_config as plain dict (OmegaConf interpolation resolved)."""
    if hasattr(cfg, "wandb_config") and cfg.wandb_config is not None:
        return OmegaConf.to_container(cfg.wandb_config, resolve=True)
    return {}

def init_wandb_if_enabled(cfg):
    """
    Initialize W&B based on cfg.wandb_config.
    Only runs on rank0. Returns wandb module or None.
    """
    if not is_main():
        return None

    wb = _get_wandb_cfg(cfg)
    enabled = bool(wb.get("enabled", False))
    if not enabled:
        return None

    try:
        import wandb
    except Exception as e:
        print(f"[W&B] wandb import failed, disable logging. err={e}")
        return None

    wandb_key = wb.get("wandb_key", None)
    if wandb_key:
        os.environ["WANDB_API_KEY"] = str(wandb_key)
        try:
            wandb.login(key=str(wandb_key), relogin=True)
        except Exception:
            pass

    project = wb.get("wandb_project", None)
    if not project:
        raise ValueError("wandb_config.enabled=true but wandb_project is missing in YAML.")

    run_name = wb.get("experiment_name", None)
    notes = wb.get("notes", None)

    wandb.init(
        project=str(project),
        name=str(run_name) if run_name else None,
        notes=str(notes) if notes else None,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    print(f"[W&B] enabled: project={project}, name={run_name}")
    return wandb

def wandb_log(wandb_mod, metrics: dict, step: int):
    """Log metrics to W&B if enabled (rank0 only)."""
    if wandb_mod is None:
        return
    if not is_main():
        return
    try:
        if wandb_mod.run is None:
            return
        wandb_mod.log(metrics, step=step)
    except Exception:
        return


# -------------------------
# YAML merge helpers
# -------------------------
def _cli_overrides(argv: list[str], key: str) -> bool:
    # detects --key ... or --key=...
    flag = f"--{key}"
    return any(a == flag or a.startswith(flag + "=") for a in argv)

def load_cfg_and_merge_with_cli(parser: argparse.ArgumentParser) -> OmegaConf:
    """
    If --config_path is provided, load YAML as cfg.
    Then apply CLI overrides for known args (CLI > YAML).
    """
    args, _ = parser.parse_known_args()
    cfg = None

    if getattr(args, "config_path", None):
        cfg = OmegaConf.load(args.config_path)
    else:
        cfg = OmegaConf.create({})

    # Parse again to get full args
    full_args = parser.parse_args()

    # Apply YAML -> args for keys that exist in args and not overridden in CLI
    yaml_dict = OmegaConf.to_container(cfg, resolve=True) if cfg is not None else {}
    for k, v in (yaml_dict or {}).items():
        if hasattr(full_args, k) and (not _cli_overrides(os.sys.argv[1:], k)):
            setattr(full_args, k, v)

    # Rebuild cfg from merged args + keep nested blocks from YAML (experiment/model/wandb_config)
    merged_flat = OmegaConf.create(vars(full_args))
    cfg = OmegaConf.merge(cfg, merged_flat)
    return cfg


# -------------------------
# RQ load
# -------------------------
def load_rq_model(args: argparse.Namespace, device: torch.device) -> RQModel:
    ckpt = torch.load(args.rq_ckpt, map_location="cpu")

    cfg = None
    if getattr(args, "rq_yaml", None) and os.path.exists(args.rq_yaml):
        cfg = OmegaConf.load(args.rq_yaml)
        if hasattr(args, "emb_dim") and args.emb_dim is not None:
            cfg.rq_config.emb_dim = int(args.emb_dim)
        cfg.rq_config.return_codes = True

    if cfg is None and isinstance(ckpt, dict):
        for k in ["cfg", "config", "args"]:
            if k in ckpt:
                cfg = ckpt[k]
                break

    if cfg is None:
        cfg = build_rq_cfg_for_codegen(
            emb_dim=args.emb_dim,
            codebook_vocab=args.codebook_vocab,
            codebook_level_wo_modality=args.codebook_level_wo_modality,
        )

    rq = RQModel(cfg).to(device)
    rq.eval()

    if isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    missing, unexpected = rq.load_state_dict(sd, strict=False)
    if is_main():
        print(f"[RQ load] missing={len(missing)} unexpected={len(unexpected)}")

    return rq


# -------------------------
# stage control
# -------------------------
def set_train_stage(model: torch.nn.Module, stage: str) -> None:
    m = unwrap_ddp(model)

    for p in m.parameters():
        p.requires_grad = False

    if stage == "invariant":
        for p in m.order_invariant.parameters():
            p.requires_grad = True
    elif stage == "t5":
        for p in m.projector.parameters():
            p.requires_grad = True
        for p in m.t5.parameters():
            p.requires_grad = True
    else:
        raise ValueError(stage)


def build_optim_sched_for_stage(args, model: torch.nn.Module, stage: str, steps_total: int):
    if stage == "invariant":
        opt = create_optimizer_genius(model, args.invariant_lr, args.invariant_wd)
    else:
        opt = create_optimizer_genius(model, args.lr, args.weight_decay)
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=args.warmup_steps, num_training_steps=steps_total)
    return opt, sch


# -------------------------
# online p_codes (DEDUP)
# -------------------------
@torch.no_grad()
def online_build_pcodes(args, model: torch.nn.Module, p_emb, p_img_mask, p_txt_mask) -> torch.Tensor:
    rq = unwrap_ddp(model).rq
    out_codes = rq.infer(p_emb, p_img_mask, p_txt_mask, return_codes=True, normalize_out=False)
    p_codes = out_codes["code"].to(torch.long)

    if p_codes.size(1) == args.codebook_level_total - 1 and args.modality_index:
        mid = modality_id_from_masks(p_img_mask, p_txt_mask).to(p_codes.device)
        p_codes = torch.cat([mid.unsqueeze(1), p_codes], dim=1)

    if p_codes.size(1) != args.codebook_level_total:
        raise RuntimeError(
            f"[p_codes shape mismatch] got {tuple(p_codes.shape)} expect L={args.codebook_level_total}. "
            f"Check RQ codebook_level/modality alignment."
        )
    return p_codes


@torch.no_grad()
def online_build_pcodes_and_labels(args, model, tokenizer, level_indicators, p_emb, p_img_mask, p_txt_mask, device):
    p_codes = online_build_pcodes(args, model, p_emb, p_img_mask, p_txt_mask)
    labels_ids = build_labels_genius_style(
        p_codes=p_codes,
        tokenizer=tokenizer,
        level_indicators=level_indicators,
        device=device,
        ignore_index=IGNORE_INDEX,
    )
    return p_codes, labels_ids


# -------------------------
# Stage-1 invariant step
# -------------------------
def train_invariant_step(args, model, q_emb, p_emb, p_codes, optimizer, scheduler, scaler, device):
    m = unwrap_ddp(model)

    with autocast(enabled=(device.type == "cuda")):
        scores_in_batch, _ = m.compute_order_invariant_inbatch_scores(q_emb=q_emb, p_codes=p_codes)
        labels_s = torch.arange(scores_in_batch.size(0), device=device)

        loss_invariant = F.cross_entropy(scores_in_batch / float(args.invariant_tau), labels_s)

        if float(args.lambda_invariant_kd) > 0:
            with torch.no_grad():
                sim_full = cosine_sim_matrix(q_emb, p_emb).float()
                sim_full.fill_diagonal_(float("-inf"))
                neg_idx = sim_full.argmax(dim=1)

            pos_s = scores_in_batch.diag()
            neg_s = scores_in_batch[torch.arange(scores_in_batch.size(0), device=device), neg_idx]
            loss_kd = F.margin_ranking_loss(
                pos_s,
                neg_s,
                torch.ones_like(pos_s),
                margin=float(args.invariant_kd_margin),
                reduction="mean",
            )
        else:
            loss_kd = torch.zeros((), device=device)

        loss_total = float(args.lambda_invariant) * loss_invariant + float(args.lambda_invariant_kd) * loss_kd

    scaler.scale(loss_total).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(m.order_invariant.parameters(), args.max_grad_norm)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()

    with torch.no_grad():
        pred = scores_in_batch.argmax(dim=-1)
        acc = (pred == torch.arange(pred.size(0), device=device)).float().mean().item()

    return {
        "loss": float(loss_total.detach().cpu()),
        "loss_invariant": float(loss_invariant.detach().cpu()),
        "loss_kd": float(loss_kd.detach().cpu()),
        "acc1": float(acc),
    }


# -------------------------
# argparse
# -------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # NEW: yaml config
    parser.add_argument("--config_path", type=str, default=None, help="Path to YAML config. CLI overrides YAML.")

    # paths
    parser.add_argument("--rq_yaml", type=str, required=False)
    parser.add_argument("--train_jsonl", type=str, required=False)
    parser.add_argument("--query_store", type=str, required=False)
    parser.add_argument("--cand_store", type=str, required=False)
    parser.add_argument("--rq_ckpt", type=str, required=False)
    parser.add_argument("--out_dir", type=str, required=False)

    # codebook
    parser.add_argument("--codebook_vocab", type=int, default=4096)
    parser.add_argument("--codebook_level_total", type=int, default=9)
    parser.add_argument("--codebook_level_wo_modality", type=int, default=8)

    # modality token control
    parser.add_argument("--modality_index", action="store_true", default=True)
    parser.add_argument("--no_modality_index", dest="modality_index", action="store_false")

    # model
    parser.add_argument("--t5_name", type=str, default="google-t5/t5-small")
    parser.add_argument("--num_prefix", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=2.0)

    # training (shared)
    parser.add_argument("--t5_epochs", type=int, default=30)
    parser.add_argument("--train_bs", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2023)

    # Stage-1: invariant
    parser.add_argument("--use_invariant_pretrain", action="store_true", default=True)
    parser.add_argument("--invariant_pretrain_epochs", type=int, default=10)
    parser.add_argument("--invariant_lr", type=float, default=2e-4)
    parser.add_argument("--invariant_wd", type=float, default=1e-4)
    parser.add_argument("--invariant_tau", type=float, default=0.01)
    parser.add_argument("--lambda_invariant", type=float, default=1.0)
    parser.add_argument("--lambda_invariant_kd", type=float, default=1.0)
    parser.add_argument("--invariant_kd_margin", type=float, default=0.1)
    parser.add_argument("--invariant_train_bs", type=int, default=512)

    # Stage-2 ranking loss
    parser.add_argument("--lambda_rank", type=float, default=1.0)
    parser.add_argument("--teacher_margin_scale", type=float, default=1.0)
    parser.add_argument("--teacher_margin_min", type=float, default=0.0)
    parser.add_argument("--mask_modality_pos0", type=int, default=1)

    parser.add_argument("--save_tokenizer_every_ckpt", action="store_true", default=False)

    return parser


def main():
    parser = build_parser()
    cfg = load_cfg_and_merge_with_cli(parser)
    args = argparse.Namespace(**OmegaConf.to_container(cfg, resolve=True))

    # basic required checks (after YAML merge)
    required = ["rq_yaml", "train_jsonl", "query_store", "cand_store", "rq_ckpt", "out_dir"]
    for k in required:
        if getattr(args, k, None) is None:
            raise ValueError(f"Missing required config field: {k}. Provide via YAML or CLI --{k}.")

    local_rank, world, device = setup_distributed()
    set_seed(int(args.seed))

    if is_main():
        os.makedirs(args.out_dir, exist_ok=True)
        print(f"[env] world={world} device={device} out_dir={args.out_dir}")

    # ---- W&B init (rank0 only) ----
    wandb_mod = init_wandb_if_enabled(cfg)

    # stores
    query_store = torch.load(args.query_store, map_location="cpu")
    cand_store = torch.load(args.cand_store, map_location="cpu")
    emb_dim = int(query_store["emb"].shape[1])
    args.emb_dim = emb_dim

    # tokenizer + code tokens
    tokenizer = T5TokenizerFast.from_pretrained(args.t5_name, model_max_length=512)
    level_indicators, code_tokens = build_code_tokens(
        codebook_vocab=int(args.codebook_vocab),
        total_levels=int(args.codebook_level_total),
        modality_index=bool(args.modality_index),
    )
    num_added = tokenizer.add_tokens(code_tokens)
    if is_main():
        print(f"[Tokenizer] Added {num_added} code tokens. Vocab size: {len(tokenizer)}")

    # RQ
    rq_for_input = load_rq_model(args, device=device)
    rq_for_init = rq_for_input

    # model
    model = T5ForGenerativeRetrieval(
        emb_dim=emb_dim,
        num_prefix=int(args.num_prefix),
        t5_name=args.t5_name,
        tokenizer=tokenizer,
        codebook_vocab=int(args.codebook_vocab),
        total_levels=int(args.codebook_level_total),
        modality_index=bool(args.modality_index),
        rq_for_init=rq_for_init,
        rq_for_input=rq_for_input,
    ).to(device)

    if is_dist():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # dataset / loaders
    if is_main():
        print("[data] building line offsets...")
    offsets = build_line_offsets(args.train_jsonl)
    dataset = Dataset(args.train_jsonl, query_store, cand_store, offsets)

    invariant_bs = int(args.invariant_train_bs) if int(args.invariant_train_bs) > 0 else int(args.train_bs)
    t5_bs = int(args.train_bs)

    sampler_invariant = DistributedSampler(dataset, shuffle=True) if is_dist() else None
    sampler_t5 = DistributedSampler(dataset, shuffle=True) if is_dist() else None

    loader_invariant = DataLoader(
        dataset,
        batch_size=invariant_bs,
        sampler=sampler_invariant,
        shuffle=(sampler_invariant is None),
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    loader_t5 = DataLoader(
        dataset,
        batch_size=t5_bs,
        sampler=sampler_t5,
        shuffle=(sampler_t5 is None),
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    if is_main():
        print(
            f"[data] invariant_bs={invariant_bs} T5_bs={t5_bs} "
            f"steps/epoch(invariant)={len(loader_invariant)} steps/epoch(t5)={len(loader_t5)}"
        )

    run_name = build_run_name(args, emb_dim)
    if is_main():
        print(f"[run] {run_name}")

    do_invariant = bool(args.use_invariant_pretrain) and int(args.invariant_pretrain_epochs) > 0
    invariant_epochs = int(args.invariant_pretrain_epochs) if do_invariant else 0
    t5_epochs = int(args.t5_epochs)

    scaler = GradScaler(enabled=(device.type == "cuda"))
    global_step = 0
    start_time = time.time()

    if is_main():
        print(f"[train] Stage-1 invariant epochs={invariant_epochs}, Stage-2 T5 epochs={t5_epochs}")

    # -------------------------
    # Stage-1: invariant
    # -------------------------
    if do_invariant:
        stage = "invariant"
        set_train_stage(model, stage)

        steps_total = len(loader_invariant) * invariant_epochs
        optim_inv, sched_inv = build_optim_sched_for_stage(args, unwrap_ddp(model), stage, steps_total)

        for epoch in range(invariant_epochs):
            if sampler_invariant is not None:
                sampler_invariant.set_epoch(epoch)

            unwrap_ddp(model).train()

            m_loss = m_linv = m_lkd = m_acc = 0.0
            m_n = 0

            for it, batch in enumerate(loader_invariant):
                q_emb = batch["q_emb"].to(device, non_blocking=True)
                p_emb = batch["p_emb"].to(device, non_blocking=True)
                p_img_mask = batch["p_img_mask"].to(device, non_blocking=True)
                p_txt_mask = batch["p_txt_mask"].to(device, non_blocking=True)

                p_codes = online_build_pcodes(args, model, p_emb, p_img_mask, p_txt_mask)

                out = train_invariant_step(
                    args=args,
                    model=model,
                    q_emb=q_emb,
                    p_emb=p_emb,
                    p_codes=p_codes,
                    optimizer=optim_inv,
                    scheduler=sched_inv,
                    scaler=scaler,
                    device=device,
                )

                global_step += 1
                m_loss += out["loss"]
                m_linv += out["loss_invariant"]
                m_lkd += out["loss_kd"]
                m_acc += out["acc1"]
                m_n += 1

                # ---- W&B (Stage-1 step) ----
                if is_main() and (global_step % int(args.print_freq) == 0):
                    lr_curr = optim_inv.param_groups[0]["lr"]
                    wandb_log(
                        wandb_mod,
                        {
                            "stage": "invariant",
                            "invariant/epoch": epoch,
                            "invariant/step": global_step,
                            "invariant/loss": out["loss"],
                            "invariant/loss_invariant": out["loss_invariant"],
                            "invariant/loss_kd": out["loss_kd"],
                            "invariant/acc1": out["acc1"],
                            "invariant/lr": float(lr_curr),
                        },
                        step=global_step,
                    )

                    print(
                        f"[Stage-INVARIANT][E{epoch:02d}][S{global_step:07d}] "
                        f"loss={out['loss']:.4f} loss_inv={out['loss_invariant']:.4f} loss_kd={out['loss_kd']:.4f} "
                        f"@1={out['acc1']:.4f} LR={lr_curr:.6g} elapsed={(time.time()-start_time)/60:.1f}m"
                    )

            # ---- W&B (Stage-1 epoch summary) ----
            if is_main():
                n = max(1, m_n)
                lr_curr = optim_inv.param_groups[0]["lr"]
                wandb_log(
                    wandb_mod,
                    {
                        "stage": "invariant",
                        "invariant/epoch_avg_loss": m_loss / n,
                        "invariant/epoch_avg_loss_invariant": m_linv / n,
                        "invariant/epoch_avg_loss_kd": m_lkd / n,
                        "invariant/epoch_avg_acc1": m_acc / n,
                        "invariant/epoch": epoch,
                        "invariant/lr": float(lr_curr),
                    },
                    step=global_step,
                )
                print(
                    f"[Stage-INVARIANT][Epoch {epoch+1:03d}/{invariant_epochs:03d} DONE] "
                    f"Avg loss={m_loss/n:.4f} loss_inv={m_linv/n:.4f} loss_kd={m_lkd/n:.4f} @1={m_acc/n:.4f} "
                    f"LR={lr_curr:.6g}"
                )

            if is_main() and ((epoch + 1) % int(args.save_every) == 0):
                save_checkpoint(
                    args=args,
                    out_dir=args.out_dir,
                    run_name=run_name,
                    stage="invariant",
                    epoch_idx=epoch,
                    stage_epoch_idx=epoch + 1,
                    global_step=global_step,
                    model=unwrap_ddp(model),
                    optimizer=optim_inv,
                    scheduler=sched_inv,
                    scaler=scaler,
                    tokenizer=(tokenizer if bool(args.save_tokenizer_every_ckpt) else None),
                )
            ddp_barrier()

    # -------------------------
    # Stage-2: T5
    # -------------------------
    if t5_epochs > 0:
        stage = "t5"
        set_train_stage(model, stage)

        steps_total = len(loader_t5) * t5_epochs
        optim_t5, sched_t5 = build_optim_sched_for_stage(args, unwrap_ddp(model), stage, steps_total)

        for e2 in range(t5_epochs):
            sampler_epoch = invariant_epochs + e2
            if sampler_t5 is not None:
                sampler_t5.set_epoch(sampler_epoch)

            unwrap_ddp(model).train()
            meters = {"loss": 0.0, "loss_rank": 0.0, "r1": 0.0, "l1": 0.0, "l12": 0.0, "l123": 0.0, "n": 0}

            for it, batch in enumerate(loader_t5):
                q_emb = batch["q_emb"].to(device, non_blocking=True)
                p_emb = batch["p_emb"].to(device, non_blocking=True)

                q_img_mask = batch["q_img_mask"].to(device, non_blocking=True)
                q_txt_mask = batch["q_txt_mask"].to(device, non_blocking=True)
                p_img_mask = batch["p_img_mask"].to(device, non_blocking=True)
                p_txt_mask = batch["p_txt_mask"].to(device, non_blocking=True)

                p_codes, labels_ids = online_build_pcodes_and_labels(
                    args, model, tokenizer, level_indicators, p_emb, p_img_mask, p_txt_mask, device
                )

                m = unwrap_ddp(model)

                with autocast(enabled=(device.type == "cuda")):
                    pref = m(
                        q_emb=q_emb,
                        p_emb=p_emb,
                        q_img_mask=q_img_mask,
                        q_txt_mask=q_txt_mask,
                        p_img_mask=p_img_mask,
                        p_txt_mask=p_txt_mask,
                        alpha=float(args.alpha),
                    )
                    loss_ce, mets, pred_ids = m.compute_loss_and_metrics(pref, labels_ids)

                with torch.no_grad():
                    sim_full = cosine_sim_matrix(q_emb, p_emb)
                    B = sim_full.size(0)
                    sim_pos = sim_full.diag()
                    sim_for_neg = sim_full.clone()
                    sim_for_neg.fill_diagonal_(-1e9)
                    neg_idx = sim_for_neg.argmax(dim=1)
                    sim_neg = sim_full[torch.arange(B, device=device), neg_idx]

                    margin = (sim_pos - sim_neg) * float(args.teacher_margin_scale)
                    margin = margin.clamp(min=float(args.teacher_margin_min))

                p_codes_neg = p_codes.index_select(0, neg_idx.to(p_codes.device))
                labels_neg = build_labels_genius_style(
                    p_codes=p_codes_neg,
                    tokenizer=tokenizer,
                    level_indicators=level_indicators,
                    device=device,
                    ignore_index=IGNORE_INDEX,
                )

                s_pos = m.sequence_logprob_score(
                    pref, labels_ids, ignore_index=IGNORE_INDEX, mask_pos0=bool(int(args.mask_modality_pos0))
                )
                s_neg = m.sequence_logprob_score(
                    pref, labels_neg, ignore_index=IGNORE_INDEX, mask_pos0=bool(int(args.mask_modality_pos0))
                )
                loss_rank = F.softplus(margin - (s_pos - s_neg)).mean()

                loss_total = loss_ce + float(args.lambda_rank) * loss_rank

                scaler.scale(loss_total).backward()
                scaler.unscale_(optim_t5)
                torch.nn.utils.clip_grad_norm_(m.parameters(), float(args.max_grad_norm))
                scaler.step(optim_t5)
                scaler.update()
                optim_t5.zero_grad(set_to_none=True)
                sched_t5.step()

                meters["loss"] += float(loss_ce.detach())
                meters["loss_rank"] += float(loss_rank.detach())
                meters["r1"] += float(mets["R_at_1"].detach())
                meters["l1"] += float(mets["Level1_acc"].detach())
                meters["l12"] += float(mets["Level12_acc"].detach())
                meters["l123"] += float(mets["Level123_acc"].detach())
                meters["n"] += 1
                global_step += 1

                if is_main() and (global_step % int(args.print_freq) == 0):
                    n = max(1, meters["n"])
                    lr_curr = optim_t5.param_groups[0]["lr"]

                    # ---- W&B (Stage-2 step) ----
                    wandb_log(
                        wandb_mod,
                        {
                            "stage": "t5",
                            "t5/epoch": e2,
                            "t5/step": global_step,
                            "t5/loss_ce": meters["loss"] / n,
                            "t5/loss_rank": meters["loss_rank"] / n,
                            "t5/R@1": meters["r1"] / n,
                            "t5/Level1_acc": meters["l1"] / n,
                            "t5/Level12_acc": meters["l12"] / n,
                            "t5/Level123_acc": meters["l123"] / n,
                            "t5/lr": float(lr_curr),
                        },
                        step=global_step,
                    )

                    print(
                        f"[Stage-T5][E{e2:02d}][S{global_step:07d}] "
                        f"Loss={meters['loss']/n:.4f} Rankloss={meters['loss_rank']/n:.4f} "
                        f"R@1={meters['r1']/n:.4f} LR={lr_curr:.6g} elapsed={(time.time()-start_time)/60:.1f}m"
                    )

                    pred_sample = pred_ids[0].detach().cpu().tolist()
                    lab_sample = labels_ids[0].detach().cpu().tolist()
                    lab_sample = [tokenizer.pad_token_id if x == IGNORE_INDEX else x for x in lab_sample]
                    print(f"Pred: {tokenizer.decode(pred_sample, skip_special_tokens=False)}")
                    print(f"Ans : {tokenizer.decode(lab_sample,  skip_special_tokens=False)}")
                    print("-" * 80)

                    for k in meters:
                        meters[k] = 0.0

            # ---- W&B (Stage-2 epoch summary) ----
            if is_main():
                n = max(1, meters["n"])
                lr_curr = optim_t5.param_groups[0]["lr"]
                wandb_log(
                    wandb_mod,
                    {
                        "stage": "t5",
                        "t5/epoch": e2,
                        "t5/epoch_end_lr": float(lr_curr),
                    },
                    step=global_step,
                )

            if is_main() and ((e2 + 1) % int(args.save_every) == 0):
                save_checkpoint(
                    args=args,
                    out_dir=args.out_dir,
                    run_name=run_name,
                    stage="t5",
                    epoch_idx=invariant_epochs + e2,
                    stage_epoch_idx=e2 + 1,
                    global_step=global_step,
                    model=unwrap_ddp(model),
                    optimizer=optim_t5,
                    scheduler=sched_t5,
                    scaler=scaler,
                    tokenizer=(tokenizer if bool(args.save_tokenizer_every_ckpt) else None),
                )
            ddp_barrier()

    if is_main():
        print("[train] done.")
        if wandb_mod is not None:
            try:
                wandb_mod.finish()
            except Exception:
                pass

    cleanup_distributed()


if __name__ == "__main__":
    main()