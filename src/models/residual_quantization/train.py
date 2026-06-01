# train.py
# -*- coding: utf-8 -*-
"""
Training Code for Residual Quantization

This module implements the training pipeline for the Residual Quantization model.
It includes functionality for:
- Setting up distributed training (DDP initialization/cleanup)
- Loading embedding stores and building datasets/dataloaders
- Creating and managing optimizers and AMP settings
- Training loop implementation with gradient accumulation
- Checkpoint saving (per-epoch and last)
- Logging metrics and optional debug printing (RQ codes / modality clustering stats)
"""

import os
import argparse

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from omegaconf import OmegaConf
from tqdm import tqdm

from dataset import RQTrainDataset, collate_rq_train
from model import ResidualQuantizerModel
from utils import (
    ddp_init,
    ddp_cleanup,
    is_dist,
    get_rank,
    get_world_size,
    save_ckpt,
    codes_to_token_str,
    update_confusion,
    resolve_paths_with_roots,
    compute_rq_stats,
)

import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*You are using `torch\.load` with `weights_only=False`.*",
    category=FutureWarning,
)


# =========================
# W&B helpers (rank0 only)
# =========================
def _get_wandb_cfg(cfg) -> dict:
    """
    Read cfg.wandb_config as a plain dict (OmegaConf interpolation resolved).
    If not present, return empty dict.
    """
    if hasattr(cfg, "wandb_config") and cfg.wandb_config is not None:
        return OmegaConf.to_container(cfg.wandb_config, resolve=True)
    return {}


def init_wandb_if_enabled(cfg):
    """
    Initialize Weights & Biases based on cfg.wandb_config.
    Only runs on rank0. Returns wandb module or None.
    """
    if get_rank() != 0:
        return None

    wb = _get_wandb_cfg(cfg)
    enabled = bool(wb.get("enabled", False))
    if not enabled:
        return None

    # Lazy import to avoid dependency on non-rank0 or if wandb not installed
    try:
        import wandb
    except Exception as e:
        print(f"[W&B] wandb import failed, disable logging. err={e}")
        return None

    # If wandb_key is provided in YAML, set env + login once on rank0
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

    wandb.init(
        project=str(project),
        name=str(run_name) if run_name else None,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    print(f"[W&B] enabled: project={project}, name={run_name}")
    return wandb


def wandb_log(wandb_mod, metrics: dict, step: int):
    """Log metrics to W&B if enabled (rank0 only)."""
    if wandb_mod is None:
        return
    if get_rank() != 0:
        return
    try:
        if wandb_mod.run is None:
            return
        wandb_mod.log(metrics, step=step)
    except Exception:
        return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default=None, help="Path to the config file (legacy arg).")
    parser.add_argument("--config_path", default=None, help="Path to the config file (preferred).")

    parser.add_argument(
        "--genir_dir",
        type=str,
        default=None,
        help="Root directory for embeddings/checkpoints/logs (used to resolve relative paths in YAML).",
    )
    parser.add_argument(
        "--mbeir_data_dir",
        type=str,
        default=None,
        help="Root directory for M-BEIR dataset (used to resolve relative paths in YAML).",
    )

    args = parser.parse_args()

    config_path = args.config_path or args.config
    if config_path is None:
        raise ValueError("You must provide --config_path (preferred) or --config (legacy).")

    cfg = OmegaConf.load(config_path)

    # - If user does not pass roots, fallback to YAML absolute paths as-is.
    if args.genir_dir is not None and args.mbeir_data_dir is not None:
        cfg = resolve_paths_with_roots(cfg, genir_dir=args.genir_dir, mbeir_data_dir=args.mbeir_data_dir)

    ddp_on, local_rank = ddp_init(cfg.dist.backend, cfg.dist.dist_url)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # DDP + GatherLayer requires identical batch shapes across ranks
    if is_dist() and not bool(cfg.dataloader.drop_last):
        raise ValueError("DDP + all_gather requires dataloader.drop_last=True (batch shapes must match across ranks).")

    # Seed
    seed = int(cfg.seed) + get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ---- W&B init (rank0 only) ----
    wandb_mod = init_wandb_if_enabled(cfg)

    if get_rank() == 0:
        print("[Config]")
        print(OmegaConf.to_yaml(cfg))
        if args.genir_dir is not None and args.mbeir_data_dir is not None:
            print("[Paths]")
            print("  query_emb_pt:", cfg.paths.query_emb_pt)
            print("  pool_emb_pt :", cfg.paths.pool_emb_pt)
            print("  query_jsonl :", cfg.paths.query_jsonl)
            print("  output_dir  :", cfg.paths.output_dir)

    # Load embedding dicts (CPU)
    query_dict = torch.load(cfg.paths.query_emb_pt, map_location="cpu")
    pool_dict = torch.load(cfg.paths.pool_emb_pt, map_location="cpu")

    # Auto infer emb_dim
    if cfg.rq_config.emb_dim is None:
        cfg.rq_config.emb_dim = int(query_dict["emb"].shape[1])
        if get_rank() == 0:
            print(f"[Auto] emb_dim = {cfg.rq_config.emb_dim}")

    # Dataset
    neg_k = int(getattr(cfg.data, "neg_k", 31))
    expand_multi_pos = bool(getattr(cfg.data, "expand_multi_pos", True))

    ds = RQTrainDataset(
        subset_query_jsonl=cfg.paths.query_jsonl,
        query_dict=query_dict,
        pool_dict=pool_dict,
        neg_k=neg_k,
        seed=int(cfg.seed),
        expand_multi_pos=expand_multi_pos,
    )

    if is_dist():
        sampler = DistributedSampler(
            ds,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
            drop_last=bool(cfg.dataloader.drop_last),
        )
    else:
        sampler = None

    dl = DataLoader(
        ds,
        batch_size=int(cfg.train.batch_size),
        sampler=sampler,
        shuffle=False,
        num_workers=int(cfg.dataloader.num_workers),
        pin_memory=bool(cfg.dataloader.pin_memory),
        drop_last=bool(cfg.dataloader.drop_last),
        collate_fn=collate_rq_train,
    )

    # Model
    model = ResidualQuantizerModel(cfg).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.optim.lr),
        weight_decay=float(cfg.optim.weight_decay),
        betas=tuple(cfg.optim.betas),
        eps=float(cfg.optim.eps),
    )

    # AMP  (DO NOT CHANGE per your request)
    amp_mode = str(cfg.train.amp).lower()
    use_amp = amp_mode in ["bf16", "fp16"]
    amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_mode == "fp16"))

    out_dir = cfg.paths.output_dir
    os.makedirs(out_dir, exist_ok=True)

    global_step = 0
    model.train()

    for epoch in range(int(cfg.train.epochs)):
        if is_dist():
            sampler.set_epoch(epoch)
        # IMPORTANT: enable deterministic negative resampling per epoch
        ds.set_epoch(epoch)

        cm_q_epoch = torch.zeros((3, 3), dtype=torch.long, device=device)
        cm_p_pos_epoch = torch.zeros((3, 3), dtype=torch.long, device=device)
        cm_p_neg_epoch = torch.zeros((3, 3), dtype=torch.long, device=device)

        pbar = tqdm(dl, desc=f"Epoch {epoch}", dynamic_ncols=True) if get_rank() == 0 else dl

        for batch in pbar:
            # Embedding names (mm removed)
            query_emb = batch["query_emb"].to(device, non_blocking=True)             # [B, D]
            cand_emb = batch["cand_emb"].to(device, non_blocking=True)               # [P, D]
            pos_index = batch["pos_index"].to(device, non_blocking=True)             # [B]
            query_pos_did = batch["query_pos_did"].to(device, non_blocking=True).long()  # [B]
            cand_did = batch["cand_did"].to(device, non_blocking=True).long()        # [P]

            query_img_mask = batch["query_img_mask"].to(device, non_blocking=True)
            query_txt_mask = batch["query_txt_mask"].to(device, non_blocking=True)
            cand_img_mask = batch["cand_img_mask"].to(device, non_blocking=True)
            cand_txt_mask = batch["cand_txt_mask"].to(device, non_blocking=True)

            # Build cand_is_pos for pos/neg debug split
            B = query_emb.size(0)
            P = cand_emb.size(0)
            assert P % B == 0, f"P must be multiple of B, got P={P}, B={B}"
            cand_is_pos = torch.zeros((P,), dtype=torch.bool, device=device)
            cand_is_pos[pos_index] = True

            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(
                    query_emb=query_emb,
                    cand_emb=cand_emb,
                    pos_index=pos_index,
                    query_pos_did=query_pos_did,
                    cand_did=cand_did,
                    query_img_mask=query_img_mask,
                    query_txt_mask=query_txt_mask,
                    cand_img_mask=cand_img_mask,
                    cand_txt_mask=cand_txt_mask,
                )
                loss = out["loss"] / int(cfg.train.grad_accum)

            # Optional code printing
            if (
                get_rank() == 0
                and bool(getattr(cfg.rq_config, "return_codes", False))
                and (global_step % int(getattr(cfg.train, "print_codes_every", 200)) == 0)
            ):
                n_show = int(getattr(cfg.train, "print_codes_samples", 3))
                Is_q = out["Is_q_full"].detach().cpu()  # [B, L]
                Is_p = out["Is_p_full"].detach().cpu()  # [P, L]

                query_id_str = batch["query_id_str"]
                cand_id_str_all = batch["cand_id_str_all"]

                print("\n[Code DBG] ===== RQ codes sample =====")
                for i in range(min(n_show, Is_q.size(0))):
                    q_code = codes_to_token_str(
                        Is_q[i],
                        modality_index=bool(cfg.rq_config.modality_index),
                        codebook_level=int(cfg.rq_config.codebook_level),
                    )
                    print(f"[Query] {query_id_str[i]} -> {q_code}")

                    K = int(getattr(cfg.data, "neg_k", 0))
                    j_pos = i * (1 + K)
                    if j_pos < Is_p.size(0):
                        p_pos_code = codes_to_token_str(
                            Is_p[j_pos],
                            modality_index=bool(cfg.rq_config.modality_index),
                            codebook_level=int(cfg.rq_config.codebook_level),
                        )
                        print(f"  [Cand-pos] {cand_id_str_all[j_pos]} -> {p_pos_code}")

                    if K > 0:
                        j_neg = j_pos + 1
                        if j_neg < Is_p.size(0):
                            p_neg_code = codes_to_token_str(
                                Is_p[j_neg],
                                modality_index=bool(cfg.rq_config.modality_index),
                                codebook_level=int(cfg.rq_config.codebook_level),
                            )
                            print(f"  [Cand-neg] {cand_id_str_all[j_neg]} -> {p_neg_code}")

            # Optional clustering stats
            if ("Is0_q" in out) and ("Is0_p_all" in out) and ("q_mid" in out) and ("p_mid_all" in out):
                cm_q_epoch = update_confusion(cm_q_epoch, out["q_mid"], out["Is0_q"])
                pmid = out["p_mid_all"]
                pis0 = out["Is0_p_all"]

                if cand_is_pos.any():
                    cm_p_pos_epoch = update_confusion(cm_p_pos_epoch, pmid[cand_is_pos], pis0[cand_is_pos])
                if (~cand_is_pos).any():
                    cm_p_neg_epoch = update_confusion(cm_p_neg_epoch, pmid[~cand_is_pos], pis0[~cand_is_pos])

            # Backward
            if amp_mode == "fp16":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (global_step + 1) % int(cfg.train.grad_accum) == 0:
                if amp_mode == "fp16":
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                optim.zero_grad(set_to_none=True)

            # Logging (console + wandb)
            if get_rank() == 0 and (global_step % int(cfg.train.log_every) == 0):
                msg = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": float(out["loss"].detach().cpu()),
                    "cl": float(out["cl_loss"].cpu()),
                    "rq": float(out["rq_loss"].cpu()),
                    "mse": float(out["mse_loss"].cpu()),
                    "acc": float(out["acc"].cpu()),
                    "acc_org": float(out["acc_org"].cpu()),
                    "lr": float(optim.param_groups[0]["lr"]),
                }
                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix(msg)

                rq_stats = {}
                if bool(getattr(cfg.rq_config, "return_codes", False)):
                    Is_q = out.get("Is_q_full", None)  # [B, L]
                    Is_p = out.get("Is_p_full", None)  # [P, L]
                    if Is_q is not None and Is_p is not None:
                        Is_all = torch.cat([Is_q, Is_p], dim=0).detach().cpu()
                        rq_stats = compute_rq_stats(Is_all, codebook_vocab=int(cfg.rq_config.codebook_vocab))

                wandb_log(
                    wandb_mod,
                    {
                        "train/epoch": msg["epoch"],
                        "train/step": msg["step"],
                        "train/loss": msg["loss"],
                        "train/cl_loss": msg["cl"],
                        "train/rq_loss": msg["rq"],
                        "train/mse_loss": msg["mse"],
                        "train/acc": msg["acc"],
                        "train/acc_org": msg["acc_org"],
                        "train/lr": msg["lr"],
                        **rq_stats,
                    },
                    step=global_step,
                )

            global_step += 1

        # Sync confusion matrices
        if is_dist():
            dist.all_reduce(cm_q_epoch, op=dist.ReduceOp.SUM)
            dist.all_reduce(cm_p_pos_epoch, op=dist.ReduceOp.SUM)
            dist.all_reduce(cm_p_neg_epoch, op=dist.ReduceOp.SUM)

        # Save per epoch
        if get_rank() == 0:
            save_ckpt(os.path.join(out_dir, f"ckpt_epoch{epoch}.pt"), model, optim, global_step, epoch, cfg)
            save_ckpt(os.path.join(out_dir, "ckpt_last.pt"), model, optim, global_step, epoch, cfg)

    ddp_cleanup()
    if get_rank() == 0:
        print(f"[Done] checkpoints saved to: {out_dir}")
        if wandb_mod is not None:
            try:
                wandb_mod.finish()
            except Exception:
                pass


if __name__ == "__main__":
    main()