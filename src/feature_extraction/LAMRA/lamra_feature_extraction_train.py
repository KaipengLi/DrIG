#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
lamra_feature_extraction_train.py

Generic LamRA-style embedding extraction for M-BEIR.

Core behaviors:
- Embeddings are NOT normalized (normalize=False in model.inference()).
- IDs follow M-BEIR hashing convention:
    qid_hash = dataset_id * 500000 + within_id
    did_hash = dataset_id * 10000000 + within_id
- Modality is exported as:
    img_mask / txt_mask / modality_id
- Optional GENIUS-style instruction prompting is supported (TSV-based).

Config philosophy:
- YAML is intended to be generic (no train/test split paths).
- Use CLI to set split paths:
    --query_data_path / --cand_pool_path
- Output filenames:
    --query_out / --pool_out (explicit) OR auto-derived from cfg paths.

DDP:
- Supports torchrun (env:// NCCL).
- Uses variable-length all_gather to collect embeddings & metadata across ranks.

Optional post-processing:
- --union_pool: After pool extraction, build/refresh a union cand pool dict in the same pool output directory:
    <pool_save_dir>/mbeir_union_cand_pool_dict.pt
  It merges files matching: mbeir_*_cand_pool_dict.pt (excluding the union file itself),
  and deduplicates by hid (keeps first occurrence).
"""

import os
import random
import argparse
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, DistributedSampler

from omegaconf import OmegaConf
from tqdm import tqdm
import warnings
import utils
from datasets import MBEIRTrainQueryDataset, MBEIRTrainPoolDataset


# ---------------------------
# Collator (Qwen2-VL processor)
# ---------------------------
def try_import_process_vision_info():
    try:
        from collators.qwen2_vision_process import process_vision_info
        return process_vision_info
    except Exception:
        raise ImportError("Cannot import process_vision_info from collators.qwen2_vision_process or qwen_vl_utils")


class LamraMbeirCollator:
    def __init__(self, processor, max_length: int = 1024):
        self.processor = processor
        self.max_length = max_length
        self.process_vision_info = try_import_process_vision_info()

    def __call__(self, batch: List[dict]) -> Dict[str, torch.Tensor]:
        messages = [x["messages"] for x in batch]
        hids = torch.tensor([x["hid"] for x in batch], dtype=torch.long)
        img_mask = torch.tensor([x["img_mask"] for x in batch], dtype=torch.long)
        txt_mask = torch.tensor([x["txt_mask"] for x in batch], dtype=torch.long)
        modality_id = torch.tensor([x["modality_id"] for x in batch], dtype=torch.long)

        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
        image_inputs, video_inputs = self.process_vision_info(messages)

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs.get("input_ids", None),
            "attention_mask": inputs.get("attention_mask", None),
            "pixel_values": inputs.get("pixel_values", None),
            "image_grid_thw": inputs.get("image_grid_thw", None),
            "hids": hids,
            "img_mask": img_mask,
            "txt_mask": txt_mask,
            "modality_id": modality_id,
        }


# ---------------------------
# Main extraction loop
# ---------------------------
@torch.inference_mode()
def run_extract(
    name: str,
    loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
):
    id_to_index: Dict[int, int] = {}
    embs: List[torch.Tensor] = []
    img_masks: List[torch.Tensor] = []
    txt_masks: List[torch.Tensor] = []
    modality_ids: List[torch.Tensor] = []

    cur_index = 0
    pbar = tqdm(loader, disable=(utils.get_rank() != 0), desc=f"Extract {name}")

    for batch in pbar:
        # Move to GPU (pixel_values cast to model.dtype like LamRA eval)
        for k in ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]:
            if k in batch and isinstance(batch[k], torch.Tensor) and batch[k] is not None:
                if k == "pixel_values":
                    batch[k] = batch[k].to(device, non_blocking=True).to(model.dtype)
                else:
                    batch[k] = batch[k].to(device, non_blocking=True)

        hids = batch["hids"].to(device, non_blocking=True)
        img_mask = batch["img_mask"].to(device, non_blocking=True)
        txt_mask = batch["txt_mask"].to(device, non_blocking=True)
        modality_id = batch["modality_id"].to(device, non_blocking=True)

        infer_kwargs = {
            "input_ids": batch.get("input_ids", None),
            "attention_mask": batch.get("attention_mask", None),
            "pixel_values": batch.get("pixel_values", None),
            "image_grid_thw": batch.get("image_grid_thw", None),
        }
        infer_kwargs = {k: v for k, v in infer_kwargs.items() if v is not None}

        model_out = model.inference(**infer_kwargs, normalize=False)
        emb = utils.extract_embedding(model_out)  # [B, D] (no normalization)

        emb = utils.gather_tensor_varlen(emb)
        hids_g = utils.gather_tensor_varlen(hids)
        img_mask_g = utils.gather_tensor_varlen(img_mask)
        txt_mask_g = utils.gather_tensor_varlen(txt_mask)
        modality_id_g = utils.gather_tensor_varlen(modality_id)

        if utils.get_rank() == 0:
            for i in range(hids_g.size(0)):
                hid = int(hids_g[i].item())
                if hid in id_to_index:
                    continue
                id_to_index[hid] = cur_index
                embs.append(emb[i].detach().cpu().float())
                img_masks.append(img_mask_g[i].detach().cpu())
                txt_masks.append(txt_mask_g[i].detach().cpu())
                modality_ids.append(modality_id_g[i].detach().cpu())
                cur_index += 1

    if utils.get_rank() == 0:
        out = {
            "emb": torch.stack(embs, dim=0),
            "img_mask": torch.stack(img_masks, dim=0),
            "txt_mask": torch.stack(txt_masks, dim=0),
            "modality_id": torch.stack(modality_ids, dim=0),
            "id_to_index": id_to_index,
            "modality_vocab": {0: "image", 1: "text", 2: "image,text"},
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(out, save_path)
        print(f"[OK] saved {name} embedding dict to: {save_path}")
        print(f"[OK] {name}: N={out['emb'].size(0)} D={out['emb'].size(1)}")


# ---------------------------
# Entry
# ---------------------------
def main():
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--mbeir_data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    # Model
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--original_model_id", type=str, default=None)  # used for processor/tokenizer
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])

    # Control switches
    parser.add_argument("--do_query", action="store_true")
    parser.add_argument("--do_pool", action="store_true")
    parser.add_argument("--max_length", type=int, default=1024)

    # Output controls
    parser.add_argument(
        "--query_save_dir",
        type=str,
        default=None,
        help="Optional. If set, save query dict under this dir instead of --save_dir.",
    )
    parser.add_argument(
        "--pool_save_dir",
        type=str,
        default=None,
        help="Optional. If set, save pool dict under this dir instead of --save_dir.",
    )
    parser.add_argument(
        "--query_out",
        type=str,
        default=None,
        help="Optional. Output filename for query dict (.pt). If None, auto-derived from config.",
    )
    parser.add_argument(
        "--pool_out",
        type=str,
        default=None,
        help="Optional. Output filename for pool dict (.pt). If None, auto-derived from config.",
    )

    # Split selection (YAML stays generic)
    parser.add_argument(
        "--query_data_path",
        type=str,
        default=None,
        help="Override cfg.data_config.query_data_path (relative to --mbeir_data_dir).",
    )
    parser.add_argument(
        "--cand_pool_path",
        type=str,
        default=None,
        help="Override cfg.data_config.cand_pool_path (relative to --mbeir_data_dir).",
    )

    # Optional post-processing
    parser.add_argument(
        "--union_pool",
        action="store_true",
        help="If set, build/refresh mbeir_union_cand_pool_dict.pt in pool output directory after pool extraction.",
    )

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config_path)

    # ---- allow CLI override for dataset paths (to keep YAML generic)
    if args.query_data_path is not None:
        cfg.data_config.query_data_path = args.query_data_path
    if args.cand_pool_path is not None:
        cfg.data_config.cand_pool_path = args.cand_pool_path

    # ---- distributed init
    _, local_rank = utils.init_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # ---- seed
    seed = int(getattr(cfg, "seed", 2023)) + utils.get_rank()
    random.seed(seed)
    torch.manual_seed(seed)

    # ---- load model + processor/tokenizer
    from transformers import AutoProcessor
    from models.lamra.qwen2_vl import Qwen2VLRetForConditionalGeneration

    torch_dtype = torch.float32
    if args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        torch_dtype = torch.float16

    original_id = args.original_model_id or args.model_name_or_path
    processor = AutoProcessor.from_pretrained(original_id, trust_remote_code=True)
    tokenizer = processor.tokenizer
    tokenizer.model_max_length = args.max_length

    model = Qwen2VLRetForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=None,
        trust_remote_code=True,
    )

    # ---- CRITICAL: inject <emb> token id into model.config (LamRA eval style)
    emb_id = utils.add_embed_token(tokenizer, model, emb_token="<emb>")
    if utils.get_rank() == 0:
        print(f"[OK] <emb> token id = {emb_id}")

    model.eval()
    model.to(device)

    # ---- build datasets
    data_cfg = cfg.data_config
    dl_cfg = cfg.dataloader_config

    enable_query_instruct = bool(getattr(data_cfg, "enable_query_instruct", False))
    shuffle_cand = bool(getattr(data_cfg, "shuffle_cand", False))

    query_ds = MBEIRTrainQueryDataset(
        mbeir_data_dir=args.mbeir_data_dir,
        query_jsonl_relpath=data_cfg.query_data_path,
        cand_pool_jsonl_relpath=data_cfg.cand_pool_path,
        instruct_tsv_relpath=data_cfg.query_instruct_path,
        enable_query_instruct=enable_query_instruct,
        shuffle_cand=shuffle_cand,
    )
    pool_ds = MBEIRTrainPoolDataset(
        mbeir_data_dir=args.mbeir_data_dir,
        cand_pool_jsonl_relpath=data_cfg.cand_pool_path,
    )

    collator = LamraMbeirCollator(processor=processor, max_length=args.max_length)

    def make_loader(ds, bs: int) -> DataLoader:
        if utils.is_dist():
            sampler = DistributedSampler(
                ds,
                num_replicas=utils.get_world_size(),
                rank=utils.get_rank(),
                shuffle=False,
                drop_last=False,
            )
        else:
            sampler = None
        return DataLoader(
            ds,
            batch_size=bs,
            num_workers=int(getattr(dl_cfg, "num_workers", 8)),
            pin_memory=True,
            shuffle=False,
            sampler=sampler,
            collate_fn=collator,
            drop_last=False,
        )

    os.makedirs(args.save_dir, exist_ok=True)

    if not args.do_query and not args.do_pool:
        args.do_query, args.do_pool = True, True

    bs = int(getattr(dl_cfg, "train_batch_size", 64))

    # resolve save dirs (fallback to --save_dir)
    query_save_dir = args.query_save_dir or args.save_dir
    pool_save_dir = args.pool_save_dir or args.save_dir

    if args.do_query:
        q_loader = make_loader(query_ds, bs=bs)
        q_fname = args.query_out or utils.derive_out_name("query", cfg)
        q_save = os.path.join(query_save_dir, q_fname)
        run_extract("query", q_loader, model, device, q_save)

    if args.do_pool:
        p_loader = make_loader(pool_ds, bs=bs)
        p_fname = args.pool_out or utils.derive_out_name("pool", cfg)
        p_save = os.path.join(pool_save_dir, p_fname)
        run_extract("pool", p_loader, model, device, p_save)

        # --- optional: build union pool dict in the same directory
        if args.union_pool and utils.get_rank() == 0:
            union_out = os.path.join(pool_save_dir, "mbeir_union_cand_pool_dict.pt")
            utils.union_cand_pool(
                input_dir=pool_save_dir,
                output_pt=union_out,
                pattern="mbeir_*_cand_pool_dict.pt",
            )

    utils.cleanup_distributed()


if __name__ == "__main__":
    main()