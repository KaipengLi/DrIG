"""
Training Code for CLIP-SF

This script generates CLIP-SF embeddings for M-BEIR queries/candidate pools and exports:
  - mbeir_{task}_{split}_embeddings.npy
  - mbeir_{task}_{split}_ids.npy

Where:
  split in {train, val, test} for query datasets
  split = cand_pool for candidate pools
  union cand pool:
    - mbeir_union_cand_pool_embeddings.npy
    - mbeir_union_cand_pool_ids.npy

Notes:
- We intentionally DO NOT save the old "*_IT_dict.pt" anymore.
- IDs are aligned with embedding rows: ids[i] corresponds to embeddings[i].
"""

# Standard library
import argparse
import logging
import os
import random
import clip  # noqa: F401  # kept for compatibility (some projects rely on this import)

# Third-party
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast
from omegaconf import OmegaConf
from tqdm import tqdm
import gc

import common.dist_utils as dist_utils
from models.uniir_clip import utils
from models.uniir_clip.clip_nofusion.clip_nf import CLIPNoFusion

from data.mbeir_dataset import (
    MBEIRMainDataset,
    MBEIRMainCollator,
    MBEIRCandidatePoolDataset,
    MBEIRCandidatePoolCollator,
    Mode,
)

# Set up logger
logger = logging.getLogger()


# --------------------------
# Reproducibility
# --------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------
# Dist helpers
# --------------------------
def _barrier_if_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


# --------------------------
# Export helpers
# --------------------------
def _to_1d_mask(x: torch.Tensor | None) -> torch.Tensor:
    """
    Convert mask tensor to shape [N] float tensor on CPU.
    Accepts shapes like [N], [N,1], [N,*].
    """
    if x is None:
        return None
    if x.dim() == 0:
        x = x.view(1)
    if x.dim() > 1:
        x = x.view(x.shape[0], -1)
        x = x[:, 0]
    return x.float().cpu()


def _fuse_img_txt_embeddings(
    img: torch.Tensor,
    txt: torch.Tensor,
    img_mask: torch.Tensor | None,
    txt_mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    Build ONE embedding per sample for export as *.npy.

    Rule:
      - If only img exists -> use img
      - If only txt exists -> use txt
      - If both exist -> average(img, txt)

    Masks are treated as boolean (mask > 0.5).
    Output: float32 tensor [N, D] on CPU.
    """
    assert img.shape == txt.shape, f"img/txt shape mismatch: {img.shape} vs {txt.shape}"
    N, _ = img.shape

    im = _to_1d_mask(img_mask)
    tm = _to_1d_mask(txt_mask)

    if im is None:
        im = torch.zeros((N,), dtype=torch.float32)
    if tm is None:
        tm = torch.zeros((N,), dtype=torch.float32)

    im = (im > 0.5).float().view(N, 1)
    tm = (tm > 0.5).float().view(N, 1)

    denom = im + tm
    denom = torch.where(denom > 0, denom, torch.ones_like(denom))

    fused = (img.float().cpu() * im + txt.float().cpu() * tm) / denom
    return fused.to(dtype=torch.float32)


def _save_npy_pair(save_dir: str, base_name: str, emb: torch.Tensor, ids: np.ndarray) -> None:
    """
    Save:
      - {base_name}_embeddings.npy
      - {base_name}_ids.npy
    """
    os.makedirs(save_dir, exist_ok=True)

    emb_path = os.path.join(save_dir, f"{base_name}_embeddings.npy")
    ids_path = os.path.join(save_dir, f"{base_name}_ids.npy")

    emb_np = emb.detach().cpu().numpy().astype(np.float32, copy=False)
    ids_np = ids.astype(np.int64, copy=False)

    np.save(emb_path, emb_np)
    np.save(ids_path, ids_np)

    print(f"Embedder Log: Saved {emb_path} shape={emb_np.shape}")
    print(f"Embedder Log: Saved {ids_path} shape={ids_np.shape}")


# --------------------------
# Main embedding generator
# --------------------------
def generate_embeds_for_config(model, img_preprocess_fn, tokenizer, config) -> None:
    """Generate embeddings for queries and candidates, and export to .npy (embeddings + ids)."""
    mbeir_data_dir = config.mbeir_data_dir
    embed_config = config.embed_config

    # NOTE: if config.model.emb_save_path is absolute, join keeps it absolute.
    save_embed_path = os.path.join(config.genir_dir, config.model.emb_save_path)

    # dataset config
    data_config = config.data_config
    query_instruct_path = data_config.query_instruct_path
    cand_pool_dir = data_config.cand_pool_dir_name
    image_size = tuple(map(int, data_config.image_size.split(",")))

    # Which datasets / splits to embed
    dataset_types = ["train", "val", "test"]
    splits = []

    # For union pool build
    all_cand_pool_name_list = [
        "visualnews_task0",
        "mscoco_task0_test",
        "fashion200k_task0",
        "webqa_task1",
        "edis_task2",
        "webqa_task2",
        "visualnews_task3",
        "mscoco_task3_test",
        "fashion200k_task3",
        "nights_task4",
        "oven_task6",
        "infoseek_task6",
        "fashioniq_task7",
        "cirr_task7",
        "oven_task8",
        "infoseek_task8",
    ]

    for split_name in dataset_types:
        split_dir_name = getattr(data_config, f"{split_name}_dir_name")
        embed_dataset_config = getattr(embed_config, f"{split_name}_datasets_config", None)
        if embed_dataset_config and embed_dataset_config.enable_embed:
            dataset_name_list = getattr(embed_dataset_config, "datasets_name", None)
            cand_pool_name_list = getattr(embed_dataset_config, "correspond_cand_pools_name", None)
            assert len(dataset_name_list) == len(cand_pool_name_list), "Mismatch between datasets and candidate pools."
            splits.append((split_name, split_dir_name, dataset_name_list, cand_pool_name_list))

    embed_cand_pool_config = embed_config.cand_pools_config
    if embed_cand_pool_config and embed_cand_pool_config.enable_embed:
        split_name = "cand_pool"
        split_dir_name = data_config.cand_pool_dir_name
        cand_pool_name_list = embed_cand_pool_config.cand_pools_name_to_embed
        splits.append((split_name, split_dir_name, [None] * len(cand_pool_name_list), cand_pool_name_list))

    if dist_utils.is_main_process():
        print("-" * 30)
        for split_name, split_dir, dataset_name_list, cand_pool_name_list in splits:
            if split_name == "cand_pool":
                print(f"Split: {split_name}, Split dir: {split_dir}, Candidate pools to embed: {cand_pool_name_list}")
            else:
                print(f"Split: {split_name}, Split dir: {split_dir}, Datasets to embed: {dataset_name_list}")
            print("-" * 30)

    def _pick_id_list_from_batch(batch: dict):
        candidates = ["did_list", "qid_list", "id_list", "ids", "hashed_qid", "hashed_did"]
        for k in candidates:
            if k in batch and batch[k] is not None:
                return batch[k]
        return None

    # --------------------------
    # Per split/dataset export
    # --------------------------
    for split_name, split_dir, dataset_name_list, cand_pool_name_list in splits:
        for dataset_name, cand_pool_name in zip(dataset_name_list, cand_pool_name_list):
            # storage: keep only final arrays + ids (aligned)
            all_img_rows = []
            all_txt_rows = []
            all_imask_rows = []
            all_tmask_rows = []
            all_ids_rows = []

            # ------------------- build dataset/collator -------------------
            if split_name == "cand_pool":
                task = cand_pool_name.lower()
                cand_pool_file_name = f"mbeir_{task}_{split_name}.jsonl"  # mbeir_xxx_cand_pool.jsonl
                cand_pool_data_path = os.path.join(cand_pool_dir, cand_pool_file_name)

                print_config = False
                if dist_utils.is_main_process():
                    print(f"\nEmbedder Log: Generating embeddings for {cand_pool_data_path}...")
                    print_config = True

                dataset = MBEIRCandidatePoolDataset(
                    mbeir_data_dir=mbeir_data_dir,
                    cand_pool_data_path=cand_pool_data_path,
                    img_preprocess_fn=img_preprocess_fn,
                    print_config=print_config,
                )
                collator = MBEIRCandidatePoolCollator(tokenizer=tokenizer, image_size=image_size)

                # eval expects:
                #   mbeir_{task}_cand_pool_embeddings.npy
                #   mbeir_{task}_cand_pool_ids.npy
                base_name = f"mbeir_{task}_cand_pool"
            else:
                task = dataset_name.lower()
                query_data_name = f"mbeir_{task}_{split_name}.jsonl"
                query_data_path = os.path.join(split_dir, query_data_name)

                cand_pool_name = cand_pool_name.lower()
                cand_pool_file_name = f"mbeir_{cand_pool_name}_cand_pool.jsonl"
                cand_pool_data_path = os.path.join(cand_pool_dir, cand_pool_file_name)

                print_config = False
                if dist_utils.is_main_process():
                    print(f"\nEmbedder Log: Generating embeddings for {query_data_path} with {cand_pool_data_path}...")
                    print_config = True

                mode = Mode.EVAL
                dataset = MBEIRMainDataset(
                    mbeir_data_dir=mbeir_data_dir,
                    query_data_path=query_data_path,
                    cand_pool_path=cand_pool_data_path,
                    query_instruct_path=query_instruct_path,
                    img_preprocess_fn=img_preprocess_fn,
                    mode=mode,
                    enable_query_instruct=data_config.enable_query_instruct,
                    shuffle_cand=data_config.shuffle_cand,
                    print_config=print_config,
                )
                collator = MBEIRMainCollator(tokenizer=tokenizer, image_size=image_size, mode=mode)

                # eval expects:
                #   mbeir_{task}_{split}_embeddings.npy  (e.g., ..._test_embeddings.npy)
                #   mbeir_{task}_{split}_ids.npy
                base_name = f"mbeir_{task}_{split_name}"

            # ------------------- dataloader -------------------
            sampler = DistributedSampler(
                dataset,
                num_replicas=dist_utils.get_world_size(),
                rank=dist_utils.get_rank(),
                shuffle=False,
            )
            data_loader = DataLoader(
                dataset,
                batch_size=config.dataloader_config.batch_size,
                num_workers=config.dataloader_config.num_workers,
                pin_memory=False,
                sampler=sampler,
                shuffle=False,
                collate_fn=collator,
                drop_last=False,
            )

            _barrier_if_dist()

            if dist_utils.is_main_process():
                print(f"Embedder Log: Generating embeddings -> {base_name}_embeddings.npy / {base_name}_ids.npy")
                print(f"Inference with half precision: {config.embed_config.use_fp16}")

            # ------------------- main loop -------------------
            for batch in tqdm(data_loader) if dist_utils.is_main_process() else data_loader:
                # move tensors to GPU
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(config.dist_config.gpu_id, non_blocking=True)

                id_list = _pick_id_list_from_batch(batch)
                if id_list is None:
                    raise KeyError(f"[{split_name}] Cannot find id list in batch. keys={list(batch.keys())}")

                txt_batched = batch["txt_batched"]
                image_batched = batch["image_batched"]
                txt_mask_batched = batch["txt_mask_batched"]
                image_mask_batched = batch["image_mask_batched"]

                with autocast(enabled=config.embed_config.use_fp16):
                    if hasattr(model, "module"):
                        img_emb, txt_emb = model.module.encode_multimodal_input(image_batched, txt_batched)
                    else:
                        img_emb, txt_emb = model.encode_multimodal_input(image_batched, txt_batched)

                # Gather across ranks (so rank0 gets global aligned arrays)
                _barrier_if_dist()
                if utils.get_world_size() > 1:
                    if not torch.is_tensor(id_list):
                        id_tensor = torch.LongTensor(id_list).to(img_emb.device)
                    else:
                        id_tensor = id_list.to(img_emb.device)

                    id_tensor = torch.cat(utils.GatherLayer.apply(id_tensor), dim=0)
                    txt_emb = torch.cat(utils.GatherLayer.apply(txt_emb), dim=0)
                    img_emb = torch.cat(utils.GatherLayer.apply(img_emb), dim=0)
                    txt_mask_batched = torch.cat(utils.GatherLayer.apply(txt_mask_batched), dim=0)
                    image_mask_batched = torch.cat(utils.GatherLayer.apply(image_mask_batched), dim=0)
                else:
                    if torch.is_tensor(id_list):
                        id_tensor = id_list
                    else:
                        id_tensor = torch.LongTensor(id_list).to(img_emb.device)

                _barrier_if_dist()

                if utils.is_main_process():
                    # IMPORTANT: we keep duplicates out, preserving first occurrence order
                    # Use a set to filter; store in python list aligned with embeddings rows.
                    # This preserves stable mapping for ids.npy <-> embeddings.npy.
                    # Since we no longer save id_to_index, we enforce uniqueness here.
                    if not hasattr(generate_embeds_for_config, "_seen_ids"):
                        generate_embeds_for_config._seen_ids = set()  # type: ignore[attr-defined]
                    seen = generate_embeds_for_config._seen_ids  # type: ignore[attr-defined]

                    for j, _id_t in enumerate(id_tensor):
                        _id = int(_id_t.item())
                        if _id in seen:
                            continue
                        seen.add(_id)
                        all_ids_rows.append(_id)
                        all_img_rows.append(img_emb[j].detach().cpu())
                        all_txt_rows.append(txt_emb[j].detach().cpu())
                        all_imask_rows.append(image_mask_batched[j].detach().cpu())
                        all_tmask_rows.append(txt_mask_batched[j].detach().cpu())

            _barrier_if_dist()

            # ------------------- save npy (rank0) -------------------
            if utils.is_main_process():
                if len(all_ids_rows) == 0:
                    raise RuntimeError(f"No samples collected for {base_name}")

                img_all = torch.stack(all_img_rows, dim=0)
                txt_all = torch.stack(all_txt_rows, dim=0)
                imask_all = torch.stack(all_imask_rows, dim=0)
                tmask_all = torch.stack(all_tmask_rows, dim=0)
                ids_all = np.asarray(all_ids_rows, dtype=np.int64)

                fused = _fuse_img_txt_embeddings(img_all, txt_all, imask_all, tmask_all)
                _save_npy_pair(save_embed_path, base_name, fused, ids_all)

            _barrier_if_dist()

            # cleanup
            del dataset, collator, data_loader
            gc.collect()
            torch.cuda.empty_cache()

            # reset global id set for next dataset export
            if utils.is_main_process() and hasattr(generate_embeds_for_config, "_seen_ids"):
                delattr(generate_embeds_for_config, "_seen_ids")  # type: ignore[attr-defined]

        # --------------------------
        # Union cand pool (rank0 only)
        # --------------------------
        if split_name == "cand_pool" and embed_cand_pool_config.embed_union_pool:
            if (not dist.is_initialized()) or dist.get_rank() == 0:
                print("\nEmbedder Log: Generating union cand pool npy...")

                union_emb_list = []
                union_ids_list = []

                for cand_pool_name in all_cand_pool_name_list:
                    task = cand_pool_name.lower()
                    emb_path = os.path.join(save_embed_path, f"mbeir_{task}_cand_pool_embeddings.npy")
                    ids_path = os.path.join(save_embed_path, f"mbeir_{task}_cand_pool_ids.npy")

                    if not os.path.exists(emb_path):
                        raise FileNotFoundError(f"Missing cand pool embeddings for union: {emb_path}")
                    if not os.path.exists(ids_path):
                        raise FileNotFoundError(f"Missing cand pool ids for union: {ids_path}")

                    e = np.load(emb_path)
                    i = np.load(ids_path).astype(np.int64, copy=False)

                    union_emb_list.append(e)
                    union_ids_list.append(i)

                    print(f"Embedder Log: Union add {task}: emb={e.shape} ids={i.shape}")

                union_emb = np.concatenate(union_emb_list, axis=0).astype(np.float32, copy=False)
                union_ids = np.concatenate(union_ids_list, axis=0).astype(np.int64, copy=False)

                np.save(os.path.join(save_embed_path, "mbeir_union_cand_pool_embeddings.npy"), union_emb)
                np.save(os.path.join(save_embed_path, "mbeir_union_cand_pool_ids.npy"), union_ids)

                print(
                    f"Embedder Log: Saved union -> "
                    f"{os.path.join(save_embed_path, 'mbeir_union_cand_pool_embeddings.npy')} shape={union_emb.shape}"
                )
                print(
                    f"Embedder Log: Saved union -> "
                    f"{os.path.join(save_embed_path, 'mbeir_union_cand_pool_ids.npy')} shape={union_ids.shape}"
                )

            _barrier_if_dist()


def main(config) -> None:
    is_distributed_mode = config.dist_config.distributed_mode

    seed = config.seed + utils.get_rank()
    set_seed(seed)

    cudnn.benchmark = True

    print("Creating CLIP model...")
    model_config = config.model
    pretrained_clip_model_dir = os.path.join(config.genir_dir, model_config.pretrained_clip_model_dir)
    logger.info(f"Downloading CLIP model to {pretrained_clip_model_dir}...")

    model = CLIPNoFusion(
        model_name=model_config.clip_vision_model_name,
        download_root=pretrained_clip_model_dir,
        config=config,
    )
    model.float()

    ckpt_config = model_config.ckpt_config
    if ckpt_config.using_pretrained:
        checkpoint_path = os.path.join(config.ckpt_root, ckpt_config.ckpt_dir, ckpt_config.ckpt_name)
        assert os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist."
        print(f"loading CLIPScoreFusion checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint["model"])

    model.eval()
    model = model.to(config.dist_config.gpu_id)

    model_without_ddp = model
    if is_distributed_mode:
        model = DDP(model, device_ids=[config.dist_config.gpu_id])
        model_without_ddp = model.module

    img_preprocess_fn = model_without_ddp.get_img_preprocess_fn()
    tokenizer = model_without_ddp.get_tokenizer()

    generate_embeds_for_config(model, img_preprocess_fn, tokenizer, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="config.yaml", help="Path to the config file.")
    parser.add_argument(
        "--genir_dir",
        type=str,
        default="/data/GENIUS",
        help="Path to GENIUS directory to save checkpoints, embeddings, etc.",
    )
    parser.add_argument(
        "--mbeir_data_dir",
        type=str,
        default="/data/GENIUS/mbeir_data",
        help="Path to mbeir dataset directory",
    )
    parser.add_argument(
        "--ckpt_root",
        type=str,
        default="",
        help="Root dir that contains checkpoints/ (e.g. /data/likaipeng/dig). If empty, fallback to genir_dir.",
    )
    args = parser.parse_args()
    print(f"Loading config from {args.config_path}")
    config = OmegaConf.load(args.config_path)

    config.genir_dir = args.genir_dir
    config.mbeir_data_dir = args.mbeir_data_dir
    config.ckpt_root = args.ckpt_root if args.ckpt_root else config.genir_dir

    # init distributed
    args.dist_url = config.dist_config.dist_url  # historical artifact
    utils.init_distributed_mode(args)
    config.dist_config.gpu_id = args.gpu
    config.dist_config.distributed_mode = args.distributed

    main(config)

    if config.dist_config.distributed_mode:
        torch.distributed.destroy_process_group()