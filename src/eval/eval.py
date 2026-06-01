#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import os
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

warnings.filterwarnings("ignore", message=r".*torch\.load.*weights_only=False.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*torch\.cuda\.amp\.GradScaler.*deprecated.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=r".*torch\.cuda\.amp\.autocast.*deprecated.*", category=FutureWarning)

from eval.utils import (
    barrier,
    compute_recall_means_by_task,
    compute_s_q_full,
    create_hash_map,
    destroy_dist,
    emb_retrieve_topk_chunked,
    get_ids_from_id_to_index,
    get_rank,
    get_world_size,
    hash_qid,
    init_distributed_if_needed,
    is_dist,
    is_main,
    load_any_emb_store,
    load_jsonl,
    load_qrel,
    load_sq,
    load_store_pt,
    print_result_block,
    retrieve_candidates_by_beams,
    write_eval_results_tsv,
)

from eval.beam import (
    LEVEL_PREFIXES,
    Trie,
    VectorizedGuidanceLogitsProcessor,
    build_E_simul_from_rq,
    build_code_tokenizer,
    build_docs_token_ids_from_cand_codes,
    build_token_id_map,
    decode_ids_to_codes,
    get_level_offsets,
    load_rq,
    load_t5_model,
    rq_emb_to_codes,
    topk_docs_by_sq_chunked,
)


class QueryIdDataset(Dataset):
    """
    Dataset of hashed query ids from query jsonl.
    Accepts either:
      - {"hashed_qid": int}
      - {"qid": original_id} -> hashed via hash_qid()
    """

    def __init__(self, query_jsonl: str):
        items = load_jsonl(query_jsonl)
        self.hashed_qids: List[int] = []
        for x in items:
            if "hashed_qid" in x:
                self.hashed_qids.append(int(x["hashed_qid"]))
            elif "qid" in x:
                self.hashed_qids.append(hash_qid(x["qid"]))

    def __len__(self) -> int:
        return len(self.hashed_qids)

    def __getitem__(self, idx: int) -> int:
        return self.hashed_qids[idx]


def collate_qids(batch: List[int]) -> torch.Tensor:
    return torch.tensor(batch, dtype=torch.long)


def _pick_existing(candidates: List[str], what: str) -> str:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    msg = f"{what} not found. Tried:\n" + "\n".join([f"  - {p}" for p in candidates])
    raise FileNotFoundError(msg)


def emb_retrieve_topk_chunked_tensor(
    *,
    q_emb: torch.Tensor,          # [B, D] on GPU
    cand_emb_cpu: torch.Tensor,   # [N, D] on CPU
    topk: int,
    normalize: bool,
    chunk_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Streaming top-k retrieval for CPU cand embeddings (torch.Tensor).
    Candidates are moved to GPU chunk-by-chunk, maintaining global topk.
    """
    assert cand_emb_cpu.device.type == "cpu", "cand_emb_cpu must be on CPU"
    assert q_emb.device.type == "cuda", "q_emb should be on GPU for speed"

    B, _D = q_emb.shape
    N = int(cand_emb_cpu.shape[0])
    k = int(min(topk, N))

    qx = q_emb.float()
    if normalize:
        qx = F.normalize(qx, dim=-1)

    top_scores = torch.full((B, k), -1e9, device=qx.device, dtype=torch.float32)
    top_indices = torch.full((B, k), -1, device=qx.device, dtype=torch.long)

    for st in range(0, N, int(chunk_size)):
        ed = min(N, st + int(chunk_size))
        c_chunk = cand_emb_cpu[st:ed].to(device=qx.device, non_blocking=True).float()
        if normalize:
            c_chunk = F.normalize(c_chunk, dim=-1)

        sims = qx @ c_chunk.t()
        local_k = min(k, sims.shape[1])
        local_scores, local_idx = torch.topk(sims, k=local_k, dim=1)
        local_idx = local_idx + st

        merged_scores = torch.cat([top_scores, local_scores], dim=1)
        merged_indices = torch.cat([top_indices, local_idx], dim=1)
        new_scores, new_pos = torch.topk(merged_scores, k=k, dim=1)
        new_indices = torch.gather(merged_indices, 1, new_pos)

        top_scores, top_indices = new_scores, new_indices

        del (
            c_chunk,
            sims,
            local_scores,
            local_idx,
            merged_scores,
            merged_indices,
            new_scores,
            new_pos,
            new_indices,
        )

    return top_scores, top_indices


def _parse_metrics(metrics: str) -> Tuple[List[str], List[str], int]:
    metric_list = [m.strip() for m in metrics.split(",") if m.strip()]
    recall_metrics = [m for m in metric_list if "recall@" in m.lower()]
    if len(recall_metrics) == 0:
        raise ValueError(f"No recall metrics in --metrics: {metrics}")
    k_eval = max(int(m.split("@")[1]) for m in recall_metrics)
    return metric_list, recall_metrics, k_eval


def _parse_guide_topk_list(s: str) -> List[int]:
    out: List[int] = []
    for x in (s or "").split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out if len(out) > 0 else [0, 1000]


def _parse_rerank_k_list(x: Any) -> List[int]:
    """
    Support:
      - 50
      - "50"
      - "3,5,10"
      - ["3", "5", "10"]
      - [3, 5, 10]
    """
    if x is None:
        return [50]

    vals: List[int] = []

    if isinstance(x, int):
        vals = [x]
    elif isinstance(x, str):
        for t in x.split(","):
            t = t.strip()
            if t:
                vals.append(int(t))
    elif isinstance(x, (list, tuple)):
        for item in x:
            if isinstance(item, int):
                vals.append(item)
            else:
                for t in str(item).split(","):
                    t = t.strip()
                    if t:
                        vals.append(int(t))
    else:
        raise TypeError(f"Unsupported rerank_k type: {type(x)}")

    vals = sorted(set(int(v) for v in vals if int(v) > 0))
    if len(vals) == 0:
        vals = [50]
    return vals


def _is_union_cand_store(path: str) -> bool:
    if not path:
        return False
    base = os.path.basename(path).lower()
    return ("union" in base) or ("mbeir_union" in base)


def _parse_rerank_sources(s: str) -> Tuple[bool, bool]:
    """
    Return (do_lamra, do_clip_sf).

    Accepted (case-insensitive):
      - lamra
      - clip-sf  (also: clip_sf / clipsf)
      - lamra,clip-sf
    """
    s = (s or "").lower().replace(" ", "")
    parts = [p for p in s.split(",") if p]
    do_lamra = "lamra" in parts
    do_clip = ("clip-sf" in parts) or ("clip_sf" in parts) or ("clipsf" in parts)
    return do_lamra, do_clip


def _maybe_autofill_task_paths(args: argparse.Namespace) -> None:
    """
    Keep original behavior: if --task is provided, auto-fill:
      - query_store/cand_store (LamRA dict.pt) unless emb_only
      - eval_query_jsonl/qrels if absent
    """
    if not getattr(args, "task", ""):
        return

    task = args.task

    query_candidates = [
        os.path.join(args.embed_root, task, f"mbeir_{task}_test_dict.pt"),
        os.path.join(args.embed_root, f"mbeir_{task}_test_dict.pt"),
    ]

    embed_root_norm = os.path.normpath(args.embed_root)
    lamra_root = os.path.dirname(embed_root_norm)
    cand_root = os.path.join(lamra_root, "cand")
    cand_candidates = [
        os.path.join(args.embed_root, task, f"mbeir_{task}_cand_pool_dict.pt"),
        os.path.join(args.embed_root, f"mbeir_{task}_cand_pool_dict.pt"),
        os.path.join(cand_root, task, f"mbeir_{task}_cand_pool_dict.pt"),
        os.path.join(cand_root, f"mbeir_{task}_cand_pool_dict.pt"),
    ]

    if args.query_store is None and (not args.emb_only):
        args.query_store = _pick_existing(query_candidates, what="query_store")
    if args.cand_store is None and (not args.emb_only):
        args.cand_store = _pick_existing(cand_candidates, what="cand_store")

    if args.eval_query_jsonl is None:
        args.eval_query_jsonl = os.path.join(args.mbeir_data_dir, "query", "test", f"mbeir_{task}_test.jsonl")
    if args.qrels is None:
        args.qrels = os.path.join(args.mbeir_data_dir, "qrels", "test", f"mbeir_{task}_test_qrels.txt")


def _maybe_fill_external_rerank_paths(args: argparse.Namespace) -> None:
    """
    If external_rerank_dir is set and rerank_sources contains external,
    auto-resolve per-task external npy paths.
    """
    if not getattr(args, "task", ""):
        return

    _do_lamra, do_ext = _parse_rerank_sources(getattr(args, "rerank_sources", "lamra"))
    if (not getattr(args, "rerank", False)) or (not do_ext):
        return

    if args.rerank_query_store not in (None, "") and args.rerank_cand_store not in (None, ""):
        return
    if args.external_rerank_dir in (None, ""):
        return

    d = args.external_rerank_dir
    task = args.task

    q_emb = os.path.join(d, f"mbeir_{task}_test_embeddings.npy")
    q_ids = os.path.join(d, f"mbeir_{task}_test_ids.npy")

    if args.external_use_union_cand:
        c_emb = os.path.join(d, "mbeir_union_cand_pool_embeddings.npy")
        c_ids = os.path.join(d, "mbeir_union_cand_pool_ids.npy")
    else:
        c_emb = os.path.join(d, f"mbeir_{task}_cand_pool_embeddings.npy")
        c_ids = os.path.join(d, f"mbeir_{task}_cand_pool_ids.npy")

    args.rerank_query_store = q_emb
    args.rerank_query_ids = q_ids
    args.rerank_cand_store = c_emb
    args.rerank_cand_ids = c_ids


def _print_rerank_stores(
    *,
    args: argparse.Namespace,
    base_query_store: Optional[str] = None,
    base_cand_store: Optional[str] = None,
    ext_enabled: bool = False,
    ext_q_type: Optional[str] = None,
    ext_c_type: Optional[str] = None,
) -> None:
    if not is_main():
        return
    if not args.rerank:
        return

    do_lamra, do_ext = _parse_rerank_sources(args.rerank_sources)
    rk_list = _parse_rerank_k_list(args.rerank_k)

    print(f"[rerank] enabled (rerank_k_list={rk_list}) sources={args.rerank_sources}")

    if do_lamra:
        print("  lamra_store: enabled (dict.pt)")
        if base_query_store:
            print(f"    query_store: {base_query_store} ")
        if base_cand_store:
            print(f"    cand_store : {base_cand_store} ")
    else:
        print("  lamra_store: disabled")

    if do_ext:
        print(f"  external_store: {'enabled' if ext_enabled else 'requested_but_missing'}")
        print(f"    query_store: {args.rerank_query_store}")
        if args.rerank_query_ids not in (None, ""):
            print(f"    query_ids  : {args.rerank_query_ids}")
        print(f"    cand_store : {args.rerank_cand_store}")
        if args.rerank_cand_ids not in (None, ""):
            print(f"    cand_ids   : {args.rerank_cand_ids}")
    else:
        print("  external_store: disabled")


def _get_query_emb_from_store_by_qids(
    *,
    qids: np.ndarray,
    store_type: str,
    store,
    id2idx: Dict[int, int],
    device: torch.device,
) -> torch.Tensor:
    idx = [id2idx[int(q)] for q in qids.tolist()]
    if store_type == "npy":
        arr = np.asarray(store[idx])
        return torch.from_numpy(arr).to(device=device, non_blocking=True)
    else:
        idx_t = torch.tensor(idx, dtype=torch.long, device="cpu")
        return store[idx_t].to(device=device, non_blocking=True)


def _get_cand_emb_from_store_by_ids(
    *,
    cand_ids: List[int],
    store_type: str,
    store,
    id2idx: Dict[int, int],
    device: torch.device,
) -> torch.Tensor:
    idx = [id2idx[int(cid)] for cid in cand_ids]
    if store_type == "npy":
        arr = np.asarray(store[idx])
        return torch.from_numpy(arr).to(device=device, non_blocking=True)
    else:
        idx_t = torch.tensor(idx, dtype=torch.long, device="cpu")
        return store[idx_t].to(device=device, non_blocking=True)


def _compute_rerank_scores_lamra(
    *,
    retrieved0: List[List[int]],
    batch_qids: np.ndarray,
    max_rerank_k: int,
    query_store: Dict[str, Any],
    cand_store: Dict[str, Any],
    lamra_id2idx: Dict[int, int],
    device: torch.device,
    query_emb_override: Optional[torch.Tensor] = None,
) -> List[np.ndarray]:
    """
    For each query, compute similarity scores only for the first max_rerank_k retrieved candidates.
    """
    if query_emb_override is not None:
        q_emb_all = query_emb_override
    else:
        q_idx = torch.tensor(
            [query_store["id_to_index"][int(q)] for q in batch_qids],
            device=device,
            dtype=torch.long,
        )
        q_emb_all = query_store["emb"][q_idx]

    q_emb_all = F.normalize(q_emb_all.float(), dim=-1)

    out_scores: List[np.ndarray] = []
    for i, cand_list in enumerate(retrieved0):
        prefix = cand_list[:max_rerank_k]
        if len(prefix) == 0:
            out_scores.append(np.zeros((0,), dtype=np.float32))
            continue

        idx_cpu = torch.tensor(
            [lamra_id2idx[int(cid)] for cid in prefix],
            dtype=torch.long,
            device="cpu",
        )
        cand_emb = cand_store["emb"][idx_cpu].to(device=device, non_blocking=True).float()
        cand_emb = F.normalize(cand_emb, dim=-1)

        scores = torch.matmul(cand_emb, q_emb_all[i])  # [K]
        out_scores.append(scores.detach().cpu().numpy().astype(np.float32))

    return out_scores


def _compute_rerank_scores_external(
    *,
    retrieved0: List[List[int]],
    batch_qids: np.ndarray,
    max_rerank_k: int,
    ext_q_type,
    ext_q_store,
    ext_q_id2idx,
    ext_c_type,
    ext_c_store,
    ext_c_id2idx,
    device: torch.device,
) -> List[np.ndarray]:
    q_emb_all = _get_query_emb_from_store_by_qids(
        qids=batch_qids,
        store_type=ext_q_type,
        store=ext_q_store,
        id2idx=ext_q_id2idx,
        device=device,
    )
    q_emb_all = F.normalize(q_emb_all.float(), dim=-1)

    out_scores: List[np.ndarray] = []
    for i, cand_list in enumerate(retrieved0):
        prefix = cand_list[:max_rerank_k]
        if len(prefix) == 0:
            out_scores.append(np.zeros((0,), dtype=np.float32))
            continue

        cand_emb = _get_cand_emb_from_store_by_ids(
            cand_ids=prefix,
            store_type=ext_c_type,
            store=ext_c_store,
            id2idx=ext_c_id2idx,
            device=device,
        )
        cand_emb = F.normalize(cand_emb.float(), dim=-1)

        scores = torch.matmul(cand_emb, q_emb_all[i])  # [K]
        out_scores.append(scores.detach().cpu().numpy().astype(np.float32))

    return out_scores


def _build_reranked_results_for_k(
        *,
        retrieved0: List[List[int]],
        scores_per_query: List[np.ndarray],
        rerank_k: int,
        k_eval: int,
) -> List[List[int]]:
    """
    修改后的版本：只评估重排后的 k 个候选项（截断式）。
    """
    out: List[List[int]] = []
    for cand_list, score_arr in zip(retrieved0, scores_per_query):
        # 1. 仅提取前 rerank_k 个候选项
        prefix = cand_list[:rerank_k]

        # --- 注意：这里删除了 tail = cand_list[rerank_k:] ---

        if len(prefix) > 0:
            # 2. 对这 k 个候选项按分数进行重排
            order = np.argsort(-score_arr[:len(prefix)], kind="mergesort")
            new_prefix = [prefix[j] for j in order.tolist()]
        else:
            new_prefix = []

        # 3. 关键修改：merged 仅包含重排后的前 k 个，不再拼接 tail
        # 这样当 k=3 时，列表总长度就是 3，计算 R@5 和 R@10 的结果自然是一样的
        merged = new_prefix

        # 返回前 k_eval 个（如果 merged 长度小于 k_eval，则全部返回）
        out.append(merged[:k_eval])

    return out


def _rank0_process_batch_multi_rerank(
    *,
    args: argparse.Namespace,
    retrieved0: List[List[int]],
    batch_qids: np.ndarray,
    k_eval: int,
    rerank_k_list: List[int],
    query_store: Dict[str, Any],
    cand_store: Dict[str, Any],
    lamra_id2idx: Dict[int, int],
    device: torch.device,
    use_external_rerank: bool,
    ext_q_type,
    ext_q_store,
    ext_q_id2idx,
    ext_c_type,
    ext_c_store,
    ext_c_id2idx,
    query_emb_override: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    Return:
      {
        "base": List[List[int]],
        "lamra": {k: List[List[int]], ...},
        "ext":   {k: List[List[int]], ...},
      }
    """
    do_lamra, do_ext = _parse_rerank_sources(args.rerank_sources)

    out = {
        "base": [x[:k_eval] for x in retrieved0],
        "lamra": {},
        "ext": {},
    }

    if not args.rerank:
        return out

    max_rerank_k = max(rerank_k_list)

    if do_lamra:
        lamra_scores = _compute_rerank_scores_lamra(
            retrieved0=retrieved0,
            batch_qids=batch_qids,
            max_rerank_k=max_rerank_k,
            query_store=query_store,
            cand_store=cand_store,
            lamra_id2idx=lamra_id2idx,
            device=device,
            query_emb_override=query_emb_override,
        )
        for rk in rerank_k_list:
            out["lamra"][rk] = _build_reranked_results_for_k(
                retrieved0=retrieved0,
                scores_per_query=lamra_scores,
                rerank_k=rk,
                k_eval=k_eval,
            )

    if do_ext:
        if not use_external_rerank:
            raise RuntimeError("rerank_sources includes external but external store is not loaded.")

        ext_scores = _compute_rerank_scores_external(
            retrieved0=retrieved0,
            batch_qids=batch_qids,
            max_rerank_k=max_rerank_k,
            ext_q_type=ext_q_type,
            ext_q_store=ext_q_store,
            ext_q_id2idx=ext_q_id2idx,
            ext_c_type=ext_c_type,
            ext_c_store=ext_c_store,
            ext_c_id2idx=ext_c_id2idx,
            device=device,
        )
        for rk in rerank_k_list:
            out["ext"][rk] = _build_reranked_results_for_k(
                retrieved0=retrieved0,
                scores_per_query=ext_scores,
                rerank_k=rk,
                k_eval=k_eval,
            )

    return out


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="", help="Task name like cirr_task7 / oven_task8")
    parser.add_argument("--embed_root", type=str, default="embed/test/")
    parser.add_argument("--mbeir_data_dir", type=str, default="/data/M-BEIR")

    # LamRA dict.pt stores
    parser.add_argument("--query_store", type=str, default=None)
    parser.add_argument("--cand_store", type=str, default=None)

    parser.add_argument("--eval_query_jsonl", type=str, default=None)
    parser.add_argument("--qrels", type=str, default=None)

    # Embedding-only mode (also used for external store retrieval)
    parser.add_argument("--emb_only", action="store_true", default=False, help="If set, do embedding-only retrieval.")
    parser.add_argument("--rerank_query_store", type=str, default=None, help="Query emb store: .pt dict or .npy")
    parser.add_argument("--rerank_cand_store", type=str, default=None, help="Cand emb store: .pt dict or .npy")
    parser.add_argument("--rerank_query_ids", type=str, default=None, help="(npy only) explicit ids path for query")
    parser.add_argument("--rerank_cand_ids", type=str, default=None, help="(npy only) explicit ids path for cand")
    parser.add_argument("--emb_chunk", type=int, default=200000, help="cand chunk size for emb-only retrieval")
    parser.add_argument("--emb_normalize", action="store_true", default=True, help="L2-normalize embeddings")

    # External store auto-resolve
    parser.add_argument(
        "--external_rerank_dir",
        type=str,
        default=None,
        help="Auto-resolve external npy paths by task name (only if rerank_sources includes external).",
    )
    parser.add_argument(
        "--external_use_union_cand",
        action="store_true",
        default=False,
        help="Use union cand embeddings/ids for external store when --external_rerank_dir is set.",
    )

    # Checkpoints (SQ-guided generation mode)
    parser.add_argument("--t5_ckpt", type=str, default=None)
    parser.add_argument(
        "--sq_ckpt",
        type=str,
        default=None,
        help="Optional. If not set, SQ projector weights will be loaded from --t5_ckpt.",
    )
    parser.add_argument("--t5_name", type=str, default="google-t5/t5-small")
    parser.add_argument("--rq_ckpt", type=str, default=None)
    parser.add_argument("--rq_yaml", type=str, default=None)

    # Eval options
    parser.add_argument("--save_dir", type=str, default="/data/likaipeng/gur/eval_outputs_sq_guided")
    parser.add_argument("--modality_index", action="store_true", default=True)
    parser.add_argument("--codebook_vocab", type=int, default=4096)
    parser.add_argument("--total_levels", type=int, default=9)

    parser.add_argument("--num_beams", type=int, default=50)
    parser.add_argument("--use_fp16", action="store_true", default=True)
    parser.add_argument("--use_trie", dest="use_trie", action="store_true", help="Enable trie constraint")
    parser.add_argument("--no_use_trie", dest="use_trie", action="store_false", help="Disable trie constraint")
    parser.set_defaults(use_trie=True)

    parser.add_argument("--metrics", type=str, default="Recall@1,Recall@5,Recall@10,Recall@20,Recall@50")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    # Rerank switches
    parser.add_argument("--rerank", action="store_true", default=False)
    parser.add_argument(
        "--rerank_k",
        action="append",
        default=None,
        help="Support repeated args or comma-separated string, e.g. --rerank_k 3 --rerank_k 5 or --rerank_k 3,5,10",
    )
    parser.add_argument(
        "--rerank_sources",
        type=str,
        default="lamra",
        help="Comma-separated rerank sources when --rerank is enabled: lamra, clip-sf, or lamra,clip-sf",
    )

    # Report options
    parser.add_argument(
        "--report_base_and_rerank",
        action="store_true",
        default=False,
        help="When --rerank is enabled, also report BASE (rerank=off) results in the same run.",
    )

    # SQ guidance params
    parser.add_argument("--sq_tau", type=float, default=0.05)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--disable_guidance", action="store_true", default=False)
    parser.add_argument("--guide_topk_list", type=str, default="0,1000")

    parser.add_argument("--out_tsv", type=str, default="recall_results.tsv")

    # Candidate code cache control
    parser.add_argument("--rebuild_cand_codes", action="store_true", default=False)
    parser.add_argument("--cand_code_prefix", type=str, default="auto")

    args = parser.parse_args()

    init_distributed_if_needed()
    device = (
        torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    os.makedirs(args.save_dir, exist_ok=True)

    do_lamra, do_ext = _parse_rerank_sources(args.rerank_sources)

    gen_code_dir = os.path.join(args.save_dir, "gen_code")
    retrieval_results_dir = os.path.join(args.save_dir, "retrieval_results")
    os.makedirs(gen_code_dir, exist_ok=True)
    os.makedirs(retrieval_results_dir, exist_ok=True)

    _maybe_autofill_task_paths(args)
    _maybe_fill_external_rerank_paths(args)

    if args.eval_query_jsonl in (None, "") or args.qrels in (None, ""):
        raise ValueError("Missing --eval_query_jsonl or --qrels (or provide --task for auto-fill)")

    _metric_list, recall_metrics, k_eval = _parse_metrics(args.metrics)
    rerank_k_list = _parse_rerank_k_list(args.rerank_k)
    max_rerank_k = max(rerank_k_list) if args.rerank else 0
    retrieve_k = k_eval if not args.rerank else max(k_eval, max_rerank_k)
    guide_topk_list = _parse_guide_topk_list(args.guide_topk_list)

    # -------------------------
    # A) Embedding-only mode
    # -------------------------
    if args.emb_only:
        if args.rerank_query_store in (None, "") or args.rerank_cand_store in (None, ""):
            raise ValueError("--emb_only requires --rerank_query_store and --rerank_cand_store")

        q_type, q_store, q_ids, _q_id2idx = load_any_emb_store(args.rerank_query_store, args.rerank_query_ids)
        c_type, c_store, c_ids, _c_id2idx = load_any_emb_store(args.rerank_cand_store, args.rerank_cand_ids)

        if is_main():
            print(f"[emb_only] query_store={args.rerank_query_store} ({q_type})")
            print(f"[emb_only] cand_store ={args.rerank_cand_store} ({c_type})")
            print(f"[emb_only] |Q|={len(q_ids)} |C|={len(c_ids)} normalize={args.emb_normalize} chunk={args.emb_chunk}")

        qrels, qid_to_task = load_qrel(args.qrels)

        class NpyQueryIndexDataset(Dataset):
            def __init__(self, q_ids_arr: np.ndarray):
                self.q_ids = q_ids_arr

            def __len__(self) -> int:
                return int(self.q_ids.shape[0])

            def __getitem__(self, idx: int) -> int:
                return int(idx)

        def collate_indices(batch: List[int]) -> torch.Tensor:
            return torch.tensor(batch, dtype=torch.long)

        ds = NpyQueryIndexDataset(q_ids)
        sampler = (
            DistributedSampler(ds, num_replicas=get_world_size(), rank=get_rank(), shuffle=False)
            if is_dist()
            else None
        )
        dl = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_indices,
        )

        all_qids_h: List[int] = []
        all_retrieved_hids: List[List[int]] = []

        total_queries = 0
        total_e2e_time = 0.0
        total_rerank_time = 0.0  # emb_only 下固定为 0
        total_time = 0.0

        dl_bar = tqdm(dl) if is_main() else dl
        for batch_idx in dl_bar:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_t0 = time.perf_counter()

            batch_idx_np = batch_idx.cpu().numpy().astype(np.int64)
            batch_qids = q_ids[batch_idx_np].astype(np.int64, copy=False)

            if q_type == "npy":
                q_emb_np = np.asarray(q_store[batch_idx_np])
                q_emb = torch.from_numpy(q_emb_np).to(device=device, non_blocking=True)
            else:
                q_emb = q_store[batch_idx_np].to(device=device, non_blocking=True)

            if c_type == "npy":
                _, top_idx = emb_retrieve_topk_chunked(
                    q_emb=q_emb,
                    cand_emb_npy=c_store,
                    topk=k_eval,
                    normalize=args.emb_normalize,
                    chunk_size=int(args.emb_chunk),
                )
                top_idx_cpu = top_idx.detach().cpu().numpy()
                retrieved_hids = c_ids[top_idx_cpu]
            else:
                cand_emb_cpu = c_store  # CPU tensor [N, D]
                _, top_idx = emb_retrieve_topk_chunked_tensor(
                    q_emb=q_emb,
                    cand_emb_cpu=cand_emb_cpu,
                    topk=k_eval,
                    normalize=args.emb_normalize,
                    chunk_size=int(args.emb_chunk),
                )
                top_idx_cpu = top_idx.detach().cpu().numpy()
                retrieved_hids = c_ids[top_idx_cpu]

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_t1 = time.perf_counter()

            batch_total_time = batch_t1 - batch_t0
            total_e2e_time += batch_total_time
            total_time += batch_total_time
            total_queries += len(batch_qids)

            if is_dist():
                gathered_qids = [None for _ in range(get_world_size())]
                gathered_rets = [None for _ in range(get_world_size())]
                dist.gather_object(batch_qids, gathered_qids if is_main() else None, dst=0)
                dist.gather_object(retrieved_hids, gathered_rets if is_main() else None, dst=0)
                if is_main():
                    all_qids_h.extend(np.concatenate(gathered_qids, axis=0).tolist())
                    all_retrieved_hids.extend(np.concatenate(gathered_rets, axis=0).tolist())
            else:
                all_qids_h.extend(batch_qids.tolist())
                all_retrieved_hids.extend(retrieved_hids.tolist())

        barrier()
        if is_main():
            tsv_path = os.path.join(args.save_dir, args.out_tsv)
            setting_name = f"emb_only_norm={int(args.emb_normalize)}_chunk={args.emb_chunk}"
            base_means = compute_recall_means_by_task(
                qids_h=all_qids_h,
                retrieved_hids=all_retrieved_hids,
                qrels=qrels,
                qid_to_task=qid_to_task,
                recall_metrics=recall_metrics,
            )
            print_result_block(
                dataset_name=args.task,
                setting=setting_name,
                means_by_task=base_means,
                recall_metrics=recall_metrics,
            )

            e2e_qps = (total_queries / total_e2e_time) if total_e2e_time > 0 else 0.0
            total_qps = (total_queries / total_time) if total_time > 0 else 0.0

            print("\n" + "-" * 80)
            print("[QPS]")
            print("-" * 80)
            print(f"  total_queries      = {total_queries}")
            print(f"  total_e2e_time     = {total_e2e_time:.4f} s")
            print(f"  total_rerank_time  = {total_rerank_time:.4f} s")
            print(f"  total_time         = {total_time:.4f} s")
            print(f"  e2e_qps            = {e2e_qps:.4f}")
            print(f"  total_qps          = {total_qps:.4f}")
            print("-" * 80)

            write_eval_results_tsv(
                tsv_path=tsv_path,
                dataset=args.task,
                split="test",
                cand_pool="union" if args.external_use_union_cand else (args.task or "cand"),
                base_means=base_means,
                recall_metrics=recall_metrics,
            )
            print(f"[tsv] wrote: {tsv_path}")

        barrier()
        destroy_dist()
        return

    # -------------------------
    # B) SQ-guided generation mode
    # -------------------------
    for need in ["t5_ckpt", "rq_ckpt", "rq_yaml", "query_store", "cand_store"]:
        if getattr(args, need) in (None, ""):
            raise ValueError(f"Missing required argument: --{need} (or use --emb_only)")

    query_store = load_store_pt(args.query_store, device, to_gpu=True)
    cand_store = load_store_pt(args.cand_store, device, to_gpu=False)

    emb_dim = int(query_store["emb"].shape[1])
    if is_main():
        print(f"[env] world={get_world_size()} device={device} save_dir={args.save_dir}")
        print(f"[store] emb_dim={emb_dim} query={args.query_store} cand={args.cand_store}")

    tokenizer = build_code_tokenizer(
        args.codebook_vocab, args.total_levels, t5_name=args.t5_name, modality_index=args.modality_index
    )
    t5 = load_t5_model(args.t5_ckpt, args.t5_name, tokenizer, emb_dim, device)

    sq_src = args.sq_ckpt or args.t5_ckpt
    if is_main():
        if args.sq_ckpt in (None, ""):
            print(f"[SQ] --sq_ckpt not provided, fallback to --t5_ckpt: {sq_src}")
        else:
            print(f"[SQ] load from --sq_ckpt: {sq_src}")

    sq = load_sq(sq_src, device)
    rq = load_rq(args.rq_yaml, args.rq_ckpt, device, emb_dim=emb_dim)
    barrier()

    # Candidate codes cache (npz)
    if args.cand_code_prefix == "auto":
        prefix = "union_" if _is_union_cand_store(args.cand_store) else ""
    else:
        prefix = args.cand_code_prefix.strip()
        prefix = (prefix + "_") if (prefix and not prefix.endswith("_")) else prefix

    cand_npz_name = f"{prefix}cand_{args.task}_code.npz" if getattr(args, "task", "") else f"{prefix}cand_code.npz"
    cand_npz_path = os.path.join(gen_code_dir, cand_npz_name)

    if (not args.rebuild_cand_codes) and os.path.exists(cand_npz_path):
        data = np.load(cand_npz_path, allow_pickle=False)
        cand_codes, cand_ids = data["codes"], data["ids"]
        if is_main():
            print(f"[cand] loaded cached {cand_npz_path}, codes={cand_codes.shape}, ids={cand_ids.shape}")
    else:
        if is_main():
            print(f"[cand] build codes -> {cand_npz_path}")

        cand_ids_list = get_ids_from_id_to_index(cand_store["id_to_index"])
        local_ids = cand_ids_list[get_rank()::get_world_size()] if is_dist() else cand_ids_list
        local_codes_list: List[torch.Tensor] = []

        batch = 4096
        it = range(0, len(local_ids), batch)
        if is_main():
            it = tqdm(it, desc="[Gen Candidates]")

        for st in it:
            chunk = local_ids[st:st + batch]
            idx_cpu = torch.tensor([cand_store["id_to_index"][i] for i in chunk], device="cpu", dtype=torch.long)

            p_emb = cand_store["emb"][idx_cpu].to(device=device, non_blocking=True)
            p_img_mask = cand_store["img_mask"][idx_cpu].to(device=device, non_blocking=True)
            p_txt_mask = cand_store["txt_mask"][idx_cpu].to(device=device, non_blocking=True)

            codes = rq_emb_to_codes(rq, p_emb, p_img_mask, p_txt_mask)
            local_codes_list.append(codes.cpu())

        local_codes = torch.cat(local_codes_list, dim=0).numpy().astype(np.int32)
        local_ids_arr = np.asarray(local_ids, dtype=np.int64)

        if is_dist():
            barrier()
            gathered_codes = [None for _ in range(get_world_size())]
            gathered_ids = [None for _ in range(get_world_size())]
            dist.gather_object(local_codes, gathered_codes if is_main() else None, dst=0)
            dist.gather_object(local_ids_arr, gathered_ids if is_main() else None, dst=0)

            if is_main():
                cand_codes = np.concatenate(gathered_codes, axis=0)
                cand_ids = np.concatenate(gathered_ids, axis=0)
                np.savez(cand_npz_path, codes=cand_codes, ids=cand_ids)
                print(f"[cand] saved {cand_npz_path}, codes={cand_codes.shape}, ids={cand_ids.shape}")

            barrier()
            data = np.load(cand_npz_path, allow_pickle=False)
            cand_codes, cand_ids = data["codes"], data["ids"]
        else:
            cand_codes, cand_ids = local_codes, local_ids_arr
            np.savez(cand_npz_path, codes=cand_codes, ids=cand_ids)
            if is_main():
                print(f"[cand] saved {cand_npz_path}, codes={cand_codes.shape}, ids={cand_ids.shape}")

    trie = None
    if args.use_trie:
        id_map = build_token_id_map(tokenizer, args.total_levels, args.codebook_vocab, modality_index=args.modality_index)
        cand_token_ids: List[List[int]] = []
        iterator = cand_codes.tolist()
        if is_main():
            iterator = tqdm(iterator, desc="[Trie Conversion]")
        for row in iterator:
            seq = [id_map[l][int(v)] for l, v in enumerate(row)]
            cand_token_ids.append(seq)
        trie = Trie(cand_token_ids)
    barrier()

    # Rank0-only resources
    cand_hash = None
    lamra_id2idx = None
    qrels = None
    qid_to_task = None
    if is_main():
        cand_hash = create_hash_map(cand_codes, cand_ids)
        lamra_id2idx = {int(k): int(v) for k, v in cand_store["id_to_index"].items()}
        qrels, qid_to_task = load_qrel(args.qrels)
        print(f"[qrels] loaded {args.qrels}, qrels_q={len(qrels)}")

    # External rerank store (optional)
    use_external_rerank = False
    ext_q_type = ext_c_type = None
    ext_q_store = ext_c_store = None
    ext_q_id2idx = ext_c_id2idx = None

    if args.rerank and do_ext:
        if (args.rerank_query_store in (None, "")) or (args.rerank_cand_store in (None, "")):
            raise ValueError(
                "rerank_sources includes 'external' but rerank_query_store/cand_store is missing. "
                "Provide --rerank_query_store/--rerank_cand_store or --external_rerank_dir."
            )
        use_external_rerank = True
        ext_q_type, ext_q_store, _q_ids, ext_q_id2idx = load_any_emb_store(args.rerank_query_store, args.rerank_query_ids)
        ext_c_type, ext_c_store, _c_ids, ext_c_id2idx = load_any_emb_store(args.rerank_cand_store, args.rerank_cand_ids)

    _print_rerank_stores(
        args=args,
        base_query_store=args.query_store,
        base_cand_store=args.cand_store,
        ext_enabled=use_external_rerank,
        ext_q_type=ext_q_type,
        ext_c_type=ext_c_type,
    )

    offsets, _V_T = get_level_offsets(args.total_levels, args.codebook_vocab, args.modality_index)

    code_tokens: List[str] = []
    for li, p in enumerate(LEVEL_PREFIXES[:args.total_levels]):
        limit = 3 if (args.modality_index and li == 0) else args.codebook_vocab
        for c in range(limit):
            code_tokens.append(f"<{p}{c}>")
    code_token_ids_all = torch.tensor(tokenizer.convert_tokens_to_ids(code_tokens), dtype=torch.long, device=device)

    E_simul = build_E_simul_from_rq(rq, args.total_levels, args.codebook_vocab, args.modality_index)
    docs_token_ids_cpu = build_docs_token_ids_from_cand_codes(
        cand_codes, args.total_levels, args.codebook_vocab, args.modality_index
    )
    barrier()

    ds = QueryIdDataset(args.eval_query_jsonl)
    sampler = (
        DistributedSampler(ds, num_replicas=get_world_size(), rank=get_rank(), shuffle=False)
        if is_dist()
        else None
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_qids,
    )

    for guide_topk in guide_topk_list:
        use_guidance = (not args.disable_guidance) and (args.guidance_scale > 0) and (guide_topk > 0)

        all_qids_h: List[int] = []
        all_retrieved_base: List[List[int]] = []
        all_retrieved_lamra_by_k: Dict[int, List[List[int]]] = {k: [] for k in rerank_k_list}
        all_retrieved_ext_by_k: Dict[int, List[List[int]]] = {k: [] for k in rerank_k_list}

        total_queries = 0
        total_e2e_time = 0.0       # exclude rerank
        total_rerank_time = 0.0    # rerank only
        total_time = 0.0           # full time = e2e + rerank

        dl_bar = tqdm(dl) if is_main() else dl
        max_new_tokens = args.total_levels

        for qids_h in dl_bar:
            batch_rerank_time = 0.0
            batch_e2e_time = 0.0

            idx = torch.tensor([query_store["id_to_index"][int(q)] for q in qids_h], device=device, dtype=torch.long)

            q_emb = query_store["emb"][idx]
            q_img_mask = query_store["img_mask"][idx]
            q_txt_mask = query_store["txt_mask"][idx]

            q_enc = rq.infer(
                emb=q_emb.float(),
                img_mask=q_img_mask,
                txt_mask=q_txt_mask,
                return_codes=False,
                normalize_out=True,
            )["encode"]

            s_q_full = compute_s_q_full(
                sq=sq,
                q_lamra_emb=q_emb,
                E_simul=E_simul,
                tau=args.sq_tau,
            )

            guidance_processor = None
            if use_guidance:
                top_scores, top_indices = topk_docs_by_sq_chunked(
                    s_q_full=s_q_full,
                    docs_token_ids_cpu=docs_token_ids_cpu,
                    guide_topk=guide_topk,
                    chunk_size=50000,
                )
                top_indices_cpu = top_indices.detach().cpu().numpy()
                B2, _ = top_indices_cpu.shape
                sel_codes = np.stack([cand_codes[top_indices_cpu[b]] for b in range(B2)], axis=0)
                selected_codes = torch.from_numpy(sel_codes).to(device=device, dtype=torch.long)

                offsets_t = torch.tensor(offsets, device=device, dtype=torch.long).view(1, 1, -1)
                logical_ids = selected_codes + offsets_t
                topk_token_ids = code_token_ids_all[logical_ids]

                guidance_processor = VectorizedGuidanceLogitsProcessor(
                    topk_codes=topk_token_ids,
                    topk_scores=top_scores,
                    guidance_scale=args.guidance_scale,
                    num_beams=args.num_beams,
                )

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            gen_t0 = time.perf_counter()

            gen = t5.generate_codes(
                emb=q_enc,
                num_beams=args.num_beams,
                max_new_tokens=max_new_tokens,
                use_fp16=args.use_fp16,
                device=device,
                trie_constraint=trie,
                logits_processor=guidance_processor,
            )

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            gen_t1 = time.perf_counter()
            batch_e2e_time += (gen_t1 - gen_t0)

            B = len(qids_h)
            gen = gen.view(B, args.num_beams, -1).cpu()

            beam_codes = np.zeros((B, args.num_beams, args.total_levels), dtype=np.int32)
            for i in range(B):
                for b in range(args.num_beams):
                    codes = decode_ids_to_codes(
                        tokenizer,
                        gen[i, b],
                        args.total_levels,
                        args.codebook_vocab,
                        modality_index=args.modality_index,
                    )
                    beam_codes[i, b] = codes if codes is not None else -1

            if is_dist():
                local_beams = beam_codes
                local_qids = qids_h.numpy()

                gathered_beams = [None for _ in range(get_world_size())]
                gathered_qids = [None for _ in range(get_world_size())]
                dist.gather_object(local_beams, gathered_beams if is_main() else None, dst=0)
                dist.gather_object(local_qids, gathered_qids if is_main() else None, dst=0)

                if is_main():
                    batch_beams = np.concatenate(gathered_beams, axis=0)
                    batch_qids = np.concatenate(gathered_qids, axis=0).astype(np.int64)

                    retrieved0 = retrieve_candidates_by_beams(batch_beams, cand_hash, retrieve_k)

                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    rerank_t0 = time.perf_counter()

                    out = _rank0_process_batch_multi_rerank(
                        args=args,
                        retrieved0=retrieved0,
                        batch_qids=batch_qids,
                        k_eval=k_eval,
                        rerank_k_list=rerank_k_list,
                        query_store=query_store,
                        cand_store=cand_store,
                        lamra_id2idx=lamra_id2idx,
                        device=device,
                        use_external_rerank=use_external_rerank,
                        ext_q_type=ext_q_type,
                        ext_q_store=ext_q_store,
                        ext_q_id2idx=ext_q_id2idx,
                        ext_c_type=ext_c_type,
                        ext_c_store=ext_c_store,
                        ext_c_id2idx=ext_c_id2idx,
                    )

                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    rerank_t1 = time.perf_counter()
                    batch_rerank_time += (rerank_t1 - rerank_t0)

                    all_retrieved_base.extend(out["base"])
                    for rk in rerank_k_list:
                        if rk in out["lamra"]:
                            all_retrieved_lamra_by_k[rk].extend(out["lamra"][rk])
                        if rk in out["ext"]:
                            all_retrieved_ext_by_k[rk].extend(out["ext"][rk])

                    all_qids_h.extend(batch_qids.tolist())
                    total_queries += len(batch_qids)

            else:
                if not is_main():
                    raise RuntimeError("Non-distributed run expects main process only.")

                batch_qids = qids_h.numpy().astype(np.int64)
                retrieved0 = retrieve_candidates_by_beams(beam_codes, cand_hash, retrieve_k)

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                rerank_t0 = time.perf_counter()

                out = _rank0_process_batch_multi_rerank(
                    args=args,
                    retrieved0=retrieved0,
                    batch_qids=batch_qids,
                    k_eval=k_eval,
                    rerank_k_list=rerank_k_list,
                    query_store=query_store,
                    cand_store=cand_store,
                    lamra_id2idx=lamra_id2idx,
                    device=device,
                    use_external_rerank=use_external_rerank,
                    ext_q_type=ext_q_type,
                    ext_q_store=ext_q_store,
                    ext_q_id2idx=ext_q_id2idx,
                    ext_c_type=ext_c_type,
                    ext_c_store=ext_c_store,
                    ext_c_id2idx=ext_c_id2idx,
                    query_emb_override=q_emb,
                )

                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                rerank_t1 = time.perf_counter()
                batch_rerank_time += (rerank_t1 - rerank_t0)

                all_retrieved_base.extend(out["base"])
                for rk in rerank_k_list:
                    if rk in out["lamra"]:
                        all_retrieved_lamra_by_k[rk].extend(out["lamra"][rk])
                    if rk in out["ext"]:
                        all_retrieved_ext_by_k[rk].extend(out["ext"][rk])

                all_qids_h.extend(batch_qids.tolist())
                total_queries += len(batch_qids)

            total_e2e_time += batch_e2e_time
            total_rerank_time += batch_rerank_time
            total_time = total_e2e_time + total_rerank_time

        barrier()
        if is_main():
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            cand_pool_name = "union" if _is_union_cand_store(args.cand_store) else (args.task or "cand")

            base_means = compute_recall_means_by_task(
                qids_h=all_qids_h,
                retrieved_hids=all_retrieved_base,
                qrels=qrels,
                qid_to_task=qid_to_task,
                recall_metrics=recall_metrics,
            )

            print_result_block(
                dataset_name=args.task,
                setting="rerank = off",
                means_by_task=base_means,
                recall_metrics=recall_metrics,
            )

            if not args.rerank:
                tsv_path = os.path.join(retrieval_results_dir, f"eval_results_{ts}_base.tsv")
                write_eval_results_tsv(
                    tsv_path=tsv_path,
                    dataset=args.task,
                    split="test",
                    cand_pool=cand_pool_name,
                    base_means=base_means,
                    recall_metrics=recall_metrics,
                )
                print(f"[tsv] wrote: {tsv_path}")
            else:
                if do_lamra:
                    for rk in rerank_k_list:
                        rr_means = compute_recall_means_by_task(
                            qids_h=all_qids_h,
                            retrieved_hids=all_retrieved_lamra_by_k[rk],
                            qrels=qrels,
                            qid_to_task=qid_to_task,
                            recall_metrics=recall_metrics,
                        )

                        print_result_block(
                            dataset_name=args.task,
                            setting=f"rerank = lamra (rerank_k={rk})",
                            means_by_task=rr_means,
                            recall_metrics=recall_metrics,
                        )

                        tsv_path = os.path.join(retrieval_results_dir, f"eval_results_{ts}_lamra_rk{rk}.tsv")
                        write_eval_results_tsv(
                            tsv_path=tsv_path,
                            dataset=args.task,
                            split="test",
                            cand_pool=cand_pool_name,
                            base_means=base_means,
                            recall_metrics=recall_metrics,
                            rerank1_model=f"LAMRA@{rk}",
                            rerank1_means=rr_means,
                        )
                        print(f"[tsv] wrote: {tsv_path}")

                if do_ext:
                    for rk in rerank_k_list:
                        rr_means = compute_recall_means_by_task(
                            qids_h=all_qids_h,
                            retrieved_hids=all_retrieved_ext_by_k[rk],
                            qrels=qrels,
                            qid_to_task=qid_to_task,
                            recall_metrics=recall_metrics,
                        )

                        print_result_block(
                            dataset_name=args.task,
                            setting=f"rerank = external (rerank_k={rk})",
                            means_by_task=rr_means,
                            recall_metrics=recall_metrics,
                        )

                        tsv_path = os.path.join(retrieval_results_dir, f"eval_results_{ts}_external_rk{rk}.tsv")
                        write_eval_results_tsv(
                            tsv_path=tsv_path,
                            dataset=args.task,
                            split="test",
                            cand_pool=cand_pool_name,
                            base_means=base_means,
                            recall_metrics=recall_metrics,
                            rerank1_model=f"CLIP-SF@{rk}",
                            rerank1_means=rr_means,
                        )
                        print(f"[tsv] wrote: {tsv_path}")

            e2e_qps = (total_queries / total_e2e_time) if total_e2e_time > 0 else 0.0
            rerank_qps = (total_queries / total_rerank_time) if total_rerank_time > 0 else 0.0
            total_qps = (total_queries / total_time) if total_time > 0 else 0.0

            print("\n" + "-" * 80)
            print("[QPS]")
            print("-" * 80)
            print(f"  total_queries      = {total_queries}")
            print(f"  total_e2e_time     = {total_e2e_time:.4f} s")
            print(f"  total_rerank_time  = {total_rerank_time:.4f} s")
            print(f"  total_time         = {total_time:.4f} s")
            print(f"  e2e_qps            = {e2e_qps:.4f}")
            print(f"  rerank_qps         = {rerank_qps:.4f}")
            print(f"  total_qps          = {total_qps:.4f}")
            print("-" * 80)

        barrier()

    destroy_dist()


if __name__ == "__main__":
    main()