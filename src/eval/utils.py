# utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# =========================================================
# Distributed helpers
# =========================================================
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main() -> bool:
    return (not is_dist()) or get_rank() == 0


def init_distributed_if_needed() -> None:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def barrier() -> None:
    if is_dist():
        dist.barrier()


def destroy_dist() -> None:
    if is_dist():
        dist.destroy_process_group()


def gather_object_to_main(obj):
    """
    Gather a python object from all ranks to rank0.
    Return list on rank0, else None.
    """
    if not is_dist():
        return [obj]
    gathered = [None for _ in range(get_world_size())]
    dist.gather_object(obj, gathered if is_main() else None, dst=0)
    return gathered if is_main() else None


# =========================================================
# ID hash / unhash utils
# =========================================================
DATASET_QUERY_NUM_UPPER_BOUND = 500000
DATASET_CAN_NUM_UPPER_BOUND = 10000000


def hash_qid(qid: str) -> int:
    dataset_id, within_id = map(int, qid.split(":"))
    return dataset_id * DATASET_QUERY_NUM_UPPER_BOUND + within_id


def hash_did(did: str) -> int:
    dataset_id, within_id = map(int, did.split(":"))
    return dataset_id * DATASET_CAN_NUM_UPPER_BOUND + within_id


def unhash_qid(h: int) -> str:
    dataset_id = int(h) // DATASET_QUERY_NUM_UPPER_BOUND
    within_id = int(h) % DATASET_QUERY_NUM_UPPER_BOUND
    return f"{dataset_id}:{within_id}"


def unhash_did(h: int) -> str:
    dataset_id = int(h) // DATASET_CAN_NUM_UPPER_BOUND
    within_id = int(h) % DATASET_CAN_NUM_UPPER_BOUND
    return f"{dataset_id}:{within_id}"


# =========================================================
# IO + metrics
# =========================================================
def load_jsonl(path: str) -> List[dict]:
    out: List[dict] = []
    with open(path, "r") as f:
        for line in f:
            out.append(json.loads(line))
    return out


def load_qrel(path: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    qrel: Dict[str, List[str]] = {}
    qid_to_task: Dict[str, str] = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            qid, _, did, rel, task_id = parts[:5]
            if int(rel) <= 0:
                continue
            qrel.setdefault(qid, []).append(did)
            if qid not in qid_to_task:
                qid_to_task[qid] = task_id
    return qrel, qid_to_task


def compute_recall_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    return 1.0 if set(relevant).intersection(set(retrieved[:k])) else 0.0


TASKID_TO_TASKNAME = {
    "0": "text -> image",
    "1": "text -> text",
    "2": "text -> image,text",
    "3": "image -> text",
    "4": "image -> image",
    "5": "image,text -> text",
    "6": "image,text -> image",
    "7": "text -> image (refine)",
    "8": "image -> text (refine)",
}


def compute_recall_means_by_task(
    *,
    qids_h: List[int],
    retrieved_hids: List[List[int]],
    qrels: Dict[str, List[str]],
    qid_to_task: Dict[str, str],
    recall_metrics: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Return:
      means[task_id][metric] = mean recall
    task_id is the same as in qrels (e.g., "0","1","2"...)
    """
    recall_by_task = defaultdict(lambda: defaultdict(list))

    for qh, retrieved_list in zip(qids_h, retrieved_hids):
        qid = unhash_qid(int(qh))
        if qid not in qrels:
            continue
        task_id = qid_to_task.get(qid, "NA")

        retrieved_dids = [unhash_did(int(x)) for x in retrieved_list]
        relevant = qrels[qid]

        for m in recall_metrics:
            k = int(m.split("@")[1])
            r = compute_recall_at_k(relevant, retrieved_dids, k)
            recall_by_task[task_id][m].append(r)

    means: Dict[str, Dict[str, float]] = {}
    for task_id, mdict in recall_by_task.items():
        means[task_id] = {}
        for m in recall_metrics:
            vals = mdict.get(m, [])
            means[task_id][m] = float(np.mean(vals)) if vals else 0.0
    return means


def write_eval_results_tsv(
    *,
    tsv_path: str,
    dataset: str,
    split: str,
    cand_pool: str,
    base_means: Dict[str, Dict[str, float]],
    recall_metrics: List[str],
    rerank1_model: Optional[str] = None,
    rerank1_means: Optional[Dict[str, Dict[str, float]]] = None,
    rerank2_model: Optional[str] = None,
    rerank2_means: Optional[Dict[str, Dict[str, float]]] = None,
    value_fmt: str = "{:.4f}",
) -> None:
    os.makedirs(os.path.dirname(tsv_path), exist_ok=True)

    two_rerank = (rerank1_model is not None and rerank1_means is not None and
                  rerank2_model is not None and rerank2_means is not None)
    one_rerank = (not two_rerank) and (rerank1_model is not None and rerank1_means is not None)

    task_ids = set(base_means.keys())
    if one_rerank or two_rerank:
        task_ids |= set(rerank1_means.keys())
    if two_rerank:
        task_ids |= set(rerank2_means.keys())

    def _sort_key(x: str):
        try:
            return (0, int(x))
        except Exception:
            return (1, x)
    task_ids = sorted(task_ids, key=_sort_key)

    if two_rerank:
        header = (
            "TaskID\tTask\tDataset\tSplit\tMetric\tCandPool\tValue\t"
            "RerankModel\tRerankValue\tRerankModel\tRerankValue\n"
        )
    elif one_rerank:
        header = (
            "TaskID\tTask\tDataset\tSplit\tMetric\tCandPool\tValue\t"
            "RerankModel\tRerankValue\n"
        )
    else:
        header = "TaskID\tTask\tDataset\tSplit\tMetric\tCandPool\tValue\n"

    with open(tsv_path, "w") as f:
        f.write(header)

        for task_id in task_ids:
            task_name = TASKID_TO_TASKNAME.get(str(task_id), str(task_id))

            for m in recall_metrics:
                base_v = base_means.get(task_id, {}).get(m, 0.0)

                if two_rerank:
                    v1 = rerank1_means.get(task_id, {}).get(m, 0.0)
                    v2 = rerank2_means.get(task_id, {}).get(m, 0.0)
                    f.write(
                        f"{task_id}\t{task_name}\t{dataset}\t{split}\t{m}\t{cand_pool}\t{value_fmt.format(base_v)}\t"
                        f"{rerank1_model}\t{value_fmt.format(v1)}\t{rerank2_model}\t{value_fmt.format(v2)}\n"
                    )
                elif one_rerank:
                    v1 = rerank1_means.get(task_id, {}).get(m, 0.0)
                    f.write(
                        f"{task_id}\t{task_name}\t{dataset}\t{split}\t{m}\t{cand_pool}\t{value_fmt.format(base_v)}\t"
                        f"{rerank1_model}\t{value_fmt.format(v1)}\n"
                    )
                else:
                    f.write(
                        f"{task_id}\t{task_name}\t{dataset}\t{split}\t{m}\t{cand_pool}\t{value_fmt.format(base_v)}\n"
                    )


def print_result_block(
    *,
    dataset_name: str,
    setting: str,
    means_by_task: Dict[str, Dict[str, float]],
    recall_metrics: List[str],
) -> None:
    task_ids = [tid for tid in means_by_task.keys() if tid != "NA"]
    task_ids.sort(key=lambda x: int(x) if str(x).isdigit() else 10**9)

    overall = {}
    for m in recall_metrics:
        vals = []
        for tid in task_ids:
            if m in means_by_task.get(tid, {}):
                vals.append(float(means_by_task[tid][m]))
        overall[m] = float(np.mean(vals)) if vals else 0.0

    print("\n" + "-" * 80)
    print(f"task={dataset_name}")
    print(f"[RESULT]  {setting}")
    print("-" * 80)
    for m in recall_metrics:
        print(f"  {m} = {overall[m]:.4f}")
    print("-" * 80)


# =========================================================
# Stores (.pt dict + .npy embedding store)
# =========================================================
def load_store_pt(path: str, device: torch.device, to_gpu: bool = True) -> dict:
    obj = torch.load(path, map_location="cpu")
    if to_gpu:
        for k in ["emb", "img_mask", "txt_mask"]:
            if k in obj and torch.is_tensor(obj[k]):
                obj[k] = obj[k].to(device, non_blocking=True)
    return obj


def _infer_ids_path_from_emb_path(emb_path: str) -> str:
    if emb_path.endswith("_embeddings.npy"):
        return emb_path.replace("_embeddings.npy", "_ids.npy")
    base = emb_path[:-4] if emb_path.endswith(".npy") else emb_path
    return base + "_ids.npy"


def load_npy_embeddings(emb_path: str, ids_path: Optional[str] = None, mmap: bool = True):
    if ids_path is None:
        ids_path = _infer_ids_path_from_emb_path(emb_path)
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"embeddings npy not found: {emb_path}")
    if not os.path.exists(ids_path):
        raise FileNotFoundError(f"ids npy not found: {ids_path} (inferred from {emb_path})")

    emb = np.load(emb_path, mmap_mode="r" if mmap else None)  # [N,D]
    ids = np.load(ids_path, mmap_mode="r" if mmap else None)  # [N]

    if emb.ndim != 2:
        raise ValueError(f"embeddings must be 2D [N,D], got {emb.shape} from {emb_path}")
    if ids.ndim != 1:
        raise ValueError(f"ids must be 1D [N], got {ids.shape} from {ids_path}")
    if emb.shape[0] != ids.shape[0]:
        n = min(emb.shape[0], ids.shape[0])
        emb = emb[:n]
        ids = ids[:n]

    return emb, ids.astype(np.int64, copy=False), ids_path


def build_id2idx_from_ids(ids: np.ndarray) -> Dict[int, int]:
    return {int(ids[i]): int(i) for i in range(ids.shape[0])}


def load_any_emb_store(path: str, ids_path: Optional[str]):
    """
    Return:
      store_type: "pt" | "npy"
      emb_store: torch.Tensor (CPU) or np.ndarray (memmap)
      ids_arr: np.ndarray int64 [N]
      id2idx: Dict[id -> row]
    """
    if path.endswith(".pt"):
        st = torch.load(path, map_location="cpu")
        if "emb" not in st or "id_to_index" not in st:
            raise ValueError(f"pt store must contain 'emb' and 'id_to_index': {path}")
        emb_cpu = st["emb"]
        id_to_index = {int(k): int(v) for k, v in st["id_to_index"].items()}
        ids = np.empty((emb_cpu.shape[0],), dtype=np.int64)
        for _id, _idx in id_to_index.items():
            ids[_idx] = _id
        return "pt", emb_cpu, ids, id_to_index

    if path.endswith(".npy"):
        emb_npy, ids_npy, _ = load_npy_embeddings(path, ids_path=ids_path, mmap=True)
        id2idx = build_id2idx_from_ids(ids_npy)
        return "npy", emb_npy, ids_npy, id2idx

    raise ValueError(f"Unsupported emb store: {path} (need .pt or .npy)")


# =========================================================
# Retrieval + rerank
# =========================================================
def get_ids_from_id_to_index(id_to_index: Dict) -> List[int]:
    return [int(x) for x in list(id_to_index.keys())]


def create_hash_map(cand_codes: np.ndarray, cand_ids: np.ndarray):
    hm = defaultdict(list)
    for code, cid in zip(cand_codes, cand_ids):
        hm[tuple(int(x) for x in code)].append(int(cid))
    return hm


def retrieve_candidates_by_beams(query_beam_codes, cand_hash, topk, per_code_cap=500):
    out = []
    for beams in query_beam_codes:
        uniq = []
        seen = set()
        for code_seq in beams:
            if np.any(code_seq == -1):
                continue
            key = tuple(int(x) for x in code_seq)
            if key not in cand_hash:
                continue
            for cid in cand_hash[key][:per_code_cap]:
                cid = int(cid)
                if cid in seen:
                    continue
                seen.add(cid)
                uniq.append(cid)
        out.append(uniq[:topk])
    return out


@torch.no_grad()
def rerank_with_embeddings_external_store(
    retrieved_ids: List[List[int]],
    batch_qids: List[int],
    q_store_type: str,
    q_store: Union[np.ndarray, torch.Tensor],
    q_id2idx: Dict[int, int],
    c_store_type: str,
    c_store: Union[np.ndarray, torch.Tensor],
    c_id2idx: Dict[int, int],
    device: torch.device,
    topk: int,
    normalize: bool = True,
) -> List[List[int]]:
    q_rows = []
    for qh in batch_qids:
        qh = int(qh)
        if qh not in q_id2idx:
            raise KeyError(f"[external rerank] query id {qh} not found in rerank_query_store")
        q_rows.append(q_id2idx[qh])

    if q_store_type == "pt":
        q_emb = q_store[torch.tensor(q_rows, dtype=torch.long)].to(device=device, non_blocking=True).float()
    else:
        q_np = np.asarray(q_store[np.asarray(q_rows, dtype=np.int64)])
        q_emb = torch.from_numpy(q_np).to(device=device, non_blocking=True).float()

    if normalize:
        q_emb = F.normalize(q_emb, dim=-1)

    reranked = []
    for i, cands in enumerate(retrieved_ids):
        if not cands:
            reranked.append([])
            continue

        filt_ids = []
        rows = []
        for cid in cands:
            cid = int(cid)
            if cid in c_id2idx:
                filt_ids.append(cid)
                rows.append(c_id2idx[cid])

        if not rows:
            reranked.append([])
            continue

        if c_store_type == "pt":
            cand_mat = c_store[torch.tensor(rows, dtype=torch.long)].to(device=device, non_blocking=True).float()
        else:
            c_np = np.asarray(c_store[np.asarray(rows, dtype=np.int64)])
            cand_mat = torch.from_numpy(c_np).to(device=device, non_blocking=True).float()

        if normalize:
            cand_mat = F.normalize(cand_mat, dim=-1)

        sims = cand_mat @ q_emb[i].to(cand_mat.dtype)
        k = min(int(topk), sims.numel())
        order = torch.topk(sims, k=k, largest=True).indices.tolist()
        reranked.append([filt_ids[j] for j in order])

    return reranked


@torch.no_grad()
def rerank_with_lamra_pt_store_subset(
    retrieved_ids: List[List[int]],
    query_emb: torch.Tensor,  # [B,D] on GPU
    cand_store_pt: dict,      # contains 'emb' on CPU
    cand_id2idx: Dict[int, int],
    device: torch.device,
    topk: int,
    normalize: bool = True,
) -> List[List[int]]:
    q = query_emb.float()
    if normalize:
        q = F.normalize(q, dim=-1)

    reranked: List[List[int]] = []
    cand_emb_cpu: torch.Tensor = cand_store_pt["emb"]

    for i, cands in enumerate(retrieved_ids):
        if not cands:
            reranked.append([])
            continue

        filt_ids: List[int] = []
        rows: List[int] = []
        for cid in cands:
            cid = int(cid)
            if cid in cand_id2idx:
                filt_ids.append(cid)
                rows.append(int(cand_id2idx[cid]))

        if not rows:
            reranked.append([])
            continue

        rows_t = torch.tensor(rows, dtype=torch.long, device="cpu")
        cand_mat = cand_emb_cpu.index_select(0, rows_t).to(device=device, non_blocking=True).float()

        if normalize:
            cand_mat = F.normalize(cand_mat, dim=-1)

        sims = cand_mat @ q[i].to(cand_mat.dtype)
        k = min(int(topk), sims.numel())
        order = torch.topk(sims, k=k, largest=True).indices.tolist()
        reranked.append([filt_ids[j] for j in order])

    return reranked


@torch.no_grad()
def emb_retrieve_topk_chunked(
    q_emb: torch.Tensor,
    cand_emb_npy: np.ndarray,
    topk: int,
    normalize: bool = True,
    chunk_size: int = 200000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = q_emb.device
    B, _ = q_emb.shape
    N = cand_emb_npy.shape[0]
    K = max(1, min(int(topk), N))

    q = q_emb.float()
    if normalize:
        q = F.normalize(q, dim=-1)

    top_scores = torch.full((B, K), -1e9, device=device, dtype=torch.float32)
    top_indices = torch.full((B, K), -1, device=device, dtype=torch.long)

    for st in range(0, N, chunk_size):
        ed = min(N, st + chunk_size)
        cand_chunk = np.asarray(cand_emb_npy[st:ed])
        c = torch.from_numpy(cand_chunk).to(device=device, non_blocking=True).float()
        if normalize:
            c = F.normalize(c, dim=-1)

        sims = (q @ c.t()).float()

        merged_scores = torch.cat([top_scores, sims], dim=1)
        chunk_idx = torch.arange(st, ed, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        merged_indices = torch.cat([top_indices, chunk_idx], dim=1)

        new_scores, new_pos = torch.topk(merged_scores, k=K, dim=1, largest=True)
        new_indices = merged_indices.gather(1, new_pos)
        top_scores, top_indices = new_scores, new_indices

    return top_scores, top_indices


# =========================================================
# Order-invariant module + scoring vector builder (NEW NAMING)
# =========================================================
class OrderInvariantModule(nn.Module):
    """
    Stage-1 module:
      order_invariant: nn.Sequential(Linear -> ReLU -> Linear)
    State_dict keys:
      order_invariant.0.weight / order_invariant.0.bias / order_invariant.2.weight
    """
    def __init__(self, in_dim: int, hidden: int = 4096):
        super().__init__()
        self.in_dim = int(in_dim)
        self.order_invariant = nn.Sequential(
            nn.Linear(self.in_dim, int(hidden)),
            nn.ReLU(),
            nn.Linear(int(hidden), self.in_dim, bias=False),
        )


def load_sq(sq_ckpt_path: str, device: torch.device) -> OrderInvariantModule:
    """
    Load Stage-1 'order_invariant' module from a training checkpoint.

    IMPORTANT:
    - This function only supports NEW naming: 'order_invariant.*'
    - It does NOT support legacy 'score_q_proj.*'
    - It does NOT support prefix 'module.' (to keep code simple as requested)
      => Ensure your checkpoint is saved from unwrap_ddp(model) or without DDP-prefix.
    """
    ckpt = torch.load(sq_ckpt_path, map_location="cpu")
    sd = ckpt.get("model", ckpt)

    k_w = "order_invariant.0.weight"
    if k_w not in sd:
        raise KeyError(f"Cannot find {k_w} in {sq_ckpt_path}")

    in_dim = int(sd[k_w].shape[1])
    hidden = int(sd[k_w].shape[0])

    mod = OrderInvariantModule(in_dim=in_dim, hidden=hidden).to(device)

    sub = {}
    for k, v in sd.items():
        if k.startswith("order_invariant."):
            sub[k] = v

    missing, unexpected = mod.load_state_dict(sub, strict=False)
    if is_main():
        print(f"[SQ load][order_invariant] in_dim={in_dim} hidden={hidden} missing={len(missing)} unexpected={len(unexpected)}")

    mod.eval()
    return mod


def compute_s_q_full(
    sq: OrderInvariantModule,
    q_lamra_emb: torch.Tensor,
    E_simul: torch.Tensor,
    tau: float = 0.05,
) -> torch.Tensor:
    device = q_lamra_emb.device
    E = E_simul.to(device=device, dtype=torch.float32)

    q = F.normalize(q_lamra_emb.float(), dim=-1)
    q_h = sq.order_invariant(q)
    q_h = F.normalize(q_h, dim=-1)

    logits = (q_h @ E.t()) / float(tau)
    s_q = torch.log1p(F.relu(logits))
    return s_q