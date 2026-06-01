# dataset.py
# -*- coding: utf-8 -*-

import json
import random
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from utils import get_rank


# =========================
# Hash ID (must match GENIUS)
# =========================
DATASET_QUERY_NUM_UPPER_BOUND = 500000
DATASET_CAND_NUM_UPPER_BOUND = 10000000


def hash_qid(qid: str) -> int:
    dataset_id, within_id = map(int, qid.split(":"))
    return dataset_id * DATASET_QUERY_NUM_UPPER_BOUND + within_id


def hash_did(did: str) -> int:
    dataset_id, within_id = map(int, did.split(":"))
    return dataset_id * DATASET_CAND_NUM_UPPER_BOUND + within_id


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


class RQTrainDataset(Dataset):
    """
    Each sample returns (emb naming):

      query_emb:        [D]
      cand_emb_all:     [L, D] where L = 1 + K (pos is always first)
      query_pos_did:    long scalar (hashed did label for this query's positive)
      cand_did_all:     [L] long tensor (hashed did for each candidate)

      masks:
        query_img_mask/query_txt_mask scalar
        cand_img_mask_all/cand_txt_mask_all [L]

    Debug fields (Python strings):
      query_id_str, query_modality_str, cand_id_str_all, cand_type_all
    """

    def __init__(
        self,
        subset_query_jsonl: str,
        query_dict: dict,
        pool_dict: dict,
        neg_k: int = 31,
        seed: int = 2023,
        expand_multi_pos: bool = True,
        fill_random_negatives: bool = True,
        neg_keys: Optional[List[str]] = None,
    ):
        super().__init__()
        self.seed = int(seed)
        self.neg_k = int(neg_k)
        self.expand_multi_pos = bool(expand_multi_pos)
        self.fill_random_negatives = bool(fill_random_negatives)
        self.neg_keys = neg_keys or [
            "neg_cand_list",
            "hard_neg_cand_list",
            "neg_list",
            "hard_neg_list",
        ]

        # Deterministic negative resampling across epochs
        self._epoch = 0

        # Embedding tables + index maps
        self.query_emb_table = query_dict["emb"]             # [Nq, D]
        self.cand_emb_table = pool_dict["emb"]               # [Np, D]
        self.query_id_to_index = query_dict["id_to_index"]   # hid(qid) -> index
        self.cand_id_to_index = pool_dict["id_to_index"]     # hid(did) -> index

        # Masks must exist
        self.query_img_mask_table = query_dict.get("img_mask", None)
        self.query_txt_mask_table = query_dict.get("txt_mask", None)
        self.cand_img_mask_table = pool_dict.get("img_mask", None)
        self.cand_txt_mask_table = pool_dict.get("txt_mask", None)
        if any(
            x is None
            for x in [
                self.query_img_mask_table,
                self.query_txt_mask_table,
                self.cand_img_mask_table,
                self.cand_txt_mask_table,
            ]
        ):
            raise KeyError("Need img_mask/txt_mask in both query_dict and pool_dict.")

        # Fast random-negative population (all hashed candidate dids)
        self.all_cand_hids = list(self.cand_id_to_index.keys())
        if len(self.all_cand_hids) == 0:
            raise ValueError("pool_dict['id_to_index'] is empty.")

        qdata = load_jsonl(subset_query_jsonl)

        self.samples: List[Dict[str, Any]] = []
        dropped = 0

        for e in qdata:
            qid_str = e.get("qid")
            pos_list = e.get("pos_cand_list", []) or []

            # Negatives might be stored under different keys
            neg_list = []
            for k in self.neg_keys:
                v = e.get(k, None)
                if isinstance(v, list) and len(v) > 0:
                    neg_list = v
                    break

            if qid_str is None or len(pos_list) == 0:
                dropped += 1
                continue

            qhid = hash_qid(qid_str)
            if qhid not in self.query_id_to_index:
                dropped += 1
                continue

            query_modality_str = e.get("query_modality", None)

            # Filter positives by existence
            pos_pairs: List[Tuple[int, str]] = []
            for did_str in pos_list:
                try:
                    phid = hash_did(did_str)
                except Exception:
                    continue
                if phid in self.cand_id_to_index:
                    pos_pairs.append((int(phid), did_str))

            if len(pos_pairs) == 0:
                dropped += 1
                continue

            # Filter negatives by existence
            neg_pairs: List[Tuple[int, str]] = []
            for did_str in neg_list:
                try:
                    nhid = hash_did(did_str)
                except Exception:
                    continue
                if nhid in self.cand_id_to_index:
                    neg_pairs.append((int(nhid), did_str))

            if self.expand_multi_pos:
                for phid, pos_did_str in pos_pairs:
                    self.samples.append(
                        {
                            "qhid": int(qhid),
                            "qid_str": qid_str,
                            "query_modality_str": query_modality_str,
                            "pos_pair": (int(phid), pos_did_str),
                            "neg_pairs": neg_pairs,
                            "task_id": e.get("task_id", -1),
                        }
                    )
            else:
                phid, pos_did_str = pos_pairs[0]
                self.samples.append(
                    {
                        "qhid": int(qhid),
                        "qid_str": qid_str,
                        "query_modality_str": query_modality_str,
                        "pos_pair": (int(phid), pos_did_str),
                        "neg_pairs": neg_pairs,
                        "task_id": e.get("task_id", -1),
                    }
                )

        if len(self.samples) == 0:
            raise ValueError("No valid training samples after filtering.")

        if get_rank() == 0:
            print(
                f"[RQTrainDataset] loaded {len(qdata)} raw queries, built {len(self.samples)} samples, dropped {dropped}"
            )

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic negative resampling."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def _rng(self, idx: int) -> random.Random:
        """
        Deterministic RNG per (epoch, idx).
        Note: avoids worker-id dependence.
        """
        return random.Random(self.seed + 1000003 * self._epoch + idx)

    def _sample_negs(
        self,
        rng: random.Random,
        pos_hid: int,
        provided_neg_pairs: List[Tuple[int, str]],
    ) -> List[Tuple[int, str]]:
        K = self.neg_k
        negs: List[Tuple[int, str]] = []
        used = {int(pos_hid)}

        # 1) Use provided negatives first (shuffled)
        if provided_neg_pairs:
            cand = list(provided_neg_pairs)
            rng.shuffle(cand)
            for hid, did_str in cand:
                hid = int(hid)
                if hid in used:
                    continue
                used.add(hid)
                negs.append((hid, did_str))
                if len(negs) >= K:
                    return negs[:K]

        # 2) Randomly fill remaining negatives (recommended for fixed shape)
        if self.fill_random_negatives:
            while len(negs) < K:
                hid = int(self.all_cand_hids[rng.randrange(0, len(self.all_cand_hids))])
                if hid in used:
                    continue
                used.add(hid)
                negs.append((hid, f"RND(hid={hid})"))
        else:
            negs = negs[:K]

        return negs

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        rng = self._rng(idx)

        qhid = int(s["qhid"])
        qid_str = s["qid_str"]
        query_modality_str = s.get("query_modality_str", None)

        pos_hid, pos_did_str = s["pos_pair"]
        provided_neg_pairs = s["neg_pairs"]

        # Sample K negatives
        neg_pairs = self._sample_negs(rng, pos_hid, provided_neg_pairs) if self.neg_k > 0 else []
        neg_hids = [hid for hid, _ in neg_pairs]
        neg_did_strs = [ds for _, ds in neg_pairs]

        # Candidate list: pos first
        cand_hids = [int(pos_hid)] + [int(h) for h in neg_hids]
        cand_id_str_all = [pos_did_str] + neg_did_strs
        cand_type_all = ["pos"] + (["neg"] * len(neg_hids))

        # Indices in embedding tables
        qidx = int(self.query_id_to_index[qhid])
        cidxs = [int(self.cand_id_to_index[hid]) for hid in cand_hids]

        query_emb = self.query_emb_table[qidx]        # [D]
        cand_emb_all = self.cand_emb_table[cidxs]     # [L, D]

        # Masks
        query_img_mask = self.query_img_mask_table[qidx]
        query_txt_mask = self.query_txt_mask_table[qidx]
        cand_img_mask_all = self.cand_img_mask_table[cidxs]
        cand_txt_mask_all = self.cand_txt_mask_table[cidxs]

        return {
            "query_emb": query_emb,
            "cand_emb_all": cand_emb_all,
            "query_img_mask": query_img_mask,
            "query_txt_mask": query_txt_mask,
            "cand_img_mask_all": cand_img_mask_all,
            "cand_txt_mask_all": cand_txt_mask_all,
            "query_pos_did": torch.tensor(int(pos_hid), dtype=torch.long),
            "cand_did_all": torch.tensor(cand_hids, dtype=torch.long),
            "pos_index_local": 0,  # pos is always first
            # Debug strings
            "query_id_str": qid_str,
            "query_modality_str": query_modality_str,
            "cand_id_str_all": cand_id_str_all,
            "cand_type_all": cand_type_all,
        }


def collate_rq_train(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate into a flattened candidate matrix.

    Output keys (emb naming):
      query_emb:        [B, D]
      cand_emb:         [P, D] where P = B * L
      pos_index:        [B] indices in flattened cand_emb (pos is always first in each block)
      query_pos_did:    [B]
      cand_did:         [P]
      query_img_mask:   [B]
      query_txt_mask:   [B]
      cand_img_mask:    [P]
      cand_txt_mask:    [P]
    """
    query_emb = torch.stack([b["query_emb"] for b in batch], dim=0)  # [B, D]

    cand_emb_all = torch.stack([b["cand_emb_all"] for b in batch], dim=0)  # [B, L, D]
    B, L, D = cand_emb_all.shape
    cand_emb = cand_emb_all.view(B * L, D)  # [P, D]

    query_img_mask = torch.stack([b["query_img_mask"] for b in batch], dim=0).long().view(B)
    query_txt_mask = torch.stack([b["query_txt_mask"] for b in batch], dim=0).long().view(B)

    cand_img_mask = torch.stack([b["cand_img_mask_all"] for b in batch], dim=0).long().view(B * L)
    cand_txt_mask = torch.stack([b["cand_txt_mask_all"] for b in batch], dim=0).long().view(B * L)

    query_pos_did = torch.stack([b["query_pos_did"] for b in batch], dim=0).long().view(B)
    cand_did = torch.stack([b["cand_did_all"] for b in batch], dim=0).long().view(B * L)

    pos_index = (torch.arange(B, dtype=torch.long) * L)  # pos is 0 in each block

    # Debug strings (Python lists)
    query_id_str = [b["query_id_str"] for b in batch]
    query_modality_str = [b.get("query_modality_str", None) for b in batch]

    cand_id_str_all: List[str] = []
    cand_type_all: List[str] = []
    for i in range(B):
        cand_id_str_all.extend(batch[i]["cand_id_str_all"])
        cand_type_all.extend(batch[i]["cand_type_all"])

    cand_is_pos = torch.zeros((B * L,), dtype=torch.bool)
    cand_is_pos[pos_index] = True

    return {
        "query_emb": query_emb,
        "cand_emb": cand_emb,
        "pos_index": pos_index,
        "query_pos_did": query_pos_did,
        "cand_did": cand_did,
        "query_img_mask": query_img_mask,
        "query_txt_mask": query_txt_mask,
        "cand_img_mask": cand_img_mask,
        "cand_txt_mask": cand_txt_mask,
        "cand_is_pos": cand_is_pos,
        # Debug
        "query_id_str": query_id_str,
        "query_modality_str": query_modality_str,
        "cand_id_str_all": cand_id_str_all,
        "cand_type_all": cand_type_all,
        "block_size": L,
    }