# -*- coding: utf-8 -*-
"""
datasets.py

Dataset + collate_fn for GUR training.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset as TorchDataset

from .utils import JsonlOffsetReader, hash_qid, hash_did


class Dataset(TorchDataset):
    """
    Training sample:
      q_emb, p_emb
      q_img_mask, q_txt_mask
      p_img_mask, p_txt_mask
    """

    def __init__(
        self,
        jsonl_path: str,
        query_store: Dict[str, Any],
        cand_store: Dict[str, Any],
        offsets: List[int],
    ):
        self.reader = JsonlOffsetReader(jsonl_path, offsets)
        self.query_store = query_store
        self.cand_store = cand_store

        # Hash-id -> row index lookup
        self.qid2idx = query_store["id_to_index"]
        self.did2idx = cand_store["id_to_index"]

    def __len__(self) -> int:
        return len(self.reader.offsets)

    def _extract_hashed_ids(self, rec: Dict[str, Any]) -> Tuple[int, int]:
        """
        Return:
          qid_hid: hashed query id (int)
          did_hid: hashed positive doc id (int)
        """
        qid_hid = rec.get("hashed_qid", None)
        if qid_hid is None:
            qid = rec.get("qid", None)
            if qid is not None:
                qid_hid = hash_qid(qid)
        if qid_hid is None:
            raise KeyError("Cannot find hashed_qid (or qid).")

        did_hid = rec.get("hashed_did", None)
        if did_hid is None:
            did_hid = rec.get("pos_hashed_did", None)
        if did_hid is None:
            lst = rec.get("pos_hashed_did_list", None)
            if isinstance(lst, list) and len(lst) > 0:
                did_hid = lst[0]
        if did_hid is None:
            pos_list = rec.get("pos_cand_list", None)
            if isinstance(pos_list, list) and len(pos_list) > 0:
                did_hid = hash_did(pos_list[0])
        if did_hid is None:
            raise KeyError("Cannot find hashed_did (pos).")

        return int(qid_hid), int(did_hid)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Try a few times to skip invalid (qid/did missing in stores).
        """
        for _ in range(5):
            rec = self.reader.read(idx)
            qid_hid, did_hid = self._extract_hashed_ids(rec)

            if qid_hid not in self.qid2idx or did_hid not in self.did2idx:
                idx = random.randint(0, len(self) - 1)
                continue

            q_idx = self.qid2idx[qid_hid]
            p_idx = self.did2idx[did_hid]

            return {
                "q_emb": self.query_store["emb"][q_idx],
                "p_emb": self.cand_store["emb"][p_idx],
                "q_img_mask": self.query_store["img_mask"][q_idx],
                "q_txt_mask": self.query_store["txt_mask"][q_idx],
                "p_img_mask": self.cand_store["img_mask"][p_idx],
                "p_txt_mask": self.cand_store["txt_mask"][p_idx],
            }

        raise RuntimeError(f"Failed to fetch valid sample after retries. last_idx={idx}")


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Simple stack collate.
    """
    out: Dict[str, torch.Tensor] = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out