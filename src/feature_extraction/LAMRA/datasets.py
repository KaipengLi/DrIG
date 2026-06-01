#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
datasets.py

M-BEIR dataset wrappers for LamRA-style embedding extraction.

This file intentionally keeps the original dataset logic unchanged:
- Query dataset loads query_jsonl and cand_pool_jsonl to build pool_dict
- Optional instruction prompting is applied if enabled
- Each sample returns "messages" (LamRA chat format) plus metadata:
    hid, img_mask, txt_mask, modality_id
"""

import os
from typing import Dict, List

from torch.utils.data import Dataset

import utils


class MBEIRTrainQueryDataset(Dataset):
    def __init__(
        self,
        mbeir_data_dir: str,
        query_jsonl_relpath: str,
        cand_pool_jsonl_relpath: str,
        instruct_tsv_relpath: str,
        enable_query_instruct: bool,
        shuffle_cand: bool,
    ):
        super().__init__()
        self.mbeir_data_dir = mbeir_data_dir

        self.query_path = os.path.join(mbeir_data_dir, query_jsonl_relpath)
        self.pool_path = os.path.join(mbeir_data_dir, cand_pool_jsonl_relpath)
        self.instruct_path = os.path.join(mbeir_data_dir, instruct_tsv_relpath)

        self.query_data = utils.load_jsonl(self.query_path)
        self.pool_data = utils.load_jsonl(self.pool_path)

        # Keep original behavior (always build pool_dict)
        self.pool_dict: Dict[str, dict] = {}
        for e in self.pool_data:
            did = e.get("did")
            if did is not None:
                self.pool_dict[did] = e

        self.enable_query_instruct = enable_query_instruct
        self.shuffle_cand = shuffle_cand
        self.prompts_dict = utils.load_query_instructions(self.instruct_path) if enable_query_instruct else {}

    def __len__(self) -> int:
        return len(self.query_data)

    def __getitem__(self, idx: int):
        e = self.query_data[idx]
        qid = e["qid"]
        qid_h = utils.hash_qid(qid)

        query_txt = e.get("query_txt") or ""
        query_txt = utils.format_string(query_txt)

        query_img_rel = e.get("query_img_path", None)
        query_img_abs = os.path.join(self.mbeir_data_dir, query_img_rel) if query_img_rel else None

        query_modality = e.get("query_modality", None)
        img_mask, txt_mask, modality_id = utils.modality_to_masks_and_id(query_modality)

        # instruction (depends on target modality, use pos cand modality)
        if self.enable_query_instruct:
            pos_list: List[str] = e.get("pos_cand_list", [])
            if len(pos_list) == 0:
                prompt = ""
            else:
                # 这里做一个“确定性选择”，避免跨 rank 的随机差异（不改变 shuffle_cand 的语义）
                if self.shuffle_cand:
                    pos_did = pos_list[qid_h % len(pos_list)]
                else:
                    pos_did = pos_list[0]

                pos_entry = self.pool_dict.get(pos_did, {})
                pos_modality = pos_entry.get("modality", None)

                dataset_id = qid.split(":")[0]
                key = f"{dataset_id}, {query_modality}, {pos_modality}"
                prompts = self.prompts_dict.get(key, [])
                if not prompts:
                    prompt = ""
                else:
                    prompt = utils.format_string(utils.pick_prompt(prompts, stable_seed=qid_h))

            if prompt:
                query_txt = utils.format_string(f"{prompt} {query_txt}")

        messages = utils.build_chat_messages(query_txt, query_img_abs)

        return {
            "messages": messages,
            "hid": qid_h,
            "img_mask": img_mask,
            "txt_mask": txt_mask,
            "modality_id": modality_id,
        }


class MBEIRTrainPoolDataset(Dataset):
    def __init__(self, mbeir_data_dir: str, cand_pool_jsonl_relpath: str):
        super().__init__()
        self.mbeir_data_dir = mbeir_data_dir
        self.pool_path = os.path.join(mbeir_data_dir, cand_pool_jsonl_relpath)
        self.pool_data = utils.load_jsonl(self.pool_path)

    def __len__(self) -> int:
        return len(self.pool_data)

    def __getitem__(self, idx: int):
        e = self.pool_data[idx]
        did = e["did"]
        did_h = utils.hash_did(did)

        txt = utils.format_string(e.get("txt") or "")
        img_rel = e.get("img_path", None)
        img_abs = os.path.join(self.mbeir_data_dir, img_rel) if img_rel else None

        modality = e.get("modality", None)
        img_mask, txt_mask, modality_id = utils.modality_to_masks_and_id(modality)

        messages = utils.build_chat_messages(txt if txt != "" else None, img_abs)

        return {
            "messages": messages,
            "hid": did_h,
            "img_mask": img_mask,
            "txt_mask": txt_mask,
            "modality_id": modality_id,
        }