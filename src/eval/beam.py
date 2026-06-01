# beam.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from omegaconf import OmegaConf
from transformers import AutoConfig, T5ForConditionalGeneration, T5TokenizerFast, PreTrainedTokenizerFast
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from models.residual_quantization.model import ResidualQuantizerModel as RQModel


# =========================================================
# Trie (pure python)
# =========================================================
class Trie:
    def __init__(self, sequences: List[List[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for seq in sequences:
                Trie._add(seq, self.trie_dict)
                self.len += 1
        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id: int):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, seq: List[int]):
        Trie._add(seq, self.trie_dict)
        self.len += 1

    def get(self, prefix_seq: List[int]):
        return Trie._get(prefix_seq, self.trie_dict, self.append_trie, self.bos_token_id)

    @staticmethod
    def _add(seq: List[int], node: Dict):
        if not seq:
            return
        h = seq[0]
        if h not in node:
            node[h] = {}
        Trie._add(seq[1:], node[h])

    @staticmethod
    def _get(prefix_seq: List[int], node: Dict, append_trie=None, bos_token_id: int = None):
        if len(prefix_seq) == 0:
            out = list(node.keys())
            if append_trie and bos_token_id in out:
                out.remove(bos_token_id)
                out += list(append_trie.trie_dict.keys())
            return out
        h = prefix_seq[0]
        if h in node:
            return Trie._get(prefix_seq[1:], node[h], append_trie, bos_token_id)
        if append_trie:
            return append_trie.get(prefix_seq)
        return []


# =========================================================
# Tokenizer & code decode
# =========================================================
LEVEL_PREFIXES = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
_code_pat = re.compile(r"^<([a-z])(\d+)>$")


def build_code_tokenizer(
    codebook_vocab: int,
    total_levels: int,
    t5_name: str = "google-t5/t5-base",
    modality_index: bool = True,
) -> PreTrainedTokenizerFast:
    tokenizer = T5TokenizerFast.from_pretrained(t5_name, model_max_length=512)
    code_tokens = []
    for li, p in enumerate(LEVEL_PREFIXES[:total_levels]):
        limit = 3 if (modality_index and li == 0) else codebook_vocab
        for c in range(limit):
            code_tokens.append(f"<{p}{c}>")
    tokenizer.add_tokens(code_tokens)
    return tokenizer


def build_token_id_map(
    tokenizer: PreTrainedTokenizerFast,
    total_levels: int,
    codebook_vocab: int,
    modality_index: bool = True,
) -> List[Dict[int, int]]:
    mapping: List[Dict[int, int]] = []
    for li in range(total_levels):
        level_map = {}
        p = LEVEL_PREFIXES[li]
        limit = 3 if (modality_index and li == 0) else codebook_vocab
        for c in range(limit):
            tok = f"<{p}{c}>"
            tid = tokenizer.convert_tokens_to_ids(tok)
            level_map[c] = tid
        mapping.append(level_map)
    return mapping


def decode_ids_to_codes(
    tokenizer: PreTrainedTokenizerFast,
    seq_ids: torch.Tensor,
    total_levels: int,
    codebook_vocab: int,
    modality_index: bool = True,
) -> Optional[np.ndarray]:
    toks = tokenizer.convert_ids_to_tokens(seq_ids.tolist())
    codes = []
    for t in toks:
        if t in ("<pad>", "</s>", "<unk>"):
            continue
        m = _code_pat.match(t)
        if not m:
            continue
        p, num = m.group(1), int(m.group(2))
        codes.append((p, num))
        if len(codes) >= total_levels:
            break

    if len(codes) < total_levels:
        return None

    for i in range(total_levels):
        if codes[i][0] != LEVEL_PREFIXES[i]:
            return None

    if modality_index and (codes[0][1] not in (0, 1, 2)):
        return None

    return np.asarray([c[1] for c in codes], dtype=np.int32)


# =========================================================
# Offsets & docs token ids
# =========================================================
def get_level_offsets(total_levels: int, codebook_vocab: int, modality_index: bool) -> Tuple[List[int], int]:
    offsets = []
    cur = 0
    for li in range(total_levels):
        offsets.append(cur)
        cur += 3 if (modality_index and li == 0) else codebook_vocab
    return offsets, cur


def build_docs_token_ids_from_cand_codes(
    cand_codes: np.ndarray,
    total_levels: int,
    codebook_vocab: int,
    modality_index: bool,
) -> torch.Tensor:
    offsets, _ = get_level_offsets(total_levels, codebook_vocab, modality_index)
    docs = np.empty_like(cand_codes, dtype=np.int64)
    for li in range(total_levels):
        docs[:, li] = cand_codes[:, li].astype(np.int64) + int(offsets[li])
    return torch.from_numpy(docs).long()


# =========================================================
# RQ loader + codebook embedding builder
# =========================================================
@torch.no_grad()
def load_rq(rq_yaml: str, rq_ckpt: str, device: torch.device, emb_dim: Optional[int] = None) -> RQModel:
    cfg = OmegaConf.load(rq_yaml)
    if emb_dim is not None:
        cfg.rq_config.emb_dim = int(emb_dim)
    cfg.rq_config.return_codes = True

    rq = RQModel(cfg).to(device)
    rq.eval()

    ckpt = torch.load(rq_ckpt, map_location="cpu")
    sd = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
    rq.load_state_dict(sd, strict=False)
    return rq


@torch.no_grad()
def rq_emb_to_codes(rq: RQModel, emb: torch.Tensor, img_mask: torch.Tensor, txt_mask: torch.Tensor) -> torch.Tensor:
    out = rq.infer(emb=emb, img_mask=img_mask, txt_mask=txt_mask, return_codes=True, normalize_out=False)
    return out["code"].long()


@torch.no_grad()
def build_E_simul_from_rq(rq: RQModel, total_levels: int, codebook_vocab: int, modality_index: bool) -> torch.Tensor:
    core = None
    for name in ["residual_rq", "rq", "quantizer"]:
        if hasattr(rq, name):
            core = getattr(rq, name)
            break
    if core is None:
        if hasattr(rq, "model") and hasattr(rq.model, "residual_rq"):
            core = rq.model.residual_rq
    if core is None or not hasattr(core, "layers"):
        raise RuntimeError("[E_simul] cannot locate residual_rq.layers")

    emb_list = []
    for li in range(total_levels):
        layer = core.layers[li]
        codes_l = layer._codebook.embed.squeeze(0).detach().cpu().float()
        if modality_index and li == 0:
            codes_l = codes_l[:3, :]
        emb_list.append(codes_l)

    E = torch.cat(emb_list, dim=0)
    E = F.normalize(E, dim=-1)
    return E


# =========================================================
# SQ-guided top-k selection (chunked)
# =========================================================
@torch.no_grad()
def topk_docs_by_sq_chunked(
    s_q_full: torch.Tensor,
    docs_token_ids_cpu: torch.Tensor,
    guide_topk: int,
    chunk_size: int = 50000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = s_q_full.device
    B = s_q_full.size(0)
    N = docs_token_ids_cpu.size(0)

    K = max(1, min(int(guide_topk), N))
    top_scores = torch.full((B, K), -1e9, device=device, dtype=torch.float32)
    top_indices = torch.full((B, K), -1, device=device, dtype=torch.long)

    for st in range(0, N, chunk_size):
        ed = min(N, st + chunk_size)
        docs_chunk = docs_token_ids_cpu[st:ed].to(device=device, non_blocking=True)
        scores = s_q_full[:, docs_chunk].sum(dim=-1)

        merged_scores = torch.cat([top_scores, scores], dim=1)
        chunk_idx = torch.arange(st, ed, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        merged_indices = torch.cat([top_indices, chunk_idx], dim=1)

        new_scores, new_pos = torch.topk(merged_scores, k=K, dim=1, largest=True)
        new_indices = merged_indices.gather(1, new_pos)
        top_scores, top_indices = new_scores, new_indices

    return top_scores, top_indices


# =========================================================
# Vectorized guidance logits processor
# =========================================================
class VectorizedGuidanceLogitsProcessor(LogitsProcessor):
    """
    Additive logits guidance using TopK token sequences and per-sequence scores.
    The guidance is activated only when current prefix matches any candidate prefix.
    """
    def __init__(self, topk_codes: torch.Tensor, topk_scores: torch.Tensor, guidance_scale: float, num_beams: int):
        super().__init__()
        self.topk_codes = topk_codes
        self.topk_scores = topk_scores
        self.guidance_scale = float(guidance_scale)
        self.num_beams = int(num_beams)
        self.B, self.K, self.L = topk_codes.shape

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.guidance_scale <= 0 or self.topk_codes is None:
            return scores

        device = scores.device
        dtype = scores.dtype
        BN = scores.size(0)
        assert BN == self.B * self.num_beams

        # Remove the decoder start token for prefix matching
        prefix = input_ids[:, 1:]
        curr_step = prefix.size(1)
        if curr_step >= self.L:
            return scores

        codes_exp = self.topk_codes.to(device=device).repeat_interleave(self.num_beams, dim=0)
        scores_exp = self.topk_scores.to(device=device, dtype=torch.float32).repeat_interleave(self.num_beams, dim=0)

        if curr_step > 0:
            match_mask = (codes_exp[:, :, :curr_step] == prefix.unsqueeze(1)).all(dim=-1)
        else:
            match_mask = torch.ones((BN, self.K), device=device, dtype=torch.bool)

        next_tokens = codes_exp[:, :, curr_step]
        valid_boosts = (scores_exp * self.guidance_scale).masked_fill(~match_mask, -1e9)

        if hasattr(torch.Tensor, "scatter_reduce_"):
            max_boosts = torch.full_like(scores, -1e9)
            max_boosts.scatter_reduce_(1, next_tokens, valid_boosts, reduce="amax", include_self=False)
            valid_mask = (max_boosts > -1e8)
            scores[valid_mask] += max_boosts[valid_mask].to(dtype)
        else:
            for k in range(self.K):
                tok = next_tokens[:, k]
                val = valid_boosts[:, k]
                m = val > -1e8
                if torch.any(m):
                    scores[m, tok[m]] += val[m].to(dtype)

        return scores


# =========================================================
# T5Generator
# =========================================================
class T5Generator(nn.Module):
    def __init__(self, t5_name: str, tokenizer: PreTrainedTokenizerFast, emb_dim: int, prefix_len: int = 1):
        super().__init__()
        self.tokenizer = tokenizer
        config = AutoConfig.from_pretrained(t5_name)
        config.num_layers = 0
        self.t5 = T5ForConditionalGeneration(config)
        self.t5.resize_token_embeddings(len(tokenizer))
        self.prefix_len = int(prefix_len)

        d_model = int(self.t5.config.d_model)
        self.emb_proj = nn.Linear(emb_dim, d_model * self.prefix_len)
        self.t5.config.pad_token_id = tokenizer.pad_token_id
        self.t5.config.eos_token_id = tokenizer.eos_token_id
        if self.t5.config.decoder_start_token_id is None:
            self.t5.config.decoder_start_token_id = tokenizer.pad_token_id

    @torch.no_grad()
    def generate_codes(
        self,
        emb: torch.Tensor,
        num_beams: int,
        max_new_tokens: int,
        use_fp16: bool,
        device: torch.device,
        trie_constraint: Optional[Trie] = None,
        logits_processor: Optional[LogitsProcessor] = None,
    ) -> torch.Tensor:
        self.eval()
        emb = emb.to(device, non_blocking=True)
        B = emb.size(0)
        d_model = int(self.t5.config.d_model)

        proj = self.emb_proj(emb)
        enc = proj.view(B, self.prefix_len, d_model)
        attn = torch.ones(B, self.prefix_len, device=device, dtype=torch.long)

        prefix_allowed_tokens_fn = None
        if trie_constraint is not None:
            def constrain_fn(batch_id, sent):
                sent_list = sent.tolist()
                if len(sent_list) > 0 and sent_list[0] == self.t5.config.decoder_start_token_id:
                    sent_list = sent_list[1:]
                allowed = trie_constraint.get(sent_list)
                return allowed if allowed else [self.tokenizer.eos_token_id]
            prefix_allowed_tokens_fn = constrain_fn

        lp_list = None
        if logits_processor is not None:
            lp_list = LogitsProcessorList([logits_processor])

        with autocast(enabled=use_fp16):
            out = self.t5.generate(
                inputs_embeds=enc,
                attention_mask=attn,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max_new_tokens,
                early_stopping=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                logits_processor=lp_list,
            )
        return out


def load_t5_model(t5_ckpt: str, t5_name: str, tokenizer: PreTrainedTokenizerFast, emb_dim: int, device: torch.device) -> T5Generator:
    ckpt = torch.load(t5_ckpt, map_location="cpu")
    sd = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}

    proj_w = sd.get("projector.weight", None)
    if proj_w is None:
        proj_w = sd.get("emb_proj.weight", None)

    d_model = AutoConfig.from_pretrained(t5_name).d_model
    prefix_len = 1
    if proj_w is not None:
        out_dim = proj_w.shape[0]
        if out_dim % d_model != 0:
            raise RuntimeError(f"projector out_dim={out_dim} not divisible by d_model={d_model}")
        prefix_len = out_dim // d_model

    model = T5Generator(t5_name, tokenizer, emb_dim, prefix_len=prefix_len).to(device)

    # Backward compatible key rename: projector.* -> emb_proj.*
    if "projector.weight" in sd and "emb_proj.weight" not in sd:
        sd["emb_proj.weight"] = sd.pop("projector.weight")
    if "projector.bias" in sd and "emb_proj.bias" not in sd:
        sd["emb_proj.bias"] = sd.pop("projector.bias")

    model.load_state_dict(sd, strict=False)
    model.eval()
    return model