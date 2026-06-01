# -*- coding: utf-8 -*-
"""
model.py

GUR model:
- frozen RQ for online codes + q/p encodes
- projector: emb_dim -> num_prefix * d_model
- T5 decoder (encoder_layers=0) to predict code tokens
- Order-invariant module (Stage-1 trainable): order_invariant
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, T5ForConditionalGeneration, T5TokenizerFast

from .utils import IGNORE_INDEX, build_code_tokens


class T5ForGenerativeRetrieval(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        num_prefix: int,
        t5_name: str,
        tokenizer: T5TokenizerFast,
        codebook_vocab: int,
        total_levels: int,
        modality_index: bool,
        rq_for_init=None,
        rq_for_input=None,
    ):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.num_prefix = int(num_prefix)
        self.total_levels = int(total_levels)
        self.codebook_vocab = int(codebook_vocab)
        self.modality_index = bool(modality_index)

        cfg = AutoConfig.from_pretrained(t5_name)
        cfg.num_layers = 0
        self.t5 = T5ForConditionalGeneration(cfg)
        self.t5.config.decoder_start_token_id = tokenizer.pad_token_id
        self.t5.resize_token_embeddings(len(tokenizer))

        # Prefix projector: emb -> (num_prefix * d_model)
        self.projector = nn.Linear(self.emb_dim, self.t5.config.d_model * self.num_prefix)

        # RQ frozen
        self.rq = rq_for_input
        if self.rq is not None:
            self.rq.eval()
            for p in self.rq.parameters():
                p.requires_grad = False

        # Tokens
        self.level_indicators, self.code_tokens = build_code_tokens(
            codebook_vocab=self.codebook_vocab,
            total_levels=self.total_levels,
            modality_index=self.modality_index,
        )

        # -------------------------
        # Order-invariant module (Stage-1 trainable)
        # -------------------------
        self.order_invariant = nn.Sequential(
            nn.Linear(self.emb_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.emb_dim, bias=False),
        )

        self.level_offsets, self.V_T = self._get_level_offsets(
            total_levels=self.total_levels,
            codebook_vocab=self.codebook_vocab,
            modality_index=self.modality_index,
        )

        if self.rq is not None:
            E = self._build_E_simul_from_rq(self.rq)
            self.register_buffer("E_simul", E, persistent=False)
        else:
            self.E_simul = None

        self.ce_ignore = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

        # Optional: init code token embeddings from RQ
        if rq_for_init is not None:
            self._init_t5_code_embeddings_from_rq(
                rq=rq_for_init,
                tokenizer=tokenizer,
                code_tokens=self.code_tokens,
                emb_dim=self.emb_dim,
                scale=100.0,
            )

    def _get_level_offsets(self, total_levels: int, codebook_vocab: int, modality_index: bool):
        offsets = []
        cur = 0
        for l in range(total_levels):
            offsets.append(cur)
            cur += 3 if (modality_index and l == 0) else codebook_vocab
        return offsets, cur

    @torch.no_grad()
    def _build_E_simul_from_rq(self, rq):
        """
        Build a single lookup table E_simul by concatenating codebook vectors across levels.
        Shape: [V_T, emb_dim]
        """
        emb_list = []
        for l in range(self.total_levels):
            layer = rq.residual_rq.layers[l]
            codes = layer._codebook.embed.squeeze(0)  # [C, emb_dim]
            if self.modality_index and l == 0:
                codes = codes[:3, :]
            else:
                codes = codes[:self.codebook_vocab, :]
            emb_list.append(codes.detach().float().cpu())
        E = torch.cat(emb_list, dim=0)
        E = F.normalize(E, dim=-1)
        return E

    @torch.no_grad()
    def _init_t5_code_embeddings_from_rq(
        self,
        rq,
        tokenizer: T5TokenizerFast,
        code_tokens: List[str],
        emb_dim: int,
        scale: float = 100.0,
    ) -> None:
        """
        Initialize newly-added code-token embeddings from RQ codebook vectors.
        """
        d_model = self.t5.config.d_model

        vectors: List[torch.Tensor] = []
        layer0 = rq.residual_rq.layers[0]._codebook.embed  # [1, C, D]

        if getattr(rq, "modality_index", True):
            v0 = layer0[0, :3, :].detach().cpu()
            vectors.append(v0)
            start_li = 1
        else:
            vectors.append(layer0[0].detach().cpu())
            start_li = 1

        for li in range(start_li, rq.codebook_level):
            cb = rq.residual_rq.layers[li]._codebook.embed
            vectors.append(cb[0].detach().cpu())

        codebook_vec = torch.cat(vectors, dim=0)  # [V_T, emb_dim]
        assert codebook_vec.size(1) == emb_dim, (codebook_vec.size(), emb_dim)

        mapper = nn.Linear(emb_dim, d_model, bias=False)
        torch.manual_seed(12345)
        mapper.weight.normal_(mean=0.0, std=0.02)

        mapped = mapper(F.normalize(codebook_vec, dim=-1)).to(self.t5.shared.weight.dtype)

        token_ids = tokenizer.convert_tokens_to_ids(code_tokens)
        for idx, tid in enumerate(token_ids):
            if tid == tokenizer.unk_token_id:
                continue
            self.t5.shared.weight[tid].copy_(mapped[idx] * scale)
            self.t5.lm_head.weight[tid].copy_(mapped[idx] * scale)

        self.t5._tie_weights()

    # -------------------------
    # Forward: prefix embeds
    # -------------------------
    def forward(
        self,
        q_emb: torch.Tensor,
        p_emb: torch.Tensor,
        q_img_mask: torch.Tensor,
        q_txt_mask: torch.Tensor,
        p_img_mask: torch.Tensor,
        p_txt_mask: torch.Tensor,
        alpha: float = 2.0,
    ) -> torch.Tensor:
        """
        Build prefix embeddings from MixUp-style augmentation over RQ-encoded embeddings.
        """
        assert self.rq is not None, "rq_for_input must be provided"

        bs = q_emb.size(0)
        device = q_emb.device

        with torch.no_grad():
            q_out = self.rq.infer(q_emb, q_img_mask, q_txt_mask, return_codes=False, normalize_out=True)
            p_out = self.rq.infer(p_emb, p_img_mask, p_txt_mask, return_codes=False, normalize_out=True)

        q_encode = q_out["encode"]
        p_encode = p_out["encode"]

        if alpha <= 0:
            # 如果 alpha 小于等于 0，直接使用 query 的编码，关闭 MixUp
            aug = q_encode
        else:
            # 原始的 MixUp 逻辑
            s = torch.distributions.Beta(alpha, alpha).sample((bs, q_encode.size(1))).to(device)
            aug = torch.sqrt(s) * q_encode + torch.sqrt(1.0 - s) * p_encode
            aug = F.normalize(aug, dim=-1)

        pref = self.projector(aug).view(bs, self.num_prefix, -1)
        return pref

    # -------------------------
    # T5 losses / metrics
    # -------------------------
    def compute_loss_and_metrics(self, inputs_embeds: torch.Tensor, labels_ids: torch.Tensor):
        out = self.t5(inputs_embeds=inputs_embeds, labels=labels_ids, return_dict=True)
        loss = out.loss
        logits = out.logits
        pred = torch.argmax(logits, dim=-1)

        seq_ok = (pred == labels_ids).all(dim=1).float().mean()

        def level_acc(k: int) -> torch.Tensor:
            if pred.size(1) < k or labels_ids.size(1) < k:
                return torch.tensor(0.0, device=loss.device)
            return (pred[:, :k] == labels_ids[:, :k]).all(dim=1).float().mean()

        metrics = {
            "R_at_1": seq_ok,
            "Level1_acc": level_acc(1),
            "Level12_acc": level_acc(2),
            "Level123_acc": level_acc(3),
        }
        return loss, metrics, pred

    def sequence_logprob_score(
        self,
        inputs_embeds: torch.Tensor,
        labels_ids: torch.Tensor,
        ignore_index: int = IGNORE_INDEX,
        mask_pos0: bool = True,
    ) -> torch.Tensor:
        out = self.t5(inputs_embeds=inputs_embeds, labels=labels_ids, return_dict=True)
        logits = out.logits

        labels_g = labels_ids.clone()
        labels_g[labels_g == ignore_index] = 0

        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(dim=-1, index=labels_g.unsqueeze(-1)).squeeze(-1)

        mask = (labels_ids != ignore_index).float()
        if mask_pos0 and mask.size(1) > 0:
            mask[:, 0] = 0.0

        seq_sum = (token_lp * mask).sum(dim=-1)
        denom = mask.sum(dim=-1).clamp(min=1e-6)
        return seq_sum / denom

    # -------------------------
    # Order-invariant in-batch scoring (Stage-1)
    # -------------------------
    def compute_order_invariant_inbatch_scores(self, q_emb: torch.Tensor, p_codes: torch.Tensor):
        """
        q_emb: [B, emb_dim]
        p_codes: [B, L_total]
        returns:
          scores_in_batch: [B,B]
          h_q_full: [B, V_T]
        """
        assert self.E_simul is not None, "E_simul is None. Provide rq_for_input to build E_simul."

        device = q_emb.device
        B = q_emb.size(0)

        q_h = self.order_invariant(q_emb)
        q_h = F.normalize(q_h, dim=-1)

        E = self.E_simul.to(device=device, dtype=q_h.dtype)
        logits = (q_h @ E.t())
        h_q_full = torch.log1p(F.relu(logits))

        offsets = torch.tensor(self.level_offsets, device=device, dtype=torch.long)
        doc_logical_ids = p_codes + offsets.view(1, -1)

        rows = []
        for b in range(B):
            h = h_q_full[b]
            s = h[doc_logical_ids].sum(dim=-1)
            rows.append(s.unsqueeze(0))
        scores_in_batch = torch.cat(rows, dim=0)
        return scores_in_batch, h_q_full