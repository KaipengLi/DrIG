# model.py
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from vector_quantize_pytorch import VectorQuantize, ResidualVQ

from models.residual_quantization.utils import is_dist, get_rank, get_world_size


# =========================
# DDP gather utilities
# =========================
class GatherLayer(torch.autograd.Function):
    """
    Differentiable all_gather for tensors with the same shape on each rank.
    """
    @staticmethod
    def forward(ctx, x):
        world = get_world_size()
        outs = [torch.zeros_like(x) for _ in range(world)]
        dist.all_gather(outs, x.contiguous())
        return tuple(outs)

    @staticmethod
    def backward(ctx, *grads):
        grad = torch.stack(grads, dim=0)
        dist.all_reduce(grad)
        return grad[get_rank()]


@torch.no_grad()
def all_gather_no_grad(x: torch.Tensor) -> torch.Tensor:
    """
    Non-differentiable all-gather for labels or metadata.
    """
    if not (is_dist() and get_world_size() > 1):
        return x
    outs = [torch.zeros_like(x) for _ in range(get_world_size())]
    dist.all_gather(outs, x.contiguous())
    return torch.cat(outs, dim=0)


# =========================
# Label-based Multi-Positive Contrastive Loss
# =========================
class LabelContrastiveLoss(nn.Module):
    """
    For query->candidate:
      positives = { j | cand_did[j] == query_pos_did[i] }
    Use soft targets (uniform over positives).
    Also do candidate_pos -> query_all (reverse) using the positive candidate of each query.
    """

    def __init__(self, temperature: float = 0.01, gather: bool = True):
        super().__init__()
        self.temperature = float(temperature)
        self.gather = bool(gather)

    def _soft_ce(self, logits: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
        """
        logits:   [N, M]
        pos_mask: [N, M] bool, each row must have at least one True
        """
        logp = F.log_softmax(logits, dim=1)
        denom = pos_mask.sum(dim=1, keepdim=True).clamp_min(1)
        target = pos_mask.float() / denom
        return -(target * logp).sum(dim=1).mean()

    def forward(
        self,
        query_emb: torch.Tensor,        # [B, D]
        cand_emb: torch.Tensor,         # [P, D]
        pos_index: torch.Tensor,        # [B]
        query_pos_did: torch.Tensor,    # [B]
        cand_did: torch.Tensor,         # [P]
    ) -> torch.Tensor:
        B = query_emb.size(0)
        assert pos_index.numel() == B
        assert cand_emb.size(0) == cand_did.numel()

        qn = F.normalize(query_emb, dim=-1)
        pn = F.normalize(cand_emb, dim=-1)

        # -------- gather candidates (and labels) --------
        if self.gather and is_dist() and get_world_size() > 1:
            pn_g = torch.cat(GatherLayer.apply(pn), dim=0)           # [P_global, D] with grad
            cand_did_g = all_gather_no_grad(cand_did.long())         # [P_global] no grad

            # query->candidate
            logits_q = (qn @ pn_g.t()) / self.temperature            # [B, P_global]
            pos_mask_q = (query_pos_did.view(-1, 1).long() == cand_did_g.view(1, -1))
            loss_q = self._soft_ce(logits_q, pos_mask_q)

            # candidate_pos -> query_all
            qn_g = torch.cat(GatherLayer.apply(qn), dim=0)           # [B_global, D] with grad
            query_pos_did_g = all_gather_no_grad(query_pos_did.long())  # [B_global]

            cand_pos = pn[pos_index]                                 # [B, D]
            logits_p = (cand_pos @ qn_g.t()) / self.temperature       # [B, B_global]
            pos_mask_p = (query_pos_did.view(-1, 1).long() == query_pos_did_g.view(1, -1))
            loss_p = self._soft_ce(logits_p, pos_mask_p)

            return 0.5 * (loss_q + loss_p)

        # -------- non-DDP --------
        logits_q = (qn @ pn.t()) / self.temperature
        pos_mask_q = (query_pos_did.view(-1, 1).long() == cand_did.view(1, -1).long())
        loss_q = self._soft_ce(logits_q, pos_mask_q)

        cand_pos = pn[pos_index]
        logits_p = (cand_pos @ qn.t()) / self.temperature
        pos_mask_p = (query_pos_did.view(-1, 1).long() == query_pos_did.view(1, -1).long())
        loss_p = self._soft_ce(logits_p, pos_mask_p)

        return 0.5 * (loss_q + loss_p)


# =========================
# Encoder (fallback version; no mask gating)
# =========================
class ConditionEncoder(nn.Module):
    """
    Back-to-basic encoder:
      - Keep the original forward signature to minimize changes.
      - Ignore img/txt masks (no gating).
      - Simple FFN + residual.
    """

    def __init__(self, dim: int, expansion_factor: float = 2.0, drop_rate: float = 0.1):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop_rate),
        )

    def forward(self, x: torch.Tensor, img_mask: torch.Tensor = None, txt_mask: torch.Tensor = None) -> torch.Tensor:
        feat = self.net(x)
        return x + feat


# =========================
# Residual Quantizer Model
# =========================
class ResidualQuantizerModel(nn.Module):
    """
    Residual quantization model operating on embeddings:
      - Encodes query/candidate embeddings (optional)
      - Adds modality embedding bias (optional)
      - Applies ResidualVQ (optionally with modality index codebook)
      - Optimizes via:
          * multi-positive label contrastive loss
          * RQ commitment loss (from ResidualVQ)
          * MSE(q_quant, pos_quant)

    rq_mode controls encoder and codebook strategy:
      - "encoder_ema":         use_emb_encoder=True,  EMA codebook (not learnable)
      - "no_encoder_learnable": use_emb_encoder=False, learnable codebook (no EMA)
      - "pure_ema":            use_emb_encoder=False, EMA codebook (not learnable)
    """

    _VALID_RQ_MODES = ("encoder_ema", "no_encoder_learnable", "pure_ema")

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        dim = int(cfg.rq_config.emb_dim)
        self.dim = dim

        self.normalize_before_rq = bool(cfg.rq_config.normalize_before_rq)
        self.rq_w = float(cfg.rq_config.rq_loss_weight)
        self.mse_w = float(cfg.rq_config.mse_loss_weight)

        # ---- rq_mode dispatch ----
        rq_mode = str(getattr(cfg.rq_config, "rq_mode", "encoder_ema"))
        if rq_mode not in self._VALID_RQ_MODES:
            raise ValueError(f"Unknown rq_mode='{rq_mode}', must be one of {self._VALID_RQ_MODES}")
        self.rq_mode = rq_mode

        self.use_emb_encoder = (rq_mode == "encoder_ema")
        use_learnable = (rq_mode == "no_encoder_learnable")
        use_ema = not use_learnable

        # Embedding encoder

        class _PassThrough(nn.Module):
            def forward(self, x, *args, **kwargs):
                return x

        self.emb_encoder = ConditionEncoder(dim, expansion_factor=2.0) if self.use_emb_encoder else _PassThrough()

        # Modality conditioning (embedding bias)
        self.use_modality_cond = bool(getattr(cfg.rq_config, "use_modality_cond", True))
        self.modality_scale = float(getattr(cfg.rq_config, "modality_scale", 1.0))
        if self.use_modality_cond:
            self.modality_emb = nn.Embedding(3, dim)
            nn.init.normal_(self.modality_emb.weight, std=0.02)

        self.critic = nn.MSELoss(reduction="mean")
        self.use_cl_loss = bool(getattr(cfg.rq_config, "use_cl_loss", True))
        self.contra = LabelContrastiveLoss(
            temperature=float(getattr(cfg.rq_config, "temperature", 0.01)),
            gather=True,
        )

        codebook_vocab = int(cfg.rq_config.codebook_vocab)
        codebook_level = int(cfg.rq_config.codebook_level)
        modality_index = bool(cfg.rq_config.modality_index)

        self.modality_index = modality_index
        self.codebook_level = codebook_level + (1 if modality_index else 0)

        # Optional first codebook for modality token
        if modality_index:
            self.vq0 = VectorQuantize(
                dim=dim,
                codebook_dim=dim,
                codebook_size=3,
                kmeans_init=True,
                kmeans_iters=1000,
                learnable_codebook=use_learnable,
                ema_update=use_ema,
                threshold_ema_dead_code=0,
                decay=0.9,
            )

        self.residual_rq = ResidualVQ(
            dim=dim,
            codebook_dim=dim,
            num_quantizers=self.codebook_level,
            codebook_size=codebook_vocab,
            kmeans_init=True,
            kmeans_iters=1000,
            learnable_codebook=use_learnable,
            ema_update=use_ema,
            threshold_ema_dead_code=2,
            decay=0.9,
        )

        if modality_index:
            self.residual_rq.layers[0] = self.vq0

    @staticmethod
    def mid_from_masks(img_mask: torch.Tensor, txt_mask: torch.Tensor) -> torch.Tensor:
        """
        Convert (img_mask, txt_mask) into modality id:
          0: image-only
          1: text-only
          2: image+text
        """
        if img_mask.dim() == 2:
            img_mask = img_mask.squeeze(-1)
        if txt_mask.dim() == 2:
            txt_mask = txt_mask.squeeze(-1)

        has_img = img_mask > 0
        has_txt = txt_mask > 0
        mid = torch.where(
            has_img & has_txt,
            torch.full_like(img_mask, 2),
            torch.where(has_txt, torch.full_like(img_mask, 1), torch.full_like(img_mask, 0)),
        ).long()
        return mid

    def forward(self, **kwargs) -> dict:
        """
        New-only argument names (mm legacy removed):
          query_emb, cand_emb, pos_index, query_pos_did, cand_did,
          query_img_mask, query_txt_mask, cand_img_mask, cand_txt_mask
        """
        query_emb = kwargs.get("query_emb", None)
        cand_emb = kwargs.get("cand_emb", None)
        pos_index = kwargs.get("pos_index", None)
        query_pos_did = kwargs.get("query_pos_did", None)
        cand_did = kwargs.get("cand_did", None)

        query_img_mask = kwargs.get("query_img_mask", None)
        query_txt_mask = kwargs.get("query_txt_mask", None)
        cand_img_mask = kwargs.get("cand_img_mask", None)
        cand_txt_mask = kwargs.get("cand_txt_mask", None)

        if query_emb is None or cand_emb is None:
            raise KeyError("Missing required inputs: query_emb and cand_emb.")
        if pos_index is None or query_pos_did is None or cand_did is None:
            raise KeyError("Missing required inputs: pos_index, query_pos_did, cand_did.")
        if query_img_mask is None or query_txt_mask is None or cand_img_mask is None or cand_txt_mask is None:
            raise KeyError("Missing required masks: query_img_mask/query_txt_mask/cand_img_mask/cand_txt_mask.")

        B = query_emb.size(0)

        # Encode (encoder ignores masks, but keep signature unchanged)
        q = self.emb_encoder(query_emb, query_img_mask, query_txt_mask)
        p_all = self.emb_encoder(cand_emb, cand_img_mask, cand_txt_mask)

        q_mid = self.mid_from_masks(query_img_mask, query_txt_mask)
        p_mid = self.mid_from_masks(cand_img_mask, cand_txt_mask)

        # Add modality embedding bias
        if self.use_modality_cond:
            q = q + self.modality_scale * self.modality_emb(q_mid)
            p_all = p_all + self.modality_scale * self.modality_emb(p_mid)

        if self.normalize_before_rq:
            q = F.normalize(q, dim=-1)
            p_all = F.normalize(p_all, dim=-1)

        # Quantization on concatenated vectors
        encode = torch.cat([q, p_all], dim=0)
        quant, codes, rq_loss = self.residual_rq(encode.unsqueeze(0), rand_quantize_dropout_fixed_seed=2023)
        quant = quant.squeeze(0)
        codes = codes.squeeze(0)

        q_quant = F.normalize(quant[:B], dim=-1)
        p_quant = F.normalize(quant[B:], dim=-1)
        p_pos_quant = p_quant[pos_index]

        rq_loss = self.rq_w * rq_loss.mean()
        mse_loss = self.mse_w * self.critic(q_quant, p_pos_quant)


        # Multi-positive contrastive loss based on did equality
        if self.use_cl_loss:
            cl_loss = self.contra(q, p_all, pos_index, query_pos_did, cand_did)
        else:
            cl_loss = torch.zeros((), device=q.device, dtype=q.dtype)

        loss = cl_loss + rq_loss + mse_loss

        # Accuracy (compare did, not local index)
        qn = F.normalize(q, dim=-1)
        pn_org = F.normalize(p_all, dim=-1)
        pn_dec = p_quant

        if is_dist() and get_world_size() > 1:
            pn_org_g = torch.cat(GatherLayer.apply(pn_org), dim=0)
            pn_dec_g = torch.cat(GatherLayer.apply(pn_dec), dim=0)
            cand_did_g = all_gather_no_grad(cand_did.long())

            logits_org = qn @ pn_org_g.t()
            logits_dec = qn @ pn_dec_g.t()

            pred_org_idx = torch.argmax(logits_org, dim=1)
            pred_dec_idx = torch.argmax(logits_dec, dim=1)

            pred_org_did = cand_did_g[pred_org_idx]
            pred_dec_did = cand_did_g[pred_dec_idx]

            acc_org = (pred_org_did == query_pos_did.long()).float().mean()
            acc = (pred_dec_did == query_pos_did.long()).float().mean()
        else:
            logits_org = qn @ pn_org.t()
            logits_dec = qn @ pn_dec.t()

            pred_org_idx = torch.argmax(logits_org, dim=1)
            pred_dec_idx = torch.argmax(logits_dec, dim=1)

            pred_org_did = cand_did.long()[pred_org_idx]
            pred_dec_did = cand_did.long()[pred_dec_idx]

            acc_org = (pred_org_did == query_pos_did.long()).float().mean()
            acc = (pred_dec_did == query_pos_did.long()).float().mean()

        out = {
            "loss": loss,
            "cl_loss": cl_loss.detach(),
            "rq_loss": rq_loss.detach(),
            "mse_loss": mse_loss.detach(),
            "acc": acc.detach(),
            "acc_org": acc_org.detach(),
        }

        # Optionally return full RQ codes for debug printing
        if bool(getattr(self.cfg.rq_config, "return_codes", False)):
            out["Is_q_full"] = codes[:B].detach().long()
            out["Is_p_full"] = codes[B:].detach().long()

        # Modality clustering debug only meaningful if modality_index=True
        if self.modality_index:
            out.update(
                {
                    "q_mid": q_mid.detach().long(),
                    "p_mid_all": p_mid.detach().long(),
                    "Is0_q": codes[:B, 0].detach().long(),
                    "Is0_p_all": codes[B:, 0].detach().long(),
                }
            )

        return out

    @torch.no_grad()
    def infer(
        self,
        emb: torch.Tensor,
        img_mask: torch.Tensor,
        txt_mask: torch.Tensor,
        return_codes: bool = True,
        normalize_out: bool = True,
    ) -> dict:
        """
        Inference API:
          emb: input embedding matrix [N, D]
        """
        self.eval()

        # Encode
        x = self.emb_encoder(emb, img_mask, txt_mask)
        mid = self.mid_from_masks(img_mask, txt_mask)

        # Add modality bias
        if self.use_modality_cond:
            x = x + self.modality_scale * self.modality_emb(mid)

        if self.normalize_before_rq:
            x = F.normalize(x, dim=-1)

        encode = x

        # Quantize
        quant, codes, _ = self.residual_rq(x.unsqueeze(0), rand_quantize_dropout_fixed_seed=2023)
        quant = quant.squeeze(0)
        codes = codes.squeeze(0).long()

        if normalize_out:
            encode = F.normalize(encode, dim=-1)
            quant = F.normalize(quant, dim=-1)

        return {
            "encode": encode,
            "quant": quant,
            "code": codes if return_codes else None,
        }