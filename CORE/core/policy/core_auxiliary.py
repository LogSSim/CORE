from typing import Dict

import torch
import torch.nn.functional as F

from core.model.vision.terminal_point_encoder import TerminalPointEncoder


class TerminalAuxMixin:
    """Auxiliary terminal representation losses shared by CORE variants."""

    def _init_terminal_aux(
        self,
        pointcloud_encoder_cfg=None,
        use_pc_color=False,
        terminal_window=8,
        neg_count=4,
        lambda_nce=1.0,
        lambda_ttg=0.3,
        lambda_term=0.1,
        nce_tau=0.1,
        term_feat_dim=128,
        term_proj_dim=128,
        aux_loss_mode="all",
    ):
        self.term_module = TerminalPointEncoder(
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            feat_dim=term_feat_dim,
            proj_dim=term_proj_dim,
            use_pc_color=use_pc_color,
        )
        self.lambda_nce = float(lambda_nce)
        self.lambda_ttg = float(lambda_ttg)
        self.lambda_term = float(lambda_term)
        self.aux_loss_mode = str(aux_loss_mode)
        if self.aux_loss_mode not in {"all", "nce_ttg", "nce_term"}:
            raise ValueError(
                "aux_loss_mode must be one of all, nce_ttg, nce_term, "
                f"got {self.aux_loss_mode}"
            )
        self.nce_tau = float(nce_tau)
        self.terminal_window = int(terminal_window)
        self.neg_count = int(neg_count)
        if self.nce_tau <= 0:
            raise ValueError(f"nce_tau must be positive, got {self.nce_tau}")

    def _normalize_terminal_point_cloud(self, pc: torch.Tensor) -> torch.Tensor:
        npc = self.normalizer["point_cloud"].normalize(pc)
        if not self.use_pc_color:
            npc = npc[..., :3]
        return npc

    def _nce_from_anchor(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        # anchor/positive: [B, D], negatives: [B, M, D]
        pos = (anchor * positive).sum(dim=-1, keepdim=True) / self.nce_tau
        neg = torch.einsum("bd,bmd->bm", anchor, negatives) / self.nce_tau
        logits = torch.cat([pos, neg], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def _terminal_aux_loss(self, batch: Dict[str, torch.Tensor]):
        required_keys = (
            "term_anchor_point_cloud",
            "term_pos_point_cloud",
            "neg_point_clouds",
            "repr_point_cloud",
            "ttg_target",
            "term_label",
        )
        missing = [key for key in required_keys if key not in batch]
        if missing:
            raise KeyError(
                "Terminal auxiliary policy needs MetaworldTerminalDataset fields, "
                f"missing: {missing}"
            )

        n_term_a = self._normalize_terminal_point_cloud(batch["term_anchor_point_cloud"])
        n_term_p = self._normalize_terminal_point_cloud(batch["term_pos_point_cloud"])
        n_neg = self._normalize_terminal_point_cloud(batch["neg_point_clouds"])
        n_repr = self._normalize_terminal_point_cloud(batch["repr_point_cloud"])

        _, z_a = self.term_module.encode_proj(n_term_a)
        _, z_p = self.term_module.encode_proj(n_term_p)

        # [B, M, N, C] -> [B * M, N, C] for the shared terminal encoder.
        batch_size, neg_count = n_neg.shape[:2]
        neg_flat = n_neg.reshape(batch_size * neg_count, *n_neg.shape[2:])
        _, z_n = self.term_module.encode_proj(neg_flat)
        z_n = z_n.reshape(batch_size, neg_count, -1)

        loss_nce_a = self._nce_from_anchor(z_a, z_p, z_n)
        loss_nce_p = self._nce_from_anchor(z_p, z_a, z_n)
        loss_nce = 0.5 * (loss_nce_a + loss_nce_p)

        feat_repr = self.term_module.encode_feat(n_repr)
        ttg_pred = torch.sigmoid(self.term_module.ttg_head(feat_repr)).squeeze(-1)
        term_logit = self.term_module.term_head(feat_repr).squeeze(-1)

        ttg_target = batch["ttg_target"].to(device=ttg_pred.device, dtype=ttg_pred.dtype).reshape(-1)
        term_label = batch["term_label"].to(device=term_logit.device, dtype=term_logit.dtype).reshape(-1)

        loss_ttg = F.smooth_l1_loss(ttg_pred, ttg_target)
        loss_term = F.binary_cross_entropy_with_logits(term_logit, term_label)
        aux_loss = torch.zeros((), device=loss_nce.device, dtype=loss_nce.dtype)
        if self.aux_loss_mode in {"all", "nce_ttg", "nce_term"}:
            aux_loss = aux_loss + self.lambda_nce * loss_nce
        if self.aux_loss_mode in {"all", "nce_ttg"}:
            aux_loss = aux_loss + self.lambda_ttg * loss_ttg
        if self.aux_loss_mode in {"all", "nce_term"}:
            aux_loss = aux_loss + self.lambda_term * loss_term

        with torch.no_grad():
            term_prob = torch.sigmoid(term_logit)
            term_acc = ((term_prob > 0.5).to(term_label.dtype) == term_label).float().mean()
            ttg_mae = (ttg_pred - ttg_target).abs().mean()

        metrics = {
            "loss_nce": loss_nce.item(),
            "loss_ttg": loss_ttg.item(),
            "loss_term": loss_term.item(),
            "ttg_mae": ttg_mae.item(),
            "term_acc": term_acc.item(),
            "z_norm": z_a.norm(dim=-1).mean().item(),
        }
        return aux_loss, metrics
