"""Modèle de recommandation multimodal à deux tours (prototype de la Partie B, entraîné en Tâche 4).

Tour item  : modalités {texte, image, méta, identifiant} → projections → absences → fusion → e_i ∈ ℝ^d
Tour user  : embedding u ∈ ℝ^d + biais b_u
Score      : ŷ_ui = μ + b_u + b_i + ⟨u, e_i⟩,  b_i = w_b·e_i  (biais item calculé depuis le contenu)

L'identifiant d'item est traité comme une 4ᵉ modalité « collaborative » : présent pour les films vus à
l'entraînement, absent pour les films froids. Pendant l'entraînement il est masqué avec probabilité
`id_dropout` (esprit DropoutNet) : le modèle apprend à scorer à partir du seul contenu, ce qui est
exactement la situation du test `cold_item_test`. Les modalités de contenu subissent un modality dropout
(`modality_dropout`) pour la robustesse aux absences et contre la domination d'une modalité.

Pertes :
  L = L_rating + λ_uni · Σ_m L_rating(ŷ^{(m)}) + λ_con · L_InfoNCE(texte, image)
  L_rating : MSE sur la note ; ŷ^{(m)} : score de la modalité m seule (têtes unimodales, contre le déséquilibre)
  L_InfoNCE : alignement contrastif symétrique texte ↔ image du même film (température τ), type CLIP.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .modules import FUSIONS, Projector, apply_absent

MODALITIES = ("text", "image", "meta", "id")


@dataclass
class FusionConfig:
    d_text: int
    d_image: int
    d_meta: int
    n_items: int
    n_users: int
    d_model: int = 256
    meta_hidden: int = 128
    dropout: float = 0.2
    modality_dropout: float = 0.3
    id_dropout: float = 0.5
    fusion: str = "gmu"
    n_heads: int = 4
    lambda_unimodal: float = 0.3
    lambda_contrastive: float = 0.1
    temperature: float = 0.07
    global_mean: float = 3.5

    @classmethod
    def from_params(cls, fp: dict, dims: dict, n_items: int, n_users: int, **kw) -> "FusionConfig":
        return cls(d_text=dims["text"], d_image=dims["image"], d_meta=dims["meta"], n_items=n_items, n_users=n_users,
                   d_model=fp["d_model"], meta_hidden=fp["meta_hidden"], dropout=fp["dropout"],
                   modality_dropout=fp["modality_dropout"], id_dropout=fp["id_dropout"], n_heads=fp["n_heads"],
                   lambda_unimodal=fp["lambda_unimodal"], lambda_contrastive=fp["lambda_contrastive"],
                   temperature=fp["temperature"], **kw)


class MultimodalRecommender(nn.Module):
    def __init__(self, cfg: FusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.proj = nn.ModuleDict({
            "text": Projector(cfg.d_text, d, cfg.dropout),
            "image": Projector(cfg.d_image, d, cfg.dropout),
            "meta": Projector(cfg.d_meta, d, cfg.dropout, hidden=cfg.meta_hidden),
        })
        self.item_id = nn.Embedding(cfg.n_items, d)        # index 0 réservé : film inconnu / froid
        nn.init.normal_(self.item_id.weight, std=0.02)
        self.absent = nn.Parameter(torch.randn(len(MODALITIES), d) * 0.02)
        kw = {"n_heads": cfg.n_heads} if cfg.fusion == "attention" else {}
        self.fusion = FUSIONS[cfg.fusion](len(MODALITIES), d, **kw)
        self.item_bias = nn.Linear(d, 1)
        self.user = nn.Embedding(cfg.n_users, d)
        self.user_bias = nn.Embedding(cfg.n_users, 1)
        nn.init.normal_(self.user.weight, std=0.02)
        nn.init.zeros_(self.user_bias.weight)
        self.mu = nn.Parameter(torch.tensor(cfg.global_mean))

    # --- tour item ---------------------------------------------------------------------------------
    def project(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        """Renvoie p (B, 4, d) avant absences et le masque effectif (B, 4)."""
        p = torch.stack([self.proj["text"](batch["text"]), self.proj["image"](batch["image"]),
                         self.proj["meta"](batch["meta"]),
                         self.item_id(batch["item_idx"])], dim=1)
        mask = torch.cat([batch["mask"].float(), (batch["item_idx"] > 0).float().unsqueeze(1)], dim=1)
        if self.training:
            keep = torch.ones_like(mask)
            keep[:, :3] = (torch.rand_like(mask[:, :3]) > self.cfg.modality_dropout).float()
            keep[:, 3] = (torch.rand_like(mask[:, 3]) > self.cfg.id_dropout).float()
            mask = mask * keep
        return p, mask

    def item_tower(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        p, mask = self.project(batch)
        p_abs = apply_absent(p, mask, self.absent)
        e, z = self.fusion(p_abs, mask)
        return {"e": e, "z": z, "p": p, "mask": mask}

    # --- passe avant complète --------------------------------------------------------------------------
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        it = self.item_tower(batch)
        u = self.user(batch["user_idx"])
        bu = self.user_bias(batch["user_idx"]).squeeze(-1)
        score = self.mu + bu + self.item_bias(it["e"]).squeeze(-1) + (u * it["e"]).sum(-1)
        uni = self.mu + bu.unsqueeze(1) + (u.unsqueeze(1) * it["p"]).sum(-1)          # (B, 4) têtes unimodales
        return {"score": score, "unimodal": uni, **it}

    def loss(self, out: dict[str, Tensor], rating: Tensor) -> dict[str, Tensor]:
        c = self.cfg
        l_main = F.mse_loss(out["score"], rating)
        m = out["mask"]
        err = (out["unimodal"] - rating.unsqueeze(1)) ** 2
        l_uni = (err * m).sum() / m.sum().clamp_min(1)
        both = (m[:, 0] > 0) & (m[:, 1] > 0)
        l_con = info_nce(out["p"][both, 0], out["p"][both, 1], c.temperature) if both.sum() > 1 else rating.new_zeros(())
        total = l_main + c.lambda_unimodal * l_uni + c.lambda_contrastive * l_con
        return {"loss": total, "rating_mse": l_main.detach(), "unimodal_mse": l_uni.detach(),
                "contrastive": l_con.detach()}

    @torch.no_grad()
    def modality_contribution(self, out: dict[str, Tensor]) -> dict[str, float]:
        """Moyenne des poids de porte par modalité (somme = 1) : contribution de chaque modalité."""
        w = out["z"].mean(dim=(0, 2))
        return {m: float(x) for m, x in zip(MODALITIES, w)}


def info_nce(a: Tensor, b: Tensor, temperature: float) -> Tensor:
    """Perte contrastive symétrique (CLIP) entre deux vues appariées ligne à ligne."""
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    logits = a @ b.t() / temperature
    target = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))
