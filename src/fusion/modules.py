"""Modules de fusion (PyTorch) — prototype de la Partie B.

Notations : M modalités, x_m ∈ ℝ^{d_m} la représentation figée de la modalité m (cache
`extract_features`), μ_m ∈ {0,1} son masque de présence, d la dimension commune.

  Projection      p_m = LN(W_m^p x_m)                         ∈ ℝ^d     (par modalité)
  Absence         p_m ← μ_m p_m + (1 − μ_m) a_m                a_m : jeton « absent » appris
  GMU à M voies   h_m = tanh(W_m p_m)                          (Arevalo et al., 2017, étendu à M > 2)
                  s = W_z [p_1 ; … ; p_M] ∈ ℝ^{M×d}
                  s_m ← s_m − ∞·(1 − μ_m)                      (porte masquée : un absent ne vote pas)
                  z = softmax_m(s)                             (softmax sur les modalités, par dimension)
                  h = Σ_m z_m ⊙ h_m
  Sortie          e = LN(W_o h + b_o)                          ∈ ℝ^d     (représentation de l'item)

Si toutes les modalités sont absentes, la porte retombe sur une moyenne uniforme des jetons absents.
z est renvoyée : sa moyenne par modalité mesure la contribution de chacune (interprétabilité).
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class Projector(nn.Module):
    """Encodeur léger entraînable au-dessus d'un encodeur pré-entraîné figé."""

    def __init__(self, d_in: int, d: int, dropout: float, hidden: int | None = None) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(d_in)]
        if hidden:
            layers += [nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d)]
        else:
            layers += [nn.Dropout(dropout), nn.Linear(d_in, d)]
        layers.append(nn.LayerNorm(d))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def apply_absent(p: Tensor, mask: Tensor, absent: Tensor) -> Tensor:
    """p : (B, M, d), mask : (B, M), absent : (M, d)."""
    m = mask.unsqueeze(-1).to(p.dtype)
    return m * p + (1 - m) * absent.unsqueeze(0)


class GatedMultimodalFusion(nn.Module):
    """GMU généralisé à M modalités avec porte masquée."""

    def __init__(self, n_modalities: int, d: int) -> None:
        super().__init__()
        self.M, self.d = n_modalities, d
        self.h = nn.ModuleList([nn.Linear(d, d) for _ in range(n_modalities)])
        self.gate = nn.Linear(n_modalities * d, n_modalities * d)
        self.out = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))

    def forward(self, p: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        B = p.shape[0]
        h = torch.stack([torch.tanh(lin(p[:, m])) for m, lin in enumerate(self.h)], dim=1)   # (B, M, d)
        s = self.gate(p.reshape(B, -1)).view(B, self.M, self.d)
        all_absent = mask.sum(dim=1, keepdim=True) == 0                                     # (B, 1)
        keep = (mask.bool() | all_absent).unsqueeze(-1)                                      # (B, M, 1)
        z = torch.softmax(s.masked_fill(~keep, float("-inf")), dim=1)                        # (B, M, d)
        return self.out((z * h).sum(dim=1)), z


class ConcatFusion(nn.Module):
    """Baseline : concaténation des projections puis MLP (fusion intermédiaire la plus simple)."""

    def __init__(self, n_modalities: int, d: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_modalities * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, d),
                                 nn.LayerNorm(d))

    def forward(self, p: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        z = mask.unsqueeze(-1).expand(-1, -1, p.shape[-1]).to(p.dtype)
        z = z / z.sum(dim=1, keepdim=True).clamp_min(1)
        return self.net(p.reshape(p.shape[0], -1)), z


class CrossAttentionFusion(nn.Module):
    """Variante d'ablation : chaque modalité est un jeton ; auto-attention masquée + pooling."""

    def __init__(self, n_modalities: int, d: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.type_emb = nn.Parameter(torch.zeros(n_modalities, d))
        self.attn = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=2 * d, dropout=dropout, batch_first=True,
                                               norm_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, p: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        all_absent = mask.sum(dim=1, keepdim=True) == 0
        pad = ~(mask.bool() | all_absent)
        t = self.attn(p + self.type_emb, src_key_padding_mask=pad)
        w = (~pad).to(p.dtype).unsqueeze(-1)
        z = (w / w.sum(dim=1, keepdim=True)).expand_as(t)
        return self.norm((t * z).sum(dim=1)), z


FUSIONS = {"gmu": GatedMultimodalFusion, "concat": ConcatFusion, "attention": CrossAttentionFusion}
