from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn


DEFAULT_FEATURE_NAMES = [
    "temporal_score",
    "reid_distance",
    "kalman_gating_distance",
    "iou",
    "track_age",
]


class AssociationVetoMLP(nn.Module):
    def __init__(self, input_dim: int = 5, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(hidden_dim // 2, 4), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class AssociationVetoScorer:
    def __init__(
        self,
        model: AssociationVetoMLP,
        mean: Sequence[float],
        std: Sequence[float],
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.std[self.std < 1e-6] = 1.0

    @torch.no_grad()
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        x = (x - self.mean) / self.std
        tensor = torch.from_numpy(x).to(self.device)
        logits = self.model(tensor)
        return torch.sigmoid(logits).detach().cpu().numpy()


def load_association_veto_scorer(path: str | Path, device: str = "cpu") -> AssociationVetoScorer:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    feature_names = checkpoint.get("feature_names", DEFAULT_FEATURE_NAMES)
    model = AssociationVetoMLP(input_dim=len(feature_names), hidden_dim=int(checkpoint.get("hidden_dim", 16)))
    model.load_state_dict(checkpoint["model_state_dict"])
    return AssociationVetoScorer(
        model=model,
        mean=checkpoint["feature_mean"],
        std=checkpoint["feature_std"],
        device=device,
    )


def save_association_veto_checkpoint(
    path: str | Path,
    model: AssociationVetoMLP,
    feature_mean: Iterable[float],
    feature_std: Iterable[float],
    hidden_dim: int,
    feature_names: Sequence[str] = DEFAULT_FEATURE_NAMES,
    metadata: dict | None = None,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": np.asarray(feature_mean, dtype=np.float32),
            "feature_std": np.asarray(feature_std, dtype=np.float32),
            "feature_names": list(feature_names),
            "hidden_dim": int(hidden_dim),
            "metadata": metadata or {},
        },
        Path(path),
    )
