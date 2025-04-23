"""LoftUp Upsampler."""

import torch

from core.model.upsamplers import BaseUpsampler

from .loftup.loftup import load_loftup_checkpoint


class LoftUpUpsampler(BaseUpsampler):
    def __init__(
        self,
        upsampler_path: str,
        n_dim: int = 384,
        lr_pe_type: str = "sine",
        lr_size: int = 16,
    ) -> None:
        super(LoftUpUpsampler, self).__init__()
        self.upsampler = load_loftup_checkpoint(
            upsampler_path, n_dim, lr_pe_type, lr_size
        )

    def forward(self, source: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        return self.upsampler(source, guidance)
