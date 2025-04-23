"""JBU upsampler from FeatUp."""

import torch
import torchvision.transforms as T

from core.model.upsamplers import BaseUpsampler


class JBUFeatUpUpsampler(BaseUpsampler):
    """Learned JBU upsampler from FeatUp. Performs x16 upsampling."""

    def __init__(self, backbone_type: str = None, use_norm: bool = True) -> None:
        super().__init__()
        self.backbone_type = backbone_type
        self.use_norm = use_norm
        self.upsampler = self._load_upsampler()

    def forward(self, source: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        return self.upsampler(source, guidance)

    def _load_upsampler(self):
        assert self.backbone_type in [
            "maskclip",
            "dino16",
            "dinov2",
            "clip",
            "vit",
            "resnet50",
        ], f"Invalid model type: {self.backbone_type}"
        return torch.hub.load(
            "mhamilton723/FeatUp", self.backbone_type, use_norm=self.use_norm
        ).upsampler


### Run basic test
def test_inference():
    upsampler = JBUFeatUpUpsampler(backbone_type="dinov2").cuda()
    source = torch.rand(1, 384, 14, 14).cuda()  # low-res feature
    guidance = torch.rand(1, 3, 224, 224).cuda()  # high-res image
    output = upsampler(source, guidance)

    print("### Inference test ###")
    print(f"Input: {source.shape}")
    print(f"Guidance: {guidance.shape}")
    print(f"Output: {output.shape}")


if __name__ == "__main__":
    test_inference()
