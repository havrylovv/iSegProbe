from .base_head import BaseClassifierHead
from .conv_heads import ConvSegHead, SimpleClassifierHead, SimpleConvSegHead

__all__ = [
    "SimpleClassifierHead",
    "SimpleConvSegHead",
    "ConvSegHead",
]

# used to load heads from config
HEAD_REGISTRY = {
    "linear": SimpleClassifierHead,
    "simple_conv": SimpleConvSegHead,
    "convhead": ConvSegHead,
}
