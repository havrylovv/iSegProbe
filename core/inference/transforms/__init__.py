"""Transformations used during evaluation phase."""

from .base_transform import BaseTransform, SigmoidForPred
from .crops import Crops
from .flip import AddHorizontalFlip
from .limit_longest_side import LimitLongestSide
from .zoom_in import ZoomIn
