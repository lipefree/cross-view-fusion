"""Fusion modules for two spatially aligned modalities."""

from .fusion import CrossViewFusion
from .multiscale import MultiScaleFusion
from .patch_fusion import PatchFusion

__all__ = ["CrossViewFusion", "MultiScaleFusion", "PatchFusion"]
__version__ = "0.1.0"
