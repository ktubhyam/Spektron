"""
Vendored D-LinOSS (Damped Linear Oscillatory State-Space Models).

Based on TritonLinOSS (MIT License, KasraMazaheri/TritonLinOSS).
Import paths fixed for package-level installation compatibility.

References:
- LinOSS: arXiv 2410.03943 (Rusch & Rus, ICLR 2025 Oral)
- D-LinOSS: arXiv 2505.12171 (Boyer et al., 2025)
"""

from .layers import DampedLayer, IMLayer, IMEXLayer, LinOSSBlock, GLU

__all__ = [
    "DampedLayer",
    "IMLayer",
    "IMEXLayer",
    "LinOSSBlock",
    "GLU",
]
