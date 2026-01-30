"""
SenseInduction: Sense Induction via Simulated Self-Repair
==========================================================

A lightweight, training-free method for inducing word senses
in static embeddings using biologically-inspired self-repair.

Basic Usage:
    >>> from sense_induction import SenseInductor
    >>> si = SenseInductor.from_glove("glove.6B.100d.txt")
    >>> senses = si.induce_senses("bank")
    >>> print(senses)
    {'financial': array([...]), 'river': array([...])}

Features:
    - Zero training required
    - Automatic anchor extraction (FrameNet + WordNet)
    - Noise level as granularity control
    - Stability-based sense number selection
    - 88% accuracy with frame-based anchors

Author: Kow Kuroda (Kyorin University) & Claude (Anthropic)
License: MIT
Version: 0.3.0
"""

from .core import (
    SenseInductor,
    COMMON_POLYSEMOUS,
    load_common_polysemous
)

from .anchor_extractor import (
    HybridAnchorExtractor,
    extract_anchors,
    get_manual_anchors,
    get_frame_anchors,
    list_supported_words,
    MANUAL_ANCHORS,
    FRAME_ANCHORS
)

__version__ = "0.3.0"
__author__ = "Kow Kuroda & Claude"
__all__ = [
    'SenseInductor',
    'HybridAnchorExtractor',
    'extract_anchors',
    'get_manual_anchors',
    'get_frame_anchors',
    'list_supported_words',
    'load_common_polysemous',
    'COMMON_POLYSEMOUS',
    'MANUAL_ANCHORS',
    'FRAME_ANCHORS'
]
