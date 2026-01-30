"""
SenseExplorer: From Sense Discovery to Sense Induction via Simulated Self-Repair
=================================================================================

A lightweight, training-free framework for exploring word sense structure
in static embeddings using biologically-inspired self-repair.

Two operational modes:
  - discover_senses(): Unsupervised - true sense discovery from data alone
  - induce_senses():   Weakly supervised - anchor-guided sense induction

Basic Usage:
    >>> from sense_explorer import SenseExplorer
    >>> se = SenseExplorer.from_glove("glove.6B.100d.txt")
    
    # Unsupervised discovery (56% accuracy)
    >>> senses = se.discover_senses("bank", n_senses=2)
    
    # Knowledge-guided induction (88% accuracy)
    >>> senses = se.induce_senses("bank")

Author: Kow Kuroda (Kyorin University) & Claude (Anthropic)
License: MIT
Version: 0.4.0
"""

from .core import (
    SenseExplorer,
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

__version__ = "0.4.0"
__author__ = "Kow Kuroda & Claude"
__all__ = [
    'SenseExplorer',
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
