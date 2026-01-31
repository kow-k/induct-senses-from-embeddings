"""
SenseExplorer: From Sense Discovery to Sense Induction via Simulated Self-Repair
=================================================================================

A lightweight, training-free framework for exploring word sense structure
in static embeddings using biologically-inspired self-repair.

Key insight: Meanings behave like superposed waves. Spectral clustering
(eigenvector decomposition) outperforms statistical clustering (BIC),
confirming the wave-like nature of semantic representations.

Three capabilities:
  - discover_senses(): Unsupervised - spectral clustering (90% at 50d)
  - induce_senses():   Weakly supervised - anchor-guided (88% accuracy)
  - find_polarity():   Supervised - polarity classification (97% accuracy)

Basic Usage:
    >>> from sense_explorer import SenseExplorer
    >>> se = SenseExplorer.from_glove("glove.6B.100d.txt")
    
    # Unsupervised discovery (spectral, 90% at 50d)
    >>> senses = se.discover_senses_auto("bank")
    
    # Knowledge-guided induction (88% accuracy)
    >>> senses = se.induce_senses("bank")
    
    # Polarity classification (97% accuracy)
    >>> polarity = se.get_polarity("excellent")

Author: Kow Kuroda (Kyorin University) & Claude (Anthropic)
License: MIT
Version: 0.7.0
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

from .polarity import (
    PolarityFinder,
    DEFAULT_POLARITY_SEEDS,
    DOMAIN_POLARITY_SEEDS,
    classify_polarity
)

from .spectral import (
    spectral_clustering,
    discover_anchors_spectral,
    discover_anchors_spectral_fixed_k,
    find_k_by_eigengap
)

__version__ = "0.7.0"
__author__ = "Kow Kuroda & Claude"
__all__ = [
    # Core
    'SenseExplorer',
    # Anchors
    'HybridAnchorExtractor',
    'extract_anchors',
    'get_manual_anchors',
    'get_frame_anchors',
    'list_supported_words',
    'load_common_polysemous',
    'COMMON_POLYSEMOUS',
    'MANUAL_ANCHORS',
    'FRAME_ANCHORS',
    # Polarity
    'PolarityFinder',
    'DEFAULT_POLARITY_SEEDS',
    'DOMAIN_POLARITY_SEEDS',
    'classify_polarity',
    # Spectral
    'spectral_clustering',
    'discover_anchors_spectral',
    'discover_anchors_spectral_fixed_k',
    'find_k_by_eigengap'
]
