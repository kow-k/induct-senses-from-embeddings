#!/usr/bin/env python3
"""
SenseExplorer: From Sense Discovery to Sense Induction via Simulated Self-Repair
=================================================================================

A lightweight, training-free framework for exploring word sense structure
in static embeddings (GloVe, Word2Vec, FastText, etc.).

Inspired by DNA self-repair mechanisms: noise + self-organization
reveals latent sense structure encoded as stable attractors.

Two operational modes:
  - discover_senses(): UNSUPERVISED - true sense discovery from data alone (56%)
  - induce_senses():   WEAKLY SUPERVISED - anchor-guided induction (88%)

The key insight: Senses can be DISCOVERED (emerging from distributional
structure) or INDUCED (guided toward anchor-defined targets). Both use
the same self-repair mechanism, but differ in supervision level.

Basic Usage:
  >>> from sense_explorer import SenseExplorer
  >>> se = SenseExplorer.from_glove("glove.6B.100d.txt")
  
  # Unsupervised discovery
  >>> senses = se.discover_senses("bank", n_senses=2)
  >>> print(senses.keys())  # dict_keys(['sense_0', 'sense_1'])
  
  # Weakly supervised induction
  >>> senses = se.induce_senses("bank")
  >>> print(senses.keys())  # dict_keys(['financial', 'river'])

Author: Kow Kuroda (Kyorin University) & Claude (Anthropic)
License: MIT
Repository: https://github.com/kow-k/sense-explorer
"""

__version__ = "0.7.0"
__author__ = "Kow Kuroda & Claude"

import numpy as np
from numpy.linalg import norm
from typing import Dict, List, Tuple, Optional, Union, Set
from pathlib import Path
from collections import defaultdict, Counter
import warnings

# Import hybrid anchor extractor
try:
    from .anchor_extractor import HybridAnchorExtractor, MANUAL_ANCHORS, FRAME_ANCHORS
    HYBRID_AVAILABLE = True
except ImportError:
    HYBRID_AVAILABLE = False

# Import spectral clustering module
try:
    from .spectral import (
        discover_anchors_spectral, 
        discover_anchors_spectral_fixed_k,
        spectral_clustering
    )
    SPECTRAL_AVAILABLE = True
except ImportError:
    SPECTRAL_AVAILABLE = False


# =============================================================================
# Core SenseExplorer Class
# =============================================================================

class SenseExplorer:
    """
    Explore word sense structure via simulated self-repair.
    
    Two operational modes:
    
    1. UNSUPERVISED (discover_senses):
       - True sense discovery from distributional data alone
       - No external knowledge required
       - Senses emerge via k-means clustering of neighbors
       - Achieves ~56% accuracy
    
    2. WEAKLY SUPERVISED (induce_senses):
       - Anchor-guided sense induction
       - Uses FrameNet frames or WordNet glosses as anchors
       - Senses are induced toward anchor-defined targets
       - Achieves ~88% accuracy
    
    Both modes use the same self-repair mechanism:
      1. Creating noisy copies of a word's embedding
      2. Seeding subsets toward different sense directions
      3. Allowing copies to self-organize toward stable configurations
      4. Observing which "attractor basins" the copies settle into
    
    Noise level acts as a granularity control parameter:
      - Low noise (10-20%): Fine distinctions, may over-split
      - Medium noise (30-50%): Standard sense-level distinctions  
      - High noise (60-80%): Coarse groupings
    
    Example:
        >>> se = SenseExplorer.from_glove("glove.6B.100d.txt")
        
        # Unsupervised discovery
        >>> se.discover_senses("bank")
        {'sense_0': array([...]), 'sense_1': array([...])}
        
        # Weakly supervised induction
        >>> se.induce_senses("bank")
        {'financial': array([...]), 'river': array([...])}
    """
    
    def __init__(
        self,
        embeddings: Dict[str, np.ndarray],
        dim: int = None,
        default_n_senses: int = 2,
        n_copies: int = 100,
        noise_level: float = 0.5,
        seed_strength: float = 0.3,
        n_iterations: int = 15,
        anchor_pull: float = 0.2,
        n_anchors: int = 8,
        top_k: int = 50,
        clustering_method: str = 'spectral',
        use_hybrid_anchors: bool = True,
        verbose: bool = True
    ):
        """
        Initialize SenseExplorer with embeddings.
        
        Args:
            embeddings: Dict mapping words to numpy vectors
            dim: Embedding dimension (auto-detected if None)
            default_n_senses: Default number of senses to discover
            n_copies: Number of noisy copies for self-repair
            noise_level: Noise magnitude (fraction of embedding values)
                         Acts as granularity control: low=fine, high=coarse
            seed_strength: Strength of initial seeding toward senses
            n_iterations: Number of self-organization iterations
            anchor_pull: Strength of pull toward anchor centroids
            n_anchors: Number of anchors per sense for auto-discovery
            top_k: Number of neighbors for clustering (default: 50)
            clustering_method: Method for unsupervised discovery:
                              'spectral' (default, 90% accuracy at 50d)
                              'xmeans' (BIC-based, 64% accuracy at 50d)
                              'kmeans' (requires n_senses)
            use_hybrid_anchors: Use hybrid anchor extraction (FrameNet + WordNet)
                               Achieves 88% accuracy vs 56% for auto-discovery
            verbose: Print progress messages
        """
        self.embeddings = embeddings
        self.dim = dim or len(next(iter(embeddings.values())))
        self.vocab = set(embeddings.keys())
        self.vocab_size = len(self.vocab)
        
        # Self-repair parameters
        self.default_n_senses = default_n_senses
        self.n_copies = n_copies
        self.noise_level = noise_level
        self.seed_strength = seed_strength
        self.n_iterations = n_iterations
        self.anchor_pull = anchor_pull
        self.n_anchors = n_anchors
        self.top_k = top_k
        self.clustering_method = clustering_method
        self.use_hybrid_anchors = use_hybrid_anchors
        self.verbose = verbose
        
        # Validate clustering method
        valid_methods = ['spectral', 'xmeans', 'kmeans']
        if clustering_method not in valid_methods:
            raise ValueError(f"clustering_method must be one of {valid_methods}")
        
        if clustering_method == 'spectral' and not SPECTRAL_AVAILABLE:
            warnings.warn("Spectral clustering not available, falling back to xmeans")
            self.clustering_method = 'xmeans'
        
        # Initialize hybrid anchor extractor if available and enabled
        self._hybrid_extractor = None
        if use_hybrid_anchors and HYBRID_AVAILABLE:
            self._hybrid_extractor = HybridAnchorExtractor(
                self.vocab,
                use_manual=True,
                use_framenet=True,
                use_wordnet=True,
                verbose=verbose
            )
        
        # Caches
        self._sense_cache = {}  # word -> {sense_name: embedding}
        self._anchor_cache = {}  # word -> {sense_name: [anchor_words]}
        self._stability_cache = {}  # word -> stability analysis results
        
        # Precompute normalized embeddings for speed
        self._embeddings_norm = {}
        for word, emb in embeddings.items():
            self._embeddings_norm[word] = emb / (norm(emb) + 1e-10)
        
        if verbose:
            hybrid_status = "enabled" if (use_hybrid_anchors and HYBRID_AVAILABLE) else "disabled"
            print(f"SenseExplorer v{__version__} initialized with {self.vocab_size:,} words, dim={self.dim}")
            print(f"  Clustering method: {self.clustering_method} (top_k={self.top_k})")
            print(f"  Hybrid anchor extraction: {hybrid_status}")
            if use_hybrid_anchors and HYBRID_AVAILABLE:
                print(f"  Coverage: manual={len(MANUAL_ANCHORS)} words, framenet={len(FRAME_ANCHORS)} frames")
    
    # =========================================================================
    # Loading Methods
    # =========================================================================
    
    @classmethod
    def from_glove(cls, filepath: str, max_words: int = None, **kwargs) -> 'SenseExplorer':
        """
        Load from GloVe format (text or binary).
        
        Args:
            filepath: Path to GloVe file (.txt or .bin)
            max_words: Maximum number of words to load
            **kwargs: Additional arguments for SenseExplorer
        
        Returns:
            SenseExplorer instance
        """
        embeddings, dim = cls._load_glove(filepath, max_words, kwargs.get('verbose', True))
        return cls(embeddings, dim=dim, **kwargs)
    
    @classmethod
    def from_word2vec(cls, filepath: str, max_words: int = None, binary: bool = True, **kwargs) -> 'SenseExplorer':
        """
        Load from Word2Vec format.
        
        Args:
            filepath: Path to Word2Vec file
            max_words: Maximum number of words to load
            binary: Whether file is binary format
            **kwargs: Additional arguments for SenseExplorer
        
        Returns:
            SenseExplorer instance
        """
        embeddings, dim = cls._load_word2vec(filepath, max_words, binary, kwargs.get('verbose', True))
        return cls(embeddings, dim=dim, **kwargs)
    
    @classmethod
    def from_dict(cls, embeddings: Dict[str, Union[np.ndarray, List[float]]], **kwargs) -> 'SenseExplorer':
        """
        Create from dictionary of embeddings.
        
        Args:
            embeddings: Dict mapping words to vectors (numpy arrays or lists)
            **kwargs: Additional arguments for SenseExplorer
        
        Returns:
            SenseExplorer instance
        """
        # Convert lists to numpy arrays if needed
        emb_dict = {}
        for word, vec in embeddings.items():
            if isinstance(vec, list):
                emb_dict[word] = np.array(vec, dtype=np.float32)
            else:
                emb_dict[word] = vec.astype(np.float32)
        
        return cls(emb_dict, **kwargs)
    
    @staticmethod
    def _load_glove(filepath: str, max_words: int = None, verbose: bool = True) -> Tuple[Dict[str, np.ndarray], int]:
        """Load GloVe embeddings from file."""
        embeddings = {}
        dim = None
        
        if verbose:
            print(f"Loading embeddings from {filepath}...")
        
        filepath = Path(filepath)
        
        if filepath.suffix == '.bin':
            # Binary format (gensim-style)
            with open(filepath, 'rb') as f:
                header = f.readline().decode('utf-8').strip()
                vocab_size, dim = map(int, header.split())
                
                for _ in range(vocab_size):
                    word_bytes = b''
                    while True:
                        c = f.read(1)
                        if c == b' ' or c == b'':
                            break
                        word_bytes += c
                    word = word_bytes.decode('utf-8', errors='replace')
                    vec = np.frombuffer(f.read(dim * 4), dtype=np.float32).copy()
                    embeddings[word] = vec
                    
                    if max_words and len(embeddings) >= max_words:
                        break
        else:
            # Text format
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    word = parts[0]
                    try:
                        vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                        embeddings[word] = vec
                        if dim is None:
                            dim = len(vec)
                        if max_words and len(embeddings) >= max_words:
                            break
                    except ValueError:
                        continue
        
        if verbose:
            print(f"Loaded {len(embeddings):,} embeddings with dimension {dim}")
        
        return embeddings, dim
    
    @staticmethod
    def _load_word2vec(filepath: str, max_words: int = None, binary: bool = True, verbose: bool = True) -> Tuple[Dict[str, np.ndarray], int]:
        """Load Word2Vec embeddings from file."""
        embeddings = {}
        
        if verbose:
            print(f"Loading Word2Vec embeddings from {filepath}...")
        
        with open(filepath, 'rb') as f:
            header = f.readline().decode('utf-8').strip()
            vocab_size, dim = map(int, header.split())
            
            for _ in range(vocab_size):
                word_bytes = b''
                while True:
                    c = f.read(1)
                    if c == b' ':
                        break
                    if c == b'':
                        break
                    word_bytes += c
                
                word = word_bytes.decode('utf-8', errors='replace')
                
                if binary:
                    vec = np.frombuffer(f.read(dim * 4), dtype=np.float32).copy()
                else:
                    vec = np.array([float(x) for x in f.readline().split()], dtype=np.float32)
                
                embeddings[word] = vec
                
                if max_words and len(embeddings) >= max_words:
                    break
        
        if verbose:
            print(f"Loaded {len(embeddings):,} embeddings with dimension {dim}")
        
        return embeddings, dim
    
    # =========================================================================
    # Core Sense Discovery
    # =========================================================================
    
    def discover_senses(
        self,
        word: str,
        n_senses: int = None,
        noise_level: float = None,
        clustering_method: str = None,
        force: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        UNSUPERVISED sense discovery via clustering.
        
        This is true sense discovery: senses emerge from the distributional
        structure alone, with no external knowledge or anchor guidance.
        
        With spectral clustering (default): achieves ~90% accuracy at 50d
        With X-means: achieves ~64% accuracy at 50d
        
        Args:
            word: Target word
            n_senses: Number of senses to discover (default: 2)
                     Only used if clustering_method='kmeans'
            noise_level: Override default noise level (granularity control)
            clustering_method: Override default clustering method
                              ('spectral', 'xmeans', 'kmeans')
            force: Force rediscovery even if cached
        
        Returns:
            Dict mapping sense names (sense_0, sense_1, ...) to embeddings
        
        Example:
            >>> se = SenseExplorer.from_glove("glove.txt")
            >>> senses = se.discover_senses("bank", n_senses=2)
            >>> print(senses.keys())
            dict_keys(['sense_0', 'sense_1'])
        """
        if word not in self.vocab:
            raise ValueError(f"Word '{word}' not in vocabulary")
        
        cache_key = f"{word}_discovered"
        if not force and cache_key in self._sense_cache:
            return self._sense_cache[cache_key]
        
        n_senses = n_senses or self.default_n_senses
        noise = noise_level if noise_level is not None else self.noise_level
        method = clustering_method or self.clustering_method
        
        # Select clustering method
        if method == 'spectral' and SPECTRAL_AVAILABLE:
            anchors = discover_anchors_spectral_fixed_k(
                word, 
                self._embeddings_norm, 
                self.embeddings,
                self.vocab,
                n_senses=n_senses,
                n_anchors=self.n_anchors,
                top_k=self.top_k
            )
        elif method == 'xmeans':
            anchors, _ = self._discover_anchors_xmeans(word)
        else:  # kmeans
            anchors = self._discover_anchors_kmeans(word, n_senses)
        
        sense_centroids = self._compute_sense_centroids(anchors)
        
        if len(sense_centroids) < 2:
            emb = self._embeddings_norm[word]
            self._sense_cache[cache_key] = {'default': emb}
            return {'default': emb}
        
        sense_embs = self._simulated_repair(word, sense_centroids, noise, sense_loyal=False)
        
        self._sense_cache[cache_key] = sense_embs
        self._anchor_cache[cache_key] = anchors
        
        if self.verbose:
            print(f"  [UNSUPERVISED/{method}] Discovered {len(sense_embs)} senses for '{word}'")
        
        return sense_embs
    
    def induce_senses(
        self,
        word: str,
        anchors: Dict[str, List[str]] = None,
        n_senses: int = None,
        noise_level: float = None,
        force: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        WEAKLY SUPERVISED sense induction via anchor guidance.
        
        Senses are induced toward targets defined by anchor words.
        Uses hybrid extraction (FrameNet + WordNet) when anchors not provided.
        Achieves ~88% accuracy with frame-based anchors.
        
        Args:
            word: Target word
            anchors: Optional dict of {sense_name: [anchor_words]}
                     If None, uses hybrid extraction (FrameNet → WordNet → k-means)
            n_senses: Number of senses (used only if anchors is None)
            noise_level: Override default noise level (granularity control)
            force: Force re-induction even if cached
        
        Returns:
            Dict mapping sense names to sense-specific embeddings
        
        Example:
            >>> se = SenseExplorer.from_glove("glove.txt")
            >>> senses = se.induce_senses("bank")
            >>> print(senses.keys())
            dict_keys(['financial', 'river'])
        """
        if word not in self.vocab:
            raise ValueError(f"Word '{word}' not in vocabulary")
        
        cache_key = f"{word}_induced"
        if not force and cache_key in self._sense_cache:
            return self._sense_cache[cache_key]
        
        n_senses = n_senses or self.default_n_senses
        noise = noise_level if noise_level is not None else self.noise_level
        
        # Weakly supervised: use provided anchors or hybrid extraction
        if anchors is None:
            anchors = self._extract_anchors_hybrid(word, n_senses)
        
        sense_centroids = self._compute_sense_centroids(anchors)
        
        if len(sense_centroids) < 2:
            emb = self._embeddings_norm[word]
            self._sense_cache[cache_key] = {'default': emb}
            return {'default': emb}
        
        sense_embs = self._simulated_repair(word, sense_centroids, noise, sense_loyal=True)
        
        self._sense_cache[cache_key] = sense_embs
        self._anchor_cache[cache_key] = anchors
        
        if self.verbose:
            print(f"  [WEAKLY SUPERVISED] Induced {len(sense_embs)} senses for '{word}'")
        
        return sense_embs
    
    def explore_senses(
        self,
        word: str,
        mode: str = 'auto',
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Convenience method: explore senses using specified mode.
        
        Args:
            word: Target word
            mode: One of:
                  - 'discover': Unsupervised with specified n_senses
                  - 'discover_auto': Unsupervised, auto-detect n_senses (X-means)
                  - 'induce': Weakly supervised with anchors
                  - 'auto': Induce if anchors available, else discover_auto
            **kwargs: Passed to the underlying method
        
        Returns:
            Dict mapping sense names to sense-specific embeddings
        
        Example:
            >>> se.explore_senses("bank", mode='discover_auto')  # X-means
            >>> se.explore_senses("bank", mode='induce')         # With anchors
            >>> se.explore_senses("bank", mode='auto')           # Best available
        """
        if mode == 'discover':
            return self.discover_senses(word, **kwargs)
        elif mode == 'discover_auto':
            return self.discover_senses_auto(word, **kwargs)
        elif mode == 'induce':
            return self.induce_senses(word, **kwargs)
        elif mode == 'auto':
            # Try hybrid extraction; if fails, fall back to X-means discovery
            if self._hybrid_extractor is not None:
                anchors, source = self._hybrid_extractor.extract(word, kwargs.get('n_senses', 2))
                if anchors and source != 'auto':
                    return self.induce_senses(word, anchors=anchors, **kwargs)
            return self.discover_senses_auto(word, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'discover', 'discover_auto', 'induce', or 'auto'")
    
    def _extract_anchors_hybrid(self, word: str, n_senses: int) -> Dict[str, List[str]]:
        """
        Extract anchors using hybrid strategy (FrameNet → WordNet → k-means).
        
        This is for weakly supervised induction.
        """
        if self._hybrid_extractor is not None:
            anchors, source = self._hybrid_extractor.extract(word, n_senses)
            if anchors:
                if self.verbose:
                    print(f"  Anchors extracted via {source} for '{word}'")
                return anchors
        
        # Fall back to k-means if hybrid fails
        if self.verbose:
            print(f"  Falling back to k-means for '{word}'")
        return self._discover_anchors_kmeans(word, n_senses)

    def induce_senses_stable(
        self,
        word: str,
        n_senses: int = None,
        anchors: Dict[str, List[str]] = None,
        noise_levels: List[float] = None,
        n_trials: int = 3
    ) -> Dict:
        """
        Induce senses using stability-based method.
        
        This method runs sense induction at multiple noise levels and
        finds the STABLE sense count - the one that persists across
        different granularities. This is more robust than single-noise
        discovery.
        
        Args:
            word: Target word
            n_senses: Max number of senses to consider
            anchors: Optional dict of {sense_name: [anchor_words]}
            noise_levels: List of noise levels to test (default: 10%-70%)
            n_trials: Number of trials per noise level
        
        Returns:
            Dict with:
                - 'senses': sense embeddings at optimal noise
                - 'stable_k': stable sense count
                - 'optimal_noise': best noise level
                - 'confidence': stability confidence
                - 'stable_range': (start, end) noise range
        """
        if word not in self.vocab:
            raise ValueError(f"Word '{word}' not in vocabulary")
        
        if noise_levels is None:
            noise_levels = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
        
        n_senses = n_senses or 6  # Allow more senses for exploration
        
        # Get or induce anchors
        if anchors is None:
            anchors = self._extract_anchors_hybrid(word, n_senses)
        
        sense_centroids = self._compute_sense_centroids(anchors)
        
        if len(sense_centroids) < 2:
            return {
                'senses': {'default': self._embeddings_norm[word]},
                'stable_k': 1,
                'optimal_noise': self.noise_level,
                'confidence': 1.0,
                'stable_range': None
            }
        
        # Collect sense counts at each noise level
        sense_counts = []
        results_by_noise = {}
        
        for noise in noise_levels:
            trial_counts = []
            trial_results = []
            
            for _ in range(n_trials):
                sense_embs = self._simulated_repair(word, sense_centroids, noise, sense_loyal=True)
                k = len(sense_embs)
                trial_counts.append(k)
                trial_results.append(sense_embs)
            
            # Use mode (most common count)
            mode_count = Counter(trial_counts).most_common(1)[0][0]
            sense_counts.append((noise, mode_count))
            
            # Store result with mode count
            for embs in trial_results:
                if len(embs) == mode_count:
                    results_by_noise[noise] = embs
                    break
        
        # Find longest stable run
        best_run = None
        best_length = 0
        current_k = None
        current_start = None
        
        for noise, k in sense_counts:
            if k != current_k:
                if current_k is not None and current_start is not None:
                    length = noise - current_start
                    if length > best_length:
                        best_length = length
                        best_run = (current_start, noise, current_k)
                current_k = k
                current_start = noise
        
        # Check final run
        if current_k is not None and current_start is not None:
            length = sense_counts[-1][0] - current_start + (noise_levels[1] - noise_levels[0])
            if length > best_length:
                best_length = length
                best_run = (current_start, sense_counts[-1][0], current_k)
        
        if best_run is None:
            # No stable run found, use most common count
            all_counts = [k for _, k in sense_counts]
            stable_k = Counter(all_counts).most_common(1)[0][0]
            stable_range = None
            confidence = 0.0
            optimal_noise = self.noise_level
        else:
            stable_k = best_run[2]
            stable_range = (best_run[0], best_run[1])
            confidence = best_length / (noise_levels[-1] - noise_levels[0])
            optimal_noise = (stable_range[0] + stable_range[1]) / 2
        
        # Get senses at optimal noise
        if optimal_noise in results_by_noise:
            final_senses = results_by_noise[optimal_noise]
        else:
            final_senses = self._simulated_repair(word, sense_centroids, optimal_noise, sense_loyal=True)
        
        # Cache
        self._sense_cache[word] = final_senses
        self._stability_cache[word] = {
            'stable_k': stable_k,
            'optimal_noise': optimal_noise,
            'confidence': confidence,
            'stable_range': stable_range,
            'sense_counts': sense_counts
        }
        
        return {
            'senses': final_senses,
            'stable_k': stable_k,
            'optimal_noise': optimal_noise,
            'confidence': confidence,
            'stable_range': stable_range
        }
    
    def _discover_anchors_kmeans(self, word: str, n_senses: int) -> Dict[str, List[str]]:
        """
        UNSUPERVISED: Discover anchors via k-means clustering of nearest neighbors.
        
        This is used for true sense discovery mode (no external knowledge).
        Achieves ~56% accuracy.
        """
        target_emb = self._embeddings_norm[word]
        
        # Find nearest neighbors (excluding the word itself)
        k_neighbors = min(100, self.vocab_size - 1)
        
        similarities = []
        for w, emb in self._embeddings_norm.items():
            if w == word:
                continue
            sim = np.dot(target_emb, emb)
            similarities.append((w, sim, self.embeddings[w]))
        
        similarities.sort(key=lambda x: -x[1])
        neighbors = similarities[:k_neighbors]
        
        neighbor_words = [n[0] for n in neighbors]
        neighbor_vecs = np.array([n[2] for n in neighbors])
        
        # Normalize for clustering
        norms = np.linalg.norm(neighbor_vecs, axis=1, keepdims=True) + 1e-10
        neighbor_vecs_norm = neighbor_vecs / norms
        
        # K-means clustering
        assignments, centroids = self._kmeans(neighbor_vecs_norm, n_senses)
        
        # Group neighbors by cluster
        anchors = {}
        for sense_idx in range(n_senses):
            mask = assignments == sense_idx
            cluster_words = [neighbor_words[i] for i in range(len(neighbor_words)) if mask[i]]
            
            if cluster_words:
                # Use top n_anchors words from each cluster
                anchors[f"sense_{sense_idx}"] = cluster_words[:self.n_anchors]
        
        return anchors
    
    def _discover_anchors_xmeans(
        self, 
        word: str, 
        max_senses: int = 6,
        min_senses: int = 2
    ) -> Tuple[Dict[str, List[str]], int]:
        """
        UNSUPERVISED + PARAMETER-FREE: Discover anchors via X-means clustering.
        
        X-means automatically determines the optimal number of clusters using
        BIC (Bayesian Information Criterion). This makes sense discovery
        truly parameter-free.
        
        Args:
            word: Target word
            max_senses: Maximum number of senses to consider
            min_senses: Minimum number of senses to consider
        
        Returns:
            Tuple of (anchors dict, optimal_k)
        """
        target_emb = self._embeddings_norm[word]
        
        # Find nearest neighbors
        k_neighbors = min(100, self.vocab_size - 1)
        
        similarities = []
        for w, emb in self._embeddings_norm.items():
            if w == word:
                continue
            sim = np.dot(target_emb, emb)
            similarities.append((w, sim, self.embeddings[w]))
        
        similarities.sort(key=lambda x: -x[1])
        neighbors = similarities[:k_neighbors]
        
        neighbor_words = [n[0] for n in neighbors]
        neighbor_vecs = np.array([n[2] for n in neighbors])
        
        # Normalize for clustering
        norms = np.linalg.norm(neighbor_vecs, axis=1, keepdims=True) + 1e-10
        neighbor_vecs_norm = neighbor_vecs / norms
        
        # X-means: try different k values and select by BIC
        best_bic = float('-inf')
        best_k = min_senses
        best_assignments = None
        best_centroids = None
        
        for k in range(min_senses, max_senses + 1):
            assignments, centroids = self._kmeans(neighbor_vecs_norm, k, max_iter=50)
            bic = self._compute_bic(neighbor_vecs_norm, assignments, centroids, k)
            
            if bic > best_bic:
                best_bic = bic
                best_k = k
                best_assignments = assignments
                best_centroids = centroids
        
        # Group neighbors by cluster
        anchors = {}
        for sense_idx in range(best_k):
            mask = best_assignments == sense_idx
            cluster_words = [neighbor_words[i] for i in range(len(neighbor_words)) if mask[i]]
            
            if cluster_words:
                anchors[f"sense_{sense_idx}"] = cluster_words[:self.n_anchors]
        
        return anchors, best_k
    
    def _compute_bic(
        self,
        data: np.ndarray,
        assignments: np.ndarray,
        centroids: np.ndarray,
        k: int
    ) -> float:
        """
        Compute Bayesian Information Criterion (BIC) for clustering.
        
        BIC = log-likelihood - (penalty for model complexity)
        Higher BIC = better model (balances fit vs complexity)
        
        Args:
            data: Data points (n_samples, n_features)
            assignments: Cluster assignments
            centroids: Cluster centroids
            k: Number of clusters
        
        Returns:
            BIC score (higher is better)
        """
        n_samples, n_features = data.shape
        
        # Compute within-cluster variance
        total_variance = 0.0
        cluster_sizes = []
        
        for cluster_idx in range(k):
            mask = assignments == cluster_idx
            cluster_size = np.sum(mask)
            cluster_sizes.append(cluster_size)
            
            if cluster_size > 1:
                cluster_points = data[mask]
                centroid = centroids[cluster_idx]
                # Sum of squared distances to centroid
                distances = np.sum((cluster_points - centroid) ** 2)
                total_variance += distances
        
        # Avoid division by zero
        if n_samples <= k:
            return float('-inf')
        
        # Estimate variance
        variance = total_variance / (n_samples - k) if (n_samples - k) > 0 else 1e-10
        variance = max(variance, 1e-10)  # Avoid log(0)
        
        # Log-likelihood (assuming Gaussian clusters)
        log_likelihood = 0.0
        for cluster_idx in range(k):
            n_i = cluster_sizes[cluster_idx]
            if n_i > 0:
                # Log-likelihood for this cluster
                log_likelihood += n_i * np.log(n_i / n_samples)  # Prior
                log_likelihood -= n_i * n_features / 2 * np.log(2 * np.pi * variance)  # Gaussian
                log_likelihood -= (n_i - 1) / 2  # Correction
        
        # Number of parameters: k centroids * n_features + k-1 mixing proportions + 1 variance
        n_params = k * n_features + k
        
        # BIC = log-likelihood - penalty
        bic = log_likelihood - n_params / 2 * np.log(n_samples)
        
        return bic
    
    def discover_senses_auto(
        self,
        word: str,
        max_senses: int = 5,
        min_senses: int = 2,
        noise_level: float = None,
        clustering_method: str = None,
        force: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        PARAMETER-FREE sense discovery with automatic k selection.
        
        Automatically determines the optimal number of senses.
        
        With spectral clustering (default): uses eigengap heuristic (90% at 50d)
        With X-means: uses BIC criterion (64% at 50d)
        
        Args:
            word: Target word
            max_senses: Maximum senses to consider (default: 5)
            min_senses: Minimum senses to consider (default: 2)
            noise_level: Override default noise level
            clustering_method: Override default method ('spectral' or 'xmeans')
            force: Force rediscovery even if cached
        
        Returns:
            Dict mapping sense names to sense-specific embeddings
        
        Example:
            >>> se = SenseExplorer.from_glove("glove.txt")
            >>> senses = se.discover_senses_auto("bank")  # No n_senses needed!
            >>> print(f"Found {len(senses)} senses: {list(senses.keys())}")
            Found 2 senses: ['sense_0', 'sense_1']
        """
        if word not in self.vocab:
            raise ValueError(f"Word '{word}' not in vocabulary")
        
        cache_key = f"{word}_auto_discovered"
        if not force and cache_key in self._sense_cache:
            return self._sense_cache[cache_key]
        
        noise = noise_level if noise_level is not None else self.noise_level
        method = clustering_method or self.clustering_method
        
        # Select clustering method
        if method == 'spectral' and SPECTRAL_AVAILABLE:
            anchors, optimal_k = discover_anchors_spectral(
                word,
                self._embeddings_norm,
                self.embeddings,
                self.vocab,
                n_anchors=self.n_anchors,
                top_k=self.top_k,
                min_senses=min_senses,
                max_senses=max_senses
            )
            method_name = "SPECTRAL"
        else:
            # Fall back to X-means
            anchors, optimal_k = self._discover_anchors_xmeans(word, max_senses, min_senses)
            method_name = "X-MEANS"
        
        if self.verbose:
            print(f"  [{method_name}] Auto-discovered {optimal_k} senses for '{word}'")
        
        sense_centroids = self._compute_sense_centroids(anchors)
        
        if len(sense_centroids) < 2:
            emb = self._embeddings_norm[word]
            self._sense_cache[cache_key] = {'default': emb}
            return {'default': emb}
        
        sense_embs = self._simulated_repair(word, sense_centroids, noise, sense_loyal=False)
        
        self._sense_cache[cache_key] = sense_embs
        self._anchor_cache[cache_key] = anchors
        
        return sense_embs
    
    def _compute_sense_centroids(self, anchors: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        """Compute normalized centroid for each sense from anchor words."""
        centroids = {}
        
        for sense_name, anchor_words in anchors.items():
            anchor_vecs = [self.embeddings[w] for w in anchor_words if w in self.vocab]
            if anchor_vecs:
                centroid = np.mean(anchor_vecs, axis=0)
                centroid = centroid / (norm(centroid) + 1e-10)
                centroids[sense_name] = centroid
        
        return centroids
    
    def _simulated_repair(
        self,
        word: str,
        sense_centroids: Dict[str, np.ndarray],
        noise_level: float,
        sense_loyal: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Run simulated self-repair to discover sense-specific embeddings.
        
        This is the core algorithm:
          1. Create seeded noisy copies (each copy is seeded toward a specific sense)
          2. Iteratively pull copies toward sense centroids
          3. Average copies by sense assignment
        
        Args:
            word: Target word
            sense_centroids: Dict of sense_name -> centroid vector
            noise_level: Amount of noise to add
            sense_loyal: If True, copies stay with seeded sense (for induction).
                        If False, copies can switch to nearest sense (for discovery).
        
        Note on sense_loyal:
          - True (default): For INDUCTION - ensures all specified senses are found
          - False: For DISCOVERY - allows natural attractor dynamics
        """
        embedding = self.embeddings[word]
        senses = list(sense_centroids.keys())
        n_senses = len(senses)
        copies_per_sense = self.n_copies // n_senses
        
        # Create seeded noisy copies
        copies = []
        copy_sense_ids = []  # Track which sense each copy was seeded toward
        
        for sense_idx, sense in enumerate(senses):
            centroid = sense_centroids[sense]
            for _ in range(copies_per_sense):
                copy = embedding.copy()
                
                # Add noise
                n_perturb = np.random.randint(int(self.dim * 0.5), int(self.dim * 0.8))
                perturb_dims = np.random.choice(self.dim, n_perturb, replace=False)
                for d in perturb_dims:
                    copy[d] += np.random.randn() * abs(embedding[d]) * noise_level
                
                # Seed toward sense
                copy_norm = copy / (norm(copy) + 1e-10)
                direction = centroid - copy_norm
                copy += self.seed_strength * direction * norm(copy)
                
                copies.append(copy)
                copy_sense_ids.append(sense_idx)
        
        copies = np.array(copies)
        copy_sense_ids = np.array(copy_sense_ids)
        centroid_matrix = np.array([sense_centroids[s] for s in senses])
        
        # Self-organization iterations
        current_pull = self.anchor_pull
        
        for _ in range(self.n_iterations):
            norms = np.linalg.norm(copies, axis=1, keepdims=True) + 1e-10
            copies_norm = copies / norms
            
            if sense_loyal:
                # SENSE-LOYAL: Pull toward SEEDED sense (for induction)
                for i in range(len(copies)):
                    target = centroid_matrix[copy_sense_ids[i]]
                    direction = target - copies_norm[i]
                    copies[i] += current_pull * direction * norms[i, 0]
            else:
                # COMPETITIVE: Pull toward NEAREST sense (for discovery)
                similarities = copies_norm @ centroid_matrix.T
                assignments = np.argmax(similarities, axis=1)
                
                for i in range(len(copies)):
                    target = centroid_matrix[assignments[i]]
                    direction = target - copies_norm[i]
                    copies[i] += current_pull * direction * norms[i, 0]
            
            current_pull *= 0.95  # Decay
        
        # Final sense assignment
        norms = np.linalg.norm(copies, axis=1, keepdims=True) + 1e-10
        copies_norm = copies / norms
        
        if sense_loyal:
            # Use seeded sense assignment
            final_assignments = copy_sense_ids
            min_copies = 1  # Keep all senses that have at least 1 copy
        else:
            # Use nearest sense assignment
            similarities = copies_norm @ centroid_matrix.T
            final_assignments = np.argmax(similarities, axis=1)
            min_copies = self.n_copies * 0.05  # At least 5% threshold
        
        # Average copies by sense
        sense_embeddings = {}
        
        for sense_idx, sense in enumerate(senses):
            mask = final_assignments == sense_idx
            if mask.sum() >= min_copies:
                sense_emb = copies[mask].mean(axis=0)
                sense_emb = sense_emb / (norm(sense_emb) + 1e-10)
                sense_embeddings[sense] = sense_emb
        
        return sense_embeddings
    
    def _kmeans(self, vectors: np.ndarray, k: int, max_iter: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """K-means clustering with k-means++ initialization."""
        n, d = vectors.shape
        
        if n < k:
            k = n
        
        # K-means++ initialization
        centroids = np.zeros((k, d))
        centroids[0] = vectors[np.random.randint(n)]
        
        for i in range(1, k):
            dists = np.min([np.sum((vectors - centroids[j])**2, axis=1) 
                            for j in range(i)], axis=0)
            dists = np.maximum(dists, 1e-10)
            probs = dists / dists.sum()
            probs = probs / probs.sum()
            centroids[i] = vectors[np.random.choice(n, p=probs)]
        
        assignments = np.zeros(n, dtype=int)
        
        for _ in range(max_iter):
            distances = np.array([np.sum((vectors - centroids[j])**2, axis=1) for j in range(k)]).T
            new_assignments = np.argmin(distances, axis=1)
            
            if np.all(new_assignments == assignments):
                break
            assignments = new_assignments
            
            for j in range(k):
                mask = assignments == j
                if mask.sum() > 0:
                    centroids[j] = vectors[mask].mean(axis=0)
        
        return assignments, centroids
    
    # =========================================================================
    # Similarity Methods
    # =========================================================================
    
    def similarity(
        self,
        word1: str,
        word2: str,
        sense_aware: bool = True,
        context: List[str] = None
    ) -> float:
        """
        Compute similarity between two words.
        
        Args:
            word1: First word
            word2: Second word
            sense_aware: If True, use sense-aware similarity
            context: Optional context words for disambiguation
        
        Returns:
            Cosine similarity score
        """
        if word1 not in self.vocab or word2 not in self.vocab:
            return 0.0
        
        if not sense_aware:
            return self._cosine(word1, word2)
        
        if context:
            return self.context_similarity(word1, word2, context)[0]
        else:
            return self.max_sense_similarity(word1, word2)[0]
    
    def _cosine(self, word1: str, word2: str) -> float:
        """Standard cosine similarity."""
        return float(np.dot(self._embeddings_norm[word1], self._embeddings_norm[word2]))
    
    def max_sense_similarity(self, word1: str, word2: str) -> Tuple[float, str]:
        """
        Max-sense similarity: max_i sim(sense_i(w1), w2)
        
        Returns:
            Tuple of (similarity, selected_sense_name)
        """
        senses1 = self.induce_senses(word1)
        e2 = self._embeddings_norm[word2]
        
        best_sim = -1.0
        best_sense = 'default'
        
        for sense_name, sense_emb in senses1.items():
            sim = np.dot(sense_emb, e2)
            if sim > best_sim:
                best_sim = sim
                best_sense = sense_name
        
        return float(best_sim), best_sense
    
    def best_match_similarity(self, word1: str, word2: str) -> Tuple[float, str, str]:
        """
        Best-match similarity for two potentially polysemous words.
        
        Returns:
            Tuple of (similarity, word1_sense, word2_sense)
        """
        senses1 = self.induce_senses(word1)
        senses2 = self.induce_senses(word2)
        
        best_sim = -1.0
        best_s1 = 'default'
        best_s2 = 'default'
        
        for s1_name, s1_emb in senses1.items():
            for s2_name, s2_emb in senses2.items():
                sim = np.dot(s1_emb, s2_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_s1 = s1_name
                    best_s2 = s2_name
        
        return float(best_sim), best_s1, best_s2
    
    def context_similarity(
        self,
        word1: str,
        word2: str,
        context: List[str]
    ) -> Tuple[float, str, str]:
        """
        Context-disambiguated similarity.
        
        Args:
            word1: First word
            word2: Second word
            context: Context words for disambiguation
        
        Returns:
            Tuple of (similarity, word1_sense, word2_sense)
        """
        senses1 = self.induce_senses(word1)
        senses2 = self.induce_senses(word2)
        
        # Compute context vector
        context_vecs = [self._embeddings_norm[c] for c in context if c in self.vocab]
        if not context_vecs:
            return self.best_match_similarity(word1, word2)
        
        context_mean = np.mean(context_vecs, axis=0)
        context_mean = context_mean / (norm(context_mean) + 1e-10)
        
        # Select senses aligned with context
        best_s1 = max(senses1.items(), key=lambda x: np.dot(x[1], context_mean))[0]
        best_s2 = max(senses2.items(), key=lambda x: np.dot(x[1], context_mean))[0]
        
        sim = np.dot(senses1[best_s1], senses2[best_s2])
        
        return float(sim), best_s1, best_s2
    
    # =========================================================================
    # Analogy
    # =========================================================================
    
    def analogy(
        self,
        a: str,
        b: str,
        c: str,
        top_k: int = 10,
        sense_aware: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Solve analogy: a:b :: c:?
        
        Args:
            a, b, c: Analogy words
            top_k: Number of results to return
            sense_aware: If True, use sense-aware analogy
        
        Returns:
            List of (word, score) tuples
        """
        if a not in self.vocab or b not in self.vocab or c not in self.vocab:
            return []
        
        exclude = {a, b, c}
        
        if sense_aware:
            senses_a = self.induce_senses(a)
            senses_c = self.induce_senses(c)
            e_b = self._embeddings_norm[b]
            
            # Find which sense of 'a' relates to 'b'
            best_a_sense = max(senses_a.items(), key=lambda x: np.dot(x[1], e_b))[0]
            e_a = senses_a[best_a_sense]
            
            # Use matching sense of 'c' if available
            if best_a_sense in senses_c:
                e_c = senses_c[best_a_sense]
            else:
                e_c = max(senses_c.items(), key=lambda x: np.dot(x[1], e_a))[1]
        else:
            e_a = self._embeddings_norm[a]
            e_b = self._embeddings_norm[b]
            e_c = self._embeddings_norm[c]
        
        # Compute analogy vector
        analogy_vec = e_b - e_a + e_c
        analogy_vec = analogy_vec / (norm(analogy_vec) + 1e-10)
        
        # Find nearest neighbors
        results = []
        for word, emb in self._embeddings_norm.items():
            if word in exclude:
                continue
            score = np.dot(analogy_vec, emb)
            results.append((word, float(score)))
        
        results.sort(key=lambda x: -x[1])
        return results[:top_k]
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def get_senses(self, word: str) -> List[str]:
        """Get list of discovered sense names for a word."""
        if word in self._sense_cache:
            return list(self._sense_cache[word].keys())
        return []
    
    def get_sense_embedding(self, word: str, sense: str) -> Optional[np.ndarray]:
        """Get embedding for a specific sense of a word."""
        if word in self._sense_cache and sense in self._sense_cache[word]:
            return self._sense_cache[word][sense].copy()
        return None
    
    def get_anchors(self, word: str) -> Optional[Dict[str, List[str]]]:
        """Get anchor words used for each sense."""
        return self._anchor_cache.get(word)
    
    def get_stability_info(self, word: str) -> Optional[Dict]:
        """Get stability analysis results for a word."""
        return self._stability_cache.get(word)
    
    def clear_cache(self):
        """Clear all cached sense embeddings."""
        self._sense_cache.clear()
        self._anchor_cache.clear()
        self._stability_cache.clear()
    
    def set_anchors(self, word: str, anchors: Dict[str, List[str]]):
        """
        Manually set anchors for a word (useful for known polysemous words).
        
        Example:
            sr.set_anchors('bank', {
                'financial': ['money', 'account', 'loan'],
                'river': ['river', 'shore', 'water']
            })
        """
        # Clear existing cache for this word
        if word in self._sense_cache:
            del self._sense_cache[word]
        if word in self._stability_cache:
            del self._stability_cache[word]
        
        # Induce senses with these anchors
        self.induce_senses(word, anchors=anchors, force=True)
    
    def set_noise_level(self, noise_level: float):
        """
        Set the noise level (granularity control).
        
        Guidelines:
            - 0.1-0.2: Fine-grained (may over-split)
            - 0.3-0.5: Standard sense-level (recommended)
            - 0.6-0.8: Coarse-grained
        
        Higher noise = coarser sense distinctions
        """
        self.noise_level = noise_level
        # Clear cache since granularity changed
        self.clear_cache()
    
    # =========================================================================
    # Polarity Methods (97% accuracy)
    # =========================================================================
    
    def get_polarity(
        self,
        word: str,
        positive_seeds: List[str] = None,
        negative_seeds: List[str] = None
    ) -> Dict:
        """
        Get polarity classification for a word.
        
        This is a SUPERVISED method: requires seed words to define
        the positive/negative poles. Achieves 97% accuracy.
        
        Args:
            word: Target word
            positive_seeds: Words defining positive pole (default: sentiment words)
            negative_seeds: Words defining negative pole
        
        Returns:
            Dict with 'polarity', 'score', 'confidence'
        
        Example:
            >>> se.get_polarity("excellent")
            {'polarity': 'positive', 'score': 0.82, 'confidence': 0.91}
        """
        # Lazy import to avoid circular dependency
        from .polarity import PolarityFinder, DEFAULT_POLARITY_SEEDS
        
        # Use cached polarity finder or create new one
        cache_key = (tuple(positive_seeds) if positive_seeds else None,
                     tuple(negative_seeds) if negative_seeds else None)
        
        if not hasattr(self, '_polarity_finders'):
            self._polarity_finders = {}
        
        if cache_key not in self._polarity_finders:
            pf = PolarityFinder(
                self.embeddings,
                positive_seeds=positive_seeds,
                negative_seeds=negative_seeds,
                verbose=False
            )
            self._polarity_finders[cache_key] = pf
        
        return self._polarity_finders[cache_key].get_polarity(word)
    
    def classify_polarity(
        self,
        words: List[str],
        positive_seeds: List[str] = None,
        negative_seeds: List[str] = None,
        threshold: float = 0.0
    ) -> Dict[str, List[str]]:
        """
        Classify multiple words by polarity.
        
        Args:
            words: Words to classify
            positive_seeds: Words defining positive pole
            negative_seeds: Words defining negative pole
            threshold: Score threshold (|score| < threshold = neutral)
        
        Returns:
            Dict with 'positive', 'negative', 'neutral' word lists
        
        Example:
            >>> se.classify_polarity(['good', 'bad', 'table'])
            {'positive': ['good'], 'negative': ['bad'], 'neutral': ['table']}
        """
        from .polarity import PolarityFinder
        
        pf = PolarityFinder(
            self.embeddings,
            positive_seeds=positive_seeds,
            negative_seeds=negative_seeds,
            verbose=False
        )
        return pf.classify_words(words, threshold=threshold)
    
    def get_polarity_finder(
        self,
        positive_seeds: List[str] = None,
        negative_seeds: List[str] = None,
        domain: str = None
    ) -> 'PolarityFinder':
        """
        Get a PolarityFinder instance for advanced polarity operations.
        
        Args:
            positive_seeds: Custom positive seeds
            negative_seeds: Custom negative seeds  
            domain: Predefined domain ('sentiment', 'quality', 'morality',
                   'health', 'size', 'temperature')
        
        Returns:
            PolarityFinder instance
        
        Example:
            >>> pf = se.get_polarity_finder(domain='quality')
            >>> pf.most_polar_words(top_k=10)
        """
        from .polarity import PolarityFinder, DOMAIN_POLARITY_SEEDS
        
        if domain and domain in DOMAIN_POLARITY_SEEDS:
            positive_seeds = DOMAIN_POLARITY_SEEDS[domain]['positive']
            negative_seeds = DOMAIN_POLARITY_SEEDS[domain]['negative']
        
        return PolarityFinder(
            self.embeddings,
            positive_seeds=positive_seeds,
            negative_seeds=negative_seeds,
            verbose=self.verbose
        )
    
    def __repr__(self) -> str:
        return f"SenseExplorer(vocab_size={self.vocab_size:,}, dim={self.dim}, cached_senses={len(self._sense_cache)})"


# =============================================================================
# Predefined Anchor Sets for Common Polysemous Words
# =============================================================================

COMMON_POLYSEMOUS = {
    'bank': {
        'financial': ['money', 'account', 'deposit', 'loan', 'finance', 'credit', 'savings', 'investment'],
        'river': ['river', 'shore', 'water', 'stream', 'edge', 'slope', 'embankment', 'side']
    },
    'bat': {
        'animal': ['bird', 'wing', 'fly', 'cave', 'mammal', 'nocturnal', 'vampire', 'creature'],
        'sports': ['ball', 'hit', 'swing', 'baseball', 'cricket', 'wooden', 'player', 'game']
    },
    'cell': {
        'biology': ['organism', 'membrane', 'nucleus', 'dna', 'protein', 'tissue', 'division', 'microscope'],
        'prison': ['jail', 'prisoner', 'locked', 'bars', 'inmate', 'detention', 'solitary', 'confined'],
        'phone': ['mobile', 'telephone', 'call', 'wireless', 'smartphone', 'signal', 'battery', 'tower']
    },
    'crane': {
        'bird': ['bird', 'wing', 'fly', 'nest', 'feather', 'migrate', 'wetland', 'heron'],
        'machine': ['construction', 'lift', 'heavy', 'tower', 'operator', 'load', 'steel', 'building']
    },
    'mouse': {
        'animal': ['rat', 'rodent', 'cat', 'trap', 'cheese', 'squirrel', 'pet', 'tail'],
        'computer': ['keyboard', 'click', 'cursor', 'screen', 'computer', 'button', 'pointer', 'device']
    },
    'plant': {
        'vegetation': ['tree', 'flower', 'leaf', 'grow', 'seed', 'garden', 'root', 'soil'],
        'factory': ['manufacturing', 'production', 'industrial', 'facility', 'equipment', 'machinery', 'worker', 'output']
    },
    'spring': {
        'season': ['summer', 'winter', 'autumn', 'march', 'april', 'bloom', 'warm', 'flowers'],
        'water': ['fountain', 'well', 'source', 'flow', 'natural', 'mineral', 'hot', 'fresh']
    },
    'bass': {
        'fish': ['fish', 'fishing', 'lake', 'catch', 'trout', 'salmon', 'angler', 'pond'],
        'music': ['guitar', 'music', 'band', 'drum', 'sound', 'instrument', 'player', 'rhythm']
    },
    'match': {
        'fire': ['lighter', 'flame', 'burn', 'ignite', 'stick', 'sulfur', 'strike', 'box'],
        'competition': ['game', 'tournament', 'opponent', 'winner', 'sports', 'contest', 'play', 'team']
    },
    'bow': {
        'weapon': ['arrow', 'archer', 'shoot', 'hunting', 'string', 'target', 'quiver', 'crossbow'],
        'gesture': ['curtsy', 'nod', 'greeting', 'respect', 'bend', 'head', 'polite', 'acknowledge']
    }
}


def load_common_polysemous(sr: SenseExplorer, words: List[str] = None):
    """
    Pre-load sense anchors for common polysemous words.
    
    Args:
        sr: SenseExplorer instance
        words: List of words to load (default: all common polysemous)
    """
    words = words or list(COMMON_POLYSEMOUS.keys())
    
    for word in words:
        if word in COMMON_POLYSEMOUS and word in sr.vocab:
            sr.set_anchors(word, COMMON_POLYSEMOUS[word])


# =============================================================================
# Command-Line Interface
# =============================================================================

def main():
    """Command-line interface for SenseExplorer."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SenseExplorer: Sense Discovery via Simulated Self-Repair',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Induce senses for a word
  python -m sense_explorer --glove glove.6B.100d.txt --word bank
  
  # Use stability-based discovery
  python -m sense_explorer --glove glove.6B.100d.txt --word bank --stable
  
  # Compute sense-aware similarity
  python -m sense_explorer --glove glove.6B.100d.txt --similarity bank river
  
  # Solve analogy
  python -m sense_explorer --glove glove.6B.100d.txt --analogy bank money crane
  
  # Control granularity with noise level
  python -m sense_explorer --glove glove.6B.100d.txt --word bank --noise 0.3
        """
    )
    
    parser.add_argument('--glove', type=str, help='Path to GloVe embeddings')
    parser.add_argument('--word2vec', type=str, help='Path to Word2Vec embeddings')
    parser.add_argument('--max-words', type=int, default=50000, help='Max words to load')
    parser.add_argument('--word', type=str, help='Word to discover senses for')
    parser.add_argument('--similarity', nargs=2, metavar=('W1', 'W2'), help='Compute similarity')
    parser.add_argument('--analogy', nargs=3, metavar=('A', 'B', 'C'), help='Solve analogy a:b::c:?')
    parser.add_argument('--n-senses', type=int, default=2, help='Number of senses to discover')
    parser.add_argument('--noise', type=float, default=0.5, help='Noise level (granularity control)')
    parser.add_argument('--stable', action='store_true', help='Use stability-based sense induction')
    parser.add_argument('--use-predefined', action='store_true', help='Use predefined anchors for common words')
    
    args = parser.parse_args()
    
    # Load embeddings
    if args.glove:
        sr = SenseExplorer.from_glove(args.glove, max_words=args.max_words, noise_level=args.noise)
    elif args.word2vec:
        sr = SenseExplorer.from_word2vec(args.word2vec, max_words=args.max_words, noise_level=args.noise)
    else:
        parser.error("Must specify --glove or --word2vec")
    
    # Load predefined anchors
    if args.use_predefined:
        load_common_polysemous(sr)
    
    # Induce senses
    if args.word:
        print(f"\nDiscovering senses for '{args.word}'...")
        
        if args.stable:
            result = sr.induce_senses_stable(args.word, n_senses=args.n_senses)
            print(f"Stable sense count: {result['stable_k']}")
            print(f"Confidence: {result['confidence']:.1%}")
            print(f"Optimal noise: {result['optimal_noise']:.1%}")
            if result['stable_range']:
                print(f"Stable range: {result['stable_range'][0]:.0%}-{result['stable_range'][1]:.0%}")
            senses = result['senses']
        else:
            senses = sr.induce_senses(args.word, n_senses=args.n_senses)
        
        print(f"Found {len(senses)} senses: {list(senses.keys())}")
        
        anchors = sr.get_anchors(args.word)
        if anchors:
            print("\nAnchors used:")
            for sense, words in anchors.items():
                print(f"  {sense}: {', '.join(words[:5])}")
    
    # Compute similarity
    if args.similarity:
        w1, w2 = args.similarity
        
        std_sim = sr.similarity(w1, w2, sense_aware=False)
        max_sim, sense = sr.max_sense_similarity(w1, w2)
        
        print(f"\nSimilarity between '{w1}' and '{w2}':")
        print(f"  Standard:    {std_sim:.4f}")
        print(f"  Sense-aware: {max_sim:.4f} (using {sense} sense)")
        print(f"  Improvement: {max_sim - std_sim:+.4f}")
    
    # Solve analogy
    if args.analogy:
        a, b, c = args.analogy
        
        print(f"\nAnalogy: {a} : {b} :: {c} : ?")
        
        results = sr.analogy(a, b, c, top_k=5, sense_aware=True)
        print("Top answers (sense-aware):")
        for word, score in results:
            print(f"  {word}: {score:.4f}")


if __name__ == "__main__":
    main()
