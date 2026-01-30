# SenseInduction

**Sense Induction via Simulated Self-Repair**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.3.0-green.svg)](https://github.com/kow-k/sense-induction)

A lightweight, training-free method for inducing word senses in static embeddings (GloVe, Word2Vec, FastText).

## Key Insight

Senses are not "discovered" but **induced** toward targets defined by anchor words. The anchors act as attractors in semantic space, and the simulated self-repair mechanism reveals which attractors are stable.

## Key Features

- **Zero training required** - Works with any pre-trained static embeddings
- **Biologically inspired** - Uses DNA self-repair analogy for sense induction
- **Automatic anchor extraction** - Hybrid FrameNet + WordNet extraction (88% accuracy)
- **Noise as granularity control** - "Semantic zoom" parameter for fine/coarse senses
- **Stability-based sense selection** - Automatic sense count determination

## What's New in v0.3.0

**Hybrid Anchor Extraction** - Combines multiple strategies for 88% accuracy:

| Strategy | Accuracy | Coverage |
|----------|----------|----------|
| Auto (k-means) | 56% | Unlimited |
| WordNet gloss nouns | 69% | 117K synsets |
| **FrameNet frames** | **88%** | 13K lexical units |
| **Manual anchors** | **88%** | ~10 common words |

Key theoretical insight: **Frame Elements ≈ WordNet Gloss Nouns ≈ Distributional Co-occurrence**

All three capture "situational participants" - the words that appear together in schematic situations.

## Installation

```bash
pip install sense-induction

# For full functionality (WordNet gloss extraction):
pip install sense-induction[full]
```

Or from source:

```bash
git clone https://github.com/kow-k/sense-induction.git
cd sense-induction
pip install -e .
```

## Quick Start

```python
from sense_induction import SenseInductor

# Load embeddings
si = SenseInductor.from_glove("glove.6B.100d.txt")

# Induce senses (hybrid anchor extraction enabled by default)
senses = si.induce_senses("bank")
print(senses.keys())  # dict_keys(['financial', 'river'])

# Sense-aware similarity
sim = si.similarity("bank", "money")  # Uses best-matching sense
print(f"bank-money similarity: {sim:.3f}")

# Max sense similarity with sense info
max_sim, sense = si.max_sense_similarity("bank", "river")
print(f"bank-river: {max_sim:.3f} (via {sense} sense)")
```

## Anchor Extraction Strategies

### Automatic (Default)

By default, SenseInductor uses hybrid anchor extraction:

```python
si = SenseInductor.from_glove("glove.6B.100d.txt", use_hybrid_anchors=True)
senses = si.induce_senses("bank")
# Extraction priority: manual → FrameNet → WordNet → k-means fallback
```

### Manual Anchors

You can provide custom anchors:

```python
si.set_anchors("bank", {
    "financial": ["money", "account", "loan", "deposit"],
    "river": ["river", "water", "shore", "stream"]
})
senses = si.induce_senses("bank")
```

### Using the Anchor Extractor Directly

```python
from sense_induction import HybridAnchorExtractor, extract_anchors

# Create extractor
extractor = HybridAnchorExtractor(vocab_set, verbose=True)

# Extract anchors for a word
anchors, source = extractor.extract("bank", n_senses=2)
print(f"Source: {source}")  # 'manual', 'framenet', 'wordnet', or 'auto'
```

## Noise Level as Granularity Control

Noise level acts as a "semantic zoom" parameter:

```python
# Fine-grained (may over-split)
si.set_noise_level(0.2)
senses_fine = si.induce_senses("cell")  # May find 4-5 senses

# Standard (recommended)
si.set_noise_level(0.5)
senses_standard = si.induce_senses("cell")  # Typically 2-3 senses

# Coarse-grained
si.set_noise_level(0.7)
senses_coarse = si.induce_senses("cell")  # May merge to 2 senses
```

## Stability-Based Sense Induction

Find the "true" sense count that remains stable across noise levels:

```python
result = si.induce_senses_stable("bank")
print(f"Stable sense count: {result['stable_k']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Optimal noise: {result['optimal_noise']:.1%}")
print(f"Senses: {list(result['senses'].keys())}")
```

## Theoretical Background

### DNA Self-Repair Analogy

The method is inspired by DNA self-repair mechanisms:

1. **Damage** (noise injection): Perturb the embedding
2. **Repair** (self-organization): Allow copies to settle into stable configurations
3. **Diagnosis** (attractor identification): Observe which "attractor basins" copies settle into

Key insight: **Stability defines correctness** - senses are stable configurations that resist perturbation.

### Why "Induction" not "Discovery"?

| Term | Implies | Reality |
|------|---------|---------|
| Discovery | Finding unknown things | We need anchors to define targets |
| **Induction** | Inferring toward targets | Anchors define attractor basins |

Senses are **induced** toward anchor-defined targets, not discovered from scratch. This aligns with the established term "Word Sense Induction (WSI)" in computational linguistics.

### Frame Semantics Connection

The success of FrameNet-based anchors (88% accuracy) validates a deep correspondence:

| Frame Semantics | Distributional Semantics |
|-----------------|-------------------------|
| Frame (schematic situation) | Sense attractor basin |
| Frame Elements (roles) | Anchor words |
| Lexical Unit evokes frame | Word activates sense |

## API Reference

### SenseInductor

```python
SenseInductor(
    embeddings,              # Dict[str, np.ndarray]
    dim=None,                # Auto-detected
    default_n_senses=2,      # Default sense count
    n_copies=100,            # Noisy copies for self-repair
    noise_level=0.5,         # Granularity control (0.1-0.8)
    seed_strength=0.3,       # Initial seeding strength
    n_iterations=15,         # Self-organization iterations
    anchor_pull=0.2,         # Anchor attraction strength
    n_anchors=8,             # Anchors per sense
    use_hybrid_anchors=True, # Enable hybrid extraction (recommended)
    verbose=True
)
```

### Key Methods

| Method | Description |
|--------|-------------|
| `induce_senses(word)` | Induce sense-specific embeddings |
| `induce_senses_stable(word)` | Stability-based sense induction |
| `similarity(w1, w2)` | Sense-aware cosine similarity |
| `max_sense_similarity(w1, w2)` | Best sense match with sense info |
| `disambiguate(word, context)` | Context-based disambiguation |
| `set_anchors(word, anchors)` | Set custom anchors |
| `set_noise_level(level)` | Adjust granularity |
| `get_anchors(word)` | Get anchors used for word |

## Performance

On 8 classic polysemous words (bank, bat, crane, mouse, plant, bass, spring, cell):

| Method | Accuracy |
|--------|----------|
| Auto (k-means neighbors) | 56% |
| WordNet gloss nouns | 69% |
| **FrameNet frames** | **88%** |
| **Manual anchors** | **88%** |

## Citation

If you use SenseInduction in your research, please cite:

```bibtex
@article{kuroda2026sense,
  title={Sense Induction via Simulated Self-Repair: 
         Revealing Latent Semantic Attractors in Word Embeddings},
  author={Kuroda, Kow and Claude},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Authors

- **Kow Kuroda** - Kyorin University Medical School
- **Claude** - Anthropic (AI Research Assistant)
