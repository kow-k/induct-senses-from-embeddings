# SenseExplorer

**From Sense Discovery to Sense Induction via Simulated Self-Repair**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.4.0-green.svg)](https://github.com/kow-k/sense-explorer)

A lightweight, training-free framework for exploring word sense structure in static embeddings (GloVe, Word2Vec, FastText).

## Two Operational Modes

| Mode | Method | Supervision | Accuracy | Sense Names |
|------|--------|-------------|----------|-------------|
| **Discovery** | `discover_senses()` | Unsupervised | 56% | `sense_0`, `sense_1`, ... |
| **Induction** | `induce_senses()` | Weakly supervised | 88% | `financial`, `river`, ... |

**Key insight**: Senses can be **discovered** (emerging from distributional structure alone) or **induced** (guided toward anchor-defined targets). Both use the same self-repair mechanism but differ in supervision level.

## Installation

```bash
pip install sense-explorer

# For full functionality (WordNet gloss extraction):
pip install sense-explorer[full]
```

Or from source:

```bash
git clone https://github.com/kow-k/sense-explorer.git
cd sense-explorer
pip install -e .
```

## Quick Start

```python
from sense_explorer import SenseExplorer

# Load embeddings
se = SenseExplorer.from_glove("glove.6B.100d.txt")

# UNSUPERVISED: True sense discovery (56% accuracy)
senses = se.discover_senses("bank", n_senses=2)
print(senses.keys())  # dict_keys(['sense_0', 'sense_1'])

# WEAKLY SUPERVISED: Anchor-guided induction (88% accuracy)
senses = se.induce_senses("bank")
print(senses.keys())  # dict_keys(['financial', 'river'])

# AUTO MODE: Uses induction if anchors available, else discovery
senses = se.explore_senses("bank", mode='auto')
```

## The Two Modes Explained

### Unsupervised Discovery (`discover_senses`)

True sense discovery from distributional data alone:
- No external knowledge required
- Senses emerge via k-means clustering of neighbors
- Generic sense names (`sense_0`, `sense_1`, ...)
- ~56% accuracy

```python
# Purely data-driven - what does the embedding tell us?
senses = se.discover_senses("bank", n_senses=2)
```

### Weakly Supervised Induction (`induce_senses`)

Knowledge-guided sense induction:
- Uses FrameNet frames or WordNet glosses as anchors
- Senses induced toward anchor-defined targets
- Meaningful sense names (`financial`, `river`, ...)
- ~88% accuracy

```python
# Knowledge-guided - we hint at what senses to find
senses = se.induce_senses("bank")

# Or provide custom anchors
senses = se.induce_senses("bank", anchors={
    "financial": ["money", "account", "loan"],
    "river": ["water", "shore", "stream"]
})
```

## Theoretical Background

### DNA Self-Repair Analogy

Both modes use the same biologically-inspired mechanism:

1. **Damage** (noise injection): Perturb the embedding
2. **Repair** (self-organization): Allow copies to settle into stable configurations
3. **Diagnosis** (attractor identification): Observe which "attractor basins" copies settle into

### Discovery vs Induction

| Aspect | Discovery | Induction |
|--------|-----------|-----------|
| **Targets** | Emerge FROM data | Provided TO system |
| **Supervision** | None | Weak (anchors) |
| **Analogy** | "What senses exist?" | "Are these senses present?" |
| **Accuracy** | 56% | 88% |

### Why the Accuracy Difference?

- **Discovery** must find structure without guidance → prone to noise
- **Induction** has anchor targets → more robust to noise

Frame Elements ≈ WordNet Gloss Nouns ≈ Distributional Co-occurrence
All capture "situational participants" — enabling effective weak supervision.

## Noise as Granularity Control

```python
# Fine-grained (may over-split)
se.set_noise_level(0.2)
senses = se.induce_senses("cell")  # May find 4-5 senses

# Standard (recommended)
se.set_noise_level(0.5)
senses = se.induce_senses("cell")  # Typically 2-3 senses

# Coarse-grained
se.set_noise_level(0.7)
senses = se.induce_senses("cell")  # May merge to 2 senses
```

## API Reference

### SenseExplorer

```python
SenseExplorer(
    embeddings,              # Dict[str, np.ndarray]
    dim=None,                # Auto-detected
    default_n_senses=2,      # Default sense count
    n_copies=100,            # Noisy copies for self-repair
    noise_level=0.5,         # Granularity control (0.1-0.8)
    use_hybrid_anchors=True, # Enable hybrid extraction for induction
    verbose=True
)
```

### Key Methods

| Method | Mode | Description |
|--------|------|-------------|
| `discover_senses(word)` | Unsupervised | True sense discovery |
| `induce_senses(word)` | Weakly supervised | Anchor-guided induction |
| `explore_senses(word, mode)` | Auto | Convenience wrapper |
| `induce_senses_stable(word)` | Weakly supervised | Stability-based induction |
| `similarity(w1, w2)` | - | Sense-aware similarity |
| `disambiguate(word, context)` | - | Context-based disambiguation |

## Citation

```bibtex
@article{kuroda2026sense,
  title={From Sense Discovery to Sense Induction via Simulated Self-Repair: 
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
