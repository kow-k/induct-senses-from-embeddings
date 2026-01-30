# SenseExplorer

**From Sense Discovery to Sense Induction via Simulated Self-Repair**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.6.0-green.svg)](https://github.com/kow-k/sense-explorer)

A lightweight, training-free framework for exploring word sense structure in static embeddings (GloVe, Word2Vec, FastText).

## Three Capabilities

| Capability | Method | Supervision | Accuracy |
|------------|--------|-------------|----------|
| **Sense Discovery** | `discover_senses()` | Unsupervised | 56% |
| **Sense Discovery (Auto)** | `discover_senses_auto()` | Unsupervised + Parameter-free | 56% |
| **Sense Induction** | `induce_senses()` | Weakly supervised | 88% |
| **Polarity Classification** | `get_polarity()` | Supervised | 97% |

**Key insight**: 
- Senses can be **discovered** (from data alone) or **induced** (anchor-guided)
- **Semantic categories** self-organize; **polarity** within categories requires supervision
- **X-means** automatically determines optimal sense count (no n_senses needed!)

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

# UNSUPERVISED: Sense discovery with specified n_senses
senses = se.discover_senses("bank", n_senses=2)
print(senses.keys())  # dict_keys(['sense_0', 'sense_1'])

# PARAMETER-FREE: X-means auto-discovers optimal sense count
senses = se.discover_senses_auto("bank")  # No n_senses needed!
print(f"Found {len(senses)} senses")  # Automatically determined

# WEAKLY SUPERVISED: Anchor-guided induction (88% accuracy)
senses = se.induce_senses("bank")
print(senses.keys())  # dict_keys(['financial', 'river'])

# SUPERVISED: Polarity classification (97% accuracy)
polarity = se.get_polarity("excellent")
print(polarity)  # {'polarity': 'positive', 'score': 0.82, ...}

# AUTO MODE: Uses best available method
senses = se.explore_senses("bank", mode='auto')
```

## Polarity Classification (97% Accuracy)

Detect positive/negative valence within semantic categories:

```python
# Simple polarity check
polarity = se.get_polarity("wonderful")
# {'polarity': 'positive', 'score': 0.78, 'confidence': 0.92}

# Classify multiple words
result = se.classify_polarity(['good', 'bad', 'happy', 'sad', 'table'])
# {'positive': ['good', 'happy'], 'negative': ['bad', 'sad'], 'neutral': ['table']}

# Use domain-specific polarity (quality, morality, health, etc.)
pf = se.get_polarity_finder(domain='quality')
pf.get_polarity("excellent")  # Uses quality-specific seeds

# Advanced: PolarityFinder directly
from sense_explorer import PolarityFinder
pf = PolarityFinder(se.embeddings)
pf.most_polar_words(top_k=10)
pf.find_polar_opposites("happy")
```

### Available Polarity Domains

| Domain | Positive Pole | Negative Pole |
|--------|---------------|---------------|
| `sentiment` | happy, joy, love | sad, angry, hate |
| `quality` | excellent, superior | poor, inferior |
| `morality` | good, virtuous | evil, wicked |
| `health` | healthy, strong | sick, weak |
| `size` | big, large, huge | small, tiny, little |
| `temperature` | hot, warm | cold, freezing |

## The Three Modes Explained

### 1. Unsupervised Discovery (`discover_senses`)

True sense discovery from distributional data alone:
- No external knowledge required
- Senses emerge via k-means clustering of neighbors
- Generic sense names (`sense_0`, `sense_1`, ...)
- ~56% accuracy

```python
senses = se.discover_senses("bank", n_senses=2)
```

### 1b. Parameter-Free Discovery (`discover_senses_auto`)

**NEW in v0.6.0**: X-means clustering for automatic sense count:
- No need to specify `n_senses`!
- Uses BIC (Bayesian Information Criterion) to find optimal k
- Truly parameter-free unsupervised discovery

```python
# No n_senses needed - X-means finds optimal count automatically
senses = se.discover_senses_auto("bank")
print(f"Found {len(senses)} senses")  # Automatically determined!

# Or use explore_senses with mode='discover_auto'
senses = se.explore_senses("bank", mode='discover_auto')
```

### 2. Weakly Supervised Induction (`induce_senses`)

Knowledge-guided sense induction:
- Uses FrameNet frames or WordNet glosses as anchors
- Senses induced toward anchor-defined targets
- Meaningful sense names (`financial`, `river`, ...)
- ~88% accuracy

```python
senses = se.induce_senses("bank")

# Or provide custom anchors
senses = se.induce_senses("bank", anchors={
    "financial": ["money", "account", "loan"],
    "river": ["water", "shore", "stream"]
})
```

### 3. Supervised Polarity (`get_polarity`)

Polarity classification with seed supervision:
- Requires positive/negative seed words
- Projects words onto polarity axis
- Binary classification with confidence
- ~97% accuracy

```python
polarity = se.get_polarity("terrible")
# {'polarity': 'negative', 'score': -0.71, 'confidence': 0.88}
```

## Theoretical Background

### The Supervision Spectrum

```
Fully Unsupervised    Weakly Supervised    Fully Supervised
       │                     │                    │
  discover_senses()    induce_senses()     get_polarity()
  discover_senses_auto()                        
       │                     │                    │
   No targets          Anchor targets        Seed labels
   56% accuracy        88% accuracy          97% accuracy
       │
  X-means: auto k
  (parameter-free!)
```

### Why Polarity Needs Supervision

- **Semantic categories** (bank-financial vs bank-river) self-organize
- **Polarity** (good vs bad within a category) doesn't self-organize
- Polarity requires **contrast**: we must tell the system what "positive" means

### DNA Self-Repair Analogy

Both sense discovery and induction use the same mechanism:
1. **Damage** (noise injection): Perturb the embedding
2. **Repair** (self-organization): Settle into stable configurations
3. **Diagnosis** (attractor identification): Observe attractor basins

## API Reference

### SenseExplorer

```python
SenseExplorer(
    embeddings,              # Dict[str, np.ndarray]
    dim=None,                # Auto-detected
    default_n_senses=2,      # Default sense count
    noise_level=0.5,         # Granularity control (0.1-0.8)
    use_hybrid_anchors=True, # Enable hybrid extraction for induction
    verbose=True
)
```

### Key Methods

| Method | Mode | Description |
|--------|------|-------------|
| `discover_senses(word, n_senses)` | Unsupervised | Sense discovery (k-means) |
| `discover_senses_auto(word)` | Unsupervised | Parameter-free discovery (X-means) |
| `induce_senses(word)` | Weakly supervised | Anchor-guided induction |
| `explore_senses(word, mode)` | Auto | Convenience wrapper |
| `get_polarity(word)` | Supervised | Polarity classification |
| `classify_polarity(words)` | Supervised | Batch polarity |
| `get_polarity_finder(domain)` | Supervised | Advanced polarity ops |
| `similarity(w1, w2)` | - | Sense-aware similarity |
| `disambiguate(word, context)` | - | Context-based disambiguation |

### PolarityFinder

```python
from sense_explorer import PolarityFinder

pf = PolarityFinder(embeddings, positive_seeds, negative_seeds)
pf.get_polarity(word)           # Single word
pf.classify_words(words)        # Multiple words
pf.find_polar_opposites(word)   # Find antonyms
pf.most_polar_words(top_k=20)   # Extreme words
pf.set_domain('quality')        # Switch domain
pf.evaluate_accuracy(pos, neg)  # Test accuracy
```

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
