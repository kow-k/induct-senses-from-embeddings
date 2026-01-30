#!/usr/bin/env python3
"""
Diagnostic script for sense induction issue.
"""

from sense_explorer import SenseExplorer, MANUAL_ANCHORS
import numpy as np

# Initialize
se = SenseExplorer.from_glove("glove.6B.100d.txt")

# Check centroid similarity
word = "bank"
anchors = MANUAL_ANCHORS.get(word, {})

print("=" * 60)
print("DIAGNOSTIC: Why is 'river' sense disappearing?")
print("=" * 60)

# Compute centroids
centroids = {}
for sense, words in anchors.items():
    vecs = [se.embeddings[w] for w in words if w in se.vocab]
    centroid = np.mean(vecs, axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
    centroids[sense] = centroid
    print(f"\n{sense} centroid computed from {len(vecs)} words")

# Check similarity between centroids
if len(centroids) >= 2:
    senses = list(centroids.keys())
    sim = np.dot(centroids[senses[0]], centroids[senses[1]])
    print(f"\nSimilarity between '{senses[0]}' and '{senses[1]}' centroids: {sim:.4f}")
    
# Check word's similarity to each centroid
word_emb = se.embeddings[word]
word_emb_norm = word_emb / (np.linalg.norm(word_emb) + 1e-10)

print(f"\n'{word}' similarity to each centroid:")
for sense, centroid in centroids.items():
    sim = np.dot(word_emb_norm, centroid)
    print(f"  {sense}: {sim:.4f}")

# The issue: if 'bank' is much closer to 'financial', 
# copies will converge there during self-repair

print("\n" + "=" * 60)
print("EXPLANATION:")
print("=" * 60)
print("""
If 'bank' embedding is much closer to 'financial' centroid than 'river':
  - All copies start near 'bank'
  - During self-repair, they're pulled toward nearest centroid
  - Since 'bank' is closer to 'financial', most copies go there
  - Copies assigned to 'river' drop below 5% threshold
  - Only 'financial' sense survives!
  
FIX OPTIONS:
1. Lower the min_copies threshold
2. Use sense-specific anchoring (don't let copies switch senses)
3. Use separate self-repair runs per sense
""")
