#!/usr/bin/env python3
"""
Test script for Embedding Merger with SenseExplorer
====================================================

This script demonstrates the embedding merger functionality.
Run with actual GloVe embeddings to see cross-embedding sense alignment.

Usage:
    python test_merger.py --wiki path/to/wiki.txt --twitter path/to/twitter.txt --words bank rock plant

Author: Kow Kuroda & Claude (Anthropic)
"""

import argparse
import sys
import os

# Add parent directory if running from sense_explorer folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_with_toy_data():
    """Test merger with synthetic data."""
    import numpy as np
    from merger import EmbeddingMerger, SenseComponent
    
    print("=" * 60)
    print("TEST: Embedding Merger with Toy Data")
    print("=" * 60)
    
    np.random.seed(42)
    dim = 50
    
    # Create synthetic embeddings with known structure
    # Embedding A: "bank" has financial and river senses
    # Embedding B: "bank" has financial and slang senses
    
    # Shared "financial" direction
    financial_dir = np.random.randn(dim)
    financial_dir /= np.linalg.norm(financial_dir)
    
    # Source-specific directions
    river_dir = np.random.randn(dim)
    river_dir /= np.linalg.norm(river_dir)
    
    slang_dir = np.random.randn(dim)
    slang_dir /= np.linalg.norm(slang_dir)
    
    # Create vocabulary
    def make_embedding(base_words, directions, n_words=100):
        vectors = {}
        for i, (words, direction) in enumerate(zip(base_words, directions)):
            for word in words:
                noise = np.random.randn(dim) * 0.1
                vectors[word] = direction + noise
                vectors[word] /= np.linalg.norm(vectors[word])
        
        # Add random filler words
        for i in range(n_words - len(vectors)):
            w = f"word_{i}"
            vectors[w] = np.random.randn(dim)
            vectors[w] /= np.linalg.norm(vectors[w])
        
        return vectors
    
    # Embedding A (wiki-like): financial + river
    financial_words_a = ["money", "loan", "credit", "finance", "investment", "banking"]
    river_words_a = ["river", "stream", "shore", "water", "bank"]
    emb_a = make_embedding([financial_words_a, river_words_a], [financial_dir, river_dir])
    
    # Embedding B (twitter-like): financial + slang
    financial_words_b = ["money", "cash", "pay", "credit", "bank", "broke"]
    slang_words_b = ["lit", "fire", "dope", "sick", "bank"]  # "bank" as slang for money
    emb_b = make_embedding([financial_words_b, slang_words_b], [financial_dir, slang_dir])
    
    print(f"\nEmbedding A: {len(emb_a)} words (wiki-like: financial + river)")
    print(f"Embedding B: {len(emb_b)} words (twitter-like: financial + slang)")
    
    # Create merger
    merger = EmbeddingMerger(verbose=True)
    merger.add_embedding("wiki", emb_a)
    merger.add_embedding("twitter", emb_b)
    
    print(f"\nShared vocabulary: {len(merger.shared_vocabulary)} words")
    
    # Create sense components manually (simulating SSR output)
    senses = [
        SenseComponent(
            word="bank", sense_id="wiki_bank_financial", 
            vector=financial_dir + np.random.randn(dim) * 0.05,
            source="wiki",
            top_neighbors=[("money", 0.9), ("loan", 0.85), ("credit", 0.8)]
        ),
        SenseComponent(
            word="bank", sense_id="wiki_bank_river",
            vector=river_dir + np.random.randn(dim) * 0.05,
            source="wiki",
            top_neighbors=[("river", 0.9), ("stream", 0.85), ("shore", 0.8)]
        ),
        SenseComponent(
            word="bank", sense_id="twitter_bank_financial",
            vector=financial_dir + np.random.randn(dim) * 0.05,
            source="twitter",
            top_neighbors=[("money", 0.9), ("cash", 0.85), ("pay", 0.8)]
        ),
        SenseComponent(
            word="bank", sense_id="twitter_bank_slang",
            vector=slang_dir + np.random.randn(dim) * 0.05,
            source="twitter",
            top_neighbors=[("lit", 0.9), ("fire", 0.85), ("dope", 0.8)]
        ),
    ]
    
    # Run merger
    print("\n" + "-" * 60)
    print("Running merger...")
    result = merger.merge_senses("bank", sense_components=senses, distance_threshold=0.3)
    
    print("\n" + merger.report(result))
    
    # Expected: financial senses should converge, river and slang should be source-specific
    print("\n" + "-" * 60)
    print("EXPECTED OUTCOME:")
    print("  - wiki_bank_financial + twitter_bank_financial → CONVERGENT")
    print("  - wiki_bank_river → SOURCE-SPECIFIC (wiki only)")
    print("  - twitter_bank_slang → SOURCE-SPECIFIC (twitter only)")
    print("-" * 60)
    
    print(f"\nActual: {result.n_convergent} convergent, {result.n_source_specific} source-specific")
    
    if result.n_convergent >= 1 and result.n_source_specific >= 1:
        print("✓ Test PASSED: Found both convergent and source-specific senses")
    else:
        print("✗ Test needs review")
    
    return result


def test_with_real_embeddings(wiki_path, twitter_path, words):
    """Test merger with real GloVe embeddings."""
    from merger import EmbeddingMerger
    
    print("=" * 60)
    print("TEST: Embedding Merger with Real Embeddings")
    print("=" * 60)
    
    # Try to import SenseExplorer for SSR-based extraction
    try:
        from core import SenseExplorer
        use_ssr = True
        print("Using SenseExplorer SSR for sense extraction")
    except ImportError:
        use_ssr = False
        print("SenseExplorer not found, using simple k-means extraction")
    
    # Load embeddings
    print(f"\nLoading Wikipedia embeddings from {wiki_path}...")
    if use_ssr:
        se_wiki = SenseExplorer.from_file(wiki_path)
        wiki_vectors = se_wiki.embeddings
    else:
        wiki_vectors = load_glove_simple(wiki_path)
    print(f"  Loaded {len(wiki_vectors)} words")
    
    print(f"\nLoading Twitter embeddings from {twitter_path}...")
    if use_ssr:
        se_twitter = SenseExplorer.from_file(twitter_path)
        twitter_vectors = se_twitter.embeddings
    else:
        twitter_vectors = load_glove_simple(twitter_path)
    print(f"  Loaded {len(twitter_vectors)} words")
    
    # Create merger
    merger = EmbeddingMerger(verbose=True)
    merger.add_embedding("wiki", wiki_vectors)
    merger.add_embedding("twitter", twitter_vectors)
    
    print(f"\nShared vocabulary: {len(merger.shared_vocabulary)} words")
    
    # Process each word
    all_results = {}
    
    for word in words:
        print("\n" + "=" * 60)
        print(f"WORD: {word}")
        print("=" * 60)
        
        # Check availability
        if word not in wiki_vectors:
            print(f"  '{word}' not in Wikipedia embedding, skipping")
            continue
        if word not in twitter_vectors:
            print(f"  '{word}' not in Twitter embedding, skipping")
            continue
        
        # Merge with multiple thresholds
        results = merger.merge_senses(word, n_senses=3, return_all_thresholds=True)
        
        # Print threshold sensitivity
        print("\nThreshold sensitivity:")
        print(f"  {'Thresh':<8} {'Clusters':<10} {'Convergent':<12} {'Source-Spec':<12}")
        print("  " + "-" * 42)
        
        for thresh, res in sorted(results.items()):
            print(f"  {thresh:<8.2f} {res.n_clusters:<10} {res.n_convergent:<12} {res.n_source_specific:<12}")
        
        # Detailed report for threshold 0.05
        best = results[0.05]
        print("\n" + merger.report(best))
        
        all_results[word] = results
    
    return all_results


def load_glove_simple(path):
    """Simple GloVe loader (fallback if SenseExplorer not available)."""
    import gzip
    
    vectors = {}
    
    open_fn = gzip.open if path.endswith('.gz') else open
    mode = 'rt' if path.endswith('.gz') else 'r'
    
    with open_fn(path, mode, encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 10:  # Skip header lines
                continue
            word = parts[0]
            try:
                vec = np.array([float(x) for x in parts[1:]])
                vec /= np.linalg.norm(vec) + 1e-10
                vectors[word] = vec
            except ValueError:
                continue
            
            if (i + 1) % 100000 == 0:
                print(f"    Loaded {i + 1} words...")
    
    return vectors


def main():
    parser = argparse.ArgumentParser(description="Test Embedding Merger")
    parser.add_argument("--wiki", type=str, help="Path to Wikipedia GloVe embeddings")
    parser.add_argument("--twitter", type=str, help="Path to Twitter GloVe embeddings")
    parser.add_argument("--words", type=str, nargs="+", 
                        default=["bank", "rock", "plant", "spring", "star"],
                        help="Words to test")
    parser.add_argument("--toy", action="store_true", help="Run toy data test only")
    
    args = parser.parse_args()
    
    # Always run toy test first
    print("\n" + "#" * 70)
    print("# PART 1: TOY DATA TEST")
    print("#" * 70)
    test_with_toy_data()
    
    # Run real embedding test if paths provided
    if args.wiki and args.twitter and not args.toy:
        print("\n" + "#" * 70)
        print("# PART 2: REAL EMBEDDING TEST")
        print("#" * 70)
        
        import numpy as np  # Needed for load_glove_simple
        test_with_real_embeddings(args.wiki, args.twitter, args.words)
    elif not args.toy:
        print("\n" + "-" * 60)
        print("To test with real embeddings, provide --wiki and --twitter paths:")
        print("  python test_merger.py --wiki glove-wiki-100d.txt --twitter glove-twitter-100d.txt")
        print("-" * 60)


if __name__ == "__main__":
    main()
