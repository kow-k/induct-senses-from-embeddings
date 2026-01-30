#!/usr/bin/env python3
"""
Setup script for SenseExplorer package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sense-explorer",
    version="0.6.0",
    author="Kow Kuroda & Claude",
    author_email="kow.k@ks.kyorin-u.ac.jp",
    description="From sense discovery to sense induction via simulated self-repair",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kow-k/sense-explorer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "full": [
            "nltk>=3.6.0",  # For WordNet gloss extraction
        ],
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sense-explorer=sense_explorer.core:main",
        ],
    },
    keywords=[
        "word-embeddings",
        "word-sense-induction",
        "word-sense-discovery",
        "word-sense-disambiguation",
        "polysemy",
        "nlp",
        "distributional-semantics",
        "framenet",
        "wordnet",
    ],
)
