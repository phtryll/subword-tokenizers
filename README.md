# Subword Tokenizers

A collection of naive and optimized implementations of Byte Pair Encoding (BPE) and WordPiece tokenizers, along with a benchmarking suite to evaluate their performance, efficiency, and tokenization quality.

## Overview

This project includes several subword tokenizers that implement popular algorithms used in natural language processing:

- **Byte Pair Encoding (BPE):** Both naive and optimized implementations for learning subword vocabularies based on the most frequent pairs of bytes or characters.
- **WordPiece:** A tokenizer inspired by Google's WordPiece algorithm, commonly used in models like BERT.
- **Benchmarking Suite:** Tools to evaluate the speed, memory usage, and quality of tokenization produced by each implementation.

## Project structure

- `custom_tokenizers.py` — Implementations of the tokenizers.
- `custom_benchmarks.py` — Benchmarking and evaluation scripts.
- `main.py` — Command-line interface for training, tokenizing, and benchmarking.
- `testing.py` — This file is a temporary testing file and should be removed once the project is ready.
- `data/`
  - `raw/` — Directory for storing raw input text files (`.txt`).
  - `processed/` — Directory for storing processed data and outputs.
  - `models/` — Directory for saving trained tokenizer models.

## Installation

To set up the project, you can install the required dependencies with:

```bash
pip install -r requirements.txt
```

Alternatively, if you only want the core libraries, you can install:

```bash
pip install transformers datasets
```

## Usage

The main interface is provided by `main.py`, which supports several command-line options:

- `--models` — Specify which tokenizer models to use or train.
- `--train` — Train a tokenizer model on the provided data.
- `--max-vocab` — Set the maximum vocabulary size for training.
- `--train-data` — Path to the training data file (usually in `data/raw/`).
- `--tokenize` — Tokenize input text using a trained model.
- `--merges` — Specify the number of merge operations for BPE.

### Examples

- **Training a tokenizer:**

```bash
python main.py --train --models NaiveBPE --max-vocab 10000 --train-data data/raw/sample.txt
```

- **Tokenizing text:**

```bash
python main.py --tokenize --models TrieBPE --train-data data/raw/sample.txt
```

## Data

Place your raw text files (`.txt`) in the `data/raw/` directory. Processed outputs, including tokenized text and trained models, will be saved under `data/processed/` and `data/models/` respectively. This structure helps keep raw data separate from generated artifacts.
