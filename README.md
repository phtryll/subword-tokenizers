# Subword Tokenizers

A collection of naive and optimized implementations of Byte-Pair Encoding (BPE) and WordPiece tokenizers, plus a comprehensive benchmarking suite to evaluate their quality and performance.

## Table of Contents

- [Subword Tokenizers](#subword-tokenizers)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Training](#training)
    - [Tokenizing](#tokenizing)
    - [Benchmarking](#benchmarking)
    - [Python API Tutorial](#python-api-tutorial)
  - [CLI Reference](#cli-reference)
  - [Project Structure](#project-structure)

## Features

- **Naive BPE & Fast-BPE**  
  – Learn subword vocabularies by byte-pair merges; optimized version uses a trie for speed.  
- **Naive WordPiece & Fast-WordPiece**  
  – Implements Google’s WordPiece algorithm; fast variant uses Aho–Corasick trie for linear-time tokenization.  
- **Benchmarking Suite**  
  – Measures tokenization quality (fragmentation, compression, Zipf fit), speed & memory, training performance.  
- **Flexible CLI**  
  – Train, tokenize, and benchmark single or multiple models via simple flags.  
- **Python API**  
  – Import tokenizers and benchmarks directly in your own scripts.  

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/phtryll/subword-tokenizers.git
cd subword-tokenizers
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Training

Train one or more models on a corpus of sentences.

```bash
python cli.py --model NaiveBPE --train data/train.txt --max-vocab 1000
```

### Tokenizing

Tokenize a single sentence with one or multiple models:

```bash
python cli.py --model NaiveBPE FastBPE --tokenize "This is a test sentence."
```

Or tokenize all sentences in a file:

```bash
python cli.py --model WordPiece --tokenize data/raw/input.txt
```

You can also train before tokenization:

```bash
python cli.py --model NaiveBPE FastBPE --train data/train.txt --tokenize "This is a test sentence."
```

### Benchmarking

Run full benchmarks comparing models on test and train data:

```bash
python cli.py --model FastBPE WordPiece --benchmark data/raw/test.txt data/raw/train.txt --max-vocab 1000
```

### Python API Tutorial

You can also use the tokenizers directly from Python:

```python
from transformers import AutoTokenizer
from source.bpe import NaiveBPE
from source.wordpiece import FastWordPiece
from source.benchmarks import benchmarks

# Initialize HuggingFace tokenizer
hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Instantiate tokenizer models
naive_bpe = NaiveBPE(hf_tokenizer)
fast_wp = FastWordPiece(hf_tokenizer)

# Training
corpus = ["This is a sentence.", "Another example sentence."]
naive_bpe.train(corpus, max_vocab_size=1000)
fast_wp.train(corpus, max_vocab_size=1000)

# Tokenization
test_text = "Hello, world!"
print("NaiveBPE:", naive_bpe.tokenize(test_text))
print("FastWordPiece:", fast_wp.tokenize(test_text))

# Benchmarking
benchmarks(
    naive_bpe,
    [test_text],
    max_vocab_size=1000,
    train_corpus=corpus,
    reference_tokenizers=[fast_wp]
)
```

## CLI Reference

| Flag                   | Args                         | Default               | Description                                                                                                  |
|------------------------|------------------------------|-----------------------|--------------------------------------------------------------------------------------------------------------|
| `-h`, `--help`         |                              |                       | Show this help message and exit                                                                             |
| `--model`              | `MODEL1 [MODEL2 ...]`        | _required_            | Select primary tokenizer model (required) and optional second model for comparison: NaiveBPE, NaiveWordPiece, FastBPE, FastWordPiece |
| `--normalize_with`     | `HF_TOKENIZER`               | `bert-base-uncased`   | Select HuggingFace tokenizer to use for normalization                                                       |
| `--train`              | `TRAIN_DATA`                 |                       | Path to `.txt` file used for training (required to enable training)                                        |
| `--pretrained`         |                              |                       | NOT IMPLEMENTED — load pretrained merges and vocabulary from resources (skip training)                      |
| `--tokenize`           | `TEST_DATA`                  |                       | String to tokenize or path to `.txt` file for tokenization                                                 |
| `--max_vocab`          | `INTEGER`                    | `1000`                | Maximum vocabulary size for training                                                                        |
| `--benchmark`          | `TEST_INPUT TRAIN_INPUT`     |                       | Benchmark models: TEST_INPUT (string or .txt file) and TRAIN_INPUT (.txt file path)                        |

## Project Structure

```plaintext
.
├── data/
│   └── train.txt             # Dummy training corpus
├── resources/
│   ├── bpe/                  # Pretrained BPE vocabulary & merges
│   └── wordpiece/            # Pretrained WordPiece vocabulary & merges
├── source/
│   ├── utils.py              # Parent classes & trie implementations
│   ├── bpe.py                # Naive & optimized BPE implementations
│   ├── wordpiece.py          # Naive & fast WordPiece implementations
│   └── benchmarks.py         # Quality & performance metrics
├── cli.py                    # Main CLI entry point
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```
