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
    - [Resetting and Saving](#resetting-and-saving)
    - [Python API Tutorial](#python-api-tutorial)
  - [CLI Reference](#cli-reference)
  - [Project Structure](#project-structure)

## Features

- **Byte-Pair Encoding (BPE)**  
  Includes both naive and optimized BPE implementations. The fast variant uses a ranking map for efficient subword merging.

- **WordPiece Tokenizer**  
  Implements both standard and fast WordPiece tokenization. The fast version supports linear-time tokenization using a trie with failure links.

- **Benchmarking Suite**  
  Analyze tokenization quality (fragmentation, compression rate, Zipf distribution), vocabulary coverage, and model equivalence. Also measures training speed and efficiency.

- **Flexible Command-Line Interface (CLI)**  
  Train, tokenize, benchmark, compare, and save models using intuitive CLI flags. Supports `.json` input data and pretrained resource loading.

- **Python Code**  
  Use tokenizers and benchmarking utilities directly in Python.

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

Train one or more models on a corpus of sentences (expects a `.json` file containing a list of sentences):

```bash
python cli.py --model NaiveBPE --train data/train.json --max-vocab 1000
```

Save trained model merges/vocab for future use:

```bash
python cli.py --model NaiveBPE --train data/train.json --save my_merges_dir
```

### Tokenizing

Tokenize a single sentence with one or multiple models:

```bash
python cli.py --model NaiveBPE FastBPE --train data/train.json --tokenize "This is a test sentence."
```

Or tokenize all sentences in a `.json` file (list of strings):

```bash
python cli.py --model FastBPE --train data/train.json --tokenize data/test.json
```

Tokenize using a pretrained model (no training required):

```bash
python cli.py --model FastBPE --pretrained my_merges_dir --tokenize data/test.json
```

### Benchmarking

Benchmark a pretrained model on a test sentence:

```bash
python cli.py --model NaiveWordPiece --pretrained my_vocab_dir --benchmark "This is a test."
```

Benchmark a pretrained model on a `.json` list of sentences:

```bash
python cli.py --model NaiveWordPiece --pretrained my_vocab_dir --benchmark data/test.json
```

Compare multiple pretrained models:

```bash
python cli.py --model NaiveBPE FastBPE --pretrained my_merges_dir --benchmark data/test.json
```

Compare sequence equivalence between pretrained models:

```bash
python cli.py --model NaiveBPE FastBPE --pretrained my_merges_dir --benchmark data/test.json --compare
```

Benchmark training time using a `.json` file and save:

```bash
python cli.py --model NaiveBPE FastBPE --benchmark data/train.json --save my_testing_dir
```

### Resetting and Saving

Reset a model's saved resources:

```bash
python cli.py --model NaiveBPE --reset my_merges_dir
```

### Python API Tutorial

You can use the tokenizers directly in Python to train, save, load, and run tokenization:

```python
from transformers import AutoTokenizer
from source.bpe import NaiveBPE, FastBPE
from source.wordpiece import NaiveWP, FastWP
from source.benchmarks import benchmarks

# 1. Initialize HuggingFace tokenizer for normalization
hf_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# 2. Instantiate tokenizer implementations
naive_bpe = NaiveBPE(hf_tokenizer)
fast_bpe = FastBPE(hf_tokenizer)
naive_wp = NaiveWP(hf_tokenizer)
fast_wp = FastWP(hf_tokenizer)

# 3. Train tokenizers on a small corpus
corpus = ["This is a sentence.", "Another example sentence."]
naive_bpe.train(corpus, max_vocab=25)
fast_bpe.train(corpus, max_vocab=25)
naive_wp.train(corpus, max_vocab=25)
fast_wp.train(corpus, max_vocab=25)

# 4. Save trained models to resources/tests
naive_bpe.save_resources("resources/tests/NaiveBPE")
fast_bpe.save_resources("resources/tests/FastBPE")
naive_wp.save_resources("resources/tests/NaiveWP")
fast_wp.save_resources("resources/tests/FastWP")

# 5. Load pretrained models for inference
loaded_naive_bpe = NaiveBPE(hf_tokenizer)
loaded_naive_bpe.load_resources("resources/tests/NaiveBPE")
loaded_fast_bpe = FastBPE(hf_tokenizer)
loaded_fast_bpe.load_resources("resources/tests/FastBPE")
loaded_naive_wp = NaiveWP(hf_tokenizer)
loaded_naive_wp.load_resources("resources/tests/NaiveWP")
loaded_fast_wp = FastWP(hf_tokenizer)
loaded_fast_wp.load_resources("resources/tests/FastWP")

# 6. Tokenize single sentences
text = "This sentence is a new example."
print("NaiveBPE tokens:       ", loaded_naive_bpe.tokenize(text))
print("FastBPE tokens:        ", loaded_fast_bpe.tokenize(text))
print("NaiveWordPiece tokens: ", loaded_naive_wp.tokenize(text))
print("FastWordPiece tokens:  ", loaded_fast_wp.tokenize(text))

# 7. Benchmark tokenizers against each other (tokenization)
benchmarks(
    loaded_naive_bpe,
    max_vocab_size=25,
    test_corpus=[text],
    pretrained=True,
    reference_tokenizers=[loaded_fast_bpe, loaded_naive_wp, loaded_fast_wp]
)

# 8. Benchmark tokenizers against each other (training)
benchmarks(
    loaded_naive_bpe,
    max_vocab_size=25,
    test_corpus=[text],
    train_corpus=corpus,
    reference_tokenizers=[loaded_fast_bpe, loaded_naive_wp, loaded_fast_wp]
)
```

## CLI Reference

| Flag                   | Args                         | Default               | Description                                                                                                  |
|------------------------|------------------------------|-----------------------|--------------------------------------------------------------------------------------------------------------|
| `-h`, `--help`         |                              |                       | Show this help message and exit                                                                              |
| `--model`              | `MODEL1 [MODEL2 ...]`        | _required_            | Select primary tokenizer model (required) and optional other models for comparison: NaiveBPE, NaiveWordPiece, FastBPE, FastWordPiece |
| `--normalize_with`     | `HF_TOKENIZER`               | `bert-base-uncased`   | Select HuggingFace tokenizer to use for normalization                                                       |
| `--train`              | `TRAIN_DATA`                 |                       | Path to `.json` file (list of sentences) used for training (required to enable training)                    |
| `--save`               | `PATH`                       |                       | Save trained merges/vocab to `resources/PATH/MODEL` for later use                                           |
| `--pretrained`         | `PATH`                       |                       | Load pretrained merges and vocabulary from `resources/PATH/MODEL` (skip training)                           |
| `--reset`              | `PATH`                       |                       | Reset merges/vocabulary for selected models by deleting `resources/PATH/MODEL`                              |
| `--tokenize`           | `TEST_DATA`                  |                       | String to tokenize or path to `.json` file (list of sentences) for tokenization                             |
| `--max_vocab`          | `INTEGER`                    | `1000`                | Maximum vocabulary size for training                                                                        |
| `--benchmark`          | `INPUT`                      |                       | Benchmark the selected tokenizer(s):<br>– With `--pretrained`, INPUT is string or `.json` test data (tokenization benchmarking)<br>– Without `--pretrained`, INPUT is a `.json` file for training benchmarking |
| `--compare`            |                              |                       | With `--pretrained`, only run token-sequence equivalence between models                                     |

## Project Structure

```plaintext
.
├── data/                         # Training and test datasets
│   ├── pan_tadeusz.json          # Tokenization testing data
│   ├── pan_tadeusz.tokens.json   # Tokenization benchmarking results
│   ├── train-5k.json             # Training dataset: 5k examples
│   └── train-85k.json            # Training dataset: full 85k examples
├── resources/                    # Directory to store pretrained tokenizer models
│   ├── pretrained/               # Pretrained tokenizer vocab/merges on the total dataset with 20k vocab length
│   │   ├── FastBPE/
│   │   ├── FastWordPiece/
│   │   ├── NaiveBPE/
│   │   └── NaiveWordPiece/
│   └── tests/                    # Directory to store user pretrained models (ex: run README python API tutorial)
├── source/                       # Tokenizer implementations and benchmarks
│   ├── benchmarks.py             # Python file with the benchmarking suite
│   ├── bpe.py                    # Python file with the BPE implementations
│   ├── data.py                   # Python file for downloading the training corpus
│   ├── utils.py                  # A collection of helpful classes such as Trie and SubwordTokenizer
│   └── wordpiece.py              # Python file containing the WordPiece implementations
├── cli.py                        # Python file with the CLI commands; call this in the terminal to run the tokenizers
├── requirements.txt              # Required libraries for this project
└── README.md                     # This file
```
