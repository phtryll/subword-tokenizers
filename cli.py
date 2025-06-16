import os
import argparse
from functools import partial
from argparse import RawTextHelpFormatter
from transformers import AutoTokenizer
from source.utils import *
from source.bpe import *
from source.wordpiece import *


# Small function to transform HF dataset to List[str]
def build_toy_data(dataset, num_examples, feature_name):
        toy_data = []
        for example in dataset:
            value = example.get(feature_name)
            if value is not None:
                toy_data.append(value)
                if len(toy_data) == num_examples:
                    break
        return toy_data


# Cleaner help display
MyFormatter = partial(RawTextHelpFormatter, max_help_position=60, width=100)

# Available tokenizers
TOKENIZERS = {
    "NaiveBPE": NaiveBPE,
    "NaiveWordPiece": NaiveWP,
    "FastBPE": FastBPE,
    "FastWordPiece": FastWP
}

# Defines the CLI
def main():
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description=(
            "Subword Tokenizers CLI\n\n"
            "A command-line tool to train and/or tokenize text using various subword tokenizers.\n"
        ),
        formatter_class=MyFormatter,
        epilog=(
            "Usage examples:\n"
            "Train models:\n"
            "\tpython cli.py --models NaiveBPE FastBPE --train --max-vocab 5000 --train-data data/train.txt\n\n"
            "Tokenize a single string:\n"
            "\tpython cli.py --models FastBPE --tokenize \"Hello world.\"\n\n"
            "Batch tokenize from file:\n"
            "\tpython cli.py --models FastWordPiece --tokenize data/test.txt\n\n\n"
        )
    )
    
    # Selecting a model
    parser.add_argument(
        "--model",
        choices=TOKENIZERS,
        required=True,
        help=(f"select tokenizer to use from: {', '.join(TOKENIZERS.keys())}")
    )

    # Select normalization model
    parser.add_argument(
        "--normalize_with",
        type=str,
        metavar="HF_TOKENIZER",
        default="bert-base-uncased",
        help=("select HuggingFace tokenizer to use for normalization")
    )

    # Training
    parser.add_argument(
        "--train",
        type=str,
        metavar="TRAIN_DATA",
        help="Path to .txt file used for training (required to enable training)."
    )

    # Flag to use pretrained data from resources/
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="load pretrained merges and vocabulary from resources (skip training)"
    )

    # Tokenize a string or a list of strings in a .txt file
    parser.add_argument(
        "--tokenize",
        type=str,
        metavar="TEST_DATA",
        help="String to tokenize or path to .txt file for tokenization."
    )

    # Select maximum vocabulary size for training (hyperparameter)
    parser.add_argument(
        "--max_vocab",
        type=int,
        metavar="INTEGER",
        default=10_000,
        help="maximum vocabulary size for training"
    )
    
    # Store the arguments so that we can use them
    args = parser.parse_args()

# Runs the CLI
if __name__ == "__main__":
    main()
