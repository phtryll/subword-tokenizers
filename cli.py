import argparse
from functools import partial
from argparse import RawTextHelpFormatter
from transformers import AutoTokenizer
from source.utils import *
from source.bpe import *
from source.wordpiece import *

# Cleaner help display
MyFormatter = partial(RawTextHelpFormatter, max_help_position=60, width=100)

# Available tokenizers
TOKENIZERS = {
    "NaiveBPE": NaiveBPE,
    "NaiveWordPiece": NaiveWP,
    "FastBPE": FastBPE,
    "FastWordPiece": FastWP
}

def main():
    parser = argparse.ArgumentParser(
        prog="tokenize",
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
        default="bert-base-uncased",
        help=("select HuggingFace tokenizer to use for normalization")
    )

    # Training
    parser.add_argument(
        "--train",
        action="store_true",
        help="enable training of the selected models on provided data."
    )

    # Selecting training data
    parser.add_argument(
        "--train-data",
        type=str,
        help="path to .txt file (one sentence per line) used for training (required with --train)."
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
        help="String to tokenize or path to .txt file for tokenization."
    )

    # Select maximum vocabulary size for training (hyperparameter)
    parser.add_argument(
        "--max_vocab",
        type=int,
        default=10_000,
        help="maximum vocabulary size for training"
    )

    # Path to training file #! Duplicate, work in progress..
    parser.add_argument(
        "--from-file",
        action="store_true",
        help="treat input as path to training file"
    )

    # Text to tokenize or path #! Duplicate, work in progress..
    parser.add_argument(
        "--input",
        required=True,
        help="Text to tokenize (or path to a file)"
    )
    
    # Store the arguments so that we can use them
    args = parser.parse_args()

    # Get tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(args.normalize_with)

    # Load training corpus
    if args.from_file:
        with open(args.input, "r", encoding="utf-8") as f:
            corpus = f.read().splitlines()
    else:
        corpus = [args.input]

    # Instantiate and train
    TokenizerClass = TOKENIZERS[args.model]
    tokenizer = TokenizerClass(tokenizer=hf_tokenizer)
    tokenizer.train(max_vocab=args.max_vocab, corpus=corpus)

    # @MG Let's include options for end-to-end tokenization,
    # @MG either in an E2E implementation or in this program.
    for line in corpus:
        tokens = tokenizer.tokenize(line)
        print(f"{line} --> {' '.join(tokens)}")

    # @MG Let's include pre-trained versions.
    '''
    if args.load_pretrained:
        tokenizer = TokenizerClass.from_files("merges.txt", "vocab.json", tokenizer=hf_tokenizer)
    else:
        tokenizer = TokenizerClass(tokenizer=hf_tokenizer)
        tokenizer.train(corpus=corpus, max_vocab=args.max_vocab)
    '''


# Runs the CLI
if __name__ == "__main__":
    main()
