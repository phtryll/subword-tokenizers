import os
import argparse
from functools import partial
from argparse import RawTextHelpFormatter
from transformers import AutoTokenizer
from source.utils import *
from source.bpe import *
from source.wordpiece import *
from source.benchmarks import benchmarks


# Small function to transform HF dataset to List[str] -- UNUSED, FOR NOW KEEP
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
            "\tpython cli.py --models NaiveBPE FastBPE --train data/train.txt --max-vocab 1000\n\n"
            "Tokenize a single string (with training):\n"
            "\tpython cli.py --models FastBPE --train data/train.txt --tokenize \"Hello world.\"\n\n"
            "Benchmark models:\n"
            "\tpython cli.py --model NaiveBPE NaiveWordPiece FastBPE  --benchmark \"Ceci est un test.\" data/train.txt\n\n\n"
        )
    )
    
    # Selecting a model
    parser.add_argument(
        "--model",
        choices=TOKENIZERS,
        nargs="+",
        metavar=("MODEL1", "MODEL2"),
        required=True,
        help=(
            "select primary tokenizer model (required) and optional second model for comparison: "
            f"{', '.join(TOKENIZERS.keys())}"
        )
    )

    # Select normalization model
    parser.add_argument(
        "--normalize_with",
        type=str,
        metavar="HF_TOKENIZER",
        default="bert-base-uncased",
        help=("select HuggingFace tokenizer to use for normalization (default: 'bert-base-uncased')")
    )

    # Training
    parser.add_argument(
        "--train",
        type=str,
        metavar="TRAIN_DATA",
        help="path to .txt file used for training (required to enable training)"
    )

    # Flag to use pretrained data from resources/
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="NOT IMPLEMENTED -- load pretrained merges and vocabulary from resources (skip training)"
    )

    # Tokenize a string or a list of strings in a .txt file
    parser.add_argument(
        "--tokenize",
        type=str,
        metavar="TEST_DATA",
        help="string to tokenize or path to .txt file for tokenization"
    )

    # Select maximum vocabulary size for training (hyperparameter)
    parser.add_argument(
        "--max_vocab",
        type=int,
        metavar="INTEGER",
        default=1_000,
        help="maximum vocabulary size for training (default: 1000)"
    )
    
    # Benchmark models
    parser.add_argument(
        "--benchmark",
        nargs=2,
        type=str,
        metavar=("TEST_INPUT", "TRAIN_INPUT"),
        help="benchmark models: TEST_INPUT (string or .txt file) and TRAIN_INPUT (.txt file path)"
    )
    
    # Store the arguments so that we can use them
    args = parser.parse_args()


    # INITIALIZATION STAGE
    # Load the HF normalization tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(args.normalize_with)

    # Instantiate one tokenizer per --model entry
    tokenizer_instances = {}
    for model_name in args.model:
        # same pattern as: bpe_naive = NaiveBPE(tokenizer)
        tokenizer_instances[model_name] = TOKENIZERS[model_name](hf_tokenizer)
    
    # Print the loaded models
    print(f"Loaded tokenizer model(s): {', '.join(tokenizer_instances.keys())}")
    
    
    # Train each model if --train was provided
    if args.train:
        # Load training data from the .txt file
        with open(args.train, "r", encoding="utf-8") as f:
            corpus = f.read().splitlines()
        
        # Train each selected tokenizer
        for name, tok in tokenizer_instances.items():
            print(f"Training {name} with max_vocab={args.max_vocab} on {len(corpus)} examples...")
            tok.train(corpus, args.max_vocab)

    # Tokenization stage
    if args.tokenize:
        print("Tokenizing input...")
        # Load test data: string or .txt file
        if os.path.isfile(args.tokenize) and args.tokenize.lower().endswith('.txt'):
            with open(args.tokenize, "r", encoding="utf-8") as f:
                inputs = f.read().splitlines()
        else:
            inputs = [args.tokenize]
        # Tokenize each input with each model
        for text in inputs:
            for name, tok in tokenizer_instances.items():
                tokens = tok.tokenize(text)
                print(f"[{name}] {tokens}")

    # Benchmark stage
    if args.benchmark:
        # Extract test and train inputs
        test_input_arg, train_input_arg = args.benchmark

        # Load test inputs: string or .txt file
        if os.path.isfile(test_input_arg) and test_input_arg.lower().endswith('.txt'):
            with open(test_input_arg, "r", encoding="utf-8") as f:
                test_inputs = f.read().splitlines()
        else:
            test_inputs = [test_input_arg]

        # Load training inputs: must be .txt file path
        if not os.path.isfile(train_input_arg) or not train_input_arg.lower().endswith('.txt'):
            parser.error("--benchmark requires TRAIN_INPUT to be a valid .txt file path")
        with open(train_input_arg, "r", encoding="utf-8") as f:
            train_inputs = f.read().splitlines()

        # Determine primary and other tokenizers
        model_names = list(tokenizer_instances.keys())
        models = list(tokenizer_instances.values())
        primary = models[0]
        primary_name = model_names[0]
        others = models[1:]

        # Run benchmarks
        if not others:
            # Single model case
            print(f"Benchmarking {primary_name} alone on {len(test_inputs)} inputs and {len(train_inputs)} training examples with max_vocab={args.max_vocab}...")
            benchmarks(primary, test_inputs, args.max_vocab, train_inputs)
            print()
        else:
            # Multi-model case: benchmark primary against all others
            other_names = model_names[1:]
            print(
                f"Benchmarking {primary_name} vs {' and '.join(other_names)} "
                f"on {len(test_inputs)} inputs and {len(train_inputs)} "
                f"training examples with max_vocab={args.max_vocab}..."
            )
            benchmarks(primary, test_inputs, args.max_vocab, train_inputs, others)
            print()

# Runs the CLI
if __name__ == "__main__":
    main()
