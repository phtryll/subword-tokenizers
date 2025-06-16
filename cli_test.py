import os
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

    # Get tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(args.normalize_with)

    # Load training corpus
    if not args.train:
        parser.error("--train requires TRAIN_DATA (a string or path to a .txt file)")
    train_data = args.train
    if os.path.isfile(train_data):
        with open(train_data, "r", encoding="utf-8") as f:
            corpus = f.read().splitlines()
    else:
        corpus = [train_data]

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








### MAIN.py





import argparse
from argparse import RawTextHelpFormatter
import os
from transformers import AutoTokenizer
from source.utils import *
from source.bpe import *
from source.wordpiece import *
from source.benchmarks import *
import pickle

if __name__ == "__main__":
    # Define supported models
    supported_models = {"FastBPE", "NaiveBPE", "NaiveWP", "FastWP", "Fast_WP_E2E"}

    parser = argparse.ArgumentParser(
        description=(
            "Subword Tokenizer CLI\n\n"
            "A command-line tool to train and/or tokenize text using various subword tokenizers.\n"
        ),
        formatter_class=RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Train models:\n"
            "    python main.py --models FastBPE NaiveBPE \\\n"
            "      --train --max-vocab 5000 --train-data data/raw/train.txt --merges\n\n"
            "  Tokenize a single string:\n"
            "    python main.py --models FastBPE --tokenize \"Hello world\"\n\n"
            "  Batch tokenize from file:\n"
            "    python main.py --models NaiveBPE --tokenize data/raw/test.txt\n"
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=supported_models,
        required=True,
        help="Select one or more subword tokenizer models to use."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Enable training of the selected models on provided data."
    )
    parser.add_argument(
        "--max-vocab",
        type=int,
        help="Maximum size of subword vocabulary (required with --train)."
    )
    parser.add_argument(
        "--train-data",
        type=str,
        help="Path to .txt file (one sentence per line) used for training (required with --train)."
    )
    parser.add_argument(
        "--tokenize",
        type=str,
        help="String to tokenize or path to .txt/.pkl file for tokenization."
    )
    parser.add_argument(
        "--merges",
        action="store_true",
        help="After training, print the learned merge operations."
    )

    args = parser.parse_args()

    # Initialize train_corpus
    train_corpus: list = []

    # Validate training arguments if requested
    if args.train:
        if args.max_vocab is None or args.train_data is None:
            parser.error("--train requires both --max-vocab and --train-data")
        if not os.path.isfile(args.train_data):
            parser.error(f"Training data not found: {args.train_data}")
        # Load training corpus
        with open(args.train_data, encoding="utf-8") as f:
            train_corpus = [line.strip() for line in f if line.strip()]

    selected_models = args.models

    print(f"Selected models: {selected_models}")

    # Instantiate shared HF tokenizer and model registry
    hf_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    MODEL_REGISTRY = {
        "FastBPE": FastBPE,
        "NaiveBPE": NaiveBPE,
        "NaiveWP": NaiveWP,
        "FastWP": FastWP,
        "Fast_WP_E2E": Fast_WP_E2E,
    }

    # Create tokenizer instances
    tokenizer_instances = {
        f"tok{i+1}": MODEL_REGISTRY[name](hf_tok)
        for i, name in enumerate(selected_models)
    }

    # 2. Training phase
    if args.train:
        print(f"→ Training each model with max_vocab={args.max_vocab} on {len(train_corpus)} sentences")
        # Ensure output directory for models exists
        models_dir = "data/models"
        os.makedirs(models_dir, exist_ok=True)
        for var, model in tokenizer_instances.items():
            print(f"  • {var} ({type(model).__name__}) …", end="", flush=True)
            model.train(train_corpus, args.max_vocab)
            print(" done")
            # Save the trained model to disk
            save_path = f"{models_dir}/{type(model).__name__}.pkl"
            with open(save_path, "wb") as f:
                pickle.dump(model, f)
            print(f"    Saved model to {save_path}")
            if args.merges:
                # Retrieve merges_list or merges attribute
                merges = getattr(model, 'merges_list', getattr(model, 'merges', []))
                print(f"    Learned merges for {var}: {merges}")

    # 3. Tokenization phase
    if args.tokenize:
        # Load trained models from data/models
        trained_instances = {}
        for i, name in enumerate(selected_models):
            save_path = f"data/models/{name}.pkl"
            if not os.path.isfile(save_path):
                parser.error(f"Model {name} is not trained; missing {save_path}")
            with open(save_path, "rb") as f:
                trained_instances[f"tok{i+1}"] = pickle.load(f)
        tokenizer_instances = trained_instances
        if args.merges:
            # Verify merges loaded correctly after unpickle
            for var, model in tokenizer_instances.items():
                merges = getattr(model, 'merges_list', getattr(model, 'merges', []))
                print(f"{var} merges after loading: {merges}")
        tok_arg = args.tokenize
        # Case A: existing pickle of tokenized output
        if os.path.isfile(tok_arg) and tok_arg.lower().endswith(".pkl"):
            with open(tok_arg, "rb") as f:
                data = pickle.load(f)
            print("Loaded tokenized data from pickle:")
            print(data)
        # Case B: batch tokenization from .txt file
        elif os.path.isfile(tok_arg) and tok_arg.lower().endswith(".txt"):
            with open(tok_arg, encoding="utf-8") as f:
                texts = [line.strip() for line in f if line.strip()]
            for var, model in tokenizer_instances.items():
                print(f"== {var} ({type(model).__name__}) ==")
                for txt in texts:
                    tokens = model.tokenize(txt)
                    print(tokens)
        # Case C: direct string input
        else:
            text = tok_arg
            for var, model in tokenizer_instances.items():
                tokens = model.tokenize(text)
                print(f"{var} → {tokens}")





### test.py




from source.utils import *
from source.bpe import *
from source.wordpiece import *
from source.benchmarks import *
from datasets import load_dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer
import random

if __name__ == "__main__":
    def build_toy_data(dataset, num_examples, feature_name):
        toy_data = []
        for example in dataset:
            value = example.get(feature_name)
            if value is not None:
                toy_data.append(value)
                if len(toy_data) == num_examples:
                    break
        return toy_data

    # Build small corpus
    # Training
    fleurs_train_fr = load_dataset("google/fleurs", "fr_fr", split = "train")
    toy_corpus_fr = build_toy_data(fleurs_train_fr, 5000, 'raw_transcription')

    # Testing
    fleurs_test_fr = load_dataset("google/fleurs", "fr_fr", split = "test")
    test_text_fr = build_toy_data(fleurs_test_fr, 50, 'raw_transcription')

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Internal variables
    toy_corpus = toy_corpus_fr
    test_text = random.choice(test_text_fr)

    # Load and train models
    bpe_naive = NaiveBPE(tokenizer) # type: ignore
    bpe_optim = FastBPE(tokenizer)
    bpe_optim.train(toy_corpus, 570)
    bpe_naive.train(toy_corpus, 570)

    # print(test_text)
    # print(bpe_naive.tokenize("Hello, this is a test."))
    # print(bpe_optim.tokenize(test_text))

    benchmarks(bpe_naive, [test_text], 600, bpe_optim)