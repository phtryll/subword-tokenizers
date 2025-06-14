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
    supported_models = {"TrieBPE", "NaiveBPE", "NaiveWP", "FastWP", "Fast_WP_E2E"}

    parser = argparse.ArgumentParser(
        description=(
            "Subword Tokenizer CLI\n\n"
            "A command-line tool to train and/or tokenize text using various subword tokenizers.\n"
        ),
        formatter_class=RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Train models:\n"
            "    python main.py --models TrieBPE NaiveBPE \\\n"
            "      --train --max-vocab 5000 --train-data data/raw/train.txt --merges\n\n"
            "  Tokenize a single string:\n"
            "    python main.py --models TrieBPE --tokenize \"Hello world\"\n\n"
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
        "TrieBPE": TrieBPE,
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
