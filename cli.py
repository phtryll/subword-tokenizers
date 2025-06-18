import json
import os
import shutil
import argparse
from functools import partial
from argparse import RawTextHelpFormatter
from transformers import AutoTokenizer
from source.utils import *
from source.bpe import *
from source.wordpiece import *
from source.benchmarks import benchmarks


# Cleaner help display
MyFormatter = partial(RawTextHelpFormatter, max_help_position=70, width=100)

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
            "Usage examples:\n\n"
            "Training:\n"
            "  Train models with training data:\n"
            "    python cli.py --model NaiveBPE FastBPE --train data/train.json --max-vocab 1000\n"
            "  Save trained models for future use:\n"
            "    python cli.py --model NaiveBPE --train data/train.json --save my_merges_dir\n\n"
            "Tokenization:\n"
            "  Tokenize a single sentence:\n"
            "    python cli.py --model FastBPE --train data/train.json --tokenize \"Hello world.\"\n"
            "  Tokenize a .json list of sentences:\n"
            "    python cli.py --model FastBPE --train data/train.json --tokenize data/test.json\n"
            "  Tokenize a .json/string with a pretrained model:\n"
            "    python cli.py --model FastBPE --pretrained my_merges_dir --tokenize data/test.json\n\n"
            "Benchmarking:\n"
            "  Benchmark a pretrained model on a test sentence:\n"
            "    python cli.py --model NaiveWordPiece --pretrained my_vocab_dir --benchmark \"This is a test.\"\n"
            "  Benchmark a pretrained model on a .json list:\n"
            "    python cli.py --model NaiveWordPiece --pretrained my_vocab_dir --benchmark data/test.json\n"
            "  Compare multiple pretrained models:\n"
            "    python cli.py --model NaiveBPE FastBPE --pretrained my_merges_dir --benchmark data/test.json\n"
            "  Compare sequence equivalence between pretrained models:\n"
            "    python cli.py --model NaiveBPE FastBPE --pretrained my_merges_dir --benchmark data/test.json --compare\n"
            "  Benchmark training time using a .json file and save:\n"
            "    python cli.py --model NaiveBPE FastBPE --benchmark data/train.json --save my_testing_dir\n\n"
            "Resetting and saving:\n"
            "  Reset a model's saved resources:\n"
            "    python cli.py --model NaiveBPE --reset testing_dir\n"
            "  Save resources after training:\n"
            "    python cli.py --model NaiveBPE --train data/train.json --save another_dir\n"
        )
    )
    
    # Selecting a model
    parser.add_argument(
        "-m", "--model",
        choices=TOKENIZERS,
        nargs="+",
        metavar=("MODEL1", "MODEL2"),
        required=True,
        help=(
            "select primary tokenizer model (required) and optional other models for comparison: "
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
        help="path to .json file used for training (required to enable training)"
    )

    # Reset pretrained resources for selected models
    parser.add_argument(
        "--save",
        type=str,
        metavar="PATH",
        help="save training merges/vocab in specified path for later use"
    )

    # Flag to use pretrained data from resources/
    parser.add_argument(
        "--pretrained",
        type=str,
        metavar="PATH",
        help="load pretrained merges and vocabulary from specified path"
    )

    # Tokenize a string or a list of strings in a .json file
    parser.add_argument(
        "--tokenize",
        type=str,
        metavar="TEST_DATA",
        help="string to tokenize or path to .json file for tokenization"
    )

    # Select maximum vocabulary size for training (hyperparameter)
    parser.add_argument(
        "-v", "--max_vocab",
        type=int,
        metavar="INTEGER",
        default=1_000,
        help="maximum vocabulary size for training (default: 1000)"
    )
    
    # Benchmark models
    parser.add_argument(
        "-b", "--benchmark",
        type=str,
        metavar="INPUT",
        help=(
            "benchmark the selected tokenizer(s)\n"
            "-\tif --pretrained is provided, INPUT is treated as test data for tokenization benchmarking (string or .json)\n"
            "-\tif --pretrained is not provided, INPUT is treated as training data for benchmarking training performance (must be .json)\n"
            "-\tuse --compare to evaluate token sequence equivalence between multiple pretrained models"
        )
    )

    parser.add_argument(
        "-c", "--compare",
        action="store_true",
        help="with --pretrained, only run token-sequence equivalence between models"
    )
    
    # Reset pretrained resources for selected models
    parser.add_argument(
        "--reset",
        type=str,
        metavar="PATH",
        help="reset merges/vocabulary for selected models by deleting their specified resources directory"
    )
    
    # Store the arguments so that we can use them
    args = parser.parse_args()


    # LOAD NORMALIZATION
    # Load the HF normalization tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(args.normalize_with)


    # RESET RESOURCES
    # Handle reset of resources
    if args.reset:
        # Remove resources for each selected model
        for model_name in args.model:

            # Get the correct path: resources/MODEL_NAME
            resource_path = os.path.join("resources", args.reset, model_name)

            # If the directory exists clean it
            if os.path.isdir(resource_path):
                shutil.rmtree(resource_path)
                
                # Confirm
                print(f"Reset resources for {model_name}")
            
            else:
                # Or not...
                print(f"No resources to reset for {model_name}")
        
        return


    # INSTANTIATE MODELS
    # Instantiate one tokenizer per --model entry
    tokenizer_instances = {}
    for model_name in args.model:
        # Same pattern as: bpe_naive = NaiveBPE(tokenizer)
        tokenizer_instances[model_name] = TOKENIZERS[model_name](hf_tokenizer)


    # IF PRETRAINED
    # Load saved merges and vocab if requested
    if args.pretrained:
        for name, tok in tokenizer_instances.items():

            # Get resource path
            resource_path = os.path.join("resources", args.pretrained, name)
            
            # Load merges/vocab for given model
            tok.load_resources(resource_path)
            print(f"Loaded saved merges and vocab for {name} from {resource_path}")

    # Print the loaded models
    print(f"Loaded tokenizer model(s): {', '.join(tokenizer_instances.keys())}")


    # TRAINING
    # Train each model if --train was provided
    if args.train:

        # Load training data from the .json file
        with open(args.train, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        
        # Train each selected tokenizer
        for name, tok in tokenizer_instances.items():
            print(f"Training {name} with max_vocab={args.max_vocab} on {len(corpus)} examples...")
            
            # Call training
            tok.train(corpus, args.max_vocab)

            # If --save flag, save the merges/vocab
            if args.save:

                # Get the path in resources/DIR_NAME
                resource_path = os.path.join("resources", args.save, name)

                # Save
                tok.save_resources(resource_path)
                print(f"Saved merges and vocab for {name} to {resource_path}")


    # TOKENIZATION
    if args.tokenize:
        print("Tokenizing input...")

        # Load test data: string or .json file
        if os.path.isfile(args.tokenize) and args.tokenize.lower().endswith('.json'):
            with open(args.tokenize, "r", encoding="utf-8") as f:
                inputs = json.load(f)
        else:
            inputs = [args.tokenize]
        
        # Tokenize each input with each model
        output = {}
        # For each example
        for text in inputs:
            # For each model
            for name, tok in tokenizer_instances.items():

                # Call tokenization
                tokens = tok.tokenize(text)
                print(f"[{name}] {tokens}")

                # Optionally, store outputs for writing to .json if needed
                if name not in output:
                    output[name] = []
                output[name].append(tokens)
        
        # If input was from a .json file, write tokenized output to a .json file
        if os.path.isfile(args.tokenize) and args.tokenize.lower().endswith('.json'):
            out_path = args.tokenize.replace('.json', '.tokens.json')
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            print(f"Tokenized output written to {out_path}")


    # BENCHMARKING STAGE
    if args.benchmark:

        # Single INPUT for benchmark
        b_arg = args.benchmark

        # If pretrained flag: load the trained model and skip training benchmarking
        if args.pretrained:

            # Benchmarking input is tokenization input
            test_input_arg = b_arg

            # Read the .json file or the provided string sequence
            if os.path.isfile(test_input_arg) and test_input_arg.lower().endswith('.json'):
                with open(test_input_arg, "r", encoding="utf-8") as f:
                    test_inputs = json.load(f)
            else:
                test_inputs = [test_input_arg]

            # No training here so training argument is empty
            train_inputs = []

        # If pretrained is not passed: train and skip tokenization benchmarking
        else:

            # Benchmarking input is training input
            train_input_arg = b_arg
            
            # Read the .json file
            if not os.path.isfile(train_input_arg) or not train_input_arg.lower().endswith('.json'):
                parser.error("--benchmark requires TRAIN_INPUT to be a valid .json file path")
            with open(train_input_arg, "r", encoding="utf-8") as f:
                train_inputs = json.load(f)
            
            # No tokenizing here so training argument is empty
            test_inputs = []

        # Determine primary and additional tokenizers
        model_names = list(tokenizer_instances.keys())
        models = list(tokenizer_instances.values())
        primary = models[0]
        primary_name = model_names[0]
        others = models[1:]

        # Compare-only flag requires pretrained + at least two models
        if args.compare and not args.pretrained:
            parser.error("--compare may only be used with --pretrained")
        if args.compare and len(models) < 2:
            parser.error("--compare requires at least two tokenizers")


        # RUN BENCHMARKS
        # If only a single model
        if not others:
            print(
                f"Benchmarking {primary_name} {'(pretrained)' if args.pretrained else ''}{'' if not train_inputs else f'with {len(train_inputs)} training examples'}..."
            )
            benchmarks(
                tokenizer=primary,
                max_vocab_size=args.max_vocab,
                test_corpus=test_inputs,
                train_corpus=train_inputs,
                pretrained=bool(args.pretrained),
                pretrained_path=args.pretrained,
                reference_tokenizers=others,
                compare_only=args.compare
            )
            print()
        else:
            # If multiple models
            other_names = model_names[1:]
            print(
                f"Benchmarking {primary_name} vs {' vs '.join(other_names)} "
                f"{'(pretrained)' if args.pretrained else ''}{'' if not train_inputs else f'with {len(train_inputs)} training examples'}..."
            )
            benchmarks(
                tokenizer=primary,
                max_vocab_size=args.max_vocab,
                test_corpus=test_inputs,
                train_corpus=train_inputs,
                pretrained=bool(args.pretrained),
                pretrained_path=args.pretrained,
                reference_tokenizers=others,
                compare_only=args.compare
            )
            print()

    # Save resources if requested with --save flag
    if args.save:
        for name, tok in tokenizer_instances.items():
            resource_path = os.path.join("resources", args.save, name)
            tok.save_resources(resource_path)
            print(f"Saved merges and vocab for {name} to {resource_path}")

# Runs the CLI
if __name__ == "__main__":
    main()
