import argparse
from transformers import AutoTokenizer
from utils import *
from bpe import *
from wordpiece import *

# Available tokenizers
TOKENIZERS = {
    "Naive BPE": NaiveBPE,
    "Naive WordPiece": NaiveWP,
    "Fast BPE": TrieBPE,
    "Fast WordPiece": FastWP#,
    #"Fast end-to-end WordPiece": FastWP_E2E
}

def main():
    parser = argparse.ArgumentParser(prog="tokenize")
    parser.add_argument("--model", choices=TOKENIZERS, required=True, help=f"Select tokenizer to use from {TOKENIZERS}")
    parser.add_argument("--pretrained", type=str, default="bert-base-uncased", help="HuggingFace pretrained tokenizer")
    parser.add_argument("--input", "-i", required=True, help="Text to tokenize (or path to a file)")
    parser.add_argument("--max_vocab", type=int, default=10_000, help="Maximum vocabulary size for training")
    parser.add_argument("--from-file", action="store_true", help="Treat input as path to training file")
    args = parser.parse_args()

    # Get tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(args.pretrained)

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
        tokens = tokenizer.tokenizer(line)
        print(f"{line} --> {' '.join(tokens)}")

    # @MG Let's include pre-trained versions.
    '''
    if args.load_pretrained:
        tokenizer = TokenizerClass.from_files("merges.txt", "vocab.json", tokenizer=hf_tokenizer)
    else:
        tokenizer = TokenizerClass(tokenizer=hf_tokenizer)
        tokenizer.train(corpus=corpus, max_vocab=args.max_vocab)
    '''
