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
    bpe_optim = TrieBPE(tokenizer, verbose=False)
    bpe_optim.train(toy_corpus, 570)
    bpe_naive.train(toy_corpus, 570)

    # print(test_text)
    # print(bpe_naive.tokenize("Hello, this is a test."))
    # print(bpe_optim.tokenize(test_text))

    benchmarks(bpe_naive, [test_text], 600, bpe_optim)