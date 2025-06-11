from custom_tokenizers import (
    SubwordTokenizer,
    NaiveBPE,
    TrieNode,
    Trie,
    TrieBPE,
    NaiveWP,
    WPTrieNode,
    WPTrie,
    WPTrie_E2E,
    FastWP,
    Fast_WP_E2E
)
from custom_benchmarks import *
import random
from datasets import load_dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer

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

    fleurs_train_fr = load_dataset("google/fleurs", "fr_fr", split = "train")
    toy_corpus_fr = build_toy_data(fleurs_train_fr, 5000, 'raw_transcription')
    # print(random.choice(toy_corpus_fr))

    # Save the toy corpus to a .txt file for later use
    import os
    output_path = "data/toy_corpus_fr.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for line in toy_corpus_fr:
            out_f.write(line + "\n")
    print(f"Toy corpus saved to {output_path}")

    fleurs_test_fr = load_dataset("google/fleurs", "fr_fr", split = "test")
    test_text_fr = build_toy_data(fleurs_test_fr, 50, 'raw_transcription')
    print(random.choice(test_text_fr))

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    toy_corpus = toy_corpus_fr
    test_text = random.choice(test_text_fr)

    bpe_optim = TrieBPE(tokenizer, verbose=False)
    bpe_naive = NaiveBPE(tokenizer) # type: ignore
    wp_naive = NaiveWP(tokenizer) # type: ignore
    wp_naive.train(toy_corpus, 570)
    bpe_optim.train(toy_corpus, 570)
    bpe_naive.train(toy_corpus, 570)

    print(test_text)
    print(bpe_naive.tokenize(test_text))
    print(bpe_optim.tokenize(test_text))
    print(wp_naive.tokenize(test_text))

    print("How long are my tokenized sentences? Shorter the better.")
    print("Average tokens per sentence (BPE naïve):", avg_tokens_per_sentence(bpe_naive, [test_text]))
    print("Average tokens per sentence (BPE optime):", avg_tokens_per_sentence(bpe_optim, [test_text]))
    print("Average tokens per sentence (WP naïve):", avg_tokens_per_sentence(wp_naive, [test_text]))
    print()
    print("How many words are split? The lower the better.")
    print(f"Subword fragmentation rate (BPE naïve): {subword_fragmentation_rate(bpe_naive, [test_text]):.2f}%")
    print(f"Subword fragmentation rate (BPE optime): {subword_fragmentation_rate(bpe_optim, [test_text]):.2f}%")
    print(f"Subword fragmentation rate (WP naïve): {subword_fragmentation_rate(wp_naive, [test_text]):.2f}%")
    print()
    print("How many full words are included in the tokenizer's vocab? The higher the better.")
    print(f"Vocabulary coverage rate (BPE naïve): { vocabulary_coverage_rate(bpe_naive, [test_text]):.2f}%")
    print(f"Vocabulary coverage rate (BPE optime): { vocabulary_coverage_rate(bpe_optim, [test_text]):.2f}%")
    print(f"Vocabulary coverage rate (WP naïve): { vocabulary_coverage_rate(wp_naive, [test_text]):.2f}%")
    print()
    print("How compact is each token relative to the original text? Longer the better.")
    print(f"Compression rate (BPE naïve): {compression_rate(bpe_naive, [test_text]):.2f}")
    print(f"Compression rate (BPE optime): {compression_rate(bpe_optim, [test_text]):.2f}")
    print(f"Compression rate (WP naïve): {compression_rate(wp_naive, [test_text]):.2f}")
    print()
    print("How do the naïve and the optimized version match?")
    results = token_sequence_equivalence(bpe_naive, bpe_optim, [test_text])
    print("Positional matches/positions:", results[0], "in", results[1], f"= {results[2]:.2f}%")
    print("Unordered matches/positions:", results[3], "in", results[1], f"= {results[4]:.2f}%")
    print("Word-level matches/total words:", results[5], "in", results[6], f"= {results[7]:.2f}%")


    bpe_metrics_naive = measure_speed_memory(bpe_naive, [test_text])
    bpe_metrics_optime = measure_speed_memory(bpe_optim, [test_text])

    print()
    print("Tokenization process:")

    print("Naïve BPE Speed & Memory metrics:")
    for k, v in bpe_metrics_naive.items():
        print(f"{k}: {v:.4f}")

    print()

    print("Optimized BPE Speed & Memory Metrics:")
    for k, v in bpe_metrics_optime.items():
        print(f"{k}: {v:.4f}")

    metrics_bpe_naive = measure_training_performance(bpe_naive, toy_corpus, 570)
    metrics_bpe_optim = measure_training_performance(bpe_optim, toy_corpus, 570)

    print()
    print("Vocab building:")

    print("Naïve BPE training metrics:")
    for k, v in metrics_bpe_naive.items():
        print(f"{k}: {v:.4f}")

    print()

    print("Optimized BPE training metrics:")
    for k, v in metrics_bpe_optim.items():
        print(f"{k}: {v:.4f}")