"""Benchmark utilities for evaluating subword tokenizers."""

from typing import List, Tuple, Dict, Optional, Any
from collections import Counter
import time
import tracemalloc


def avg_tokens_per_sentence(tokenizer: Any, texts: List[str]) -> float:
    """
    Compute the average number of tokens per sentence for a given tokenizer.

    Args:
        tokenizer (Any): Tokenizer with a `tokenize` method.
        texts (List[str]): List of input sentences.

    Returns:
        float: Average number of tokens per sentence.
    """
    # 1. Input validation
    if not texts:
        return 0.0
    # 2. Tokenize each sentence and count tokens
    total_tokens = sum(len(tokenizer.tokenize(text)) for text in texts)
    # 3. Compute average tokens per sentence
    return total_tokens / len(texts)


def subword_fragmentation_rate(tokenizer: Any, texts: List[str]) -> float:
    """
    Compute the subword fragmentation rate: the percentage of words
    that are split into multiple subword tokens.

    Args:
        tokenizer (Any): Tokenizer with a `tokenize` method.
        texts (List[str]): List of input sentences.

    Returns:
        float: Percentage of words split into multiple subword tokens.
    """
    # 1. Flatten all words across all texts
    words = [word for txt in texts for word in txt.split()]
    if not words:
        return 0.0

    # 2. Count how many are split into multiple subwords
    fragmented = sum(
        1
        for word in words
        if len(tokenizer.tokenize(word)) > 1  # More than one subword means fragmented
    )

    # 3. Compute fragmentation rate as a percentage
    return fragmented / len(words) * 100


def vocabulary_coverage_rate(tokenizer: Any, texts: List[str]) -> float:
    """
    Compute vocabulary coverage: the percentage of unique full words
    in the input texts that the tokenizer recognizes as single tokens.

    Args:
        tokenizer (Any): Tokenizer with a `tokenize` method.
        texts (List[str]): List of input sentences.

    Returns:
        float: Percentage of unique words tokenized as exactly one token.
    """
    # 1. Gather unique words from all texts
    words = {word for txt in texts for word in txt.split()}
    if not words:
        return 0.0

    # 2. Count words tokenized as exactly one token (no splitting)
    covered = sum(1 for word in words if tokenizer.tokenize(word) == [word])

    # 3. Compute coverage as a percentage
    return covered / len(words) * 100


def compression_rate(tokenizer: Any, texts: List[str]) -> float:
    """
    Compute the compression rate: ratio of total non-space characters
    to total number of subword tokens.

    Args:
        tokenizer (Any): Tokenizer with a `tokenize` method.
        texts (List[str]): List of input sentences.

    Returns:
        float: Compression rate (characters per subword token).
    """
    # 1. Input validation
    if not texts:
        return 0.0

    # 2. Count total non-space characters
    total_chars = sum(len(txt.replace(" ", "")) for txt in texts)
    # 3. Count total number of subword tokens
    total_subwords = sum(len(tokenizer.tokenize(txt)) for txt in texts)

    # 4. Avoid division by zero
    if total_subwords == 0:
        return float('inf')

    # 5. Compute compression rate (characters per subword token)
    return total_chars / total_subwords


def token_sequence_equivalence(
    tokenizer1: Any,
    tokenizer2: Any,
    texts: List[str]
) -> Tuple[int, int, float, int, float, int, int, float]:
    """
    Compute equivalence metrics between two tokenizers over a list of texts.

    Args:
        tokenizer1 (Any): First tokenizer with a `tokenize` method.
        tokenizer2 (Any): Second tokenizer with a `tokenize` method.
        texts (List[str]): List of input sentences.

    Returns:
        Tuple containing:
            total_pos_matches (int): Tokens matching at the same positions.
            total_positions (int): Total positions compared.
            positional_rate (float): Percentage of positional matches.
            total_unordered_matches (int): Tokens matching regardless of position.
            unordered_rate (float): Percentage of unordered matches.
            total_word_matches (int): Words sharing at least one subword.
            total_words (int): Total words across all sentences.
            word_match_rate (float): Percentage of words with ≥1 shared subword.
    """
    # 1. Initialize counters for all metrics
    total_pos_matches = 0
    total_positions = 0
    total_unordered_matches = 0
    total_words = 0
    total_word_matches = 0

    # 2. Iterate through each text
    for text in texts:
        # 2.1 Tokenize with both tokenizers
        tokens1 = tokenizer1.tokenize(text)
        tokens2 = tokenizer2.tokenize(text)

        # 2.2 Compare tokens at same positions (positional matches)
        n = min(len(tokens1), len(tokens2))
        total_pos_matches += sum(1 for i in range(n) if tokens1[i] == tokens2[i])
        total_positions += n

        # 2.3 Compare unordered token overlap (regardless of position)
        freq1 = Counter(tokens1)
        freq2 = Counter(tokens2)
        overlap = sum(min(freq1[t], freq2[t]) for t in (freq1.keys() & freq2.keys()))
        total_unordered_matches += overlap

        # 2.4 For each word, check if both tokenizers share at least one subword
        words = text.split()
        total_words += len(words)
        for word in words:
            sub1 = tokenizer1.tokenize(word)
            sub2 = tokenizer2.tokenize(word)
            if set(sub1) & set(sub2):  # Non-empty intersection
                total_word_matches += 1

    # 3. Compute rates as percentages
    positional_rate = (total_pos_matches / total_positions * 100) if total_positions else 100.0
    unordered_rate = (total_unordered_matches / total_positions * 100) if total_positions else 100.0
    word_match_rate = (total_word_matches / total_words * 100) if total_words else 100.0

    # 4. Return all metrics as a tuple
    return (
        total_pos_matches,
        total_positions,
        positional_rate,
        total_unordered_matches,
        unordered_rate,
        total_word_matches,
        total_words,
        word_match_rate
    )


def measure_speed_memory(tokenizer: Any, texts: List[str]) -> Dict[str, float]:
    """
    Measure tokenization speed and memory usage for a tokenizer over texts.

    Args:
        tokenizer (Any): Tokenizer with a `tokenize` method.
        texts (List[str]): List of input sentences.

    Returns:
        Dict[str, float]: Metrics including total time, throughput, average latency, and peak memory in MB.
    """
    # 1. Start memory tracking
    tracemalloc.start()

    # 2. Measure tokenization time
    start_time = time.perf_counter()
    all_tokens = [tokenizer.tokenize(txt) for txt in texts]  # Tokenize all texts
    end_time = time.perf_counter()

    # 3. Get peak memory usage
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 4. Compute timing and throughput metrics
    total_time = end_time - start_time
    total_tokens = sum(len(tokens) for tokens in all_tokens)
    throughput = total_tokens / total_time if total_time > 0 else float('inf')
    avg_latency = total_time / len(texts) if texts else 0.0
    peak_memory = peak / (1024**2)  # Convert bytes to MB

    # 5. Return all metrics in a dictionary
    return {
        "total_time_s": total_time,
        "throughput_toks_per_s": throughput,
        "avg_latency_s": avg_latency,
        "peak_memory_mb": peak_memory
    }


def measure_training_performance(tokenizer_instance: Any, corpus: List[str], max_vocab_size: int) -> Dict[str, float]:
    """
    Measure training speed and memory usage for a tokenizer.

    Args:
        tokenizer_instance (Any): Tokenizer instance with a `train` method.
        corpus (List[str]): List of training sentences.
        max_vocab_size (int): Maximum vocabulary size for training.

    Returns:
        Dict[str, float]: Metrics including training time, peak memory in MB, and number of merges.
    """
    # 1. Start memory tracking
    tracemalloc.start()
    # 2. Measure training time
    start_time = time.perf_counter()
    tokenizer_instance.train(corpus, max_vocab_size)
    end_time = time.perf_counter()
    # 3. Get peak memory usage
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 4. Determine number of merges (if available)
    if hasattr(tokenizer_instance, 'merges_list'):
        num_merges = len(tokenizer_instance.merges_list)
    elif hasattr(tokenizer_instance, 'merges'):
        num_merges = len(tokenizer_instance.merges)
    else:
        num_merges = 0

    # 5. Return performance metrics as a dictionary
    return {
        "train_time_s": end_time - start_time,
        "peak_memory_mb": peak / (1024**2),
        "num_merges": float(num_merges)
    }