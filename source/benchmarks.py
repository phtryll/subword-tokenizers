"""
Various benchmarks to evaluate tokenizers:

1.  avg_tokens_per_sentence:
       Compute the average number of tokens per sentence.
2.  avg_tokens_per_word:
       Compute the average number of subword tokens per word.
3.  subword_fragmentation_rate:
       Measure the percentage of unique words split into multiple subword tokens.
4.  vocabulary_coverage_rate:
       Calculate the percentage of unique words recognized as single tokens.
5.  compression_rate:
       Determine the average number of characters per subword token.
6.  normalized_sequence_length:
       Compute the ratio of the tokenizer's sequence length to a baseline where each character is treated as an individual token.
7.  token_sequence_equivalence:
       Compare two tokenizers on positional agreement, token overlap, and word-level subword match.
8.  tokenization_performance:
       Evaluate tokenization speed, latency, and memory usage.
9.  training_performance:
       Measure training speed, memory usage, and number of merge operations.
10. zipf_distribution:
       Assess how closely token frequency follows Zipf's law.
11. benchmarks:
       Run all benchmarks and print a summary of results to the console.
"""

import time
import math
import tracemalloc
from collections import Counter
from typing import List, Tuple, Dict, Any


def avg_tokens_per_sentence(tokenizer: Any, input: List[str]) -> float:
    """
    Compute the average number of tokens per sentence for a given tokenizer.

    Args:
        tokenizer (Any): Tokenizer with a `tokenize` method.
        input (List[str]): List of input sentences.

    Returns:
        float: Average number of tokens per sentence.
    """
    
    # 1. Input validation
    if not input:
        return 0.0
    
    # 2. Tokenize each sentence and count tokens
    total_tokens = sum(len(tokenizer.tokenize(sentence)) for sentence in input)
    
    # 3. Compute average tokens per sentence
    return total_tokens / len(input)


def avg_tokens_per_word(tokenizer: Any, input: List[str]) -> float:
    """
    Compute the average number of subword tokens per word.

    Args:
        tokenizer (Any): Tokenizer with a `tokenize` method.
        input (List[str]): List of input sentences.

    Returns:
        float: Average number of tokens per word.
    """
    
    # 1. Gather unique words from all sentences
    words = {word for sentence in input for word in sentence.split()}
    if not words:
        return 0.0

    # 2. Count tokens for each word
    total_tokens = sum(len(tokenizer.tokenize(word)) for word in words)

    # 3. Compute average tokens per word
    return total_tokens / len(words)


def normalized_sequence_length(tokenizer: Any, input: List[str]) -> float:
    """
    Compute normalized sequence length: the tokenizer's total tokens divided 
    by the baseline character-level token count (each character as a token).

    Args:
        tokenizer (Any): Tokenizer with a `tokenize` method.
        input (List[str]): List of input sentences.

    Returns:
        float: Total tokens divided by total characters in the input.
    """
    
    # 1. Validate input
    if not input:
        return 0.0
    
    # 2. Compute the number of tokens
    total_tokens = sum(len(tokenizer.tokenize(sentence)) for sentence in input)
    
    # 3. Compute the baseline if the input is split into individual characters
    total_chars = sum(len(token) for sentence in input for token in sentence)
    
    # 4. Return the normalized sequence length (avoid division by zero)
    return total_tokens / total_chars if total_chars else float('inf')


def subword_fragmentation_rate(tokenizer: Any, input: List[str]) -> float:
    """
    Compute the subword fragmentation rate: the percentage of unique words
    that are split into multiple subword tokens.

    Args:
        tokenizer (Any): Tokenizer with a `tokenize` method.
        input (List[str]): List of input sentences.

    Returns:
        float: Percentage of unique words split into multiple subword tokens.
    """

    # 1. Gather unique words from all sentences
    words = {word for sentence in input for word in sentence.split()}
    if not words:
        return 0.0

    # 2. Count unique words that are split into subwords
    split_words = sum(
        1
        for word in words
        if len(tokenizer.tokenize(word)) > 1
    )

    # 3. Compute fragmentation rate as a percentage
    return split_words / len(words) * 100


def vocabulary_coverage_rate(tokenizer: Any, input: List[str]) -> float:
    """
    Compute vocabulary coverage: the percentage of unique full words
    in the input input that the tokenizer recognizes as single tokens.

    Args:
        tokenizer (Any): Tokenizer with a `tokenize` method.
        input (List[str]): List of input sentences.

    Returns:
        float: Percentage of unique words tokenized as exactly one token.
    """
    
    # 1. Gather unique words from all sentences
    words = {word for sentence in input for word in sentence.split()}
    if not words:
        return 0.0

    # 2. Count words tokenized as exactly one token (no splitting)
    covered = sum(1 for word in words if len(tokenizer.tokenize(word)) == 1) #! Same as before

    # 3. Compute coverage as a percentage
    return covered / len(words) * 100


def compression_rate(tokenizer: Any, input: List[str]) -> float:
    """
    Compute the compression rate: ratio of total non-space characters
    to total number of subword tokens. Basically computes the average number
    of characters per subword token.

    Args:
        tokenizer (Any): Tokenizer with a `tokenize` method.
        input (List[str]): List of input sentences.

    Returns:
        float: Compression rate (characters per subword token).
    """
    
    # 1. Input validation
    if not input:
        return 0.0

    # 2. Count total non-space characters
    total_chars = sum(len(sentence.replace(" ", "")) for sentence in input)
    
    # 3. Count total number of subword tokens
    total_subwords = sum(len(tokenizer.tokenize(sentence)) for sentence in input)

    # 4. Avoid division by zero
    if total_subwords == 0:
        return float('inf')

    # 5. Compute compression rate (characters per subword token)
    return total_chars / total_subwords


def token_sequence_equivalence(
    tokenizer1: Any,
    tokenizer2: Any,
    input: List[str]
) -> Tuple[int, int, float, int, float, int, int, float]:
    """
    Compute equivalence metrics between two tokenizers over a list of input.

    Args:
        tokenizer1 (Any): First tokenizer with a `tokenize` method.
        tokenizer2 (Any): Second tokenizer with a `tokenize` method.
        input (List[str]): List of input sentences.

    Returns:
        Tuple containing:
            total_pos_matches (int): Tokens matching at the same position (offsets).
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

    # 2. Iterate through each sentence in the input
    for sentence in input:
        # 2.1 Tokenize with both tokenizers and strip '##' for WordPice subword tokens
        raw1 = tokenizer1.tokenize(sentence)
        raw2 = tokenizer2.tokenize(sentence)
        tokens1 = [token[2:] if token.startswith("##") else token for token in raw1]
        tokens2 = [token[2:] if token.startswith("##") else token for token in raw2]

        # 2.2 Compare tokens at same positions (positional matches)
        n = min(len(tokens1), len(tokens2)) #! Compare up to the length of the smallest sequence
        total_pos_matches += sum(1 for i in range(n) if tokens1[i] == tokens2[i])
        total_positions += n

        # 2.3 Compare unordered token overlap (regardless of position)
        freq1 = Counter(tokens1)
        freq2 = Counter(tokens2)
        overlap = sum(min(freq1[token], freq2[token]) for token in (freq1.keys() & freq2.keys())) #! Again, we take the counts of the smallest sequence
        total_unordered_matches += overlap

        # 2.4 For each word, check if both tokenizers share at least one subword
        words = sentence.split()
        total_words += len(words)
        for word in words:
            raw_sub1 = tokenizer1.tokenize(word)
            raw_sub2 = tokenizer2.tokenize(word)
            sub1 = [token[2:] if token.startswith("##") else token for token in raw_sub1]
            sub2 = [token[2:] if token.startswith("##") else token for token in raw_sub2]
            if set(sub1) & set(sub2): #! Non-empty intersection
                total_word_matches += 1

    # 3. Compute rates as percentages
    positional_rate = (total_pos_matches / total_positions * 100) if total_positions else 0.0
    unordered_rate = (total_unordered_matches / total_positions * 100) if total_positions else 0.0
    word_match_rate = (total_word_matches / total_words * 100) if total_words else 0.0

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


def tokenization_performance(tokenizer: Any, input: List[str]) -> Dict[str, float]:
    """
    Measure tokenization speed and memory usage for a tokenizer over input.

    Args:
        tokenizer (Any): Tokenizer with a `tokenize` method.
        input (List[str]): List of input sentences.

    Returns:
        Dict[str, float]: Metrics including total time, throughput, average latency, and peak memory in MB.
    """
    
    # 1. Start memory tracking
    tracemalloc.start()

    # 2. Measure tokenization time
    start_time = time.perf_counter()
    all_tokens = [tokenizer.tokenize(sentence) for sentence in input]  # Tokenize all input
    end_time = time.perf_counter()

    # 3. Get peak memory usage
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 4. Compute timing and throughput metrics
    total_time = end_time - start_time
    total_tokens = sum(len(tokens_in_sentence) for tokens_in_sentence in all_tokens)
    throughput = total_tokens / total_time if total_time > 0 else float('inf')
    avg_latency = total_time / len(input) if input else 0.0
    peak_memory = peak / (1024**2)  # Convert bytes to MB

    # 5. Return all metrics in a dictionary
    return {
        "total_time_s": total_time,
        "throughput_tokens_per_s": throughput,
        "avg_latency_s": avg_latency,
        "peak_memory_mb": peak_memory
    }


def training_performance(tokenizer: Any, corpus: List[str], max_vocab_size: int) -> Dict[str, float]:
    """
    Measure training speed and memory usage for a tokenizer.

    Args:
        tokenizer (Any): Tokenizer instance with a `train` method.
        corpus (List[str]): List of training sentences.
        max_vocab_size (int): Maximum vocabulary size for training.

    Returns:
        Dict[str, float]: Metrics including training time, peak memory in MB, and number of merges.
    """
    
    # 1. Start memory tracking
    tracemalloc.start()
    
    # 2. Measure training time
    start_time = time.perf_counter()
    tokenizer.train(corpus, max_vocab_size)
    end_time = time.perf_counter()
    
    # 3. Get peak memory usage
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 4. Determine number of merges (if available)
    if hasattr(tokenizer, 'merges_list'): #! `merges_list` is the name in naïve BPE
        num_merges = len(tokenizer.merges_list)
    elif hasattr(tokenizer, 'merges'): #! `merges` is the name in fast BPE
        num_merges = len(tokenizer.merges)
    else:
        num_merges = 0

    # 5. Return performance metrics as a dictionary
    return {
        "train_time_s": end_time - start_time,
        "peak_memory_mb": peak / (1024**2),
        "num_merges": float(num_merges)
    }


def zipf_distribution(tokenizer: Any, input: List[str]) -> Dict[str, float]:
    """
    Analyze token frequency distribution and compute Zipf's law fit.

    Args:
        tokenizer (Any): Tokenizer with a `tokenize` method.
        input (List[str]): List of input sentences.

    Returns:
        Dict[str, float]: Contains 'slope', 'intercept', and 'correlation' of the log-log fit.
    """
    
    # 1. Tokenize all input and flatten
    all_tokens = [token for sentence in input for token in tokenizer.tokenize(sentence)]
    
    # 2. Compute token frequencies and prepare ranks
    frequency = Counter(all_tokens)
    ranks = list(range(1, len(frequency) + 1))
    
    # 3. Sort frequencies in descending order and prepare ranks
    frequency_sorted = [count for _, count in frequency.most_common()]
    
    # 4. Compute log-rank and log-frequency
    log_ranks = [math.log(rank) for rank in ranks]
    log_frequencies = [math.log(freq) for freq in frequency_sorted]
    
    # 5. Linear regression on log-log data
    n = len(log_ranks)
    mean_x = sum(log_ranks) / n
    mean_y = sum(log_frequencies) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_ranks, log_frequencies))
    var_x = sum((x - mean_x) ** 2 for x in log_ranks)
    var_y = sum((y - mean_y) ** 2 for y in log_frequencies)
    slope = cov / var_x if var_x else 0.0
    intercept = mean_y - slope * mean_x
    correlation = cov / math.sqrt(var_x * var_y) if var_x and var_y else 0.0
    return {"slope": slope, "intercept": intercept, "correlation": correlation}


def benchmarks(
    tokenizer: Any,
    corpus: List[str],
    max_vocab_size: int,
    reference_tokenizer: Any = None
) -> None:
    """
    Run all benchmark functions and print results to the console.

    Args:
        tokenizer (Any): Tokenizer with a `tokenize` and `train` method.
        corpus (List[str]): List of input sentences.
        max_vocab_size (int): Maximum vocabulary size for training.
        reference_tokenizer (Any, optional): Second tokenizer for equivalence metrics.
    """

    # Get the names of the tokenizers
    name1 = tokenizer.__class__.__name__

    # Run metrics
    print(f"=== Tokenization Benchmarks for {name1} ===")
    
    avg_tokens = avg_tokens_per_sentence(tokenizer, corpus)
    print(f"Average tokens per sentence:        {avg_tokens:.2f}")
    
    avg_tpw = avg_tokens_per_word(tokenizer, corpus)
    print(f"Average tokens per word:            {avg_tpw:.2f}")

    comp_rate = compression_rate(tokenizer, corpus)
    print(f"Compression rate (chars per token): {comp_rate:.2f}")
    
    ns_len = normalized_sequence_length(tokenizer, corpus)
    print(f"Normalized sequence length:         {ns_len:.4f}")

    frag_rate = subword_fragmentation_rate(tokenizer, corpus)
    print(f"Subword fragmentation rate:         {frag_rate:.2f}%")

    vocab_cov = vocabulary_coverage_rate(tokenizer, corpus)
    print(f"Vocabulary coverage rate:           {vocab_cov:.2f}%")

    # If a second tokenizer was provided, run the same metrics for it
    if reference_tokenizer is not None:
        name2 = reference_tokenizer.__class__.__name__
        print(f"\n=== Tokenization Benchmarks for {name2} ===")
        avg_tokens2 = avg_tokens_per_sentence(reference_tokenizer, corpus)
        print(f"Average tokens per sentence:        {avg_tokens2:.2f}")
        avg_tpw2 = avg_tokens_per_word(reference_tokenizer, corpus)
        print(f"Average tokens per word:            {avg_tpw2:.2f}")
        comp_rate2 = compression_rate(reference_tokenizer, corpus)
        print(f"Compression rate (chars per token): {comp_rate2:.2f}")
        ns_len2 = normalized_sequence_length(reference_tokenizer, corpus)
        print(f"Normalized sequence length:         {ns_len2:.4f}")
        frag_rate2 = subword_fragmentation_rate(reference_tokenizer, corpus)
        print(f"Subword fragmentation rate:         {frag_rate2:.2f}%")
        vocab_cov2 = vocabulary_coverage_rate(reference_tokenizer, corpus)
        print(f"Vocabulary coverage rate:           {vocab_cov2:.2f}%")

    # Optional equivalence metrics if a reference tokenizer is provided
    if reference_tokenizer is not None:
        name2 = reference_tokenizer.__class__.__name__
        print(f"\n=== Token Sequence Equivalence ({name1} vs. {name2}) ===")
        (
            total_pos_matches,
            total_positions,
            positional_rate,
            total_unordered_matches,
            unordered_rate,
            total_word_matches,
            total_words,
            word_match_rate
        ) = token_sequence_equivalence(tokenizer, reference_tokenizer, corpus)
        print(f"Positional match rate: {positional_rate:.2f}% ({total_pos_matches}/{total_positions})")
        print(f"Unordered match rate:  {unordered_rate:.2f}% ({total_unordered_matches}/{total_positions})")
        print(f"Word match rate:       {word_match_rate:.2f}% ({total_word_matches}/{total_words})")

    print("\n=== Tokenization Performance ===")
    perf = tokenization_performance(tokenizer, corpus)
    print(f"Total time:     {perf['total_time_s']:.4f}s")
    print(f"Throughput:     {perf['throughput_tokens_per_s']:.2f} tokens/s")
    print(f"Avg. latency:   {perf['avg_latency_s']:.6f}s per sentence")
    print(f"Peak memory:    {perf['peak_memory_mb']:.2f} MB")

    print("\n=== Training Performance ===")
    train_perf = training_performance(tokenizer, corpus, max_vocab_size)
    print(f"Training time:  {train_perf['train_time_s']:.4f}s")
    print(f"Peak memory:    {train_perf['peak_memory_mb']:.2f} MB")
    print(f"Num. merges:    {int(train_perf['num_merges'])}")

    print("\n=== Zipf Distribution Fit ===")
    zipf_res = zipf_distribution(tokenizer, corpus)
    print(f"Slope:          {zipf_res['slope']:.4f}")
    print(f"Intercept:      {zipf_res['intercept']:.4f}")
    print(f"Correlation:    {zipf_res['correlation']:.4f}")

    # If a second tokenizer was provided...
    if reference_tokenizer is not None:
        name2 = reference_tokenizer.__class__.__name__
        print(f"\n=== Tokenization Performance for {name2} ===")
        perf2 = tokenization_performance(reference_tokenizer, corpus)
        print(f"Total time:     {perf2['total_time_s']:.4f}s")
        print(f"Throughput:     {perf2['throughput_tokens_per_s']:.2f} tokens/s")
        print(f"Avg. latency:   {perf2['avg_latency_s']:.6f}s per sentence")
        print(f"Peak memory:    {perf2['peak_memory_mb']:.2f} MB")

        print(f"\n=== Training Performance for {name2} ===")
        train_perf2 = training_performance(reference_tokenizer, corpus, max_vocab_size)
        print(f"Training time:  {train_perf2['train_time_s']:.4f}s")
        print(f"Peak memory:    {train_perf2['peak_memory_mb']:.2f} MB")
        print(f"Num. merges:    {int(train_perf2['num_merges'])}")

        print(f"\n=== Zipf Distribution Fit for {name2} ===")
        zipf_res2 = zipf_distribution(reference_tokenizer, corpus)
        print(f"Slope:          {zipf_res2['slope']:.4f}")
        print(f"Intercept:      {zipf_res2['intercept']:.4f}")
        print(f"Correlation:    {zipf_res2['correlation']:.4f}")
