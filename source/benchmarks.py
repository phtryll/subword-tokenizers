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

from timeit import default_timer as timer
from collections import Counter
from typing import List, Tuple, Dict, Any


def avg_tokens_per_sentence(tokenized_sents: List[List[str]]) -> float:
    """
    Compute the average number of tokens per sentence from pre-tokenized data.
    Args:
        tokenized_sents (List[List[str]]): List of tokenized sentences.
    Returns:
        float: Average number of tokens per sentence.
    """
    if not tokenized_sents:
        return 0.0
    return sum(len(ts) for ts in tokenized_sents) / len(tokenized_sents)


def avg_tokens_per_word(tokenized_words: Dict[str, List[str]]) -> float:
    """
    Compute the average number of subword tokens per word from pre-tokenized words.
    Args:
        tokenized_words (Dict[str, List[str]]): Mapping from word to its tokenized form.
    Returns:
        float: Average number of tokens per word.
    """
    if not tokenized_words:
        return 0.0
    return sum(len(ws) for ws in tokenized_words.values()) / len(tokenized_words)


def normalized_sequence_length(total_tokens: int, total_chars: int) -> float:
    """
    Compute normalized sequence length: total tokens divided by total characters.
    Args:
        total_tokens (int): Total number of tokens.
        total_chars (int): Total number of characters.
    Returns:
        float: Total tokens divided by total characters.
    """
    return total_tokens / total_chars if total_chars else float('inf')


def subword_fragmentation_rate(tokenized_words: Dict[str, List[str]]) -> float:
    """
    Compute the subword fragmentation rate from pre-tokenized words.
    Args:
        tokenized_words (Dict[str, List[str]]): Mapping from word to its tokenized form.
    Returns:
        float: Percentage of unique words split into multiple subword tokens.
    """
    if not tokenized_words:
        return 0.0
    split_words = sum(1 for ws in tokenized_words.values() if len(ws) > 1)
    return split_words / len(tokenized_words) * 100


def vocabulary_coverage_rate(tokenized_words: Dict[str, List[str]]) -> float:
    """
    Compute vocabulary coverage from pre-tokenized words.
    Args:
        tokenized_words (Dict[str, List[str]]): Mapping from word to its tokenized form.
    Returns:
        float: Percentage of unique words tokenized as exactly one token.
    """
    if not tokenized_words:
        return 0.0
    covered = sum(1 for ws in tokenized_words.values() if len(ws) == 1)
    return covered / len(tokenized_words) * 100


def compression_rate(total_chars: int, tokenized_sents: List[List[str]]) -> float:
    """
    Compute the compression rate: ratio of total non-space characters
    to total number of subword tokens, using pre-tokenized sentences.
    Args:
        total_chars (int): Total number of non-space characters.
        tokenized_sents (List[List[str]]): List of tokenized sentences.
    Returns:
        float: Compression rate (characters per subword token).
    """
    total_subs = sum(len(ts) for ts in tokenized_sents)
    return total_chars / total_subs if total_subs else float('inf')


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
    # Measure memory before and after
    start_time = timer()
    all_tokens = [tokenizer.tokenize(sentence) for sentence in input]
    end_time = timer()

    total_time = end_time - start_time
    total_tokens = sum(len(tokens_in_sentence) for tokens_in_sentence in all_tokens)
    throughput = total_tokens / total_time if total_time > 0 else float('inf')
    avg_latency = total_time / len(input) if input else 0.0

    return {
        "total_time_s": total_time,
        "throughput_tokens_per_s": throughput,
        "avg_latency_s": avg_latency,
    }


def training_performance(tokenizer: Any, test_corpus: List[str], max_vocab_size: int) -> Dict[str, float]:
    """
    Measure training speed and memory usage for a tokenizer.

    Args:
        tokenizer (Any): Tokenizer instance with a `train` method.
        test_corpus (List[str]): List of training sentences.
        max_vocab_size (int): Maximum vocabulary size for training.

    Returns:
        Dict[str, float]: Metrics including training time, peak memory in MB, and number of merges.
    """
    start_time = timer()
    tokenizer.train(test_corpus, max_vocab_size)
    end_time = timer()

    if hasattr(tokenizer, 'merges_list'): #! `merges_list` is the name in naïve BPE
        num_merges = len(tokenizer.merges_list)
    elif hasattr(tokenizer, 'merges'): #! `merges` is the name in fast BPE
        num_merges = len(tokenizer.merges)
    else:
        num_merges = 0

    return {
        "train_time_s": end_time - start_time,
        "num_merges": float(num_merges)
    }


def zipf_distribution(tokenized_sents: List[List[str]]) -> Dict[str, float]:
    """
    Analyze token frequency distribution and compute Zipf's law fit from pre-tokenized sentences.
    Args:
        tokenized_sents (List[List[str]]): List of tokenized sentences.
    Returns:
        Dict[str, float]: Contains 'slope', 'intercept', and 'correlation' of the log-log fit.
    """
    all_tokens = [tok for seq in tokenized_sents for tok in seq]
    frequency = Counter(all_tokens)
    frequency_sorted = [count for _, count in frequency.most_common()]
    ranks = list(range(1, len(frequency_sorted) + 1))
    import math
    log_ranks = [math.log(r) for r in ranks]
    log_freqs = [math.log(f) for f in frequency_sorted]
    n = len(ranks)
    mean_x = sum(log_ranks) / n
    mean_y = sum(log_freqs) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_ranks, log_freqs))
    var_x = sum((x - mean_x) ** 2 for x in log_ranks)
    var_y = sum((y - mean_y) ** 2 for y in log_freqs)
    slope = cov / var_x if var_x else 0.0
    intercept = mean_y - slope * mean_x
    correlation = cov / math.sqrt(var_x * var_y) if var_x and var_y else 0.0
    return {"slope": slope, "intercept": intercept, "correlation": correlation}


def benchmarks(
    tokenizer: Any,
    test_corpus: List[str],
    max_vocab_size: int,
    train_corpus: List[str] = [],
    reference_tokenizers: List[Any] = [],
    pretrained: bool = False,
    pretrained_path: str = "",
    compare_only: bool = False
) -> None:
    """
    Run all benchmark functions and print results to the console.

    Args:
        tokenizer (Any): Tokenizer with a `tokenize` and `train` method.
        test_corpus (List[str]): List of input sentences.
        max_vocab_size (int): Maximum vocabulary size for training.
        train_corpus (List[str]): List of training sentences.
        reference_tokenizers (List[Any], optional): List of additional tokenizers for equivalence and comparison metrics.
    """

    # Determine tokenizer name
    name1 = tokenizer.__class__.__name__
    # Compare-only mode: just token-sequence equivalence
    if pretrained and compare_only:
        if not reference_tokenizers:
            print("No reference tokenizers provided for comparison.")
            return
        for opt_tok in reference_tokenizers:
            name2 = opt_tok.__class__.__name__
            (
                total_pos, total_positions, pos_rate,
                total_unord, unord_rate,
                total_wmatch, total_words, wmatch_rate
            ) = token_sequence_equivalence(tokenizer, opt_tok, test_corpus)
            print(f"=== Token Sequence Equivalence ({name1} vs {name2}) ===")
            print(f"Positional match rate: {pos_rate:.2f}% ({total_pos}/{total_positions})")
            print(f"Unordered match rate:  {unord_rate:.2f}% ({total_unord}/{total_positions})")
            print(f"Word match rate:       {wmatch_rate:.2f}% ({total_wmatch}/{total_words})")
        return

    if pretrained:
        # Tokenization-only mode
        tokenizer.load_resources(pretrained_path)
        # Pre-tokenize once for this tokenizer
        tokenized_sents = [tokenizer.tokenize(s) for s in test_corpus]
        unique_words = {w for s in test_corpus for w in s.split()}
        tokenized_words = {w: tokenizer.tokenize(w) for w in unique_words}
        total_chars = sum(len(s.replace(' ', '')) for s in test_corpus)
        total_tokens = sum(len(ts) for ts in tokenized_sents)

        print(f"=== Tokenization Metrics for {name1} ===")
        print(f"Average tokens per sentence:        {avg_tokens_per_sentence(tokenized_sents):.2f}")
        print(f"Average tokens per word:            {avg_tokens_per_word(tokenized_words):.2f}")
        print(f"Compression rate (chars per token): {compression_rate(total_chars, tokenized_sents):.2f}")
        print(f"Normalized sequence length:         {normalized_sequence_length(total_tokens, total_chars):.4f}")
        print(f"Subword fragmentation rate:         {subword_fragmentation_rate(tokenized_words):.2f}%")
        print(f"Vocabulary coverage rate:           {vocabulary_coverage_rate(tokenized_words):.2f}%")

        print("\n=== Tokenization Performance ===")
        perf = tokenization_performance(tokenizer, test_corpus)
        print(f"Total time:     {perf['total_time_s']:.4f}s")
        print(f"Throughput:     {perf['throughput_tokens_per_s']:.2f} tokens/s")
        print(f"Avg. latency:   {perf['avg_latency_s']:.6f}s per sentence")
        print(f"Peak memory:    {perf['peak_memory_mb']:.2f} MB")

        print("\n=== Zipf Distribution Fit ===")
        zipf_res = zipf_distribution(tokenized_sents)
        print(f"Slope:          {zipf_res['slope']:.4f}")
        print(f"Intercept:      {zipf_res['intercept']:.4f}")
        print(f"Correlation:    {zipf_res['correlation']:.4f}")

        # Compare additional tokenizers on tokenization metrics
        if reference_tokenizers:
            for opt_tok in reference_tokenizers:
                name2 = opt_tok.__class__.__name__
                opt_tok.load_resources(pretrained_path)
                # Pre-tokenize once for this reference tokenizer
                tokenized_sents2 = [opt_tok.tokenize(s) for s in test_corpus]
                unique_words2 = {w for s in test_corpus for w in s.split()}
                tokenized_words2 = {w: opt_tok.tokenize(w) for w in unique_words2}
                total_chars2 = total_chars
                total_tokens2 = sum(len(ts) for ts in tokenized_sents2)

                print(f"\n=== Tokenization Metrics for {name2} ===")
                print(f"Average tokens per sentence:        {avg_tokens_per_sentence(tokenized_sents2):.2f}")
                print(f"Average tokens per word:            {avg_tokens_per_word(tokenized_words2):.2f}")
                print(f"Compression rate (chars per token): {compression_rate(total_chars2, tokenized_sents2):.2f}")
                print(f"Normalized sequence length:         {normalized_sequence_length(total_tokens2, total_chars2):.4f}")
                print(f"Subword fragmentation rate:         {subword_fragmentation_rate(tokenized_words2):.2f}%")
                print(f"Vocabulary coverage rate:           {vocabulary_coverage_rate(tokenized_words2):.2f}%")

                print("\n=== Tokenization Performance ===")
                perf2 = tokenization_performance(opt_tok, test_corpus)
                print(f"Total time:     {perf2['total_time_s']:.4f}s")
                print(f"Throughput:     {perf2['throughput_tokens_per_s']:.2f} tokens/s")
                print(f"Avg. latency:   {perf2['avg_latency_s']:.6f}s per sentence")
                print(f"Peak memory:    {perf2['peak_memory_mb']:.2f} MB")

                print("\n=== Zipf Distribution Fit ===")
                zipf_res2 = zipf_distribution(tokenized_sents2)
                print(f"Slope:          {zipf_res2['slope']:.4f}")
                print(f"Intercept:      {zipf_res2['intercept']:.4f}")
                print(f"Correlation:    {zipf_res2['correlation']:.4f}")

    else:
        # Training-only mode
        if not train_corpus:
            raise ValueError("train_corpus is required for training metrics.")
        print(f"=== Training Performance for {name1} ===")
        train_perf = training_performance(tokenizer, train_corpus, max_vocab_size)
        print(f"Training time:  {train_perf['train_time_s']:.4f}s")
        print(f"Peak memory:    {train_perf['peak_memory_mb']:.2f} MB")
        print(f"Num. merges:    {int(train_perf['num_merges'])}")

        # Compare additional tokenizers on training performance
        if reference_tokenizers:
            for opt_tok in reference_tokenizers:
                name2 = opt_tok.__class__.__name__
                print(f"\n=== Training Performance for {name2} ===")
                train_perf2 = training_performance(opt_tok, train_corpus, max_vocab_size)
                print(f"Training time:  {train_perf2['train_time_s']:.4f}s")
                print(f"Peak memory:    {train_perf2['peak_memory_mb']:.2f} MB")
                print(f"Num. merges:    {int(train_perf2['num_merges'])}")
