import math
from timeit import default_timer as timer
from collections import Counter
from typing import List, Tuple, Dict, Any


def avg_tokens_per_sentence(tokenized_inputs: List[List[str]]) -> float:
    """
    Compute the average number of tokens per sentence from pre-tokenized data.
    Args:
        tokenized_inputs (List[List[str]]): List of tokenized sentences.
    Returns:
        float: Average number of tokens per sentence.
    """
    
    # Check the input
    if not tokenized_inputs:
        return 0.0
    
    # Take the sum of the number subword tokens in each input and divide it by the number of inputs
    return sum(len(tokenized_input) for tokenized_input in tokenized_inputs) / len(tokenized_inputs)


def avg_tokens_per_word(tokenized_words: Dict[str, List[str]]) -> float:
    """
    Compute the average number of subword tokens per word from pre-tokenized words.
    Args:
        tokenized_words (Dict[str, List[str]]): Mapping from word to its tokenized form.
    Returns:
        float: Average number of tokens per word.
    """

    # Check the input
    if not tokenized_words:
        return 0.0
    
    # Take the sum of the number of subword tokens in each input and divide it by the number of words
    return sum(len(tokens) for tokens in tokenized_words.values()) / len(tokenized_words)


def normalized_sequence_length(total_tokens: int, total_chars: int) -> float:
    """
    Compute normalized sequence length: total tokens divided by total characters.
    Args:
        total_tokens (int): Total number of tokens.
        total_chars (int): Total number of characters.
    Returns:
        float: Total tokens divided by total characters.
    """

    # Average number of tokens per character
    return total_tokens / total_chars if total_chars else float('inf')


def subword_fragmentation_rate(tokenized_words: Dict[str, List[str]]) -> float:
    """
    Compute the subword fragmentation rate from pre-tokenized words.
    Args:
        tokenized_words (Dict[str, List[str]]): Mapping from word to its tokenized form.
    Returns:
        float: Percentage of unique words split into multiple subword tokens.
    """

    # Check the input
    if not tokenized_words:
        return 0.0
    
    # Number of words split in subword tokens
    split_words = sum(1 for tokens in tokenized_words.values() if len(tokens) > 1)
    
    # Compute percentage
    return split_words / len(tokenized_words) * 100


def vocabulary_coverage_rate(tokenized_words: Dict[str, List[str]]) -> float:
    """
    Compute vocabulary coverage from pre-tokenized words.
    Args:
        tokenized_words (Dict[str, List[str]]): Mapping from word to its tokenized form.
    Returns:
        float: Percentage of unique words tokenized as exactly one token.
    """

    # Check the input
    if not tokenized_words:
        return 0.0
    
    # Number of full words tokenized as such
    covered = sum(1 for tokens in tokenized_words.values() if len(tokens) == 1)
    
    # Compute percentage
    return covered / len(tokenized_words) * 100


def compression_rate(total_chars: int, tokenized_inputs: List[List[str]]) -> float:
    """
    Compute the compression rate: ratio of total non-space characters
    to total number of subword tokens, using pre-tokenized sentences.
    Args:
        total_chars (int): Total number of non-space characters.
        tokenized_inputs (List[List[str]]): List of tokenized sentences.
    Returns:
        float: Compression rate (characters per subword token).
    """

    # Average number of characters per subword token
    total_subword_tokens = sum(len(tokenized_input) for tokenized_input in tokenized_inputs)
    
    # Compute average
    return total_chars / total_subword_tokens if total_subword_tokens else float('inf')


def token_sequence_equivalence(tokenizer1: Any, tokenizer2: Any, input: List[str] ) -> Tuple[int, int, float, int, float, int, int, float]:
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
    
    # Initialize counters for all metrics
    total_pos_matches = 0
    total_positions = 0
    total_unordered_matches = 0
    total_words = 0
    total_word_matches = 0

    # Iterate through each sentence in the input
    for sentence in input:
        # Tokenize with both tokenizers and strip '##' for WordPice subword tokens
        raw1 = tokenizer1.tokenize(sentence)
        raw2 = tokenizer2.tokenize(sentence)
        tokens1 = [token[2:] if token.startswith("##") else token for token in raw1]
        tokens2 = [token[2:] if token.startswith("##") else token for token in raw2]

        # Compare tokens at same positions (positional matches)
        n = min(len(tokens1), len(tokens2)) # Compare up to the length of the smallest sequence
        total_pos_matches += sum(1 for i in range(n) if tokens1[i] == tokens2[i])
        total_positions += n

        # Compare unordered token overlap (regardless of position)
        freq1 = Counter(tokens1)
        freq2 = Counter(tokens2)
        overlap = sum(min(freq1[token], freq2[token]) for token in (freq1.keys() & freq2.keys())) # Again, we take the counts of the smallest sequence
        total_unordered_matches += overlap

        # For each word, check if both tokenizers share at least one subword
        words = sentence.split()
        total_words += len(words)
        for word in words:
            raw_sub1 = tokenizer1.tokenize(word)
            raw_sub2 = tokenizer2.tokenize(word)
            sub1 = [token[2:] if token.startswith("##") else token for token in raw_sub1]
            sub2 = [token[2:] if token.startswith("##") else token for token in raw_sub2]
            if set(sub1) & set(sub2): # Non-empty intersection
                total_word_matches += 1

    # Compute rates as percentages
    positional_rate = (total_pos_matches / total_positions * 100) if total_positions else 0.0
    unordered_rate = (total_unordered_matches / total_positions * 100) if total_positions else 0.0
    word_match_rate = (total_word_matches / total_words * 100) if total_words else 0.0

    # Return all metrics as a tuple
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

    # Time tokenization process
    start_time = timer()
    all_tokens = [tokenizer.tokenize(sentence) for sentence in input]
    end_time = timer()

    # Total time elapsed
    total_time = end_time - start_time
    
    # Total subword tokens computed
    total_tokens = sum(len(tokens_in_sentence) for tokens_in_sentence in all_tokens)
    
    # Compute the number of outputs computed per second
    throughput = total_tokens / total_time if total_time > 0 else float('inf')
    
    # Compute average time to process one input
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

    # Time training process
    start_time = timer()
    tokenizer.train(test_corpus, max_vocab_size)
    end_time = timer()

    return {"train_time_s": end_time - start_time}


def zipf_distribution(tokenized_inputs: List[List[str]]) -> Dict[str, float]:
    """
    Analyze token frequency distribution and compute Zipf's law fit from pre-tokenized sentences.
    Args:
        tokenized_inputs (List[List[str]]): List of tokenized sentences.
    Returns:
        Dict[str, float]: Contains 'slope', 'intercept', and 'correlation' of the log-log fit.
    """
    
    # Flatten the list of tokenized inputs to a single list of tokens
    all_tokens = [token for sentence in tokenized_inputs for token in sentence]

    # Count frequency of each token
    token_frequencies = Counter(all_tokens)

    # Sort frequencies in descending order
    sorted_frequencies = [count for _, count in token_frequencies.most_common()]

    # Assign rank to each token based on its frequency
    token_ranks = list(range(1, len(sorted_frequencies) + 1))

    # Take logarithm of ranks and frequencies for log-log analysis
    log_ranks = [math.log(rank) for rank in token_ranks]
    log_frequencies = [math.log(freq) for freq in sorted_frequencies]

    # Calculate mean of log values
    num_tokens = len(token_ranks)
    mean_log_rank = sum(log_ranks) / num_tokens
    mean_log_freq = sum(log_frequencies) / num_tokens

    # Calculate covariance and variances
    covariance = sum((x - mean_log_rank) * (y - mean_log_freq) for x, y in zip(log_ranks, log_frequencies))
    variance_log_rank = sum((x - mean_log_rank) ** 2 for x in log_ranks)
    variance_log_freq = sum((y - mean_log_freq) ** 2 for y in log_frequencies)

    # Compute linear regression slope and intercept (Zipf's law approximation)
    slope = covariance / variance_log_rank if variance_log_rank else 0.0
    intercept = mean_log_freq - slope * mean_log_rank

    # Compute Pearson correlation coefficient
    correlation = covariance / math.sqrt(variance_log_rank * variance_log_freq) if variance_log_rank and variance_log_freq else 0.0

    return {"slope": slope, "intercept": intercept, "correlation": correlation}


def benchmarks(
    tokenizer: Any,
    max_vocab_size: int,
    test_corpus: List[str],
    train_corpus: List[str] = [],
    pretrained: bool = False,
    pretrained_path: str = "",
    reference_tokenizers: List[Any] = [],
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

    # Determine main tokenizer name
    name1 = tokenizer.__class__.__name__

    # Compare-only mode: just token-sequence equivalence
    if pretrained and compare_only:
        
        # Add safety check
        if not reference_tokenizers:
            print("No reference tokenizers provided for comparison.")
            return
        
        # For each secondary tokenizer that was included
        for optional_tokenizer in reference_tokenizers:

            # Determine its name
            name2 = optional_tokenizer.__class__.__name__
            
            # Compute equivalence metrics
            (
                total_positional_matches, total_positions, pos_rate,
                total_unordered_matches, unordered_rate,
                total_word_matches, total_words, word_match_rate
            ) = token_sequence_equivalence(tokenizer, optional_tokenizer, test_corpus)

            print(f"=== Token Sequence Equivalence ({name1} vs {name2}) ===")
            print(f"Positional match rate: {pos_rate:.2f}% ({total_positional_matches}/{total_positions})")
            print(f"Unordered match rate:  {unordered_rate:.2f}% ({total_unordered_matches}/{total_positions})")
            print(f"Word match rate:       {word_match_rate:.2f}% ({total_word_matches}/{total_words})")
        
        return

    # If pretrained data provided: compute only tokenization metrics
    if pretrained:

        # Load the vocab/merges list (depending of the tokenizer)
        tokenizer.load_resources(pretrained_path)

        # Pre-tokenize once for this tokenizer
        tokenized_inputs = [tokenizer.tokenize(sentence) for sentence in test_corpus]
        unique_words = {word for sentence in tokenizer.preprocessing(test_corpus) for word, _ in sentence}
        tokenized_words = {word: tokenizer.tokenize(word) for word in unique_words}
        total_chars = sum(len(s.replace(' ', '')) for s in test_corpus)
        total_tokens = sum(len(tokenized_input) for tokenized_input in tokenized_inputs)

        print(f"=== Tokenization Metrics for {name1} ===")
        print(f"Average tokens per sentence:        {avg_tokens_per_sentence(tokenized_inputs):.2f}")
        print(f"Average tokens per word:            {avg_tokens_per_word(tokenized_words):.2f}")
        print(f"Compression rate (chars per token): {compression_rate(total_chars, tokenized_inputs):.2f}")
        print(f"Normalized sequence length:         {normalized_sequence_length(total_tokens, total_chars):.4f}")
        print(f"Subword fragmentation rate:         {subword_fragmentation_rate(tokenized_words):.2f}%")
        print(f"Vocabulary coverage rate:           {vocabulary_coverage_rate(tokenized_words):.2f}%")

        print("\n=== Tokenization Performance ===")
        perf = tokenization_performance(tokenizer, test_corpus)
        print(f"Total time:     {perf['total_time_s']:.4f}s")
        print(f"Throughput:     {perf['throughput_tokens_per_s']:.2f} tokens/s")
        print(f"Avg. latency:   {perf['avg_latency_s']:.6f}s per sentence")

        print("\n=== Zipf Distribution Fit ===")
        zipf_res = zipf_distribution(tokenized_inputs)
        print(f"Slope:          {zipf_res['slope']:.4f}")
        print(f"Intercept:      {zipf_res['intercept']:.4f}")
        print(f"Correlation:    {zipf_res['correlation']:.4f}")

        # Compare additional tokenizers on tokenization metrics
        if reference_tokenizers:

            # For each secondary tokenizer that was provided
            for optional_tokenizer in reference_tokenizers:

                # Get the name
                name2 = optional_tokenizer.__class__.__name__
                
                # Load the vocab/merges list for this tokenizer
                optional_tokenizer.load_resources(pretrained_path)

                # Pre-tokenize once for this reference tokenizer
                tokenized_inputs2 = [optional_tokenizer.tokenize(sentence) for sentence in test_corpus]
                unique_words2 = {word for sentence in optional_tokenizer.preprocessing(test_corpus) for word, _ in sentence}
                tokenized_words2 = {word: optional_tokenizer.tokenize(word) for word in unique_words2}
                total_chars2 = total_chars
                total_tokens2 = sum(len(tokenized_input) for tokenized_input in tokenized_inputs2)

                print(f"\n=== Tokenization Metrics for {name2} ===")
                print(f"Average tokens per sentence:        {avg_tokens_per_sentence(tokenized_inputs2):.2f}")
                print(f"Average tokens per word:            {avg_tokens_per_word(tokenized_words2):.2f}")
                print(f"Compression rate (chars per token): {compression_rate(total_chars2, tokenized_inputs2):.2f}")
                print(f"Normalized sequence length:         {normalized_sequence_length(total_tokens2, total_chars2):.4f}")
                print(f"Subword fragmentation rate:         {subword_fragmentation_rate(tokenized_words2):.2f}%")
                print(f"Vocabulary coverage rate:           {vocabulary_coverage_rate(tokenized_words2):.2f}%")

                print("\n=== Tokenization Performance ===")
                perf2 = tokenization_performance(optional_tokenizer, test_corpus)
                print(f"Total time:     {perf2['total_time_s']:.4f}s")
                print(f"Throughput:     {perf2['throughput_tokens_per_s']:.2f} tokens/s")
                print(f"Avg. latency:   {perf2['avg_latency_s']:.6f}s per sentence")

                print("\n=== Zipf Distribution Fit ===")
                zipf_res2 = zipf_distribution(tokenized_inputs2)
                print(f"Slope:          {zipf_res2['slope']:.4f}")
                print(f"Intercept:      {zipf_res2['intercept']:.4f}")
                print(f"Correlation:    {zipf_res2['correlation']:.4f}")

    # Training-only mode
    else:

        # Safety check
        if not train_corpus:
            raise ValueError("train_corpus is required for training metrics.")
        
        # Compute training performance metrics
        train_perf = training_performance(tokenizer, train_corpus, max_vocab_size)

        print(f"=== Training Performance for {name1} ===")
        print(f"Training time:  {train_perf['train_time_s']:.4f}s")

        # Compare additional tokenizers on training performance
        if reference_tokenizers:

            # For each secondary tokenizer
            for optional_tokenizer in reference_tokenizers:

                # Get name
                name2 = optional_tokenizer.__class__.__name__

                # Compute performance metrics
                train_perf2 = training_performance(optional_tokenizer, train_corpus, max_vocab_size)

                print(f"\n=== Training Performance for {name2} ===")
                print(f"Training time:  {train_perf2['train_time_s']:.4f}s")
