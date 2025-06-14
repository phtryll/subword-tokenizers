import heapq
from source.utils import SubwordTokenizer, Trie
from collections import Counter, defaultdict
from transformers import PreTrainedTokenizerFast
from typing import List, Tuple, Dict

class NaiveBPE(SubwordTokenizer):
    """A subword tokenizer based on the Byte-Pair Encoding (BPE) algorithm."""

    def __init__(self, tokenizer: PreTrainedTokenizerFast) -> None:
        """
        Initialize the Naive_BPE tokenizer.

        Args:
            tokenizer (PreTrainedTokenizerFast): A tokenizer instance for preprocessing.
        """
        super().__init__(tokenizer)

    def _replace_pair(self, pair: Tuple[str, str], word: List[str]) -> List[str]:
        """
        Replace a pair of symbols with a single merged symbol in a word.

        Args:
            pair (Tuple[str, str]): The symbol pair to merge.
            word (List[str]): The word represented as a list of symbols.

        Returns:
            List[str]: The word after merging the specified symbol pair.
        """
        merged = pair[0] + pair[1]
        new_word = []
        i = 0

        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i + 1]) == pair:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        return new_word

    def train(self, corpus: List[str], max_vocab_size: int = 30_000) -> None:
        """
        Train the BPE tokenizer on the corpus.

        Args:
            corpus (List[str]): A list of strings used to train the tokenizer.
            max_vocab_size (int): The maximum size of the subword vocabulary.

        This method modifies the internal state by building the merge list; it does not return anything.
        """
        # 1. Input validation
        if not isinstance(corpus, list) or not all(isinstance(example, str) for example in corpus):
            raise TypeError("Corpus must be a list of strings.")

        if not isinstance(max_vocab_size, int):
            raise TypeError("Maximum vocabulary size must be an integer.")

        # 2. Preprocess corpus into tokens with character offsets
        processed_corpus = super().preprocessing(corpus)

        # 3. Initialize vocabulary and count word frequencies
        self.merges_list: List[Tuple[str, str]] = []
        corpus_as_words = [word for example in processed_corpus for word, _ in example]
        vocab = {symbol for word in corpus_as_words for symbol in word}
        word_freqs = Counter(corpus_as_words)

        # 4. Convert each word to a list of character symbols
        corpus_as_symbols = [
            ([symbol for symbol in word], frequency)
            for word, frequency in word_freqs.items()
        ]

        # 5. Iteratively merge the most frequent symbol pairs
        while len(vocab) < max_vocab_size:
            # 5.1 Count symbol pair frequencies
            get_pair_freqs = Counter(
                (word[0][i], word[0][i + 1])
                for word in corpus_as_symbols
                for i in range(len(word[0]) - 1)
                for _ in range(word[1])
            )

            # 5.4 Exit if no more pairs can be merged
            if not get_pair_freqs:
                break

            # 5.2 Find most frequent pair and merge
            most_frequent_pair = get_pair_freqs.most_common(1)[0][0]
            vocab.add(most_frequent_pair[0] + most_frequent_pair[1])
            self.merges_list.append(most_frequent_pair)

            # 5.3 Replace symbol pairs in corpus
            corpus_as_symbols = [
                (self._replace_pair(most_frequent_pair, word), frequency)
                for word, frequency in corpus_as_symbols
            ]

    def encode_word(self, word: str) -> List[str]:
        """
        Encode a word into subword tokens using the learned merges.

        Args:
            word (str): The input word to encode.

        Returns:
            List[str]: The encoded subword sequence.
        """
        word_split = [symbol for symbol in word]

        for pair in self.merges_list:
            word_split = self._replace_pair(pair, word_split)

        return word_split

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize input text into subword units.

        Args:
            text (str): The text to tokenize.

        Returns:
            List[str]: A list of subword tokens.
        """
        # 1. Input validation
        if not isinstance(text, str):
            raise TypeError("Text to tokenize must be a string.")

        # 2. Pre-tokenize the input text using the parent class method
        pre_tokenized_corpus = self.preprocessing([text])

        # 3. Extract individual words from the pre-tokenized output
        pre_tokenized_text = [word for word, _ in pre_tokenized_corpus[0]]

        # 4. Encode each word into subword tokens
        encoded_words = [self.encode_word(word) for word in pre_tokenized_text]

        # 5. Flatten the list of subword tokens
        return sum(encoded_words, [])


class TrieBPE(SubwordTokenizer):
    """
    A subword tokenizer using Byte-Pair Encoding (BPE) with Trie-based lookup.

    Learns merges from a training corpus and uses a Trie for greedy longest-prefix
    matching during tokenization.
    """
    def __init__(self, tokenizer, verbose: bool = False):
        super().__init__(tokenizer)
        self.verbose: bool = verbose
        self.logs: List[str] = []
        self.merges: List[Tuple[str, str]] = []  # history of merges
        self.vocab: set[str] = set()
        self.trie = Trie()  # built in training
        # for lazy/incremental loading
        self.temp_corpus: List[List[str]] = []
        self.pair_freq: Counter[Tuple[str,str]] = Counter()
        self.pair_pos: Dict[Tuple[str,str], List[Tuple[int,int]]] = defaultdict(list)
        self.heap: List[Tuple[int, Tuple[str,str]]] = []

    def train(self, corpus: List[str], max_vocab: int = 30_000):
        """
        Train the BPE tokenizer on the input corpus.

        Args:
            corpus (List[str]): A list of strings to learn subword merges from.
            max_vocab (int): The maximum size of the subword vocabulary.
        """
        # 1. Input validation
        if not isinstance(corpus, list) or not all(isinstance(x, str) for x in corpus):
            raise TypeError("Corpus must be a list of strings.")
        if not isinstance(max_vocab, int):
            raise TypeError("max_vocab must be an integer")

        # 2. Preprocess corpus into character sequences
        prepared = super().preprocessing(corpus)
        # Accumulate corpus in a lazy way
        new_corpus = [list(word) for example in prepared for word, _ in example]
        start_idx = len(self.temp_corpus)
        self.temp_corpus.extend(new_corpus)

        # 3. Initialize vocabulary from single characters
        self.vocab.update({ch for word in new_corpus for ch in word})

        # 4. Update symbol pair freqs/positions
        for w_offset, word in enumerate(new_corpus, start=start_idx):
            for p_idx in range(len(word)-1):
                pair = (word[p_idx], word[p_idx+1])
                self.pair_freq[pair] += 1
                self.pair_pos[pair].append((w_offset, p_idx))

        # 5. Build or update heap of symbol pairs by frequency
        if not self.heap:
            self.heap = [(-freq, pair) for pair, freq in self.pair_freq.items()]
            heapq.heapify(self.heap)
        else:
            for pair, freq in self.pair_freq.items():
                heapq.heappush(self.heap, (-freq, pair))

        if self.verbose:
            print(f"[TRAIN] starting merges at heap size {len(self.heap)}")

        # 6. Iteratively perform merges
        while len(self.vocab) < max_vocab and self.heap:
            # 6.1 Skip invalid pairs
            freq_neg, pair = heapq.heappop(self.heap)
            freq = -freq_neg
            # Skip stale entries or pairs too rare
            if self.pair_freq.get(pair, 0) != freq:
                if self.verbose:
                    self.logs.append(f"Skipping {pair} (stale or freq<2)")
                continue
            if freq < 2:
                break

            # 6.2 Merge most frequent pair and update vocabulary
            a, b = pair
            new_sym = a + b  # merged token
            self.vocab.add(new_sym)
            self.merges.append(pair)

            if self.verbose:
                print(f"[MERGE] {pair} -> '{new_sym}', freq={freq}, vocab -> {len(self.vocab)}")

            # 6.3 Update affected words and pair frequencies
            for w_idx, p_idx in self.pair_pos[pair]:
                word = self.temp_corpus[w_idx]
                # Skip if the word has changed already
                if p_idx >= len(word) - 1 or (word[p_idx], word[p_idx + 1]) != pair:
                    continue
                # Remember neighboring pairs before replacement
                left_pair = (word[p_idx - 1], word[p_idx]) if p_idx > 0 else None
                right_pair = (word[p_idx + 1], word[p_idx + 2]) if p_idx + 2 < len(word) else None
                # Replace the two symbols with the merged symbol
                word[p_idx: p_idx + 2] = [new_sym]
                # Decrement counts for old neighbor pairs
                for old in (left_pair, pair, right_pair):
                    if old:
                        self.pair_freq[old] -= 1
                # Increment counts for newly-formed neighbor pairs
                new_left = (word[p_idx - 1], word[p_idx]) if p_idx > 0 else None
                new_right = (word[p_idx], word[p_idx + 1]) if p_idx + 1 < len(word) else None
                for new in (new_left, new_right):
                    if new:
                        self.pair_freq[new] += 1
                        heapq.heappush(self.heap, (-self.pair_freq[new], new))
                        self.pair_pos[new].append((w_idx, p_idx - 1 if new == new_left else p_idx))
            # Mark this pair as consumed
            self.pair_freq[pair] = 0
            self.pair_pos.pop(pair, None)

        if self.verbose:
            print(f"[TRAIN] completed {len(self.merges)} merges. Final vocabulary size: {len(self.vocab)}")

        # 7. Build Trie for encoding
        for tok in self.vocab:
            # Each tok is a string; insert its character sequence
            self.trie.insert(list(tok), tok)

        if self.verbose:
            print("[TRAIN] Trie built for encoding.")

    def _encode_seq(self, seq: List[str]) -> List[str]:
        """
        Encode a list of symbols using greedy longest-prefix matching.

        Args:
            seq (List[str]): A list of character symbols to encode.

        Returns:
            List[str]: The sequence of matched subword tokens.
        """
        out, i = [], 0
        while i < len(seq):
            tok, length = self.trie.longest_match(seq, i)
            if tok is None:
                # no multi-char token matches here: emit single char
                out.append(seq[i])
                i += 1
            else:
                # emit the matched token, advance by its length
                out.append(tok)
                i += length
        return out

    def encode_word(self, word: str) -> List[str]:
        """
        Encode a single word into subword tokens.

        Args:
            word (str): The word to encode.

        Returns:
            List[str]: The subword tokens representing the input word.
        """
        return self._encode_seq(list(word))

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into a flat list of subword tokens.

        Args:
            text (str): The input text to tokenize.

        Returns:
            List[str]: The list of subword tokens extracted from the text.
        """
        if not isinstance(text, str):
            raise TypeError("Text must be a string.")
        
        prepared = self.preprocessing([text])[0]
        # Encode each word in turn and flatten
        return [tok for w, _ in prepared for tok in self.encode_word(w)]