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
        # for lazy/incremental training
        self.merges_list: List[Tuple[str,str]] = []
        self.vocab: set[str] = set()
        self.corpus_as_symbols: List[Tuple[List[str], int]] = []

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

        # 3. Update vocabulary & corpus symbols with new data
        new_words = [word for example in processed_corpus for word, _ in example]
        # expand vocab
        self.vocab.update({ch for w in new_words for ch in w})
        # count new word frequencies
        new_freqs = Counter(new_words)
        # convert into list-of-symbols form and append
        for word, freq in new_freqs.items():
            symbols = [s for s in word]
            self.corpus_as_symbols.append((symbols, freq))
        # self.corpus_as_symbols contains past and new data now

        # 4. Iteratively merge the most frequent symbol pairs
        while len(self.vocab) < max_vocab_size:
            # 4.1 Count symbol pair frequencies
            get_pair_freqs = Counter(
                (seq[i], seq[i+1])
                for seq, freq in self.corpus_as_symbols
                for i in range(len(seq) - 1)
                for _ in range(freq)
            )

            # 4.1b Exit if no more pairs can be merged
            if not get_pair_freqs:
                break

            # 4.2 Find most frequent pair and merge
            most_frequent_pair = get_pair_freqs.most_common(1)[0][0]
            self.vocab.add(most_frequent_pair[0] + most_frequent_pair[1])
            self.merges_list.append(most_frequent_pair)

            # 4.3 Apply merge to entire corpus
            self.corpus_as_symbols = [
                (self._replace_pair(most_frequent_pair, seq), freq)
                for seq, freq in self.corpus_as_symbols
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

    def reset(self) -> None:
        """Reset all learned state."""
        self.merges_list.clear()
        self.vocab.clear()
        self.corpus_as_symbols.clear()

class FastBPE(NaiveBPE):
    """
    A subword tokenizer using Byte-Pair Encoding (BPE) with Trie-based lookup.
    Learns merges from a training corpus and uses a Trie for greedy longest-prefix
    matching during tokenization.
    """
    def __init__(self, tokenizer: PreTrainedTokenizerFast) -> None:
        super().__init__(tokenizer)
        self.trie = Trie()  # built in training

    def train(self, corpus: List[str], max_vocab_size: int = 30_000) -> None:
        """
        Train the BPE tokenizer on the input corpus.

        Args:
            corpus (List[str]): A list of strings to learn subword merges from.
            max_vocab (int): The maximum size of the subword vocabulary.
        """
        super().train(corpus, max_vocab_size)
        # Build Trie over vocabulary
        self.trie = Trie()
        for token in self.vocab:
            self.trie.insert(list(token), token)

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
        # Try fast matching first
        out = self._encode_seq(list(word))
        if any(tok not in self.vocab for tok in out):
            return super().encode_word(word)
        return out

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

    def reset(self) -> None:
        """Reset all learned state."""
        # Clear BPE state
        super().reset()
        # Rebuild an empty trie
        self.trie = Trie()