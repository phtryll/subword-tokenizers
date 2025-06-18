import os
import json
from source.utils import SubwordTokenizer
from itertools import chain
from collections import Counter, defaultdict
from typing import List, Tuple, Dict
from tqdm import tqdm

class NaiveBPE(SubwordTokenizer):
    """A subword tokenizer based on the Byte-Pair Encoding (BPE) algorithm."""

    def __init__(self, tokenizer) -> None:
        """
        Initialize the Naive_BPE tokenizer.

        Args:
            tokenizer (AutoTokenizer): A tokenizer instance for preprocessing.
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

    def train(self, corpus: List[str], max_vocab: int = 30_000) -> None:
        """
        Train the BPE tokenizer on the corpus.

        Args:
            corpus (List[str]): A list of strings used to train the tokenizer.
            max_vocab (int): The maximum size of the subword vocabulary.

        This method modifies the internal state by building the merge list; it does not return anything.
        """
        # 1. Input validation
        if not isinstance(corpus, list) or not all(isinstance(example, str) for example in corpus):
            raise TypeError("Corpus must be a list of strings.")

        if not isinstance(max_vocab, int):
            raise TypeError("Maximum vocabulary size must be an integer.")
        
        self.reset()

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

        # Progress bar setup
        initial_vocab_size = len(self.vocab)
        pbar = tqdm(total=max_vocab - initial_vocab_size, desc="Training BPE")
        # 4. Iteratively merge the most frequent symbol pairs
        while len(self.vocab) < max_vocab:
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
            pbar.update(1)

            # 4.3 Apply merge to entire corpus
            self.corpus_as_symbols = [
                (self._replace_pair(most_frequent_pair, seq), freq)
                for seq, freq in self.corpus_as_symbols
            ]
        pbar.close()

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

        if len(word_split)>1:
          word_split[1::] = ['##' + s for s in word_split[1::]]

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


    def save_resources(self, path: str) -> None:
        """
        Save the learned BPE merges to JSON files in the given directory.

        Args:
            path (str): Directory where 'merges.json' will be written.
        """
        os.makedirs(path, exist_ok=True)
        merges_file = os.path.join(path, "merges.json")
        with open(merges_file, "w", encoding="utf-8") as f:
            json.dump(self.merges_list, f, ensure_ascii=False)

    def load_resources(self, path: str) -> None:
        """
        Load BPE merges from JSON files in the specified directory.

        Args:
            path (str): Directory from which 'merges.json' will be read.
        """
        merges_file = os.path.join(path, "merges.json")
        if os.path.isfile(merges_file):
            with open(merges_file, "r", encoding="utf-8") as f:
                self.merges_list = [tuple(pair) for pair in json.load(f)]


class FastBPE(NaiveBPE):
    """Faster inference-only subclass"""
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self._bpe_ranks: Dict[Tuple[str, str], int] = {}

    def train(self, corpus: List[str], max_vocab: int = 30_000) -> None:
        super().train(corpus, max_vocab)
        self._bpe_ranks = {pair: i for i, pair in enumerate(self.merges_list)}

    def _pairs(self, seq: List[str]) -> set[Tuple[str, str]]:
        return {(seq[i], seq[i+1]) for i in range(len(seq)-1)}

    def encode_word(self, word: str) -> List[str]:
        symbols = list(word)
        if len(symbols) < 2:
            return symbols or [""]

        pairs = self._pairs(symbols)
        while True:
            best_pair = None
            best_rank = float("inf")
            for p in pairs:
                r = self._bpe_ranks.get(p)
                if r is not None and r < best_rank:
                    best_rank, best_pair = r, p
            if best_pair is None:
                break

            merged = best_pair[0] + best_pair[1]
            new_syms: List[str] = []
            i = 0
            while i < len(symbols):
                if (
                    i < len(symbols) - 1
                    and symbols[i] == best_pair[0]
                    and symbols[i + 1] == best_pair[1]
                ):
                    new_syms.append(merged)
                    i += 2
                else:
                    new_syms.append(symbols[i])
                    i += 1
            symbols = new_syms
            if len(symbols) == 1:
                break
            pairs = self._pairs(symbols)

        if len(symbols)>1:
          symbols[1::] = ['##' + s for s in symbols[1::]]

        return symbols

    def tokenize(self, text: str) -> List[str]:
        if not isinstance(text, str):
            raise TypeError("Text must be a string.")
        pre = [w for w, _ in self.preprocessing([text])[0]]
        return [tok for w in pre for tok in self.encode_word(w)]

    def load_resources(self, path: str) -> None:
        """
        Load BPE merges and vocabulary, and rebuild BPE ranks for FastBPE.
        """
        super().load_resources(path)
        # Rebuild BPE ranks for inference
        self._bpe_ranks = {pair: i for i, pair in enumerate(self.merges_list)}

    def save_resources(self, path: str) -> None:
        """
        Save BPE merges and vocabulary using the NaiveBPE implementation.
        """
        super().save_resources(path)
