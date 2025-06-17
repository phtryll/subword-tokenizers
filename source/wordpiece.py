import os
import json
from source.utils import SubwordTokenizer, WPTrie_E2E
from collections import Counter
from typing import List, Tuple, Dict

class NaiveWP(SubwordTokenizer):
    """
    A naive subword tokenizer based on the WordPiece algorithm.

    Implements WordPiece training and encoding without optimization, using a simple
    scoring-based merge strategy and '##' prefix notation for subword tokens.
    """

    def __init__(self, tokenizer):
        '''
        Initialize the tokenizer.
            tokenizer (AutoTokenizer): helper tokenizer for preprocessing.
        '''

        # Initialize the parent class
        super().__init__(tokenizer)
        # For lazy/incremental training
        self.vocab: set[str] = set()
        # Hold (symbol_sequence, frequency) across calls
        self.corpus_as_symbols: List[Tuple[List[str], int]] = []

    def train(self, corpus, max_vocab: int = 30_000):
        """
        Train the WordPiece tokenizer on the input corpus.

        Args:
            corpus (List[str]): A list of input strings.
            max_vocab (int): The maximum number of vocabulary entries.
        """

        if not isinstance(corpus, list) or not all(isinstance(example, str) for example in corpus):
            raise TypeError("corpus must be a list of strings.")

        if not isinstance(max_vocab, int):
            raise TypeError("max_vocab must be an int.")

        self.reset()

        # Preprocess the corpus using parent method
        prepd_corpus = super().preprocessing(corpus)
        # Accumulate data into state
        corpus_as_words = [word for example in prepd_corpus for word, position in example]
        word_freqs = Counter(corpus_as_words)
        
        # Represent each word as a list of symbols
        new_symbols = [
            # Prefix all but the first with ##
            ([word[0]] + [f"##{c}" for c in word[1:]], freq)
            for word, freq in word_freqs.items()
        ]
        self.corpus_as_symbols.extend(new_symbols)

        # Initialize the vocabulary with all symbols (allow incremental training)
        initial_symbols = {s for symbols, _ in new_symbols for s in symbols}
        self.vocab |= initial_symbols
        
        # -- Main loop - initialize until max_vocab is reached
        while len(self.vocab) < max_vocab:
            # Get frequencies of adjacent symbol pairs
            pair_freqs = Counter()
            for symbols, freq in self.corpus_as_symbols:
                for i in range(len(symbols) - 1):
                    pair_freqs[(symbols[i], symbols[i+1])] += freq
            if not pair_freqs:
                break
                
            # Count frequencies of individual symbols
            symbol_freqs = Counter()
            for symbols, freq in self.corpus_as_symbols:
                for s in symbols:
                    symbol_freqs[s] += freq

            # Calculate scores for each pair
            scores = {
                pair: freq / (symbol_freqs[pair[0]] * symbol_freqs[pair[1]])
                for pair, freq in pair_freqs.items()
            }
            if not scores:
                break

            # Choose pair with highest score
            best_pair = max(scores, key=scores.get) # type:ignore

            # Add merged token to vocab
            merged_token = best_pair[0] + best_pair[1][2:]
            self.vocab.add(merged_token)
            # Replace the pair in corpus
            self.corpus_as_symbols = [
                (self._replace_pair(best_pair, symbols), freq)
                for symbols, freq in self.corpus_as_symbols
            ]

    def _replace_pair(self, pair, word):
        """
        Replace a pair of symbols with a merged token.

        Args:
            pair (Tuple[str, str]): The pair of symbols to be merged.
            word (List[str]): The input word as a list of symbols.

        Returns:
            List[str]: The updated word after merging the pair.
        """

        merged = pair[0] + pair[1][2:]
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

    def encode_word(self, word):
        """
        Encode a single word into subword tokens using the vocabulary.

        Args:
            word (str): Input word to tokenize.

        Returns:
            List[str]: A list of subword tokens, or ["[UNK]"] if not tokenizable.
        """
        tokens = []

        # Try longest possible substring in vocabulary
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in self.vocab:
                i -= 1        
            if i == 0:
                return ["[UNK]"]

            tokens.append(word[:i])
            word = word[i:]
    
            # Prepend '##' if inside word
            if len(word) > 0:
                word = f"##{word}"

        return tokens

    def tokenize(self, text):
        """
        Tokenize input text using WordPiece tokenization.

        Args:
            text (str): Input text.

        Returns:
            List[str]: A flat list of subword tokens.
        """

        if not isinstance(text, str):
            raise TypeError("Text to tokenize must be a string.")

        # Preprocess input text
        pre_tokenized_corpus = self.preprocessing([text])
        pre_tokenized_text = [word for word, offset in pre_tokenized_corpus[0]]
        # Encode each word and flatten the result
        encoded_words: List[List[str]] = [self.encode_word(word) for word in pre_tokenized_text]
        return [tok for sub in encoded_words for tok in sub]

    def reset(self) -> None:
        """Reset all learned state."""
        self.vocab.clear()
        self.corpus_as_symbols.clear()
    
    def save_resources(self, path: str) -> None:
        """
        Save the current vocabulary to JSON files.

        Args:
            path (str): Directory path where 'vocab.json' will be saved.
        """
        os.makedirs(path, exist_ok=True)
        vocab_file = os.path.join(path, "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(list(self.vocab), f, ensure_ascii=False)

    def load_resources(self, path: str) -> None:
        """
        Load vocabulary from JSON files in the specified directory.

        Args:
            path (str): Directory path from which 'vocab.json' will be loaded.
        """
        vocab_file = os.path.join(path, "vocab.json")
        if os.path.isfile(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                self.vocab = set(json.load(f))


class FastWP(NaiveWP):
    """
    A fast, boundary-aware WordPiece tokenizer with end-to-end punctuation handling.

    Extends Naive_WP by leveraging a trie structure (WPTrie) to enable
    linear-time subword tokenization via longest-prefix matching.
    """

    def __init__(self, tokenizer):
        '''
        Initialize the FastWP tokenizer.
            tokenizer (AutoTokenizer): A tokenizer instance to assist with preprocessing.
        '''
        # Initialize the parent class
        super().__init__(tokenizer)

    def train(self, corpus, max_vocab=30_000):
        # Train using the NaiveWP logic to build the vocabulary
        super().train(corpus, max_vocab)
        # Build the custom E2E trie using the trained vocabulary
        self.vocab_trie = WPTrie_E2E(self.vocab)

    def tokenize(self, text):
        """
        Tokenize text with end-to-end WordPiece using punctuation-aware trie.

        Args:
            text (str): Input text.

        Returns:
            List[str]: A list of subword tokens, honoring word and punctuation boundaries.
        """
        if not isinstance(text, str):
            raise TypeError("Text to tokenize must be a string.")

        # Prepare: lowercase the input and append a space to mark the end
        result = []
        s = text.lower() + " "
        i = 0

        while i < len(s):
            # Step 1: Use the trie to match as many tokens as possible from position i
            tokens, u, i = self.matchloop(s, i)
            # Step 2: Validate if the match ends at a word boundary and at an allowed node
            if not self.iswdbndry(s, i) or u not in {self.vocab_trie.root, self.vocab_trie.root_sharp, self.vocab_trie.root_p}:
                # If not a valid boundary, treat as unknown
                tokens = ["['UNK']"]
            else:
                # If at root_sharp and no tokens, fallback to encoding "##"
                if u == self.vocab_trie.root_sharp and len(tokens) == 0:
                    tokens = super().encode_word("##")
            # Step 3: Add the tokens from this segment to the result
            result.extend(tokens)
            # Step 4: Advance i to the next word/punctuation boundary
            while i < len(s) and not self.iswdbndry(s, i):
                i = i + 1
            # Step 5: Skip whitespace
            while i < len(s) and s[i].isspace():
                i = i + 1
        return result

    def iswdbndry(self, s, i):
        """
        Check if the current index is at a word or punctuation boundary.

        Args:
            s (str): Input string.
            i (int): Index in the string.

        Returns:
            bool: True if it's a word boundary, else False.
        """
        # At or past end of string, or previous char is not alnum, or current char is space or not alnum
        return i > len(s) or (i > 0 and not s[i - 1].isalnum() or s[i].isspace() or not s[i].isalnum())


    def matchloop(self, s: str, i: int):
        """
        Traverse the WordPiece trie to find all matching subword tokens.

        Args:
            s (str): Input string with trailing space.
            i (int): Starting index.

        Returns:
            Tuple[List[str], TrieNode, int]: Tokens found, final node, and end index.
        """
        node = self.vocab_trie.root
        tokens: List[str] = []
        while i < len(s):
            # no outgoing edge, follow failure links
            while s[i] not in node.children:
                if node.failure_link is None:
                    return tokens, node, i
                tokens.extend(node.failure_pops)
                node = node.failure_link
            node = node.children[s[i]]
            i = i + 1
        return tokens, node, i

    def load_resources(self, path: str) -> None:
        """
        Load WordPiece merges and vocabulary, and rebuild vocabulary trie for FastWP.
        """
        super().load_resources(path)
        # Rebuild vocabulary trie
        self.vocab_trie = WPTrie_E2E(self.vocab)

    def save_resources(self, path: str) -> None:
        """
        Save WordPiece merges and vocabulary using the NaiveWP implementation.
        """
        super().save_resources(path)
