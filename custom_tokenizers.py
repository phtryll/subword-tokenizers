"""Tokenization utilities for subword tokenizers."""

import heapq
from collections import Counter, defaultdict
from transformers import PreTrainedTokenizerFast
from typing import List, Tuple, Dict, Optional


class SubwordTokenizer:
    """A parent class for subword tokenizers."""

    def __init__(self, tokenizer: PreTrainedTokenizerFast) -> None:
        """
        Args:
            tokenizer (PreTrainedTokenizerFast): A tokenizer object, e.g., from Hugging Face AutoTokenizer.
        """
        self.tokenizer = tokenizer

    def preprocessing(self, corpus: List[str]) -> List[List[Tuple[str, Tuple[int, int]]]]:
        """
        Preprocess the input corpus.

        Args:
            corpus (List[str]): A list of sentences to preprocess.

        Returns:
            List[List[Tuple[str, Tuple[int, int]]]]: The preprocessed corpus where each
            sentence is a list of tuples containing a token and its character offset.
        """
        return [
            self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(example.lower())
            for example in corpus
        ]

    def vocab_length(self, corpus: List[str]) -> int:
        """
        Calculate the vocabulary size based on individual symbols in the corpus.

        Args:
            corpus (List[str]): A list of strings used to derive the vocabulary.

        Returns:
            int: The number of unique symbols (characters) in the corpus.
        """
        return len({symbol for example in corpus for symbol in example})


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


class TrieNode(defaultdict):
    """
    A node in the Trie used for storing subword tokens.

    Each node represents a symbol in a token sequence and optionally stores the complete
    token string if it marks the end of a valid subword.
    """
    __slots__ = ("token",)

    def __init__(self, default_factory=None, *args, **kwargs):
        super().__init__(TrieNode, *args, **kwargs)
        # If this node marks the end of valid token, store it here
        self.token: Optional[str] = None


class Trie:
    """
    A prefix tree (Trie) for storing and retrieving subword tokens.

    Supports efficient longest-prefix matching for greedy subword tokenization.
    """
    def __init__(self):
        # Root has no associated token
        self.root = TrieNode()

    def insert(self, token_sequence: List[str], token: str) -> None:
        """
        Insert a token into the trie.

        Args:
            token_sequence (List[str]): A list of symbols representing the token.
            token (str): The full string representation of the token.
        """
        node = self.root
        for sym in token_sequence:
            # Traverse or create child for this symbol
            node = node[sym]
        # Mark this node as end of a valid token
        node.token = token

    def longest_match(self, seq: List[str], start: int = 0) -> Tuple[Optional[str], int]:
        """
        Find the longest matching token starting from a given index.

        Args:
            seq (List[str]): The input sequence of symbols.
            start (int): The index to start matching from.

        Returns:
            Tuple[Optional[str], int]: A tuple containing the matched token (if any)
            and the length of the matched segment in symbols.
        """
        node = self.root
        matched: Optional[str] = None
        length = 0

        # Go down the trie as far as possible
        for i in range(start, len(seq)):
            sym = seq[i]
            if sym not in node:
                # No further path, stop
                break
            node = node[sym]
            if node.token is not None:
                # Found valid token, record it
                matched = node.token
                length = i - start + 1

        return matched, length


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
        # Temporary corpus
        temp_corpus = [list(word) for example in prepared for word, _ in example]

        # 3. Initialize vocabulary from single characters
        self.vocab = {ch for word in temp_corpus for ch in word}

        # 4. Count initial symbol pair frequencies and positions
        pair_freq: Counter[Tuple[str, str]] = Counter()
        pair_pos: Dict[Tuple[str, str], List[Tuple[int, int]]] = defaultdict(list)
        for w_idx, word in enumerate(temp_corpus):
            for p_idx in range(len(word) - 1):
                pair = (word[p_idx], word[p_idx + 1])
                pair_freq[pair] += 1
                pair_pos[pair].append((w_idx, p_idx))

        # 5. Build heap of symbol pairs by frequency
        heap = [(-freq, pair) for pair, freq in pair_freq.items()]
        heapq.heapify(heap)

        if self.verbose:
            print(f"[TRAIN] starting merges at heap size {len(heap)}")

        # 6. Iteratively perform merges
        while len(self.vocab) < max_vocab and heap:
            # 6.1 Skip invalid pairs
            freq_neg, pair = heapq.heappop(heap)
            freq = -freq_neg
            # Skip stale entries or pairs too rare
            if pair_freq.get(pair, 0) != freq:
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
            for w_idx, p_idx in pair_pos[pair]:
                word = temp_corpus[w_idx]
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
                        pair_freq[old] -= 1
                # Increment counts for newly-formed neighbor pairs
                new_left = (word[p_idx - 1], word[p_idx]) if p_idx > 0 else None
                new_right = (word[p_idx], word[p_idx + 1]) if p_idx + 1 < len(word) else None
                for new in (new_left, new_right):
                    if new:
                        pair_freq[new] += 1
                        heapq.heappush(heap, (-pair_freq[new], new))
                        pair_pos[new].append((w_idx, p_idx - 1 if new is new_left else p_idx))
            # Mark this pair as consumed
            pair_freq[pair] = 0
            pair_pos.pop(pair, None)

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


class NaiveWP(SubwordTokenizer):
    """
    A naive subword tokenizer based on the WordPiece algorithm.

    Implements WordPiece training and encoding without optimization, using a simple
    scoring-based merge strategy and '##' prefix notation for subword tokens.
    """

    def __init__(self, tokenizer):
        '''
        Initialize the Naive_WP tokenizer.
            corpus (List[str]): A list of strings used to train the tokenizer.
            tokenizer (AutoTokenizer): A tokenizer instance to assist with preprocessing.
        '''

        # Initialize the parent class
        super().__init__(tokenizer)

    def train(self, corpus, max_vocab_size=30_000):
        """
        Train the WordPiece tokenizer on the input corpus.

        Args:
            corpus (List[str]): A list of input strings.
            max_vocab_size (int): The maximum number of vocabulary entries.
        """

        if not isinstance(corpus, list) or not all(isinstance(example, str) for example in corpus):
            raise TypeError("Corpus must be a list of strings.")

        if not isinstance(max_vocab_size, int):
            raise TypeError("Maximum vocabulary size must be an integer.")

        # Preprocess the corpus using parent method
        prepd_corpus = super().preprocessing(corpus)

        # Flatten the preprocessed corpus into a list of words
        corpus_as_words = [word for example in prepd_corpus for word, position in example]
        # Count frequency of each word
        word_freqs = Counter(corpus_as_words)
        # Represent each word as a list of symbols
        corpus_as_symbols = [
            ([symbol for symbol in word], freq)
            for word, freq in word_freqs.items()
        ]

        # Prefix all but the first with '##'
        for word in corpus_as_symbols:
            word[0][1:] = ['##' + symbol for symbol in word[0][1:]]

        # Initialize the vocabulary with all symbols
        self.vocab = {symbol for word, freq in corpus_as_symbols for symbol in word}

        # Main loop — continue until vocabulary size limit is reached
        while len(self.vocab) < max_vocab_size:

            # Get frequencies of adjacent symbol pairs
            pair_freqs = Counter(
                (word[0][i], word[0][i + 1])
                for word in corpus_as_symbols
                for i in range(len(word[0]) - 1)
                for _ in range(word[1])
            )

            # Frequency of individual symbols
            get_freqs = Counter(
                symbol
                for word, freq in corpus_as_symbols
                for symbol in word
                for _ in range(freq)
            )

            ## PROPOSITION:
            '''
            # Frequencies of individual symbols across all items
            # In this version, we avoid creating freq number of duplicates per symbol
            get_freqs = Counter()
            for word, freq in corpus_as_symbols:
                for symbol in word:
                    get_freqs[symbol] += freq
            '''

            # Calculate scores for each pair
            scores = {
                pair: freq / (get_freqs[pair[0]] * get_freqs[pair[1]])
                for pair, freq in pair_freqs.items()
            }

            ## PROPOSITION:
            '''
            if not scores:
                break
            '''

            # Choose the pair with the highest score
            high_score = max(scores, key=scores.get) # type: ignore

            # Add merged token to the vocabulary
            self.vocab.add(high_score[0] + high_score[1][2:])

            ## PROPOSITION:
            '''
            if merged_token not in self.vocab:
                self.vocab.add(merged_token)
            '''

            # Replace the pair in all corpus words
            corpus_as_symbols = [
                (self._replace_pair(high_score, word), freq)
                for word, freq in corpus_as_symbols
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

        while len(word) > 0:
            i = len(word)

            # Try longest possible substring in vocabulary
            while i > 0 and word[:i] not in self.vocab:
                i -= 1

            if i == 0:
                return ["[UNK]"]

            tokens.append(word[:i])
            word = word[i:]

            # Prepend '##' to mark continuation if needed
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
        encoded_words = [self.encode_word(word) for word in pre_tokenized_text]

        return sum(encoded_words, [])


class WPTrieNode:
    """
    A node in the WordPiece trie structure.

    Supports basic trie functionality along with failure links and token collection
    for efficient longest-prefix match in WordPiece tokenization.
    """

    def __init__(self, char):
        # Store the character for this node
        self.char = char
        # Indicates if this node marks the end of a valid token
        self.is_end = False
        # Dictionary of child nodes (char -> WPTrieNode)
        self.children = {}
        # For linear/max-match tokenization: failure link and pops
        self.failure_pops = []
        self.failure_link = None
        # Vx is the string spelled out by the path to this node
        self.Vx = char

class WPTrie(object):
    """
    Trie structure for efficient WordPiece tokenization.

    Builds failure links and supports token matching using a variant of the
    Aho-Corasick algorithm for fast lookup.
    """

    def __init__(self, vocab={}):
        # Initialize the root node (empty string)
        self.root = WPTrieNode("")
        # Insert the special "##" prefix and store its node
        self.root_sharp = self.insert("##")
        # Insert all tokens from the vocabulary into the trie
        for v in vocab:
            self.insert(v)
        # Precompute failure links for efficient matching
        self.precompute()

    def insert(self, word):
        """
        Insert a word into the trie, creating nodes as needed.
        Marks the last node as the end of a valid token.
        """
        node = self.root
        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                # Create a new node for this character
                new_node = WPTrieNode(char)
                # Set Vx to be the path string up to this node
                new_node.Vx = node.Vx + char
                node.children[char] = new_node
                node = new_node
        # Mark this node as the end of a valid token
        node.is_end = True
        return node

    def precompute(self):
        """
        Compute failure links and failure pops for all nodes in the trie.
        This enables efficient Aho-Corasick-style matching.
        """
        r = self.root
        r_sharp = self.root_sharp
        # Use a queue to traverse all trie nodes in BFS order
        v_queue = [r, r_sharp]
        while len(v_queue) > 0:
            u = v_queue.pop(0)
            for c, v in u.children.items():
                if v == r_sharp:
                    # Skip the root_sharp node itself
                    continue
                if v.is_end:
                    # If this node ends a token, set failure link to r_sharp and pop its token
                    v.failure_link = r_sharp
                    v.failure_pops = [v.Vx]
                else:
                    # Otherwise, walk up failure links to find node with c as child
                    z = u.failure_link
                    Z = []
                    while z is not None and c not in z.children:
                        # Collect failure pops along the way
                        Z.extend(z.failure_pops)
                        z = z.failure_link
                    if z is not None:
                        # Set failure link to matching child, accumulate pops
                        v.failure_link = z.children[c]
                        v.failure_pops = u.failure_pops + Z
                # Add this child node to the queue for further processing
                v_queue.append(v)


class FastWP(NaiveWP):
    """
    A fast WordPiece tokenizer using a trie for efficient encoding.

    Extends Naive_WP by leveraging a trie structure (WPTrie) to enable
    linear-time subword tokenization via longest-prefix matching.
    """

    def __init__(self, tokenizer):
        '''
        Initialize the Fast_WP tokenizer.
            corpus (List[str]): A list of strings used to train the tokenizer.
            tokenizer (AutoTokenizer): A tokenizer instance to assist with preprocessing.
        '''
        # Initialize the parent class
        super().__init__(tokenizer)

    def train(self, corpus, max_vocab_size=30_000):
        """
        Train the fast WordPiece tokenizer and build the vocabulary trie.

        Args:
            corpus (List[str]): List of training strings.
            max_vocab_size (int): Maximum size of the subword vocabulary.
        """
        super().train(corpus, max_vocab_size)
        self.vocab_trie = WPTrie(self.vocab)

    def encode_word(self, word):
        """
        Encode a single word using the WordPiece trie.

        Args:
            word (str): The word to tokenize.

        Returns:
            List[str]: A list of subword tokens, or ["[UNK]"] if not matched.
        """
        tokens, u, i = self.matchloop(word + ' ', 0)
        if i < len(word) or u not in {self.vocab_trie.root, self.vocab_trie.root_sharp}:
            tokens = ["[UNK]"]
        else:
            if u == self.vocab_trie.root_sharp and len(tokens) == 0:
                tokens = super().encode_word(word)  # Should call original WordPiece
        return tokens

    def matchloop(self, s, i):
        """
        Traverse the WordPiece trie to find all matching subword tokens.

        Args:
            s (str): Input string with trailing space.
            i (int): Starting index.

        Returns:
            Tuple[List[str], WPTrieNode, int]: Tokens found, final node, and end index.
        """
        u = self.vocab_trie.root
        tokens = []
        while i < len(s):
            while s[i] not in u.children:
                if u.failure_link is None:
                    return tokens, u, i
                tokens.extend(u.failure_pops)
                u = u.failure_link
            u = u.children[s[i]]
            i = i + 1
        return tokens, u, i


class WPTrie_E2E(WPTrie):
    """
    Extended WPTrie for end-to-end WordPiece tokenization.

    Adds special handling for punctuation and boundary-aware failure links.
    """

    def __init__(self, vocab={}):
        # Special root node for punctuation transitions
        self.root_p = WPTrieNode("")
        super().__init__(vocab)

    # Modified for E2E tokenization
    def precompute(self):
        """
        Precompute failure links and pops, with special handling for punctuation.
        Extends the base precompute by assigning punctuation nodes' failure links to root_p.
        """
        # Original Precomputation (copied and extended)
        r = self.root
        r_sharp = self.root_sharp
        v_queue = [r, r_sharp]
        while len(v_queue) > 0:
            u = v_queue.pop(0)
            for c, v in u.children.items():
                if v == r_sharp:
                    continue
                if v.is_end:
                    v.failure_link = r_sharp
                    v.failure_pops = [v.Vx]
                else:
                    z = u.failure_link
                    Z = []
                    while z is not None and c not in z.children:
                        Z.extend(z.failure_pops)
                        z = z.failure_link
                    if z is not None:
                        v.failure_link = z.children[c]
                        v.failure_pops = u.failure_pops + Z
                # Modification for E2E:
                # For nodes whose character is punctuation (not alphanumeric),
                # set their failure link to the special punctuation root node.
                if not v.char.isalnum():
                    v.failure_link = self.root_p
                v_queue.append(v)


class Fast_WP_E2E(FastWP):
    """
    A fast, boundary-aware WordPiece tokenizer with end-to-end punctuation handling.

    Uses WPTrie_E2E to tokenize input while respecting word boundaries and punctuation,
    enabling more robust token segmentation for natural text.
    """

    def __init__(self, tokenizer):
        '''
        Initialize the Fast_WP_E2E tokenizer.
            tokenizer (AutoTokenizer): A tokenizer instance to assist with preprocessing.
        '''
        # Initialize the parent class
        super().__init__(tokenizer)

    def train(self, corpus, max_vocab_size=30_000):
        # Train using the NaiveWP/FastWP logic to build the vocabulary
        super(FastWP, self).train(corpus, max_vocab_size)
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