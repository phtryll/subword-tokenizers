from utils import SubwordTokenizer, WPTrie, WPTrie_E2E
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