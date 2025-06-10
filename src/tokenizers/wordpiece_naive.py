from .subword_tokenizer import SubwordTokenizer

class NaiveWP(SubwordTokenizer):
    '''A subword tokenizer based on the WordPiece algorithm.'''

    def __init__(self, tokenizer):
        '''
        Initialize a naive WordPiece tokenizer.
            corpus (List[str]): A list of strings used to train the tokenizer.
            tokenizer (AutoTokenizer): A tokenizer instance to assist with preprocessing.
        '''

        # Initialize the parent class
        super().__init__(tokenizer)

    def train(self, corpus, max_vocab_size=30_000):
        '''
        Train the WordPiece tokenizer on the corpus.
            max_vocab_size (int): The maximum size of the vocabulary.
        '''

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
            high_score = max(scores, key=scores.get)

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
        '''
        Replace a symbol pair with a merged symbol in a word.
            pair (Tuple[str, str]): The pair of symbols to merge.
            word (List[str]): The word (as a list of symbols).
        Returns:
            List[str]: The word after merging the specified symbol pair.
        '''

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
        '''
        Encode a single word using the trained WordPiece vocabulary.
            word (str): The word to encode.
        Returns:
            List[str]: The list of subword tokens. Returns ["[UNK]"] if the word
            cannot be tokenized with the current vocabulary.
        '''

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
        '''
        Tokenize a text into subword tokens.
            text (str): The text to tokenize.
        Returns:
            List[str]: A flat list of subword tokens.
        '''

        if not isinstance(text, str):
            raise TypeError("Text to tokenize must be a string.")

        # Preprocess input text
        pre_tokenized_corpus = self.preprocessing([text])
        pre_tokenized_text = [word for word, offset in pre_tokenized_corpus[0]]

        # Encode each word and flatten the result
        encoded_words = [self.encode_word(word) for word in pre_tokenized_text]

        return sum(encoded_words, [])
