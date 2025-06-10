from .subword_tokenizer import SubwordTokenizer

class NaiveBPE(SubwordTokenizer):
    '''A subword tokenizer based on the Byte-Pair Encoding (BPE) algorithm.'''

    def __init__(self, tokenizer):
        '''
        Initialize a naive BPE tokenizer.
            corpus (List[str]): A list of strings used to train the tokenizer.
            tokenizer (AutoTokenizer): A tokenizer instance to assist with preprocessing.
        '''

        # Initialize the parent class with the tokenizer
        super().__init__(tokenizer)

    def _replace_pair(self, pair, word):
        '''
        Replace a pair of symbols with a single merged symbol in a word.
            pair (Tuple[str, str]): The symbol pair to merge.
            word (List[str]): The word represented as a list of symbols.
        Returns:
            List[str]: The word after merging the specified symbol pair.
        '''

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

    def train(self, corpus, max_vocab_size=30_000):
        '''
        Train the BPE tokenizer on the corpus.
            max_vocab_size (int): The maximum size of the subword vocabulary.
        Returns
            List[Tuple[str, str]]: a list of merged symbol pairs representing subword units.
        '''

        # Check that corpus is a list of strings
        if not isinstance(corpus, list) or not all(isinstance(example, str) for example in corpus):
            raise TypeError("Corpus must be a list of strings.")

        if not isinstance(max_vocab_size, int):
            raise TypeError("Maximum vocabulary size must be an integer.")

        # Prepare corpus for training
        prepd_corpus = super().preprocessing(corpus)

        # Initialize the list that will store merged symbol pairs
        self.merges_list = []

        # Flatten corpus into a list of words
        corpus_as_words = [word for example in prepd_corpus for word, position in example]

        # Build the initial vocabulary from all individual symbols
        vocab = {symbol for word in corpus_as_words for symbol in word}

        # Count the frequency of each word
        word_freqs = Counter(corpus_as_words)

        # Convert each word into a list of symbols, retaining their frequency
        corpus_as_symbols = [
            ([symbol for symbol in word], frequency)
            for word, frequency in word_freqs.items()
        ]

        # Iteratively merge the most frequent symbol pairs until vocab limit is reached
        while len(vocab) < max_vocab_size:

            # Count all adjacent symbol pair frequencies in the corpus
            get_pair_freqs = Counter(
                [
                    (word[0][i], word[0][i + 1])
                    for word in corpus_as_symbols
                    for i in range(len(word[0]) - 1)
                    for _ in range(word[1])
                ]
            )

            # Select the most frequent pair
            most_frequent_pair = get_pair_freqs.most_common(1)[0][0]

            # Add the new merged symbol to the vocabulary
            vocab.add(most_frequent_pair[0] + most_frequent_pair[1])

            # Record the merge operation
            self.merges_list.append(most_frequent_pair)

            # Replace the merged pair in all words in the corpus
            corpus_as_symbols = [
                (self._replace_pair(most_frequent_pair, word), frequency)
                for word, frequency in corpus_as_symbols
            ]

    def encode_word(self, word):
        '''
        Encode a word into subword tokens using the learned merges.
            word (str): The input word to encode.
        Returns:
            List[str]: The encoded subword sequence.
        '''

        # Split word into individual symbols
        word_split = [symbol for symbol in word]

        # Apply each merge operation sequentially
        for pair in self.merges_list:
            word_split = self._replace_pair(pair, word_split)

        return word_split

    def tokenize(self, text):
        '''
        Tokenize input text into subword units.
            text (str): The text to tokenize.
        Returns:
            List[str]: A list of subword tokens.
        '''

        if not isinstance(text, str):
            raise TypeError("Text to tokenize must be a string.")

        # Preprocess the input text using the parent class method
        pre_tokenized_corpus = self.preprocessing([text])

        # Extract words from the pre-tokenized output
        pre_tokenized_text = [word for word, offset in pre_tokenized_corpus[0]]

        # Encode each word using the merge rules
        encoded_words = [self.encode_word(word) for word in pre_tokenized_text]

        # Flatten the list of encoded words into a single token list
        return sum(encoded_words, [])
