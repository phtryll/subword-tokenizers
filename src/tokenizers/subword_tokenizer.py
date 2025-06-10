class SubwordTokenizer:
    '''A parent class for subword tokenizers.'''

    def __init__(self, tokenizer):
        '''tokenizer: use AutoTokenizer'''

        self.tokenizer = tokenizer

    def preprocessing(self, corpus):
        '''
        Preprocess the input corpus.
            corpus (List[str]): A list of sentences (strings) to preprocess.
        Returns
            List[List[Tuple[str, Tuple[int, int]]]]: the preprocessed corpus where each
        sentence is a list of tuples, each containing a token and its character offset.
        '''

        # Lowercase each sentence, tokenize it into subwords, and extract character offsets
        corpus_preprocessed = [
            self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(str.lower(example))
            for example in corpus
        ]

        return corpus_preprocessed

    def vocab_length(self, corpus):
        '''
        Calculate the vocabulary size based on individual symbols in the corpus.
            corpus (List[str]): A list of strings used to derive the vocabulary.
        Returns
            int The number of unique symbols (characters) in the corpus.
        '''

        # Create a set of unique characters (symbols) across the entire corpus
        vocab_length = len({symbol for example in corpus for symbol in example})

        return vocab_length
