from .wordpiece_fast import FastWP

class FastWP_E2E(FastWP):
    '''A subword tokenizer based on the WordPiece algorithm.'''

    def __init__(self, tokenizer):
        '''
        Initialize the Fast_WP_E2E tokenizer.
            tokenizer (AutoTokenizer): A tokenizer instance to assist with preprocessing.
        '''

        # Initialize the parent class
        super().__init__(tokenizer)

    def train(self, corpus, max_vocab_size):
      super().super().train(corpus, max_vocab_size)
      self.vocab_trie = WPTrie_E2E(self.vocab)

    def tokenize(self, text):
      '''
      Tokenize a text into subword tokens.
          text (str): The text to tokenize.
      Returns:
          List[str]: A flat list of subword tokens.
      '''

      if not isinstance(text, str):
          raise TypeError("Text to tokenize must be a string.")

      result = []
      s = text.lower() + " " # keep lowercasing
      i = 0

      while i < len(s):
        tokens, u, i = self.matchloop(s, i)
        if not self.iswdbndry(s, i) or u not in {self.vocab_trie.root, self.vocab_trie.root_sharp, self.vocab_trie.root_p}:
          tokens = ["['UNK']"]
        else:
          if u==self.vocab_trie.root_sharp and len(tokens)==0:
            tokens = super().encode_word("##")
        result.extend(tokens)
        while i < len(s) and not self.iswdbndry(s,i):
          i = i + 1
        while i < len(s) and s[i].isspace():
          i = i + 1
      return result

    def iswdbndry(self, s,i):
      return i > len(s) or (i>0 and not s[i-1].isalnum() or s[i].isspace() or not s[i].isalnum())
