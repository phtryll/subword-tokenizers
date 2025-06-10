from .wordpiece_naive import NaiveWP
from .trie_wp import WPTrie

class FastWP(NaiveWP):
    '''A subword tokenizer based on the WordPiece algorithm.'''

    def __init__(self, tokenizer):
        '''
        Initialize the Fast_WP tokenizer.
            corpus (List[str]): A list of strings used to train the tokenizer.
            tokenizer (AutoTokenizer): A tokenizer instance to assist with preprocessing.
        '''

        # Initialize the parent class
        super().__init__(tokenizer)

    def train(self, corpus, max_vocab_size):
      super().train(corpus, max_vocab_size)
      self.vocab_trie = WPTrie(self.vocab)

    def encode_word(self, word):
      '''
        Encode a single word using the trained WordPiece vocabulary.
            word (str): The word to encode.
        Returns:
            List[str]: The list of subword tokens. Returns ["[UNK]"] if the word
            cannot be tokenized with the current vocabulary.
      '''
      tokens, u, i = self.matchloop(word + ' ', 0)
      if i < len(word) or u not in {self.vocab_trie.root, self.vocab_trie.root_sharp}:
        tokens = ["[UNK]"]
      else:
        if u==self.vocab_trie.root_sharp and len(tokens)==0:
          tokens = super().encode_word(word) # Should call original wordpiece
      return tokens


    def matchloop(self, s, i):
      u = self.vocab_trie.root
      tokens = []
      while i < len(s):
        while s[i] not in u.children:
          if u.failure_link == None:
            return tokens, u, i
          tokens.extend(u.failure_pops)
          u = u.failure_link
        u = u.children[s[i]]
        i = i + 1
      return tokens, u, i
