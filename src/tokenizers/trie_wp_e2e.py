from .trie_wp import WPTrie

class WPTrie_E2E(WPTrie):

    def __init__(self, vocab={}):

        self.root_p = WPTrieNode("")
        super().__init__(vocab)


    # Modified for E2E tokenization
    def precompute(self):
      # Original Precomputation
      r = self.root
      r_sharp = self.root_sharp
      v_queue = [r, r_sharp]
      while len(v_queue)>0:
        u = v_queue.pop(0)
        for c, v  in u.children.items():
          if v==r_sharp:
            continue
          if v.is_end:
            v.failure_link = r_sharp
            v.failure_pops = [v.Vx]
          else:
            z = u.failure_link
            Z = []
            while z!=None and c not in z.children:
              Z.extend(z.failure_pops)
              z = z.failure_link
            if z!=None:
              v.failure_link = z.children[c]
              v.failure_pops = u.failure_pops + Z
          # Modification for E2E
          # for punctuation nodes, set failure links to
          # special node r_p (do not change failure pops)
          if not v.char.isalnum():
            v.failure_link = self.root_p
          v_queue.append(v)
