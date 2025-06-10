# Trie implementation for optimized wordpiece
class WPTrieNode:

    def __init__(self, char):
        # Basic trie
        self.char = char
        self.is_end = False
        self.children = {}
        # For LinMaxMatch Tokenization
        self.failure_pops = []
        self.failure_link = None
        self.Vx = char

class WPTrie(object):

    def __init__(self, vocab={}):

        self.root = WPTrieNode("")
        self.root_sharp = self.insert("##")
        for v in vocab:
          self.insert(v)

        self.precompute()

    def insert(self, word):

        node = self.root
        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                new_node = WPTrieNode(char)
                new_node.Vx = node.Vx + char
                node.children[char] = new_node
                node = new_node

        node.is_end = True
        return node

    def precompute(self):
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
          v_queue.append(v)
