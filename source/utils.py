from transformers import PreTrainedTokenizerFast
from collections import defaultdict, deque
from typing import List, Tuple, Optional

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

class ACNode(dict):
    __slots__ = ("failure_link", "output_token")
    def __init__(self):
        super().__init__()
        self.failure_link: "ACNode" | None = None
        self.output_token: Optional[str]   = None

class AhoCorasick:
    def __init__(self, token_vocab: List[str] = ()):
        self.root = ACNode()
        for token in token_vocab:
            self.insert(token)
        self.build_failure_links()

    def insert(self, token: str) -> None:
        state = self.root
        for symbol in token:
            state = state.setdefault(symbol, ACNode())
        state.output_token = token

    def build_failure_links(self) -> None:
        queue = deque()
        for state in self.root.values():
            state.failure_link = self.root
            queue.append(state)
        while queue:
            current_state = queue.popleft()
            for symbol, next_state in current_state.items():
                queue.append(next_state)
                fallback = current_state.failure_link
                while fallback and symbol not in fallback:
                    fallback  = fallback.failure_link
                next_state.failure_link = fallback[symbol] if fallback and symbol in fallback else self.root
                if next_state.failure_link.output_token and not next_state.output_token:
                    next_state.output_token = next_state.failure_link.output_token

    def find_matches(self, input_sequence: List[str]):
        state = self.root
        for end_index, symbol in enumerate(input_sequence):
            while state and symbol not in state:
                state = state.failure_link
            state = state[symbol] if state else self.root
            match_state = state
            while match_state:
                if match_state.output_token:
                    yield end_index, match_state.output_token
                match_state = match_state.failure_link if match_state is not self.root else None