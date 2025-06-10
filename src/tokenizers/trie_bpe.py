class TrieNode(defaultdict):
    __slots__ = ("token",)
    def __init__(self):
        super().__init__(TrieNode)
        # If this node marks the end of valid token, store it here
        self.token: Optional[str] = None

class Trie:
    """Minimal prefix tree that maps char sequences to the full token string."""
    def __init__(self):
        # Root has no associated token
        self.root = TrieNode()

    def insert(self, token_sequence: List[str], token: str) -> None:
        """
        token_sequence: e.g. ['t', 'h', 'e'] or ['th', 'e']
        token:          full merged token string, e.g. 'the'
        """
        node = self.root
        for sym in token_sequence:
            # Traverse or create child for this symbol
            node = node[sym]
        # Mark this node as end of a valid token
        node.token = token

    def longest_match(self, seq: List[str], start: int = 0) -> Tuple[Optional[str], int]:
        """
        Greedy longest-prefix match from seq[start:].
        Returns (matched_token, length_in_symbols) or (None, o).
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
