from .subword_tokenizer import SubwordTokenizer
from .trie import Trie

class TrieBPE(SubwordTokenizer):
    """Fast Byte-Pair Encoding with greedy longest-match via a trie: O(n)."""
    def __init__(self, tokenizer, verbose: bool = False):
        super().__init__(tokenizer)
        self.verbose: bool                = verbose
        self.logs: List[str]              = []
        self.merges: List[Tuple[str,str]] = [] # history of merges
        self.vocab: set[str]              = set()
        self.trie                         = Trie() # built in training

    def train(self, corpus: List[str], max_vocab: int = 30_000):
        if not isinstance(corpus, list) or not all(isinstance(x, str) for x in corpus):
            raise TypeError("Corpus must be a list of strings.")
        if not isinstance(max_vocab, int):
            raise TypeError("max_vocab must be an integer")

        # Pre-tokenize (whitespace, punctuation, offsets)
        prepared = super().preprocessing(corpus)
        # Temporary corpus
        temp_corpus = [list(word) for example in prepared for word, _ in example]
        # Initialize vocabulary: all unique single characters
        self.vocab = {ch for word in temp_corpus for ch in word}

        # Count all adjacent pair frequencies and track their positions
        pair_freq: Counter[Tuple[str,str]]                     = Counter()
        pair_pos:   Dict[Tuple[str,str], List[Tuple[int,int]]] = defaultdict(list)
        for w_idx, word in enumerate(temp_corpus):
            for p_idx in range(len(word) - 1):
                pair = (word[p_idx], word[p_idx + 1])
                pair_freq[pair] += 1
                pair_pos[pair].append((w_idx, p_idx))

        # Build a max heap of (-frequency, pair) so the most frequent has priority
        heap = [(-freq, pair) for pair, freq in pair_freq.items()]
        heapq.heapify(heap)

        if self.verbose:
            print(f"[TRAIN] starting merges at heap size {len(heap)}")

        # Merge
        while len(self.vocab) < max_vocab and heap:
            freq_neg, pair = heapq.heappop(heap)
            freq = -freq_neg
            # Skip stale entries or pairs too rare
            if pair_freq.get(pair, 0) != freq:
                if self.verbose:
                    self.logs.append(f"Skipping {pair} (stale or freq<2)")
                continue
            if freq < 2:
                break

            a, b = pair
            new_sym = a + b # merged token
            self.vocab.add(new_sym)
            self.merges.append(pair)

            if self.verbose:
                print(f"[MERGE] {pair} -> '{new_sym}', freq={freq}, vocab -> {len(self.vocab)}")

            # Update every word where this pair occurs
            for w_idx, p_idx in pair_pos[pair]:
                word = temp_corpus[w_idx]
                # Skip if the word has changed already
                if p_idx >= len(word) - 1 or (word[p_idx], word[p_idx + 1]) != pair:
                    continue
                # Remember neighboring pairs before replacement
                left_pair  = (word[p_idx - 1], word[p_idx])     if p_idx > 0 else None
                right_pair = (word[p_idx + 1], word[p_idx + 2]) if p_idx + 2 < len(word) else None
                # Replace the two symbols with the merged symbol
                word[p_idx : p_idx + 2] = [new_sym]
                # Decrement counts for old neighbor pairs
                for old in (left_pair, pair, right_pair):
                    if old:
                        pair_freq[old] -= 1
                # Increment counts for newly-formed neighbor pairs
                new_left  = (word[p_idx - 1], word[p_idx]) if p_idx > 0 else None
                new_right = (word[p_idx], word[p_idx + 1]) if p_idx + 1 < len(word) else None
                for new in (new_left, new_right):
                    if new:
                        pair_freq[new] += 1
                        heapq.heappush(heap, (-pair_freq[new], new))
                        pair_pos[new].append((w_idx, p_idx - 1 if new is new_left else p_idx))
            # Mark this pair as consumed
            pair_freq[pair] = 0
            pair_pos.pop(pair, None)

        if self.verbose:
            print(f"[TRAIN] completed {len(self.merges)} merges. Final vocabulary size: {len(self.vocab)}")

        # Build trie for fast encoding
        for tok in self.vocab:
            # Each tok is a string; insert its character sequence
            self.trie.insert(list(tok), tok)

        if self.verbose:
            print("[TRAIN] Trie built for encoding.")

    def _encode_seq(self, seq: List[str]) -> List[str]:
        """Greedy longest-match using the trie (seq is a list of chars)."""
        out, i = [], 0
        while i < len(seq):
            tok, length = self.trie.longest_match(seq, i)
            if tok is None:
                # no multi-char token matches here: emit single char
                out.append(seq[i])
                i += 1
            else:
                # emit the matched token, advance by its length
                out.append(tok)
                i += length
        return out

    def encode_word(self, word: str) -> List[str]:
        """Encode a single word into subword tokens."""
        return self._encode_seq(list(word))

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text: preprocess, encode per word, flatten."""
        if not isinstance(text, str):
            raise TypeError("Text must be a string.")
        prepared = self.preprocessing([text])[0]
        # Encode each word in turn and flatten
        return [tok for w, _ in prepared for tok in self.encode_word(w)]
