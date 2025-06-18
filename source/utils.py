from transformers import PreTrainedTokenizerFast
from typing import List, Tuple, Optional
import re

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


class TrieNode:
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
        # Dictionary of child nodes (char -> TrieNode)
        self.children = {}
        # For linear/max-match tokenization: failure link and pops
        self.failure_pops = []
        self.failure_link = None
        # chars_seen is the string spelled out by the path to this node
        self.chars_seen = char
        

class WPTrie_E2E(object):
    """
    Trie structure for efficient for end-to-end WordPiece tokenization

    Handles precomputation step for LinMaxMatch fast WordPiece tokenization algorithm.

    Includes special handling for punctuation and boundary-aware failure links.
    """

    def __init__(self, vocab={}):
        # Initialize the root node (empty string)
        self.root = TrieNode("")
        # Special root node for punctuation transitions
        self.root_p = TrieNode("")
        # Insert the special "##" prefix and store its node
        self.root_sharp = self.insert("##")
        # Insert all tokens from the vocabulary into the trie
        for v in vocab:
            self.insert(v)
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
                new_node = TrieNode(char)
                # Set chars_seen to be the path string up to this node
                new_node.chars_seen = node.chars_seen + char
                node.children[char] = new_node
                node = new_node
        # Mark this node as the end of a valid token
        node.is_end = True
        return node

    # Modified for E2E tokenization
    def precompute(self):
        """
        Precompute failure links and pops, with special handling for punctuation.
        Extends the base precompute by assigning punctuation nodes' failure links to root_p.
        """
        # Original Precomputation (copied and extended)
        r = self.root
        r_sharp = self.root_sharp
        node_queue = [r, r_sharp]
        while len(node_queue) > 0:
            curr_node = node_queue.pop(0)
            for child_char, child_node in curr_node.children.items():
                if child_node == r_sharp:
                    continue
                if child_node.is_end:
                    child_node.failure_link = r_sharp
                    child_node.failure_pops = [child_node.chars_seen]
                else:
                    fail_node = curr_node.failure_link
                    fail_pops = []
                    while fail_node is not None and child_char not in fail_node.children:
                        fail_pops.extend(fail_node.failure_pops)
                        fail_node = fail_node.failure_link
                    if fail_node is not None:
                        child_node.failure_link = fail_node.children[child_char]
                        child_node.failure_pops = curr_node.failure_pops + fail_pops
                # Modification for E2E:
                # For nodes whose character is punctuation (defined here as not alphanumeric),
                # set their failure link to the special punctuation root node.
                if not child_node.char.isalnum():
                    child_node.failure_link = self.root_p
                node_queue.append(child_node)

def recover_sentence(tokens):
  # This function does NOT reliably reconstruct the
  # original sentence, as whitespace information is lost
  # Does some common-sense punctuation handling

  outseq = ' '.join(tokens)
  # Combine subwords
  outseq = re.sub(r'\s##(\S)', r'\g<1>', outseq)
  # Remove left space before appropriate punctuation
  outseq = re.sub(r'\s(\.|,|\)|\]|\\|’|-|\'|\\|/)', r'\g<1>', outseq)
  # Remove right space after appropriate punctuation
  outseq = re.sub(r'(\(|\[|\\|’|-|\'|\\|/)\s', r'\g<1>', outseq)

  return outseq