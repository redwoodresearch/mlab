from __future__ import annotations

from typing import Dict, List

from collections import defaultdict
import days.w2d4.tokenizer as tokenizer
import re

def extract_tokens(string: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", string)

def corpus_common_tokens(string_list: List[str], top_k=30_000) -> List[str]:
    """Takes a list of strings and outputs the top_k most common "tokens"."""
    token_dict = defaultdict(int)

    for string in string_list:
        tokens = extract_tokens(string)
        for token in tokens:
            token_dict[token] += 1

    distinct_tokens = list(token_dict.keys())
    distinct_tokens.sort(key=lambda token : token_dict[token], reverse=True)
    return distinct_tokens[:top_k]


class Tokenizer:
    def __init__(self, token_list: List[Dict]):
        """
        See ~/mlab/days/w2d4/bpe_tokens.json for an example of a token_list.
        """
        self.token_list = token_list
        self.id_to_string = {}
        self.string_to_id = {}

        for piece_idx_dict in token_list:
            piece = piece_idx_dict["piece"]
            idx = piece_idx_dict["id"]
            self.id_to_string[idx] = piece
            self.string_to_id[piece] = idx

    def decode(self, id_list: List[int]) -> str:
        ret = ""
        for idx in id_list:
            ret += self.id_to_string[idx]
        return ret

    def tokenize(self, string: str) -> List[int]:
        """Missing tokens are tokenized as 3 ('[UNK]')."""
        tokens = re.findall(r"\w+|[^\w\s]", string)
        return [self.string_to_id.get(token, 3) for token in tokens]

# class Make a class BPETokenizer that inherits from your earlier Tokenizer.
# We’re inheriting to use our previous padding, start/end tokens, batching,
# and decoding implementations.

# Step 1: Tokenize BPE

# Make the function tokenize of your BPETokenizer class.
# It implements this algorithm: turn your string into a list of tokens that are each one char, then loop through the list of tokens, and for every token in the vocabulary, and if two adjacent tokens concatenate to that token, replace them with that token.

class BPETokenizer(Tokenizer):

    def tokenize(self, text: str) -> List[int]:
        tokens: List[str] = list(text)
        # for vocab_token in sorted(self.string_to_id.keys(), key=lambda s: len(s)):
        for vocab_token in self.string_to_id.keys():
            new_tokens: List[str] = []
            i: int = 0
            while i < len(tokens):
                if i + 1 < len(tokens):
                    if tokens[i] + tokens[i + 1] == vocab_token:
                        new_tokens.append(vocab_token)
                        i += 2
                        continue
                
                new_tokens.append(tokens[i])
                i += 1
        
            tokens = new_tokens
        
        return [self.string_to_id[t] for t in tokens]

    @classmethod
    def from_corpus(cls, corpus: List[str], concats: int = 10) -> BPECorpus:
        corpus: List[List[str]] = [[char for char in text] for text in corpus]
        concatted: List[str] = []

        def concat(corpus, concatted):
            pair_freqs = defaultdict(int)
            for text in corpus:
                for i in range(len(text) - 1):
                    pair_freqs[(text[i], text[i+1])] += 1
            pair_freqs_list = [key for key in pair_freqs.keys() if key[0] + key[1] not in concatted]
            pair_freqs_list.sort(key=lambda pair : pair_freqs[pair])

            if len(pair_freqs_list) > 0:
                concatted.append(pair_freqs_list[-1])
            else:
                return new_corpus, concatted

            new_corpus: List[List[str]] = []
            for text in corpus:
                i = 0
                new_text: List[str] = []
                while i < len(text):
                    if i == len(text) - 1:
                        new_text.append(text[-1])
                        break
                    if text[i] + text[i-1] == concatted[-1]:
                        new_text.append(text[i] + text[i-1])
                        i += 2
                    else:
                        new_text.append(text[i])
                        i += 1
                new_corpus.append(new_text)

            return new_corpus, concatted

        for _ in range(concats):
            corpus, concatted = concat(corpus, concatted)

        return cls(token_list=concatted)


def custom_test_bpe_tokenize():
    tokenizer = BPETokenizer([
        dict(piece="abb", id=3),
        dict(piece="a", id=0),
        dict(piece="b", id=1),
        dict(piece="ab", id=2),
    ])
    print(tuple(tokenizer.tokenize("abb")))
    assert tuple(tokenizer.tokenize("abb")) == (3,)

def foo1():
    xs = [1, 2, 3]
    def bar2():
        nonlocal xs
        xs = [4, 5, 6]
    bar()
    print(xs) # [4, 5, 6]


def foo2():
    xs = [1, 2, 3]
    def bar2(xs):
        xs = [4, 5, 6]
    bar(xs)
    print(xs) # [1, 2, 3]


def foo3():
    xs = [1, 2, 3]
    def bar(xs):
        xs.append(1)
        # xs = [4, 5, 6]
    bar(xs)
    print(xs) # [1, 2, 3]

if __name__ == "__main__":
    tokenizer.test_tokenizer_from_corpus_fn(corpus_common_tokens)
    tokenizer.test_tokenizer(Tokenizer)

    # custom_test_bpe_tokenize()

    tokenizer.test_bpe_tokenizer(BPETokenizer)
    tokenizer.test_bpe_tokenizer_from_corpus(BPETokenizer)

    foo1()
    foo2()
