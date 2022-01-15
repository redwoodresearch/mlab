from typing import Counter
import transformers
import numpy as np
import sentencepiece as spm
from transformers.models.auto.tokenization_auto import AutoTokenizer
from days.utils import Timer, getprops
import json
import functools
import torch as t
from unidecode import unidecode
from collections import defaultdict
import itertools
import re
from pathlib import Path

corpus = open(Path.home() / "mlab/shakespeare.txt").readlines()
minicorpus = corpus[5000:6000]


def normalizer(str):
    return unidecode(str.lower().replace("\t", "    ").replace("\n", ""))


class Tokenizer:
    def __init__(
        self,
        token_list,
        pad=0,
        sep=1,
        cls=None,
        bot=2,
        eot=2,
        unk=3,
        mask=4,
        normalizer=normalizer,
    ):
        self.replacements = {"▁": " "}
        for obj in token_list:
            obj["piece"] = self._replace_all(obj["piece"])
        self.token_list = token_list
        self.pad = pad
        if cls:
            self.eot = cls
            self.bot = cls
        else:
            self.eot = eot
            self.bot = bot
        self.mask = mask
        self.unk = unk
        self.sep = sep
        self.vocab = {x["piece"]: x for x in token_list}
        self.vocab_by_id = {x["id"]: x for x in token_list}

        self.normalizer = normalizer

    def _replace_all(self, text):
        return functools.reduce(
            lambda a, x: a.replace(x[0], x[1]), self.replacements.items(), text
        )

    def _pad_and_shit(self, ids, ends=False, pad_length=None):
        if ends:
            ids = [self.bot] + ids + [self.eot]
        if pad_length is not None:
            ids.extend([self.pad] * (pad_length - len(ids)))
        return ids

    def tokenize(self, texts, **kwargs):
        if isinstance(texts, str):
            if "pad_longest" in kwargs:
                del kwargs["pad_longest"]
            return self._pad_and_shit(self._tokenize(texts), **kwargs)
        results = []
        maxlen = 0
        for text in texts:
            seq_ids = self._tokenize(text)
            maxlen = max(maxlen, len(seq_ids))
            results.append(new_ids)
        if kwargs.get("pad_longest"):
            results = [
                self._pad_and_shit(
                    seq_ids, ends=kwargs.get("ends", False), pad_length=maxlen
                )
                for seq_ids in results
            ]
        return results

    def from_corpus(texts, n=30000):
        splitted = itertools.chain(
            *[re.findall(r"\w+|[^\w\s]", text) for text in texts]
        )
        counts = Counter(splitted)
        tokens = [
            {"piece": x, "id": i}
            for i, (x, _) in zip(range(5, 10000000), counts.most_common(n))
        ]
        return Tokenizer(tokens)

    def _tokenize(self, text):
        splitted = re.findall(r"\w+|[^\w\s]", text)
        ids = [self.vocab.get(x, {"id": self.unk})["id"] for x in splitted]
        return ids

    def __call__(self, texts, **kwargs):
        return self.tokenize(texts, **kwargs)

    def decode(self, ids):
        ids = list(ids)
        return "".join([self.vocab_by_id[id]["piece"] for id in ids])


# bpe and wordpiece work by splitting to chars then merging by tokens in order! i didn't know this!

# main tokenizers: bpe, wordpiece, sentencepiece, unigram lm
# how unigram lm works: pick the tokenization such that a unigram lm on them has the best loss
# https://everdark.github.io/k9/notebooks/ml/natural_language_understanding/subword_units/subword_units.nb.html
class UnigramLmTokenizer(Tokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _tokenize(self, text):
        if self.normalizer is not None:
            text = self.normalizer(text)
        best_subw_slices = self.viterbi_forward(text)
        ids = self.viterbi_backward(best_subw_slices)

        return ids

    def viterbi_forward(self, sequence):
        """Forward step of Viterbi."""
        # Create storage array for best substring recorded at each end-of-character position.
        best_subw_slices = t.IntTensor(len(sequence) + 1, 3).fill_(-1)
        print(best_subw_slices.shape)
        neg_loglik = np.zeros(len(sequence) + 1)
        # Search the best substring given every possible end of position along the word.
        for eow in range(1, len(sequence) + 1):
            # For every end-of-word position:
            neg_loglik[eow] = np.inf
            for bow in range(eow):
                # For every possible beginning-of-word position given the end-of-word position:
                subw = sequence[bow:eow]
                if subw in self.vocab:
                    vocel = self.vocab[subw]
                    logp = vocel["score"]
                    id = vocel["id"]
                    # Compute subword probability:
                    # P(current segment) + P(the best segment just before the current segment).
                    s = neg_loglik[bow] - logp
                    if s < neg_loglik[eow]:
                        neg_loglik[eow] = s
                        best_subw_slices[eow, 0] = bow
                        best_subw_slices[eow, 1] = eow
                        best_subw_slices[eow, 2] = id
        return best_subw_slices

    def viterbi_backward(self, subw_slices):
        """Backward step of Viterbi to return the best path."""
        subwords = []
        next_slices = subw_slices[-1]
        while next_slices[0] != -1:
            subwords.append(next_slices[2].item())
            next_slices = subw_slices[next_slices[0]]
        subwords.reverse()
        return subwords


class BPETokenizer(Tokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _tokenize(self, text):
        tokens = list(text)
        for token in self.token_list:
            token = token["piece"]
            i = 0  # using jank loop to iterate through list while changing its length
            while i < len(tokens) - 1:
                if tokens[i] + tokens[i + 1] == token:
                    tokens[i] = token
                    tokens.pop(i + 1)
                i += 1
        return [self.vocab[x]["id"] for x in tokens]

    def from_corpus(texts, num_tokens=1000):
        seqs = [list(x) for x in texts]

        tokens = list(set(itertools.chain(*seqs)))
        merge_pair = 12345678990
        merge_string = None
        while len(tokens) < num_tokens:
            pair_counts = defaultdict(lambda: 0)
            for seq in seqs:
                i = 0
                while i < len(seq) - 1:
                    tottup = (seq[i], seq[i + 1])
                    if tottup == merge_pair:
                        seq[i] = merge_string
                        seq.pop(i + 1)
                        i -= 1
                    else:
                        pair_counts[tottup] += 1
                    i += 1
            if len(pair_counts) == 0:
                break
            merge_pair = max(pair_counts.items(), key=lambda x: x[1])[0]
            merge_string = merge_pair[0] + merge_pair[1]
            tokens.append(merge_string)
        tokenizer = BPETokenizer(
            [{"id": i, "piece": p} for i, p in zip(range(5, 1000000), tokens)]
        )
        return tokenizer


def test_tokenizer_from_corpus(tokenizer):
    reference = Tokenizer.from_corpus(minicorpus)
    yours = tokenizer.from_corpus(minicorpus)
    assert tuple([x["piece"] for x in reference.token_list]) == tuple(
        [x["piece"] for x in yours.token_list]
    )


def test_tokenizer_from_corpus_fn(fn):
    reference = Tokenizer.from_corpus(minicorpus).vocab.keys()
    yours = fn(minicorpus)
    assert set(reference) == set(yours)


def test_tokenizer(tokenizer):
    reference = Tokenizer.from_corpus(minicorpus)
    yours = tokenizer(reference.token_list)
    assert tuple(reference.tokenize("hello, my name is tom trundlewich")) == tuple(
        yours.tokenize("hello, my name is tom trundlewich")
    )


def test_tokenizer_convenience(tokenizer):
    reference = Tokenizer.from_corpus(minicorpus)
    yours = tokenizer(reference.token_list)
    assert tuple(
        reference(
            [
                "hello, my name is tom trundlewich",
                "farewell, my dearest tom trundlewich",
            ]
        )
    ) == tuple(
        yours(
            [
                "hello, my name is tom trundlewich",
                "farewell, my dearest tom trundlewich",
            ]
        )
    )
    assert tuple(
        reference(
            [
                "hello, my name is tom trundlewich",
                "farewell, my dearest tom trundlewich",
            ],
            pad_longest=True,
        )
    ) == tuple(
        yours(
            [
                "hello, my name is tom trundlewich",
                "farewell, my dearest tom trundlewich",
            ],
            pad_longest=True,
        )
    )
    assert tuple(reference("farewell, my dearest tom trundlewich")) == tuple(
        yours("farewell, my dearest tom trundlewich")
    )


def test_bpe_tokenizer_from_corpus(tokenizer):
    reference = BPETokenizer.from_corpus(minicorpus)
    yours = tokenizer.from_corpus(minicorpus)
    assert tuple(reference.vocab.items()) == tuple(yours.vocab.items())


def test_bpe_tokenizer(tokenizer):
    reference = BPETokenizer.from_corpus(minicorpus)
    yours = tokenizer(reference.token_list)
    assert tuple(reference.tokenize("hello, my name is tom trundlewich")) == tuple(
        yours.tokenize("hello, my name is tom trundlewich")
    )


if __name__ == "__main__":
    # Syntax-1
    print("loading shakespeare")
    tokenizer = Tokenizer.from_corpus(minicorpus)
    print(tokenizer("i eat bread for breakfast every day of the week."))
    raise AssertionError("hi")
    bpe_shakespeare = BPETokenizer.from_corpus(minicorpus, num_tokens=500)
    model_file = "/home/tao/mlab/days/spiece.model"
    s = spm.SentencePieceProcessor(model_file=model_file)
    model_file_bytes = open(model_file, "rb").read()
    example_str = "Hello, my name is Amy the shoemaker, and I love eating oligarchies and monopolies and oceanography, RandomRedditor, and getting paid $250,000."
    toks = s.encode(example_str)
    print(toks)
    dct = json.load(
        open(
            "./days/sentencepiece.json",
        )
    )
    my_tokenizer = UnigramLmTokenizer(token_list=dct)
    btxt = """Gregory, any update on this? Maybe you can poll python-ideas.

Collin, any download stats and feedback on your package?
msg111176 - (view)	Author: Amaury Forgeot d'Arc (amaury.forgeotdarc) * (Python committer)	Date: 2010-07-22 14:28
The proposed code may be useful sometimes, but is not generic enough for the standard library.  For example, the f() function can only take one argument, when g() can accept any number.

Implementations of this kind are so easy to write, They are better described by their implementation rather than documentation.
IMO they show the expressiveness of python, and don't need to be hidden in a C module.
msg111400 - (view)	Author: Raymond Hettinger (rhettinger) * (Python committer)	Date: 2010-07-23 23:48
I agree with Amaury that this should be closed.  It has been previously discussed and rejected in other forums.  One the issues is that the usual mathematical order is unintuitive and not self-documenting  -- i.e. is compose(f,g)  the same as f(g(x)) or g(f(x))?  Also, it is already dirt simple to create your own compose function or to do the composition directly:  h = lambda x: f(g(x))."""
    with Timer():
        my_tokens = my_tokenizer(btxt, ends=True)
    # print(my_tokens)
    # print(my_tokenizer.decode(my_tokens))
    import transformers

    vocab_my_way = json.load(open("bpe_tokens.json"))
    print(vocab_my_way[600:700])
    bpe_tokenizer = BPETokenizer(vocab_my_way)
    tokens = bpe_tokenizer._tokenize("hi, my name is tao")
    print([bpe_tokenizer.vocab_by_id[i]["piece"] for i in tokens])
