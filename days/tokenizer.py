import transformers
import numpy as np
import sentencepiece as spm
from utils import Timer, getprops
import json
import functools

# bpe and wordpiece work by splitting to chars then merging by tokens in order! i didn't know this!

# main tokenizers: bpe, wordpiece, sentencepiece, unigram lm
# how unigram lm works: pick the tokenization such that a unigram lm on them has the best loss
# https://everdark.github.io/k9/notebooks/ml/natural_language_understanding/subword_units/subword_units.nb.html
class ViterbiTokenizer:
    def __init__(self, token_list, pad=0, sep=1, cls=None, bot=2, eot = 2, unk=3, mask=4, normalizer=str.lower):
        self.replacements = {"▁": " "}
        for obj in token_list:
            obj["piece"] = self._replace_all(obj["piece"])
        self.token_list = token_list
        self.text_to_id = None
        self.id_to_text = None
        self.pad = pad
        if cls:
            self.eot = cls
            self.bot = cls
        else:
            self.eot = eot
            self.bot = bot
        self.mask = mask
        self.unk = unk
        self.vocab = {x["piece"]: x for x in token_list}
        self.vocab_by_id = {x["id"]: x for x in token_list}
        self.normalizer = normalizer

    def _replace_all(self, text):
        return functools.reduce(lambda a, x: a.replace(x[0], x[1]), self.replacements.items(), text)

    def tokenize(self, texts, **kwargs):
        if isinstance(texts, str):
            return self._tokenize(texts,**kwargs)
        results = []
        for text in texts:
            results.append(self._tokenize(text))
        return results

    def __call__(self, texts,**kwargs):
        return self.tokenize(texts,**kwargs)

    def _tokenize(self, text, pad_length = None, ends = False):
        if self.normalizer is not None:
            text = self.normalizer(text)
        neg_loglik, best_subw_slices = self.viterbi_forward(text)
        ids = self.viterbi_backward(text, best_subw_slices)
        print("".join([self.vocab_by_id[id]["piece"] for id in ids]))
        if ends:
            ids  = [self.bot] + ids + [self.eot]
        if pad_length:
            ids.extend([self.pad]*(pad_length-len(ids)))
        return ids

    def viterbi_forward(self, sequence):
        """Forward step of Viterbi."""
        # Create storage array for best substring recorded at each end-of-character position.
        best_subw_slices = [None] * (len(sequence) + 1)
        neg_loglik = np.zeros(len(sequence) + 1)
        # Search the best substring given every possible end of position along the word.
        for eow in range(1, len(sequence) + 1):
            # For every end-of-word position:
            neg_loglik[eow] = 9999999999
            for bow in range(eow):
                # For every possible beginning-of-word position given the end-of-word position:
                subw = sequence[bow:eow]
                if subw in self.vocab:
                    print("found in vocab", subw)

                    logp = self.vocab[subw]["score"]
                    # Compute subword probability:
                    # P(current segment) + P(the best segment just before the current segment).
                    s = neg_loglik[bow] - logp
                    print("here", s, "there", neg_loglik[eow])
                    if s < neg_loglik[eow]:
                        neg_loglik[eow] = s
                        best_subw_slices[eow] = (bow, eow)
        return neg_loglik, best_subw_slices

    def viterbi_backward(self, sequence, subw_slices):
        """Backward step of Viterbi to return the best path."""
        subwords = []
        subword_slices = []
        next_slices = subw_slices[-1]
        while next_slices is not None:
            subw = sequence[next_slices[0] : next_slices[1]]
            subwords.append(self.vocab[subw]["id"])
            subword_slices.append(next_slices)
            next_slices = subw_slices[next_slices[0]]
        subwords.reverse()
        return subwords


# "protoc -I=. --python_out=./proto ./spiece.proto"
if __name__ == "__main__":
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
    my_tokenizer = ViterbiTokenizer(token_list=dct)
    with Timer():
        my_tokens = my_tokenizer(example_str)
    print(my_tokens)
    btxt = """int:   git config pull.rebase false  # merge (the default strategy)
hint:   git config pull.rebase true   # rebase
hint:   git config pull.ff only       # fast-forward only
hint: 
hint: You can replace "git config" with "git config --global" to set a default
hint: preference for all repositories. You can also pass --rebase, --no-rebase,
hint: or --ff-only on the command line to override the configured default per
hint: invocation.
Updating e067129..b21c5a9
Fast-forward
 .gitignore       |   3 +-
 days/bert.py     |  20 +-----
 days/bert_run.py |  19 ++++--
 days/gpt2.py     | 138 +++++++++++++++++++++++++++++++++++++++++
 utils.py         |  15 +++++
 5 files changed, 173 insertions(+), 22 deletions(-)
 create mode 100644 days/gpt2.py
tao@Taos-MacBook-Air mlab % git pull
hint: Pulling without specifying how to reconcile divergent branches is
hint: discouraged. You can squelch this message by running one of the following
hint: commands sometime before your next pull:
hint: 
hint:   git config pull.rebase false  # merge (the default strategy)
hint:   git config pull.rebase true   # rebase
hint:   git config pull.ff only       # fast-forward only
hint: 
hint: You can replace "git config" with "git config --global" to set a default
hint: preference for all repositories. You can also pass --rebase, --no-rebase,
hint: or --ff-only on the command line to override the configured default per
hint: invocation.
Updating b21c5a9..4caa896
Fast-forward
 days/bert.py            |     46 +-
 days/gpt2.py            |     92 +-
 days/modules.py         |     75 +-
 days/resnet.py          |     64 +-
 days/sentencepiece.json | 180002 +++++++++++++++++++++++++++++++
 days/spiece.model       |    Bin 0 -> 760289 bytes
 days/tokenizer.py       |    106 +
 test_all.py             |     30 +-
 utils.py                |     29 +-
 9 files changed, 180344 insertions(+), 100 deletions(-)"""
    with Timer():
        my_tokens = my_tokenizer(example_str)
    print(my_tokens)
    