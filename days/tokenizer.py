import transformers
import numpy as np
import sentencepiece as spm
from utils import Timer, getprops
import json
import functools
import torch as t
from unidecode import unidecode
from collections import defaultdict


def normalizer(str):
    return unidecode(str.lower().replace("\t", "    ").replace("\n", ""))


# bpe and wordpiece work by splitting to chars then merging by tokens in order! i didn't know this!

# main tokenizers: bpe, wordpiece, sentencepiece, unigram lm
# how unigram lm works: pick the tokenization such that a unigram lm on them has the best loss
# https://everdark.github.io/k9/notebooks/ml/natural_language_understanding/subword_units/subword_units.nb.html
class ViterbiTokenizer:
    def __init__(self, token_list, pad=0, sep=1, cls=None, bot=2, eot=2, unk=3, mask=4, normalizer=normalizer):
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

    def create_viterbi_tokenizer_from_corpus(corpus_string, max_token_length=20):
        corpus_sentences = [x + "." for x in corpus_string.split(".")]
        substring_frequencies = ViterbiTokenizer.get_substring_frequencies(corpus_sentences, max_token_length)

    def get_substring_frequencies(sentences, max_length):
        counter = defaultdict(lambda: 0)
        for sentence in sentences:
            for i in range(len(sentence)):
                for ln in range(1, min(len(sentence) - i, max_length + 1)):
                    counter[sentence[i:ln]] += 1
        return counter

    def _replace_all(self, text):
        return functools.reduce(lambda a, x: a.replace(x[0], x[1]), self.replacements.items(), text)

    def tokenize(self, texts, **kwargs):
        if isinstance(texts, str):
            return self._tokenize(texts, **kwargs)
        results = []
        for text in texts:
            results.append(self._tokenize(text))
        return results

    def __call__(self, texts, **kwargs):
        return self.tokenize(texts, **kwargs)

    def _tokenize(self, text, pad_length=None, ends=False):
        if self.normalizer is not None:
            text = self.normalizer(text)
        print(text)
        best_subw_slices = self.viterbi_forward(text)
        print("did forward")
        ids = self.viterbi_backward(best_subw_slices)
        if ends:
            ids = [self.bot] + ids + [self.eot]
        if pad_length:
            ids.extend([self.pad] * (pad_length - len(ids)))
        return ids

    def decode(self, ids):
        ids = list(ids)
        return "".join([self.vocab_by_id[id]["piece"] for id in ids])

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
    print(my_tokens)
    print(my_tokenizer.decode(my_tokens))
