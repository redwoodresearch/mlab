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
    def __init__(self, token_list, pad=0, sep=1, cls=2, unk=3, mask=4, normalizer=str.lower):
        self.replacements = {"▁": " "}
        for obj in token_list:
            obj["piece"] = self._replace_all(obj["piece"])
        self.token_list = token_list
        self.text_to_id = None
        self.id_to_text = None
        self.pad = pad
        self.cls = cls
        self.mask = mask
        self.unk = unk
        self.vocab = {x["piece"]: x for x in token_list}
        self.vocab_by_id = {x["id"]: x for x in token_list}
        self.normalizer = normalizer

    def _replace_all(self, text):
        return functools.reduce(lambda a, x: a.replace(x[0], x[1]), self.replacements.items(), text)

    def tokenize(self, texts):
        if isinstance(texts, str):
            return self._tokenize(texts)
        results = []
        for text in texts:
            results.append(self._tokenize(text))
        return results

    def __call__(self, texts):
        return self.tokenize(texts)

    def _tokenize(self, text):
        if self.normalizer is not None:
            text = self.normalizer(text)
        neg_loglik, best_subw_slices = self.viterbi_forward(text)
        ids = self.viterbi_backward(text, best_subw_slices)
        print("".join([self.vocab_by_id[id]["piece"] for id in ids]))
        return ids

    def viterbi_forward(self, sequence):
        """Forward step of Viterbi given a single word."""
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
