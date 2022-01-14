from typing import *
import torch as t
import numpy as np
from torch.nn import (
    Module,
    Parameter,
    ModuleList,
    Sequential,
)  # not allowed to use other stuff from nn
from transformers import AutoTokenizer

from days.modules import (
    Embedding,
    Dropout,
    Linear,
    LayerNorm,
    gelu,
    log_softmax,
    softmax,
)
from torchtyping import TensorType

# from torch.nn import LayerNorm
# from torch.nn.functional import gelu, softmax
from days.utils import getprops, tpeek, copy_weight_bias
from dataclasses import dataclass
import transformers

from torch import nn
from einops import reduce, rearrange, repeat


class GPT2Attention(Module):
    def __init__(self, config):
        super().__init__()
        config = convert_hf_to_my_config(config)
        self.config = config
        hidden_size = config["hidden_size"]
        max_positions = config["max_position_embeddings"]
        self.c_attn = Linear(hidden_size, hidden_size * 3)
        self.c_proj = Linear(hidden_size, hidden_size)
        self.dropout = Dropout(config["dropout"])
        self.register_buffer(
            "mask",
            t.tril(t.ones((max_positions, max_positions), dtype=t.bool))
            .flip(0)
            .flip(1)
            .unsqueeze(0)
            .unsqueeze(0),
        )
        self.masked_bias = t.tensor(-1e4)

    def forward(
        self,
        encodings,
        attention_masks=None,
        past_key_values: Optional[Tuple[TensorType["h", "s", "head_size"], ...]] = None,
    ):
        num_heads = self.config["num_heads"]
        head_size = self.config["hidden_size"] // num_heads
        output_sequence_length = encodings.shape[1]
        input_sequence_length = encodings.shape[1]
        cattn = self.c_attn(encodings)
        query, key, value = t.split(cattn, self.config["hidden_size"], -1)
        query = rearrange(query, "b s (h c) -> b h s c", h=num_heads)

        key = rearrange(key, "b s (h c) -> b h s c", h=num_heads)
        value = rearrange(value, "b s (h c) -> b h s c", h=num_heads)

        unidirectional_mask = self.mask[
            :, :, :output_sequence_length, :output_sequence_length
        ]

        if past_key_values is not None:
            new_kvs = (key[0], value[0])
            key = t.cat([past_key_values[0].unsqueeze(0), key], dim=2)
            value = t.cat([past_key_values[1].unsqueeze(0), value], dim=2)
            input_sequence_length += past_key_values[0].shape[1]
            unidirectional_mask = t.cat(
                [
                    t.full(
                        (
                            1,
                            1,
                            input_sequence_length - output_sequence_length,
                            output_sequence_length,
                        ),
                        True,
                    ),
                    unidirectional_mask,
                ],
                2,
            )

        attention_raw = t.einsum("bhtc,bhfc->bhft", query, key) / np.sqrt(head_size)

        attention_raw = t.where(unidirectional_mask, attention_raw, self.masked_bias)
        if attention_masks is not None:
            attention_raw = attention_raw * attention_masks
        attention_patterns = nn.Softmax(dim=-2)(attention_raw)

        # print("value shape", value.shape)
        # print("attention shape", attention_raw.shape)
        context_layer = t.einsum("bhft,bhfc->bhtc", attention_patterns, value)
        attention_values = rearrange(context_layer, "b h s c -> b s (h c)")
        attention_values = self.c_proj(attention_values)
        if past_key_values is not None:
            return attention_values, new_kvs
        return attention_values


class GPT2Layer(Module):
    def __init__(self, config):
        super().__init__()
        config = convert_hf_to_my_config(config)
        self.config = config
        hidden_size = config["hidden_size"]
        self.layer_norm_1 = LayerNorm((hidden_size,), eps=config["layer_norm_epsilon"])
        self.attention = GPT2Attention(config)
        # gpt2 calls these fully connected layers "conv1d", but they're actually 100% identical to linear layers
        self.fc1 = Linear(hidden_size, hidden_size * 4)
        self.fc2 = Linear(hidden_size * 4, hidden_size)
        self.layer_norm_2 = LayerNorm((hidden_size,), eps=config["layer_norm_epsilon"])
        self.dropout = Dropout(config["dropout"])

    def forward(self, x: t.Tensor, past_key_values=None):
        if past_key_values is not None:
            attention_output, new_kvs = self.attention(
                self.layer_norm_1(x), past_key_values=past_key_values
            )
        else:
            attention_output = self.attention(self.layer_norm_1(x))
        x = x + attention_output
        after_1_mlp = gelu(self.fc1(self.layer_norm_2(x)))
        mlpout = self.dropout(self.fc2(after_1_mlp))
        x = x + mlpout
        if past_key_values is not None:
            return x, new_kvs
        return x


@dataclass
class GPT2Output:
    logits: TensorType["batch_size", "seq_length", "vocab_size"]
    final_encoding: TensorType["batch_size", "seq_length", "hidden_size"]


class GPT2(Module):
    def __init__(self, config, tokenizer=None):
        super().__init__()
        config = convert_hf_to_my_config(config)
        default_config = {
            "bos_token_id": 50256,
            "eos_token_id": 50256,
            "initializer_range": 0.02,
            "layer_norm_epsilon": 1e-05,
            "hidden_size": 768,
            "num_heads": 12,
            "num_layers": 12,
            "max_position_embeddings": 1024,
            "dropout": 0.1,
            "scale_attn_weights": True,
            "use_cache": False,
            "vocab_size": 50257,
        }
        self.tokenizer = tokenizer
        # convert config params from HF
        config = {**default_config, **config}
        self.config = config
        self.token_embedding = Embedding(config["vocab_size"], config["hidden_size"])
        self.position_embedding = Embedding(
            config["max_position_embeddings"], config["hidden_size"]
        )
        self.dropout = Dropout(config["dropout"])
        self.blocks = Sequential(
            *[GPT2Layer(config) for _ in range(config["num_layers"])]
        )
        self.layer_norm_final = LayerNorm(
            (config["hidden_size"],), eps=config["layer_norm_epsilon"]
        )
        # cache is some sequences of tokens
        # how much space per token? hidden_size*seq_length*(layers*2+1)
        # 768*100*25 = 10M, aka very little. Definitely have space for 10 sequences

        # I'm only caching batch size one rn because simple for teaching
        self.cache = []

    def forward(self, input_ids: t.LongTensor):
        seq_length = input_ids.shape[1]
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(
            t.arange(seq_length).to(next(self.parameters()).device)
        )

        embeddings = token_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)

        # cache only works for batch size 1
        if self.config["use_cache"] and input_ids.shape[0] == 1:
            print("using cache")
            input_ids = input_ids[0]
            for idx, (current_id, (id, layer_kvs, _last_embedding)) in enumerate(
                zip(input_ids, self.cache)
            ):
                if current_id != id:
                    self.cache = self.cache[:idx]
                    break
            encodings = embeddings[:, len(self.cache) :]
            new_cache_stuff = [
                [id.item(), [], None] for id in input_ids[len(self.cache) :]
            ]
            for layer_num, block in enumerate(self.blocks):
                # this silly if because t.cat doesn't work for 0 length list (which kinda makes sense)
                if len(self.cache) > 0:
                    past_keys = t.stack(
                        [layer_kvs[layer_num][0] for id, layer_kvs, _ in self.cache],
                        dim=1,
                    )
                    past_values = t.stack(
                        [layer_kvs[layer_num][1] for id, layer_kvs, _ in self.cache],
                        dim=1,
                    )
                else:
                    past_keys = t.zeros(
                        12, 0, self.config["hidden_size"] // self.config["num_heads"]
                    )
                    past_values = t.zeros(
                        12, 0, self.config["hidden_size"] // self.config["num_heads"]
                    )
                encodings, kvs = block(
                    encodings, past_key_values=(past_keys, past_values)
                )
                for id_num, (id, layer_kvs, _last_embedding) in enumerate(
                    new_cache_stuff
                ):
                    layer_kvs.append((kvs[0][:, id_num], kvs[1][:, id_num]))
            encodings = self.layer_norm_final(encodings)
            for idx, thing in enumerate(new_cache_stuff):
                thing[2] = encodings[0, idx]
            if len(self.cache) > 0:
                encodings = t.cat(
                    [
                        t.stack([x[2] for x in self.cache], dim=0).unsqueeze(0),
                        encodings,
                    ],
                    dim=1,
                )
            self.cache.extend(new_cache_stuff)
        else:
            encodings = self.blocks(embeddings)
            encodings = self.layer_norm_final(encodings)

        final_encoding = encodings[:, -1, :]
        logits = t.einsum("...i,ji->...j", encodings, self.token_embedding.weight)
        return GPT2Output(logits=logits, final_encoding=final_encoding)

    def next_token(self, input_ids, temperature, freq_penalty=0):
        last_outputs = self(input_ids=input_ids.unsqueeze(0)).logits[0][-1]
        last_outputs /= temperature
        if freq_penalty > 0:
            id_freqs = (
                t.bincount(input_ids, minlength=self.config["vocab_size"])
                * freq_penalty
            )
            last_outputs -= id_freqs
        choice = t.distributions.categorical.Categorical(
            probs=t.nn.functional.softmax(last_outputs, dim=-1)
        )
        c = choice.sample()
        return c

    def generate_ids(self, input_ids, max_length=30, temperature=1, freq_penalty=2):
        ids = input_ids
        for _ in range(max_length):
            c = self.next_token(
                input_ids=ids, temperature=temperature, freq_penalty=freq_penalty
            )
            if (
                self.tokenizer.eos_token_id is not None
                and c == self.tokenizer.eos_token_id
            ):
                break
            ids = t.cat([ids, c.reshape(1)], dim=0)
        return ids[input_ids.shape[0] :]

    def generate(self, text, max_length=30, temperature=1, freq_penalty=2):
        assert self.tokenizer is not None
        print(self.tokenizer(text))
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"][0]
        tpeek("iids", input_ids)
        generated_ids = self.generate_ids(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            freq_penalty=freq_penalty,
        )
        completion_text = self.tokenizer.decode(generated_ids)
        return completion_text

    def _get_logprob_of_logits(self, input_ids, logits):
        logprobs = log_softmax(logits, dim=-1)
        input_ids = input_ids[1:]
        logits = logits[:-1]
        scores = t.gather(logprobs, -1, input_ids.unsqueeze(-1)).squeeze(-1)
        scores = t.nan_to_num(scores, nan=-30, posinf=-30, neginf=-30)
        return t.sum(scores)

    def generate_beam_search_ids(
        self, input_ids, max_length=10, beam_width=5, freq_penalty=2
    ):
        candidates = [(0, input_ids)]
        for i in range(max_length):
            next_candidates = []
            for score, input_ids in candidates:
                outputs = self(input_ids=input_ids.unsqueeze(0)).logits[0]
                cost = self._get_logprob_of_logits(input_ids, outputs)
                last_outputs = log_softmax(outputs[-1])
                if freq_penalty > 0:
                    id_freqs = (
                        t.bincount(input_ids, minlength=self.config["vocab_size"])
                        * freq_penalty
                    )
                    last_outputs -= id_freqs
                # Prune to the top k for each candidate because no more than k be in the top k of all candidates
                topk_values, topk_indices = t.topk(last_outputs, dim=0, k=beam_width)
                next_candidates.extend(
                    [
                        (cost + value, t.cat([input_ids, index.reshape(1)], dim=0))
                        for value, index in zip(topk_values, topk_indices)
                    ]
                )
            next_candidates = sorted(next_candidates, key=lambda x: x[0])[:beam_width]
            candidates = next_candidates
        return sorted(candidates, key=lambda x: x[0])[0][1]

    def generate_beam_search(self, text, max_length=10, beam_width=5, freq_penalty=2):
        input_ids = self.tokenizer([text], return_tensors="pt")["input_ids"][0]
        return self.tokenizer.decode(
            self.generate_beam_search_ids(
                input_ids,
                max_length=max_length,
                beam_width=beam_width,
                freq_penalty=freq_penalty,
            )
        )

    def specific_completion_probs_ids(self, input_ids, completions_ids):
        result = []
        for completion_ids in completions_ids:
            ids = t.cat([input_ids, completion_ids], dim=0)
            outputs = self(input_ids=ids.reshape(1, -1)).logits[0]
            completion_outputs = outputs[input_ids.shape[0] :]
            completion_logprobs = t.nn.functional.log_softmax(
                completion_outputs, dim=-1
            )
            indices = t.stack(
                [t.arange(0, completion_ids.shape[0]), completion_ids], dim=-1
            )
            gathered = t.gather(completion_logprobs, -1, indices)
            gathered = t.nan_to_num(gathered, nan=-15, posinf=-15, neginf=-15)
            prob = reduce(gathered, "...->", "sum")
            result.append(prob.item())
        return result

    def specific_completion_probs(self, text, completion_texts):
        assert self.tokenizer is not None
        input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"][0]
        completion_ids = [
            self.tokenizer(t, return_tensors="pt")["input_ids"][0]
            for t in completion_texts
        ]
        return self.specific_completion_probs_ids(
            input_ids, completions_ids=completion_ids
        )


def convert_hf_to_my_config(hf_config):
    if isinstance(hf_config, dict):
        return hf_config
    hf_config = hf_config.to_dict()
    key_map = {
        "resid_pdrop": "dropout",  # mine doesn't confiure attention and resid dropout seperately
        "n_layer": "num_layers",
        "n_embd": "hidden_size",
        "n_head": "num_heads",
        "n_ctx": "max_position_embeddings",
    }
    return {(key_map.get(k, k)): v for k, v in hf_config.items()}


def copy_gpt2_attention_weights(my_attn, their_attn):
    copy_weight_bias(my_attn.c_attn, their_attn.c_attn, transpose=True)
    copy_weight_bias(my_attn.c_proj, their_attn.c_proj, transpose=True)


def copy_gpt2_layer_weights(my_layer, their_layer):
    copy_weight_bias(my_layer.fc1, their_layer.mlp.c_fc, transpose=True)
    copy_weight_bias(my_layer.fc2, their_layer.mlp.c_proj, transpose=True)

    copy_weight_bias(my_layer.layer_norm_1, their_layer.ln_1)
    copy_weight_bias(my_layer.layer_norm_2, their_layer.ln_2)
    copy_gpt2_attention_weights(my_layer.attention, their_layer.attn)


def copy_gpt2_weights(to_model, from_model):
    their_model: transformers.models.gpt2.modeling_gpt2.GPT2Model = (
        from_model.transformer
    )
    to_model.token_embedding.weight = their_model.wte.weight
    to_model.position_embedding.weight = their_model.wpe.weight
    for their_layer, my_layer in zip(their_model.h, to_model.blocks):
        copy_gpt2_layer_weights(my_layer, their_layer)

    copy_weight_bias(to_model.layer_norm_final, their_model.ln_f)


def my_gpt_from_hf_weights():
    their_lm_model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    their_lm_model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

    my_model = GPT2({"use_cache": False}, tokenizer=tokenizer)
    my_model.eval()
    copy_gpt2_weights(my_model, their_lm_model)
    # not supporting cross attention
    return my_model, their_lm_model
