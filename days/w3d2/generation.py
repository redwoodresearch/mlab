"""Utilities for generating text with the day's models."""
import json
from typing import List

import torch as t
from transformers import GPT2Tokenizer

from w3d2_tests import MiniGPT

# 50258 is the pad_token_id
# 50257 is the BEGIN token id
BEGIN_TOKEN_ID = 50257
PAD_TOKEN_ID = 50258


def gen_completion(
    model: MiniGPT,
    tokenizer: GPT2Tokenizer,
    base_text: str,
    n_tokens_to_gen: int,
    temperature: float,
    freq_penalty: float,
) -> List[str]:
    # base_ids: List[int] = [BEGIN_TOKEN_ID] + tokenizer(base_text)["input_ids"]
    base_ids: List[int] = tokenizer(f"[BEGIN] {base_text}")["input_ids"]

    completion_ids: List[int] = []
    for _ in range(n_tokens_to_gen):
        input_ids_tensor = t.tensor(
            base_ids + completion_ids,
            dtype=t.long,
            device=model.device,
        )
        logits = model(input_ids_tensor.reshape(1, -1))[0, -1]
        assert logits.shape == (model.vocab_size,)

        id_freqs = t.bincount(input_ids_tensor, minlength=model.vocab_size)
        probs = t.softmax(logits / temperature - id_freqs * freq_penalty, dim=0)

        next_token_id: int = t.multinomial(probs, 1)[0].item()
        # next_token_id: int = t.argmax(logits).item()
        completion_ids.append(next_token_id)

    return tokenizer.decode(base_ids + completion_ids)
