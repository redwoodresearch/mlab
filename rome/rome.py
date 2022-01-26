import random
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, Generator, List, Optional, Tuple
from numpy import indices

import torch as t
import transformers
from black import json
from datasets import load_dataset
from einops import rearrange, repeat
from torch import nn


@dataclass
class Fact:
    subject: str  # do the subjects all need to be the same length?
    predicate: str
    object: Optional[str]

    @classmethod
    def load(cls, json_fp):  # -> Generator["Fact"]:
        for d in json.load(json_fp):
            yield cls(**d)


ALL_FACTS = list(Fact.load(open("fact_examples.json")))
ALL_SUBJECTS = [
    "Megan Rappinoe",
    "Patrick Mahomes",
    "Tom Brady",
    "Alex Rodriguez",
    "David Beckham",
]  # [fact.subject for fact in ALL_FACTS]

NUM_SUBJECTS_TO_SAMPLE = 5

# this could actually use unit tests


def inputs(tokenizer: transformers.PreTrainedTokenizer, fact: Fact) -> t.Tensor:
    tokenizer.pad_token = tokenizer.eos_token
    perturbations = generate_random_pertubations(fact)
    inputs = tokenizer(
        [fact_to_sentence_without_object(f) for f in [fact] + perturbations],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )["input_ids"]
    print(tokenizer.decode(inputs[0]))
    return inputs


def find_most_activated_layer(
    tokenizer: transformers.PreTrainedTokenizer,
    model: nn.Module,
    layers: List[nn.Module],
    inputs: t.Tensor,
    fact: Fact,
) -> t.Tensor:
    """ """
    model.eval()

    patch_activations: Dict[int, (int, t.Tensor)] = {}
    activations_per_layer: List[t.Tensor] = [t.tensor(0) for _ in range(len(layers))]

    def gen_hook(layer_index: int):
        def hook(
            model: nn.Module, input: t.Tensor, output: t.Tensor
        ) -> Optional[t.Tensor]:
            activations_per_layer[layer_index] = output[0]
            if layer_index in patch_activations:
                index, activations = patch_activations[layer_index]
                output[0][:, index] = activations
                return output

        return hook

    handles = []
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(gen_hook(i)))

    try:
        # we can guarantee that all the facts & perturbations have the same length
        with t.no_grad():
            # do we even need to run anything through the model except the correct answer?
            logits = model(inputs.to(model.device)).logits

        # double check that the answers are indeed different
        if logits[0, -1].argmax() in logits[1:, -1].argmax(dim=-1):
            print("Some of your perturbations return the target value")

        # double check that the fact returns the right answer
        if not (" " + fact.object).startswith(tokenizer.decode(logits[0, -1].argmax())):
            print("Your fact is not in the model")

        # save output of activations
        reference_activations = activations_per_layer.copy()

        desired_token = logits[0, -1].argmax(dim=-1).item()
        p_desired_token = []

        # todo: try to reduce time spent by parallelizing the double for loop below?

        # patch original values at each layer of each pertubation and find loss for each
        for i, layer in enumerate(layers):
            p_desired_token_for_layer = []
            for token_idx in range(inputs.shape[1]):
                # 0 is the index of the fact we're interested in

                activations = reference_activations[i][:1, token_idx]
                patch_activations[i] = (
                    token_idx,
                    activations,
                )

                with t.no_grad():
                    replaced_logits = model(inputs.to(model.device)).logits
                    del patch_activations[i]

                    p_desired_token_for_layer.append(
                        t.softmax(replaced_logits[1:, -1], dim=-1)[
                            :, desired_token
                        ].mean(dim=0)
                    )
            p_desired_token.append(t.stack(p_desired_token_for_layer))
        return t.stack(p_desired_token)

    finally:
        for handle in handles:
            handle.remove()


def generate_random_pertubations(fact: Fact) -> List[str]:
    """
    Going to be easiest to just substitute different subjects in here.

    Should we be operating on the tokens instead?
    """
    return [
        Fact(subject=subject, predicate=fact.predicate, object=fact.object)
        for subject in ALL_SUBJECTS
    ]


def fact_to_sentence_without_object(fact: Fact) -> str:
    """
    Super dumb for now. May get smarter someday
    """
    return f"{fact.subject} {fact.predicate}"


def find_region_of_max_causal_effect(probs: t.Tensor) -> Tuple[int, int]:
    """Takes the output of find_most_activated_layer. Returns point (layer, token_idx)"""
    best_token_idx = probs.sum(dim=0).argmax().item()
    best_layer_idx = probs[:, best_token_idx].argmax().item()
    return (best_layer_idx, best_token_idx)


def maximize_activations_at_layer(
    tokenizer: transformers.PreTrainedTokenizer,
    model: nn.Module,
    layers: List[nn.Module],
    layer_idx: int,
    token_idx: int,
    inputs: t.Tensor,
    new_fact: Fact,
) -> t.Tensor:
    epochs = 1000

    inputs = inputs[:1]

    # should i use smarter initialization here? like divide by sqrt n or whatever
    activations = t.rand((1, 1600)).to(model.device).requires_grad_(True)

    def hook(model: nn.Module, input: t.Tensor, output: t.Tensor) -> Optional[t.Tensor]:
        # trying to retain grad-ability here
        output[0][:, token_idx] = t.zeros_like(activations)
        output[0][:, token_idx] += activations
        return output

    handle = layers[layer_idx].register_forward_hook(hook)

    # this better only be one token long!
    target = tokenizer.encode(" " + new_fact.object, return_tensors="pt")[0, 0]
    target = target.unsqueeze(0).to(model.device)
    print(tokenizer.decode(target))

    loss_fn = nn.CrossEntropyLoss()

    try:
        optim = t.optim.Adam(lr=1e-3, params=[activations])
        for _ in range(epochs):
            optim.zero_grad()
            logits = model(inputs.to(model.device)).logits
            loss = loss_fn(logits[:, -1], target)
            loss.backward()
            print(loss)
            optim.step()

        print(loss)
        print(t.softmax(logits[0, -1], dim=0).topk(10).indices)
        print(tokenizer.decode(t.softmax(logits[0, -1], dim=0).topk(10).indices))
        print(t.softmax(logits[0, -1], dim=0).topk(10).values)
        return activations

    finally:
        handle.remove()


def find_layer_hidden_state(
    model: nn.Module,
    layer_idx: int,
    token_idx: int,
    inputs: t.Tensor,
):
    inputs = inputs[:1]

    hidden_state = t.zeros((4 * 1600,))
    # output_to_return = t.zeros(
    #     (
    #         6,
    #         1600,
    #     )
    # )

    def hook(model: nn.Module, input: t.Tensor, output: t.Tensor) -> Optional[t.Tensor]:
        hidden_state[:] = input[0][0, token_idx]
        # output_to_return[:] = output[0]

    handle = model.transformer.h[layer_idx].mlp.c_proj.register_forward_hook(hook)

    try:
        with t.no_grad():
            model(inputs.to(model.device))
        return hidden_state  # , output_to_return
    finally:
        handle.remove()


def find_average_inputs_to_layer(
    tokenizer: transformers.PreTrainedTokenizer,
    model: nn.Module,
    layer_idx: int,
):
    """Probably gonna want to cache this somewhere for the next time"""
    batch_size = 8
    batches = 200
    model.eval()
    # load all of fucking wikipedia

    dataset = load_dataset("wikitext", "wikitext-103-v1")

    tokenizer.pad_token = tokenizer.eos_token

    hidden_states = []

    def hook(model: nn.Module, input: t.Tensor, output: t.Tensor) -> Optional[t.Tensor]:
        hidden_states.append(rearrange(input[0], "b s h -> (b s) h"))

    handle = model.transformer.h[layer_idx].mlp.c_proj.register_forward_hook(hook)
    try:
        for batch_idx in range(batches):
            inputs = tokenizer(
                dataset["train"][batch_idx * batch_size : (batch_idx + 1) * batch_size][
                    "text"
                ],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )["input_ids"].to(model.device)

            with t.no_grad():
                model(inputs.to(model.device))
    finally:
        handle.remove()

    K = t.cat(hidden_states, dim=0)

    return t.cov(K.T @ K)


def new_weights_from_activation(
    tokenizer: transformers.PreTrainedTokenizer,
    model: nn.Module,
    layer_idx: int,
    hidden_state: t.Tensor,
    activations: t.Tensor,
    C: t.Tensor,
) -> None:
    layer = model.transformer.h[layer_idx]
    weights = layer.mlp.c_proj.weight

    k_star = hidden_state.to(model.device)
    v_star = activations.to(model.device)

    u_T = t.einsum("ij, j -> i", t.inverse(C), k_star)

    I_concat = t.cat(
        [t.eye(k_star.shape[0]).to(model.device), k_star.unsqueeze(1)], dim=1
    )
    u_T_plus_zero = t.cat([u_T, t.zeros((1,)).to(model.device)], dim=0).unsqueeze(0)
    I_concat = t.cat([I_concat, -u_T_plus_zero], dim=0)

    W_hat_plus_v = t.einsum(
        "ij, ik -> kj", t.cat([weights, v_star], dim=0), t.inverse(I_concat)
    )

    return W_hat_plus_v[:-1, :]

    # another method
    # v = W_hat_plus_v[-1, :]
    # other_W_hat = weights + t.einsum("i, j -> ij", v, u_T).T
    # return other_W_hat


def test_on_fact(
    model: nn.Module, tokenizer: transformers.PreTrainedTokenizer, fact: Fact
) -> str:
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(
        [fact_to_sentence_without_object(fact)],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )["input_ids"].to(model.device)

    model.eval()
    with t.no_grad():
        logits = model(inputs.to(model.device)).logits
    return logits

    # return tokenizer.decode(logits[0, -1].argmax(dim=0))
