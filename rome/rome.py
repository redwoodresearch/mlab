import random
from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, Dict, Generator, List, Optional, Tuple

import torch as t
import transformers
from black import json
from einops import repeat
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


def find_most_activated_layer(
    tokenizer: transformers.PreTrainedTokenizer,
    model: nn.Module,
    layers: List[nn.Module],
    fact: Fact,
) -> t.Tensor:
    """ """
    model.eval()

    perturbations = generate_random_pertubations(fact, tokenizer)

    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(
        [fact_to_sentence_without_object(f) for f in [fact] + perturbations],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )["input_ids"]
    print(tokenizer.decode(inputs[0]))

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
            print("Some of your perturbations return the same value")

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


def generate_random_pertubations(
    fact: Fact, tokenizer: transformers.PreTrainedTokenizer
) -> List[str]:
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
