from re import X
import torch as t
from days.w3d5 import gpt_mod
from days.w3d5.hook_handler import HookHandler
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast


def get_gpt2_contribs(
    text: str,
    gpt2: gpt_mod.GPT2,
    tokenizer: GPT2TokenizerFast,
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """Returns (res, head, ln_bias, sum)."""
    res_embedding = None
    head_embeddings = []

    def he_hook(
        block: gpt_mod.GPT2Block,
        inputs: tuple[t.Tensor],
        outputs: t.Tensor,
    ):
        _, out_headwise = block.forward(inputs[0], return_headwise=True)
        head_embeddings.append(out_headwise)

    with HookHandler() as hh:

        def foo(_, inputs, __):
            nonlocal res_embedding
            res_embedding = inputs[0]

        hh.add_hook(gpt2.blocks[0], foo)

        for block in gpt2.blocks:
            hh.add_hook(block, he_hook)

        input_ids = tokenizer.encode(text, return_tensors="pt")
        gpt_out = gpt2(input_ids)

        head_embeddings = t.stack(head_embeddings, dim=1)  # stack layers together
        head_embeddings = head_embeddings.squeeze(0)  # eliminate batch dim
        res_embedding = res_embedding.squeeze(0)

    # we subtract off the mean of each term going through layer norm
    # this is because layer norm subtracts off the mean of the vector
    # and thus we don't care about the relative magnitudes of the terms
    res_embedding -= res_embedding.mean(dim=-1, keepdim=True)
    head_embeddings -= head_embeddings.mean(dim=-1, keepdim=True)

    sum_of_embeddings = head_embeddings.sum(dim=(0, 1)) + res_embedding

    l_norms = t.sqrt(
        t.var(
            sum_of_embeddings,
            dim=-1,
            keepdim=True,
            unbiased=False,
        )
        + gpt2.ln.eps
    )

    ln_bias = gpt2.ln.bias
    ln_weight = gpt2.ln.weight
    unembed_matrix = gpt2.token_embedding.weight

    def embed_to_contribs(x):
        # layer norm calculations, ignoring bias (which is a separate term)
        # we've also already subtracted off the means
        ln_out = ln_weight * (x / l_norms)
        return t.einsum("ve, ...e -> ...v", unembed_matrix, ln_out)

    ln_bias_contrib = unembed_matrix @ ln_bias
    res_contrib = embed_to_contribs(res_embedding)
    head_contribs = embed_to_contribs(head_embeddings)

    summed_logits = ln_bias_contrib + res_contrib + head_contribs.sum(dim=(0, 1))
    assert t.allclose(summed_logits, gpt_out.all_logits)

    return res_contrib, head_contribs, ln_bias_contrib, summed_logits, gpt_out.all_logits