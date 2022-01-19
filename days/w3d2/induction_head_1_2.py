import torch as t
import einops

layer = 1
head = 2


def get_attention(tokens: t.LongTensor) -> t.Tensor:
    """
    Returns A[k, q].

    Mixes three strategies:
    1. Inductively attend to previous parts of the sequence that look like
       our current context. Attention is stronger if more of the current
       context matches.
    2. Uniformly attend to previous tokens (a little).
    3. Attend to begin token (strongly).
    """
    seq_len = tokens.shape[0]
    neg1 = t.tensor([-1], dtype=t.long, device=tokens.device)
    neg2 = t.tensor([-2], dtype=t.long, device=tokens.device)

    tokens_rshift0 = tokens
    tokens_rshift1 = t.cat((neg1, tokens[:-1]))
    tokens_rshift2 = t.cat((neg2, neg2, tokens[:-2]))

    # induct_1[k, q] = (tokens[q] == tokens[k - 1])
    lhs_1 = einops.repeat(tokens_rshift0, "q -> k q", k=seq_len)
    rhs_1 = einops.repeat(tokens_rshift1, "k -> k q", q=seq_len)
    induct_1 = lhs_1 == rhs_1

    # induct_2[k, q] = (tokens[q - 1] == tokens[k - 2])
    lhs_2 = einops.repeat(tokens_rshift1, "q -> k q", k=seq_len)
    rhs_2 = einops.repeat(tokens_rshift2, "k -> k q", q=seq_len)
    induct_2 = lhs_2 == rhs_2

    A = 2.0 * induct_1 + 4.0 * (induct_1 & induct_2)  # induction
    A += 0.01  # uniform attention
    A[0, :] = 5.0  # attending to begin token
    A = t.triu(A)  # proper masking

    return A / A.sum(dim=0, keepdim=True)  # normalize and return


def attention_adjuster(
    layer_index: int,
    # shape: [batch, heads, seq_len (key), seq_len (query)]
    attention_prob: t.tensor,
    # shape: [batch, seq_len]
    input_ids: t.tensor,
):
    if layer_index == layer:
        for batch_idx in range(input_ids.shape[0]):
            attention_prob[batch_idx, head, :, :] = get_attention(input_ids[batch_idx])


model.attention_adjuster = attention_adjuster
