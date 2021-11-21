# Making Bert


Bert is a transformer trained on masked language modelling, which is where the model is given a text with some of the tokens replaced with `[MASK]`, and it has to predict the original token what was masked. 

```

Jane and her dog Ralph went do the dog park.

- tokenization ->

Jane [MASK] her dog Ralph went to the dog park.

- BERT ->

____  and   ___ ___ _____ ____ __ ___ ___ ____ _

```
You will make an nn.Module called Bert, which has the following type signature: `(LongTensor[batch_size, sequence_length], LongTensor[batch_size, sequence_length]) -> FloatTensor[batch_size, sequence_length, vocab_size]`

It will be composed of the following modules: BertEmbedding, BertAttention, and BertMLP.

## Embedding

`(LongTensor[batch_size, sequence_length], LongTensor[batch_size, sequence_length]) -> FloatTensor[batch_size, sequence_length, hidden_size]`
The embedding converts the descrete tokens ( token number 11, '.'), into embedding vectors, which are a combination of token embeddings, position embeddings (there's an embedding vector for every sequence position), and token type embeddings, which are used to differentiate multiple texts in some NLP tasks like entailment.
It has three Embedding modules, for tokens, positions, and token_types. It sums the embeddings from these, then layer norms them, and then applies dropout.

Theory: this can be identical to concatentation if the model learns to only use a set fraction of the dimensions for each of the three embeddings. Or, might be beneficial to make the embeddings overlap. Note: the embedding dimensions aren't a privilaged basis, so any subspacing would happen in an arbitrary projection, not a subset of neurons.



## Block
`FloatTensor[batch_size, sequence_length, embedding_size]->FloatTensor[batch_size, sequence_length, embedding_size]`

This performs the core computation Bert. It is a residual network with an Attention module and an MLP module. Multiple of these are stacked together.

## Attention
`FloatTensor[batch_size, sequence_length, embedding_size]->FloatTensor[batch_size, sequence_length, embedding_size]`
The Attention layer is the core of the Transformer architecture. It allows each token to integrate information from each other token in the network.
## MLP
`FloatTensor[batch_size, sequence_length, embedding_size]->FloatTensor[batch_size, sequence_length, embedding_size]`

## Unembedding
`FloatTensor[batch_size, sequence_length, embedding_size]->FloatTensor[batch_size, sequence_length, vocab_size]`
BERT is a bidirectional transoformer

Here's the rough outline:
```
Bert(
  (embedding): BertEmbedding(
    (token_embedding): Embedding(vocab_size, hidden_size)
    (position_embedding): Embedding(512, hidden_size)
    (token_type_embedding): Embedding(2, hidden_size)
    (layer_norm): LayerNorm(hidden_size)
    (dropout): Dropout()
  )
  (transformer): Sequential(
    (0): BertLayer(
      (attention): SelfAttentionLayer(
        (attention): PureSelfAttentionLayer(
          (project_query): Linear()
          (project_key): Linear()
          (project_value): Linear()
          (dropout): Dropout()
        )
        (mlp): Linear()
        (layer_norm): LayerNorm(hidden_size)
        (dropout): Dropout()
      )
      (residual): NormedResidualLayer(
        (mlp1): Linear(in_features=hidden_size, out_features=intermediate_size)
        (mlp2): Linear(in_features=intermediate_size, out_features=hidden_size)
        (layer_norm): LayerNorm(hidden_size)
        (dropout): Dropout()
      )
    )
    ... more layers
  )
  (lm_head): BertLMHead(
    (mlp): Linear()
    (unembedding): Linear(in_features=hidden_size, out_features=vocab_size)
    (layer_norm): LayerNorm((hidden_size,), elementwise_affine=True)
  )
)
```