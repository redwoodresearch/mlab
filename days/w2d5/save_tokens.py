import json
import transformers


if __name__ == "__main__":
    seq_len = 1024
    tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
    with open("../../../lw_corpus.json") as f:
        data = json.load(f)
    text = ""
    for sequence in data:
        text += sequence['text'] + "<|endoftext|>"
    tokens = tokenizer.encode(text)
    # Note that this cuts off the end of the last element
    tokens = [tokens[i:i+seq_len] for i in range(0, len(tokens) // seq_len, seq_len)]
    assert all(len(i) == 1024 for i in tokens)
    with open("lw_corpus_tokens.json", "w") as f:
        json.dump(tokens, f)
        