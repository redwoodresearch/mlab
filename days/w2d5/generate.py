import torch
import transformers

# this is someone else's GPU so check to make sure they're not using it before
# running this file
DEVICE = "cuda:7"
MODEL_FILENAME = "/home/ubuntu/alwin_tom_mlab/model.pt"

tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")

model_config = transformers.GPT2Config()
model = transformers.GPT2LMHeadModel(model_config).to(DEVICE)
model.load_state_dict(torch.load(MODEL_FILENAME, map_location=DEVICE))

PROMPT = "This year I donated to"
input_ids = tokenizer.encode(PROMPT, return_tensors='pt').to(DEVICE)

# args copied from the last example of https://huggingface.co/blog/how-to-generate
sample_outputs = model.generate(
        input_ids,
        do_sample=True, 
        max_length=1000, 
        top_k=50, 
        top_p=0.95, 
        num_return_sequences=5
)

print("### Output:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
    sentence = tokenizer.decode(sample_output, skip_special_tokens=True)
    print(f"Sample {i}:\n{sentence}\n\n")
