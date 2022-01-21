from comet_ml import Experiment

# This block is so that comet_ml is always imported first
if True:
    pass

import random
from typing import Any

import gin
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
import torch as t
from torch import nn, optim
from einops import rearrange


EXPERIMENT = Experiment(
    api_key="qjxcybqq2HsGHbEwATgNiqWgE",
    project_name="mlab_gpt2_periods_v0",
    workspace="ttwang",
    auto_metric_logging=False,
)


def count_periods(s: str) -> int:
    return s.count(".")


@gin.configurable
def train(
    num_steps: int,
    batch_size: int,
    gen_len: int,
    temp: float,
    lr: float,
    seed: int,
):
    t.manual_seed(seed)
    random.seed(seed)

    gpt2: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    prompt_tensor = t.tensor(
        [gpt2.config.eos_token_id],
        dtype=t.long,
        device=gpt2.device,
    )

    # TODO: Decay learning rate
    opt = optim.Adam(gpt2.parameters(), lr=lr)

    for step in range(num_steps):
        gen_tensor: t.Tensor = gpt2.generate(
            input_ids=prompt_tensor.reshape(1, -1),
            min_length=len(prompt_tensor) + gen_len,
            max_length=len(prompt_tensor) + gen_len,
            do_sample=True,
            temperature=temp,
            top_k=gpt2.config.vocab_size,
            top_p=1.0,
            num_return_sequences=batch_size,
            use_cache=True,
        )

        gen_texts: list[str] = tokenizer.batch_decode(gen_tensor[:, 1:])
        rewards = t.tensor(
            [count_periods(t) for t in gen_texts],
            dtype=t.long,
            device=gpt2.device,
        )

        loss = None

        opt.zero_grad()
        if True:
            loss.backward()
            nn.utils.clip_grad.clip_grad_norm_(gpt2.parameters())
        opt.step()

        EXPERIMENT.log_metric(name="loss", value=loss.item(), step=step)

    # t.save(gpt2.state_dict(), "models/gpt2_periods.pt")
    # EXPERIMENT.log_asset("models/gpt2_periods.pt")


def flatten_gin_config(
    d: dict[tuple[str, str], tuple[str, Any]],
) -> dict[str, Any]:
    ret_dict = {}
    for (_, sub_name), sub_dict in d.items():
        small_sub_name = sub_name.split(".")[-1]
        ret_dict.update({f"{small_sub_name}.{k}": v for k, v in sub_dict.items()})
    return ret_dict


if __name__ == "__main__":
    with gin.unlock_config():
        gin.parse_config_files_and_bindings(
            config_files=["atari.gin"],
            bindings=None,
        )
        EXPERIMENT.log_parameters(flatten_gin_config(gin.config._CONFIG))
        train()
