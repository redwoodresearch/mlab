import torch as t
import torch.nn as nn
import torch.nn.functional as F
import itertools
import functools
import copy
from einops import rearrange, repeat


def init_env(size=3):
    return {
        "size": size,
        "board": [[0 for _ in range(size)] for _ in range(size)],
        "scalars": [0],  # player turn, 0 or 1
        "action_size": 1,
        "input_size": 3,
    }


def get_valid_actions(env):
    return [
        (x * env["size"] + y)
        for x, y in itertools.product(range(env["size"]), range(env["size"]))
        if env["board"][x][y] == 0
    ]


def print_env(env):
    string = "----" * env["size"] + "-\n"
    for row in range(env["size"]):
        for col in range(env["size"]):
            string += f"| {env['board'][row][col]} "
        string += "|\n"
    string += "----" * env["size"] + "-"
    print(string)
    return string


def get_winner(env):
    thing = 1


def step_env(env, action, player):
    i, j, c = (
        action // (env["size"] * env["action_size"]),
        (action // env["action_size"]) % env["size"],
        action % (env["size"] * env["size"]),
    )
    if action not in get_valid_actions(env):
        raise AssertionError("inavlid action", action, env)
    env = copy.deepcopy(env)
    env["board"][i][j] = player + 1  # this is tictactoe specific
    return env, get_winner(env)


# prob interface: env->Tensor[action_size]


def actsample_naive(x):
    return t.randint(0, x["action_size"] * x["size"] * x["size"])


class AlphaZeroModel(nn.Module):
    def __init__(self, board_size, input_channels, output_channels, input_scalars):
        super().__init__()
        self.board_size = board_size
        self.n_channels = 128
        self.n_layers = board_size
        self.input_scalars = input_scalars
        self.input_channels = input_channels + input_scalars
        self.output_channels = output_channels
        self.input_layer = nn.Conv2d(
            self.input_channels, self.n_channels, (3, 3), padding="same", bias=False
        )
        self.blocks = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        self.n_channels,
                        self.n_channels,
                        (3, 3),
                        padding="same",
                        bias=False,
                    ),
                    nn.ReLU(),
                )
                for _ in range(self.n_layers)
            ]
        )
        self.policy_layer = nn.Conv2d(
            self.n_channels, self.output_channels, (3, 3), padding="same"
        )
        self.value_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(self.n_channels, 1)
        )

    def forward(self, board, scalars):
        input = t.cat(
            [
                board,
                repeat(scalars, "b s -> b s h w", h=self.board_size, w=self.board_size),
            ],
            dim=1,
        )
        x = self.input_layer(input)
        x = self.blocks(x)
        policy = self.policy_layer(x)
        value = self.value_layer(x)
        return policy, value


def tmd(x, m):
    return x.to(next(m.parameters()).device)


def actsample_alphazero(model):
    def fun(envs):
        x = t.stack(
            [
                F.one_hot(t.tensor(env["board"]), num_classes=env["input_size"])
                for env in envs
            ]
        )
        scalars = t.stack([t.Tensor(env["scalars"]) for env in envs])
        policy, _value = model(tmd(x, model), tmd(scalars, model))
        policy_softmax = t.softmax(rearrange(policy, "b c h w -> b (h w c)"), dim=1)
        samples = t.distributions.Categorical(policy_softmax).sample()
        print("samples", samples)
        return samples.tolist()

    return fun


def mcts_beam_search_game(
    beam_width=32,
):
    rollouts = [{"actions": [], "score": 0, "env": init_env()}]


if __name__ == "__main__":
    env = init_env()
    print_env(env)
    model = AlphaZeroModel(
        board_size=3, input_channels=3, output_channels=1, input_scalars=1
    )
    initial_state = t.randn(2, 3, 3, 3)
    initial_scalars = t.randn(2, 1)
    policy, value = model(initial_state, initial_scalars)
    print("p", policy.shape, "v", value.shape)
    actions = actsample_alphazero(model)([env])
    print(
        "actprobs",
    )
    env2, winner = step_env(env, actions[0], 0)
    print("winner", winner)
    print_env(env2)
