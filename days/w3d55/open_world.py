# import comet_ml at the top of your file
from comet_ml import Experiment

# Create an experiment with your api key
experiment = Experiment(
    api_key="vABV7zo6pqS7lfzZBhyabU2Xe",
    project_name="playing_around",
    workspace="redwood",
)
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
import matplotlib.pyplot as plt
import crafter

device = "cpu"

world_size = 64
view = 8
img_size = 64
action_space_size = 12
input_scalar_size = 8
tile_pixels = img_size // view
env = crafter.Env(
    area=(world_size, world_size), view=(view, view), size=(img_size, img_size), seed=0
)
obs = env.reset()


def get_pos(env):
    return env._player.pos


class DomainSpecificModel(nn.Module):
    def __init__(
        self,
        tile_encoding_dim,
        tile_pixels,
        n_heads,
        input_scalar_size,
        action_space_size,
        world_size,
        view_size,
        num_layers,
    ):
        super().__init__()
        self.tile_encoding_dim = tile_encoding_dim
        self.tile_pixels = tile_pixels
        self.world_size = world_size
        self.view_size = view_size
        self.pos_encodings = nn.Embedding(world_size * world_size, tile_encoding_dim)
        self.scalar_input_proj = nn.Linear(input_scalar_size, tile_encoding_dim)

        self.tile_embedder = nn.Linear(tile_pixels * tile_pixels * 3, tile_encoding_dim)

        self.encoder_layers = nn.Sequential(
            *[
                nn.TransformerEncoderLayer(
                    tile_encoding_dim, n_heads, tile_encoding_dim * 4, activation=F.gelu
                )
                for _ in range(num_layers)
            ]
        )
        self.action_head = nn.Linear(tile_encoding_dim, action_space_size)
        self.transition_neck = nn.TransformerEncoderLayer(
            tile_encoding_dim, n_heads, tile_encoding_dim * 4, activation=F.gelu
        )
        self.value_head = nn.Linear(tile_encoding_dim, 1)
        self.reward_head = nn.Linear(tile_encoding_dim, 1)

    def forward(self, imgs):
        img_blocks = rearrange(
            imgs,
            "b (x e1) (y e2) c -> b (x y) (e1 e2 c)",
            e1=self.tile_pixels,
            e2=self.tile_pixels,
        )
        img_encodings = self.tile_embedder(img_blocks)
        print(img_blocks.shape)
        scalar_encodings = t.zeros(img_blocks.shape[0], 1, self.tile_encoding_dim)
        encodings = t.cat([scalar_encodings, img_encodings], dim=1)
        encodings = self.encoder_layers(encodings)

        action_logits = self.action_head(encodings[:, :0])
        value = self.value_head(encodings[:, 0])

        transition_encodings = self.transition_neck(encodings)
        next_img = rearrange(
            transition_encodings[:, 1:] @ self.tile_embedder.weight,
            "b (x y) (e1 e2 c) -> b (x e1) (y e2) c",
            x=self.view_size,
            e1=self.tile_pixels,
            e2=self.tile_pixels,
        )
        next_reward = self.reward_head(transition_encodings[:, 1])

        return action_logits, value, (next_img, next_reward)


model = DomainSpecificModel(
    32, tile_pixels, 2, input_scalar_size, action_space_size, world_size, view, 3
)
model.to(device)
optimizer = t.optim.Adam(model.parameters(), lr=1e-4)
print("model # params", sum([p.numel() for p in model.parameters()]))

print(get_pos(env))

random_action_fraction = 0.1
env_batch_size = 2


def reinforcement_learn():
    envs = [
        crafter.Env(
            area=(world_size, world_size),
            view=(view, view),
            size=(img_size, img_size),
            seed=0,
        )
        for _ in range(env_batch_size)
    ]

    obses = t.stack([t.Tensor(env.reset()) for env in envs])
    print(obses[0].shape)
    action = 1
    for episode_batch in range(10):
        actions, values, (next_imgs, next_rewards) = model(obses)
        print(actions.shape)
        act_sample = t.distributions.Categorical(logits=actions).sample()
        print("act shape", act_sample.shape)
        outies = [envs[i].step(act_sample[i]) for i in range(env_batch_size)]
        obses = [outie[0] for outie in outies]
        transition_loss = (next_imgs - obses) ** 2
        experiment.log_metric("transition_loss", transition_loss)
        transition_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    obs, reward, done, info = env.step(action)


reinforcement_learn()
