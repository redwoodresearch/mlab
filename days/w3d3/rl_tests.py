

#import solution.dqn_minimal as sol
import torch
import torch.nn as nn

from days.utils import tpeek


def allclose(my_out, their_out, name, tol=1e-5):
 
    if not torch.allclose(my_out, their_out, rtol=1e-4, atol=tol):
        errstring = f'error in {name}\n{tpeek("", my_out, ret=True)} \n!=\n{tpeek("", their_out, ret=True)}'
        raise AssertionError(errstring)
    else:
        tpeek(f"{name} MATCH!!!!!!!!\n", my_out)


def test_q_net(model_class):
    in_size = torch.randint(100, (1,))
    hidden_size = torch.randint(100, (1,))
    out_size = torch.randint(100, (1,))
    torch.manual_seed(1011)
    target_model = nn.Sequential(
        nn.Linear(in_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, out_size),
    )
    torch.manual_seed(1011)
    model = model_class(in_size, hidden_size, out_size)
    input = torch.rand((16, in_size))
    my_out = target_model(input).detach()
    their_out = model(input).detach()
    allclose(my_out, their_out, "test_q_net")

class q_mlp(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.model = nn.Sequential(
                nn.Linear(in_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, out_size),
            )
    
    def forward(self, input):
        return self.model(input)


def force_batched(x):
    if not isinstance(x, torch.Tensor):
        return torch.Tensor(x).unsqueeze(0)
    if len(x.shape) == 1:
        return x.unsqueeze(0)
    return x


def calc_loss(net, gamma, obs, act, reward, obs_next, done):
    all_qs = net(force_batched(obs)) #(b, num_actions)

    pred = torch.gather(all_qs, -1, force_batched(act).unsqueeze(-1)).squeeze(-1) # (b,)
    with torch.no_grad():
        is_non_done = torch.logical_not(force_batched(done)) # (b,)
        target = reward + is_non_done * gamma * torch.max(net(obs_next), dim=-1)
    return (target - pred) ** 2

if __name__ == "__main__":
    test_q_net(q_mlp)
    


