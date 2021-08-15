import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

from typing import Tuple

from building_blocks import CGM, LGM
import aesmc.state as aes

class SSM(nn.Module):
    def __init__(self, state_dim: int = 2, obs_dim: int = 1,
                trans_hidden_dim: int = 50, trans_hidden_layer: int = 1,
                obs_hidden_dim: int = 50, obs_hidden_layer: int = 1):
        super().__init__()
        self.prior_mean = nn.Parameter(torch.zeros(state_dim), requires_grad=True)
        self.prior_scale = nn.Parameter(torch.ones(state_dim), requires_grad=True)
        # self.trans_net = CGM(state_dim, state_dim, trans_hidden_dim, trans_hidden_layer)
        # self.obs_net = CGM(state_dim, obs_dim, obs_hidden_dim, obs_hidden_layer)
        self.trans_net = LGM(state_dim, state_dim)
        self.obs_net = LGM(state_dim, obs_dim)

    def prior(self) -> D.Distribution:
        return D.Independent(D.Normal(self.prior_mean, F.softplus(self.prior_scale)), 1)
    
    def transition(self, previous_latents, time = None, previous_observations = None) -> D.Distribution:
        return self.trans_net(previous_latents[-1])

    def observation(self, latents, time = None, previous_observations = None) -> D.Distribution:
        return self.obs_net(latents[-1])

class Proposal(nn.Module):
    def __init__(self, state_dim: int = 2, obs_dim: int = 1, hidden_dim: int = 50, num_hidden_layer=1):
        super().__init__()
        # self.prior_net = CGM(obs_dim, state_dim, hidden_dim, num_hidden_layer)
        # self.trans_net = CGM(state_dim+obs_dim, state_dim, hidden_dim, num_hidden_layer)
        self.prior_net = LGM(obs_dim, state_dim)
        self.trans_net = LGM(state_dim+obs_dim, state_dim)
    
    def forward(self, previous_latents=None, time=None, observations=None):
        if time == 0:
            return aes.set_batch_shape_mode(self.prior_net(observations[0]), aes.BatchShapeMode.BATCH_EXPANDED)
        else:
            num_particles = previous_latents[-1].shape[1]
            inputs = torch.cat((previous_latents[-1], observations[time].unsqueeze(1).expand(-1, num_particles, -1)), dim=-1)
            return aes.set_batch_shape_mode(self.trans_net(inputs), aes.BatchShapeMode.FULLY_EXPANDED)