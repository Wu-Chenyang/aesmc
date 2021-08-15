import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from torch import autograd

import time
import os
import fire
import random

import numpy as np
import matplotlib.pyplot as plt

# from datasets import BatteryDataset
from processed_datasets import BatteryDataset

# import multiprocessing

import aesmc.train as train
import aesmc
from models import SSM, Proposal
from aesmc.statistics import sample_from_prior


class LGSSMTester:
    def run(self,
            run_dir: str = './runs/',
            state_dim: int = 5,
            # action_dim: int = 3,
            obs_dim: int = 13,
            sequence_length: int = 50,
            batch_size: int = 32,
            device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
            seed: int = 100):
        # np.random.seed(seed)
        # random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)

        device = torch.device(device_name)
        model = SSM(state_dim, obs_dim, 5, 1, 5, 1)

        model.to(device)

        os.makedirs(run_dir, exist_ok=True)
        checkpoint_path = os.path.join(run_dir, 'checkpoint.pt')
        log_dir = None
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            log_dir = checkpoint['log_dir']

        model.double()
        model.eval()
        observations = sample_from_prior(model.prior, model.transition, model.observation, sequence_length, batch_size)[1]
        for batch in range(batch_size):
            plt.plot(range(len(observations)), [obs[batch, -1].item() for obs in observations])
        plt.gcf().savefig("runs/prior_sample.jpg")
        plt.close()

if __name__ == '__main__':
    fire.Fire(LGSSMTester)
