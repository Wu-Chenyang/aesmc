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
import pickle as pkl
from aesmc.statistics import sample_from_prior


class LGSSMTrainer:
    def run(self,
            run_dir: str = './runs/',
            model_lr: float = 1e-6,
            proposal_lr: float = 1e-5,
            state_dim: int = 5,
            # action_dim: int = 3,
            obs_dim: int = 13,
            iterations: int = 100000,
            epochs_per_iter = 4,
            save_interval: int = 10,
            test_interval: int = 10,
            num_particles: int = 1000,
            sequence_length = 50,
            batch_size: int = 32,
            device_name: str = "cuda" if torch.cuda.is_available() else "cpu",
            data_dir: str = "battery/processed_data",
            seed: int = 100):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # cell_list = [os.path.join(data_dir, item) for item in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, item))]
        if isinstance(sequence_length, int):
            sequence_length = [sequence_length,]
        elif isinstance(sequence_length, str):
            sequence_length = [int(seq_len) for seq_len in sequence_length.strip('[]').split(' ')]
        if isinstance(epochs_per_iter, int):
            epochs_per_iter = [epochs_per_iter,] * len(sequence_length)
        elif isinstance(epochs_per_iter, str):
            epochs_per_iter = [int(epoch) for epoch in epochs_per_iter.strip('[]').split(' ')]
        cell_lists = []
        for seq_len in sequence_length:
            cell_list = []
            for item in os.listdir(data_dir):
                if os.path.isfile(os.path.join(data_dir, item)):
                    cell = os.path.join(data_dir, item)
                    with open(cell, 'rb') as f:
                        data = pkl.load(f)
                        if len(data['state_information']) >= seq_len:
                            cell_list.append(cell)
            cell_lists.append(cell_list)

        device = torch.device(device_name)
        model = SSM(state_dim, obs_dim, 5, 1, 5, 1)
        proposal = Proposal(state_dim, obs_dim, 50, 1)

        proposal.to(device)
        model.to(device)

        model_optimizer = torch.optim.Adam([{'params': model.parameters(), 'lr': model_lr}])
        proposal_optimizer = torch.optim.Adam([{'params': proposal.parameters(), 'lr': proposal_lr}])

        os.makedirs(run_dir, exist_ok=True)
        checkpoint_path = os.path.join(run_dir, 'checkpoint.pt')
        log_dir = None
        step = 0
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            proposal.load_state_dict(checkpoint['proposal'])
            model.load_state_dict(checkpoint['model'])
            # model_optimizer.load_state_dict(checkpoint['model_optimizer'])
            # proposal_optimizer.load_state_dict(checkpoint['proposal_optimizer'])
            step = checkpoint['step']
            log_dir = checkpoint['log_dir']

        summary_writer = SummaryWriter(log_dir)

        recorder = TrainingStats(num_particles, model, proposal, summary_writer, checkpoint_path, model_optimizer, proposal_optimizer, step, save_interval, test_interval)

        for i in range(iterations):
            for cell_list, seq_len, epoch_per_iter in zip(cell_lists, sequence_length, epochs_per_iter):
                dataloader =  DataLoader(BatteryDataset(cell_list, seq_len), batch_size=batch_size, shuffle=True, num_workers=4)
                train.train(dataloader=dataloader,
                            num_particles=num_particles,
                            algorithm="aesmc",
                            initial=model.prior,
                            transition=model.transition,
                            emission=model.observation,
                            proposal=proposal,
                            num_epochs=epoch_per_iter,
                            optimizer=model_optimizer,
                            num_iterations_per_epoch=10000,
                            callback=recorder,
                            device=device)
                train.train(dataloader=dataloader,
                            num_particles=10,
                            algorithm="iwae",
                            initial=model.prior,
                            transition=model.transition,
                            emission=model.observation,
                            proposal=proposal,
                            num_epochs=epoch_per_iter,
                            optimizer=proposal_optimizer,
                            num_iterations_per_epoch=10000,
                            callback=recorder,
                            device=device)

class TrainingStats(object):
    def __init__(self, test_inference_num_particles, model, proposal,
                summary_writer, checkpoint_path, model_optimizer, proposal_optimizer, step=0,
                 save_interval=100, test_interval=100, sample_batch_size=3, sample_sequence_length=2000):
        self.save_interval = save_interval
        self.test_interval = test_interval
        self.test_inference_num_particles = test_inference_num_particles
        self.step = step
        self.model = model
        self.proposal = proposal
        self.summary_writer = summary_writer
        self.checkpoint_path = checkpoint_path
        self.model_optimizer = model_optimizer
        self.proposal_optimizer = proposal_optimizer
        self.sample_batch_size = sample_batch_size
        self.sample_sequence_length = sample_sequence_length

    def __call__(self, epoch_idx, epoch_iteration_idx, loss, initial,
                 transition, emission, proposal):
        self.step += 1
        if self.step % self.save_interval == 0:
            torch.save(
                dict(proposal=self.proposal.state_dict(),
                    model=self.model.state_dict(),
                    step=self.step,
                    model_optimizer=self.model_optimizer.state_dict(),
                    proposal_optimizer=self.proposal_optimizer.state_dict(),
                    log_dir=self.summary_writer.log_dir), self.checkpoint_path)
        self.summary_writer.add_scalar('loss/train', loss, self.step)
        model_grad = global_grad_norm(self.model.parameters())
        proposal_grad = global_grad_norm(self.proposal.parameters())
        self.summary_writer.add_scalar('model_gradient/train', model_grad, self.step)
        self.summary_writer.add_scalar('proposal_gradient/train', proposal_grad, self.step)
        self.model.eval()
        self.model.double()
        try:
            if self.step % self.test_interval == 0:
                observations = sample_from_prior(self.model.prior, self.model.transition, self.model.observation, self.sample_sequence_length, self.sample_batch_size)[1]
                for batch in range(self.sample_batch_size):
                    plt.plot(range(self.sample_sequence_length), [obs[batch, -1].item() * 3.996 + 47.025 for obs in observations])
                self.summary_writer.add_figure('filtering/test', plt.gcf(), self.step)
                plt.close()
        except:
            pass
        self.model.train()
        self.model.float()

        print('Step {} : Loss = {} Model Gradient = {} Proposal Gradient = {}'.format(self.step, loss, model_grad, proposal_grad))
    
def global_grad_norm(params):
    grad_norm = 0.0
    for param in params:
        if param.grad == None:
            continue
        grad_norm = max(grad_norm, torch.max(torch.abs(param.grad.data)).item())
    return grad_norm

if __name__ == '__main__':
    fire.Fire(LGSSMTrainer)
