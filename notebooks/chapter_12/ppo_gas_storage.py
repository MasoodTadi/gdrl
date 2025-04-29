import warnings ; warnings.filterwarnings('ignore')
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import threading
from torch.distributions import Normal

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import linprog
from IPython.display import display
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from itertools import cycle, count
from textwrap import wrap

#import pybullet_envs
import pybullet
import matplotlib
import subprocess
import os.path
import tempfile
import random
import base64
import pprint
import glob
import time
import json
import sys
import gymnasium as gym
import io
import os
import gc

from gym import wrappers
from skimage.transform import resize
from skimage.color import rgb2gray
from subprocess import check_output
from IPython.display import display, HTML

import cvxpy as cp

import datetime

LEAVE_PRINT_EVERY_N_SECS = 300
ERASE_LINE = '\x1b[2K'
EPS = 1e-6
BEEP = lambda: os.system("printf '\a'")
RESULTS_DIR = os.path.join('..', 'results')
SEEDS = (12, 34, 56, 78, 90)

plt.style.use('fivethirtyeight')
params = {
    'figure.figsize': (15, 8),
    'font.size': 24,
    'legend.fontsize': 20,
    'axes.titlesize': 28,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20
}
pylab.rcParams.update(params)
np.set_printoptions(suppress=True)

from ttf_gas_env import TTFGasStorageEnv

def get_make_env_fn(params):
    def make_env_fn(seed=None, render=False):
        env = TTFGasStorageEnv(params)
        if seed is not None:
            env.seed(seed)
        return env
    return make_env_fn

class MultiprocessEnv(object):
    def __init__(self, make_env_fn, make_env_kargs, seed, n_workers):
        self.make_env_fn = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.n_workers = n_workers
        self.pipes = [mp.Pipe() for rank in range(self.n_workers)]
        self.workers = [mp.Process(target=self.work, args=(rank, self.pipes[rank][1])) for rank in range(self.n_workers)]
        [w.start() for w in self.workers]
        self.dones = {rank:False for rank in range(self.n_workers)}

    # def reset(self, ranks=None, **kwargs):
    #     if not (ranks is None):
    #         [self.send_msg(('reset', {}), rank) for rank in ranks]            
    #         return np.stack([parent_end.recv() for rank, (parent_end, _) in enumerate(self.pipes) if rank in ranks])

    #     self.broadcast_msg(('reset', kwargs))
    #     return np.stack([parent_end.recv() for parent_end, _ in self.pipes])
    def reset(self, ranks=None, **kwargs):
        if ranks is not None:
            [self.send_msg(('reset', kwargs), rank) for rank in ranks]
            obs = [parent_end.recv()[0] for rank, (parent_end, _) in enumerate(self.pipes) if rank in ranks]
            return np.stack(obs)

        self.broadcast_msg(('reset', kwargs))
        obs = [parent_end.recv()[0] for parent_end, _ in self.pipes]  # ← grab just obs
        return np.stack(obs)

    # def step(self, actions):
    #     assert len(actions) == self.n_workers
    #     [self.send_msg(('step', {'action':actions[rank]}), rank) for rank in range(self.n_workers)]
    #     results = []
    #     for rank in range(self.n_workers):
    #         parent_end, _ = self.pipes[rank]
    #         # o, r, d, i = parent_end.recv()
    #         o, r, d, truncated, i = parent_end.recv()
    #         results.append((o,
    #                         float(r),
    #                         #float(d),
    #                         float(d or truncated),
    #                         i))
    #     return [np.stack(block).squeeze() for block in np.array(results).T]
    def step(self, actions):
        assert len(actions) == self.n_workers
        [self.send_msg(('step', {'action': actions[rank]}), rank) for rank in range(self.n_workers)]
    
        # obs_list, reward_list, done_list, info_list = [], [], [], []
        obs_list, reward_list, terminated_list, truncated_list, info_list = [], [], [], [], []
        for rank in range(self.n_workers):
            parent_end, _ = self.pipes[rank]
            obs, reward, terminated, truncated, info = parent_end.recv()
            # done = terminated or truncated
            obs_list.append(obs)
            reward_list.append(reward)
            # done_list.append(done)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            info_list.append(info)
    
        obs_array = np.stack(obs_list)
        reward_array = np.array(reward_list, dtype=np.float32)
        # done_array = np.array(done_list, dtype=np.float32)
        terminated_array = np.array(terminated_list, dtype=bool)
        truncated_array = np.array(truncated_list, dtype=bool)
        # return obs_array, reward_array, done_array, info_list
        return obs_array, reward_array, terminated_array, truncated_array, info_list

    def close(self, **kwargs):
        self.broadcast_msg(('close', kwargs))
        [w.join() for w in self.workers]
    
    def work(self, rank, worker_end):
        env = self.make_env_fn(**self.make_env_kargs, seed=self.seed+rank)
        while True:
            cmd, kwargs = worker_end.recv()
            if cmd == 'reset':
                worker_end.send(env.reset(**kwargs))
            elif cmd == 'step':
                # worker_end.send(env.step(**kwargs))
                worker_end.send(env.step(kwargs['action']))
            # elif cmd == '_past_limit':
            #     worker_end.send(env._elapsed_steps >= env._max_episode_steps)
            else:
                # including close command 
                env.close(**kwargs) ; del env ; worker_end.close()
                break

    def send_msg(self, msg, rank):
        parent_end, _ = self.pipes[rank]
        parent_end.send(msg)

    def broadcast_msg(self, msg):    
        [parent_end.send(msg) for parent_end, _ in self.pipes]

class EpisodeBuffer():
    def __init__(self,
                 state_dim,
                 action_dim, # <<< ADD THIS
                 gamma,
                 tau,
                 n_workers,
                 max_episodes,
                 max_episode_steps):
        
        assert max_episodes >= n_workers

        self.state_dim = state_dim
        self.action_dim = action_dim # <<< SAVE IT
        self.gamma = gamma
        self.tau = tau
        self.n_workers = n_workers
        self.max_episodes = max_episodes
        self.max_episode_steps = max_episode_steps

        # self._truncated_fn = np.vectorize(lambda x: 'TimeLimit.truncated' in x and x['TimeLimit.truncated'])
        self.discounts = np.logspace(
            0, max_episode_steps+1, num=max_episode_steps+1, base=gamma, endpoint=False, dtype=np.float64)
        self.tau_discounts = np.logspace(
            0, max_episode_steps+1, num=max_episode_steps+1, base=gamma*tau, endpoint=False, dtype=np.float64)

        device = 'cpu'
        if torch.cuda.is_available():
            device = 'cuda:0'
        self.device = torch.device(device)

        self.clear()

    def clear(self):
        self.states_mem = np.empty(shape=np.concatenate(((self.max_episodes, self.max_episode_steps), self.state_dim)), dtype=np.float64)
        self.states_mem[:] = np.nan

        # self.actions_mem = np.empty(shape=(self.max_episodes, self.max_episode_steps), dtype=np.uint8)
        self.actions_mem = np.empty(shape=np.concatenate(((self.max_episodes, self.max_episode_steps), self.action_dim)), dtype=np.uint8)
        # self.actions_mem[:] = np.nan

        self.returns_mem = np.empty(shape=(self.max_episodes,self.max_episode_steps), dtype=np.float32)
        self.returns_mem[:] = np.nan

        self.gaes_mem = np.empty(shape=(self.max_episodes, self.max_episode_steps), dtype=np.float32)
        self.gaes_mem[:] = np.nan

        self.logpas_mem = np.empty(shape=(self.max_episodes, self.max_episode_steps), dtype=np.float32)
        self.logpas_mem[:] = np.nan

        self.episode_steps = np.zeros(shape=(self.max_episodes), dtype=np.uint16)
        self.episode_reward = np.zeros(shape=(self.max_episodes), dtype=np.float32)
        self.episode_exploration = np.zeros(shape=(self.max_episodes), dtype=np.float32)
        self.episode_seconds = np.zeros(shape=(self.max_episodes), dtype=np.float64)

        self.current_ep_idxs = np.arange(self.n_workers, dtype=np.uint16)
        gc.collect()


    def fill(self, envs, policy_model, value_model):
        states = envs.reset()

        worker_rewards = np.zeros(shape=(self.n_workers, self.max_episode_steps), dtype=np.float32)
        worker_exploratory = np.zeros(shape=(self.n_workers, self.max_episode_steps), dtype=np.bool_)
        worker_steps = np.zeros(shape=(self.n_workers), dtype=np.uint16)
        worker_seconds = np.array([time.time(),] * self.n_workers, dtype=np.float64)

        buffer_full = False
        while not buffer_full and len(self.episode_steps[self.episode_steps > 0]) < self.max_episodes/2:
            with torch.no_grad():
                actions, logpas, are_exploratory = policy_model.np_pass(states)
                values = value_model(states)

            # next_states, rewards, terminals, infos = envs.step(actions)
            next_states, rewards, terminals, truncateds, infos = envs.step(actions)
            self.states_mem[self.current_ep_idxs, worker_steps] = states
            self.actions_mem[self.current_ep_idxs, worker_steps] = actions
            self.logpas_mem[self.current_ep_idxs, worker_steps] = logpas

            worker_exploratory[np.arange(self.n_workers), worker_steps] = are_exploratory
            worker_rewards[np.arange(self.n_workers), worker_steps] = rewards

            for w_idx in range(self.n_workers):
                if worker_steps[w_idx] + 1 == self.max_episode_steps:
                    terminals[w_idx] = 1
                    # infos[w_idx]['TimeLimit.truncated'] = True
                    truncateds[w_idx] = True  # <- directly mark as truncated

            #Adding these lines
            next_values = np.zeros(shape=(self.n_workers,), dtype=np.float32)
            idx_truncated = np.flatnonzero(truncateds)
            if idx_truncated.size > 0:
                with torch.no_grad():
                    next_values[idx_truncated] = value_model(next_states[idx_truncated]).cpu().numpy()
                    
            # if terminals.sum():
            #     idx_terminals = np.flatnonzero(terminals)
            #     next_values = np.zeros(shape=(self.n_workers))
            #     truncated = self._truncated_fn(infos)
            #     if truncated.sum():
            #         idx_truncated = np.flatnonzero(truncated)
            #         with torch.no_grad():
            #             next_values[idx_truncated] = value_model(
            #                 next_states[idx_truncated]).cpu().numpy()

            states = next_states
            worker_steps += 1

            # if terminals.sum():
            #     new_states = envs.reset(ranks=idx_terminals)
            #     states[idx_terminals] = new_states

            #     for w_idx in range(self.n_workers):
            #         if w_idx not in idx_terminals:
            #             continue

            #         e_idx = self.current_ep_idxs[w_idx]
            #         T = worker_steps[w_idx]
            #         self.episode_steps[e_idx] = T
            #         self.episode_reward[e_idx] = worker_rewards[w_idx, :T].sum()
            #         self.episode_exploration[e_idx] = worker_exploratory[w_idx, :T].mean()
            #         self.episode_seconds[e_idx] = time.time() - worker_seconds[w_idx]

            #         ep_rewards = np.concatenate(
            #             (worker_rewards[w_idx, :T], [next_values[w_idx]]))
            #         ep_discounts = self.discounts[:T+1]
            #         ep_returns = np.array(
            #             [np.sum(ep_discounts[:T+1-t] * ep_rewards[t:]) for t in range(T)])
            #         self.returns_mem[e_idx, :T] = ep_returns

            #         ep_states = self.states_mem[e_idx, :T]
            #         with torch.no_grad():
            #             ep_values = torch.cat((value_model(ep_states),
            #                                    torch.tensor([next_values[w_idx]],
            #                                                 device=value_model.device,
            #                                                 dtype=torch.float32)))
            #         np_ep_values = ep_values.view(-1).cpu().numpy()
            #         ep_tau_discounts = self.tau_discounts[:T]
            #         deltas = ep_rewards[:-1] + self.gamma * np_ep_values[1:] - np_ep_values[:-1]
            #         gaes = np.array(
            #             [np.sum(self.tau_discounts[:T-t] * deltas[t:]) for t in range(T)])
            #         self.gaes_mem[e_idx, :T] = gaes

            #         worker_exploratory[w_idx, :] = 0
            #         worker_rewards[w_idx, :] = 0
            #         worker_steps[w_idx] = 0
            #         worker_seconds[w_idx] = time.time()

            #         new_ep_id = max(self.current_ep_idxs) + 1
            #         if new_ep_id >= self.max_episodes:
            #             buffer_full = True
            #             break

            #         self.current_ep_idxs[w_idx] = new_ep_id
            if terminals.sum():
                idx_terminals = np.flatnonzero(terminals)
    
                new_states = envs.reset(ranks=idx_terminals)
                states[idx_terminals] = new_states
    
                for w_idx in idx_terminals:
                    e_idx = self.current_ep_idxs[w_idx]
                    T = worker_steps[w_idx]
    
                    self.episode_steps[e_idx] = T
                    self.episode_reward[e_idx] = worker_rewards[w_idx, :T].sum()
                    self.episode_exploration[e_idx] = worker_exploratory[w_idx, :T].mean()
                    self.episode_seconds[e_idx] = time.time() - worker_seconds[w_idx]
    
                    ep_rewards = np.concatenate((worker_rewards[w_idx, :T], [next_values[w_idx]]))
                    ep_returns = np.array(
                        [np.sum(self.discounts[:T+1-t] * ep_rewards[t:]) for t in range(T)]
                    )
                    self.returns_mem[e_idx, :T] = ep_returns
    
                    ep_states = self.states_mem[e_idx, :T]
                    with torch.no_grad():
                        ep_values = torch.cat((
                            value_model(ep_states),
                            torch.tensor([next_values[w_idx]], device=value_model.device, dtype=torch.float32)
                        ))
                    np_ep_values = ep_values.view(-1).cpu().numpy()
                    deltas = ep_rewards[:-1] + self.gamma * np_ep_values[1:] - np_ep_values[:-1]
                    gaes = np.array(
                        [np.sum(self.tau_discounts[:T-t] * deltas[t:]) for t in range(T)]
                    )
                    self.gaes_mem[e_idx, :T] = gaes
    
                    # Reset worker
                    worker_rewards[w_idx, :] = 0
                    worker_exploratory[w_idx, :] = 0
                    worker_steps[w_idx] = 0
                    worker_seconds[w_idx] = time.time()
    
                    # Assign new episode id
                    new_ep_id = max(self.current_ep_idxs) + 1
                    if new_ep_id >= self.max_episodes:
                        buffer_full = True
                        break
    
                    self.current_ep_idxs[w_idx] = new_ep_id

        ep_idxs = self.episode_steps > 0
        ep_t = self.episode_steps[ep_idxs]

        self.states_mem = [row[:ep_t[i]] for i, row in enumerate(self.states_mem[ep_idxs])]
        self.states_mem = np.concatenate(self.states_mem)
        self.actions_mem = [row[:ep_t[i]] for i, row in enumerate(self.actions_mem[ep_idxs])]
        self.actions_mem = np.concatenate(self.actions_mem)
        self.returns_mem = [row[:ep_t[i]] for i, row in enumerate(self.returns_mem[ep_idxs])]
        self.returns_mem = torch.tensor(np.concatenate(self.returns_mem), 
                                        device=value_model.device)
        self.gaes_mem = [row[:ep_t[i]] for i, row in enumerate(self.gaes_mem[ep_idxs])]
        self.gaes_mem = torch.tensor(np.concatenate(self.gaes_mem), 
                                     device=value_model.device)
        self.logpas_mem = [row[:ep_t[i]] for i, row in enumerate(self.logpas_mem[ep_idxs])]
        self.logpas_mem = torch.tensor(np.concatenate(self.logpas_mem), 
                                       device=value_model.device)

        ep_r = self.episode_reward[ep_idxs]
        ep_x = self.episode_exploration[ep_idxs]
        ep_s = self.episode_seconds[ep_idxs]
        return ep_t, ep_r, ep_x, ep_s

    def get_stacks(self):
        return (self.states_mem, self.actions_mem, 
                self.returns_mem, self.gaes_mem, self.logpas_mem)

    def __len__(self):
        return self.episode_steps[self.episode_steps > 0].sum()

class FCCA(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,   # output_dim = tuple like (n_actions_per_dim,)
                 hidden_dims=(32, 32),
                 activation_fc=F.relu):
        super(FCCA, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_dim[0], hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        
        # Important: output layer should output sum of all action logits
        self.output_layer = nn.Linear(hidden_dims[-1], sum(output_dim))

        self.output_dim = tuple(output_dim)  # (n1, n2, ..., n12)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)

    def _format(self, states):
        x = states
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            if len(x.size()) == 1:
                x = x.unsqueeze(0)
        return x

    def forward(self, states):
        x = self._format(states)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        return self.output_layer(x)  # Output shape: (batch_size, sum of action logits)

    def split_logits(self, logits):
        # Split logits into list [logits_dim1, logits_dim2, ..., logits_dim12]
        splits = torch.split(logits, self.output_dim, dim=-1)
        return splits

    def np_pass(self, states):
        logits = self.forward(states)
        split_logits = self.split_logits(logits)

        actions, logpas, is_exploratory = [], [], []
        for logit in split_logits:
            dist = torch.distributions.Categorical(logits=logit)
            action = dist.sample()
            logpa = dist.log_prob(action)
            exploratory = (action != logit.argmax(dim=-1))

            actions.append(action)
            logpas.append(logpa)
            is_exploratory.append(exploratory)

        actions = torch.stack(actions, dim=-1)          # shape (batch_size, n_actions)
        logpas = torch.stack(logpas, dim=-1).sum(dim=-1) # sum log probs for joint policy
        is_exploratory = torch.stack(is_exploratory, dim=-1).any(dim=-1) # if any action exploratory

        return actions.cpu().numpy(), logpas.cpu().numpy(), is_exploratory.cpu().numpy()

    def select_action(self, states):
        logits = self.forward(states)
        split_logits = self.split_logits(logits)

        actions = []
        for logit in split_logits:
            dist = torch.distributions.Categorical(logits=logit)
            action = dist.sample()
            actions.append(action)

        actions = torch.stack(actions, dim=-1)
        return actions.squeeze(0).detach().cpu().numpy()  # if input single state, return (n_actions,)

    def get_predictions(self, states, actions):
        states = self._format(states)
        logits = self.forward(states)
        split_logits = self.split_logits(logits)

        # If actions input is numpy, convert
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, device=self.device, dtype=torch.long)

        logpas = []
        for idx, logit in enumerate(split_logits):
            dist = torch.distributions.Categorical(logits=logit)
            logpas.append(dist.log_prob(actions[:, idx]))

        logpas = torch.stack(logpas, dim=-1).sum(dim=-1)
        entropies = torch.stack([torch.distributions.Categorical(logits=logit).entropy() for logit in split_logits], dim=-1).mean(dim=-1)

        return logpas, entropies

    def select_greedy_action(self, states):
        logits = self.forward(states)
        split_logits = self.split_logits(logits)

        greedy_actions = []
        for logit in split_logits:
            greedy_action = logit.argmax(dim=-1)
            greedy_actions.append(greedy_action)

        greedy_actions = torch.stack(greedy_actions, dim=-1)
        return greedy_actions.squeeze(0).detach().cpu().numpy()

class FCV(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dims=(32,32), 
                 activation_fc=F.relu):
        super(FCV, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_dim[0], hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)
        
    def _format(self, states):
        x = states
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            if len(x.size()) == 1:
                x = x.unsqueeze(0)
        return x

    def forward(self, states):
        x = self._format(states)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        return self.output_layer(x).squeeze()

class PPO():
    def __init__(self, 
                 policy_model_fn, 
                 policy_model_max_grad_norm,
                 policy_optimizer_fn,
                 policy_optimizer_lr,
                 policy_optimization_epochs,
                 policy_sample_ratio,
                 policy_clip_range,
                 policy_stopping_kl,
                 value_model_fn, 
                 value_model_max_grad_norm,
                 value_optimizer_fn,
                 value_optimizer_lr,
                 value_optimization_epochs,
                 value_sample_ratio,
                 value_clip_range,
                 value_stopping_mse,
                 episode_buffer_fn,
                 max_buffer_episodes,
                 max_buffer_episode_steps,
                 entropy_loss_weight,
                 tau,
                 n_workers):
        assert n_workers > 1
        assert max_buffer_episodes >= n_workers

        self.policy_model_fn = policy_model_fn
        self.policy_model_max_grad_norm = policy_model_max_grad_norm
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr
        self.policy_optimization_epochs = policy_optimization_epochs
        self.policy_sample_ratio = policy_sample_ratio
        self.policy_clip_range = policy_clip_range
        self.policy_stopping_kl = policy_stopping_kl

        self.value_model_fn = value_model_fn
        self.value_model_max_grad_norm = value_model_max_grad_norm
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.value_optimization_epochs = value_optimization_epochs
        self.value_sample_ratio = value_sample_ratio
        self.value_clip_range = value_clip_range
        self.value_stopping_mse = value_stopping_mse

        self.episode_buffer_fn = episode_buffer_fn
        self.max_buffer_episodes = max_buffer_episodes
        self.max_buffer_episode_steps = max_buffer_episode_steps

        self.entropy_loss_weight = entropy_loss_weight
        self.tau = tau
        self.n_workers = n_workers

    def optimize_model(self):
        states, actions, returns, gaes, logpas = self.episode_buffer.get_stacks()
        values = self.value_model(states).detach()
        gaes = (gaes - gaes.mean()) / (gaes.std() + EPS)
        n_samples = len(actions)
        
        for _ in range(self.policy_optimization_epochs):
            batch_size = int(self.policy_sample_ratio * n_samples)
            batch_idxs = np.random.choice(n_samples, batch_size, replace=False)
            states_batch = states[batch_idxs]
            actions_batch = actions[batch_idxs]
            gaes_batch = gaes[batch_idxs]
            logpas_batch = logpas[batch_idxs]

            logpas_pred, entropies_pred = self.policy_model.get_predictions(states_batch,
                                                                            actions_batch)

            ratios = (logpas_pred - logpas_batch).exp()
            pi_obj = gaes_batch * ratios
            pi_obj_clipped = gaes_batch * ratios.clamp(1.0 - self.policy_clip_range,
                                                       1.0 + self.policy_clip_range)
            policy_loss = -torch.min(pi_obj, pi_obj_clipped).mean()
            entropy_loss = -entropies_pred.mean() * self.entropy_loss_weight

            self.policy_optimizer.zero_grad()
            (policy_loss + entropy_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 
                                           self.policy_model_max_grad_norm)
            self.policy_optimizer.step()
            
            with torch.no_grad():
                logpas_pred_all, _ = self.policy_model.get_predictions(states, actions)
                kl = (logpas - logpas_pred_all).mean()
                if kl.item() > self.policy_stopping_kl:
                    break

        for _ in range(self.value_optimization_epochs):
            batch_size = int(self.value_sample_ratio * n_samples)
            batch_idxs = np.random.choice(n_samples, batch_size, replace=False)
            states_batch = states[batch_idxs]
            returns_batch = returns[batch_idxs]
            values_batch = values[batch_idxs]

            values_pred = self.value_model(states_batch)
            values_pred_clipped = values_batch + (values_pred - values_batch).clamp(-self.value_clip_range, 
                                                                                    self.value_clip_range)
            v_loss = (returns_batch - values_pred).pow(2)
            v_loss_clipped = (returns_batch - values_pred_clipped).pow(2)
            value_loss = torch.max(v_loss, v_loss_clipped).mul(0.5).mean()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 
                                           self.value_model_max_grad_norm)
            self.value_optimizer.step()

            with torch.no_grad():
                values_pred_all = self.value_model(states)
                mse = (values - values_pred_all).pow(2).mul(0.5).mean()
                if mse.item() > self.value_stopping_mse:
                    break

    def train(self, make_envs_fn, make_env_fn, make_env_kargs, seed, gamma, 
              max_minutes, max_episodes, goal_mean_100_reward):
        training_start, last_debug_time = time.time(), float('-inf')
        
        # Safe and persistent checkpoint directory in home
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = os.path.expanduser(f"~/ddpg_checkpoints/run_{timestamp}")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
        print(f"Running on: {os.uname().nodename}")
        print(f"[INFO] Checkpoints will be saved to: {self.checkpoint_dir}")
        # self.checkpoint_dir = tempfile.mkdtemp()
        self.make_envs_fn = make_envs_fn
        self.make_env_fn = make_env_fn
        self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.gamma = gamma
        
        env = self.make_env_fn(**self.make_env_kargs, seed=self.seed)
        envs = self.make_envs_fn(make_env_fn, make_env_kargs, self.seed, self.n_workers)
        torch.manual_seed(self.seed) ; np.random.seed(self.seed) ; random.seed(self.seed)
    
        # nS, nA = env.observation_space.shape, env.action_space.n
        nS, nA = env.observation_space.shape, env.action_space.nvec
        self.episode_timestep, self.episode_reward = [], []
        self.episode_seconds, self.episode_exploration = [], []
        self.evaluation_scores = []

        self.policy_model = self.policy_model_fn(nS, nA)
        self.policy_optimizer = self.policy_optimizer_fn(self.policy_model, self.policy_optimizer_lr)

        self.value_model = self.value_model_fn(nS)
        self.value_optimizer = self.value_optimizer_fn(self.value_model, self.value_optimizer_lr)

        # self.episode_buffer = self.episode_buffer_fn(nS, self.gamma, self.tau,
        #                                              self.n_workers, 
        #                                              self.max_buffer_episodes,
        #                                              self.max_buffer_episode_steps)
        self.episode_buffer = self.episode_buffer_fn(nS, nA.shape, self.gamma, self.tau,
                                                     self.n_workers, 
                                                     self.max_buffer_episodes,
                                                     self.max_buffer_episode_steps)

        result = np.empty((max_episodes, 5))
        result[:] = np.nan
        training_time = 0
        episode = 0

        # collect n_steps rollout
        while True:
            episode_timestep, episode_reward, episode_exploration, \
            episode_seconds = self.episode_buffer.fill(envs, self.policy_model, self.value_model)
            
            n_ep_batch = len(episode_timestep)
            self.episode_timestep.extend(episode_timestep)
            self.episode_reward.extend(episode_reward)
            self.episode_exploration.extend(episode_exploration)
            self.episode_seconds.extend(episode_seconds)
            self.optimize_model()
            self.episode_buffer.clear()

            # stats
            evaluation_score, _ = self.evaluate(self.policy_model, env)
            self.evaluation_scores.extend([evaluation_score,] * n_ep_batch)
            for e in range(episode, episode + n_ep_batch):
                self.save_checkpoint(e, self.policy_model)
            training_time += episode_seconds.sum()

            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])
            mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
            std_100_eval_score = np.std(self.evaluation_scores[-100:])
            mean_100_exp_rat = np.mean(self.episode_exploration[-100:])
            std_100_exp_rat = np.std(self.episode_exploration[-100:])
            
            total_step = int(np.sum(self.episode_timestep))
            wallclock_elapsed = time.time() - training_start
            result[episode:episode+n_ep_batch] = total_step, mean_100_reward, \
                mean_100_eval_score, training_time, wallclock_elapsed

            episode += n_ep_batch

            # debug stuff
            reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS
            reached_max_minutes = wallclock_elapsed >= max_minutes * 60            
            reached_max_episodes = episode + self.max_buffer_episodes >= max_episodes
            reached_goal_mean_reward = mean_100_eval_score >= goal_mean_100_reward
            training_is_over = reached_max_minutes or \
                               reached_max_episodes or \
                               reached_goal_mean_reward
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
            debug_message = 'el {}, ep {:04}, ts {:07}, '
            debug_message += 'ar 10 {:05.1f}\u00B1{:05.1f}, '
            debug_message += '100 {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'ex 100 {:02.1f}\u00B1{:02.1f}, '
            debug_message += 'ev {:05.1f}\u00B1{:05.1f}'
            debug_message = debug_message.format(
                elapsed_str, episode-1, total_step, mean_10_reward, std_10_reward, 
                mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
                mean_100_eval_score, std_100_eval_score)
            print(debug_message, end='\r', flush=True)
            if reached_debug_time or training_is_over:
                print(ERASE_LINE + debug_message, flush=True)
                last_debug_time = time.time()
            if training_is_over:
                if reached_max_minutes: print(u'--> reached_max_minutes \u2715')
                if reached_max_episodes: print(u'--> reached_max_episodes \u2715')
                if reached_goal_mean_reward: print(u'--> reached_goal_mean_reward \u2713')
                break

        final_eval_score, score_std = self.evaluate(self.policy_model, env, n_episodes=100)
        wallclock_time = time.time() - training_start
        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(
                  final_eval_score, score_std, training_time, wallclock_time))
        env.close() ; del env
        envs.close() ; del envs
        self.get_cleaned_checkpoints()
        return result, final_eval_score, training_time, wallclock_time

    def evaluate(self, eval_model, eval_env, n_episodes=1, greedy=True):
        rs = []
        for _ in range(n_episodes):
            # s, d = eval_env.reset(), False
            s, _ = eval_env.reset()
            d = False
            rs.append(0)
            for _ in count():
                if greedy:
                    a = eval_model.select_greedy_action(s)
                else: 
                    a = eval_model.select_action(s)
                # s, r, d, _ = eval_env.step(a)
                s, r, d, truncated, _ = eval_env.step(a)
                rs[-1] += r
                # if d: break
                if d or truncated: break
        return np.mean(rs), np.std(rs)

    def get_cleaned_checkpoints(self, n_checkpoints=4):
        try: 
            return self.checkpoint_paths
        except AttributeError:
            self.checkpoint_paths = {}

        paths = glob.glob(os.path.join(self.checkpoint_dir, '*.tar'))
        paths_dic = {int(path.split('.')[-2]):path for path in paths}
        last_ep = max(paths_dic.keys())
        # checkpoint_idxs = np.geomspace(1, last_ep+1, n_checkpoints, endpoint=True, dtype=np.int)-1
        checkpoint_idxs = np.linspace(1, last_ep+1, n_checkpoints, endpoint=True, dtype=int)-1

        for idx, path in paths_dic.items():
            if idx in checkpoint_idxs:
                self.checkpoint_paths[idx] = path
            else:
                os.unlink(path)

        return self.checkpoint_paths

    def demo_last(self, title='Fully-trained {} Agent', n_episodes=2, max_n_videos=2):
        env = self.make_env_fn(**self.make_env_kargs, monitor_mode='evaluation', render=True, record=True)

        checkpoint_paths = self.get_cleaned_checkpoints()
        last_ep = max(checkpoint_paths.keys())
        self.policy_model.load_state_dict(torch.load(checkpoint_paths[last_ep]))

        self.evaluate(self.policy_model, env, n_episodes=n_episodes)
        env.close()
        data = get_gif_html(env_videos=env.videos, 
                            title=title.format(self.__class__.__name__),
                            max_n_videos=max_n_videos)
        del env
        return HTML(data=data)

    def demo_progression(self, title='{} Agent progression', max_n_videos=4):
        env = self.make_env_fn(**self.make_env_kargs, monitor_mode='evaluation', render=True, record=True)

        checkpoint_paths = self.get_cleaned_checkpoints()
        for i in sorted(checkpoint_paths.keys()):
            self.policy_model.load_state_dict(torch.load(checkpoint_paths[i]))
            self.evaluate(self.policy_model, env, n_episodes=1)

        env.close()
        data = get_gif_html(env_videos=env.videos, 
                            title=title.format(self.__class__.__name__),
                            subtitle_eps=sorted(checkpoint_paths.keys()),
                            max_n_videos=max_n_videos)
        del env
        return HTML(data=data)

    def save_checkpoint(self, episode_idx, model):
        torch.save(model.state_dict(),
                   os.path.join(self.checkpoint_dir, 'model.{}.tar'.format(episode_idx)))

ppo_results = []
best_agent, best_eval_score = None, float('-inf')
SEEDS = [90]
for seed in SEEDS:
    environment_settings = {
        # 'env_name': 'LunarLander-v3',
        'gamma': 0.99,
        'max_minutes': np.inf,
        'max_episodes': 3_000,
        'goal_mean_100_reward': np.inf
    }

    policy_model_fn = lambda nS, nA: FCCA(nS, nA, hidden_dims=(256,256))
    policy_model_max_grad_norm = float('inf')
    policy_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
    policy_optimizer_lr = 0.0003
    policy_optimization_epochs = 80
    policy_sample_ratio = 0.8
    policy_clip_range = 0.1
    policy_stopping_kl = 0.02

    value_model_fn = lambda nS: FCV(nS, hidden_dims=(256,256))
    value_model_max_grad_norm = float('inf')
    value_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
    value_optimizer_lr = 0.0005
    value_optimization_epochs = 80
    value_sample_ratio = 0.8
    value_clip_range = float('inf')
    value_stopping_mse = 25

    episode_buffer_fn = lambda sd, ad, g, t, nw, me, mes: EpisodeBuffer(sd, ad, g, t, nw, me, mes)
    max_buffer_episodes = 256#16
    max_buffer_episode_steps = 12#1000
    
    entropy_loss_weight = 0.01
    tau = 0.97
    n_workers = 64#8

    params = {
        'n_months': 12,
        'V_min': 0,
        'V_max': 1,
        'V_0': 0,
        'W_max': 0.4,
        'I_max': 0.4,
        # 'storage_capacity': 100000,
        'kappa_r': 0.492828372105622,
        'sigma_r': 0.655898616135014,
        'theta_r': 0.000588276156660185,
        'kappa_delta': 1.17723166341479,
        'sigma_delta': 1.03663918307669,
        'theta_delta': -0.213183673388138,
        'sigma_s': 0.791065501973918,
        'rho_1': 0.899944474373156,
        'rho_2': -0.306810849087325,
        'sigma_v': 0.825941396204049,
        'theta_v': 0.0505685591761352,
        'theta': 0.00640705687096142,
        'kappa_v': 2.36309244973169,
        'lam': 0.638842070975342,
        'sigma_j': 0.032046147726045,
        'mu_j': 0.0137146728855484,
        'seed': seed,
        'initial_spot_price': np.exp(2.9479),
        'initial_r': 0.15958620269619,
        'initial_delta': 0.106417288572204,
        'initial_v': 0.0249967313173077,
        'penalty_lambda1': 20.0,#0.2,#2.0,#0.2,#10.0,
        'penalty_lambda2': 100.,#1,#10.0,#1.0,#50.0,
        'monthly_seasonal_factors': np.array([-0.106616824924423, -0.152361004102492, -0.167724706188117, -0.16797984045645,
                                     -0.159526180248348, -0.13927943487493, -0.0953402986114613, -0.0474646801238288, 
                                     -0.0278622280543003, 0.000000, -0.00850263509128089, -0.0409638719325969])
    }
    
    gamma, max_minutes, \
    max_episodes, goal_mean_100_reward = environment_settings.values()
    agent = PPO(policy_model_fn, 
                policy_model_max_grad_norm,
                policy_optimizer_fn,
                policy_optimizer_lr,
                policy_optimization_epochs,
                policy_sample_ratio,
                policy_clip_range,
                policy_stopping_kl,
                value_model_fn, 
                value_model_max_grad_norm,
                value_optimizer_fn,
                value_optimizer_lr,
                value_optimization_epochs,
                value_sample_ratio,
                value_clip_range,
                value_stopping_mse,
                episode_buffer_fn,
                max_buffer_episodes,
                max_buffer_episode_steps,
                entropy_loss_weight,
                tau,
                n_workers)

    make_envs_fn = lambda mef, mea, s, n: MultiprocessEnv(mef, mea, s, n)
    # make_env_fn, make_env_kargs = get_make_env_fn(env_name=env_name)
    make_env_fn = get_make_env_fn(params)
    make_env_kargs = {}
    result, final_eval_score, training_time, wallclock_time = agent.train(make_envs_fn,
                                                                          make_env_fn,
                                                                          make_env_kargs,
                                                                          seed,
                                                                          gamma,
                                                                          max_minutes,
                                                                          max_episodes,
                                                                          goal_mean_100_reward)
    ppo_results.append(result)
    if final_eval_score > best_eval_score:
        best_eval_score = final_eval_score
        best_agent = agent
ppo_results = np.array(ppo_results)
_ = BEEP()
