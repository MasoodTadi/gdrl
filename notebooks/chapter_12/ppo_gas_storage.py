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

        self._truncated_fn = np.vectorize(lambda x: 'TimeLimit.truncated' in x and x['TimeLimit.truncated'])
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
            next_states, rewards, terminateds, truncateds, infos = envs.step(actions)
            terminals = np.logical_or(terminateds, truncateds)
            self.states_mem[self.current_ep_idxs, worker_steps] = states
            self.actions_mem[self.current_ep_idxs, worker_steps] = actions
            self.logpas_mem[self.current_ep_idxs, worker_steps] = logpas

            worker_exploratory[np.arange(self.n_workers), worker_steps] = are_exploratory
            worker_rewards[np.arange(self.n_workers), worker_steps] = rewards

            for w_idx in range(self.n_workers):
                if worker_steps[w_idx] + 1 == self.max_episode_steps:
                    terminals[w_idx] = 1
                    infos[w_idx]['TimeLimit.truncated'] = True

            if terminals.sum():
                idx_terminals = np.flatnonzero(terminals)
                next_values = np.zeros(shape=(self.n_workers))
                truncated = self._truncated_fn(infos)
                if truncated.sum():
                    idx_truncated = np.flatnonzero(truncated)
                    with torch.no_grad():
                        next_values[idx_truncated] = value_model(
                            next_states[idx_truncated]).cpu().numpy()

            states = next_states
            worker_steps += 1

            if terminals.sum():
                new_states = envs.reset(ranks=idx_terminals)
                states[idx_terminals] = new_states

                for w_idx in range(self.n_workers):
                    if w_idx not in idx_terminals:
                        continue

                    e_idx = self.current_ep_idxs[w_idx]
                    T = worker_steps[w_idx]
                    self.episode_steps[e_idx] = T
                    self.episode_reward[e_idx] = worker_rewards[w_idx, :T].sum()
                    self.episode_exploration[e_idx] = worker_exploratory[w_idx, :T].mean()
                    self.episode_seconds[e_idx] = time.time() - worker_seconds[w_idx]

                    ep_rewards = np.concatenate(
                        (worker_rewards[w_idx, :T], [next_values[w_idx]]))
                    ep_discounts = self.discounts[:T+1]
                    ep_returns = np.array(
                        [np.sum(ep_discounts[:T+1-t] * ep_rewards[t:]) for t in range(T)])
                    self.returns_mem[e_idx, :T] = ep_returns

                    ep_states = self.states_mem[e_idx, :T]
                    with torch.no_grad():
                        ep_values = torch.cat((value_model(ep_states),
                                               torch.tensor([next_values[w_idx]],
                                                            device=value_model.device,
                                                            dtype=torch.float32)))
                    np_ep_values = ep_values.view(-1).cpu().numpy()
                    ep_tau_discounts = self.tau_discounts[:T]
                    deltas = ep_rewards[:-1] + self.gamma * np_ep_values[1:] - np_ep_values[:-1]
                    gaes = np.array(
                        [np.sum(self.tau_discounts[:T-t] * deltas[t:]) for t in range(T)])
                    self.gaes_mem[e_idx, :T] = gaes

                    worker_exploratory[w_idx, :] = 0
                    worker_rewards[w_idx, :] = 0
                    worker_steps[w_idx] = 0
                    worker_seconds[w_idx] = time.time()

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

envs = MultiprocessEnv(
    make_env_fn=get_make_env_fn,
    make_env_kargs=params,   # Nothing needed here, already inside `make_env_fn`
    seed=42, 
    n_workers=4
)

env = TTFGasStorageEnv(params)

obs = envs.reset()
print("Initial Observations:")
print(obs)
n_steps = 5

for step in range(n_steps):
    print(f"\n--- Step {step + 1} ---")

    random_actions = []
    for _ in range(envs.n_workers):
        random_action_indices = np.array([
            np.random.randint(0, env.action_space.nvec[i]) for i in range(env.n_months)
        ], dtype=np.int32)
        random_actions.append(random_action_indices)
    random_actions = np.stack(random_actions)

    # === Pretty print random actions ===
    print("Random Actions (indices):")
    for worker_idx in range(envs.n_workers):
        action_indices = random_actions[worker_idx]
        action_str = ", ".join(f"{a:2d}" for a in action_indices)
        print(f"Worker {worker_idx}: [{action_str}]")

    print("\nRandom Actions (real values):")
    for worker_idx in range(envs.n_workers):
        real_action = [
            env.action_meanings_list[i][random_actions[worker_idx][i]] for i in range(env.n_months)
        ]
        real_action_str = ", ".join(f"{val:+.2f}" for val in real_action)
        print(f"Worker {worker_idx}: [{real_action_str}]")

    # Now step environments
    obs, rewards, dones, infos = envs.step(random_actions)
envs.close()
