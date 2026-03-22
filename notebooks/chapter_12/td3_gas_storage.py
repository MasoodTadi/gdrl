import warnings ; warnings.filterwarnings('ignore')
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import torch.multiprocessing as mp
#import threading
#from torch.distributions import Normal

import numpy as np
#import pandas as pd
#from scipy.interpolate import CubicSpline
from scipy.optimize import linprog
#from IPython.display import display
#from collections import namedtuple, deque
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from itertools import cycle, count
from textwrap import wrap

#import pybullet_envs
#import pybullet
#import matplotlib
#import subprocess
#import os.path
#import tempfile
import random
#import base64
#import pprint
import glob
import time
#import json
#import sys
import gymnasium as gym
#import io
import gc

#from gym import wrappers
#from skimage.transform import resize
#from skimage.color import rgb2gray
#from subprocess import check_output
#from IPython.display import display, HTML

import datetime

LEAVE_PRINT_EVERY_N_SECS = 300
ERASE_LINE = '\x1b[2K'
EPS = 1e-6
BEEP = lambda: os.system("printf '\a'")
#RESULTS_DIR = os.path.join('..', 'results')

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

class TTFGasStorageEnv(gym.Env):
    def __init__(self, params):
        super().__init__()

        self.max_timesteps = 30 * 12 + 1
        self.seed_value = params.get("seed", None)
        self.dt = 1.0 / self.max_timesteps

        # Storage parameters
        self.price_mean = params["price_mean"]
        self.price_std = params["price_std"]
        self.V_min = params["V_min"]
        self.V_max = params["V_max"]
        self.V_0 = params["V_0"]
        self.I_max = params["I_max"]
        self.W_max = params["W_max"]
        self.n_months = params["n_months"]

        # Yan (2002) model parameters
        self.initial_r = params["initial_r"]
        self.theta_r = params["theta_r"]
        self.kappa_r = params["kappa_r"]
        self.sigma_r = params["sigma_r"]

        self.initial_delta = params["initial_delta"]
        self.theta_delta = params["theta_delta"]
        self.kappa_delta = params["kappa_delta"]
        self.sigma_delta = params["sigma_delta"]

        self.initial_v = params["initial_v"]
        self.kappa_v = params["kappa_v"]
        self.sigma_v = params["sigma_v"]
        self.theta_v = params["theta_v"]

        self.initial_spot_price = params["initial_spot_price"]
        self.sigma_s = params["sigma_s"]

        self.lam = params["lam"]
        self.mu_j = params["mu_j"]
        self.sigma_j = params["sigma_j"]
        self.theta = params["theta"]

        self.rho_1 = params["rho_1"]
        self.rho_2 = params["rho_2"]

        self.ksi_r = np.sqrt(self.kappa_r**2 + 2 * self.sigma_r**2)

        # Still keep RIV penalty if you want that benchmark regularization
        self.penalty_lambda_riv = params["penalty_lambda_riv"]

        self.seasonal_factors = params["monthly_seasonal_factors"]

        # Numerical tolerance for feasibility checks
        self.feas_tol = params.get("feas_tol", 1e-5)

        self.seed(self.seed_value)

        # Action space
        # action[0] can only inject or do nothing at the first maturity
        # last action can only withdraw or do nothing at the final maturity
        low = np.array([0.0] + [-self.W_max] * (self.n_months - 2) + [-self.W_max], dtype=np.float32)
        high = np.array([self.I_max] + [self.I_max] * (self.n_months - 2) + [0.0], dtype=np.float32)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # Observation space: [month, 12 futures prices, V_t]
        self.observation_space = gym.spaces.Box(
            low=np.concatenate(([0.0], np.zeros(12, dtype=np.float32), [self.V_min])).astype(np.float32),
            high=np.concatenate(([12.0], np.full(12, np.inf, dtype=np.float32), [self.V_max])).astype(np.float32),
            shape=(14,),
            dtype=np.float32,
        )

        self.reset()

    def seed(self, seed=None):
        if seed is not None:
            self.seed_value = seed
        self.W = np.random.default_rng(seed=self.seed_value)
        return [self.seed_value]

    def _get_obs(self):
        return np.concatenate(([self.month], self.F_t, [self.V_t]), dtype=np.float32)

    def compute_futures_curve(self):
        futures_list = np.full(12, 0.0, dtype=np.float32)

        for k in range(12):
            expiration_day = (k + 1) * 30
            tau = (expiration_day - self.day) / 360.0

            if tau < 0:
                continue

            beta_r = (2 * (1 - np.exp(-self.ksi_r * tau))) / (
                2 * self.ksi_r - (self.ksi_r - self.kappa_r) * (1 - np.exp(-self.ksi_r * tau))
            )
            beta_delta = -(1 - np.exp(-self.kappa_delta * tau)) / self.kappa_delta
            beta_0 = (
                (self.theta_r / self.sigma_r**2)
                * (
                    2
                    * np.log(
                        1
                        - (self.ksi_r - self.kappa_r)
                        * (1 - np.exp(-self.ksi_r * tau))
                        / (2 * self.ksi_r)
                    )
                    + (self.ksi_r - self.kappa_r) * tau
                )
                + (self.sigma_delta**2 * tau) / (2 * self.kappa_delta**2)
                - (self.sigma_s * self.sigma_delta * self.rho_1 + self.theta_delta) * tau / self.kappa_delta
                - (self.sigma_s * self.sigma_delta * self.rho_1 + self.theta_delta)
                * np.exp(-self.kappa_delta * tau)
                / self.kappa_delta**2
                + (
                    4 * self.sigma_delta**2 * np.exp(-self.kappa_delta * tau)
                    - self.sigma_delta**2 * np.exp(-2 * self.kappa_delta * tau)
                )
                / (4 * self.kappa_delta**3)
                + (self.sigma_s * self.sigma_delta * self.rho_1 + self.theta_delta) / self.kappa_delta**2
                - 3 * self.sigma_delta**2 / (4 * self.kappa_delta**3)
            )

            F_tk = np.exp(
                np.log(self.S_t)
                + self.seasonal_factors[k]
                + beta_0
                + beta_r * self.r_t
                + beta_delta * self.delta_t
            )
            futures_list[k] = F_tk

        return futures_list

    def _validate_action(self, action):
        """
        Numerical feasibility check only.
        Returns diagnostics; does not reshape reward anymore.
        """
        action = np.asarray(action, dtype=np.float32)

        valid_box = np.all(action >= self.action_space.low - self.feas_tol) and np.all(
            action <= self.action_space.high + self.feas_tol
        )

        # Expired maturities must be zero
        # At month m, maturities 0,...,m-1 are expired
        expired_ok = True
        if self.month > 0:
            expired_ok = np.all(np.abs(action[: self.month]) <= self.feas_tol)

        running = 0.0
        prefix_ok = True
        min_prefix_volume = np.inf
        max_prefix_volume = -np.inf

        for i in range(self.n_months):
            running += float(action[i])
            vol_i = self.V_t + running
            min_prefix_volume = min(min_prefix_volume, vol_i)
            max_prefix_volume = max(max_prefix_volume, vol_i)
            if vol_i < self.V_min - self.feas_tol or vol_i > self.V_max + self.feas_tol:
                prefix_ok = False

        equality_ok = abs(self.V_t + np.sum(action)) <= self.feas_tol

        feasible = valid_box and expired_ok and prefix_ok and equality_ok

        return {
            "feasible_action": feasible,
            "valid_box": valid_box,
            "expired_ok": expired_ok,
            "prefix_ok": prefix_ok,
            "equality_ok": equality_ok,
            "min_prefix_volume": min_prefix_volume,
            "max_prefix_volume": max_prefix_volume,
            "terminal_balance_error": float(self.V_t + np.sum(action)),
        }

    def compute_episode_riv(self):
        """
        Rolling Intrinsic + Extrinsic Value benchmark.
        """
        decision_times = np.arange(0, len(self.F_trajectory), 30)
        n = self.n_months
        V_t = self.V_0
        V_max, V_min = self.V_max, self.V_min
        I_max, W_max = self.I_max, self.W_max
        V_T = 0.0

        L_n = np.tril(np.ones((n, n)))
        A = np.vstack([L_n, -L_n])
        b = np.hstack([(V_max - V_t) * np.ones(n), -(V_min - V_t) * np.ones(n)])
        A_eq = np.ones((1, n))
        b_eq = np.array([V_T - V_t])
        bounds = [(-W_max, I_max)] * n

        CF = 0.0
        X_tau = np.zeros((len(decision_times), n))

        for i, tau in enumerate(decision_times):
            if i == 0:
                prices = self.F_trajectory[tau].copy()
                X_tau[i] = linprog(
                    prices, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
                ).x
                continue

            prev_tau = decision_times[i - 1]
            CF += np.dot((self.F_trajectory[tau] - self.F_trajectory[prev_tau]), X_tau[i - 1])

            prices = self.F_trajectory[tau].copy()
            zero_price_indices = prices == 0
            adjusted_bounds = [(0, 0) if zero_price_indices[k] else bounds[k] for k in range(len(prices))]

            if i < 12:
                X_tau[i] = linprog(
                    prices, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=adjusted_bounds, method="highs"
                ).x
            else:
                X_tau[i] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -V_t], dtype=np.float32)
                CF += -self.F_trajectory[tau][i - 1] * X_tau[i, i - 1]

            V_t += X_tau[i, i - 1]
            b = np.hstack([(V_max - V_t) * np.ones(n), -(V_min - V_t) * np.ones(n)])
            b_eq = np.array([V_T - V_t])

        return CF

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self.month = 0
        self.day = 0

        self.S_t = self.initial_spot_price
        self.r_t = self.initial_r
        self.delta_t = self.initial_delta
        self.v_t = self.initial_v
        self.V_t = self.V_0

        self.F_t = self.compute_futures_curve()

        self.S_trajectory = [self.S_t]
        self.r_trajectory = [self.r_t]
        self.delta_trajectory = [self.delta_t]
        self.v_trajectory = [self.v_t]
        self.F_trajectory = [self.F_t.copy()]

        self.rl_cumulative_reward = 0.0

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(self.n_months)

        # Numerical safeguard only
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Diagnostics only; actor should already ensure feasibility
        diag = self._validate_action(action)

        old_month = self.month
        old_F_t = self.F_t.copy()
        old_V_t = self.V_t

        reward = 0.0
        riv_penalty = 0.0

        # Economic reward from rolling futures strip
        # This uses the strip decided at time t over the next month.
        for _ in range(30):
            dW_1 = self.W.normal(0, np.sqrt(self.dt))
            dW_r = self.W.normal(0, np.sqrt(self.dt))
            dW_2 = self.W.normal(0, np.sqrt(self.dt))
            dW_delta = self.rho_1 * dW_1 + np.sqrt(1 - self.rho_1**2) * self.W.normal(0, np.sqrt(self.dt))
            dW_v = self.rho_2 * dW_2 + np.sqrt(1 - self.rho_2**2) * self.W.normal(0, np.sqrt(self.dt))

            dq = self.W.choice([0, 1], p=[1 - self.lam * self.dt, self.lam * self.dt])

            ln_1_plus_J = self.W.normal(
                np.log(1 + self.mu_j) - 0.5 * self.sigma_j**2,
                self.sigma_j,
            )
            J = np.exp(ln_1_plus_J) - 1
            J_v = self.W.exponential(scale=self.theta)

            dS_t = (
                (self.r_t - self.delta_t - self.lam * self.mu_j) * self.S_t * self.dt
                + self.sigma_s * self.S_t * dW_1
                + np.sqrt(max(self.v_t, 0)) * self.S_t * dW_2
                + J * self.S_t * dq
            )
            self.S_t += dS_t
            self.S_trajectory.append(self.S_t)

            dr_t = (
                (self.theta_r - self.kappa_r * self.r_t) * self.dt
                + self.sigma_r * np.sqrt(max(self.r_t, 0)) * dW_r
            )
            self.r_t += dr_t
            self.r_trajectory.append(self.r_t)

            ddelta_t = (
                (self.theta_delta - self.kappa_delta * self.delta_t) * self.dt
                + self.sigma_delta * dW_delta
            )
            self.delta_t += ddelta_t
            self.delta_trajectory.append(self.delta_t)

            dv_t = (
                (self.theta_v - self.kappa_v * self.v_t) * self.dt
                + self.sigma_v * np.sqrt(max(self.v_t, 0)) * dW_v
                + J_v * dq
            )
            self.v_t += dv_t
            self.v_trajectory.append(self.v_t)

            self.day += 1
            self.F_t = self.compute_futures_curve()
            self.F_trajectory.append(self.F_t.copy())

        # Futures MtM over one month
        reward += float(np.dot((self.F_t - old_F_t), action))

        # Realize current front-month physical/storage move after one month passes
        # At month m>0, the contract m-1 has just matured over this interval.
        if old_month > 0:
            self.V_t = self.V_t + action[old_month - 1]

        self.month += 1
        self.rl_cumulative_reward += reward

        terminated = False
        truncated = False

        # Terminal handling
        if self.month == 12:
            terminated = True

            # Final settlement of the last maturity
            reward += float(-self.F_t[-1] * action[-1])
            self.rl_cumulative_reward += float(-self.F_t[-1] * action[-1])

            self.V_t += action[-1]

            riv = self.compute_episode_riv()
            if self.rl_cumulative_reward < riv:
                riv_penalty = -self.penalty_lambda_riv * (riv - self.rl_cumulative_reward)
                reward += riv_penalty

        info = {
            **diag,
            "old_month": old_month,
            "old_V_t": float(old_V_t),
            "new_V_t": float(self.V_t),
            "riv_penalty": float(riv_penalty),
        }

        return self._get_obs(), float(reward), terminated, truncated, info

class FCDPAutoregressive(nn.Module):
    def __init__(
        self,
        input_dim,
        action_bounds,
        price_mean,
        price_std,
        hidden_dims=(512, 512, 256, 128),
        activation_fc=F.leaky_relu,
        V_min=0.0,
        V_max=1.0,
        I_max=0.4,
        W_max=0.4,
    ):
        super().__init__()
        self.activation_fc = activation_fc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        low, high = action_bounds
        self.env_min = torch.tensor(low, device=self.device, dtype=torch.float32)
        self.env_max = torch.tensor(high, device=self.device, dtype=torch.float32)

        self.V_min = torch.tensor(V_min, device=self.device, dtype=torch.float32)
        self.V_max = torch.tensor(V_max, device=self.device, dtype=torch.float32)
        self.I_max = torch.tensor(I_max, device=self.device, dtype=torch.float32)
        self.W_max = torch.tensor(W_max, device=self.device, dtype=torch.float32)

        self.price_mean = torch.tensor(price_mean, device=self.device, dtype=torch.float32)
        self.price_std = torch.tensor(price_std, device=self.device, dtype=torch.float32)

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.LayerNorm(hidden_dims[i + 1])
            ))

        # 11 latent outputs; feasible_map determines the 12th action from equality
        self.output_layer = nn.Linear(hidden_dims[-1], len(self.env_max) - 1)

        self.to(self.device)

    def _normalize(self, state: torch.Tensor) -> torch.Tensor:
        state = state.clone()

        # t in {0,...,11} -> [0,1]
        state[:, 0] = state[:, 0] / 11.0

        # Replace per-sample normalization by global mean/std normalization
        prices = state[:, 1:13]
        prices = (prices - self.price_mean) / self.price_std
        state[:, 1:13] = prices

        # Normalize V_t
        state[:, -1] = state[:, -1] / self.V_max

        return state

    def _format(self, state) -> torch.Tensor:
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        else:
            state = state.to(self.device, dtype=torch.float32)

        if state.ndim == 1:
            state = state.unsqueeze(0)

        return self._normalize(state)

    def net(self, state) -> torch.Tensor:
        """
        Returns unconstrained latent z with shape [batch_size, 11].
        """
        state = self._format(state)

        x = self.activation_fc(self.input_layer(state))
        for layer in self.hidden_layers:
            x = self.activation_fc(layer(x))

        z = self.output_layer(x)
        return z

    def feasible_map(self, state, z: torch.Tensor) -> torch.Tensor:
        """
        Maps latent vector z to a feasible action X.
        Noise should be added to z before calling this function.
        """
        state = self._format(state)
        z = z.to(self.device, dtype=torch.float32)

        # Recover original t and V_t from normalized state
        t = torch.round(state[:, 0] * 11.0).long()     # shape [batch]
        V_t = state[:, -1] * self.V_max                # shape [batch]

        batch_size = state.size(0)
        actions = []
        cum_V = V_t.clone()

        for j in range(11):
            # expired futures forced to zero
            active_mask = (j + 1 >= t).float()

            # Current one-step feasible interval from storage and rate constraints
            lower_volume = self.V_min - cum_V
            upper_volume = self.V_max - cum_V

            final_min = torch.maximum(lower_volume, -self.W_max)
            final_max = torch.minimum(upper_volume, self.I_max)

            # Future reachability so terminal equality remains feasible
            # Remaining actions after current one: indices j+1,...,11 plus x12 is handled by equality
            r = 11 - j
            reach_low = -cum_V - r * self.I_max
            reach_high = -cum_V + r * self.W_max

            final_min = torch.maximum(final_min, reach_low)
            final_max = torch.minimum(final_max, reach_high)

            # Safety guard in case numerical issues create inverted bounds
            final_max = torch.maximum(final_max, final_min)

            # Latent -> [0,1] fraction -> feasible action
            s_j = torch.sigmoid(z[:, j])
            a_j = final_min + (final_max - final_min) * s_j

            # Expired contracts are exactly zero
            a_j = a_j * active_mask

            actions.append(a_j)
            cum_V = cum_V + a_j

        # Last action from equality
        x12 = -V_t - torch.stack(actions, dim=1).sum(dim=1)

        # Keep same final clamp as your original implementation
        x12 = torch.clamp(x12, min=self.env_min[-1], max=self.env_max[-1])
        actions.append(x12)

        return torch.stack(actions, dim=1)

    def forward(self, state) -> torch.Tensor:
        z = self.net(state)
        X = self.feasible_map(state, z)
        return X

class FCQV(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        action_bounds,
        price_mean,
        price_std,
        hidden_dims=(512, 512, 256, 128),
        activation_fc=F.leaky_relu,
        V_max=1.0,
    ):
        super().__init__()
        self.activation_fc = activation_fc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.price_mean = torch.tensor(price_mean, device=self.device, dtype=torch.float32)
        self.price_std = torch.tensor(price_std, device=self.device, dtype=torch.float32)
        self.V_max = torch.tensor(V_max, device=self.device, dtype=torch.float32)

        low, high = action_bounds
        self.action_low = torch.tensor(low, device=self.device, dtype=torch.float32)
        self.action_high = torch.tensor(high, device=self.device, dtype=torch.float32)

        # Safer action scale for asymmetric bounds
        self.action_scale = torch.maximum(
            self.action_high.abs(),
            self.action_low.abs()
        )
        self.action_scale = torch.clamp(self.action_scale, min=1e-6)

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.input_norm = nn.LayerNorm(hidden_dims[0])

        self.hidden_layers = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()

        for i in range(len(hidden_dims) - 1):
            in_dim = hidden_dims[i]
            if i == 0:
                in_dim += output_dim  # concatenate action after first state block
            hidden_layer = nn.Linear(in_dim, hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)
            self.hidden_norms.append(nn.LayerNorm(hidden_dims[i + 1]))

        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        self.to(self.device)

    def _normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        state = state.clone()

        # time
        state[:, 0] = state[:, 0] / 11.0

        # prices
        state[:, 1:13] = (state[:, 1:13] - self.price_mean) / (self.price_std + 1e-6)

        # inventory
        state[:, -1] = state[:, -1] / self.V_max

        return state

    def _normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return action / self.action_scale

    def _format(self, state, action):
        x, u = state, action

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device, dtype=torch.float32)

        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, dtype=torch.float32, device=self.device)
        else:
            u = u.to(self.device, dtype=torch.float32)

        if x.ndim == 1:
            x = x.unsqueeze(0)
        if u.ndim == 1:
            u = u.unsqueeze(0)

        x = self._normalize_state(x)
        u = self._normalize_action(u)

        return x, u

    def forward(self, state, action):
        x, u = self._format(state, action)

        x = self.activation_fc(self.input_norm(self.input_layer(x)))

        for i, hidden_layer in enumerate(self.hidden_layers):
            if i == 0:
                x = torch.cat((x, u), dim=1)
            x = hidden_layer(x)
            x = self.hidden_norms[i](x)
            x = self.activation_fc(x)

        return self.output_layer(x)

    def load(self, experiences):
        states, actions, rewards, new_states, is_terminals = experiences

        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)

        return states, actions, rewards, new_states, is_terminals

class PrioritizedReplayBuffer:
    def __init__(
        self,
        max_samples=10000,
        batch_size=64,
        rank_based=False,
        alpha=0.6,
        beta0=0.1,
        beta_rate=0.99992,
    ):
        self.max_samples = max_samples
        self.batch_size = batch_size

        self.memory = np.empty((self.max_samples, 2), dtype=object)

        self.n_entries = 0
        self.next_index = 0

        self.td_error_index = 0
        self.sample_index = 1

        self.rank_based = rank_based
        self.alpha = alpha
        self.beta = beta0
        self.beta0 = beta0
        self.beta_rate = beta_rate

    def update(self, idxs, td_errors):
        idxs = np.asarray(idxs).reshape(-1)
        td_errors = np.asarray(td_errors).reshape(-1)

        self.memory[idxs, self.td_error_index] = np.abs(td_errors) + EPS

        if self.rank_based:
            sorted_arg = np.argsort(self.memory[:self.n_entries, self.td_error_index])[::-1]
            self.memory[:self.n_entries] = self.memory[sorted_arg]

    def store(self, sample):
        """
        sample is usually:
        (state, action, reward, next_state, done)
        """
        priority = 1.0
        if self.n_entries > 0:
            priority = np.max(self.memory[:self.n_entries, self.td_error_index])

        self.memory[self.next_index, self.td_error_index] = priority
        self.memory[self.next_index, self.sample_index] = tuple(sample)

        self.n_entries = min(self.n_entries + 1, self.max_samples)
        self.next_index = (self.next_index + 1) % self.max_samples

    def _update_beta(self):
        self.beta = min(1.0, self.beta * (self.beta_rate ** -1))
        return self.beta

    def sample(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        self._update_beta()

        entries = self.memory[:self.n_entries]

        if self.rank_based:
            priorities = 1.0 / (np.arange(self.n_entries) + 1.0)
        else:
            priorities = entries[:, self.td_error_index].astype(np.float64) + EPS

        scaled_priorities = priorities ** self.alpha
        probs = scaled_priorities / np.sum(scaled_priorities)

        idxs = np.random.choice(self.n_entries, batch_size, replace=False, p=probs)

        weights = (self.n_entries * probs[idxs]) ** (-self.beta)
        weights = weights / np.max(weights)

        batch = [entries[idx, self.sample_index] for idx in idxs]

        # unzip transitions
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.stack(states).astype(np.float32)
        actions = np.stack(actions).astype(np.float32)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
        next_states = np.stack(next_states).astype(np.float32)
        dones = np.asarray(dones, dtype=np.float32).reshape(-1, 1)

        idxs = idxs.reshape(-1, 1)
        weights = weights.astype(np.float32).reshape(-1, 1)

        samples_stacks = [states, actions, rewards, next_states, dones]
        return idxs, weights, samples_stacks

    def __len__(self):
        return self.n_entries

    def __repr__(self):
        return str(self.memory[:self.n_entries])

    def __str__(self):
        return str(self.memory[:self.n_entries])

class NormalNoiseStrategy:
    """
    Adds decaying Gaussian noise in LATENT space, then maps to a feasible action.
    """

    def __init__(
        self,
        action_bounds,
        exploration_noise_ratio,
        final_noise_ratio,
        max_episode,
        noise_free_last,
    ):
        self.low = np.array(action_bounds[0], dtype=np.float32)
        self.high = np.array(action_bounds[1], dtype=np.float32)
        self.action_range = self.high - self.low

        self.exploration_noise_ratio = exploration_noise_ratio
        self.final_noise_ratio = final_noise_ratio
        self.max_episode = max_episode
        self.noise_free_last = noise_free_last

        self.episodes_with_noise = max_episode - noise_free_last
        if self.episodes_with_noise > 0:
            self.decay_rate = (final_noise_ratio / exploration_noise_ratio) ** (1 / self.episodes_with_noise)
        else:
            self.decay_rate = 1.0

        self.current_episode = 0
        self.ratio_noise_injected = 0.0

    def decay_step(self):
        self.current_episode += 1

    @property
    def noise_ratio(self):
        if self.current_episode >= self.episodes_with_noise:
            return 0.0
        return self.exploration_noise_ratio * (self.decay_rate ** self.current_episode)

    def _active_mask_from_state(self, state: np.ndarray) -> np.ndarray:
        """
        Active maturities based on your convention:
        state = [t, F_1,...,F_12, V_t]
        active futures have nonzero price.
        """
        F_t = state[1:-1]
        return (F_t != 0)

    def _compute_noise_diagnostics(self, action, greedy_action, mask):
        if np.any(mask):
            ratio = np.abs((greedy_action[mask] - action[mask]) / np.maximum(self.action_range[mask], 1e-8))
            self.ratio_noise_injected = float(np.mean(ratio))
        else:
            self.ratio_noise_injected = 0.0

    def select_action(self, model: FCDPAutoregressive, state, max_exploration=False):
        """
        1) z = actor.net(state)
        2) z = z + noise
        3) X = actor.feasible_map(state, z)
        """
        state_tensor = torch.tensor(state, dtype=torch.float32, device=model.device).unsqueeze(0)

        # Use latent noise, not action-space noise
        latent_dim = len(self.high) - 1  # 11
        latent_scale = 1.0 if max_exploration else self.noise_ratio

        with torch.no_grad():
            z_greedy = model.net(state_tensor)
            greedy_action = model.feasible_map(state_tensor, z_greedy).cpu().numpy().squeeze()

            if latent_scale > 0.0:
                noise = torch.randn_like(z_greedy) * latent_scale
                z_noisy = z_greedy + noise
            else:
                z_noisy = z_greedy

            action = model.feasible_map(state_tensor, z_noisy).cpu().numpy().squeeze()

        mask = self._active_mask_from_state(np.asarray(state, dtype=np.float32))
        self._compute_noise_diagnostics(action, greedy_action, mask)

        return action.astype(np.float32)

class GreedyStrategy:
    def __init__(self, bounds):
        self.low = np.array(bounds[0], dtype=np.float32)
        self.high = np.array(bounds[1], dtype=np.float32)
        self.ratio_noise_injected = 0.0

    def select_action(self, model, state):
        with torch.no_grad():
            action = model(state).cpu().numpy().squeeze()

        # Optional numerical safety only.
        action = np.clip(action, self.low, self.high)
        return np.reshape(action, self.high.shape).astype(np.float32)

class TD3:
    def __init__(
        self,
        replay_buffer_fn,
        policy_model_fn,
        policy_max_grad_norm,
        policy_optimizer_fn,
        policy_optimizer_lr,
        value_model_fn,
        value_max_grad_norm,
        value_optimizer_fn,
        value_optimizer_lr,
        training_strategy_fn,
        evaluation_strategy_fn,
        n_warmup_batches,
        # update_target_every_steps,
        tau,
        policy_update_freq=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
    ):
        self.replay_buffer_fn = replay_buffer_fn

        self.policy_model_fn = policy_model_fn
        self.policy_max_grad_norm = policy_max_grad_norm
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr

        self.value_model_fn = value_model_fn
        self.value_max_grad_norm = value_max_grad_norm
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr

        self.training_strategy_fn = training_strategy_fn
        self.evaluation_strategy_fn = evaluation_strategy_fn

        self.n_warmup_batches = n_warmup_batches
        # self.update_target_every_steps = update_target_every_steps
        self.tau = tau

        self.policy_update_freq = policy_update_freq
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip

        self.total_env_steps = 0
        self.total_gradient_steps = 0

    def _target_action(self, next_states):
        """
        TD3 target policy smoothing in LATENT space:
            z = target_actor.net(next_states)
            z = z + clipped noise
            a = target_actor.feasible_map(next_states, z)
        """
        with torch.no_grad():
            z = self.target_policy_model.net(next_states)
            noise = torch.randn_like(z) * self.target_policy_noise
            noise = torch.clamp(noise, -self.target_noise_clip, self.target_noise_clip)
            z_noisy = z + noise
            next_actions = self.target_policy_model.feasible_map(next_states, z_noisy)
        return next_actions

    def optimize_model(self, idxs_weights_samples):
        idxs, weights, experiences = idxs_weights_samples
        states, actions, rewards, next_states, is_terminals = experiences

        weights = torch.as_tensor(
            weights,
            device=self.online_value_model_1.device,
            dtype=torch.float32,
        )

        # -----------------------------
        # 1) Critic targets
        # -----------------------------
        with torch.no_grad():
            next_actions = self._target_action(next_states)

            target_q1 = self.target_value_model_1(next_states, next_actions)
            target_q2 = self.target_value_model_2(next_states, next_actions)
            target_q = torch.minimum(target_q1, target_q2)

            target_q_sa = rewards + self.gamma * target_q * (1.0 - is_terminals)

        # -----------------------------
        # 2) Critic 1 update
        # -----------------------------
        current_q1 = self.online_value_model_1(states, actions)
        td_error1 = current_q1 - target_q_sa
        value_loss1 = (td_error1.pow(2) * weights).mean()

        self.value_optimizer_1.zero_grad()
        value_loss1.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_value_model_1.parameters(),
            self.value_max_grad_norm,
        )
        self.value_optimizer_1.step()

        # -----------------------------
        # 3) Critic 2 update
        # -----------------------------
        current_q2 = self.online_value_model_2(states, actions)
        td_error2 = current_q2 - target_q_sa
        value_loss2 = (td_error2.pow(2) * weights).mean()

        self.value_optimizer_2.zero_grad()
        value_loss2.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_value_model_2.parameters(),
            self.value_max_grad_norm,
        )
        self.value_optimizer_2.step()

        # PER update: use average absolute TD error from both critics
        avg_td_error = 0.5 * (td_error1.abs() + td_error2.abs())
        self.replay_buffer.update(
            idxs.squeeze(),
            avg_td_error.detach().cpu().numpy().squeeze(),
        )

        # -----------------------------
        # 4) Delayed actor + target update
        # -----------------------------
        self.total_gradient_steps += 1

        if self.total_gradient_steps % self.policy_update_freq == 0:
            policy_actions = self.online_policy_model(states)
            policy_loss = -self.online_value_model_1(states, policy_actions).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.online_policy_model.parameters(),
                self.policy_max_grad_norm,
            )
            self.policy_optimizer.step()

            self.update_networks()

    def interaction_step(self, state, env):
        min_samples = self.replay_buffer.batch_size * self.n_warmup_batches

        action = self.training_strategy.select_action(
            self.online_policy_model,
            state,
            max_exploration=(len(self.replay_buffer) < min_samples),
        )

        new_state, reward, terminated, truncated, info = env.step(action)
        done = float(terminated or truncated)

        experience = (state, action, reward, new_state, done)
        self.replay_buffer.store(experience)

        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += self.training_strategy.ratio_noise_injected

        self.total_env_steps += 1
        return new_state, terminated, truncated, info

    def update_networks(self, tau=None):
        tau = self.tau if tau is None else tau

        for target, online in zip(
            self.target_value_model_1.parameters(),
            self.online_value_model_1.parameters(),
        ):
            target.data.copy_((1.0 - tau) * target.data + tau * online.data)

        for target, online in zip(
            self.target_value_model_2.parameters(),
            self.online_value_model_2.parameters(),
        ):
            target.data.copy_((1.0 - tau) * target.data + tau * online.data)

        for target, online in zip(
            self.target_policy_model.parameters(),
            self.online_policy_model.parameters(),
        ):
            target.data.copy_((1.0 - tau) * target.data + tau * online.data)

    def train(self, env, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward):
        training_start, last_debug_time = time.time(), float("-inf")

        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # self.checkpoint_dir = os.path.expanduser(f"~/td3_checkpoints/")
        # os.makedirs(self.checkpoint_dir, exist_ok=True)

        # print(f"Running on: {os.uname().nodename}")
        # print(f"[INFO] Checkpoints will be saved to: {self.checkpoint_dir}")

        self.seed = seed
        self.gamma = gamma

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        nS = env.observation_space.shape[0]
        nA = env.action_space.shape[0]
        action_bounds = (env.action_space.low, env.action_space.high)

        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []
        self.episode_exploration = []

        price_mean = env.price_mean
        price_std = env.price_std

        # Two critics
        self.target_value_model_1 = self.value_model_fn(nS, nA, action_bounds, price_mean, price_std)
        self.online_value_model_1 = self.value_model_fn(nS, nA, action_bounds, price_mean, price_std)

        self.target_value_model_2 = self.value_model_fn(nS, nA, action_bounds, price_mean, price_std)
        self.online_value_model_2 = self.value_model_fn(nS, nA, action_bounds, price_mean, price_std)

        # One actor
        self.target_policy_model = self.policy_model_fn(nS, action_bounds, price_mean, price_std)
        self.online_policy_model = self.policy_model_fn(nS, action_bounds, price_mean, price_std)

        self.update_networks(tau=1.0)

        self.value_optimizer_1 = self.value_optimizer_fn(
            self.online_value_model_1,
            self.value_optimizer_lr,
        )
        self.value_optimizer_2 = self.value_optimizer_fn(
            self.online_value_model_2,
            self.value_optimizer_lr,
        )
        self.policy_optimizer = self.policy_optimizer_fn(
            self.online_policy_model,
            self.policy_optimizer_lr,
        )

        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = self.training_strategy_fn(action_bounds)
        self.evaluation_strategy = self.evaluation_strategy_fn(action_bounds)

        result = np.empty((max_episodes, 5))
        result[:] = np.nan

        training_time = 0.0
        self.total_env_steps = 0
        self.total_gradient_steps = 0

        for episode in range(1, max_episodes + 1):
            episode_start = time.time()

            state, _ = env.reset()
            terminated = False
            truncated = False

            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)

            for _ in count():
                state, terminated, truncated, info = self.interaction_step(state, env)

                min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
                if len(self.replay_buffer) >= min_samples:
                    idxs_weights_samples = self.replay_buffer.sample()
                    samples = self.online_value_model_1.load(idxs_weights_samples[2])
                    self.optimize_model((idxs_weights_samples[0], idxs_weights_samples[1], samples))

                if terminated or truncated:
                    gc.collect()
                    break

            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed

            evaluation_score, _ = self.evaluate(self.online_policy_model, env)
            # self.save_checkpoint(episode - 1, self.online_policy_model)

            total_step = int(np.sum(self.episode_timestep))
            self.evaluation_scores.append(evaluation_score)

            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])
            mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
            std_100_eval_score = np.std(self.evaluation_scores[-100:])

            lst_100_exp_rat = np.array(self.episode_exploration[-100:]) / np.maximum(
                np.array(self.episode_timestep[-100:]), 1.0
            )
            mean_100_exp_rat = np.mean(lst_100_exp_rat)
            std_100_exp_rat = np.std(lst_100_exp_rat)

            wallclock_elapsed = time.time() - training_start
            result[episode - 1] = (
                total_step,
                mean_100_reward,
                mean_100_eval_score,
                training_time,
                wallclock_elapsed,
            )

            reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS
            reached_max_minutes = wallclock_elapsed >= max_minutes * 60
            reached_max_episodes = episode >= max_episodes
            reached_goal_mean_reward = mean_100_eval_score >= goal_mean_100_reward

            training_is_over = (
                reached_max_minutes
                or reached_max_episodes
                or reached_goal_mean_reward
            )

            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
            debug_message = "el {}, ep {:04}, ts {:07}, "
            debug_message += "ar 10 {:05.1f}\u00B1{:05.1f}, "
            debug_message += "100 {:05.1f}\u00B1{:05.1f}, "
            debug_message += "ex 100 {:02.1f}\u00B1{:02.1f}, "
            debug_message += "ev {:05.1f}\u00B1{:05.1f}"
            debug_message = debug_message.format(
                elapsed_str,
                episode - 1,
                total_step,
                mean_10_reward,
                std_10_reward,
                mean_100_reward,
                std_100_reward,
                mean_100_exp_rat,
                std_100_exp_rat,
                mean_100_eval_score,
                std_100_eval_score,
            )

            print(debug_message, end="\r", flush=True)
            if reached_debug_time or training_is_over:
                print(ERASE_LINE + debug_message, flush=True)
                last_debug_time = time.time()

            self.training_strategy.decay_step()

            if training_is_over:
                if reached_max_minutes:
                    print("--> reached_max_minutes ✕")
                if reached_max_episodes:
                    print("--> reached_max_episodes ✕")
                if reached_goal_mean_reward:
                    print("--> reached_goal_mean_reward ✓")
                break

        final_eval_score, score_std = self.evaluate(self.online_policy_model, env, n_episodes=100)
        wallclock_time = time.time() - training_start

        print("Training complete.")
        print(
            "Final evaluation score {:.2f}±{:.2f} in {:.2f}s training time, {:.2f}s wall-clock time.\n".format(
                final_eval_score,
                score_std,
                training_time,
                wallclock_time,
            )
        )

        env.close()
        del env
        # self.get_cleaned_checkpoints()
        return result, final_eval_score, training_time, wallclock_time

    def evaluate(self, eval_policy_model, eval_env, n_episodes=1):
        rs = []
        for _ in range(n_episodes):
            s, _ = eval_env.reset()
            terminated = False
            truncated = False
            rs.append(0.0)

            for _ in count():
                a = self.evaluation_strategy.select_action(eval_policy_model, s)
                s, r, terminated, truncated, _ = eval_env.step(a)
                rs[-1] += r
                if terminated or truncated:
                    break

        return np.mean(rs), np.std(rs)

    def get_cleaned_checkpoints(self, n_checkpoints=4):
        try:
            return self.checkpoint_paths
        except AttributeError:
            self.checkpoint_paths = {}

        paths = glob.glob(os.path.join(self.checkpoint_dir, "*.tar"))
        paths_dic = {int(path.split(".")[-2]): path for path in paths}
        last_ep = max(paths_dic.keys())

        checkpoint_idxs = np.linspace(1, last_ep + 1, n_checkpoints, endpoint=True, dtype=int) - 1

        for idx, path in paths_dic.items():
            if idx in checkpoint_idxs:
                self.checkpoint_paths[idx] = path
            else:
                os.unlink(path)

        return self.checkpoint_paths

    def save_checkpoint(self, episode_idx, model):
        torch.save(
            model.state_dict(),
            os.path.join(self.checkpoint_dir, f"model.{episode_idx}.tar"),
        )

SEEDS = (1, 78, 90, 1024, 2048)
td3_results = []
best_agent, best_eval_score = None, float("-inf")

for seed in SEEDS:
    environment_settings = {
        "env_name": "TTFGasStorageEnv",
        "gamma": 0.99,
        "max_minutes": np.inf,
        "max_episodes": 20_000,
        "goal_mean_100_reward": 4.2,
    }

    params = {
        "price_mean": 17.49715440346992,
        "price_std": 5.874272071888386,
        "n_months": 12,
        "V_min": 0,
        "V_max": 1,
        "V_0": 0,
        "W_max": 0.4,
        "I_max": 0.4,
        "kappa_r": 0.492828372105622,
        "sigma_r": 0.655898616135014,
        "theta_r": 0.000588276156660185,
        "kappa_delta": 1.17723166341479,
        "sigma_delta": 1.03663918307669,
        "theta_delta": -0.213183673388138,
        "sigma_s": 0.791065501973918,
        "rho_1": 0.899944474373156,
        "rho_2": -0.306810849087325,
        "sigma_v": 0.825941396204049,
        "theta_v": 0.0505685591761352,
        "theta": 0.00640705687096142,
        "kappa_v": 2.36309244973169,
        "lam": 0.638842070975342,
        "sigma_j": 0.032046147726045,
        "mu_j": 0.0137146728855484,
        "seed": seed,
        "initial_spot_price": np.exp(2.9479),
        "initial_r": 0.15958620269619,
        "initial_delta": 0.106417288572204,
        "initial_v": 0.0249967313173077,
        "penalty_lambda_riv": 0.0,
        "monthly_seasonal_factors": np.array([
            -0.106616824924423, -0.152361004102492, -0.167724706188117, -0.16797984045645,
            -0.159526180248348, -0.13927943487493, -0.0953402986114613, -0.0474646801238288,
            -0.0278622280543003, 0.000000, -0.00850263509128089, -0.0409638719325969
        ], dtype=np.float32),
    }

    env = TTFGasStorageEnv(params)

    policy_model_fn = lambda nS, bounds, price_mean, price_std: FCDPAutoregressive(
        nS,
        bounds,
        price_mean=17.49715440346992,
        price_std=5.874272071888386,
        hidden_dims=(512, 512, 256, 128),
    )
    policy_max_grad_norm = float('inf')
    policy_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
    policy_optimizer_lr = 3e-5

    value_model_fn = lambda nS, nA, bounds, price_mean, price_std: FCQV(
        nS,
        nA,
        action_bounds=bounds,
        price_mean=17.49715440346992,
        price_std=5.874272071888386,
        hidden_dims=(512, 512, 256, 128),
    )
    value_max_grad_norm = float('inf')
    value_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
    value_optimizer_lr = 5e-4

    training_strategy_fn = lambda bounds: NormalNoiseStrategy(
        action_bounds=bounds,
        exploration_noise_ratio=0.20,
        final_noise_ratio=0.01,
        max_episode=environment_settings["max_episodes"],
        noise_free_last=int(0.1 * environment_settings["max_episodes"]),
    )

    evaluation_strategy_fn = lambda bounds: GreedyStrategy(bounds)

    replay_buffer_fn = lambda: PrioritizedReplayBuffer(
        max_samples=200_000,
        batch_size=256,
    )

    n_warmup_batches = 200
    # update_target_every_steps = 1
    tau = 0.005

    policy_update_freq = 2
    target_policy_noise = 0.2
    target_noise_clip = 0.1

    gamma = environment_settings["gamma"]
    max_minutes = environment_settings["max_minutes"]
    max_episodes = environment_settings["max_episodes"]
    goal_mean_100_reward = environment_settings["goal_mean_100_reward"]

    agent = TD3(
        replay_buffer_fn=replay_buffer_fn,
        policy_model_fn=policy_model_fn,
        policy_max_grad_norm=policy_max_grad_norm,
        policy_optimizer_fn=policy_optimizer_fn,
        policy_optimizer_lr=policy_optimizer_lr,
        value_model_fn=value_model_fn,
        value_max_grad_norm=value_max_grad_norm,
        value_optimizer_fn=value_optimizer_fn,
        value_optimizer_lr=value_optimizer_lr,
        training_strategy_fn=training_strategy_fn,
        evaluation_strategy_fn=evaluation_strategy_fn,
        n_warmup_batches=n_warmup_batches,
        # update_target_every_steps=update_target_every_steps,
        tau=tau,
        policy_update_freq=policy_update_freq,
        target_policy_noise=target_policy_noise,
        target_noise_clip=target_noise_clip,
    )

    result, final_eval_score, training_time, wallclock_time = agent.train(
        env,
        seed,
        gamma,
        max_minutes,
        max_episodes,
        goal_mean_100_reward,
    )

    td3_results.append(result)

    if final_eval_score > best_eval_score:
        best_eval_score = final_eval_score
        best_agent = agent

td3_results = np.array(td3_results)

torch.save(
    best_agent.online_policy_model.state_dict(),
    "online_policy_model_autoregressive_td3.pth",
)

td3_max_t, td3_max_r, td3_max_s, \
td3_max_sec, td3_max_rt = np.max(td3_results, axis=0).T
td3_min_t, td3_min_r, td3_min_s, \
td3_min_sec, td3_min_rt = np.min(td3_results, axis=0).T
td3_mean_t, td3_mean_r, td3_mean_s, \
td3_mean_sec, td3_mean_rt = np.mean(td3_results, axis=0).T
td3_x = np.arange(len(td3_mean_s))

fig, axs = plt.subplots(2, 1, figsize=(15,10), sharey=False, sharex=True)

# TD3
axs[0].plot(td3_max_r, 'r', linewidth=1)
axs[0].plot(td3_min_r, 'r', linewidth=1)
axs[0].plot(td3_mean_r, 'r:', label='TD3', linewidth=2)
axs[0].fill_between(
    td3_x, td3_min_r, td3_max_r, facecolor='r', alpha=0.3)

axs[1].plot(td3_max_s, 'r', linewidth=1)
axs[1].plot(td3_min_s, 'r', linewidth=1)
axs[1].plot(td3_mean_s, 'r:', label='TD3', linewidth=2)
axs[1].fill_between(
     td3_x, td3_min_s, td3_max_s, facecolor='r', alpha=0.3)

# ALL
axs[0].set_title('Moving Avg Reward (Training)')
axs[1].set_title('Moving Avg Reward (Evaluation)')
plt.xlabel('Episodes')
axs[0].legend(loc='upper left')
plt.savefig("Moving_Average_Reward_Autoregressive_Penalized_latent_noise.png")

def compute_futures_curve(day, S_t, r_t, delta_t):
    futures_list = np.full((N_simulations,12), 0.0, dtype=np.float32)  # Initialize all values as 0.0
    #remaining_futures = max(12 - (day // 30), 0)  # Shrinks every 30 days
    for k in range(12):
        expiration_day = (k+1) * 30  # Expiration at the end of month (1-based index)
        tau = (expiration_day - day) / 360.0
        if tau < 0:  # Contract expired, skip (remains 0.0)
            continue
        beta_r = (2 * (1 - np.exp(-ksi_r * tau))) / (2 * ksi_r - (ksi_r - kappa_r) * (1 - np.exp(-ksi_r * tau)))
        beta_delta = -(1 - np.exp(-kappa_delta * tau)) / kappa_delta
        beta_0 = (theta_r / sigma_r**2) * (2 * np.log(1 - (ksi_r - kappa_r) * (1 - np.exp(-ksi_r * tau)) / (2 * ksi_r))
                                                     + (ksi_r - kappa_r) * tau) \
                 + (sigma_delta**2 * tau) / (2 * kappa_delta**2) \
                 - (sigma_s * sigma_delta * rho_1 + theta_delta) * tau / kappa_delta \
                 - (sigma_s * sigma_delta * rho_1 + theta_delta) * np.exp(-kappa_delta * tau) / kappa_delta**2 \
                 + (4 * sigma_delta**2 * np.exp(-kappa_delta * tau) - sigma_delta**2 * np.exp(-2 * kappa_delta * tau)) / (4 * kappa_delta**3) \
                 + (sigma_s * sigma_delta * rho_1 + theta_delta) / kappa_delta**2 \
                 - 3 * sigma_delta**2 / (4 * kappa_delta**3)
        F_tk = np.exp(np.log(S_t) + seasonal_factors[k] + beta_0 + beta_r * r_t + beta_delta * delta_t)
        futures_list[:,k] = F_tk
    return futures_list
    
def compute_futures_curve_scalar(day, S_t, r_t, delta_t):
    futures = np.full(12, 0.0, dtype=np.float32)
    for k in range(12):
        expiration_day = (k + 1) * 30
        tau = (expiration_day - day) / 360.0
        if tau < 0:
            continue

        beta_r = (2 * (1 - np.exp(-ksi_r * tau))) / (2 * ksi_r - (ksi_r - kappa_r) * (1 - np.exp(-ksi_r * tau)))
        beta_delta = -(1 - np.exp(-kappa_delta * tau)) / kappa_delta
        beta_0 = (theta_r / sigma_r**2) * (2 * np.log(1 - (ksi_r - kappa_r) * (1 - np.exp(-ksi_r * tau)) / (2 * ksi_r))
                 + (ksi_r - kappa_r) * tau) \
            + (sigma_delta**2 * tau) / (2 * kappa_delta**2) \
            - (sigma_s * sigma_delta * rho_1 + theta_delta) * tau / kappa_delta \
            - (sigma_s * sigma_delta * rho_1 + theta_delta) * np.exp(-kappa_delta * tau) / kappa_delta**2 \
            + (4 * sigma_delta**2 * np.exp(-kappa_delta * tau) - sigma_delta**2 * np.exp(-2 * kappa_delta * tau)) / (4 * kappa_delta**3) \
            + (sigma_s * sigma_delta * rho_1 + theta_delta) / kappa_delta**2 \
            - 3 * sigma_delta**2 / (4 * kappa_delta**3)

        F_tk = np.exp(np.log(S_t) + seasonal_factors[k] + beta_0 + beta_r * r_t + beta_delta * delta_t)
        futures[k] = F_tk

    return futures

# Parameters for the Yan (2002) model
N_simulations = 100 # Number of simulations
T = 360  
dt = 1/(T+1)
# Model Parameters (Assumed)
kappa_r = 0.492828372105622
sigma_r = 0.655898616135014
theta_r = 0.000588276156660185
kappa_delta= 1.17723166341479
sigma_delta = 1.03663918307669
theta_delta = -0.213183673388138
sigma_s = 0.791065501973918
rho_1 = 0.899944474373156
rho_2 = -0.306810849087325
sigma_v = 0.825941396204049
theta_v = 0.0505685591761352
theta = 0.00640705687096142
kappa_v = 2.36309244973169
lam = 0.638842070975342
sigma_j = 0.032046147726045
mu_j = 0.0137146728855484
seed = 1
initial_spot_price = np.exp(2.9479)
initial_r = 0.15958620269619
initial_delta =  0.106417288572204
initial_v =  0.0249967313173077

ksi_r = np.sqrt(kappa_r**2 + 2*sigma_r**2)
seasonal_factors = np.array([ -0.106616824924423, -0.152361004102492, -0.167724706188117, -0.16797984045645,
                             -0.159526180248348, -0.13927943487493, -0.0953402986114613, -0.0474646801238288,
                             -0.0278622280543003, 0.000000, -0.00850263509128089, -0.0409638719325969  ])

# Simulate state variables using Euler-Maruyama
# Pre-allocate arrays
S_t = np.zeros((N_simulations, T + 1))
r_t = np.zeros((N_simulations, T + 1))
delta_t = np.zeros((N_simulations, T + 1))
v_t = np.zeros((N_simulations, T + 1))
F_t = np.zeros((N_simulations, T + 1, 12))

# Set initial state
S_t[:, 0] = initial_spot_price
r_t[:, 0] = initial_r
delta_t[:, 0] = initial_delta
v_t[:, 0] = initial_v
F_t[:, 0, :] = compute_futures_curve(0, S_t[:, 0], r_t[:, 0], delta_t[:, 0])

# Create one RNG per simulation (aligns with how env would run one episode at a time)
rngs = [np.random.default_rng(seed + i) for i in range(N_simulations)]

for sim in range(N_simulations):
    S, r, delta, v = S_t[sim, 0], r_t[sim, 0], delta_t[sim, 0], v_t[sim, 0]
    
    for day in range(1, T + 1):
        rng = rngs[sim]
        
        # Match the environment’s RNG call sequence
        dW_1 = rng.normal(0, np.sqrt(dt))
        dW_r = rng.normal(0, np.sqrt(dt))
        dW_2 = rng.normal(0, np.sqrt(dt))
        dW_delta = rho_1 * dW_1 + np.sqrt(1 - rho_1 ** 2) * rng.normal(0, np.sqrt(dt))
        dW_v = rho_2 * dW_2 + np.sqrt(1 - rho_2 ** 2) * rng.normal(0, np.sqrt(dt))

        dq = rng.choice([0, 1], p=[1 - lam * dt, lam * dt])
        ln_1_plus_J = rng.normal(np.log(1 + mu_j) - 0.5 * sigma_j**2, sigma_j)
        J = np.exp(ln_1_plus_J) - 1
        J_v = rng.exponential(scale=theta)

        dS = (r - delta - lam * mu_j) * S * dt + sigma_s * S * dW_1 + np.sqrt(max(v, 0)) * S * dW_2 + J * S * dq
        dr = (theta_r - kappa_r * r) * dt + sigma_r * np.sqrt(max(r, 0)) * dW_r
        ddelta = (theta_delta - kappa_delta * delta) * dt + sigma_delta * dW_delta
        dv = (theta_v - kappa_v * v) * dt + sigma_v * np.sqrt(max(v, 0)) * dW_v + J_v * dq

        # Update states
        S += dS
        r += dr
        delta += ddelta
        v += dv

        S_t[sim, day] = S
        r_t[sim, day] = r
        delta_t[sim, day] = delta
        v_t[sim, day] = v
        F_t[sim, day, :] = compute_futures_curve_scalar(day, S, r, delta)

# Rolling Intrinsic valuation
N_simulations = 100  # Number of simulations
n = 12  # Number of months
T = 360  
N_maturities = 12
decision_times = np.arange(0, T+1, 30)  # Decision points tau = [0, 30, 60, ..., 360]
V_max = 1.0
V_min = 0.0
I_max = 0.4
W_max = 0.4
V_0 = 0  # Initial reservoir balance
V_T = 0  # Final balance condition

# Initialize results
V_I_Rolling = np.zeros((N_simulations, len(decision_times)))  # Intrinsic value per maturity
CF_IE_Rolling = np.zeros(N_simulations)  # Total cash-flow per simulation
X_tau = np.zeros((N_simulations, len(decision_times), N_maturities))

# Loop over all simulations
for j in range(N_simulations):
    V_t = V_0
    L_n = np.tril(np.ones((n, n)))  # Lower triangular matrix for cumulative sums
    A = np.vstack([L_n, -L_n])  # Stacking for V_max and V_min constraints
    b = np.hstack([(V_max - V_t) * np.ones(n), -(V_min - V_t) * np.ones(n)])
    # Equality constraint: Final balance VT = 0
    A_eq = np.ones((1, n))
    b_eq = np.array([V_T - V_t])
    # Box constraints: -W_max ≤ xi ≤ I_max
    bounds = [(-W_max, I_max) for _ in range(n)]
    CF = 0
    for i, tau in enumerate(decision_times):
        if i == 0:
            # Compute initial intrinsic value V_I,0 = -F_0 * X_0 for all maturities
            prices = F_t[j, tau, :]
            X_tau[j, i, :] = linprog(prices, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds,  method="highs").x 
            V_I_Rolling[j, i] = -np.dot(prices , X_tau[j, i, :])
            continue
            
        prev_tau = decision_times[i-1]
        
        # Compute realized P/L from futures position for each maturity
        CF += np.dot((F_t[j, tau, :] - F_t[j, prev_tau, :]) , X_tau[j, i-1, :])

        # Update intrinsic value for each maturity
        prices = F_t[j, tau, :]
        # Identify zero-price variables
        zero_price_indices = (prices == 0)
        # Modify bounds: Force zero-price variables to be zero (fix them)
        adjusted_bounds = [(0, 0) if zero_price_indices[i] else bounds[i] for i in range(len(prices))]
        
        if i < 12:
            X_tau[j, i, :] = linprog(prices, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=adjusted_bounds, method="highs").x 
        else:
            X_tau[j, i, :] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -V_t])
            CF += -F_t[j, tau, i-1] * X_tau[j, i, i-1]

        V_I_Rolling[j, i] = -np.dot(prices , X_tau[j, i, :])
        
        V_t += X_tau[j, i, i-1]
        b = np.hstack([(V_max - V_t) * np.ones(n), -(V_min - V_t) * np.ones(n)])
        b_eq = np.array([V_T - V_t])

        # Compute spot cash-flow using front-month futures (maturity=0)
        # CF += -F_t[j, tau, i-1] * X_tau[j, i, i-1]
            
    # Update total cash-flows for this simulation
    CF_IE_Rolling[j] = CF

# Compute estimated total intrinsic + extrinsic value
V_IE_Rolling = np.mean(CF_IE_Rolling)

# Display results
print(f"Estimated Rolling Intrinsic + Extrinsic Value: {V_IE_Rolling:.4f}")

# DRL Valuation
N_simulations = 100 # Number of simulations
T = 360  
N_maturities = 12
decision_times = np.arange(0, T+1, 30)  # Decision points tau = [0, 30, 60, ..., 360]
V_max = 1.0
V_min = 0.0
I_max = 0.4
W_max = 0.4
V_0 = 0  # Initial reservoir balance
V_T = 0  # Final balance condition

# Initialize results
V_I = np.zeros((N_simulations, len(decision_times)))  # Intrinsic value per maturity
CF_IE = np.zeros(N_simulations)  # Total cash-flow per simulation
X_tau = np.zeros((N_simulations, len(decision_times), N_maturities))

# Loop over all simulations
for j in range(N_simulations):
    V_t = V_0
    # L_n = np.tril(np.ones((n, n)))  # Lower triangular matrix for cumulative sums
    # A = np.vstack([L_n, -L_n])  # Stacking for V_max and V_min constraints
    # b = np.hstack([(V_max - V_t) * np.ones(n), -(V_min - V_t) * np.ones(n)])
    # # Equality constraint: Final balance VT = 0
    # A_eq = np.ones((1, n))
    # b_eq = np.array([V_T - V_t])
    # # Box constraints: -W_max ≤ xi ≤ I_max
    # bounds = [(-W_max, I_max) for _ in range(n)]
    CF = 0
    for i, tau in enumerate(decision_times):
        if i == 0:
            # Compute initial intrinsic value V_I,0 = -F_0 * X_0 for all maturities
            prices = F_t[j, tau, :]
            state = np.concatenate((np.array([i]),prices,np.array([V_t])),dtype=np.float32)
            # env.month = state[0]
            # env.V_t = state[-1]
            X_tau[j, i, :] = best_agent.evaluation_strategy.select_action(agent.online_policy_model, state)
            # X_tau[j, i, -1] = np.clip(-X_tau[j, i, :-1].cumsum()[-1]-V_t,-W_max, I_max)
            X_tau[j, i, :] = np.round(X_tau[j, i, :],2)
            # X_tau[j, i, :] = trained_network(state).detach().cpu().numpy()
            V_I[j, i] = -np.dot(prices , X_tau[j, i, :])
            continue
            
        prev_tau = decision_times[i-1]
        # Compute realized P/L from futures position for each maturity
        CF += np.dot((F_t[j, tau, :] - F_t[j, prev_tau, :]) , X_tau[j, i-1, :])

        # Update intrinsic value for each maturity
        prices = F_t[j, tau, :]
        state = np.concatenate((np.array([i]),prices,np.array([V_t])),dtype=np.float32)
        # env.month = state[0]
        # env.V_t = state[-1]
        X_tau[j, i, :] = best_agent.evaluation_strategy.select_action(best_agent.online_policy_model, state)
        # X_tau[j, i, -1] = np.clip(-X_tau[j, i, :-1].cumsum()[-1]-V_t,-W_max, I_max)
        X_tau[j, i, :] = np.round(X_tau[j, i, :],2)
        # if i < 12:
        #     X_tau[j, i, :] = env.rescale_actions(agent.evaluation_strategy.select_action(agent.online_policy_model, state))
        #     # X_tau[j, i, :] = trained_network(state).detach().cpu().numpy()
        # else:
        #     X_tau[j, i, :] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.clip(-V_t,-W_max,I_max)])
        if i == 12:           
            X_tau[j, i, :] = np.zeros(12,dtype=np.float32)
            X_tau[j, i, -1] = X_tau[j, i-1, -1]
            X_tau[j, i, :] = np.round(X_tau[j, i, :],2)
            # print("X_tau[j, i]:  ",np.round(X_tau[j, i],2))
            CF += -F_t[j, tau, i-1] * X_tau[j, i-1, i-1] 
            V_I[j, i] = -np.dot(prices , X_tau[j, i, :])

        V_I[j, i] = -np.dot(prices , X_tau[j, i, :])
        
        V_t += np.round(X_tau[j, i, i-1],2)
        # b = np.hstack([(V_max - V_t) * np.ones(n), -(V_min - V_t) * np.ones(n)])
        # b_eq = np.array([V_T - V_t])

        # Compute spot cash-flow using front-month futures (maturity=0)
        # CF += -F_t[j, tau, i-1] * X_tau[j, i, i-1]
            
    # Update total cash-flows for this simulation
    CF_IE[j] = CF

# Compute estimated total intrinsic + extrinsic value
V_IE = np.mean(CF_IE)

# Display results
print(f"Estimated Rolling Intrinsic + Extrinsic Value: {V_IE:.4f}")

plt.style.use('default')  # Reset to classic white background style 
plt.figure(figsize=(14, 6))
plt.plot(CF_IE, color='grey', label="Realized RL Value")
plt.plot(CF_IE_Rolling, color='black', label="Rolling Intrinsic Value")
plt.axhline(V_IE, color='green', linestyle='--', linewidth=2, label="RL Average Value")
plt.axhline(V_IE_Rolling, color='red', linestyle='--', linewidth=2, label="RI Average Value")
plt.axhline(V_I_Rolling[0][0], color='blue', linestyle='--', linewidth=2, label="Intrinsic Value")
plt.xlabel("Simulation ID")
plt.ylabel("Realized Reservoir Value")
plt.title("Reinforcement Learning Value Calculation")
plt.legend()
plt.grid(True)
plt.savefig("Reinforcement_Learning_Value_Autoregressive_Penalized_td3.png")



