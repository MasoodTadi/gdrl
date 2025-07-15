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
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import StepLR

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

import datetime

class TTFGasStorageEnv(gym.Env):
    def __init__(self, params):
        super(TTFGasStorageEnv, self).__init__()
        
        self.max_timesteps = 30 * 12 + 1 #Simplistic Case
        self.seed_value = params.get('seed', None)
        self.dt = 1.0 / self.max_timesteps
        # self.storage_capacity = params['storage_capacity']
        self.V_min = params['V_min']
        self.V_max = params['V_max']
        self.V_0 = params['V_0']
        self.I_max = params['I_max']
        self.W_max = params['W_max']
        self.n_months = params['n_months']
        
        ## Yan (2002) Model Parameters
        # Short rate: r_t
        self.initial_r = params['initial_r'] # Initial short rate
        self.theta_r = params['theta_r'] # Long‐run mean level of r_t
        self.kappa_r = params['kappa_r'] # Speed of mean reversion for r_t
        self.sigma_r = params['sigma_r'] # Volatility coefficient (diffusion) for r_t

        # Convenience Yield: delta_t
        self.initial_delta = params['initial_delta'] # Initial convenience yield
        self.theta_delta = params['theta_delta'] # Long‐run mean of delta_t
        self.kappa_delta = params['kappa_delta'] # Speed of mean reversion for delta_t
        self.sigma_delta = params['sigma_delta'] # Volatility coefficient (diffusion) for delta_t

        # Stochastic Variance v_t
        self.initial_v = params['initial_v'] # Initial variance
        self.kappa_v = params['kappa_v'] # Long‐run mean of v_t
        self.sigma_v = params['sigma_v'] # Speed of mean reversion for v_t
        self.theta_v = params['theta_v'] # Volatility coefficient (diffusion) for v_t

        # Spot Price Factor S_t
        self.initial_spot_price = params['initial_spot_price'] # Initial (de‐seasoned) spot price.
        self.sigma_s = params['sigma_s'] # Factor loading” or volatility parameter on S_t

        # Jump Process Parameters
        self.lam = params['lam'] # Jump intensity (Poisson arrival rate)
        self.mu_j = params['mu_j'] # average size of spot‐price jumps
        self.sigma_j = params['sigma_j'] # dispersion (volatility) of spot‐price jumps
        self.theta = params['theta'] # v_t jump size
        
        # Correlations among Brownian increments
        self.rho_1 = params['rho_1'] # Correlation between dW_1 and dW_delta
        self.rho_2 = params['rho_2'] # Correlation between dW_2 and dW_v

        # ksi_r constant in futures formula
        self.ksi_r = np.sqrt(self.kappa_r**2 + 2*self.sigma_r**2)

        ## Penalty parameters
        self.penalty_lambda1 = params['penalty_lambda1']  # For inequality constraint violation
        self.penalty_lambda2 = params['penalty_lambda2']  # For final sum violation
        # self.bonus_lambda = params['bonus_lambda']      # For hitting final sum exactly (optional bonus)
        self.penalty_lambda_riv = params['penalty_lambda_riv']

        # Seasonal Factors (Month 1 is April, Month 12 is March)
        # self.seasonal_factors = pd.read_csv('interpolated_seasonal_factors.csv',index_col=0)
        self.seasonal_factors = params['monthly_seasonal_factors']

        # Set the seed for reproducibility
        self.seed(self.seed_value)

        # ----- ACTION SPACE -----
        # Each action is an array of length self.n_months (injection/withdrawal for each month).
        # Box constraints: [-W_max, I_max] for each entry.
        # low = np.array([-self.W_max]*(self.n_months), dtype=np.float32)
        # high = np.array([ self.I_max]*(self.n_months), dtype=np.float32)
        # self.action_space = gym.spaces.Box(low=low, high=high, shape=(self.n_months,), dtype=np.float32, seed=self.seed_value)
        low = np.array([0.0] + [-self.W_max]*(self.n_months - 2) + [-self.W_max])
        high = np.array([self.I_max] + [self.I_max]*(self.n_months - 2) + [0.0])
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # ----- OBSERVATION SPACE -----
        # 1) month index t (discrete 1..12) - will store as an integer
        # 2) futures term structure: 12 prices max
        # 3) storage level V_t
        # We'll flatten them into one array for Gym:
        # Structure: [month (1), futures prices (12), storage level (1)] → Total = 14 elements
        self.observation_space = gym.spaces.Box(
            low=np.concatenate(([0], [-np.inf] * 12, [self.V_min])),  # Lower bounds
            high=np.concatenate(([12], [np.inf] * 12, [self.V_max])),  # Upper bounds
            shape=(14,), dtype=np.float32, seed=self.seed_value
        )
        
        # Initialize environment variables
        self.reset()
    
    def seed(self, seed=None):
        """
        Seed the environment, ensuring reproducibility of the randomness in the environment.
        """
        if seed is not None:
            self.seed_value = seed  # Update seed if provided
        self.W = np.random.default_rng(seed=self.seed_value)  # Seed the random generator
        #self.action_space.seed(self.seed_value)  # Seed the action space random generator
        return [self.seed_value]   

    def compute_futures_curve(self):
        """
        Computes the futures curve at the current day t, ensuring:
        - The length of the futures curve is always 12.
        - Expired futures are replaced with 0.0.
        - Futures prices are determined using the Yan (2002) model.
    
        Returns:
          np.ndarray : Array of 12 futures prices (valid prices or 0.0 for expired contracts).
        """
        futures_list = np.full(12, 0.0, dtype=np.float32)  # Initialize all values as 0.0
    
        # Determine the number of remaining valid futures contracts
        remaining_futures = max(12 - (self.day // 30), 0)  # Shrinks every 30 days
    
        for k in range(12):
            expiration_day = (k+1) * 30  # Expiration at the end of month (1-based index)
    
            # Compute time to expiration tau in years
            tau = (expiration_day - self.day) / 360.0
    
            if tau < 0:  # Contract expired, skip (remains 0.0)
                continue
    
            # Compute beta functions using Yan (2002) model
            beta_r = (2 * (1 - np.exp(-self.ksi_r * tau))) / (2 * self.ksi_r - (self.ksi_r - self.kappa_r) * (1 - np.exp(-self.ksi_r * tau)))
            beta_delta = -(1 - np.exp(-self.kappa_delta * tau)) / self.kappa_delta
            beta_0 = (self.theta_r / self.sigma_r**2) * (2 * np.log(1 - (self.ksi_r - self.kappa_r) * (1 - np.exp(-self.ksi_r * tau)) / (2 * self.ksi_r))
                                                         + (self.ksi_r - self.kappa_r) * tau) \
                     + (self.sigma_delta**2 * tau) / (2 * self.kappa_delta**2) \
                     - (self.sigma_s * self.sigma_delta * self.rho_1 + self.theta_delta) * tau / self.kappa_delta \
                     - (self.sigma_s * self.sigma_delta * self.rho_1 + self.theta_delta) * np.exp(-self.kappa_delta * tau) / self.kappa_delta**2 \
                     + (4 * self.sigma_delta**2 * np.exp(-self.kappa_delta * tau) - self.sigma_delta**2 * np.exp(-2 * self.kappa_delta * tau)) / (4 * self.kappa_delta**3) \
                     + (self.sigma_s * self.sigma_delta * self.rho_1 + self.theta_delta) / self.kappa_delta**2 \
                     - 3 * self.sigma_delta**2 / (4 * self.kappa_delta**3)
    
            # Compute the forward price F_tk
            F_tk = np.exp(np.log(self.S_t) + self.seasonal_factors[k] + beta_0 + beta_r * self.r_t + beta_delta * self.delta_t)
    
            # Store the computed futures price
            futures_list[k] = F_tk
    
        return futures_list

    def compute_episode_riv(self):
        """
        Compute Rolling Intrinsic + Extrinsic Value (RIV) using LP at decision times.
        Returns scalar RIV.
        """
        decision_times = np.arange(0, len(self.F_trajectory), 30)
        n = self.n_months
        V_t = self.V_0
        # print('V_t; ',V_t)
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
                # Compute initial intrinsic value V_I,0 = -F_0 * X_0 for all maturities
                prices = self.F_trajectory[tau].copy()
                X_tau[i] = linprog(prices, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds,  method="highs").x 
                continue

            prev_tau = decision_times[i-1]
            # Compute realized P/L from futures position for each maturity
            CF += np.dot((self.F_trajectory[tau] - self.F_trajectory[prev_tau]) , X_tau[i-1])
            prices = self.F_trajectory[tau].copy()
            zero_price_indices = (prices == 0)
            adjusted_bounds = [(0, 0) if zero_price_indices[k] else bounds[k] for k in range(len(prices))]
            if i < 12:
                X_tau[i] = linprog(prices, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=adjusted_bounds, method="highs").x 
            else:
                X_tau[i] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -V_t])
                CF += -self.F_trajectory[tau][i-1] * X_tau[i, i-1]
            
            V_t += X_tau[i, i-1]
            b = np.hstack([(V_max - V_t) * np.ones(n), -(V_min - V_t) * np.ones(n)])
            b_eq = np.array([V_T - V_t])
        return CF

    def reset(self):
        self.month = 0
        self.day = 0
        
        self.S_t = self.initial_spot_price
        self.r_t = self.initial_r
        self.delta_t = self.initial_delta
        self.v_t = self.initial_v
        self.F_t = self.compute_futures_curve()
        
        self.S_trajectory = [self.S_t]
        self.r_trajectory = [self.r_t]
        self.delta_trajectory = [self.delta_t]
        self.v_trajectory = [self.v_t]
        self.F_trajectory = [self.F_t]
                
        self.V_t = self.V_0

        self.rl_cumulative_reward = 0.0
        
        return np.concatenate(([self.month], self.F_t, [self.V_t]), dtype=np.float32), {}

    def step(self, action):
        
        # Validate action dimension
        assert len(action) == self.n_months, "Action must have length = n_months"

        # last_action = -self.V_t - action.cumsum()[-1]
        # action = np.concatenate((action, np.array([last_action], dtype=np.float32)))
        
        is_terminal = False
        is_truncated = False
        
        # # ---- APPLY ACTION AND CHECK CONSTRAINTS ----
        # running_sum = 0.0
        # for i in range(max(self.month - 1, 0), self.n_months): # Ensure valid range
        #     # Step 1 (τ_1): Full vector X_0 = [X_0^1, ..., X_0^12]
        #     if self.month == 0:
        #         running_sum += action[i]
        #         new_volume = self.V_t + running_sum  # Here self.V_t == self.V_0
        #     else:
        #         running_sum += action[i]
        #         new_volume = self.V_t + running_sum  # Update storage


        #     # ---- HARD REJECT: Cumulative constraint violation ----
        #     if new_volume < self.V_min or new_volume > self.V_max:
        #         self.CF -= 10#1_000_000.0  # Huge negative penalty
        #         is_truncated = True
        #         return np.concatenate(([self.month+1], self.F_t, [self.V_t]), dtype=np.float32), self.CF, is_terminal, is_truncated, {}
        #     # Except for the first loop, updating V_t: V_t = V_{t-1} + X_{t}^t   
        #     if self.month != 0 and i == max(self.month - 1, 0):
        #         self.V_t = new_volume
        #         running_sum = 0

        # ---- APPLY ACTION AND CHECK CONSTRAINTS ----
        running_sum = 0.0
        cost1 = 0.0
        cost2 = 0.0
        riv_penalty = 0.0
        reward = 0.0
        for i in range(max(self.month - 1, 0), self.n_months): # Ensure valid range
            running_sum += action[i]
            new_volume = self.V_t + running_sum  # Update storage

            # ---- SOFT REJECT: Cumulative constraint violation ----
            if new_volume + 0.001 < self.V_min or new_volume - 0.001 > self.V_max:
                # print("new_volume: ",new_volume)
                # print("min(new_volume - self.V_min, self.V_max - new_volume): ",min(new_volume - self.V_min, self.V_max - new_volume))
                cost1 += min(new_volume - self.V_min, self.V_max - new_volume) * self.penalty_lambda1
            # Except for the first loop, updating V_t: V_t = V_{t-1} + X_{t}^t   
            if self.month != 0 and i == max(self.month - 1, 0):
                self.V_t = new_volume
                running_sum = 0
        # Computing the 12th action (dependent action) and adding penalties for the last dependent action decision
        # last_action = np.clip(-self.V_t, -self.W_max, self.I_max)
        
        whole_volume = new_volume

        # ---- FINAL STORAGE BALANCE CONSTRAINT (SOFT PENALTY) ----
        # if abs(self.V_t + last_action) > 0.005: # If balance is not exactly zero
        if abs(whole_volume) > 0.001: # If balance is not exactly zero
            # print('cost2', whole_volume)
            # cost2 += -abs(self.V_t + last_action) * self.penalty_lambda2  # Soft penalty
            cost2 += -abs(whole_volume) * self.penalty_lambda2  # Soft penalty
            
        reward += cost1
        reward += cost2
        # if abs(total_balance) > 1e-7:  # If balance is not exactly zero
            # cost2 = -abs(total_balance) * self.penalty_lambda2  # Soft penalty
            # reward += cost2  # Apply penalty to cumulative cost
                    

        # # Update storage level:
        # if self.month != 0:
            
        #     chosen_action = action[self.month - 1]
            
        #     # Update storage volume
        #     old_V_t = self.V_t
        #     self.V_t += chosen_action
        #     #self.CF_S += - self.F_t[self.month -1] * action[self.month - 1] #This is the spot cash-flow from gas purchases or sales

        # Store state at time t
        month = self.month
        F_t = self.F_t
        V_t = self.V_t

        for i in range(30):
            # Generate independent Brownian increments
            dW_1 = self.W.normal(0, np.sqrt(self.dt))  # For dW_1
            dW_r = self.W.normal(0, np.sqrt(self.dt))  # For dW_r (interest rate)
            dW_2 = self.W.normal(0, np.sqrt(self.dt))  # For dW_2
            dW_delta = self.rho_1 * dW_1 + np.sqrt(1 - self.rho_1 ** 2) * self.W.normal(0, np.sqrt(self.dt))  # For dW_delta (correlated with dW_1)
            dW_v = self.rho_2 * dW_2 + np.sqrt(1 - self.rho_2 ** 2) * self.W.normal(0, np.sqrt(self.dt))  # For dW_v (correlated with dW_2)
            
            # Probability of jump occurrence
            dq = self.W.choice([0, 1], p=[1 - self.lam * self.dt, self.lam * self.dt])
    
            # Jump magnitude: ln(1 + J) ~ N[ln(1 + mu_J) - 0.5 * sigma_J^2, sigma_J^2]
            ln_1_plus_J = self.W.normal(np.log(1 + self.mu_j) - 0.5 * self.sigma_j ** 2, self.sigma_j)
            J = np.exp(ln_1_plus_J) - 1  # Jump size for the spot price

            J_v = self.W.exponential(scale=self.theta)
    
            # Stochastic differential equations (SDEs)
            # dS_t = (r_t - delta_t - \lambda \mu_J)S_t dt + \sigma_s S_t dW_1 + \sqrt{V_t} S_t dW_2 + J S_t dq
            dS_t = (self.r_t - self.delta_t - self.lam * self.mu_j) * self.S_t * self.dt + self.sigma_s * self.S_t * dW_1 + np.sqrt(max(self.v_t,0)) * self.S_t * dW_2 + J * self.S_t * dq
            self.S_t += dS_t
            self.S_trajectory.append(self.S_t)
            
            # dr_t = (\theta_r - \kappa_r r_t) dt + \sigma_r \sqrt{r_t} dW_r
            dr_t = (self.theta_r - self.kappa_r * self.r_t) * self.dt + self.sigma_r * np.sqrt(max(self.r_t, 0)) * dW_r
            self.r_t += dr_t
            self.r_trajectory.append(self.r_t)
    
            # ddelta_t = (\theta_delta - \kappa_delta \delta_t) dt + \sigma_delta dW_delta
            ddelta_t = (self.theta_delta - self.kappa_delta * self.delta_t) * self.dt + self.sigma_delta * dW_delta
            self.delta_t += ddelta_t
            self.delta_trajectory.append(self.delta_t)
    
            # dv_t = (\theta_v - \kappa_v v_t) dt + \sigma_v \sqrt{v_t} dW_v + J_v dq
            dv_t = (self.theta_v - self.kappa_v * self.v_t) * self.dt + self.sigma_v * np.sqrt(max(self.v_t, 0)) * dW_v + J_v * dq
            self.v_t += dv_t
            self.v_trajectory.append(self.v_t)  

            self.day += 1
            self.F_t = self.compute_futures_curve()
            self.F_trajectory.append(self.F_t)
        self.month += 1
        reward += np.dot((self.F_t - F_t), action)
        # self.CF = self.CF_S + self.CF_F
        # Check termination condition
        self.rl_cumulative_reward += reward
        is_terminal = False
        is_truncated = False
        if self.month == 12: 
            is_terminal = True
            #last_action = np.clip(-self.V_t, -self.W_max, self.I_max)
            # reward += - self.F_t[-1] * action[-1]
            reward += - self.F_t[-1] * action[-1]
            self.rl_cumulative_reward += reward
            self.V_t += action[-1]
            riv = self.compute_episode_riv()
            if self.rl_cumulative_reward < riv:
                riv_penalty =  -self.penalty_lambda_riv * (riv - self.rl_cumulative_reward)
                reward += riv_penalty
            # new_volume = self.V_t + last_action  # Update storage
            # if new_volume + 1e-7 < self.V_min or new_volume - 1e-7 > self.V_max:
            #     cost1 += min(new_volume - self.V_min, self.V_max - new_volume) * self.penalty_lambda1
            #     reward += cost1
            # self.V_t = new_volume
            # if abs(self.V_t) > 1e-7:  # If balance is not exactly zero
            #     cost2 = -abs(self.V_t) * self.penalty_lambda2  # Soft penalty
            #     reward += cost2  # Apply penalty to cumulative cost
            # self.month += 1
            # self.F_t = np.full(12, 0.0, dtype=np.float32)
            # return np.concatenate(([self.month], self.F_t, [self.V_t]), dtype=np.float32), reward, is_terminal, is_truncated, info
        info = {'cost1':cost1, 'cost2':cost2, 'riv_penalty': riv_penalty}
        return np.concatenate(([self.month], self.F_t, [self.V_t]), dtype=np.float32), reward, is_terminal, is_truncated, info

class FCDPAutoregressive(nn.Module):
    def __init__(self, 
                 input_dim,
                 action_bounds,
                 hidden_dims=(512, 512, 256, 128), 
                 activation_fc=F.leaky_relu):
        super(FCDPAutoregressive, self).__init__()
        self.activation_fc = activation_fc
        self.env_min, self.env_max = action_bounds

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.LayerNorm(hidden_dims[i + 1])
            ))

        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dims[-1] + 1, 1) for _ in range(len(self.env_max) - 1)
        ])

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.to(self.device)

        self.env_min = torch.tensor(self.env_min, device=self.device, dtype=torch.float32)
        self.env_max = torch.tensor(self.env_max, device=self.device, dtype=torch.float32)

        self.V_min = torch.tensor(0.0, device=self.device)
        self.V_max = torch.tensor(1.0, device=self.device)
        self.I_max = torch.tensor(0.4, device=self.device)
        self.W_max = torch.tensor(0.4, device=self.device)

    def _normalize(self, state):
        state = state.clone()
        state[:, 0] /= 11.0  # Normalize month
        state[:, 1:13] /= torch.clamp(state[:, 1:13].max(dim=1, keepdim=True).values, min=1e-6)  # Normalize prices
        state[:, -1] /= self.V_max  # Normalize V_t
        return state

    def _format(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        return self._normalize(state)

    def forward(self, state):
        state = self._format(state)
        batch_size = state.size(0)
        t = (state[:, 0] * 11).long()
        V_t = state[:, -1] * self.V_max  # De-normalize

        x = self.activation_fc(self.input_layer(state))
        for layer in self.hidden_layers:
            x = self.activation_fc(layer(x))

        actions = []
        cum_V = V_t.clone()

        for j in range(11):
            mask = (j + 1 >= t).float().unsqueeze(1)
            upper_volume = torch.minimum((12 - j - 1) * self.I_max, self.V_max) - cum_V
            lower_volume = self.V_min - cum_V

            final_min = torch.maximum(lower_volume, -self.W_max)
            final_max = torch.minimum(upper_volume, self.I_max)

            xj_input = torch.cat([x, cum_V.unsqueeze(1)], dim=1)
            raw_output = self.output_heads[j](xj_input).squeeze(1)
            output = torch.tanh(raw_output)
            scaled = (output + 1) / 2 * (final_max - final_min) + final_min
            bounded = scaled * mask.squeeze(1)
            actions.append(bounded)
            cum_V += bounded

        x12 = -V_t - torch.stack(actions, dim=1).sum(dim=1)
        x12 = torch.clamp(x12, min=-self.W_max, max=self.I_max)
        actions.append(x12)

        return torch.stack(actions, dim=1)

def compute_futures_curve(day, S_t, r_t, delta_t):
    futures_list = np.full((N_simulations,12), 0.0, dtype=np.float32)  # Initialize all values as 0.0
    remaining_futures = max(12 - (day // 30), 0)  # Shrinks every 30 days
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
N_simulations = 5000 # Number of simulations
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
N_simulations = 5000  # Number of simulations
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
X_tau_Rolling = np.zeros((N_simulations, len(decision_times), N_maturities))

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
            X_tau_Rolling[j, i, :] = linprog(prices, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds,  method="highs").x 
            V_I_Rolling[j, i] = -np.dot(prices , X_tau_Rolling[j, i, :]) 
            continue
            
        prev_tau = decision_times[i-1]
        
        # Compute realized P/L from futures position for each maturity
        CF += np.dot((F_t[j, tau, :] - F_t[j, prev_tau, :]) , X_tau_Rolling[j, i-1, :])

        # Update intrinsic value for each maturity
        prices = F_t[j, tau, :]
        # Identify zero-price variables
        zero_price_indices = (prices == 0)
        # Modify bounds: Force zero-price variables to be zero (fix them)
        adjusted_bounds = [(0, 0) if zero_price_indices[i] else bounds[i] for i in range(len(prices))]
        
        if i < 12:
            X_tau_Rolling[j, i, :] = linprog(prices, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=adjusted_bounds, method="highs").x 
        else:
            X_tau_Rolling[j, i, :] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -V_t])
            CF += -F_t[j, tau, i-1] * X_tau_Rolling[j, i, i-1]

        V_I_Rolling[j, i] = -np.dot(prices , X_tau_Rolling[j, i, :])
        
        V_t += X_tau_Rolling[j, i, i-1]
        b = np.hstack([(V_max - V_t) * np.ones(n), -(V_min - V_t) * np.ones(n)])
        b_eq = np.array([V_T - V_t])

        # Compute spot cash-flow using front-month futures (maturity=0)
        # CF += -F_t[j, tau, i-1] * X_tau_Rolling[j, i, i-1]
            
    # Update total cash-flows for this simulation
    CF_IE_Rolling[j] = CF

# Compute estimated total intrinsic + extrinsic value
V_IE_Rolling = np.mean(CF_IE_Rolling)

# Display results
print(f"Estimated Rolling Intrinsic + Extrinsic Value: {V_IE_Rolling:.4f}")

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
    'sigma_delta': 0.103663918307669, #1.03663918307669,
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
    'penalty_lambda1': 10,#0.2,#2.0,#0.2,#10.0,
    'penalty_lambda2': 50.,#1,#10.0,#1.0,#50.0,
    'penalty_lambda_riv': 10, #5.0,
    'monthly_seasonal_factors': np.array([-0.106616824924423, -0.152361004102492, -0.167724706188117, -0.16797984045645,
                                 -0.159526180248348, -0.13927943487493, -0.0953402986114613, -0.0474646801238288, 
                                 -0.0278622280543003, 0.000000, -0.00850263509128089, -0.0409638719325969])
}
env = TTFGasStorageEnv(params)
trained_network = FCDPAutoregressive(env.observation_space.shape[0], (env.action_space.low, env.action_space.high), hidden_dims=(512, 512, 256, 128))
# trained_network.load_state_dict(torch.load("fcdp_actor_ri.pth"))
trained_network.load_state_dict(torch.load("online_policy_model_autoregressive_penalized.pth"))
# trained_network.to(device)
trained_network.eval()

# DRL Valuation
N_simulations = 5000 # Number of simulations
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
            state = np.concatenate((np.array([i]),prices,np.array([V_t])),dtype=np.float32)
            # env.month = state[0]
            # env.V_t = state[-1]
            # X_tau[j, i, :] = best_agent.evaluation_strategy.select_action(agent.online_policy_model, state)
            # X_tau[j, i, :] = linprog(prices, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds,  method="highs").x
            X_tau[j, i, :] = trained_network(state).detach().cpu().numpy()
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
        # X_tau[j, i, :] = best_agent.evaluation_strategy.select_action(best_agent.online_policy_model, state)
        X_tau[j, i, :] = trained_network(state).detach().cpu().numpy()
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

# Mix Valuation
N_simulations = 5000 # Number of simulations
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
V_I_Mix = np.zeros((N_simulations, len(decision_times)))  # Intrinsic value per maturity
CF_IE_Mix = np.zeros(N_simulations)  # Total cash-flow per simulation
X_tau_Mix = np.zeros((N_simulations, len(decision_times), N_maturities))
mix_dataset = []
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
    if CF_IE[j] > CF_IE_Rolling[j]:
        for i, tau in enumerate(decision_times):
            if i == 0:
                # Compute initial intrinsic value V_I,0 = -F_0 * X_0 for all maturities
                prices = F_t[j, tau, :]
                state = np.concatenate((np.array([i]),prices,np.array([V_t])),dtype=np.float32)
                # env.month = state[0]
                # env.V_t = state[-1]
                # X_tau[j, i, :] = best_agent.evaluation_strategy.select_action(agent.online_policy_model, state)
                # X_tau[j, i, :] = linprog(prices, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds,  method="highs").x
                X_tau_Mix[j, i, :] = trained_network(state).detach().cpu().numpy()
                # X_tau[j, i, -1] = np.clip(-X_tau[j, i, :-1].cumsum()[-1]-V_t,-W_max, I_max)
                X_tau_Mix[j, i, :] = np.round(X_tau_Mix[j, i, :],2)
                # mix_dataset.append((state.copy(), X_tau_Mix[j, i, :].copy()))
                # X_tau[j, i, :] = trained_network(state).detach().cpu().numpy()
                V_I_Mix[j, i] = -np.dot(prices , X_tau_Mix[j, i, :])
                continue
                
            prev_tau = decision_times[i-1]
            # Compute realized P/L from futures position for each maturity
            CF += np.dot((F_t[j, tau, :] - F_t[j, prev_tau, :]) , X_tau_Mix[j, i-1, :])
    
            # Update intrinsic value for each maturity
            prices = F_t[j, tau, :]
            state = np.concatenate((np.array([i]),prices,np.array([V_t])),dtype=np.float32)
            # env.month = state[0]
            # env.V_t = state[-1]
            # X_tau[j, i, :] = best_agent.evaluation_strategy.select_action(best_agent.online_policy_model, state)
            X_tau_Mix[j, i, :] = trained_network(state).detach().cpu().numpy()
            # X_tau[j, i, -1] = np.clip(-X_tau[j, i, :-1].cumsum()[-1]-V_t,-W_max, I_max)
            X_tau_Mix[j, i, :] = np.round(X_tau_Mix[j, i, :],2)
            # if i < 12:
            #     X_tau[j, i, :] = env.rescale_actions(agent.evaluation_strategy.select_action(agent.online_policy_model, state))
            #     # X_tau[j, i, :] = trained_network(state).detach().cpu().numpy()
            # else:
            #     X_tau[j, i, :] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, np.clip(-V_t,-W_max,I_max)])
            if i == 12:           
                X_tau_Mix[j, i, :] = np.zeros(12,dtype=np.float32)
                X_tau_Mix[j, i, -1] = X_tau_Mix[j, i-1, -1]
                X_tau_Mix[j, i, :] = np.round(X_tau_Mix[j, i, :],2)
                # print("X_tau[j, i]:  ",np.round(X_tau[j, i],2))
                CF += -F_t[j, tau, i-1] * X_tau_Mix[j, i-1, i-1] 
                V_I_Mix[j, i] = -np.dot(prices , X_tau_Mix[j, i, :])
            mix_dataset.append((state.copy(), X_tau_Mix[j, i, :].copy()))
            V_I_Mix[j, i] = -np.dot(prices , X_tau_Mix[j, i, :])
            
            V_t += np.round(X_tau_Mix[j, i, i-1],2)
            # b = np.hstack([(V_max - V_t) * np.ones(n), -(V_min - V_t) * np.ones(n)])
            # b_eq = np.array([V_T - V_t])
    
            # Compute spot cash-flow using front-month futures (maturity=0)
            # CF += -F_t[j, tau, i-1] * X_tau[j, i, i-1]
            
    else:
        for i, tau in enumerate(decision_times):
            if i == 0:
                # Compute initial intrinsic value V_I,0 = -F_0 * X_0 for all maturities
                prices = F_t[j, tau, :]
                X_tau_Mix[j, i, :] = linprog(prices, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=bounds,  method="highs").x 
                V_I_Mix[j, i] = -np.dot(prices , X_tau_Mix[j, i, :]) 
                continue
                
            prev_tau = decision_times[i-1]
            
            # Compute realized P/L from futures position for each maturity
            CF += np.dot((F_t[j, tau, :] - F_t[j, prev_tau, :]) , X_tau_Mix[j, i-1, :])
    
            # Update intrinsic value for each maturity
            prices = F_t[j, tau, :]
            # Identify zero-price variables
            zero_price_indices = (prices == 0)
            # Modify bounds: Force zero-price variables to be zero (fix them)
            adjusted_bounds = [(0, 0) if zero_price_indices[i] else bounds[i] for i in range(len(prices))]
            
            if i < 12:
                X_tau_Mix[j, i, :] = linprog(prices, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=adjusted_bounds, method="highs").x 
            else:
                X_tau_Mix[j, i, :] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -V_t])
                CF += -F_t[j, tau, i-1] * X_tau_Mix[j, i, i-1]

            mix_dataset.append((state.copy(), X_tau_Mix[j, i, :].copy()))
    
            V_I_Mix[j, i] = -np.dot(prices , X_tau_Mix[j, i, :])
            
            V_t += X_tau_Mix[j, i, i-1]
            b = np.hstack([(V_max - V_t) * np.ones(n), -(V_min - V_t) * np.ones(n)])
            b_eq = np.array([V_T - V_t])
    
            # Compute spot cash-flow using front-month futures (maturity=0)
            # CF += -F_t[j, tau, i-1] * X_tau_Rolling[j, i, i-1]
    # Update total cash-flows for this simulation
    CF_IE_Mix[j] = CF

# Compute estimated total intrinsic + extrinsic value
V_IE_Mix = np.mean(CF_IE_Mix)

# Display results
print(f"Estimated Rolling Intrinsic + Extrinsic Value: {V_IE_Mix:.4f}")

states = torch.tensor([s for s, _ in mix_dataset], dtype=torch.float32)
actions = torch.tensor([a for _, a in mix_dataset], dtype=torch.float32)

batch_size = 100
dataset = TensorDataset(states, actions)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

nS = env.observation_space.shape[0]
action_bounds = env.action_space.low, env.action_space.high
actor = FCDPAutoregressive(nS, action_bounds, hidden_dims=(512, 512, 256, 128)) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
actor.to(device)

actor.train()
# optimizer = optim.AdamW(actor.parameters(), lr=1e-3)
optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)  # Halve LR every 100 epochs
loss_fn = nn.MSELoss()

for epoch in range(600):
    total_loss = 0.0
    for batch_states, batch_actions in loader:
        batch_states = batch_states.to(device)
        batch_actions = batch_actions.to(device)

        optimizer.zero_grad()
        pred_actions = actor(batch_states)
        # loss = loss_fn(pred_actions, batch_actions)
        weights = batch_actions.abs().mean(dim=1, keepdim=True) + 1e-2
        loss = ((pred_actions - batch_actions)**2 * weights).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    scheduler.step()  # Update learning rate
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(actor.state_dict(), "fcdp_actor_mix.pth")