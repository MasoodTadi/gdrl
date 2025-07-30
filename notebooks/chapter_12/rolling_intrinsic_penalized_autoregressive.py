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

import datetime

LEAVE_PRINT_EVERY_N_SECS = 300
ERASE_LINE = '\x1b[2K'
EPS = 1e-6
BEEP = lambda: os.system("printf '\a'")
RESULTS_DIR = os.path.join('..', 'results')

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

class GreedyStrategy():
    def __init__(self, bounds):
        self.low, self.high = bounds
        self.ratio_noise_injected = 0

    def select_action(self, model, state):
        with torch.no_grad():
            if hasattr(model, "set_noise_active"):
                model.set_noise_active(False)
            # if state[0]==0:
            #     greedy_action = np.array([0. , 0. ,  0.4,  0.4,  0.2, 0. , 0. , 0. , -0.2, -0.4, -0.4, 0.], dtype=np.float32)
            # else:
            greedy_action = model(state).cpu().detach().numpy().squeeze()

        action = np.clip(greedy_action, self.low, self.high)
        return np.reshape(action, self.high.shape)

class FCQV(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 hidden_dims=(512, 512, 256, 128),
                 activation_fc=F.leaky_relu):
        super(FCQV, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.input_norm = nn.LayerNorm(hidden_dims[0])
        
        self.hidden_layers = nn.ModuleList()
        self.hidden_norms = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            in_dim = hidden_dims[i]
            if i == 0: 
                in_dim += output_dim
            hidden_layer = nn.Linear(in_dim, hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
            self.hidden_norms.append(nn.LayerNorm(hidden_dims[i + 1]))
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)
    
    def _format(self, state, action):
        x, u = state, action
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, 
                             device=self.device, 
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, 
                             device=self.device, 
                             dtype=torch.float32)
            u = u.unsqueeze(0)
        return x, u

    def forward(self, state, action):
        x, u = self._format(state, action)
        x = self.activation_fc(self.input_norm(self.input_layer(x)))
        for i, hidden_layer in enumerate(self.hidden_layers):
            if i == 0:
                x = torch.cat((x, u), dim=1)
            x = self.activation_fc(hidden_layer(x))
        return self.output_layer(x)
    
    def load(self, experiences):
        states, actions, rewards, new_states, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, rewards, new_states, is_terminals

# class FCDPAutoregressive(nn.Module):
#     def __init__(self, 
#                  input_dim,
#                  action_bounds,
#                  hidden_dims=(32, 32), 
#                  activation_fc=F.relu):
#         super(FCDPAutoregressive, self).__init__()
#         self.activation_fc = activation_fc
#         self.env_min, self.env_max = action_bounds

#         self.input_layer = nn.Linear(input_dim, hidden_dims[0])
#         self.hidden_layers = nn.ModuleList()
#         for i in range(len(hidden_dims) - 1):
#             self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

#         # One output head per action dimension
#         self.output_heads = nn.ModuleList([
#             nn.Linear(hidden_dims[-1] + 1, 1) for _ in range(len(self.env_max) - 1)
#         ])

#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.device = torch.device(device)
#         self.to(self.device)

#         self.env_min = torch.tensor(self.env_min, device=self.device, dtype=torch.float32)
#         self.env_max = torch.tensor(self.env_max, device=self.device, dtype=torch.float32)

#         self.V_min = torch.tensor(0.0, device=self.device)
#         self.V_max = torch.tensor(1.0, device=self.device)
#         self.I_max = torch.tensor(0.4, device=self.device)
#         self.W_max = torch.tensor(0.4, device=self.device)

#     def _format(self, state):
#         if not isinstance(state, torch.Tensor):
#             state = torch.tensor(state, dtype=torch.float32, device=self.device)
#         if state.ndim == 1:
#             state = state.unsqueeze(0)
#         return state

#     def forward(self, state):
#         state = self._format(state)
#         batch_size = state.size(0)
#         t = state[:, 0].long()
#         V_t = state[:, -1].clone()

#         # Shared base network
#         x = self.activation_fc(self.input_layer(state))
#         for hidden_layer in self.hidden_layers:
#             x = self.activation_fc(hidden_layer(x))

#         actions = []
#         cum_V = V_t.clone()

#         for j in range(11):
#             mask = (j + 1 >= t).float().unsqueeze(1)
#             upper_volume = torch.minimum((12 - j -1) * self.I_max, self.V_max) - cum_V
#             lower_volume = self.V_min - cum_V

#             # Combine with box constraints
#             final_min = torch.maximum(lower_volume, -self.W_max)
#             final_max = torch.minimum(upper_volume, self.I_max)

#             xj_input = torch.cat([x, cum_V.unsqueeze(1)], dim=1)
#             raw_output = self.output_heads[j](xj_input).squeeze(1)
#             output = torch.tanh(raw_output)  # in [-1, 1]
#             scaled = (output + 1) / 2 * (final_max - final_min) + final_min
#             bounded = scaled * mask.squeeze(1)
#             actions.append(bounded)
#             cum_V += bounded

#         # Compute last action to satisfy equality constraint
#         x12 = -V_t - torch.stack(actions, dim=1).sum(dim=1)
#         x12 = torch.clamp(x12, min=-self.W_max, max=self.I_max)
#         actions.append(x12)

#         return torch.stack(actions, dim=1)

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
        
class PrioritizedReplayBuffer():
    def __init__(self, 
                 max_samples=10000, 
                 batch_size=64, 
                 rank_based=False,
                 alpha=0.6, 
                 beta0=0.1, 
                 beta_rate=0.99992):
        self.max_samples = max_samples
        self.memory = np.empty(shape=(self.max_samples, 2), dtype=np.ndarray)
        self.batch_size = batch_size
        self.n_entries = 0
        self.next_index = 0
        self.td_error_index = 0
        self.sample_index = 1
        self.rank_based = rank_based # if not rank_based, then proportional
        self.alpha = alpha # how much prioritization to use 0 is uniform (no priority), 1 is full priority
        self.beta = beta0 # bias correction 0 is no correction 1 is full correction
        self.beta0 = beta0 # beta0 is just beta's initial value
        self.beta_rate = beta_rate

    def update(self, idxs, td_errors):
        self.memory[idxs, self.td_error_index] = np.abs(td_errors)
        if self.rank_based:
            sorted_arg = self.memory[:self.n_entries, self.td_error_index].argsort()[::-1]
            self.memory[:self.n_entries] = self.memory[sorted_arg]

    def store(self, sample):
        priority = 1.0
        if self.n_entries > 0:
            priority = self.memory[
                :self.n_entries, 
                self.td_error_index].max()
        self.memory[self.next_index, 
                    self.td_error_index] = priority
        self.memory[self.next_index, 
                    self.sample_index] = np.array(sample, dtype=object)
        self.n_entries = min(self.n_entries + 1, self.max_samples)
        self.next_index += 1
        self.next_index = self.next_index % self.max_samples

    def _update_beta(self):
        self.beta = min(1.0, self.beta * self.beta_rate**-1)
        return self.beta

    def sample(self, batch_size=None):
        batch_size = self.batch_size if batch_size == None else batch_size
        self._update_beta()
        entries = self.memory[:self.n_entries]

        if self.rank_based:
            priorities = 1/(np.arange(self.n_entries) + 1)
        else: # proportional
            priorities = entries[:, self.td_error_index] + EPS
        scaled_priorities = priorities**self.alpha        
        probs = np.array(scaled_priorities/np.sum(scaled_priorities), dtype=np.float64)

        weights = (self.n_entries * probs)**-self.beta
        normalized_weights = weights/weights.max()
        idxs = np.random.choice(self.n_entries, batch_size, replace=False, p=probs)
        samples = np.array([entries[idx] for idx in idxs])
        
        samples_stacks = [np.vstack(batch_type) for batch_type in np.vstack(samples[:, self.sample_index]).T]
        idxs_stack = np.vstack(idxs)
        weights_stack = np.vstack(normalized_weights[idxs])
        return idxs_stack, weights_stack, samples_stacks

    def __len__(self):
        return self.n_entries
    
    def __repr__(self):
        return str(self.memory[:self.n_entries])
    
    def __str__(self):
        return str(self.memory[:self.n_entries])

class DDPG():
    def __init__(self, 
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
                 update_target_every_steps,
                 tau):
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
        self.update_target_every_steps = update_target_every_steps
        self.tau = tau

    # def optimize_model(self, experiences):
    #     states, actions, rewards, next_states, is_terminals = experiences
    #     batch_size = len(is_terminals)

    #     argmax_a_q_sp = self.target_policy_model(next_states)
    #     max_a_q_sp = self.target_value_model(next_states, argmax_a_q_sp)
    #     target_q_sa = rewards + self.gamma * max_a_q_sp * (1 - is_terminals)
    #     q_sa = self.online_value_model(states, actions)
    #     td_error = q_sa - target_q_sa.detach()
    #     value_loss = td_error.pow(2).mul(0.5).mean()
    #     self.value_optimizer.zero_grad()
    #     value_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.online_value_model.parameters(), 
    #                                    self.value_max_grad_norm)
    #     self.value_optimizer.step()

    #     argmax_a_q_s = self.online_policy_model(states)
    #     max_a_q_s = self.online_value_model(states, argmax_a_q_s)
    #     policy_loss = -max_a_q_s.mean()
    #     self.policy_optimizer.zero_grad()
    #     policy_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.online_policy_model.parameters(), 
    #                                    self.policy_max_grad_norm)        
    #     self.policy_optimizer.step()

    def optimize_model(self, idxs_weights_samples):
        idxs, weights, experiences = idxs_weights_samples
        states, actions, rewards, next_states, is_terminals = experiences
        batch_size = len(is_terminals)

        argmax_a_q_sp = self.target_policy_model(next_states)
        max_a_q_sp = self.target_value_model(next_states, argmax_a_q_sp)
        target_q_sa = rewards + self.gamma * max_a_q_sp * (1 - is_terminals)
        q_sa = self.online_value_model(states, actions)
        td_error = q_sa - target_q_sa.detach()
        weights = torch.tensor(weights, device=td_error.device, dtype=td_error.dtype)
        value_loss = (td_error.pow(2) * weights).mean()  # apply importance weights
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_value_model.parameters(), 
                                       self.value_max_grad_norm)
        self.value_optimizer.step()

        argmax_a_q_s = self.online_policy_model(states)
        max_a_q_s = self.online_value_model(states, argmax_a_q_s)
        policy_loss = -max_a_q_s.mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_policy_model.parameters(), 
                                       self.policy_max_grad_norm)        
        self.policy_optimizer.step()

        # Update TD errors in buffer
        self.replay_buffer.update(idxs.squeeze(), td_error.detach().cpu().numpy().squeeze())


    def interaction_step(self, state, env):
        min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
        action = self.training_strategy.select_action(self.online_policy_model, 
                                                      state, 
                                                      len(self.replay_buffer) < min_samples)
        new_state, reward, is_terminal, is_truncated, info = env.step(action)
        #is_truncated = 'TimeLimit.truncated' in info and info['TimeLimit.truncated']
        is_failure = is_terminal and not is_truncated
        experience = (state, action, reward, new_state, float(is_failure))
        self.replay_buffer.store(experience)
        self.episode_reward[-1] += reward
        self.episode_timestep[-1] += 1
        self.episode_exploration[-1] += self.training_strategy.ratio_noise_injected
        #print("self.episode_exploration[-1]: ", self.episode_exploration[-1])
        return new_state, is_terminal
    
    def update_networks(self, tau=None):
        tau = self.tau if tau is None else tau
        for target, online in zip(self.target_value_model.parameters(), 
                                  self.online_value_model.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

        for target, online in zip(self.target_policy_model.parameters(), 
                                  self.online_policy_model.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

    #def train(self, make_env_fn, make_env_kargs, seed, gamma, 
    def train(self, env, seed, gamma,
              max_minutes, max_episodes, goal_mean_100_reward):
        training_start, last_debug_time = time.time(), float('-inf')

        # Safe and persistent checkpoint directory in home
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = os.path.expanduser(f"~/ddpg_checkpoints/run_{timestamp}")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
        print(f"Running on: {os.uname().nodename}")
        print(f"[INFO] Checkpoints will be saved to: {self.checkpoint_dir}")
        # self.checkpoint_dir = tempfile.mkdtemp()
        #print(self.checkpoint_dir)
        #self.make_env_fn = make_env_fn
        #self.make_env_kargs = make_env_kargs
        self.seed = seed
        self.gamma = gamma
        
        #env = self.make_env_fn(**self.make_env_kargs, seed=self.seed)
        torch.manual_seed(self.seed) ; np.random.seed(self.seed) ; random.seed(self.seed)
    
        nS, nA = env.observation_space.shape[0], env.action_space.shape[0]
        action_bounds = env.action_space.low, env.action_space.high
        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []        
        self.episode_exploration = []
        
        self.target_value_model = self.value_model_fn(nS, nA)
        self.online_value_model = self.value_model_fn(nS, nA)
        self.target_policy_model = self.policy_model_fn(nS, action_bounds)
        self.online_policy_model = self.policy_model_fn(nS, action_bounds)

        # Load pretrained actor weights into both online and target policy models
        pretrained_path = "fcdp_actor_ri.pth"  # path to your pretrained file
        self.online_policy_model.load_state_dict(torch.load(pretrained_path, map_location=self.online_policy_model.device))
        self.target_policy_model.load_state_dict(torch.load(pretrained_path, map_location=self.target_policy_model.device))

        self.update_networks(tau=1.0)
        self.value_optimizer = self.value_optimizer_fn(self.online_value_model, 
                                                       self.value_optimizer_lr)        
        self.policy_optimizer = self.policy_optimizer_fn(self.online_policy_model, 
                                                         self.policy_optimizer_lr)

        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = training_strategy_fn(action_bounds)
        # self.training_strategy = self.training_strategy_fn() # No action bounds here
        self.evaluation_strategy = evaluation_strategy_fn(action_bounds)
        # self.evaluation_strategy = self.evaluation_strategy_fn() # No action bounds here
                    
        result = np.empty((max_episodes, 5))
        result[:] = np.nan
        training_time = 0
        for episode in range(1, max_episodes + 1):
            episode_start = time.time()
            
            state, _ = env.reset()
            is_terminal = False
            self.episode_reward.append(0.0)
            self.episode_timestep.append(0.0)
            self.episode_exploration.append(0.0)
            for step in count():
                state, is_terminal = self.interaction_step(state, env)

                min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
                if len(self.replay_buffer) > min_samples:
                    # experiences = self.replay_buffer.sample()
                    # experiences = self.online_value_model.load(experiences)
                    # self.optimize_model(experiences)
                    idxs_weights_samples = self.replay_buffer.sample()
                    samples = self.online_value_model.load(idxs_weights_samples[2])
                    self.optimize_model((idxs_weights_samples[0], idxs_weights_samples[1], samples))

                if np.sum(self.episode_timestep) % self.update_target_every_steps == 0:
                    self.update_networks()

                if is_terminal:
                    gc.collect()
                    break
            
            # stats
            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed
            evaluation_score, _ = self.evaluate(self.online_policy_model, env)
            self.save_checkpoint(episode-1, self.online_policy_model)

            total_step = int(np.sum(self.episode_timestep))
            self.evaluation_scores.append(evaluation_score)
            
            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])
            mean_100_eval_score = np.mean(self.evaluation_scores[-100:])
            std_100_eval_score = np.std(self.evaluation_scores[-100:])
            lst_100_exp_rat = np.array(
                self.episode_exploration[-100:])/np.array(self.episode_timestep[-100:])
            mean_100_exp_rat = np.mean(lst_100_exp_rat)
            std_100_exp_rat = np.std(lst_100_exp_rat)
            
            wallclock_elapsed = time.time() - training_start
            result[episode-1] = total_step, mean_100_reward, \
                mean_100_eval_score, training_time, wallclock_elapsed
            
            reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS
            reached_max_minutes = wallclock_elapsed >= max_minutes * 60
            reached_max_episodes = episode >= max_episodes
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
                
        final_eval_score, score_std = self.evaluate(self.online_policy_model, env, n_episodes=100)
        wallclock_time = time.time() - training_start
        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(
                  final_eval_score, score_std, training_time, wallclock_time))
        env.close() ; del env
        self.get_cleaned_checkpoints()
        return result, final_eval_score, training_time, wallclock_time
    
    def evaluate(self, eval_policy_model, eval_env, n_episodes=1):
        rs = []
        for _ in range(n_episodes):
            s, _ = eval_env.reset()
            d = False
            rs.append(0)
            for _ in count():
                a = self.evaluation_strategy.select_action(eval_policy_model, s)
                s, r, d, _, _ = eval_env.step(a)
                rs[-1] += r
                if d: break
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

    def save_checkpoint(self, episode_idx, model):
        torch.save(model.state_dict(), 
                   os.path.join(self.checkpoint_dir, 'model.{}.tar'.format(episode_idx)))

class NormalNoiseStrategy:
    """
    Adds exponentially decaying Gaussian noise to actions in a DDPG agent for exploration.
    """
    def __init__(self, bounds, exploration_noise_ratio, final_noise_ratio,
                 max_episode, noise_free_last):
        self.low = np.array(bounds[0])
        self.high = np.array(bounds[1])
        self.action_range = self.high - self.low

        self.I_max = 0.4
        self.W_max = 0.4
        self.V_min = 0.0
        self.V_max = 1.0

        self.exploration_noise_ratio = exploration_noise_ratio
        self.final_noise_ratio = final_noise_ratio
        self.max_episode = max_episode
        self.noise_free_last = noise_free_last

        self.episodes_with_noise = max_episode - noise_free_last
        self.decay_rate = (final_noise_ratio / exploration_noise_ratio) ** (1 / self.episodes_with_noise)

        self.current_episode = 0
        self.ratio_noise_injected = 0.0  # for diagnostics

    def decay_step(self):
        """Call after each episode to increment the noise decay tracker."""
        self.current_episode += 1

    @property
    def noise_ratio(self):
        """Dynamically compute current noise level."""
        if self.current_episode >= self.episodes_with_noise:
            return 0.0
        return self.exploration_noise_ratio * (self.decay_rate ** self.current_episode)

    def _mask_action(self, action, state):
        """Masks actions based on problem-specific logic using F_t."""
        F_t = state[1:-1]
        mask = (F_t != 0)
        action[~mask] = 0.0
        return action, mask

    def _compute_noise_diagnostics(self, action, greedy_action, mask):
        if np.any(mask):
            ratio = np.abs((greedy_action[mask] - action[mask]) / self.action_range[mask])
            self.ratio_noise_injected = np.mean(ratio)
        else:
            self.ratio_noise_injected = 0.0

    def select_action(self, model, state, max_exploration=False):
        """
        Returns a noisy action using Gaussian noise scaled by the current noise ratio.
        """
        noise_scale = self.action_range if max_exploration else self.noise_ratio * self.action_range

        with torch.no_grad():
            greedy_action = model(state).cpu().detach().numpy().squeeze()
            if np.isnan(greedy_action).any():
                print("⚠️ Warning: NaN detected in greedy action")

        t = int(state[0])
        V_t = float(state[-1])
        action = []
        cum_V = V_t
        for j in range(11):
            if j < t - 1:
                a_j = 0.0
            else:
                upper_volume = min((12 - j - 1) * self.I_max, self.V_max) - cum_V
                lower_volume = self.V_min - cum_V
                upper_volume = min(upper_volume, self.I_max)
                lower_volume = max(lower_volume, -self.W_max)

                g_j = greedy_action[j]
                noise = np.random.normal(0.0, noise_scale[j])
                a_j = np.clip(g_j + noise, lower_volume, upper_volume)
                cum_V += a_j
            action.append(a_j)
        
        # Final action to satisfy equality constraint exactly
        a_12 = -V_t - sum(action)
        a_12 = np.clip(a_12, -self.W_max, self.I_max)
        action.append(a_12)
        action = np.array(action, dtype=np.float32)
        
        # noise = np.random.normal(loc=0.0, scale=noise_scale)
        # noisy_action = greedy_action + noise
        # clipped_action = np.clip(noisy_action, self.low, self.high)

        _, mask = self._mask_action(action, state)
        self._compute_noise_diagnostics(action, greedy_action, mask)

        # return final_action
        return action

# SEEDS = (34, 56, 78, 90)
SEEDS = (56, 78, 90)
# SEEDS = [90]
ddpg_results = []
best_agent, best_eval_score = None, float('-inf')
for seed in SEEDS:
    environment_settings = {
        'env_name': 'TTFGasStorageEnv',
        'gamma': 1.0,
        'max_minutes': np.inf,#20,
        'max_episodes': 25_000, #15_000,
        'goal_mean_100_reward': np.inf#4.1#-15#-150
    }

    # policy_model_fn = lambda nS, bounds: FCDPAutoregressive(nS, bounds, hidden_dims=(256,256)) 
    policy_model_fn = lambda nS, bounds: FCDPAutoregressive(nS, bounds, hidden_dims=(512, 512, 256, 128)) 
    policy_max_grad_norm = 1#float('inf')
    policy_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
    policy_optimizer_lr = 0.0003#0.0005#0.003

    value_model_fn = lambda nS, nA: FCQV(nS, nA, hidden_dims=(512, 512, 256, 128)) 
    value_max_grad_norm = 1#float('inf')
    value_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
    value_optimizer_lr = 0.0005#0.0003#0.0005#0.003
    training_strategy_fn = lambda bounds: NormalNoiseStrategy(bounds, exploration_noise_ratio=0.3, final_noise_ratio = 1e-6, max_episode=environment_settings['max_episodes'], 
                                                                                                                              noise_free_last=0.2 * environment_settings['max_episodes'])
    # training_strategy_fn = lambda: NormalNoiseStrategy(exploration_noise_ratio=0.1)
    evaluation_strategy_fn = lambda bounds: GreedyStrategy(bounds)
    # evaluation_strategy_fn = lambda: GreedyStrategy()

    # replay_buffer_fn = lambda: ReplayBuffer(max_size=100_000, batch_size=32) #max_size=100000
    replay_buffer_fn = lambda: PrioritizedReplayBuffer(max_samples=100_000, batch_size=32)
    n_warmup_batches = 1#200#5
    update_target_every_steps = 1
    tau = 0.005
    
    env_name, gamma, max_minutes, \
    max_episodes, goal_mean_100_reward = environment_settings.values()

    agent = DDPG(replay_buffer_fn,
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
                 update_target_every_steps,
                 tau)

    #make_env_fn, make_env_kargs = get_make_env_fn(env_name=env_name)
    # Example usage
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
        'penalty_lambda1': 10,#0.2,#2.0,#0.2,#10.0,
        'penalty_lambda2': 50.,#1,#10.0,#1.0,#50.0,
        'penalty_lambda_riv': 0.0, #5.0,
        'monthly_seasonal_factors': np.array([-0.106616824924423, -0.152361004102492, -0.167724706188117, -0.16797984045645,
                                     -0.159526180248348, -0.13927943487493, -0.0953402986114613, -0.0474646801238288, 
                                     -0.0278622280543003, 0.000000, -0.00850263509128089, -0.0409638719325969])
    }

    # params = {
    #     'n_months': 12,
    #     'V_min': 0,
    #     'V_max': 1,
    #     'V_0': 0,
    #     'W_max': 0.4,
    #     'I_max': 0.4,
    #     # 'storage_capacity': 100000,
    #     'kappa_r': 0.492828372105622,
    #     'sigma_r': 0.655898616135014,
    #     'theta_r': 0.000588276156660185,
    #     'kappa_delta': 1.17723166341479,
    #     'sigma_delta': 0.103663918307669, #1.03663918307669,
    #     'theta_delta': -0.213183673388138,
    #     'sigma_s': 0.791065501973918,
    #     'rho_1': 0.899944474373156,
    #     'rho_2': -0.306810849087325,
    #     'sigma_v': 0.825941396204049,
    #     'theta_v': 0.0505685591761352,
    #     'theta': 0.00640705687096142,
    #     'kappa_v': 2.36309244973169,
    #     'lam': 0.638842070975342,
    #     'sigma_j': 0.032046147726045,
    #     'mu_j': 0.0137146728855484,
    #     'seed': seed,
    #     'initial_spot_price': np.exp(2.9479),
    #     'initial_r': 0.15958620269619,
    #     'initial_delta': 0.0106417288572204, #0.106417288572204,
    #     'initial_v': 0.0249967313173077,
    #     'penalty_lambda1': 10,#0.2,#2.0,#0.2,#10.0,
    #     'penalty_lambda2': 50.,#1,#10.0,#1.0,#50.0,
    #     'penalty_lambda_riv': 100, #5.0,
    #     'monthly_seasonal_factors': np.array([-0.106616824924423/6, -0.152361004102492/6, -0.167724706188117/6, -0.16797984045645/6,
    #                                  -0.159526180248348/6, -0.13927943487493/6, -0.0953402986114613/6, -0.0474646801238288/6, 
    #                                  -0.0278622280543003/6, 0.000000/6, -0.00850263509128089/6, -0.0409638719325969/6])
    # }
    # params = {
    #     'n_months': 12,
    #     'V_min': 0,
    #     'V_max': 1,
    #     'V_0': 0,
    #     'W_max': 0.4,
    #     'I_max': 0.4,
    #     # 'storage_capacity': 100000,
    #     'kappa_r': 0.492828372105622,
    #     'sigma_r': 0.655898616135014,
    #     'theta_r': 0.000588276156660185,
    #     'kappa_delta': 1.17723166341479,
    #     'sigma_delta': 1.03663918307669,
    #     'theta_delta': -0.213183673388138,
    #     'sigma_s': 0.791065501973918,
    #     'rho_1': 0.899944474373156,
    #     'rho_2': -0.306810849087325,
    #     'sigma_v': 0.825941396204049,
    #     'theta_v': 0.0505685591761352 * 10,
    #     'theta': 0.00640705687096142,
    #     'kappa_v': 2.36309244973169,
    #     'lam': 0.638842070975342,
    #     'sigma_j': 0.032046147726045,
    #     'mu_j': 0.0137146728855484,
    #     'seed': seed,
    #     'initial_spot_price': np.exp(2.9479),
    #     'initial_r': 0.15958620269619,
    #     'initial_delta': 0.106417288572204,
    #     'initial_v': 0.0249967313173077 * 10,
    #     'penalty_lambda1': 10,#0.2,#2.0,#0.2,#10.0,
    #     'penalty_lambda2': 50.,#1,#10.0,#1.0,#50.0,
    #     'penalty_lambda_riv': 10.0,
    #     'monthly_seasonal_factors': np.array([0,0,0,0,0,0,0,0,0,0,0,0])
    # }
    env = TTFGasStorageEnv(params)
    #result, final_eval_score, training_time, wallclock_time = agent.train(make_env_fn, make_env_kargs, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward)
    result, final_eval_score, training_time, wallclock_time = agent.train(env, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward)
    ddpg_results.append(result)
    if final_eval_score > best_eval_score:
        best_eval_score = final_eval_score
        best_agent = agent
ddpg_results = np.array(ddpg_results)
_ = BEEP()

torch.save(best_agent.online_policy_model.state_dict(), "online_policy_model_autoregressive_penalized.pth")

ddpg_max_t, ddpg_max_r, ddpg_max_s, \
ddpg_max_sec, ddpg_max_rt = np.max(ddpg_results, axis=0).T
ddpg_min_t, ddpg_min_r, ddpg_min_s, \
ddpg_min_sec, ddpg_min_rt = np.min(ddpg_results, axis=0).T
ddpg_mean_t, ddpg_mean_r, ddpg_mean_s, \
ddpg_mean_sec, ddpg_mean_rt = np.mean(ddpg_results, axis=0).T
ddpg_x = np.arange(len(ddpg_mean_s))

fig, axs = plt.subplots(2, 1, figsize=(15,10), sharey=False, sharex=True)

# DDPG
axs[0].plot(ddpg_max_r, 'r', linewidth=1)
axs[0].plot(ddpg_min_r, 'r', linewidth=1)
axs[0].plot(ddpg_mean_r, 'r:', label='DDPG', linewidth=2)
axs[0].fill_between(
    ddpg_x, ddpg_min_r, ddpg_max_r, facecolor='r', alpha=0.3)

axs[1].plot(ddpg_max_s, 'r', linewidth=1)
axs[1].plot(ddpg_min_s, 'r', linewidth=1)
axs[1].plot(ddpg_mean_s, 'r:', label='DDPG', linewidth=2)
axs[1].fill_between(
     ddpg_x, ddpg_min_s, ddpg_max_s, facecolor='r', alpha=0.3)

# ALL
axs[0].set_title('Moving Avg Reward (Training)')
axs[1].set_title('Moving Avg Reward (Evaluation)')
plt.xlabel('Episodes')
axs[0].legend(loc='upper left')
plt.savefig("Moving_Average_Reward_Autoregressive_Penalized.png")

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

# kappa_r = 0.492828372105622
# sigma_r = 0.655898616135014
# theta_r = 0.000588276156660185
# kappa_delta= 1.17723166341479
# sigma_delta = 0.103663918307669 #1.03663918307669
# theta_delta = -0.213183673388138
# sigma_s = 0.791065501973918
# rho_1 = 0.899944474373156
# rho_2 = -0.306810849087325
# sigma_v = 0.825941396204049
# theta_v = 0.0505685591761352
# theta = 0.00640705687096142
# kappa_v = 2.36309244973169
# lam = 0.638842070975342
# sigma_j = 0.032046147726045
# mu_j = 0.0137146728855484
# seed = 34
# initial_spot_price = np.exp(2.9479)
# initial_r = 0.15958620269619
# initial_delta =  0.0106417288572204 #0.106417288572204
# initial_v =  0.0249967313173077

# ksi_r = np.sqrt(kappa_r**2 + 2*sigma_r**2)
# seasonal_factors = np.array([ -0.106616824924423/6, -0.152361004102492/6, -0.167724706188117/6, -0.16797984045645/6,
#                              -0.159526180248348/6, -0.13927943487493/6, -0.0953402986114613/6, -0.0474646801238288/6,
#                              -0.0278622280543003/6, 0.000000/6, -0.00850263509128089/6, -0.0409638719325969/6  ])

# kappa_r = 0.492828372105622
# sigma_r = 0.655898616135014
# theta_r = 0.000588276156660185
# kappa_delta= 1.17723166341479
# sigma_delta = 1.03663918307669
# theta_delta = -0.213183673388138
# sigma_s = 0.791065501973918
# rho_1 = 0.899944474373156
# rho_2 = -0.306810849087325
# sigma_v = 0.825941396204049
# theta_v = 10 * 0.0505685591761352
# theta = 0.00640705687096142
# kappa_v = 2.36309244973169
# lam = 0.638842070975342
# sigma_j = 0.032046147726045
# mu_j = 0.0137146728855484
# seed = 34
# initial_spot_price = np.exp(2.9479)
# initial_r = 0.15958620269619
# initial_delta =  0.106417288572204
# initial_v =  10 * 0.0249967313173077

# ksi_r = np.sqrt(kappa_r**2 + 2*sigma_r**2)
# seasonal_factors = np.array([0,0,0,0,0,0,0,0,0,0,0,0])

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
plt.savefig("Reinforcement_Learning_Value_Autoregressive_Penalized.png")