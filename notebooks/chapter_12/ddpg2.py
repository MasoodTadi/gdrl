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

print(torch.cuda.is_available())

# Define the months and corresponding seasonal factors (as you provided)
month_midpoints = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349, 380])  # Middle of each month
seasonal_factors = np.array([0.000000, -0.00850263509128089, -0.0409638719325969, -0.106616824924423, -0.152361004102492, -0.167724706188117, -0.16797984045645,
                             -0.159526180248348, -0.13927943487493, -0.0953402986114613, -0.0474646801238288, -0.0278622280543003, 0.0])

# Create a cubic spline interpolation
cs = CubicSpline(month_midpoints, seasonal_factors, bc_type='periodic')
# Generate daily data for a year (365 days)
days_in_year = np.linspace(1, 397, 397)
daily_seasonal_factors = cs(days_in_year)

# Create a DataFrame for the interpolated daily seasonal factors
dates = pd.date_range(start="2024-01-01", periods=366)  # Create dates for one year
df = pd.DataFrame(daily_seasonal_factors[:366], index=dates, columns=["Seasonal_Factor"])

# Save the data to a CSV file
df.to_csv('interpolated_seasonal_factors.csv', index_label='Timestamp')

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(days_in_year, daily_seasonal_factors, 'r-', label="Interpolated Daily Seasonal Factors (Cubic Spline)")
plt.scatter(month_midpoints, seasonal_factors, color='blue', label="Monthly Seasonal Factors (Mid-month)")
plt.xlabel("Day of Year")
plt.ylabel("Seasonal Factor")
plt.title("Cubic Spline Interpolation of Seasonal Factors")
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig('seasonal_factors_interpolation.png', dpi=300, bbox_inches='tight')

# Optionally, comment out plt.show() if you don't want to display the plot
# plt.show()


class GasStorageEnv(gym.Env):
    def __init__(self, params):
        super(GasStorageEnv, self).__init__()

        # Calculate the number of business days in 2024
        self.start_date = pd.Timestamp('2023-04-01')
        self.end_date = pd.Timestamp('2024-03-31')
        # List of holidays between April 1, 2023 and March 31, 2024 in the Czech Republic
        self.holidays = [
            pd.Timestamp('2023-04-07'),  # Good Friday
            pd.Timestamp('2023-04-10'),  # Easter Monday
            pd.Timestamp('2023-05-01'),  # Labour Day
            pd.Timestamp('2023-05-08'),  # Liberation Day
            pd.Timestamp('2023-07-05'),  # Saints Cyril and Methodius Day
            pd.Timestamp('2023-07-06'),  # Jan Hus Day
            pd.Timestamp('2023-09-28'),  # Czech Statehood Day
            pd.Timestamp('2023-10-28'),  # Independent Czechoslovak State Day
            pd.Timestamp('2023-11-17'),  # Struggle for Freedom and Democracy Day
            pd.Timestamp('2023-12-24'),  # Christmas Eve (Sunday)
            pd.Timestamp('2023-12-25'),  # Christmas Day
            pd.Timestamp('2023-12-26'),  # St. Stephen's Day
            pd.Timestamp('2024-01-01'),  # New Year's Day
            pd.Timestamp('2024-03-29'),  # Good Friday
        ]
        self.all_trading_days = self.trading_days(self.start_date, self.end_date, self.holidays)
        self.date_to_index = {date: idx for idx, date in enumerate(self.all_trading_days)}
        self.max_timesteps = len(self.all_trading_days) # Number of business days 

        self.seasonal_factors = pd.read_csv('interpolated_seasonal_factors.csv', index_col='Timestamp', parse_dates=True)

        # Define filling target dates and their required storage levels
        self.filling_targets = {
            pd.Timestamp('2023-05-02'): 0.10,  # 10% stock by 1st May
            pd.Timestamp('2023-07-03'): 0.30,  # 30% stock by 1st July
            pd.Timestamp('2023-09-01'): 0.60,  # 60% stock by 1st September
            pd.Timestamp('2023-11-01'): 0.90,  # 90% stock by 1st November
            pd.Timestamp('2024-01-02'): 0.30,  # 30% stock by 2nd January
        }
        # Parameters
        #self.max_timesteps = params['max_timesteps']
        self.seed_value = params.get('seed', None)
        self.dt = 1.0 / self.max_timesteps
        self.storage_capacity = params['storage_capacity']
        self.alpha = params['alpha']
        self.kappa_r = params['kappa_r']
        self.sigma_r = params['sigma_r']
        self.theta_r = params['theta_r']
        self.kappa_delta = params['kappa_delta']
        self.sigma_delta = params['sigma_delta']
        self.theta_delta = params['theta_delta']
        self.sigma_s = params['sigma_s']
        self.rho_1 = params['rho_1']
        self.rho_2 = params['rho_2']
        self.sigma_v = params['sigma_v']
        self.theta_v = params['theta_v']
        self.kappa_v = params['kappa_v']
        self.lam = params['lam']
        self.sigma_j = params['sigma_j']
        self.mu_j = params['mu_j']
        self.theta = params['theta']
        self.penalty_lambda = params['penalty_lambda']
        self.bonus_lambda = params['bonus_lambda']
        # self.W = np.random.default_rng() # Default; this will be seeded in `seed()` method

        # Initial date setup
        # self.current_date = pd.Timestamp('2024-01-02')  # First business day of January 2024

        # State variables as parameters
        self.initial_spot_price = params['initial_spot_price']
        self.initial_r = params['initial_r']
        self.initial_delta = params['initial_delta']
        self.initial_V = params['initial_V']
        
        # Initialize trajectories
        self.S_trajectory = []
        self.SS_trajectory = []
        self.r_trajectory = []
        self.delta_trajectory = []
        self.V_trajectory = []
        self.F_trajectory = []

        self.delivery_quantities = []

        # Set the seed for reproducibility
        self.seed(self.seed_value)

        # Action and Observation spaces
        action_space_low = np.array([0, -1])
        action_space_high = np.array([1, 1])
        self.action_space = gym.spaces.Box(low=action_space_low, high=action_space_high, dtype=np.float32, seed=self.seed_value)
        self.observation_space = gym.spaces.Box(low=np.array([0, 0.0, 0.0, 0.0]), high=np.array([self.max_timesteps, self.storage_capacity, 100, 100]), dtype=np.float32,seed=self.seed_value)

        # Initialize state
        #self.reset()
        
    def seed(self, seed=None):
        """
        Seed the environment, ensuring reproducibility of the randomness in the environment.
        """
        if seed is not None:
            self.seed_value = seed  # Update seed if provided
        self.W = np.random.default_rng(seed=self.seed_value)  # Seed the random generator
        #self.action_space.seed(self.seed_value)  # Seed the action space random generator
        return [self.seed_value]  
        
    def trading_days(self, start_date, end_date, holidays):      
        all_business_days = pd.date_range(start=start_date, end=end_date, freq='B')
        # Exclude holidays
        trading_days = all_business_days.difference(holidays)
        return trading_days
        
    def get_first_last_trading_days(self, current_date):
        """
        Calculate the first and last trading days of the next nearby month based on current_date.
        """
        # Get the first day of the next month
        next_month = current_date + pd.offsets.MonthBegin(1)
        
        # Create a date range for the entire next month
        next_month_range = pd.date_range(next_month, next_month + pd.offsets.MonthEnd(1), freq='B')  # Business days only

        # Exclude holidays
        next_month_range = next_month_range.difference(self.holidays)
        
        # tau_1 is the first trading day, tau_2 is the last trading day
        tau_1 = next_month_range[0]
        tau_2 = next_month_range[-1]
        
        return tau_1, tau_2

    def compute_real_actions(self, h_S_k_star, h_j_k_star, d_previous_month, next_month_days):
        # Constants
        max_injection_rate = 1447  # Maximum injection rate (MWh/day)
        max_withdrawal_rate = 1813  # Maximum withdrawal rate (MWh/day)
    
        # Calculate current stock percentage based on the current storage level and total capacity
        stock_percent_k = self.storage_level / self.storage_capacity  # Fraction of total capacity (0 to 1)
    
        # Compute Withdrawal Bound (l_k) based on stock level
        if stock_percent_k <= 0.4:  # Stock level <= 40%
            withdrawal_fraction = 0.4 + (1.0 - 0.4) * (stock_percent_k / 0.4)
        else:  # Stock level > 40%
            withdrawal_fraction = 1.0  # Max withdrawal rate
    
        l_k = -withdrawal_fraction * max_withdrawal_rate
    
        # Compute Injection Bound (u_k) based on stock level
        if stock_percent_k <= 0.7:  # Stock level <= 70%
            injection_fraction = 0.7 + (1.0 - 0.7) * (stock_percent_k / 0.7)
        else:  # Stock level > 70%
            injection_fraction = 1.0 + (0.5 - 1.0) * ((stock_percent_k - 0.7) / (1.0 - 0.7))
    
        u_k = injection_fraction * max_injection_rate
    
        # Initialize action space bounds
        # action_space_low = np.array([0.0 , 0.0])
        # action_space_high = np.array([0.0, 0.0])
    
        # Ensure bounds respect current storage level and capacity
        tilde_l_k = max(l_k, -self.storage_level)  # Don't withdraw more than available storage
        tilde_u_k = min(u_k, self.storage_capacity - self.storage_level)  # Don't inject more than available space

        # Map h_k^S* (normalized storage action) to the real-world storage action (h_k^S)
        h_S_k = h_S_k_star * (tilde_u_k - tilde_l_k) + (tilde_l_k - d_previous_month)
    
        # Update action space bounds based on storage bounds and previous month demand
        # action_space_low[0] = tilde_l_k - d_previous_month
        # action_space_high[0] = tilde_u_k - d_previous_month
    
        # Handle forward market bounds for March (no trading allowed)
        if next_month_days == 0:
            max_h_j_k = 0.0  # No forward market action allowed in March
            # action_space_low[1] = 0.0  # No action on forward market in December
            # action_space_high[1] = 0.0
        else:
            max_h_j_k = self.alpha * self.storage_capacity / next_month_days
            # action_space_low[1] = -max_h_j_k  # Limit forward market actions
            # action_space_high[1] = max_h_j_k
    
        # Map h_k^j* (normalized forward market action) to real-world forward market action (h_k^j)
        h_j_k = h_j_k_star * max_h_j_k
        # return action_space_low, action_space_high
        return h_S_k, h_j_k
        
    # def compute_action_space_bounds(self, k, d_previous_month, next_month_days):
    #     # Initialize action space bounds
    #     action_space_low = np.array([-self.storage_capacity, -self.storage_capacity])
    #     action_space_high = np.array([self.storage_capacity, self.storage_capacity])
        
    #     # Compute bounds for h_S_k
    #     if k <= 126:
    #         l_k = -600
    #     else:
    #         l_k = -3072

    #     if k <= 148:
    #         u_k = 2808
    #     else:
    #         u_k = 408
        
    #     tilde_l_k = max(l_k, -self.storage_level)
    #     tilde_u_k = min(u_k, self.storage_capacity - self.storage_level)
        
    #     action_space_low[0] = tilde_l_k - d_previous_month
    #     action_space_high[0] = tilde_u_k - d_previous_month

    #     if next_month_days == 0:
    #         action_space_low[1] = 0.0  # No action on forward market in December
    #         action_space_high[1] = 0.0 # No action on forward market in December
    #     else:
    #         max_h_j_k = self.alpha * self.storage_capacity / next_month_days
    #         action_space_low[1] = -max_h_j_k # instead of -self.storage_capacity  
    #         action_space_high[1] = max_h_j_k
    #     # self.action_space = gym.spaces.Box(low=self.action_space_low, high=self.action_space_high, dtype=np.float32, seed=self.seed_value)
    #     return action_space_low, action_space_high
    
    def forward_price(self, current_date, S_t, r_t, delta_t):
        # Ensure that sample_spot_price has been called
        # if len(self.S_trajectory) == 0:
        #     raise RuntimeError("You must call sample_spot_price() before calling forward_price()")
            
        # Assuming t and tau are in terms of time steps
        # if t < 0 or t >= len(self.S_trajectory):
        #     raise ValueError("Time t is out of range")

        # # Spot price at time t
        # S_t = self.S_trajectory[t]
        # r_t = self.r_trajectory[t]
        # delta_t = self.delta_trajectory[t]
        tau_1, tau_2 = self.get_first_last_trading_days(current_date)
        
        # Convert tau_1 and tau_2 to years
        tau_1_years = (tau_1 - self.current_date).days / self.max_timesteps
        tau_2_years = (tau_2 - self.current_date).days / self.max_timesteps        

        # Calculate the number of business days (except holidays) in a month 
        business_days = pd.date_range(start=tau_1, end=tau_2, freq='B')
        holidays_in_range = [holiday for holiday in self.holidays if tau_1 <= holiday <= tau_2]
        business_days = business_days.difference(holidays_in_range)
        num_business_days = len(business_days)

        forward_prices_for_day = []
        for tau_i, trading_day in zip(np.linspace(tau_1_years, tau_2_years, num=num_business_days), business_days):
        #for tau_i in np.linspace(tau_1_years, tau_2_years, num=num_business_days):
            # Implement the Yan (2002) forward price calculation
            ksi_r = np.sqrt(self.kappa_r**2 + 2*self.sigma_r**2)
            beta_r = (2*(1 - np.exp(-ksi_r * tau_i))) / (2 * ksi_r - (ksi_r - self.kappa_r) * (1 - np.exp(-ksi_r * tau_i)))
            beta_delta = -(1 - np.exp(-self.kappa_delta * tau_i)) / self.kappa_delta
            beta_0 = (self.theta_r / self.sigma_r**2) * (2 * np.log(1 - (ksi_r - self.kappa_r) * (1 - np.exp(-ksi_r * tau_i)) / (2 * ksi_r))
                                                         + (ksi_r - self.kappa_r) * tau_i) + (self.sigma_delta**2 * tau_i) / (2 * self.kappa_delta**2) \
                     - (self.sigma_s * self.sigma_delta * self.rho_1 + self.theta_delta) * tau_i / self.kappa_delta \
                     - (self.sigma_s * self.sigma_delta * self.rho_1 + self.theta_delta) * np.exp(-self.kappa_delta * tau_i) / self.kappa_delta**2 \
                     + (4 * self.sigma_delta**2 * np.exp(self.kappa_delta * tau_i) - self.sigma_delta**2 * np.exp(-2 * self.kappa_delta * tau_i)) / (4 * self.kappa_delta**3) \
                     + (self.sigma_s * self.sigma_delta * self.rho_1 + self.theta_delta) / self.kappa_delta**2 \
                     - 3 * self.sigma_delta**2 / (4 * self.kappa_delta**3)

            # Get the daily seasonal factor for the current trading day
            daily_seasonal_factor = self.seasonal_factors[(self.seasonal_factors.index.month==trading_day.month)&(self.seasonal_factors.index.day==trading_day.day)].values[0][0]
            
            # Forward price calculation
            forward_price = np.exp(np.log(S_t) + daily_seasonal_factor  + beta_0 + beta_r * r_t + beta_delta * delta_t)
            forward_prices_for_day.append(forward_price)
        return np.mean(forward_prices_for_day)
        
    def compute_delivery_quantity(self, month_start, month_end):
        # Get the indices for month_start and month_end
        start_index = self.date_to_index[month_start]
        end_index = self.date_to_index[month_end]
        # Sum the forward actions from start_index to end_index (inclusive)
        delivery_quantity = sum(self.forward_actions[start_index:end_index + 1])
        if np.isnan(delivery_quantity):
            print("month_start: ", month_start)
            print("month_end: ", month_end)
            print("start_index: ", start_index)
            print("end_index: ", end_index)
            print("self.forward_actions[start_index:end_index + 1]: ", self.forward_actions[start_index:end_index + 1])
            print(len(self.forward_actions[start_index:end_index + 1]))
        return delivery_quantity  
        
    def reset(self):
        # Initialize trajectories
        self.S_t = self.initial_spot_price
        self.SS_t = np.exp(np.log(self.S_t) + 
                           self.seasonal_factors[(self.seasonal_factors.index.month==self.all_trading_days[0].month)&(self.seasonal_factors.index.day==self.all_trading_days[0].day)].values[0][0])
        self.r_t = self.initial_r
        self.delta_t = self.initial_delta
        self.V_t = self.initial_V
        self.step_count = 0
        self.current_date = self.all_trading_days[self.step_count]
        self.F_t = self.forward_price(self.current_date, self.S_t, self.r_t, self.delta_t)
        
        self.S_trajectory.append(self.S_t)
        self.SS_trajectory.append(self.SS_t)
        self.r_trajectory.append(self.r_t)
        self.delta_trajectory.append(self.delta_t)
        self.V_trajectory.append(self.V_t)
        self.F_trajectory.append(self.F_t)
        

        self.delivery_quantities.append(0.0)
        self.forward_actions = []
        
        
        self.storage_level = 0

        self.W_S = 0
        self.W_F = 0

        self.reward = 0

        # Compute action space bounds based on constraints
        # next_month_days = len(self.all_trading_days[self.all_trading_days.month == self.current_date.month+1])
        # self.bounds = self.compute_action_space_bounds(self.step_count, 0, next_month_days)

        return np.array([self.step_count, self.storage_level, self.S_t, self.F_t]), {}

    def step(self, action):
        # Calculate d^{I-1} for delivery quantities
        current_month = self.current_date.month
        if current_month == 4:
            d_previous_month = 0  # April has no deliveries
        elif current_month == 1:
            previous_month_start = self.all_trading_days[self.all_trading_days.month == 12][0]
            previous_month_end = self.all_trading_days[self.all_trading_days.month == 12][-1]
            d_previous_month = self.compute_delivery_quantity(previous_month_start, previous_month_end)
        else:
            # For February and onward, calculate d^{I-1}
            previous_month_start = self.all_trading_days[self.all_trading_days.month == current_month-1][0]
            previous_month_end = self.all_trading_days[self.all_trading_days.month == current_month-1][-1]
            d_previous_month = self.compute_delivery_quantity(previous_month_start, previous_month_end)

        if current_month == 3:
            next_month_days = 0
        elif current_month == 12:
            next_month_days = len(self.all_trading_days[self.all_trading_days.month == 1])
        else:
            next_month_days = len(self.all_trading_days[self.all_trading_days.month == current_month+1])

        # Action: [h_S_k, h_j_k]
        h_S_k_star, h_j_k_star = action
        h_S_k, h_j_k = self.compute_real_actions(h_S_k_star, h_j_k_star, d_previous_month, next_month_days)
            
        self.forward_actions.append(h_j_k)

        # Updating storage level
        self.storage_level += h_S_k + d_previous_month 

        # Updating P&L
        self.W_S += -h_S_k * self.S_t
        self.W_F += -h_j_k * self.F_t * next_month_days
        # Compute reward
        if self.current_date == self.all_trading_days[-2]:#self.max_timesteps - 1:
            W_terminal = self.W_S + self.W_F
            # print("self.W_S: ",self.W_S)
            # print("self.W_F: ",self.W_F)
            # print("W_terminal: ",W_terminal)
            self.reward += self.utility_function(W_terminal)
            # print("self.utility_function(W_terminal): ", self.utility_function(W_terminal))
            # print("self.reward: ",self.reward)
            # Penalize if storage level is not zero at terminal time
            if self.storage_level != 0:
                penalty = -self.penalty_lambda * abs(self.storage_level)
                self.reward += penalty
            #print("self.current_date: ",self.current_date)
            #print("reward: ",reward)
        else:
            # Check if current date is a filling target date
            if self.current_date in self.filling_targets:
                target_storage_level = self.filling_targets[self.current_date]
                
                # If storage level is less than the required target, apply a penalty
                if self.storage_level < (target_storage_level * self.storage_capacity):
                    penalty = -self.penalty_lambda * (target_storage_level * self.storage_capacity - self.storage_level)
                    self.reward += penalty
    
                # Optional: If storage level exceeds the target, give a bonus (optional)
                elif self.storage_level >= target_storage_level * self.storage_capacity:
                    bonus = self.bonus_lambda * (self.storage_level - target_storage_level * self.storage_capacity)
                    self.reward += bonus
        #print("current_date: ",self.current_date, "reward: ", self.reward)
        
        # Generate independent Brownian increments
        dW_1 = self.W.normal(0, np.sqrt(self.dt))  # For dW_1
        dW_r = self.W.normal(0, np.sqrt(self.dt))  # For dW_r (interest rate)
        dW_2 = self.W.normal(0, np.sqrt(self.dt))  # For dW_2
        dW_delta = self.rho_1 * dW_1 + np.sqrt(1 - self.rho_1 ** 2) * self.W.normal(0, np.sqrt(self.dt))  # For dW_delta (correlated with dW_1)
        dW_V = self.rho_2 * dW_2 + np.sqrt(1 - self.rho_2 ** 2) * self.W.normal(0, np.sqrt(self.dt))  # For dW_V (correlated with dW_2)
        
        # Probability of jump occurrence
        dq = self.W.choice([0, 1], p=[1 - self.lam * self.dt, self.lam * self.dt])

        # Jump magnitude: ln(1 + J) ~ N[ln(1 + mu_J) - 0.5 * sigma_J^2, sigma_J^2]
        ln_1_plus_J = self.W.normal(np.log(1 + self.mu_j) - 0.5 * self.sigma_j ** 2, self.sigma_j)
        J = np.exp(ln_1_plus_J) - 1  # Jump size for the spot price

        # Stochastic differential equations (SDEs)
        # dS_t = (r_t - delta_t - \lambda \mu_J)S_t dt + \sigma_s S_t dW_1 + \sqrt{V_t} S_t dW_2 + J S_t dq
        dS_t = (self.r_t - self.delta_t - self.lam * self.mu_j) * self.S_t * self.dt + self.sigma_s * self.S_t * dW_1 + np.sqrt(max(self.V_t,0)) * self.S_t * dW_2 + J * self.S_t * dq
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

        # dV_t = (\theta_V - \kappa_V V_t) dt + \sigma_V \sqrt{V_t} dW_V + J_V dq
        dV_t = (self.theta_v - self.kappa_v * self.V_t) * self.dt + self.sigma_v * np.sqrt(max(self.V_t, 0)) * dW_V + np.exp(self.theta) * dq
        self.V_t += dV_t
        self.V_trajectory.append(self.V_t)
        
        self.step_count += 1
        #self.current_date += pd.Timedelta(days=1)
        #self.current_date += pd.offsets.BDay()
        self.current_date = self.all_trading_days[self.step_count]

        self.F_t = self.forward_price(self.current_date, self.S_t, self.r_t, self.delta_t)
        self.F_trajectory.append(self.F_t)

        self.SS_t = np.exp(np.log(self.S_t) + 
                           self.seasonal_factors[(self.seasonal_factors.index.month==self.all_trading_days[self.step_count].month)&(self.seasonal_factors.index.day==self.all_trading_days[self.step_count].day)].values[0][0])
        self.SS_trajectory.append(self.SS_t)

        # Check if done
        
        is_terminal = self.step_count >= self.max_timesteps - 1 # Whether the episode is over
        # is_truncated can indicate whether the episode was cut short due to external conditions.
        is_truncated = False  # Assuming no time limit or other truncation factors
        
        ###############################################################
        # Update constraints of the next action
        ###############################################################
        # Calculate d^{I-1} for delivery quantities
        # current_month = self.current_date.month
        # if current_month == 4:
        #     d_previous_month = 0  # April has no deliveries
        # elif current_month == 1:
        #     previous_month_start = self.all_trading_days[self.all_trading_days.month == 12][0]
        #     previous_month_end = self.all_trading_days[self.all_trading_days.month == 12][-1]
        #     d_previous_month = self.compute_delivery_quantity(previous_month_start, previous_month_end)
        # else:
        #     # For other months, calculate d^{I-1}
        #     previous_month_start = self.all_trading_days[self.all_trading_days.month == current_month-1][0]
        #     previous_month_end = self.all_trading_days[self.all_trading_days.month == current_month-1][-1]
        #     d_previous_month = self.compute_delivery_quantity(previous_month_start, previous_month_end)
        #     if np.isnan(d_previous_month):
        #         raise ValueError(f"Invalid bounds: d_previous_month={d_previous_month}")
        # if current_month == 3:
        #     next_month_days = 0
        # elif current_month == 12:
        #     next_month_days = len(self.all_trading_days[self.all_trading_days.month == 1])            
        # else:
        #     next_month_days = len(self.all_trading_days[self.all_trading_days.month == current_month+1])
        # # Now we compute the constraints for the next actions
        # self.bounds = self.compute_action_space_bounds(self.step_count, d_previous_month, next_month_days)
        
        return np.array([self.step_count, self.storage_level, self.SS_t, self.F_t]), self.reward, is_terminal, is_truncated, {}
    
    # def adjust_h_S_k(self, h_S_k, d_previous_month):
    #     # Compute constraints for h_S_k
    #     k = self.step_count
    #     if k <= 126:
    #         l_k = -600
    #     else:
    #         l_k = -3072

    #     if k <= 148:
    #         u_k = 2808
    #     else:
    #         u_k = 408
        
    #     tilde_l_k = max(l_k, -self.storage_level)
    #     tilde_u_k = min(u_k, self.storage_capacity - self.storage_level)
        
    #     return np.clip(h_S_k, tilde_l_k- d_previous_month, tilde_u_k - d_previous_month)
    
    # def adjust_h_j_k(self, h_j_k, next_month_days):
    #     # Check forward trading action constraint
    #     if next_month_days == 0:
    #         return 0 # No action on forward market in December
    #     else:
    #         max_h_j_k = self.alpha * self.storage_capacity / next_month_days
    #         return np.clip(h_j_k, -np.storage_capacity, max_h_j_k)
    
    # def utility_function(self, W):
    #     # Logarithmic utility function
    #     if W > 0:
    #         return np.log(W)
    #     else:
    #         # In case W is zero or negative, you can decide on how to handle it.
    #         # One option is to return a large negative value (strong penalty).
    #         # Alternatively, you could add a small epsilon to avoid log(0).
    #         return -np.inf  # Or some very large negative number, like -1e6
    def utility_function(self, W):
        return W


class FCQV(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 hidden_dims=(32,32), 
                 activation_fc=F.relu):
        super(FCQV, self).__init__()
        self.activation_fc = activation_fc

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            in_dim = hidden_dims[i]
            if i == 0: 
                in_dim += output_dim
            hidden_layer = nn.Linear(in_dim, hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
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
        x = self.activation_fc(self.input_layer(x))
        for i, hidden_layer in enumerate(self.hidden_layers):
            if i == 0:
                x = torch.cat((x, u), dim=1)
            x = self.activation_fc(hidden_layer(x))
        return self.output_layer(x)
    
    def load(self, experiences):
        states, actions, new_states, rewards, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, new_states, rewards, is_terminals

class FCDP(nn.Module):
    def __init__(self, 
                 input_dim,
                 action_bounds,
                 hidden_dims=(32,32), 
                 activation_fc=F.relu,
                 out_activation_fc=F.tanh):
        super(FCDP, self).__init__()
        self.activation_fc = activation_fc
        self.out_activation_fc = out_activation_fc
        self.env_min, self.env_max = action_bounds

        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(hidden_dims[i], hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], len(self.env_max))

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)
        
        self.env_min = torch.tensor(self.env_min,
                                    device=self.device, 
                                    dtype=torch.float32)

        self.env_max = torch.tensor(self.env_max,
                                    device=self.device, 
                                    dtype=torch.float32)
        
        self.nn_min = self.out_activation_fc(
            torch.Tensor([float('-inf')])).to(self.device)
        self.nn_max = self.out_activation_fc(
            torch.Tensor([float('inf')])).to(self.device)
        self.rescale_fn = lambda x: (x - self.nn_min) * (self.env_max - self.env_min) / \
                                    (self.nn_max - self.nn_min) + self.env_min

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, 
                             device=self.device, 
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, state):
        x = self._format(state)
        x = self.activation_fc(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fc(hidden_layer(x))
        x = self.output_layer(x)
        x = self.out_activation_fc(x)
        return self.rescale_fn(x)

class ReplayBuffer():
    def __init__(self, 
                 max_size=100000,
                 batch_size=128):
        self.ss_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.as_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.rs_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.ps_mem = np.empty(shape=(max_size), dtype=np.ndarray)
        self.ds_mem = np.empty(shape=(max_size), dtype=np.ndarray)

        self.max_size = max_size
        self.batch_size = batch_size
        self._idx = 0
        self.size = 0
    
    def store(self, sample):
        s, a, r, p, d = sample
        self.ss_mem[self._idx] = s
        self.as_mem[self._idx] = a
        self.rs_mem[self._idx] = r
        self.ps_mem[self._idx] = p
        self.ds_mem[self._idx] = d
        
        self._idx += 1
        self._idx = self._idx % self.max_size

        self.size += 1
        self.size = min(self.size, self.max_size)

    def sample(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        idxs = np.random.choice(
            self.size, batch_size, replace=False)
        experiences = np.vstack(self.ss_mem[idxs]), \
                      np.vstack(self.as_mem[idxs]), \
                      np.vstack(self.rs_mem[idxs]), \
                      np.vstack(self.ps_mem[idxs]), \
                      np.vstack(self.ds_mem[idxs])
        return experiences

    def __len__(self):
        return self.size

class GreedyStrategy():
    def __init__(self, bounds):
        self.low, self.high = bounds
        self.ratio_noise_injected = 0

    def select_action(self, model, state):
        with torch.no_grad():
            greedy_action = model(state).cpu().detach().data.numpy().squeeze()

        action = np.clip(greedy_action, self.low, self.high)
        return np.reshape(action, self.high.shape)

class NormalNoiseStrategy():
    def __init__(self, bounds, exploration_noise_ratio=0.1):
        self.low, self.high = bounds
        self.exploration_noise_ratio = exploration_noise_ratio
        self.ratio_noise_injected = 0
        
    def select_action(self, model, state, max_exploration=False):          
        if max_exploration:
            noise_scale = self.high
        else:
            noise_scale = self.exploration_noise_ratio * self.high

        with torch.no_grad():
            greedy_action = model(state).cpu().detach().data.numpy().squeeze()
            if np.isnan(greedy_action).any():
                print("greedy action is NaN")

        noise = np.random.normal(loc=0, scale=noise_scale, size=len(self.high))
        noisy_action = greedy_action + noise
        action = np.clip(noisy_action, self.low, self.high)
        
        self.ratio_noise_injected = np.mean(abs((greedy_action - action)/(self.high - self.low)))
        return action

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

    def optimize_model(self, experiences):
        states, actions, rewards, next_states, is_terminals = experiences
        batch_size = len(is_terminals)

        argmax_a_q_sp = self.target_policy_model(next_states)
        max_a_q_sp = self.target_value_model(next_states, argmax_a_q_sp)
        target_q_sa = rewards + self.gamma * max_a_q_sp * (1 - is_terminals)
        q_sa = self.online_value_model(states, actions)
        td_error = q_sa - target_q_sa.detach()
        value_loss = td_error.pow(2).mul(0.5).mean()
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

        self.checkpoint_dir = tempfile.mkdtemp()
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
        self.update_networks(tau=1.0)
        self.value_optimizer = self.value_optimizer_fn(self.online_value_model, 
                                                       self.value_optimizer_lr)        
        self.policy_optimizer = self.policy_optimizer_fn(self.online_policy_model, 
                                                         self.policy_optimizer_lr)

        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = self.training_strategy_fn(action_bounds)
        # self.training_strategy = self.training_strategy_fn() # No action bounds here
        self.evaluation_strategy = self.evaluation_strategy_fn(action_bounds)
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
                    experiences = self.replay_buffer.sample()
                    experiences = self.online_value_model.load(experiences)
                    self.optimize_model(experiences)

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

    def demo_last(self, title='Fully-trained {} Agent', n_episodes=2, max_n_videos=2):
        env = self.make_env_fn(**self.make_env_kargs, monitor_mode='evaluation', render=True, record=True)

        checkpoint_paths = self.get_cleaned_checkpoints()
        last_ep = max(checkpoint_paths.keys())
        self.online_policy_model.load_state_dict(torch.load(checkpoint_paths[last_ep]))

        self.evaluate(self.online_policy_model, env, n_episodes=n_episodes)
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
            self.online_policy_model.load_state_dict(torch.load(checkpoint_paths[i]))
            self.evaluate(self.online_policy_model, env, n_episodes=1)

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

import multiprocessing

# Define a function to create the policy model
def create_policy_model(nS, bounds):
    return FCDP(nS, bounds, hidden_dims=(256, 256))

# Define the policy optimizer function
def create_policy_optimizer(net, lr):
    return optim.Adam(net.parameters(), lr=lr)

# Define a function to create the value model
def create_value_model(nS, nA):
    return FCQV(nS, nA, hidden_dims=(256, 256))

# Define the value optimizer function
def create_value_optimizer(net, lr):
    return optim.Adam(net.parameters(), lr=lr)

# Define the training strategy function
def create_training_strategy(bounds):
    return NormalNoiseStrategy(bounds, exploration_noise_ratio=0.1)

# Define the evaluation strategy function
def create_evaluation_strategy(bounds):
    return GreedyStrategy(bounds)

# Define the replay buffer function
def create_replay_buffer():
    return ReplayBuffer(max_size=100000, batch_size=256)


# Define a function to run the DDPG training for a single seed
def train_ddpg_for_seed(seed):
    # Environment settings
    environment_settings = {
        'env_name': 'GasStorageEnv',
        'gamma': 0.99,
        'max_minutes': np.inf,
        'max_episodes': 1000,
        'goal_mean_100_reward': np.inf
    }

    # policy_model_fn = lambda nS, bounds: FCDP(nS, bounds, hidden_dims=(256,256))
    # policy_max_grad_norm = float('inf')
    # policy_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
    # policy_optimizer_lr = 0.0003

    # value_model_fn = lambda nS, nA: FCQV(nS, nA, hidden_dims=(256,256))
    # value_max_grad_norm = float('inf')
    # value_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
    # value_optimizer_lr = 0.0003

    # training_strategy_fn = lambda: NormalNoiseStrategy(exploration_noise_ratio=0.1)
    # evaluation_strategy_fn = lambda: GreedyStrategy()

    # replay_buffer_fn = lambda: ReplayBuffer(max_size=100000, batch_size=256)

    policy_model_fn = create_policy_model
    policy_max_grad_norm = float('inf')
    policy_optimizer_fn = create_policy_optimizer
    policy_optimizer_lr = 0.0003
    
    value_model_fn = create_value_model
    value_max_grad_norm = float('inf')
    value_optimizer_fn = create_value_optimizer
    value_optimizer_lr = 0.0003
    
    training_strategy_fn = create_training_strategy
    evaluation_strategy_fn = create_evaluation_strategy
    
    replay_buffer_fn = create_replay_buffer

    n_warmup_batches = 50
    update_target_every_steps = 1
    tau = 0.005

    env_name, gamma, max_minutes, max_episodes, goal_mean_100_reward = environment_settings.values()

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

    # Parameters for the environment
    params = {
        'seed': seed,
        'storage_capacity': 100000,
        'alpha': 0.05,
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
        'initial_spot_price': np.exp(2.9479),
        'initial_r': 0.15958620269619,
        'initial_delta': 0.106417288572204,
        'initial_V': 0.0249967313173077,
        'penalty_lambda': 15,
        'bonus_lambda': 5,
    }
    env = GasStorageEnv(params)
    
    result, final_eval_score, training_time, wallclock_time = agent.train(env, seed, gamma, max_minutes, max_episodes, goal_mean_100_reward)
    
    return result, final_eval_score, agent  # Return the results for further analysis

# Main execution block
if __name__ == '__main__':
    SEEDS = (12, 34, 56, 78, 90)
    ddpg_results = []
    best_agent, best_eval_score = None, float('-inf')

    for seed in SEEDS:
        result, final_eval_score, agent = train_ddpg_for_seed(seed)  # Sequential execution
        ddpg_results.append(result)
        if final_eval_score > best_eval_score:
            best_eval_score = final_eval_score
            best_agent = agent
        
    # with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    #     results = pool.map(train_ddpg_for_seed, SEEDS)  # Parallel execution

    # for result, final_eval_score in results:
    #     ddpg_results.append(result)
    #     if final_eval_score > best_eval_score:
    #         best_eval_score = final_eval_score
    #         # Here you would also need to save or reference the best agent if needed

    ddpg_results = np.array(ddpg_results)
    _ = BEEP()
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
    
    # Save the plot to a file
    plt.savefig('ddpg_results_plot_v2.png', dpi=300, bbox_inches='tight')
    
    # Optionally, you can comment out plt.show() if you don't want to display the plot
    # plt.show()
    
    fig, axs = plt.subplots(3, 1, figsize=(15,15), sharey=False, sharex=True)
    
    # DDPG
    axs[0].plot(ddpg_max_t, 'r', linewidth=1)
    axs[0].plot(ddpg_min_t, 'r', linewidth=1)
    axs[0].plot(ddpg_mean_t, 'r:', label='DDPG', linewidth=2)
    axs[0].fill_between(
        ddpg_x, ddpg_min_t, ddpg_max_t, facecolor='r', alpha=0.3)
    
    axs[1].plot(ddpg_max_sec, 'r', linewidth=1)
    axs[1].plot(ddpg_min_sec, 'r', linewidth=1)
    axs[1].plot(ddpg_mean_sec, 'r:', label='DDPG', linewidth=2)
    axs[1].fill_between(
        ddpg_x, ddpg_min_sec, ddpg_max_sec, facecolor='r', alpha=0.3)
    
    axs[2].plot(ddpg_max_rt, 'r', linewidth=1)
    axs[2].plot(ddpg_min_rt, 'r', linewidth=1)
    axs[2].plot(ddpg_mean_rt, 'r:', label='DDPG', linewidth=2)
    axs[2].fill_between(
        ddpg_x, ddpg_min_rt, ddpg_max_rt, facecolor='r', alpha=0.3)
    
    # ALL
    axs[0].set_title('Total Steps')
    axs[1].set_title('Training Time')
    axs[2].set_title('Wall-clock Time')
    plt.xlabel('Episodes')
    axs[0].legend(loc='upper left')
    
    # Save the plot to a file
    plt.savefig('ddpg_timings_plot_v2.png', dpi=300, bbox_inches='tight')
    
    # Optionally, comment out plt.show() if you don't want to display the plot
    # plt.show()
    
    ddpg_root_dir = os.path.join(RESULTS_DIR, 'ddpg')
    not os.path.exists(ddpg_root_dir) and os.makedirs(ddpg_root_dir)
    
    np.save(os.path.join(ddpg_root_dir, 'x_v2'), ddpg_x)
    
    np.save(os.path.join(ddpg_root_dir, 'max_r_v2'), ddpg_max_r)
    np.save(os.path.join(ddpg_root_dir, 'min_r_v2'), ddpg_min_r)
    np.save(os.path.join(ddpg_root_dir, 'mean_r_v2'), ddpg_mean_r)
    
    np.save(os.path.join(ddpg_root_dir, 'max_s_v2'), ddpg_max_s)
    np.save(os.path.join(ddpg_root_dir, 'min_s_v2'), ddpg_min_s )
    np.save(os.path.join(ddpg_root_dir, 'mean_s_v2'), ddpg_mean_s)
    
    np.save(os.path.join(ddpg_root_dir, 'max_t_v2'), ddpg_max_t)
    np.save(os.path.join(ddpg_root_dir, 'min_t_v2'), ddpg_min_t)
    np.save(os.path.join(ddpg_root_dir, 'mean_t_v2'), ddpg_mean_t)
    
    np.save(os.path.join(ddpg_root_dir, 'max_sec_v2'), ddpg_max_sec)
    np.save(os.path.join(ddpg_root_dir, 'min_sec_v2'), ddpg_min_sec)
    np.save(os.path.join(ddpg_root_dir, 'mean_sec_v2'), ddpg_mean_sec)
    
    np.save(os.path.join(ddpg_root_dir, 'max_rt_v2'), ddpg_max_rt)
    np.save(os.path.join(ddpg_root_dir, 'min_rt_v2'), ddpg_min_rt)
    np.save(os.path.join(ddpg_root_dir, 'mean_rt_v2'), ddpg_mean_rt)