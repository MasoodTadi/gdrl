# ttf_gas_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TTFGasStorageEnv(gym.Env):
    def __init__(self, params):
        super(TTFGasStorageEnv, self).__init__()
        
        self.max_timesteps = 30 * 12
        self.seed_value = params.get('seed', None)
        self.dt = 1.0 / self.max_timesteps
        
        self.V_min = params['V_min']
        self.V_max = params['V_max']
        self.V_0 = params['V_0']
        self.I_max = params['I_max']
        self.W_max = params['W_max']
        self.n_months = params['n_months']

        self.initial_r = params['initial_r']
        self.theta_r = params['theta_r']
        self.kappa_r = params['kappa_r']
        self.sigma_r = params['sigma_r']
        self.initial_delta = params['initial_delta']
        self.theta_delta = params['theta_delta']
        self.kappa_delta = params['kappa_delta']
        self.sigma_delta = params['sigma_delta']
        self.initial_v = params['initial_v']
        self.kappa_v = params['kappa_v']
        self.sigma_v = params['sigma_v']
        self.theta_v = params['theta_v']
        self.initial_spot_price = params['initial_spot_price']
        self.sigma_s = params['sigma_s']
        self.lam = params['lam']
        self.mu_j = params['mu_j']
        self.sigma_j = params['sigma_j']
        self.theta = params['theta']
        self.rho_1 = params['rho_1']
        self.rho_2 = params['rho_2']
        self.ksi_r = np.sqrt(self.kappa_r**2 + 2*self.sigma_r**2)

        self.penalty_lambda1 = params['penalty_lambda1']
        self.penalty_lambda2 = params['penalty_lambda2']

        self.seasonal_factors = params['monthly_seasonal_factors']

        self.seed(self.seed_value)

        # === ACTION SPACE ===
        # Discrete actions
        # 0 -> -0.4
        # 1 -> -0.2
        # 2 -> 0.0
        # 3 -> +0.2
        # 4 -> +0.4
        self.action_meanings_list = []
        
        for i in range(self.n_months):
            if i == 0:
                # Month 0 only allows 0.0, +0.2, +0.4
                self.action_meanings_list.append([0.0, 0.2, 0.4])
            elif i == self.n_months - 1:
                # Month 11 only allows -0.4, -0.2, 0.0
                self.action_meanings_list.append([-0.4, -0.2, 0.0])
            else:
                # Middle months allow full range
                self.action_meanings_list.append([-0.4, -0.2, 0.0, 0.2, 0.4])

        # For each month, number of valid actions = length of that list
        nvec = [len(meaning) for meaning in self.action_meanings_list]
        self.action_space = spaces.MultiDiscrete(nvec)

        # === OBSERVATION SPACE ===
        self.observation_space = spaces.Box(
            low=np.concatenate(([0], [-np.inf] * 12, [self.V_min])),
            high=np.concatenate(([12], [np.inf] * 12, [self.V_max])),
            shape=(14,),
            dtype=np.float32,
            seed=self.seed_value
        )

        self.reset()

    def seed(self, seed=None):
        if seed is not None:
            self.seed_value = seed
        self.W = np.random.default_rng(seed=self.seed_value)
        return [self.seed_value]

    def compute_futures_curve(self):
        futures_list = np.full(12, 0.0, dtype=np.float32)
        remaining_futures = max(12 - (self.day // 30), 0)

        for k in range(12):
            expiration_day = (k+1) * 30
            tau = (expiration_day - self.day) / 360.0
            if tau < 0:
                continue

            beta_r = (2 * (1 - np.exp(-self.ksi_r * tau))) / (2 * self.ksi_r - (self.ksi_r - self.kappa_r) * (1 - np.exp(-self.ksi_r * tau)))
            beta_delta = -(1 - np.exp(-self.kappa_delta * tau)) / self.kappa_delta
            beta_0 = (self.theta_r / self.sigma_r**2) * (2 * np.log(1 - (self.ksi_r - self.kappa_r) * (1 - np.exp(-self.ksi_r * tau)) / (2 * self.ksi_r)) + (self.ksi_r - self.kappa_r) * tau)
            F_tk = np.exp(np.log(self.S_t) + self.seasonal_factors[k] + beta_0 + beta_r * self.r_t + beta_delta * self.delta_t)
            futures_list[k] = F_tk
        return futures_list

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
        return np.concatenate(([self.month], self.F_t, [self.V_t]), dtype=np.float32), {}

    def step(self, action_code):
        
        # Validate action dimension
        assert len(action_code) == self.n_months, "Action must have length = n_months"

        action = np.array([self.action_meanings_list[i][action_code[i]] for i in range(self.n_months)])
        
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
                self.V_t = np.clip(new_volume, self.V_min, self.V_max)
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
        is_terminal = False
        is_truncated = False
        if self.month == 12: 
            is_terminal = True
            # last_action = np.clip(-self.V_t, -self.W_max, self.I_max)
            reward += - self.F_t[-1] * action[-1]
            # reward += - self.F_t[-1] * last_action
            # self.V_t += last_action
            self.V_t += action[-1]
            self.V_t = np.clip(self.V_t, self.V_min, self.V_max)
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
        info = {'cost1':cost1, 'cost2':cost2}
        return np.concatenate(([self.month], self.F_t, [self.V_t]), dtype=np.float32), reward, is_terminal, is_truncated, info
