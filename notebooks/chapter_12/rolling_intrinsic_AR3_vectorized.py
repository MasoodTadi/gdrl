import warnings;

warnings.filterwarnings('ignore')
import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torch.multiprocessing as mp
# import threading
# from torch.distributions import Normal

import numpy as np
# import pandas as pd
# from scipy.interpolate import CubicSpline
from scipy.optimize import linprog
# from IPython.display import display
# from collections import namedtuple, deque
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from itertools import cycle, count
from textwrap import wrap

# import pybullet_envs
# import pybullet
# import matplotlib
# import subprocess
# import os.path
# import tempfile
import random
# import base64
# import pprint
import glob
import time
# import json
# import sys
import gymnasium as gym
# import io
import gc

# from gym import wrappers
# from skimage.transform import resize
# from skimage.color import rgb2gray
# from subprocess import check_output
# from IPython.display import display, HTML

import datetime

LEAVE_PRINT_EVERY_N_SECS = 300
ERASE_LINE = '\x1b[2K'
EPS = 1e-6
BEEP = lambda: os.system("printf '\a'")
# RESULTS_DIR = os.path.join('..', 'results')

# plt.style.use('fivethirtyeight')
plt.style.use('default')
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
    """
    ===================== CHANGE 5: VECTORISED OVER PATHS ==============================
    The class, its method names and their meaning are unchanged.  What changed is that
    every internal state variable now carries a leading path axis of length
    `params['n_paths']`, so one call to step() advances N independent Monte-Carlo paths
    at once:

        S_t, r_t, delta_t, v_t, V_t   ->  shape (N,)
        F_t                           ->  shape (N, 12)
        reset()                       ->  state of shape (N, N_STATE)
        step(action)                  ->  action of shape (N, 12), reward of shape (N,)

    With n_paths = 1 the arithmetic is identical to the original scalar code, and the
    random-number stream is identical too (every scalar draw becomes a draw of size N,
    which for N = 1 consumes exactly the same uniforms).  `verify_vec.py` checks this.

    Two further interior changes, both mathematically exact:

      (a) compute_futures_curve() used to recompute beta_0, beta_r, beta_delta from
          scratch for all 12 contracts on every one of the 361 days of every episode.
          Those three functions depend only on tau = (30(k+1) - day)/360, which is
          deterministic, so they are now tabulated once in __init__ as (361, 12) arrays
          and the curve is a single vectorised exp().  This is where most of the speed-up
          comes from; the formula is copied verbatim from the original.

      (b) F_trajectory now stores the curve on the 13 DECISION days only (day 0, 30, ...,
          360) instead of all 361 days.  compute_episode_riv() only ever looked at
          F_trajectory[::30], so its value is unchanged; the daily curves were pure
          memory cost (361 x N x 12 floats).
    ====================================================================================
    """

    def __init__(self, params):
        super(TTFGasStorageEnv, self).__init__()

        self.max_timesteps = 30 * 12 + 1  # Simplistic Case
        self.seed_value = params.get('seed', None)
        self.dt = 1.0 / self.max_timesteps
        # self.storage_capacity = params['storage_capacity']
        self.V_min = params['V_min']
        self.V_max = params['V_max']
        self.V_0 = params['V_0']
        self.I_max = params['I_max']
        self.W_max = params['W_max']
        self.n_months = params['n_months']

        # CHANGE 5: number of independent paths advanced per step().  Default 1 = original.
        self.n_paths = int(params.get('n_paths', 1))

        ## Yan (2002) Model Parameters
        # Short rate: r_t
        self.initial_r = params['initial_r']  # Initial short rate
        self.theta_r = params['theta_r']  # Long-run mean level of r_t
        self.kappa_r = params['kappa_r']  # Speed of mean reversion for r_t
        self.sigma_r = params['sigma_r']  # Volatility coefficient (diffusion) for r_t

        # Convenience Yield: delta_t
        self.initial_delta = params['initial_delta']  # Initial convenience yield
        self.theta_delta = params['theta_delta']  # Long-run mean of delta_t
        self.kappa_delta = params['kappa_delta']  # Speed of mean reversion for delta_t
        self.sigma_delta = params['sigma_delta']  # Volatility coefficient (diffusion) for delta_t

        # Stochastic Variance v_t
        self.initial_v = params['initial_v']  # Initial variance
        self.kappa_v = params['kappa_v']  # Speed of mean reversion for v_t
        self.sigma_v = params['sigma_v']  # Volatility coefficient (diffusion) for v_t
        self.theta_v = params['theta_v']  # Long-run mean of v_t

        # Spot Price Factor S_t
        self.initial_spot_price = params['initial_spot_price']  # Initial (de-seasoned) spot price.
        self.sigma_s = params['sigma_s']  # Factor loading or volatility parameter on S_t

        # Jump Process Parameters
        self.lam = params['lam']  # Jump intensity (Poisson arrival rate)
        self.mu_j = params['mu_j']  # average size of spot-price jumps
        self.sigma_j = params['sigma_j']  # dispersion (volatility) of spot-price jumps
        self.theta = params['theta']  # v_t jump size

        # Correlations among Brownian increments
        self.rho_1 = params['rho_1']  # Correlation between dW_1 and dW_delta
        self.rho_2 = params['rho_2']  # Correlation between dW_2 and dW_v

        # ============================ CHANGE 1: PHYSICAL MEASURE ============================
        # Under the physical measure P the spot drift carries an extra term
        #     phi * (delta_t - delta_bar),        delta_bar = theta_delta / kappa_delta,
        # i.e. the expected spot return rises when the convenience yield is above its
        # long-run mean (normal backwardation / hedging pressure).  phi = 0.0 reproduces
        # the ORIGINAL risk-neutral dynamics exactly.  compute_futures_curve() is a
        # Q-pricing functional and is therefore NOT modified.
        self.phi = params.get('phi', 0.0)
        self.delta_bar = params['theta_delta'] / params['kappa_delta']
        # ====================================================================================

        # ksi_r constant in futures formula
        self.ksi_r = np.sqrt(self.kappa_r ** 2 + 2 * self.sigma_r ** 2)

        ## Penalty parameters
        self.penalty_lambda1 = params['penalty_lambda1']  # For inequality constraint violation
        self.penalty_lambda2 = params['penalty_lambda2']  # For final sum violation
        self.penalty_lambda_riv = params['penalty_lambda_riv']

        # Seasonal Factors (Month 1 is April, Month 12 is March)
        self.seasonal_factors = np.asarray(params['monthly_seasonal_factors'], dtype=np.float64)

        # Set the seed for reproducibility
        self.seed(self.seed_value)

        # ----- ACTION SPACE ----- (UNCHANGED)
        low = np.array([0.0] + [-self.W_max] * (self.n_months - 2) + [-self.W_max])
        high = np.array([self.I_max] + [self.I_max] * (self.n_months - 2) + [0.0])
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        # ============================ CHANGE 2: 33-DIM STATE ================================
        self.N_STATE = params.get('n_state', 33)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate(([0], [-np.inf] * (self.N_STATE - 2), [self.V_min])),
            high=np.concatenate(([12], [np.inf] * (self.N_STATE - 2), [self.V_max])),
            shape=(self.N_STATE,), dtype=np.float32, seed=self.seed_value
        )
        # ====================================================================================

        # ---------------- CHANGE 5(a): tabulate the Yan beta functions once ----------------
        # tau[d, k] = ((k+1)*30 - d)/360 for d = 0..360, k = 0..11.  beta_0, beta_r and
        # beta_delta below are transcribed verbatim from the original
        # compute_futures_curve(); only the loop over (day, k) has been lifted out.
        d = np.arange(self.max_timesteps, dtype=np.float64)[:, None]  # (361, 1)
        kk = np.arange(12, dtype=np.float64)[None, :]  # (1, 12)
        tau = ((kk + 1) * 30.0 - d) / 360.0  # (361, 12)
        self._LIVE = tau >= 0
        taus = np.where(self._LIVE, tau, 0.0)
        e_ksi = np.exp(-self.ksi_r * taus)
        e_kd = np.exp(-self.kappa_delta * taus)
        self._BR = (2 * (1 - e_ksi)) / (2 * self.ksi_r - (self.ksi_r - self.kappa_r) * (1 - e_ksi))
        self._BD = -(1 - e_kd) / self.kappa_delta
        sd2 = self.sigma_delta ** 2
        c = self.sigma_s * self.sigma_delta * self.rho_1 + self.theta_delta
        self._B0 = ((self.theta_r / self.sigma_r ** 2)
                    * (2 * np.log(1 - (self.ksi_r - self.kappa_r) * (1 - e_ksi) / (2 * self.ksi_r))
                       + (self.ksi_r - self.kappa_r) * taus)
                    + (sd2 * taus) / (2 * self.kappa_delta ** 2)
                    - c * taus / self.kappa_delta
                    - c * e_kd / self.kappa_delta ** 2
                    + (4 * sd2 * e_kd - sd2 * np.exp(-2 * self.kappa_delta * taus)) / (4 * self.kappa_delta ** 3)
                    + c / self.kappa_delta ** 2
                    - 3 * sd2 / (4 * self.kappa_delta ** 3))
        # ----------------------------------------------------------------------------------

        # Initialize environment variables
        self.reset()

    def seed(self, seed=None):
        """
        Seed the environment, ensuring reproducibility of the randomness in the environment.
        """
        if seed is not None:
            self.seed_value = seed  # Update seed if provided
        self.W = np.random.default_rng(seed=self.seed_value)  # Seed the random generator
        return [self.seed_value]

    def compute_futures_curve(self):
        """
        Computes the futures curve at the current day t, ensuring:
        - The length of the futures curve is always 12.
        - Expired futures are replaced with 0.0.
        - Futures prices are determined using the Yan (2002) model.

        CHANGE 5(a): identical formula, now a table lookup + one vectorised exp over all
        paths and all 12 contracts.  Returns (n_paths, 12).
        """
        live = self._LIVE[self.day]  # (12,)
        F = np.exp(np.log(np.maximum(self.S_t, 1e-300))[:, None]
                   + self.seasonal_factors[None, :]
                   + self._B0[self.day][None, :]
                   + self._BR[self.day][None, :] * self.r_t[:, None]
                   + self._BD[self.day][None, :] * self.delta_t[:, None])
        return (np.where(live[None, :], F, 0.0)).astype(np.float32)

    # ---------------------------------------------------------------------------------
    # CHANGE 5: internal helper.  Solves the INTRINSIC linear program of the original
    # compute_episode_riv() for one (prices, inventory, month) triple.  It is the exact
    # same scipy.optimize.linprog call with the exact same constraint matrices that the
    # original code built inline; it is factored out only because it is now needed from
    # two places (compute_episode_riv, and the expert-guided exploration branch of
    # NormalNoiseStrategy.select_action).  No behaviour is changed.
    # ---------------------------------------------------------------------------------
    def _intrinsic_lp(self, prices, V_t, month):
        n = self.n_months
        L_n = np.tril(np.ones((n, n)))
        A = np.vstack([L_n, -L_n])
        b = np.hstack([(self.V_max - V_t) * np.ones(n), -(self.V_min - V_t) * np.ones(n)])
        A_eq = np.ones((1, n))
        b_eq = np.array([0.0 - V_t])
        bounds = [(-self.W_max, self.I_max)] * n
        if month > 0:
            zero_price_indices = (prices == 0)
            bounds = [(0, 0) if zero_price_indices[k] else bounds[k] for k in range(n)]
        out = linprog(prices, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
        return out.x if out.x is not None else np.zeros(n)

    def compute_episode_riv(self):
        """
        Compute Rolling Intrinsic + Extrinsic Value (RIV) using LP at decision times.

        CHANGE 5: returns an ARRAY of length n_paths instead of a scalar.  The linear
        program, the constraint matrices, the decision-time grid and the cash-flow
        accounting are byte-for-byte the original ones; only the outer loop over paths
        is new.  F_trajectory now holds the 13 decision-day curves (see class docstring),
        so `decision_times` is simply range(13).
        """
        n = self.n_months
        Fdec = np.stack(self.F_trajectory, axis=1)  # (n_paths, 13, 12)
        N = Fdec.shape[0]
        CF = np.zeros(N)
        for p in range(N):
            Fp = Fdec[p]
            V_t = float(self.V_0)
            X_tau = np.zeros((13, n))
            for i in range(13):
                if i == 0:
                    X_tau[i] = self._intrinsic_lp(Fp[0].astype(np.float64), V_t, 0)
                    continue
                CF[p] += np.dot(Fp[i] - Fp[i - 1], X_tau[i - 1])
                if i < 12:
                    X_tau[i] = self._intrinsic_lp(Fp[i].astype(np.float64), V_t, i)
                else:
                    X_tau[i] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -V_t])
                    CF[p] += -Fp[i][i - 1] * X_tau[i, i - 1]
                V_t += X_tau[i, i - 1]
        return CF

    # ============================ CHANGE 2: 33-DIM STATE ================================
    def _build_state(self):
        """Observation handed to the agent.  CHANGE 5: returns (n_paths, N_STATE).

        N_STATE == 14 -> the original [month, F^(1..12), V_t].
        N_STATE == 33 -> the extended state, laid out as

            0      month / 12
            1-12   F^(k) / Fbar                       (Fbar = 20, a fixed price scale)
            13     V_t
            14-25  one-month log returns ln(F^(k)_t / F^(k)_{t-1}), 0 if either leg is dead
            26     slope      (F_back - F_front) / Fbar
            27     curvature  [(F_front + F_back)/2 - mean of live legs] / Fbar
            28     spread     (max - min over live legs) / Fbar
            29     realised-volatility proxy: mean |log return| over live legs
            30-31  sin(2 pi t / 12), cos(2 pi t / 12)
            32     V_t / (R_t * W_max), R_t = max(11 - max(t-1, 0), 1)

        The slope block is the operative one: because beta_delta(tau) < 0, the slope is a
        decreasing affine read-out of the convenience yield once the level is removed, and
        the convenience yield is what drives the premium introduced in CHANGE 1.
        """
        Fbar = 20.0
        t = float(self.month)
        F = np.asarray(self.F_t, dtype=np.float64)  # (N, 12)
        Fp = np.asarray(self.F_prev, dtype=np.float64)  # (N, 12)
        N = F.shape[0]
        if self.N_STATE == 14:
            return np.concatenate(
                (np.full((N, 1), t), F, self.V_t[:, None]), axis=1).astype(np.float32)

        alive = self._LIVE[self.day]  # (12,) shared by all paths
        na = max(int(alive.sum()), 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            ret = np.where((F > 0) & (Fp > 0),
                           np.log(np.maximum(F, 1e-9) / np.maximum(Fp, 1e-9)), 0.0)
        ret = np.nan_to_num(ret)
        if alive.any():
            idx = np.flatnonzero(alive)
            front, back = F[:, idx[0]], F[:, idx[-1]]
            mx, mn = F[:, alive].max(axis=1), F[:, alive].min(axis=1)
        else:
            front = back = mx = mn = np.zeros(N)
        mid = (F * alive[None, :]).sum(axis=1) / na
        slope = (back - front) / Fbar
        curv = ((front + back) / 2.0 - mid) / Fbar
        spread = (mx - mn) / Fbar if alive.any() else np.zeros(N)
        rv = np.abs(ret).sum(axis=1) / na
        R = max(11 - max(t - 1, 0), 1)
        vrel = self.V_t / (R * self.W_max)
        col = lambda a: np.asarray(a, dtype=np.float64).reshape(N, 1)
        return np.concatenate((
            np.full((N, 1), t / 12.0), F / Fbar, col(self.V_t), ret,
            col(slope), col(curv), col(spread), col(rv),
            np.full((N, 1), np.sin(2 * np.pi * t / 12.0)),
            np.full((N, 1), np.cos(2 * np.pi * t / 12.0)), col(vrel)
        ), axis=1).astype(np.float32)

    # ====================================================================================

    def reset(self):
        N = self.n_paths
        self.month = 0
        self.day = 0

        self.S_t = np.full(N, self.initial_spot_price, dtype=np.float64)
        self.r_t = np.full(N, self.initial_r, dtype=np.float64)
        self.delta_t = np.full(N, self.initial_delta, dtype=np.float64)
        self.v_t = np.full(N, self.initial_v, dtype=np.float64)
        self.F_t = self.compute_futures_curve()

        # CHANGE 5(b): decision-day curves only (day 0, 30, ..., 360).
        self.S_trajectory = [self.S_t.copy()]
        self.r_trajectory = [self.r_t.copy()]
        self.delta_trajectory = [self.delta_t.copy()]
        self.v_trajectory = [self.v_t.copy()]
        self.F_trajectory = [self.F_t.copy()]

        self.V_t = np.full(N, float(self.V_0), dtype=np.float64)

        self.rl_cumulative_reward = np.zeros(N)

        # CHANGE 2: at t = 0 there is no previous curve, so returns are taken against the
        # current one (i.e. they are zero), and the observation is built by _build_state().
        self.F_prev = self.F_t
        return self._build_state(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.ndim == 1:
            action = action[None, :]
        assert action.shape[1] == self.n_months, "Action must have length = n_months"
        N = self.n_paths

        is_terminal = False
        is_truncated = False

        # ---- APPLY ACTION AND CHECK CONSTRAINTS ----
        # CHANGE 5: identical logic, now over the path axis.  cost1/cost2 are arrays.
        running_sum = np.zeros(N)
        cost1 = np.zeros(N)
        cost2 = np.zeros(N)
        riv_penalty = np.zeros(N)
        reward = np.zeros(N)
        m0 = max(self.month - 1, 0)
        for i in range(m0, self.n_months):
            running_sum = running_sum + action[:, i]
            new_volume = self.V_t + running_sum

            # ---- SOFT REJECT: Cumulative constraint violation ----
            viol = (new_volume + 0.001 < self.V_min) | (new_volume - 0.001 > self.V_max)
            cost1 = cost1 + np.where(
                viol,
                np.minimum(new_volume - self.V_min, self.V_max - new_volume) * self.penalty_lambda1,
                0.0)
            # Except for the first loop, updating V_t: V_t = V_{t-1} + X_{t}^t
            if self.month != 0 and i == m0:
                self.V_t = new_volume
                running_sum = np.zeros(N)

        whole_volume = new_volume

        # ---- FINAL STORAGE BALANCE CONSTRAINT (SOFT PENALTY) ----
        cost2 = cost2 + np.where(np.abs(whole_volume) > 0.001,
                                 -np.abs(whole_volume) * self.penalty_lambda2, 0.0)

        reward = reward + cost1
        reward = reward + cost2

        # Store state at time t
        F_t = self.F_t
        # CHANGE 2: remember the curve observed at the START of this month.
        self.F_prev = F_t

        sqdt = np.sqrt(self.dt)
        for _ in range(30):
            # Generate independent Brownian increments (CHANGE 5: size N)
            dW_1 = self.W.normal(0, sqdt, size=N)
            dW_r = self.W.normal(0, sqdt, size=N)
            dW_2 = self.W.normal(0, sqdt, size=N)
            dW_delta = self.rho_1 * dW_1 + np.sqrt(1 - self.rho_1 ** 2) * self.W.normal(0, sqdt, size=N)
            dW_v = self.rho_2 * dW_2 + np.sqrt(1 - self.rho_2 ** 2) * self.W.normal(0, sqdt, size=N)

            # Probability of jump occurrence
            dq = self.W.choice([0, 1], p=[1 - self.lam * self.dt, self.lam * self.dt], size=N)

            # Jump magnitude: ln(1 + J) ~ N[ln(1 + mu_J) - 0.5 * sigma_J^2, sigma_J^2]
            ln_1_plus_J = self.W.normal(np.log(1 + self.mu_j) - 0.5 * self.sigma_j ** 2,
                                        self.sigma_j, size=N)
            J = np.exp(ln_1_plus_J) - 1  # Jump size for the spot price

            J_v = self.W.exponential(scale=self.theta, size=N)

            # CHANGE 1: physical-measure premium.  phi = 0.0 gives the original line back.
            premium = self.phi * (self.delta_t - self.delta_bar)
            dS_t = (self.r_t - self.delta_t - self.lam * self.mu_j + premium) * self.S_t * self.dt \
                   + self.sigma_s * self.S_t * dW_1 \
                   + np.sqrt(np.maximum(self.v_t, 0)) * self.S_t * dW_2 + J * self.S_t * dq
            self.S_t = self.S_t + dS_t

            dr_t = (self.theta_r - self.kappa_r * self.r_t) * self.dt \
                   + self.sigma_r * np.sqrt(np.maximum(self.r_t, 0)) * dW_r
            self.r_t = self.r_t + dr_t

            ddelta_t = (self.theta_delta - self.kappa_delta * self.delta_t) * self.dt \
                       + self.sigma_delta * dW_delta
            self.delta_t = self.delta_t + ddelta_t

            dv_t = (self.theta_v - self.kappa_v * self.v_t) * self.dt \
                   + self.sigma_v * np.sqrt(np.maximum(self.v_t, 0)) * dW_v + J_v * dq
            self.v_t = self.v_t + dv_t

            self.day += 1

        self.F_t = self.compute_futures_curve()
        # CHANGE 5(b): record the decision-day curve only.
        self.S_trajectory.append(self.S_t.copy())
        self.r_trajectory.append(self.r_t.copy())
        self.delta_trajectory.append(self.delta_t.copy())
        self.v_trajectory.append(self.v_t.copy())
        self.F_trajectory.append(self.F_t.copy())

        self.month += 1
        reward = reward + np.einsum('nk,nk->n', (self.F_t - F_t).astype(np.float64), action)
        self.rl_cumulative_reward = self.rl_cumulative_reward + reward
        is_terminal = False
        is_truncated = False
        if self.month == 12:
            is_terminal = True
            reward = reward + (-self.F_t[:, -1].astype(np.float64) * action[:, -1])
            self.rl_cumulative_reward = self.rl_cumulative_reward + reward
            self.V_t = self.V_t + action[:, -1]
            # CHANGE 5: the RIV shaping term is skipped when penalty_lambda_riv == 0.
            # It contributes exactly 0.0 in that case (both branches are multiplied by
            # penalty_lambda_riv), and it is by far the most expensive call in the file.
            if self.penalty_lambda_riv != 0.0:
                riv = self.compute_episode_riv()
                riv_penalty = -self.penalty_lambda_riv * (riv - self.rl_cumulative_reward)
                reward = reward + riv_penalty
        info = {'cost1': cost1, 'cost2': cost2, 'riv_penalty': riv_penalty}
        # CHANGE 2: observation built by _build_state().  The reward above is UNCHANGED.
        return self._build_state(), reward, is_terminal, is_truncated, info


class GreedyStrategy():
    def __init__(self, bounds):
        self.low, self.high = bounds
        self.ratio_noise_injected = 0

    def select_action(self, model, state):
        """CHANGE 6: vectorised.  `state` is (N, nS); the returned action is (N, 12).
        The actor already emits a feasible schedule, so the clip below is a no-op safety
        net exactly as in the original."""
        with torch.no_grad():
            if hasattr(model, "set_noise_active"):
                model.set_noise_active(False)
            greedy_action = model(state).cpu().detach().numpy()
        greedy_action = np.atleast_2d(greedy_action)
        action = np.clip(greedy_action, self.low[None, :], self.high[None, :])
        return action.astype(np.float64)


class FCQV(nn.Module):
    """
    ===================== CHANGE 10: CRITIC ARCHITECTURE ==============================
    Same class name, same constructor signature (input_dim, output_dim, hidden_dims,
    activation_fc), same forward(state, action) -> (B, 1), same load().  What changed:

      * the action is concatenated with the state at the INPUT layer.  The previous
        version embedded the state through Linear + LayerNorm and only then concatenated
        the raw action.  After LayerNorm the 512 state features have unit scale while the
        twelve action entries live in [-0.4, 0.4], so the action contributed a ~0.4-scale
        signal among unit-scale features.  DDPG's actor gradient is dQ/da, so weakening
        the action pathway weakens exactly the signal that trains the policy.

      * LayerNorm removed and leaky_relu -> relu.  With reward_scale = 10 the critic
        targets are O(0.3) and need no internal normalisation.

      * default hidden_dims (512, 512, 256, 128) -> (256, 256, 256).

    This is the critic that produced the reported 3.914.
    ==================================================================================
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dims=(256, 256, 256),
                 activation_fc=F.relu):
        super(FCQV, self).__init__()
        self.activation_fc = activation_fc

        layers, prev = [], input_dim + output_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)

    def _format(self, state, action):
        x, u = state, action
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
            if x.ndim == 1:
                x = x.unsqueeze(0)
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, device=self.device, dtype=torch.float32)
            if u.ndim == 1:
                u = u.unsqueeze(0)
        return x, u

    def forward(self, state, action):
        x, u = self._format(state, action)
        return self.net(torch.cat([x, u], dim=1))

    def load(self, experiences):
        states, actions, rewards, new_states, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, rewards, new_states, is_terminals


class FCDPAutoregressive(nn.Module):
    """
    ===================== CHANGE 10: ACTOR ARCHITECTURE ===============================
    Same class name, same constructor signature (input_dim, action_bounds, hidden_dims,
    activation_fc), same forward(state) -> (B, 12) schedule.  What changed:

      * the eleven per-leg output heads with a running-inventory input are replaced by a
        single trunk emitting a twelve-dimensional LATENT z, which is then pushed through
        the feasibility map below.  The leg-by-leg admissible interval, including the
        reachability term, is IDENTICAL to your "MINIMAL REACHABILITY FIX"; only the way
        the network parameterises a point inside that interval changed.  Eleven separate
        heads whose inputs depend on the previous head's output make the actor gradient a
        product of eleven Jacobians, which is what was training slowly.

      * squashing tanh -> sigmoid, so a zero latent maps to the midpoint of the interval
        rather than requiring the head to learn an offset.

      * default hidden_dims (512, 512, 256, 128) -> (256, 256, 256), relu.

      * `_explore_noise`: exploration is now injected into the LATENT z, before the
        feasibility map, so a perturbed action is still exactly feasible.  It is set by
        NormalNoiseStrategy.select_action and cleared immediately afterwards; it is 0.0
        during every optimisation step and every greedy evaluation.

    The feasibility map: at leg k, with Vc the inventory accumulated so far and
    R = 11 - k the number of months left after k,

        lo = max(-W_max, V_min - Vc)
        hi = min( I_max, V_max - Vc, R*W_max - Vc)
        x_k = lo + (hi - lo) * sigmoid(z_k),   pinned to 0 while k < start

    The third term in `hi` is what makes the terminal balance a property of the
    parameterisation rather than an assumption: at k = 10 it forces Vc <= W_max, so at
    k = 11 the interval collapses to the single point -Vc and sum(x) = -V_t exactly.
    ==================================================================================
    """

    def __init__(self,
                 input_dim,
                 action_bounds,
                 hidden_dims=(256, 256, 256),
                 activation_fc=F.relu):
        super(FCDPAutoregressive, self).__init__()
        self.activation_fc = activation_fc
        self.env_min, self.env_max = action_bounds

        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers += [nn.Linear(prev, len(self.env_max))]
        self.f = nn.Sequential(*layers)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.to(self.device)

        self.env_min = torch.tensor(self.env_min, device=self.device, dtype=torch.float32)
        self.env_max = torch.tensor(self.env_max, device=self.device, dtype=torch.float32)

        self.V_min = torch.tensor(0.0, device=self.device)
        self.V_max = torch.tensor(1.0, device=self.device)
        self.I_max = torch.tensor(0.4, device=self.device)
        self.W_max = torch.tensor(0.4, device=self.device)

        self._explore_noise = 0.0

    def set_noise_active(self, flag):
        """Hook already called by GreedyStrategy.select_action."""
        if not flag:
            self._explore_noise = 0.0

    def _normalize(self, state):
        # ===================== CHANGE 3: state indexing for the 33-dim state ==============
        # The extended observation is already scaled at source (month/12, prices/20,
        # returns, slope, ...), so re-normalising it here would destroy the very slope
        # information the agent needs.  For the original 14-dim observation the historical
        # normalisation is applied unchanged.
        if state.shape[1] != 14:
            return state
        state = state.clone()
        state[:, 0] /= 11.0
        state[:, 1:13] /= torch.clamp(state[:, 1:13].max(dim=1, keepdim=True).values, min=1e-6)
        state[:, -1] /= self.V_max
        return state
        # ==================================================================================

    def _format(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        return self._normalize(state)

    def _feasible_map(self, z, start, V):
        """Differentiable surjection from R^12 onto the storage polytope."""
        Vc = V
        xs = []
        for k in range(12):
            R = 11 - k
            lo = torch.maximum(torch.full_like(Vc, -float(self.W_max)), self.V_min - Vc)
            hi = torch.minimum(torch.minimum(torch.full_like(Vc, float(self.I_max)),
                                             self.V_max - Vc),
                               R * float(self.W_max) - Vc)
            hi = torch.maximum(hi, lo)
            xk = lo + (hi - lo) * torch.sigmoid(z[:, k])
            xk = xk * (k >= start).to(z.dtype)
            xs.append(xk)
            Vc = Vc + xk
        return torch.stack(xs, dim=1)

    def forward(self, state):
        state = self._format(state)
        # ===================== CHANGE 3: read month and inventory by layout ===============
        if state.shape[1] == 14:
            t = (state[:, 0] * 11).round().long()
            V_t = state[:, -1] * self.V_max
        else:
            t = (state[:, 0] * 12).round().long()
            V_t = state[:, 13]
        # ==================================================================================
        z = self.f(state)
        if self._explore_noise > 0.0:
            z = z + self._explore_noise * torch.randn_like(z)
        start = torch.clamp(t - 1, min=0)
        return self._feasible_map(z, start, V_t)


class PrioritizedReplayBuffer():
    """
    ===================== CHANGE 7: ARRAY-BACKED, BULK STORE ==========================
    Same class, same method names, same prioritised-replay semantics (proportional
    priorities, alpha, annealed beta, importance weights, priority refresh from the TD
    error).  Two interior changes, both forced by the vectorised rollout:

      (a) the backing store was `np.empty((max_samples, 2), dtype=np.ndarray)`, an object
          array holding one Python tuple per transition.  At 4x10^5 entries the
          per-sample boxing dominated the run time.  It is now five preallocated
          contiguous arrays (states, actions, rewards, next states, terminal flags) plus
          one float array of priorities.  Nothing about the algorithm changes.

      (b) store() accepts a BATCH.  If the state argument is 2-D it stores the whole
          block of N transitions in one ring-buffer write instead of N calls.

      (c) sampling without replacement proportional to p^alpha used
          np.random.choice(..., replace=False, p=probs), which is O(n * batch) and takes
          seconds at n = 4x10^5.  It is replaced by the Gumbel top-k trick,
          argpartition(log p + Gumbel), which draws from EXACTLY the same distribution
          (successive proportional sampling without replacement) in one O(n) pass.
    ==================================================================================
    """

    def __init__(self,
                 max_samples=10000,
                 batch_size=64,
                 rank_based=False,
                 alpha=0.6,
                 beta0=0.1,
                 beta_rate=0.99992):
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.n_entries = 0
        self.next_index = 0
        self.td_error_index = 0
        self.sample_index = 1
        self.rank_based = rank_based  # if not rank_based, then proportional
        self.alpha = alpha  # how much prioritization to use 0 is uniform, 1 is full priority
        self.beta = beta0  # bias correction 0 is no correction 1 is full correction
        self.beta0 = beta0
        self.beta_rate = beta_rate
        # CHANGE 7(a): contiguous storage, allocated lazily on the first store().
        self._prio = np.zeros(max_samples, dtype=np.float64)
        self._s = self._a = self._r = self._s2 = self._d = None
        self._rng = np.random.default_rng(0)

    def _alloc(self, nS, nA):
        m = self.max_samples
        self._s = np.zeros((m, nS), dtype=np.float32)
        self._a = np.zeros((m, nA), dtype=np.float32)
        self._r = np.zeros((m, 1), dtype=np.float32)
        self._s2 = np.zeros((m, nS), dtype=np.float32)
        self._d = np.zeros((m, 1), dtype=np.float32)

    def update(self, idxs, td_errors):
        self._prio[np.asarray(idxs).ravel()] = np.abs(np.asarray(td_errors)).ravel()

    def store(self, sample):
        """CHANGE 7(b): `sample` may be a single transition or a batch of N."""
        state, action, reward, new_state, is_failure = sample
        state = np.atleast_2d(np.asarray(state, dtype=np.float32))
        action = np.atleast_2d(np.asarray(action, dtype=np.float32))
        new_state = np.atleast_2d(np.asarray(new_state, dtype=np.float32))
        reward = np.asarray(reward, dtype=np.float32).reshape(-1, 1)
        done = np.asarray(is_failure, dtype=np.float32).reshape(-1, 1)
        n = state.shape[0]
        if self._s is None:
            self._alloc(state.shape[1], action.shape[1])

        priority = 1.0 if self.n_entries == 0 else float(self._prio[:self.n_entries].max())
        # ring-buffer write, wrapping if the block crosses the end
        pos = (self.next_index + np.arange(n)) % self.max_samples
        self._s[pos] = state
        self._a[pos] = action
        self._r[pos] = reward
        self._s2[pos] = new_state
        self._d[pos] = done
        self._prio[pos] = priority
        self.n_entries = min(self.n_entries + n, self.max_samples)
        self.next_index = (self.next_index + n) % self.max_samples

    def _update_beta(self):
        self.beta = min(1.0, self.beta * self.beta_rate ** -1)
        return self.beta

    def sample(self, batch_size=None):
        batch_size = self.batch_size if batch_size == None else batch_size
        self._update_beta()
        n = self.n_entries

        if self.rank_based:
            priorities = 1 / (np.arange(n) + 1)
        else:  # proportional
            priorities = self._prio[:n] + EPS
        scaled_priorities = priorities ** self.alpha
        probs = scaled_priorities / scaled_priorities.sum()

        weights = (n * probs) ** -self.beta
        normalized_weights = weights / weights.max()
        # CHANGE 7(c): Gumbel top-k == sampling without replacement proportional to probs
        k = min(batch_size, n)
        g = self._rng.gumbel(size=n)
        idxs = np.argpartition(np.log(probs) + g, -k)[-k:]

        samples_stacks = [self._s[idxs], self._a[idxs], self._r[idxs],
                          self._s2[idxs], self._d[idxs]]
        idxs_stack = idxs.reshape(-1, 1)
        weights_stack = normalized_weights[idxs].reshape(-1, 1)
        return idxs_stack, weights_stack, samples_stacks

    def __len__(self):
        return self.n_entries

    def __repr__(self):
        return f'PrioritizedReplayBuffer(n_entries={self.n_entries})'

    def __str__(self):
        return self.__repr__()


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
        """CHANGE 8: one vectorised transition for ALL n_paths at once."""
        min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
        action = self.training_strategy.select_action(self.online_policy_model,
                                                      state,
                                                      len(self.replay_buffer) < min_samples)
        new_state, reward, is_terminal, is_truncated, info = env.step(action)
        is_failure = is_terminal and not is_truncated
        n = np.asarray(reward).size
        # CHANGE 8: `reward_scale` divides the reward stored for the CRITIC only.  It is a
        # change of units for Q, exactly equivalent to rescaling the critic's output, and
        # leaves the optimal policy and every reported P&L untouched (all P&L figures are
        # summed from env.step's raw reward).  reward_scale = 1.0 restores the original.
        experience = (state, action, np.asarray(reward) / self.reward_scale, new_state,
                      np.full(n, float(is_failure)))
        self.replay_buffer.store(experience)
        self._ep_reward = self._ep_reward + np.asarray(reward)
        self._ep_steps += 1
        self._ep_expl += self.training_strategy.ratio_noise_injected
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

    # def train(self, make_env_fn, make_env_kargs, seed, gamma,
    def train(self, env, seed, gamma,
              max_minutes, max_episodes, goal_mean_500_reward):
        """
        ===================== CHANGE 8: ROUND-BASED VECTORISED TRAINING ===================
        The signature, the returned tuple and the debug line are unchanged.  What changed
        inside:

          * `env` now advances env.n_paths independent paths per step(), so one pass over
            the 12 decision dates produces 12 * n_paths transitions instead of 12.  A
            "round" is one such pass; `max_episodes` is still counted in PATH-EPISODES, so
            the number of rounds is max_episodes // n_paths.

          * gradient steps are decoupled from environment steps.  The original took one
            optimisation step per environment step (12 per episode); here each round does
            `iters_per_round` steps drawn from the replay buffer.  This is the update-to-
            data ratio, and it is what makes a vectorised rollout worth doing at all.

          * the exploration noise and the expert ratio are annealed once per ROUND rather
            than once per episode, so `max_episode` passed to NormalNoiseStrategy should
            be the number of rounds.

          * model selection: the actor that scores best on a FIXED validation environment
            (`validation_env`, common random numbers, evaluated every round) is kept and
            reloaded at the end.  The original returned whatever the last episode left
            behind.

        Everything else -- the networks, the optimisers, the Polyak update, the prioritised
        replay, the reward, the action space -- is the original code.
        ==================================================================================
        """
        training_start, last_debug_time = time.time(), float('-inf')

        print(f"Running on: {os.uname().nodename}")
        self.seed = seed
        self.gamma = gamma
        self.reward_scale = float(globals().get('reward_scale', 1.0))

        torch.manual_seed(self.seed);
        np.random.seed(self.seed);
        random.seed(self.seed)

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
        self.training_strategy = training_strategy_fn(action_bounds)
        self.evaluation_strategy = evaluation_strategy_fn(action_bounds)

        # ---- CHANGE 8: round bookkeeping -------------------------------------------
        n_paths = env.n_paths
        n_rounds = max(int(max_episodes) // n_paths, 1)
        iters_per_round = int(globals().get('iters_per_round', 3000))
        e0 = float(globals().get('expert_ratio0', 0.0))
        e1 = float(globals().get('expert_ratio1', 0.0))
        val_env = globals().get('validation_env', None)
        self.training_strategy.expert_env = env
        self.training_strategy.rng = np.random.default_rng(self.seed)
        self.replay_buffer._rng = np.random.default_rng(self.seed + 1)
        best_val, best_state = -np.inf, None
        # -----------------------------------------------------------------------------

        result = np.empty((n_rounds * n_paths, 5))
        result[:] = np.nan
        training_time = 0
        episode = 0
        for rd in range(n_rounds):
            episode_start = time.time()
            frac = rd / max(n_rounds - 1, 1)
            self.training_strategy.expert_ratio = e0 + (e1 - e0) * frac

            state, _ = env.reset()
            self._ep_reward = np.zeros(n_paths)
            self._ep_steps = 0
            self._ep_expl = 0.0
            for step in count():
                state, is_terminal = self.interaction_step(state, env)
                if is_terminal:
                    gc.collect()
                    break

            min_samples = self.replay_buffer.batch_size * self.n_warmup_batches
            if len(self.replay_buffer) > min_samples:
                for it in range(iters_per_round):
                    idxs_weights_samples = self.replay_buffer.sample()
                    samples = self.online_value_model.load(idxs_weights_samples[2])
                    self.optimize_model((idxs_weights_samples[0], idxs_weights_samples[1], samples))
                    if (it + 1) % self.update_target_every_steps == 0:
                        self.update_networks()
            self.training_strategy.decay_step()

            # stats -- one entry per PATH-episode so the log means the same thing
            episode_elapsed = time.time() - episode_start
            self.episode_seconds.extend([episode_elapsed / n_paths] * n_paths)
            training_time += episode_elapsed
            self.episode_reward.extend(list(self._ep_reward))
            self.episode_timestep.extend([self._ep_steps] * n_paths)
            self.episode_exploration.extend([self._ep_expl] * n_paths)

            if val_env is not None:
                evaluation_score, _ = self.evaluate(self.online_policy_model, val_env)
                if evaluation_score > best_val:
                    best_val = evaluation_score
                    best_state = {k: v.detach().clone()
                                  for k, v in self.online_policy_model.state_dict().items()}
            else:
                evaluation_score, _ = self.evaluate(self.online_policy_model, env)
            self.evaluation_scores.extend([evaluation_score] * n_paths)
            episode += n_paths

            total_step = int(np.sum(self.episode_timestep))
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])
            mean_500_reward = np.mean(self.episode_reward[-500:])
            std_500_reward = np.std(self.episode_reward[-500:])
            mean_500_eval_score = np.mean(self.evaluation_scores[-500:])
            std_500_eval_score = np.std(self.evaluation_scores[-500:])
            lst_500_exp_rat = np.array(
                self.episode_exploration[-500:]) / np.array(self.episode_timestep[-500:])
            mean_500_exp_rat = np.mean(lst_500_exp_rat)
            std_500_exp_rat = np.std(lst_500_exp_rat)

            wallclock_elapsed = time.time() - training_start
            result[episode - 1] = total_step, mean_500_reward, \
                mean_500_eval_score, training_time, wallclock_elapsed

            reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS
            reached_max_minutes = wallclock_elapsed >= max_minutes * 60
            reached_max_episodes = episode >= max_episodes
            reached_goal_mean_reward = mean_500_eval_score >= goal_mean_500_reward
            training_is_over = reached_max_minutes or \
                               reached_max_episodes or \
                               reached_goal_mean_reward

            debug_message = 'el {}, ep {:04}, ts {:07}, '
            debug_message += 'ar 100 {:05.1f}±{:05.1f}, '
            debug_message += '500 {:05.1f}±{:05.1f}, '
            debug_message += 'ex 500 {:02.1f}±{:02.1f}, '
            debug_message += 'ev {:05.1f}±{:05.1f}'
            debug_message = debug_message.format(
                str(datetime.timedelta(seconds=int(wallclock_elapsed))).rjust(8, '0'),
                episode - 1, total_step, mean_100_reward, std_100_reward,
                mean_500_reward, std_500_reward, mean_500_exp_rat, std_500_exp_rat,
                mean_500_eval_score, std_500_eval_score)
            print(ERASE_LINE + debug_message, flush=True)
            if reached_debug_time or training_is_over:
                last_debug_time = time.time()
            if training_is_over:
                if reached_max_minutes: print(u'--> reached_max_minutes ✕')
                if reached_max_episodes: print(u'--> reached_max_episodes ✕')
                if reached_goal_mean_reward: print(u'--> reached_goal_mean_reward ✓')
                break

        # CHANGE 8: restore the best-on-validation actor before the final evaluation.
        if best_state is not None:
            self.online_policy_model.load_state_dict(best_state)
            print(f'--> restored best-on-validation actor (val P&L {best_val:.4f})')

        final_eval_score, score_std = self.evaluate(self.online_policy_model, env)
        wallclock_time = time.time() - training_start
        print('Training complete.')
        print('Final evaluation score {:.2f}±{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(
            final_eval_score, score_std, training_time, wallclock_time))
        return result, final_eval_score, training_time, wallclock_time

    def evaluate(self, eval_policy_model, eval_env, n_episodes=1, paired_riv=False):
        # ===================== CHANGE 4 + CHANGE 8: vectorised paired evaluation ==========
        # One call runs eval_env.n_paths complete episodes greedily and returns the mean
        # and standard deviation of the COMPLETE pathwise P&L, equation (26).  With
        # paired_riv=True the rolling-intrinsic value of the SAME realised price paths is
        # recorded alongside via the environment's own compute_episode_riv(), so the
        # DDPG - RIV difference is a paired statistic under common random numbers.
        # `n_episodes` is kept in the signature for compatibility and repeats the sweep.
        # =================================================================================
        rs, rivs = [], []
        for _ in range(n_episodes):
            s, _ = eval_env.reset()
            tot = np.zeros(eval_env.n_paths)
            for _ in count():
                a = self.evaluation_strategy.select_action(eval_policy_model, s)
                s, r, d, _, _ = eval_env.step(a)
                tot = tot + np.asarray(r)
                if d:
                    break
            rs.append(tot)
            if paired_riv:
                rivs.append(eval_env.compute_episode_riv())
        rs = np.concatenate(rs)
        if paired_riv:
            rivs = np.concatenate(rivs)
            diff = rs - rivs
            return dict(ddpg=rs.mean(), ddpg_se=rs.std(ddof=1) / np.sqrt(len(rs)),
                        ddpg_sd=rs.std(ddof=1),
                        riv=rivs.mean(), riv_se=rivs.std(ddof=1) / np.sqrt(len(rivs)),
                        riv_sd=rivs.std(ddof=1),
                        diff=diff.mean(), diff_se=diff.std(ddof=1) / np.sqrt(len(diff)),
                        win=float((diff > 0).mean()), n=len(rs),
                        ddpg_cvar5=float(rs[rs <= np.quantile(rs, .05)].mean()),
                        riv_cvar5=float(rivs[rivs <= np.quantile(rivs, .05)].mean()))
        return float(np.mean(rs)), float(np.std(rs))

    def get_cleaned_checkpoints(self, n_checkpoints=4):
        try:
            return self.checkpoint_paths
        except AttributeError:
            self.checkpoint_paths = {}

        paths = glob.glob(os.path.join(self.checkpoint_dir, '*.tar'))
        paths_dic = {int(path.split('.')[-2]): path for path in paths}
        last_ep = max(paths_dic.keys())
        # checkpoint_idxs = np.geomspace(1, last_ep+1, n_checkpoints, endpoint=True, dtype=np.int)-1
        checkpoint_idxs = np.linspace(1, last_ep + 1, n_checkpoints, endpoint=True, dtype=int) - 1

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
    Exploration strategy for DDPG.

    ===================== CHANGE 6 + CHANGE 11 =======================================
    Same class name, same constructor signature, same select_action(model, state,
    max_exploration) contract.  Interior changes:

      (a) vectorised: `state` is (N, nS) and the returned action is (N, 12).

      (b) the noise is injected into the actor's LATENT vector (via the actor's
          `_explore_noise` hook), before the feasibility map, instead of being added to
          the twelve flows and clipped afterwards.  A perturbed action is therefore
          still exactly feasible: cost1 = cost2 = 0 for every exploratory episode and
          the reward the agent sees is exactly equation (26).  This is the "deal with
          the constraints in another way" change -- projection by construction rather
          than penalisation after the fact.

      (c) expert-guided exploration: with probability `expert_ratio` a path takes the
          INTRINSIC linear-programming action instead of the noisy actor action.  It is
          the same LP the rolling-intrinsic benchmark solves.  Pure exploration: it
          shapes which transitions enter the replay buffer, and touches neither the
          reward, the action set, nor the evaluation policy.  0.0 switches it off.

      (d) CHANGE 11: the noise level is interpolated LINEARLY between
          exploration_noise_ratio and final_noise_ratio over `episodes_with_noise`,
          rather than geometrically.  This is the schedule used for the reported run.
    ==================================================================================
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

        self.episodes_with_noise = max(max_episode - noise_free_last, 1)

        self.current_episode = 0
        self.ratio_noise_injected = 0.0  # for diagnostics

        # CHANGE 6(c)
        self.expert_ratio = 0.0
        self.expert_env = None
        self.rng = np.random.default_rng(0)
        # latent noise used while the replay buffer is below the warm-up threshold
        self.max_exploration_noise = 3.0

    def decay_step(self):
        """Call after each round to advance the noise schedule."""
        self.current_episode += 1

    @property
    def noise_ratio(self):
        """CHANGE 11: linear interpolation from exploration_ to final_noise_ratio."""
        f = min(self.current_episode / self.episodes_with_noise, 1.0)
        return self.exploration_noise_ratio + \
            (self.final_noise_ratio - self.exploration_noise_ratio) * f

    def _mask_action(self, action, state):
        """Masks actions based on problem-specific logic using F_t."""
        # CHANGE 3: the futures curve is state[:, 1:13] in BOTH layouts.
        F_t = np.asarray(state)[:, 1:13]
        mask = (F_t != 0)
        action = np.where(mask, action, action)  # the map already pins expired legs
        return action, mask

    def _compute_noise_diagnostics(self, action, greedy_action, mask):
        if np.any(mask):
            ratio = np.abs((greedy_action - action) / self.action_range[None, :])
            self.ratio_noise_injected = float(ratio[mask].mean())
        else:
            self.ratio_noise_injected = 0.0

    def select_action(self, model, state, max_exploration=False):
        """Returns a feasible, noisy schedule of shape (N, 12)."""
        state = np.atleast_2d(np.asarray(state, dtype=np.float32))
        N = state.shape[0]
        nz = self.max_exploration_noise if max_exploration else self.noise_ratio

        with torch.no_grad():
            model._explore_noise = 0.0
            greedy_action = np.atleast_2d(model(state).cpu().numpy()).astype(np.float64)
            model._explore_noise = float(nz)
            action = np.atleast_2d(model(state).cpu().numpy()).astype(np.float64)
            model._explore_noise = 0.0
        if np.isnan(action).any():
            print("!! Warning: NaN detected in action")

        # ---- CHANGE 6(c): expert-guided exploration ----
        if (not max_exploration) and self.expert_ratio > 0.0 and self.expert_env is not None:
            if state.shape[1] == 14:
                t = int(round(float(state[0, 0])))
                V_t = state[:, -1].astype(np.float64)
                prices = state[:, 1:13].astype(np.float64)
            else:
                t = int(round(float(state[0, 0]) * 12))
                V_t = state[:, 13].astype(np.float64)
                prices = state[:, 1:13].astype(np.float64) * 20.0
            idx = np.flatnonzero(self.rng.random(N) < self.expert_ratio)
            for p in idx:
                action[p] = self.expert_env._intrinsic_lp(prices[p], V_t[p], t)

        _, mask = self._mask_action(action, state)
        self._compute_noise_diagnostics(action, greedy_action, mask)
        return action


# ============================ CHANGE 9: DRIVER =====================================
# The driver is the original SEEDS loop with the module-level factory functions the
# original DDPG.train() already reads from module scope.  What is new: the phi and
# n_state switches are command-line arguments, a separate validation and a separate
# test environment are built with their own seeds (common random numbers across
# methods), and the reported figure is the PAIRED DDPG - RIV statistic on the test set.
#
#   python rolling_intrinsic_AR3_vectorized.py <phi> <14|33> [rounds] [train_paths]
#                                              [test_paths] [seed]
# ==================================================================================
import json, sys

torch.set_num_threads(2)

PHI = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
N_STATE = int(sys.argv[2]) if len(sys.argv) > 2 else 33
N_ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
TRAIN_PATHS = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
TEST_PATHS = int(sys.argv[5]) if len(sys.argv) > 5 else 20000
SEEDS = [int(sys.argv[6])] if len(sys.argv) > 6 else [78]
VAL_PATHS = 4000
TAG = f'phi{PHI}_s{N_STATE}_seed{SEEDS[0]}'

BASE_PARAMS = {
    'n_months': 12,
    'V_min': 0,
    'V_max': 1,
    'V_0': 0,
    'W_max': 0.4,
    'I_max': 0.4,
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
    'initial_v': 0.0249967313173077,
    'penalty_lambda1': 10,
    'penalty_lambda2': 50.,
    'penalty_lambda_riv': 0.0,
    'monthly_seasonal_factors': np.array([
        -0.106616824924423, -0.152361004102492, -0.167724706188117, -0.16797984045645,
        -0.159526180248348, -0.13927943487493, -0.0953402986114613, -0.0474646801238288,
        -0.0278622280543003, 0.0, -0.00850263509128089, -0.0409638719325969]),
    # CHANGE 1 + CHANGE 2 switches
    'phi': PHI,
    'n_state': N_STATE,
}


def make_params(seed, n_paths):
    p = dict(BASE_PARAMS)
    p['seed'] = seed
    p['n_paths'] = n_paths
    return p


ddpg_results = []
best_agent, best_eval_score = None, float('-inf')
for seed in SEEDS:
    environment_settings = {
        'env_name': 'TTFGasStorageEnv',
        # CHANGE 9: undiscounted.  Equation (26) is an undiscounted sum over 12 monthly
        # steps; gamma = 0.99 would shrink the terminal settlement leg by 0.99^11 = 0.895
        # and bias the critic against late withdrawals.
        'gamma': 1.0,
        'max_minutes': np.inf,
        'max_episodes': N_ROUNDS * TRAIN_PATHS,
        # CHANGE 9: no reward-threshold early stop; the best actor is selected on the
        # fixed validation environment instead.
        'goal_mean_500_reward': np.inf
    }

    # CHANGE 10: the actor/critic architecture that produced the reported 3.914.
    policy_model_fn = lambda nS, bounds: FCDPAutoregressive(nS, bounds, hidden_dims=(256, 256, 256))
    policy_max_grad_norm = 1
    policy_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
    policy_optimizer_lr = 0.00003

    value_model_fn = lambda nS, nA: FCQV(nS, nA, hidden_dims=(256, 256, 256))
    value_max_grad_norm = 1
    value_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
    value_optimizer_lr = 0.0005

    # CHANGE 9: the noise is annealed once per ROUND, so max_episode = N_ROUNDS.
    training_strategy_fn = lambda bounds: NormalNoiseStrategy(
        bounds, exploration_noise_ratio=0.60, final_noise_ratio=0.05,
        max_episode=N_ROUNDS, noise_free_last=0)
    evaluation_strategy_fn = lambda bounds: GreedyStrategy(bounds)

    # CHANGE 12: alpha = 0 makes every priority equal, i.e. UNIFORM replay with unit
    # importance weights -- prioritised replay over-samples the terminal settlement
    # transitions, which carry by far the largest TD errors in a 12-step episode.
    replay_buffer_fn = lambda: PrioritizedReplayBuffer(max_samples=400_000, batch_size=512,
                                                       alpha=0.0)
    # CHANGE 12: no uniform-random warm-up phase.  The gate
    # `len(buffer) > batch_size * n_warmup_batches` is still evaluated after the first
    # round, when the buffer already holds 12 * TRAIN_PATHS transitions, so learning
    # starts at round 1 and exploration is the actor + noise + expert from the outset.
    n_warmup_batches = 0
    update_target_every_steps = 1
    tau = 0.005

    # CHANGE 9: knobs read from module scope by the round-based train() loop.
    iters_per_round = 3000  # gradient steps per round
    expert_ratio0 = 0.5  # RIV-guided exploration at round 0 ...
    expert_ratio1 = 0.0  # ... decaying to none at the last round
    reward_scale = 10.0  # critic-side units only; see interaction_step

    env_name, gamma, max_minutes, \
        max_episodes, goal_mean_500_reward = environment_settings.values()

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

    env = TTFGasStorageEnv(make_params(101, TRAIN_PATHS))
    validation_env = TTFGasStorageEnv(make_params(555, VAL_PATHS))
    print(f'[{TAG}] phi={PHI} n_state={N_STATE} obs_dim={env.observation_space.shape[0]} '
          f'rounds={N_ROUNDS} train_paths={TRAIN_PATHS}', flush=True)

    t0 = time.time()
    result, final_eval_score, training_time, wallclock_time = agent.train(
        env, seed, gamma, max_minutes, max_episodes, goal_mean_500_reward)

    # ---- out-of-sample, paired against rolling intrinsic on the same paths ----
    test_env = TTFGasStorageEnv(make_params(202, TEST_PATHS))
    res = agent.evaluate(agent.online_policy_model, test_env, paired_riv=True)

    # static intrinsic value: locked in at t = 0 by the same linear program
    F0 = test_env.F_trajectory[0][0].astype(np.float64)
    X0 = test_env._intrinsic_lp(F0, 0.0, 0)
    res['intrinsic'] = float(-np.dot(F0, X0))
    res.update(phi=PHI, n_state=N_STATE, seed=seed, rounds=N_ROUNDS,
               train_paths=TRAIN_PATHS, test_paths=TEST_PATHS,
               train_seconds=wallclock_time, total_seconds=time.time() - t0)

    print(f'[{TAG}] DDPG   = {res["ddpg"]:.4f} ({res["ddpg_se"]:.4f})', flush=True)
    print(f'[{TAG}] RIV    = {res["riv"]:.4f} ({res["riv_se"]:.4f})', flush=True)
    print(f'[{TAG}] paired = {res["diff"]:+.4f} ({res["diff_se"]:.4f})  win={res["win"]:.3f}',
          flush=True)
    print(f'[{TAG}] intrinsic = {res["intrinsic"]:.4f}', flush=True)
    json.dump(res, open(f'vec_{TAG}.json', 'w'), indent=2)
    torch.save(agent.online_policy_model.state_dict(), f'vec_actor_{TAG}.pt')
    ddpg_results.append(res)

