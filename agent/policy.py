#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project     : DT_SAC_reconstruct
@File        : actor.py
@Author      : Dong Tian
@E-mail      : dong.tian@outlook.de
@Data        : 25.11.23 21:05
@Description : This file defines the policy class of DT_SAC
"""
import os
import sys
import time

import torch as th

from config.parse_config import ParseConfig


class SAC_Policy(th.nn.Module):
    def __init__(
        self,
        env,
    ):
        super().__init__()

        # Get the needed configurations from .yaml
        ParseConfig()

        # Get the network configuration
        hidden_size = ParseConfig.get_neural_network_config()["hidden_layer_size"]
        hidden_num = ParseConfig.get_neural_network_config()["hidden_layer_number"]
        device = ParseConfig.get_neural_network_config()["device"]
        random_seed = ParseConfig.get_neural_network_config()["random_seed"]
        activation_function = ParseConfig.get_neural_network_config()["activation_function"]

        # Get the optimizer configuration
        optimizer = ParseConfig.get_training_config()["optimizer"]
        learning_rate = ParseConfig.get_training_config()["learning_rate"]

        # Get the environment configuration
        self.env = env
        action_size = self.env.action_size
        observation_size = self.env.observation_size

        # Set the seed for the network to ensure the reproducibility
        # The randomness just comes from PyTorch for the current developing stage
        if random_seed is not None:
            th.manual_seed(random_seed)

        # check whether the device is available
        if device == "cuda":
            self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        else:
            self.device = th.device("cpu")

        # Initialize the network
        # Encoder Network
        self.encoder_layers = th.nn.ModuleList()
        self.encoder_layers.append(th.nn.Linear(observation_size, hidden_size, device=self.device))
        for _ in range(hidden_num):
            self.encoder_layers.append(th.nn.Linear(hidden_size, hidden_size, device=self.device))
            if hasattr(th.nn, activation_function):
                self.encoder_layers.append(getattr(th.nn, activation_function)())
            else:
                raise ValueError(f"Activation function {activation_function} is not valid!")
        # mu Network
        if ParseConfig.get_neural_network_config()["enhance_mu"]:
            self.mu_layers = th.nn.ModuleList()
            for _ in range(hidden_num):
                self.mu_layers.append(th.nn.Linear(hidden_size, hidden_size, device=self.device))
                if hasattr(th.nn, activation_function):
                    self.mu_layers.append(getattr(th.nn, activation_function)())
                else:
                    raise ValueError(f"Activation function {activation_function} is not valid!")
            self.mu_layers.append(th.nn.Linear(hidden_size, action_size, device=self.device))
        else:
            self.mu_layer = th.nn.Linear(hidden_size, action_size, device=self.device)

        # sigma Network
        # TODO:Should be optimized here, the situation that action_size is not 1 is not considered!
        self.log_sigma_layer = th.nn.Linear(hidden_size, action_size, device=self.device)

        # Define the optimizer
        if hasattr(th.optim, optimizer):
            self.optimizer = getattr(th.optim, optimizer)(self.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Optimizer {optimizer} is not valid!")

        # For logger
        self.mu = None
        self.sigma = None

    def forward(self, observation, deterministic: bool = False):
        batch_size = observation.shape[0]
        state = observation.view(batch_size, -1)

        # Encoder part
        for layer in self.encoder_layers:
            state = layer(state)

        # mu and sigma part
        if ParseConfig.get_neural_network_config()["enhance_mu"]:
            mu_state = state
            for layer in self.mu_layers:
                mu_state = layer(mu_state)
            mu = mu_state
            mu = th.sigmoid(mu)
            action_max = self.env.action_space_high * 3
            action_min = self.env.action_space_low * 3
            mu = (action_max - action_min) * mu + action_min
        else:
            mu = self.mu_layer(state)
        # Save mu for logger
        self.mu = mu
        log_sigma = self.log_sigma_layer(state)
        # Following the instruction from Stable-Baselines3
        log_sigma = th.clamp(log_sigma, -20, 2)
        if ParseConfig.get_neural_network_config()["gSDE"]:
            sigma = self._get_std(log_sigma)
        else:
            sigma = th.exp(log_sigma)
        # Save sigma for logger
        self.sigma = sigma

        # sample from the normal distribution
        normal_distribution = th.distributions.Normal(mu, sigma)
        if deterministic:
            action = mu
        else:
            action = normal_distribution.rsample()
        # get the log probability of the action
        log_prob = normal_distribution.log_prob(action) + 1e-7  # avoid log(0)

        # Based on the original SAC article, tanh is applied to the action
        action = th.tanh(action)

        return action, log_prob

    @staticmethod
    def _get_std(log_std: th.Tensor) -> th.Tensor:
        """
        The way to deal with the std is based on the article of gSDE
        It prevents the std from changing too quickly
        :return:
        """
        below_threshold = th.where(log_std <= 0, th.exp(log_std), 0.0)
        safe_log_std = log_std * (log_std > 0) + 1.0e-6
        above_threshold = (th.log1p(safe_log_std) + 1.0) * (log_std > 0)
        std = below_threshold + above_threshold
        return std


if __name__ == "__main__":  # test function, DO NOT use it in a normal way
    from env.gym.gym_init import gym_Init
    env = gym_Init()
    policy_net = SAC_Policy(env)
    observation, _ = env.env.reset(seed=0)
    observation = th.tensor(observation, dtype=th.float32, device="cuda").view(1, -1)
    action, log_prob = policy_net(observation)
    print(action)
    print(log_prob)
