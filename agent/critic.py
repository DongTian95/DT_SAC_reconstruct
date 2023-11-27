#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project     : DT_SAC_reconstruct
@File        : critic.py
@Author      : Dong Tian
@E-mail      : dong.tian@outlook.de
@Data        : 25.11.23 21:05
@Description : This file defines the critic network of the SAC algorithm.
"""
import os
import sys
import time

import torch as th

from config.parse_config import ParseConfig


class SAC_Critic(th.nn.Module):
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

        # Define the critic network
        self.layers = th.nn.ModuleList()
        self.layers.append(th.nn.Linear(observation_size + action_size, hidden_size, device=self.device))
        for _ in range(hidden_num):
            self.layers.append(th.nn.Linear(hidden_size, hidden_size, device=self.device))
            if hasattr(th.nn, activation_function):
                self.layers.append(getattr(th.nn, activation_function)())
            else:
                raise ValueError(f"Activation function {activation_function} is not valid!")
        self.layers.append(th.nn.Linear(hidden_size, 1, device=self.device))
        
        # Define the optimizer
        if hasattr(th.optim, optimizer):
            self.optimizer = getattr(th.optim, optimizer)(self.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Optimizer {optimizer} is not valid!")

    def forward(self, observation, action):
        state = th.cat([observation, action], dim=1)
        for layer in self.layers:
            state = layer(state)
        return state


if __name__ == "__main__":  # test function, DO NOT use it in a normal way
    from env.gym.gym_init import gym_Init
    env = gym_Init()
    critic = SAC_Critic(env)
    observation, _ = env.env.reset(seed=0)
    observation = th.tensor(observation, dtype=th.float32).to("cuda:0").view(1, -1)
    action = 0
    action = th.tensor(action, dtype=th.float32).to("cuda:0").view(1, -1)
    print(critic(observation, action))

