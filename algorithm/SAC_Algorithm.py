#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project     : DT_SAC_reconstruct
@File        : SAC_Algorithm.py
@Author      : Dong Tian
@E-mail      : dong.tian@outlook.de
@Data        : 27.11.23 17:17
@Description : This file is the SAC algorithm implementation
"""
import os
import sys
import time

import torch as th
import copy
import numpy as np

from config.parse_config import ParseConfig
from agent.policy import SAC_Policy
from agent.critic import SAC_Critic
from replay_buffer.SAC_replay_buffer import ReplayBuffer

class SAC_Algorithm:
    def __init__(
        self,
        env,
    ):
        self.env = env

        # Get the needed configurations from .yaml
        ParseConfig()
        self.training_config = ParseConfig.get_training_config()
        self.reward_scale = self.training_config["reward_scale"]
        self.eta = self.training_config["eta"]

        # decide the device
        if ParseConfig.get_neural_network_config()["device"] == "cuda":
            self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        else:
            self.device = th.device("cpu")

        # Actor part
        self.actor_net = SAC_Policy(env=self.env)
        # Critic part
        self.critic_net_1 = SAC_Critic(env=self.env)
        self.critic_net_2 = SAC_Critic(env=self.env)
        # Target Critic part
        self.critic_net_1_target = copy.deepcopy(self.critic_net_1)
        self.critic_net_2_target = copy.deepcopy(self.critic_net_2)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Initial date saver
        env_seed = ParseConfig.get_env_config()["random_seed"]
        if env_seed is not None:
            state, _ = self.env.env.reset(seed=env_seed)
        else:
            state, _ = self.env.env.reset()
        self.state_tensor = th.from_numpy(state).float().to(self.device).view(1, -1)
        action = (self.env.action_space_high + self.env.action_space_low)/2
        self.action_tensor = th.from_numpy(action).float().to(self.device).view(1, -1)
        
    def train(self):
        next_state, reward, done, truncated, info = self.env.env.step(self.action_tensor.cpu().detach().numpy())
        next_state_tensor = th.from_numpy(next_state).float().to(self.device).view(1, -1)
        # Scale the reward if necessary
        reward = self._scale_reward(reward)

        # Update the replay buffer
        # save to RB
        with th.no_grad():
            self.replay_buffer.add(
                state=self.state_tensor,
                action=self.action,
                reward=reward,
                next_state=next_state_tensor,
                done=done,
            )

        # Update the network
        if len(self.replay_buffer) > self.training_config["batch_size"] * 20:
            self.update_network()

        # Update the state for next step
        # Reset the environment is needed
        if done or truncated:
            action = (self.env.action_space_high + self.env.action_space_low)/2
            self.action_tensor = th.from_numpy(action).float().to(self.device).view(1, -1)
            state, _ = self.env.env.reset()
            self.state_tensor = th.from_numpy(state).float().to(self.device).view(1, -1)
        else:
            self.action_tensor, _ = self.actor_net(next_state_tensor)
            self.state_tensor = next_state_tensor

    def update_network(self):
        # Sample from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        print(states.size())

        # Update the critic network
        # Calculate the q value
        q_values_1 = self.critic_net_1(observation=states, action=actions)
        q_values_2 = self.critic_net_2(observation=states, action=actions)
        q_values = th.min(q_values_1, q_values_2)
        # Calculate the target q value
        next_actions, log_probs = self.actor_net(observation=next_states)
        next_q_values_1_target = self.critic_net_1_target(observation=next_states, action=next_actions)
        next_q_values_2_target = self.critic_net_2_target(observation=next_states, action=next_actions)
        next_q_values_target = th.min(next_q_values_1_target, next_q_values_2_target)
        target_q_values = rewards + (1 - dones.float()) * self.training_config["gamma"] * next_q_values_target
        target_q_values = target_q_values + log_probs.mean() * self.training_config["alpha"]
        # Calculate the loss
        critic_loss = th.nn.functional.mse_loss(q_values, target_q_values)
        # Update the network
        self.critic_net_1.optimizer.zero_grad()
        self.critic_net_2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_net_1.optimizer.step()
        self.critic_net_2.optimizer.step()

        # Update the actor network
        generated_actions, log_probs = self.actor_net(observation=states)
        q_values_1_actor = self.critic_net_1(observation=states, action=generated_actions)
        q_values_2_actor = self.critic_net_2(observation=states, action=generated_actions)
        q_values_actor = th.min(q_values_1_actor, q_values_2_actor)
        actor_loss = (self.training_config["alpha"] * log_probs - q_values_actor).mean()
        # Update the network
        self.actor_net.optimizer.zero_grad()
        actor_loss.backward()
        self.actor_net.optimizer.step()

        # Update the target network
        self._soft_copy(self.critic_net_1_target, self.critic_net_1)
        self._soft_copy(self.critic_net_2_target, self.critic_net_2)

    def _soft_copy(self, target_model, source_model):
        """
        Perform a soft-copy of the target model using the source model.

        Args:
            target_model (nn.Module): The target model to be updated.
            source_model (nn.Module): The source model to copy from.

        Returns:
            None
        """
        # Iterate over the parameters of the target and source models
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            # Perform the soft-copy by updating the target parameter data
            target_param.data.copy_((1 - self.eta) * target_param.data + self.eta * source_param.data)
            
    def _scale_reward(self, reward):
        return reward * self.reward_scale


if __name__ == "__main__":  # test function, DO NOT use it in a normal way
    from env.gym.gym_init import gym_Init
    env = gym_Init()
    algo = SAC_Algorithm(env)
    for i in range(2000):
        print("Episode: ", i)
        algo.train()
