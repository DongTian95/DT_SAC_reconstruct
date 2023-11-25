#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project     : DT_GPM
@File        : gpm_algorithm.py
@Author      : Dong Tian
@E-mail      : dong.tian@outlook.de
@Data        : 02.11.23 19:58
@Description : This file implements the whole algorithm of GPM(Generative Planning Method)
"""

import torch

from config.parse_config import ParseConfig
from agent.sac_agent import SAC_Agent
from replay_buffer.replay_buffer import ReplayBuffer
from py_wandb.wandb_init import Wandb_Init


class SAC_Algorithm:
    def __init__(self, envir, wandb_on: bool = False):
        """
        Initialize the class with the given environment.

        Args:
            envir (object): The environment to be run, such as gymnasium.
        """
        # Parse the configuration file
        ParseConfig()

        # Get the needed configurations from .yaml
        training_config = ParseConfig.get_training_config()
        self.batch_size = training_config["batch_size"]
        self.plan_length = training_config["plan_length"]
        self.eta = training_config["eta"]
        self.gamma = training_config["gamma"]
        self.reward_scale = training_config["reward_scale"]

        # Initialize the agent
        self.agent = SAC_Agent(environment=envir)

        # Initialize the replay buffer
        self.replay_buffer = ReplayBuffer()

        # Reset the environment
        observation, _ = self.agent.env.env.reset()
        self.observation_tensor = torch.from_numpy(observation).float().to(self.agent.device).view(1, -1)
        self.action_sequence_tensor = torch.tensor([], device=self.agent.device).view(1, -1)

        # Wandb initialization
        if wandb_on:
            self.wandb_log = Wandb_Init()
        else:
            self.wandb_log = None

    def algorithm_body(self):
        """
        The algorithm body of the whole process, after the initialization of all the configurations that are
        necessary, the real training should start from here

        Returns:
            None
        """
        # Get samples from the environment and save them to the replay buffer
        self.get_sample_from_env()

        # Train the network
        # Replay Buffer should at least contain 20 * batch_size samples
        if len(self.replay_buffer) > 20 * ParseConfig.get_training_config()["batch_size"]:
            self.train()

    def get_sample_from_env(self):
        """
        This function is to get samples from environment and store the samples in Replay Buffer
        :return: None
        """
        # Set the network mode to eval, otherwise it may cause the problem because Batch Normalization is implemented
        self.agent.set_network_mode("eval")

        generated_action_sequence = self.agent.run_gru(self.observation_tensor)
        if self.action_sequence_tensor.size(0) <= 1:
            self.action_sequence_tensor = generated_action_sequence
        else:
            self.action_sequence_tensor = self.action_sequence_tensor[1:]
            old_critic_value = self.agent.run_lstm(self.observation_tensor, self.action_sequence_tensor).mean()
            new_critic_value = self.agent.run_lstm(self.observation_tensor, generated_action_sequence).mean()
            if new_critic_value > old_critic_value + self.agent.epsilon_threshold:
                self.action_sequence_tensor = generated_action_sequence
        action = self.action_sequence_tensor[0]

        next_observation, reward, done, truncated, info = self.agent.env.env.step(action.cpu().detach().numpy())
        reward = self._scale_reward(reward)

        # Add the sample to the replay buffer
        with torch.no_grad():
            self.replay_buffer.add(
                state=self.observation_tensor.detach(),
                action_sequence=generated_action_sequence.detach(),
                current_action_sequence_length=self.action_sequence_tensor.size(0),
                reward=reward,
                next_state=torch.from_numpy(next_observation).float().to(self.agent.device).view(1, -1),
                done=done,
            )

        if truncated or done:
            # Reset the environment
            observation, info = self.agent.env.env.reset()
            self.observation_tensor = torch.from_numpy(observation).float().to(self.agent.device).view(1, -1)
            self.action_sequence_tensor = torch.tensor([], device=self.agent.device).view(1, -1)
        else:
            observation = next_observation
            self.observation_tensor = torch.from_numpy(observation).float().to(self.agent.device).view(1, -1)

    def train(self):
        """
        this function is used to train the neural networks defined in gpm_agent.py
        It calculates the loss function, performs gradient descent, and updates the neural networks

        Returns:
            None
        """
        # set the network mode to train
        self.agent.set_network_mode("train")

        # get sample from replay buffer
        (states_tensor, actions_sequence_tensor, current_action_sequence_length,
         rewards_tensor, next_states_tensor, dones_tensor) = self.replay_buffer.sample()

        # update the critic part
        q_values = self.agent.run_lstm(s=states_tensor, action_sequence=actions_sequence_tensor)
        next_actions_sequence_target_tensor = self.agent.run_gru(s=next_states_tensor)
        q_values_target = self.agent.run_lstm_target(s=next_states_tensor,
                                                     action_sequence=next_actions_sequence_target_tensor)
        # To deal with the problem when the next state is terminal
        q_values_target = q_values_target * (1 - dones_tensor.float())
        entropy = -self.agent.alpha * self.agent.log_prob()
        target = rewards_tensor + self.gamma * q_values_target - entropy.mean()
        critic_loss = torch.nn.functional.smooth_l1_loss(q_values, target)
        self.agent.optimizer_lstm.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.agent.optimizer_lstm.step()

        # update the plan generator part
        generated_actions_sequence = self.agent.run_gru(s=states_tensor)
        entropy_loss = -self.agent.alpha * self.agent.log_prob()
        plan_loss = -self.agent.run_lstm(s=states_tensor, action_sequence=generated_actions_sequence).mean()
        plan_loss = plan_loss - entropy_loss.mean()
        self.agent.optimizer_gru.zero_grad(set_to_none=True)
        plan_loss.backward()
        self.agent.optimizer_gru.step()

        # update the alpha part
        self.agent.run_gru(s=states_tensor)
        entropy_loss = -self.agent.log_prob().mean()
        alpha_loss = -self.agent.alpha * (self.agent.env.action_size - entropy_loss)
        self.agent.optimizer_alpha.zero_grad(set_to_none=True)
        alpha_loss.backward()
        # self.agent.optimizer_alpha.step()
        if self.agent.alpha <= 0.0 or self.agent.alpha >= ParseConfig.config["Training"]["alpha"] * 5.0:
            with torch.no_grad():   # Make sure this step isn't tracked by autograd
                self.agent.alpha.data.clamp_(min=0.0, max=ParseConfig.config["Training"]["alpha"] * 5.0)

        # update the epsilon part
        self.agent.optimizer_epsilon.zero_grad(set_to_none=True)
        epsilon_loss = self.agent.epsilon_threshold * (current_action_sequence_length - self.l_commit_target)
        epsilon_loss = epsilon_loss.mean()
        epsilon_loss.backward()
        self.agent.optimizer_epsilon.step()
        if self.agent.epsilon_threshold <= 0.0:
            with torch.no_grad():   # Make sure this step isn't tracked by autograd
                self.agent.epsilon_threshold.data.clamp_(min=0.0)

        # soft-copy of target network
        # update the target network 1
        self._soft_copy(self.agent.mlp_s2lstm_target_1, self.agent.mlp_s2lstm_1)
        self._soft_copy(self.agent.lstm_target_1, self.agent.lstm_1)
        self._soft_copy(self.agent.mlp_lstm_output_target_1, self.agent.mlp_lstm_output_1)
        self._soft_copy(self.agent.mlp_lstm_output_first_net_target_1, self.agent.mlp_lstm_output_first_net_1)
        # update the target network 2
        self._soft_copy(self.agent.mlp_s2lstm_target_2, self.agent.mlp_s2lstm_2)
        self._soft_copy(self.agent.lstm_target_2, self.agent.lstm_2)
        self._soft_copy(self.agent.mlp_lstm_output_target_2, self.agent.mlp_lstm_output_2)
        self._soft_copy(self.agent.mlp_lstm_output_first_net_target_2, self.agent.mlp_lstm_output_first_net_2)

    def evaluate(self):
        # Set the network mode to eval, otherwise it may cause the problem because Batch Normalization is implemented
        self.agent.set_network_mode("eval")

        total_reward = 0.0
        sigma_average = 0.0
        alpha_average = 0.0
        entropy = 0.0
        mu = 0.0

        observation, info = self.agent.env.env.reset()
        observation_tensor = torch.from_numpy(observation).float().to(self.agent.device).view(1, -1)
        current_action_sequence_tensor = torch.tensor([]).float().to(self.agent.device)

        for i in range(100000):
            generated_action_sequence = self.agent.run_gru(observation_tensor, deterministic=True)
            if current_action_sequence_tensor.size(0) <= 1:
                current_action_sequence_tensor = generated_action_sequence
            else:
                current_action_sequence_tensor = current_action_sequence_tensor[1:]
                old_critic_value = self.agent.run_lstm(observation_tensor, current_action_sequence_tensor).mean()
                new_critic_value = self.agent.run_lstm(observation_tensor, generated_action_sequence).mean()
                if new_critic_value > old_critic_value + self.agent.epsilon_threshold:
                    current_action_sequence_tensor = generated_action_sequence
            action = current_action_sequence_tensor[0]

            next_observation, reward, done, truncated, info = self.agent.env.env.step(action.cpu().detach().numpy())
            reward = self._scale_reward(reward)
            total_reward += reward
            entropy += torch.log(self.agent.sigma_to_log)
            sigma_average += self.agent.sigma_to_log
            alpha_average += self.agent.alpha
            mu += self.agent.mu

            if truncated or done:
                # Reset the environment
                observation, info = self.agent.env.env.reset()
                self.observation_tensor = torch.from_numpy(observation).float().to(self.agent.device).view(1, -1)
                self.action_sequence_tensor = torch.tensor([], device=self.agent.device)
                if self.wandb_log is not None:
                    self.wandb_log.log_to_wandb({
                        "total_reward": total_reward,
                        "sigma": sigma_average / (i + 1.0),
                        "alpha": alpha_average / (i + 1.0),
                        # "epsilon": self.agent.epsilon_threshold,
                        "entropy": entropy/(i + 1.0),
                        "mu": mu / (i + 1.0),
                    })
                return
            else:
                observation = next_observation
                observation_tensor = torch.from_numpy(observation).float().to(self.agent.device).view(1, -1)

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


if __name__ == "__main__":  # test.py function, DO NOT use it in a normal way
    from env.gym.gym_init import gym_Init

    environment = gym_Init()

    test = SAC_Algorithm(envir=environment)
    for i in range(1003):
        print("This is the iteration: ", i)
        test.algorithm_body()
