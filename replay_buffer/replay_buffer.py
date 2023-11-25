#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project     : DT_GPM
@File        : GPM_replay_buffer.py
@Author      : Dong Tian
@E-mail      : dong.tian@outlook.de
@Data        : 22.10.23 17:35
@Description : This file is to implement the replay buffer of GPM (Generative Planning Method)
"""
import os
import sys
import time

import random
from collections import namedtuple, deque
import torch

from config.parse_config import ParseConfig

# Define a named tuple to represent a single experience
Experience = namedtuple('Experience', ('state', 'action_sequence',
                                       'current_action_sequence_length', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, buffer_size: int = None, batch_size: int = None, device: str = None):
        """
        Initialize a ReplayBuffer and will use the configurations in YAML to initialize by default
        :param buffer_size: the maximum size of buffer
        :param batch_size: batch size for batch training
        :param device: cuda or cpu, defines device to move tensors to
        """
        ParseConfig()   # ParseConfig is a singleton, if it is already initialized, this step will have no effect
        # by default, configs in YAML will be used
        self.config = ParseConfig.get_replay_buffer_config()
        buffer_size = buffer_size or self.config["buffer_size"]
        batch_size = batch_size or self.config["batch_size"]
        device = device or self.config["device"]

        self._memory = deque(maxlen=buffer_size)
        self._batch_size = batch_size
        if device == "cuda" or device == "cpu":
            self._device = device
        else:
            raise TypeError("To be used device must be cuda or cpu")

    def add(self, state, action_sequence, current_action_sequence_length, reward, next_state, done):
        """
        Add a new experience to memory
        :return: None
        """
        experience = Experience(
            state=state,
            action_sequence=action_sequence,
            current_action_sequence_length=current_action_sequence_length,
            reward=reward,
            next_state=next_state,
            done=done,
        )
        self._memory.append(experience)

    def sample(self):
        """
        Randomly sample a batch of experiences from memory
        :return: sampled memories
        """
        # randomly select samples
        experiences = random.sample(self._memory, k=self._batch_size)

        # transform them to the expected tensor devices
        states = torch.stack([e.state for e in experiences]).to(self._device).view(self._batch_size, -1)
        actions_sequence = torch.stack(
            [e.action_sequence for e in experiences]).to(self._device).view(self._batch_size, -1)
        current_action_sequence_length = torch.tensor([
            e.current_action_sequence_length
            for e in experiences], dtype=torch.float).to(self._device).view(self._batch_size, -1)
        rewards = torch.tensor(
            [e.reward for e in experiences], dtype=torch.float).to(self._device).view(self._batch_size, -1)
        next_states = torch.stack([e.next_state for e in experiences]).to(self._device).view(self._batch_size, -1)
        dones = torch.tensor([e.done for e in experiences]).to(self._device).view(self._batch_size, -1)

        return states, actions_sequence, current_action_sequence_length, rewards, next_states, dones

    def __len__(self):
        """
        :return: the current size of the memory
        """
        return len(self._memory)


if __name__ == "__main__":  # test.py function, DO NOT use it in a normal way
    ParseConfig()
    test = ReplayBuffer()
    print(len(test))
