#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project     : DT_GPM
@File        : gym_init.py
@Author      : Dong Tian
@E-mail      : dong.tian@outlook.de
@Data        : 09.10.23 19:13
@Description : This file is used to initialize the gym(gymnasium) environment
"""
import os
import sys

import gymnasium as gym

from config.parse_config import ParseConfig
from env.gym.gym_wrapper import Gym_Wrappers


def Singleton(cls):
    """
    Decorator to ensure a class follows the Singleton pattern.

    Args:
        cls (class): The class to be decorated.

    Returns:
        class: The decorated class.

    """
    _instances = {}

    class SingletonWrapper(cls):
        """
        Wrapper class that ensures only one instance of the decorated class is created.
        """
        def __new__(cls, *args, **kwargs):
            """
            Create a new instance of the decorated class if it doesn't exist yet.

            Returns:
                object: The instance of the decorated class.

            """
            if cls not in _instances:
                _instances[cls] = super(SingletonWrapper, cls).__new__(cls)
            return _instances[cls]

    return SingletonWrapper


@Singleton
class gym_Init:
    action_size = None
    observation_size = None
    action_space_low = None
    action_space_high = None

    def __init__(self, visualize: bool = False):
        """
        Initializes the environment for the agent.

        Args:
            visualize (bool, optional): Whether to visualize the environment. Defaults to False.

        Returns:
            None
        """
        # Parse the environment configuration
        ParseConfig()

        # Get the environment configuration
        _env_config: dict = ParseConfig.get_env_config()

        # Create the environment using the environment name from the configuration
        self.env = gym.make(_env_config["env_name"], render_mode=_env_config["render_mode"])
        if visualize:
            self.env = gym.make(_env_config["env_name"], render_mode=_env_config["render_mode"])
        else:
            self.env = gym.make(_env_config["env_name"])

        # Set the seed for the environment
        self.set_seed_unwrapped(_env_config.get("seed", 0))

        # Apply wrappers to the environment
        gym_wrappers = Gym_Wrappers()
        self.env = gym_wrappers.do_wrappers(env=self.env)

        # Initialize class attributes that may be useful
        if gym_Init.action_size is None:
            gym_Init.action_size = self.env.action_space.shape[0]
        if gym_Init.observation_size is None:
            gym_Init.observation_size = self.env.observation_space.shape[0]
        if gym_Init.action_space_low is None:
            gym_Init.action_space_low = self.env.action_space.low[0]
        if gym_Init.action_space_high is None:
            gym_Init.action_space_high = self.env.action_space.high[0]

        # Determine the action dimension
        self.action_dimension = self._get_action_dimension()

    def _get_action_dimension(self):
        """
        Returns the dimension of the action space.

        This function checks the type of the action space and returns the appropriate dimension.
        For discrete action spaces, it returns the number of available actions.
        For continuous action spaces, it returns the dimension of the action space.

        Args:
            None

        Returns:
            int: The dimension of the action space.

        Raises:
            TypeError: If the action space type is unknown.
        """
        # Check if the action space is discrete
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            num_actions = self.env.action_space.n
            return num_actions

        # Check if the action space is continuous
        elif isinstance(self.env.action_space, gym.spaces.Box):
            action_dim = self.env.action_space.shape[0]
            return action_dim

        # Raise an error for unknown action space types
        else:
            raise TypeError("Unknown action space type")

    def set_seed_unwrapped(self, seed=None):
        """
        Set the seed for the environment.

        Args:
            seed (int): The seed number.

        Returns:
            None
        """
        # Check if the environment has an 'unwrapped' attribute
        if hasattr(self.env.unwrapped, "seed"):
            # Set the seed for the unwrapped environment
            self.env.unwrapped.seed(seed)
        else:
            # Return if the environment does not have an 'unwrapped' attribute
            return

    def close(self):
        """
        This function is used to close the gym env that is used.
        It is recommended to close the env after the training is finished.
        """
        self.env.close()


if __name__ == "__main__":  # Testing function, DO not use it in normal situation
    ParseConfig()
    test = gym_Init()
    test.env.reset(seed=0)
    try:
        for i in range(100):
            test.env.step([-1.0])
    finally:
        test.close()
