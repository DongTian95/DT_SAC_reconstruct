#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project     : DT_GPM
@File        : gym_wrapper.py
@Author      : Dong Tian
@E-mail      : dong.tian@outlook.de
@Data        : 09.10.23 20:20
@Description : This file defines the interface for the environment wrapper of gym
"""

import gymnasium as gym
from config.parse_config import ParseConfig


class Gym_Wrappers:
    def __init__(self):
        """
        Initializes the object.

        Returns:
            None
        """
        ParseConfig()
        self.env_config = ParseConfig.get_env_config()

    def do_wrappers(self, env):
        """
        Wraps the given environment based on the specified configuration.

        Args:
            env: The unwrapped gym environment.

        Returns:
            The wrapped environment.
        """
        # Check if ImageChannelFirst wrapper is required
        if self.env_config["wrapper_ImageChannelFirst"]:
            pass  # No use in the current works

        # Check if RescaleAction wrapper is required
        if self.env_config["wrapper_RescaleAction"]:
            # Get the maximum and minimum action values from the configuration
            max_action = float(self.env_config.get("wrapper_RescaleAction_max", 1))
            min_action = float(self.env_config.get("wrapper_RescaleAction_min", -1))

            # Wrap the environment with RescaleAction wrapper
            env = gym.wrappers.RescaleAction(env=env, max_action=max_action, min_action=min_action)

        # Check if ContinuousActionClip wrapper is required
        if self.env_config["wrapper_ContinuousActionClip"]:
            pass  # In the original program, the interval is set to -1e9 to 1e9, so this part will also be dropped

        # Check if AutoReset wrapper is required
        if self.env_config["wrapper_AutoReset"]:
            # Wrap the environment with AutoResetWrapper
            env = gym.wrappers.AutoResetWrapper(env=env)

        return env
