#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project     : DT_GPM
@File        : parse_config.py
@Author      : Dong Tian
@E-mail      : dong.tian@outlook.de
@Data        : 08.10.23 12:29
@Description : This file is used to parse the configuration file .yaml and should be imported anywhere needed
"""
import os
import sys

import yaml


class ParseConfig:
    """
    All configurations are stored here
    """
    config = None
    _instances = {}

    def __new__(cls, *args, **kwargs):
        """
        To ensure only one instance is created
        """
        if cls not in cls._instances:
            cls._instances[cls] = super(ParseConfig, cls).__new__(cls)
        return cls._instances[cls]

    def __init__(self, yaml_file: str = "/home/dong/Desktop/DT_SAC_reconstruct/config/Config.yaml"):
        if ParseConfig.config is None:
            with open(yaml_file, 'r') as file:
                ParseConfig.config = yaml.safe_load(file)
                ParseConfig.config = ParseConfig.config["configs"]
        self._config_verification()

    @classmethod
    def del_instance(cls):
        """
        This function is defined to make the pipeline work properly
        :return: None
        """
        if cls in cls._instances:
            del cls._instances[cls]
            ParseConfig.config = None

    @staticmethod
    def _config_verification():
        """
        This function is to verify the format of self.config
        :return: None
        """
        if not isinstance(ParseConfig.config, dict):
            raise TypeError("ParseConfig.config doesn't have the right format, please check!")

    @staticmethod
    def get_training_config():
        return ParseConfig.config["Training"]
    
    @staticmethod
    def get_wandb_config():
        return ParseConfig.config["Wandb"]
    
    @staticmethod
    def get_env_config():
        return ParseConfig.config["Env"]
    
    @staticmethod
    def get_neural_network_config():
        return ParseConfig.config["NeuralNetwork"]

    @staticmethod
    def get_replay_buffer_config():
        return ParseConfig.config["ReplayBuffer"]


if __name__ == "__main__":
    """
    This part is to test.py the class ParseConfig and doesn't really have usage
    """
    ps = ParseConfig(yaml_file='Config.yaml')
    print(ParseConfig.get_training_config())

