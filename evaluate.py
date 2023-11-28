#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project     : DT_GPM
@File        : evaluate.py
@Author      : Dong Tian
@E-mail      : dong.tian@outlook.de
@Data        : 13.11.23 21:11
@Description : 
"""
import os
import sys

from algorithm.SAC_Algorithm import SAC_Algorithm
from config.parse_config import ParseConfig
from save_and_load.load import LoadModel

if __name__ == "__main__":  # test function, DO NOT use it in a normal way
    current_dir = os.getcwd()
    yaml_path = os.path.join(current_dir, "./config/Config.yaml")
    ParseConfig(yaml_path)
    from env.gym.gym_init import gym_Init

    environment = gym_Init(visualize=True)

    test = SAC_Algorithm(env=environment)
    load = LoadModel()
    load.load_model(agent=test)
    while True:
        test.evaluate()
