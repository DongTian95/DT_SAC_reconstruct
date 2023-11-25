#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project     : DT_GPM
@File        : start.py
@Author      : Dong Tian
@E-mail      : dong.tian@outlook.de
@Data        : 07.11.23 17:53
@Description : 
"""
import os
import sys

from absl import app
from absl import flags
import time

from algorithm.sac_algorithm import SAC_Algorithm
from config.parse_config import ParseConfig
from save_and_load.save import SaveModel

FLAGS = flags.FLAGS
flags.DEFINE_bool("wandb_on", False, "define whether to use wandb")
flags.DEFINE_bool("visualize_on", False, "define whether to visualize the environment")


def main(_):
    current_dir = os.getcwd()
    yaml_path = os.path.join(current_dir, "./config/Config.yaml")
    ParseConfig(yaml_path)
    from env.gym.gym_init import gym_Init

    environment = gym_Init(visualize=FLAGS.visualize_on)

    test = SAC_Algorithm(envir=environment, wandb_on=FLAGS.wandb_on)
    time_start = time.time()
    for i in range(ParseConfig.get_training_config()["max_steps"]):
        if i % 1000 == 0:
            print("################################", "Epoch: ", i, "################################")
            time_end = time.time()
            print("1000 steps taken: {}".format(time_end - time_start))
            time_start = time.time()
        test.algorithm_body()
        if i % ParseConfig.get_training_config()["logging_frequency"] == 0:
            test.evaluate()

    save_model = SaveModel("save_and_load/models_params")
    save_model.save_model(test.agent)


if __name__ == "__main__":
    app.run(main)
