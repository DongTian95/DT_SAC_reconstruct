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

from algorithm.SAC_Algorithm import SAC_Algorithm
from config.parse_config import ParseConfig
from save_and_load.save import SaveModel

FLAGS = flags.FLAGS
flags.DEFINE_bool("wandb_on", True, "define whether to use wandb")
flags.DEFINE_bool("visualize_on", False, "define whether to visualize the environment")


def main(_):
    current_dir = os.getcwd()
    yaml_path = os.path.join(current_dir, "./config/Config.yaml")
    ParseConfig(yaml_path)
    from env.gym.gym_init import gym_Init

    environment = gym_Init(visualize=FLAGS.visualize_on)

    test = SAC_Algorithm(env=environment, wandb_on=FLAGS.wandb_on)
    time_start = time.time()
    for i in range(ParseConfig.get_training_config()["max_steps"]):
        if i % 100 == 0:
            print("################################", "Epoch: ", i, "################################")
        test.train()
        if i % ParseConfig.get_training_config()["logging_frequency"] == 0:
            test.evaluate()

    save_model = SaveModel("save_and_load/models_params")
    models_dict = {
        "actor": test.actor_net.state_dict(),
        "critic_1": test.critic_net_1.state_dict(),
        "critic_2": test.critic_net_2.state_dict(),
        "critic_1_target": test.critic_net_1_target.state_dict(),
        "critic_2_target": test.critic_net_2_target.state_dict(),
    }
    save_model.save_model(models_dict)


if __name__ == "__main__":
    app.run(main)
