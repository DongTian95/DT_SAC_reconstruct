#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project     : DT_SAC_reconstruct
@File        : pipeline.py
@Author      : Dong Tian
@E-mail      : dong.tian@outlook.de
@Data        : 28.11.23 23:05
@Description : This file is used to run different configurations defined in config directory, namely pipeline
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
flags.DEFINE_bool("wandb_on", False, "define whether to use wandb")
flags.DEFINE_bool("visualize_on", False, "define whether to visualize the environment")


def main(_):
    current_dir = os.getcwd()
    yaml_path = current_dir + "/config/pipeline/Config_pipeline_"
    for i in range(100):
        yaml_file = yaml_path + str(i) + ".yaml"
        if not os.path.exists(yaml_file):
            print(f"The configuration {yaml_file} does not exist.")
            print("The pipeline is finished, no other configurations needs to be tested.")
            return
        ParseConfig.del_instance()
        ParseConfig(yaml_file=yaml_file)

        from env.gym.gym_init import gym_Init
        environment = gym_Init(visualize=FLAGS.visualize_on)

        test = SAC_Algorithm(env=environment, wandb_on=FLAGS.wandb_on)
        for j in range(ParseConfig.get_training_config()["max_steps"]):
            print("###################", "Epoch: ", j, "###################")
            test.train()
            if j % ParseConfig.get_training_config()["logging_frequency"] == 0:
                test.evaluate()

        save_model = SaveModel(saving_dir="save_and_load/models_params", number=str(i))
        models_dict = {
            "actor": test.actor_net.state_dict(),
            "critic_1": test.critic_net_1.state_dict(),
            "critic_2": test.critic_net_2.state_dict(),
            "critic_1_target": test.critic_net_1_target.state_dict(),
            "critic_2_target": test.critic_net_2_target.state_dict(),
        }
        save_model.save_model(models_dict)

        del test


if __name__ == "__main__":
    app.run(main)
