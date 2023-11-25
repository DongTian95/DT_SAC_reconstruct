#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project     : DT_GPM
@File        : gym_init.py
@Author      : Dong Tian
@E-mail      : dong.tian@outlook.de
@Data        : 08.10.23 11:37
@Description : This file defines the initialization of wandb.ai
"""
import os
import sys

import wandb

from config.parse_config import ParseConfig


class Wandb_Init:
    def __init__(self, yaml_file: str = "/home/dong/Desktop/DT_GPM/config/Config.yaml"):
        """
        send project name and config info to wandb
        :param yaml_file: define where to find the yaml file
        """
        _wandb_config = ParseConfig.get_wandb_config()
        wandb.init(
            project=_wandb_config['project'],
            config=_wandb_config['config'],
            settings=wandb.Settings(),
        )
        self.average_return = None

    @staticmethod
    def log_to_wandb(
        info: dict = None,
    ) -> None:
        """
        Logs the provided information to WandB.
        WandB is a tool for visualizing and tracking machine learning experiments.

        Args:
            info (dict): A dictionary containing the information to be logged.
        """
        wandb.log(info)

    def __del__(self) -> None:
        """
        close wandb connection
        please call
        del <instance name>
        at the end to free up the connection
        :return: None
        """
        wandb.finish()


if __name__ == "__main__":  # Testing function
    test = Wandb_Init(yaml_file="/home/dong/Desktop/DT_GPM/config/Config.yaml")
    import random
    epochs = 10
    offset = random.random()/5
    for epoch in range(1, epochs):
        test.acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        test.loss = 2 ** -epoch + random.random() / epoch + offset
        test.log_to_wandb()
    del test
