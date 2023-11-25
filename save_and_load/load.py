#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project     : DT_GPM
@File        : save.py
@Author      : Dong Tian
@E-mail      : dong.tian@outlook.de
@Data        : 13.11.23 16:04
@Description : 
"""
import os
import sys

import torch

from config.parse_config import ParseConfig
from agent.gpm_agent import GPM_Agent


class LoadModel:
    def __init__(self, saving_dir="save_and_load/models_params"):
        ParseConfig()

        # Define where to save
        self.saving_dir = os.getcwd()
        self.saving_dir = os.path.join(self.saving_dir, saving_dir)
        if not os.path.exists(self.saving_dir):
            raise FileNotFoundError(f"Directory {self.saving_dir} does not exist.")
        self.saving_dir = os.path.join(self.saving_dir, "model.pth")

    def load_model(self, agent: GPM_Agent):
        checkpoint = torch.load(self.saving_dir)
        for key in checkpoint:
            setattr(agent, key, checkpoint[key])


if __name__ == "__main__":  # test function, DO NOT use it in a normal way
    pass
