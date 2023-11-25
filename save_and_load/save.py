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


class SaveModel:
    def __init__(self, saving_dir="save_and_load/models_params", number: str = None):
        ParseConfig()

        # Define where to save
        self.saving_dir = os.getcwd()
        self.saving_dir = os.path.join(self.saving_dir, saving_dir)
        if not os.path.exists(self.saving_dir):
            raise FileNotFoundError(f"Directory {self.saving_dir} does not exist.")
        if number is not None:
            self.saving_dir = os.path.join(self.saving_dir, "pipeline/model" + number + ".pth")
        else:
            self.saving_dir = os.path.join(self.saving_dir, "model.pth")

        self.name_dict = {}
        self.name_dict.update(ParseConfig.config["SavingModel"]["gru_params"])
        self.name_dict.update(ParseConfig.config["SavingModel"]["lstm_params"])
        self.name_dict.update(ParseConfig.config["SavingModel"]["lstm_target_params"])
        self.name_dict.update(ParseConfig.config["SavingModel"]["others"])

    def save_model(self, agent: GPM_Agent):
        for key in self.name_dict:
            self.name_dict[key] = getattr(agent, key)
        torch.save(self.name_dict, self.saving_dir)
        print("The model is successfully saved in: ", self.saving_dir)

if __name__ == "__main__":  # test function, DO NOT use it in a normal way
    test = SaveModel(saving_dir="models_params")
