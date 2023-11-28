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

    def save_model(self, to_be_saved_models: dict):
        torch.save(to_be_saved_models, self.saving_dir)
        print("The model is successfully saved in: ", self.saving_dir)

if __name__ == "__main__":  # test function, DO NOT use it in a normal way
    test = SaveModel(saving_dir="models_params")
