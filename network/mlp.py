#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project     : DT_GPM
@File        : mlp.py
@Author      : Dong Tian
@E-mail      : dong.tian@outlook.de
@Data        : 11.10.23 21:06
@Description : This file defines how is one FC MLP initialized and trained (based on PyTorch)
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_layer_size: int, hidden_layer_number: int,
                 hidden_layer_size: int, output_layer_size: int, device: torch.device = "cuda",
                 batch_norm: bool = False):
        """
        Initialize a user-defined FC MLP with ReLU in all hidden layers.

        :param input_layer_size: defines the input size
        :param hidden_layer_number: defines the hidden layer number
        :param hidden_layer_size: defines the hidden layer size
        :param output_layer_size: defines the output size
        """
        super().__init__()
        self.layers = nn.ModuleList()

        # Add input layer
        self.layers.append(nn.Linear(input_layer_size, hidden_layer_size, device=device))

        # Add hidden layers
        for _ in range(hidden_layer_number):
            self.layers.append(nn.Linear(hidden_layer_size, hidden_layer_size, device=device))
            if batch_norm:
                self.layers.append(nn.BatchNorm1d(hidden_layer_size, device=device))
            self.layers.append(nn.ReLU())

        # Add output layer
        self.layers.append(nn.Linear(hidden_layer_size, output_layer_size, device=device))

    def forward(self, mlp_input):
        """
        Forward function, consider it as a magical function defined in PyTorch
        :param mlp_input: Input to the whole MLP
        :return: Output of the whole MLP
        """
        # in case it is a 1D tensor, add a dimension
        if len(mlp_input.shape) == 1:
            mlp_input = mlp_input.unsqueeze(0)

        for layer in self.layers:
            mlp_input = layer(mlp_input)

        return mlp_input


if __name__ == "__main__":      # Test function, DO NOT use it in a normal way
    model = MLP(
        input_layer_size=10,
        hidden_layer_number=2,
        hidden_layer_size=128,
        output_layer_size=1
    )
    model.eval()
    random_data = torch.rand(1, 1, 10).to("cuda")
    result = model(random_data)
    print(result)
