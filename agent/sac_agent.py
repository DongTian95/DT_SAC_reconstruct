#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project     : DT_GPM
@File        : generative_planning_method.py
@Author      : Dong Tian
@E-mail      : dong.tian@outlook.de
@Data        : 11.10.23 22:04
@Description : This file is the implementation of the agent of GPM (Generative Panning Method) based on the
               description on original article.
"""
import os
import sys

import torch
import torch.nn as nn
import copy

from config.parse_config import ParseConfig
from env.gym.gym_init import gym_Init
from network.mlp import MLP


class SAC_Agent(nn.Module):  # need to use .parameters(), subclass nn.Module is needed
    def __init__(self, environment):
        """
        Initializes the program and defines the input_layer_size, which is the size of the observation
        space of the environment.

        :param environment: The environment object.
        """
        super().__init__()

        # Get all the needed configurations
        self.training_config = ParseConfig.get_training_config()
        self.neural_network_config = ParseConfig.get_neural_network_config()

        # Set z size to 3, try at 15.okt.2023
        self.env = environment
        self.input_layer_size = self.env.observation_size

        # check whether the device is available
        if self.neural_network_config["device"] == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        # change the self.neural_network_config["device"] to device
        # This line is to ensure when GPU is not available, the device is therefore set to CPU
        self.neural_network_config["device"] = self.device

        # Initialize neural networks
        self._initialize_nn()

        # Define alpha and epsilon as hyperparameters
        # Alpha is the temperature parameter and epsilon is the threshold parameter to determine
        # whether a new plan is needed
        self.alpha = torch.tensor(ParseConfig.get_training_config()["alpha"], requires_grad=True)
        self.epsilon_threshold = torch.tensor(
            ParseConfig.get_training_config()["epsilon_threshold"], requires_grad=True)

        # Initialize action and Q sequences
        self.action_sequence = None
        self.Q_sequence = None
        self.Q_sequence_target = None

        # Initialize variables for gaussian distribution
        self.sigma = None
        self.mu = None

        # Define the optimizers
        self.optimizer_method = ParseConfig.get_training_config()["optimizer"]
        self.learning_rate = float(ParseConfig.get_training_config()["learning_rate"])
        self._define_optimizer()

        # Debug Line, these are not really necessary, and should be removed to produce a clean code
        self.rand = None
        self.sigma_to_log = None

    def _initialize_nn(self):
        """
        Initialize all the neural networks used in the program.
        """
        # Define the GRU part
        self.mlp_s2z = MLP(
            input_layer_size=self.input_layer_size,
            output_layer_size=self._z_size,
            **self.neural_network_config
        )
        self.mlp_z2mu = MLP(
            input_layer_size=self._z_size,
            output_layer_size=1,
            **self.neural_network_config
        )
        self.mlp_z2sigma = MLP(
            input_layer_size=self._z_size,
            output_layer_size=1,
            **self.neural_network_config,
        )
        self.mlp_z2h = MLP(
            input_layer_size=self._z_size,
            output_layer_size=self.neural_network_config["hidden_layer_size"],
            **self.neural_network_config
        )
        self.gru = nn.GRU(
            input_size=self.env.action_size,
            hidden_size=self.neural_network_config["hidden_layer_size"],  # hidden layer should be of one layer
            device=self.device,
            batch_first=True,
        )
        self.mlp_gru_output2a = MLP(
            input_layer_size=self.neural_network_config["hidden_layer_size"],
            output_layer_size=self.env.action_size,
            **self.neural_network_config
        )

        # Define the LSTM part1
        self.mlp_s2lstm_1 = MLP(
            input_layer_size=self.input_layer_size,
            output_layer_size=self.neural_network_config["hidden_layer_size"],
            **self.neural_network_config
        )
        self.lstm_1 = nn.LSTM(
            input_size=self.env.action_size,
            hidden_size=self.neural_network_config["hidden_layer_size"],
            device=self.device,
            batch_first=True,
        )
        # the output nn of the first step should be dealt separately because it doesn't take the output (or should say
        # no output) of the previous step and therefore should be treated the same
        self.mlp_lstm_output_first_net_1 = MLP(
            input_layer_size=self.neural_network_config["hidden_layer_size"],
            output_layer_size=1,
            **self.neural_network_config
        )
        # the output nn of the other steps
        self.mlp_lstm_output_1 = MLP(
            input_layer_size=self.neural_network_config["hidden_layer_size"],
            output_layer_size=1,
            **self.neural_network_config
        )

        # Define the LSTM part2
        self.mlp_s2lstm_2 = MLP(
            input_layer_size=self.input_layer_size,
            output_layer_size=self.neural_network_config["hidden_layer_size"],
            **self.neural_network_config
        )
        self.lstm_2 = nn.LSTM(
            input_size=1,  # action and state encoder
            hidden_size=self.neural_network_config["hidden_layer_size"],
            device=self.device,
            batch_first=True,
        )
        self.mlp_lstm_output_first_net_2 = MLP(
            input_layer_size=self.neural_network_config["hidden_layer_size"],
            output_layer_size=1,
            **self.neural_network_config
        )
        self.mlp_lstm_output_2 = MLP(
            input_layer_size=self.neural_network_config["hidden_layer_size"],
            output_layer_size=1,
            **self.neural_network_config
        )

        # Define the target network 1
        self.mlp_s2lstm_target_1 = copy.deepcopy(self.mlp_s2lstm_1)
        self.lstm_target_1 = copy.deepcopy(self.lstm_1)
        self.mlp_lstm_output_first_net_target_1 = copy.deepcopy(self.mlp_lstm_output_first_net_1)
        self.mlp_lstm_output_target_1 = copy.deepcopy(self.mlp_lstm_output_1)

        # Define the target network 2
        self.mlp_s2lstm_target_2 = copy.deepcopy(self.mlp_s2lstm_2)
        self.lstm_target_2 = copy.deepcopy(self.lstm_2)
        self.mlp_lstm_output_first_net_target_2 = copy.deepcopy(self.mlp_lstm_output_first_net_2)
        self.mlp_lstm_output_target_2 = copy.deepcopy(self.mlp_lstm_output_2)

    def _define_optimizer(self):
        """
        Initializes all optimizers required in the back propagation step.
        Only SGD and Adam optimizers are supported and should be configured inside .YAML.
        :return: None
        """
        # Map of supported optimizer methods
        OPTIMIZER_MAP = {
            "SGD": torch.optim.SGD,
            "Adam": torch.optim.Adam,
        }

        # Check if optimizer method is supported
        if self.optimizer_method not in OPTIMIZER_MAP:
            raise ValueError(f"Optimizer '{self.optimizer_method}' not recognized")

        # Define parameters for plan generator
        gru_params = (
            list(self.mlp_s2z.parameters()) +
            list(self.mlp_z2mu.parameters()) +
            list(self.mlp_z2sigma.parameters()) +
            list(self.mlp_z2h.parameters()) +
            list(self.gru.parameters()) +
            list(self.mlp_gru_output2a.parameters())
        )

        # Initialize plan generator optimizer
        self.optimizer_gru = OPTIMIZER_MAP[self.optimizer_method](gru_params, lr=self.learning_rate)

        # Define parameters for Q-values part
        lstm_params = (
            list(self.mlp_s2lstm_1.parameters()) +
            list(self.lstm_1.parameters()) +
            list(self.mlp_lstm_output_1.parameters()) +
            list(self.mlp_lstm_output_first_net_1.parameters()) +
            list(self.mlp_s2lstm_2.parameters()) +
            list(self.lstm_2.parameters()) +
            list(self.mlp_lstm_output_2.parameters()) +
            list(self.mlp_lstm_output_first_net_2.parameters())
        )

        # Initialize Q-values optimizer
        self.optimizer_lstm = OPTIMIZER_MAP[self.optimizer_method](lstm_params, lr=self.learning_rate)

        # Convert alpha to a list and initialize alpha optimizer
        self.optimizer_alpha = OPTIMIZER_MAP[self.optimizer_method]([self.alpha], lr=self.learning_rate)

        # Initialize epsilon optimizer
        self.optimizer_epsilon = OPTIMIZER_MAP[self.optimizer_method]([self.epsilon_threshold], lr=self.learning_rate)

    @staticmethod
    def gaussian_entropy(sigma: torch.Tensor) -> torch.Tensor:
        """
        Calculate the entropy of a Gaussian distribution.

        Args:
            sigma (torch.Tensor): The standard deviation of the Gaussian distribution.

        Returns:
            torch.Tensor: The entropy of the Gaussian distribution.
        """
        # Calculate the entropy using the formula: 0.5 * log(2 * pi * sigma^2 * e)
        entropy = 0.5 * torch.log(2 * torch.pi * sigma ** 2 * torch.e)
        return entropy

    def forward(self, s):
        """
        This function defines the forward path of the whole process according to the original article.

        Args:
            s (torch.Tensor): Input vector (here, it should be the observation of the env) (state)

        Returns:
            None
        """
        # Verify the input vector
        self._input_verification(s)

        # Plan Generator Part
        self.action_sequence = self.run_gru(s)

        # Critic part
        # Run LSTM with the action sequence
        self.Q_sequence = self.run_lstm(s=s, action_sequence=self.action_sequence)

        # Run LSTM target with the action sequence
        self.Q_sequence_target = self.run_lstm_target(s=s, action_sequence=self.action_sequence)

    @staticmethod
    def _input_verification(s):
        """
        Verify the input vector for the forward path of the whole process.

        Args:
            s (torch.Tensor): Input vector of the whole nn.
                (Normally, this should be the observation of the env.)

        Raises:
            ValueError: If the input vector is not a torch.Tensor
        """
        # Check if the input vector is a torch.Tensor
        if not isinstance(s, torch.Tensor):
            raise ValueError("Input vector must be a torch.Tensor")

    def _get_random_action_according_gaussian(self, mu, sigma):
        """
        Get a random action according to the given mean and standard deviation.

        Args:
            mu (torch.Tensor): Mean of the Gaussian distribution.
            sigma (torch.Tensor): Logarithm of the standard deviation of the Gaussian distribution.

        Returns:
            torch.Tensor: Randomly chosen action.

        Raises:
            TypeError: If mu or sigma is not a torch.Tensor.

        Notes:
            Re-parameterization trick is used to make the back propagation possible.
        """
        # Check if mu and sigma are torch.Tensor
        if not isinstance(mu, torch.Tensor) or not isinstance(sigma, torch.Tensor):
            raise TypeError("mu and sigma should ne torch.Tensor, please check!")

        # Re-parameterization trick
        rand = torch.randn_like(mu)
        action = mu + sigma * rand
        self.rand = rand

        # Clip the action within the action space bounds
        # In the original SAC paper, tanh is used to clip the action
        action = torch.tanh(action)

        return action

    @staticmethod
    def bounded_sigmoid_tensor(x, low, high):
        """
        Returns a bounded sigmoid function of x between low and high.

        Args:
            x (torch.Tensor): The input value.
            low (float): The lower bound of the output range.
            high (float): The upper bound of the output range.

        Returns:
            torch.Tensor: The bounded sigmoid function of x.
        """
        # Compute the sigmoid of x
        sigmoid_x = torch.sigmoid(x)

        # Compute the bounded value by scaling and shifting the sigmoid of x
        bounded_value = sigmoid_x * (high - low) + low

        return bounded_value

    def run_gru(self, s, deterministic=False):
        """
        Run the GRU part to generate the actions plan.

        Args:
            s (torch.Tensor): The observation of the environment
            deterministic (bool, optional): Whether to use the deterministic version by action selection.

        Returns:
            None
        """
        # get the input vector size
        input_size = s.size(0)

        # Generative actor part
        # Encode the input vector
        z = self.mlp_s2z(mlp_input=s)

        # Generate mu and sigma
        self.mu = self.mlp_z2mu(mlp_input=z)
        # Clamp of mu to avoid the inconsistency of NN
        try:
            if ParseConfig.get_training_config()["mu_clip"]:
                # min = 3 * self.env.action_space_low
                min = self.env.action_space_low
                # max = 3 * self.env.action_space_high
                max = self.env.action_space_high
                self.mu = torch.sigmoid(self.mu) * (max - min) + min
        except:
            pass
        # Define the maximum value of sigma (clip) to avoid the training in the orientation of enlarging sigma
        sigma_max = 20.0 * (self.env.action_space_high - self.env.action_space_low)
        if not deterministic:
            # Here the method from gSDE is used
            log_sigma = self.mlp_z2sigma(mlp_input=z)
            # clamp the log_sigma to avoid numeric instability
            log_sigma = torch.clamp(log_sigma, min=-0.1, max=sigma_max)
            self.sigma = self._get_std(log_std=log_sigma)
        else:
            self.sigma = torch.zeros_like(self.mu)
            self.sigma_to_log = self.mlp_z2sigma(mlp_input=z)
            self.sigma_to_log = torch.clamp(self.sigma_to_log, min=-3.0, max=sigma_max)
            self.sigma_to_log = self._get_std(log_std=self.sigma_to_log)

        # Generate random action according to Gaussian distribution
        action_t = self._get_random_action_according_gaussian(mu=self.mu, sigma=self.sigma)

        # Initialize action sequence with the generated action
        action_sequence = action_t

        # Initialize the transparent output of RNN
        _output_current = action_t.view(input_size, 1, self.env.action_size)
        _hidden_state_current = self.mlp_z2h(mlp_input=z)

        # Generate the action sequence
        for i in range(self.training_config["plan_length"]):
            _output_current, _hidden_state_current = self.gru(
                _output_current.view(input_size, 1, self.env.action_size),
                _hidden_state_current.view(1, input_size, self.neural_network_config["hidden_layer_size"]))
            _output_current = self.mlp_gru_output2a(
                mlp_input=_output_current.view(input_size, self.neural_network_config["hidden_layer_size"]))
            # clip the action
            _output_current = torch.tanh(_output_current)
            action_sequence = torch.cat((action_sequence, _output_current), dim=1)

        return action_sequence

    def run_lstm(self, s, action_sequence) -> torch.Tensor:
        """
        This function is the LSTM part of GPM, which generates the Q_value of the action sequence.
        Two Q networks are used to leverage the overestimation problem.

        Args:
            s: environment observation
            action_sequence: action sequence generated by GRU

        Returns: Q_value sequence (torch.Tensor)
        """
        # get the input vector size
        input_size = s.size(0)
        # Reshape s to match the expected input shape
        s = s.view(input_size, self.env.observation_size)

        # Initialize LSTM cell states to zero
        _c_current_1 = torch.zeros(1, input_size, self.neural_network_config["hidden_layer_size"], device=self.device)
        _c_current_2 = torch.zeros_like(_c_current_1)

        # Get hidden states
        _hidden_state_current_1 = self.mlp_s2lstm_1(s).view(1, input_size,
                                                            self.neural_network_config["hidden_layer_size"])
        _hidden_state_current_2 = self.mlp_s2lstm_2(s).view(1, input_size,
                                                            self.neural_network_config["hidden_layer_size"])

        # Initialize Q_sequence
        Q_sequence = torch.tensor([], device=self.device)

        # Calculate Q value for the first action in the sequence
        lstm_out_1, (_hidden_state_current_1, _c_current_1) = (
            self.lstm_1(action_sequence[:, 0].view(input_size, 1, self.env.action_size),
                        (_hidden_state_current_1, _c_current_1)))
        lstm_out_2, (_hidden_state_current_2, _c_current_2) = (
            self.lstm_2(action_sequence[:, 0].view(input_size, 1, self.env.action_size),
                        (_hidden_state_current_2, _c_current_2)))
        # For MLP, the input should be (batch_size, features) sequence_length here is 1
        # But not as in RNN, (batch_size, sequence_length, features)
        lstm_out_1 = lstm_out_1.view(input_size, self.neural_network_config["hidden_layer_size"])
        lstm_out_2 = lstm_out_2.view(input_size, self.neural_network_config["hidden_layer_size"])

        # Get output of LSTM for the first action
        _output_current_1 = self.mlp_lstm_output_first_net_1(lstm_out_1)
        _output_current_2 = self.mlp_lstm_output_first_net_2(lstm_out_2)

        # Save the smaller value (restrain overestimation)
        _min_values, _ = torch.min(torch.stack((_output_current_1, _output_current_2)), dim=0)
        Q_sequence = torch.cat((Q_sequence, _min_values), dim=1)

        # Calculate Q values for the rest of the action sequence
        for i in range(action_sequence.size(1) - 1):  # -1 is because the first action is already calculated
            lstm_out_1, (_hidden_state_current_1, _c_current_1) = (
                self.lstm_1(action_sequence[:, i].view(input_size, 1, self.env.action_size),
                            (_hidden_state_current_1, _c_current_1)))
            lstm_out_2, (_hidden_state_current_2, _c_current_2) = (
                self.lstm_2(action_sequence[:, i].view(input_size, 1, self.env.action_size),
                            (_hidden_state_current_2, _c_current_2)))
            # For MLP, the input should be (batch_size, features) sequence_length here is 1
            # But not as in RNN, (batch_size, sequence_length, features)
            lstm_out_1 = lstm_out_1.view(input_size, self.neural_network_config["hidden_layer_size"])
            lstm_out_2 = lstm_out_2.view(input_size, self.neural_network_config["hidden_layer_size"])

            # Get the smaller values of critic output
            _min_values, _ = torch.min(torch.stack((self.mlp_lstm_output_1(lstm_out_1),
                                                    self.mlp_lstm_output_2(lstm_out_2))), dim=0)
            Q_sequence = torch.cat((Q_sequence, Q_sequence[:, i - 1] + _min_values), dim=1)

        return Q_sequence

    def run_lstm_target(self, s: torch.Tensor, action_sequence: torch.Tensor) -> list:
        """
        This function calculates the target values of the target neural networks.

        Args:
            s (torch.Tensor): The observation state.
            action_sequence (torch.Tensor): The action sequence generated by the GRU.

        Returns:
            list: The target Q-value sequence.
        """

        # get the input vector size
        input_size = s.size(0)

        # Reshape s to match the expected input shape
        s = s.view(input_size, self.env.observation_size)

        # Initialize LSTM cell states to zero
        _c_current_1 = torch.zeros(1, input_size,
                                   self.neural_network_config["hidden_layer_size"], device=self.device)
        _c_current_2 = torch.zeros_like(_c_current_1)

        # Get hidden states
        _hidden_state_current_1 = self.mlp_s2lstm_target_1(s).view(1,
                                                                   input_size,
                                                                   self.neural_network_config["hidden_layer_size"])
        _hidden_state_current_2 = self.mlp_s2lstm_target_2(s).view(1,
                                                                   input_size,
                                                                   self.neural_network_config["hidden_layer_size"])

        # Initialize Q_sequence
        Q_sequence_target = torch.tensor([], device=self.device)

        # Calculate Q value for the first action in the sequence
        lstm_out_1, (_hidden_state_current_1, _c_current_1) = (
            self.lstm_target_1(action_sequence[:, 0].view(input_size, 1, self.env.action_size),
                               (_hidden_state_current_1, _c_current_1)))
        lstm_out_2, (_hidden_state_current_2, _c_current_2) = (
            self.lstm_target_2(action_sequence[:, 0].view(input_size, 1, self.env.action_size),
                               (_hidden_state_current_2, _c_current_2)))
        # For MLP, the input should be (batch_size, features) sequence_length here is 1
        # But not as in RNN, (batch_size, sequence_length, features)
        lstm_out_1 = lstm_out_1.view(input_size, self.neural_network_config["hidden_layer_size"])
        lstm_out_2 = lstm_out_2.view(input_size, self.neural_network_config["hidden_layer_size"])

        # Get output of LSTM for the first action
        _output_current_1 = self.mlp_lstm_output_first_net_target_1(lstm_out_1)
        _output_current_2 = self.mlp_lstm_output_first_net_target_2(lstm_out_2)

        # Save the smaller value (restrain overestimation)
        _min_values, _ = torch.min(torch.stack((_output_current_1, _output_current_2)), dim=0)
        Q_sequence_target = torch.cat((Q_sequence_target, _min_values), dim=1)

        # Calculate Q values for the rest of the action sequence
        for i in range(self.training_config["plan_length"]):
            lstm_out_1, (_hidden_state_current_1, _c_current_1) = (
                self.lstm_target_1(action_sequence[:, i].view(input_size, 1, self.env.action_size),
                                   (_hidden_state_current_1, _c_current_1)))
            lstm_out_2, (_hidden_state_current_2, _c_current_2) = (
                self.lstm_target_2(action_sequence[:, i].view(input_size, 1, self.env.action_size),
                                   (_hidden_state_current_2, _c_current_2)))
            # For MLP, the input should be (batch_size, features) sequence_length here is 1
            # But not as in RNN, (batch_size, sequence_length, features)
            lstm_out_1 = lstm_out_1.view(input_size, self.neural_network_config["hidden_layer_size"])
            lstm_out_2 = lstm_out_2.view(input_size, self.neural_network_config["hidden_layer_size"])

            # Get the smaller values of critic output
            _min_values, _ = torch.min(torch.stack((self.mlp_lstm_output_target_1(lstm_out_1),
                                                    self.mlp_lstm_output_target_2(lstm_out_2))), dim=0)
            Q_sequence_target = torch.cat((Q_sequence_target, Q_sequence_target[:, i - 1] + _min_values), dim=1)

        return Q_sequence_target

    def set_network_mode(self, mode: str) -> None:
        """
        Set the network mode of the model

        Args:
            mode (str): The mode to set the network to. If "eval", the model is eval. If "train", the model is train.

        Returns:
            None
        """
        # List of NN names
        NN_list = [
            "mlp_s2z",
            "mlp_z2mu",
            "mlp_z2sigma",
            "mlp_z2h",
            "gru",
            "mlp_gru_output2a",
            "mlp_s2lstm_1",
            "lstm_1",
            "mlp_lstm_output_first_net_1",
            "mlp_lstm_output_1",
            "mlp_s2lstm_2",
            "lstm_2",
            "mlp_lstm_output_first_net_2",
            "mlp_lstm_output_2",
            "mlp_s2lstm_target_1",
            "lstm_target_1",
            "mlp_lstm_output_first_net_target_1",
            "mlp_lstm_output_target_1",
            "mlp_s2lstm_target_2",
            "lstm_target_2",
            "mlp_lstm_output_first_net_target_2",
            "mlp_lstm_output_target_2",
        ]

        # Set network mode based on the mode parameter
        if mode == "train":
            for name in NN_list:
                self.__setattr__(name, self.__getattr__(name).train())
        elif mode == "eval":
            for name in NN_list:
                self.__setattr__(name, self.__getattr__(name).eval())
        else:
            raise ValueError("mode should be either 'train' or 'eval'")

    @staticmethod
    def _get_std(log_std) -> torch.Tensor:

        below_threshold = torch.where(log_std <= 0, torch.exp(log_std), 0.0)
        if torch.isnan(below_threshold).any():
            print("NaN detected in below_threshold")

        safe_log_std = log_std * (log_std > 0) + 1.0e-6
        if torch.isnan(safe_log_std).any():
            print("NaN detected in safe_log_std")

        above_threshold = (torch.log1p(safe_log_std) + 1.0) * (log_std > 0)
        if torch.isnan(above_threshold).any():
            print("NaN detected in above_threshold")

        std = below_threshold + above_threshold
        if torch.isnan(std).any():
            print("NaN detected in std")

        return std

    def log_prob(self) -> torch.Tensor:
        """
        This function calculates the log probability of the action sequence.
        """
        log_prob = torch.distributions.Normal(0, 1).log_prob(self.rand) - torch.log(self.sigma)
        return log_prob


if __name__ == "__main__":  # test.py function, DO NOT use it in a normal way
    ParseConfig()
    env = gym_Init()
    test = GPM_Agent(environment=env)
    observation, _ = env.env.reset()
    observation = torch.tensor(observation, requires_grad=True, device="cuda").view(1, -1)
    test.set_network_mode("eval")
    test.forward(s=observation)
    del env
