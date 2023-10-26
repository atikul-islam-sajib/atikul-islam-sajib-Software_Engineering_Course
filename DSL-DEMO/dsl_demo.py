"""Import all important package"""
import torch
import torch.nn as nn


class DSL:
    """DSL - Domain Specific Language class for NN"""

    def __init__(self):
        """
        Initialize the DSL class

        Attributes:
        - model : Sequential model
        - prams : in contains epochs that is default = 10 and lr(Learning Rate) and that is default 0.001
        """
        self.model = nn.Sequential()
        self.params = {"epochs": 5, "lr": 0.001}

    def _insert_activation(self, tokens):
        """
        Update the activation functions for Neural Network

        Args:
        - tokens: A list of tokens parsed from the command line

        Returns:
        - model
        """
        if tokens[1] == "relu":
            self.model.add_module(f"ReLU {len(self.model)}", nn.ReLU())
        elif tokens[1] == "leaky relu":
            self.model.add_module(f"Leaky ReLU {len(self.model)}", nn.LeakyReLU())
        elif tokens[1] == "sigmoid":
            self.model.add_module(f"Sigmoid {len(self.model)}", nn.Sigmoid())
        elif tokens[1] == "softmax":
            self.model.add_module(f"Softmax {len(self.model)}", nn.Softmax(dim=1))

        return self.model

    def _add_layers(self, command_line, tokens):
        """
        Update the model and hyperparameters

        Args:
        - command line: DSL command to be processed
        - tokens: A list of tokens parsed from the command line

        Returns:
        -models and parameters
        """
        if command_line == "layer":
            input_features, output_features = map(int, tokens[1:])
            linear_layer = nn.Linear(
                in_features=input_features, out_features=output_features
            )
            self.model.add_module(f"linear_{len(self.model)}", linear_layer)

        elif command_line == "activation":
            self.model = self._insert_activation(tokens=tokens)

        elif command_line == "dropout":
            self.model.add_module(
                f"Droput {len(self.model)}", nn.Dropout(p=float(tokens[1]))
            )

        elif command_line == "train":
            self.params["epochs"] = int(tokens[1])
            self.params["lr"] = float(tokens[2])

        return self.model, self.params

    def model_defined(self, dsl_input):
        """
        Update the model and hyperparameters

        Args:
        - dsl input: DSL command to be processed

        Returns:
        - models and parameters(that is dict format)
        """
        lines = dsl_input.strip().split("\n")

        for each_line in lines:
            tokens = each_line.split()
            command_line = tokens[0]

            model, params = self._add_layers(command_line=command_line, tokens=tokens)

        return model, params


if __name__ == "__main__":
    dsl_input = """
    layer 4 32
    activation relu 
    dropout 0.3
    layer 32 16
    activation relu
    dropout 0.3
    layer 16 3
    activation softmax
    train 100 0.001
    """

    # Create the object of DSL
    dsl_demo = DSL()
    model, params = dsl_demo.model_defined(dsl_input=dsl_input)

    print("Model architecture is given below.\n", model)
    print("Model parameters # ".format(params))
