"""Import the necessity libraries"""
import torch
import torch.nn as nn


class ANN(nn.Module):
    """
    Custom artificial neural network (ANN) model for the Iris dataset classification.

    This model consists of two separate fully connected layers (LEFT and RIGHT)
    followed by an output layer that combines their outputs.

    Args:
        None

    Attributes:
        LEFT_LAYER (nn.Sequential): Left fully connected layers.
        RIGHT_LAYER (nn.Sequential): Right fully connected layers.
        OUT_LAYER (nn.Sequential): Output layer for classification.

    Methods:
        left_fully_connected_layer: Create the left fully connected layer.
        right_fully_connected_layer: Create the right fully connected layer.
        output_layer: Create the output layer.
        forward: Forward pass through the model.
        total_trainable_parameters: Calculate and display the total trainable parameters.

    """

    def __init__(self):
        super().__init__()

        # Initialize the left and right fully connected layers
        self.left_layer = self.left_fully_connected_layer(dropout=0.2)
        self.right_layer = self.right_fully_connected_layer(dropout=0.2)

        # Initialize the output layer
        self.out_layer = self.output_layer()

        # Initialise the model
        self.model = None

    def left_fully_connected_layer(self, dropout=None):
        """
        Create the left fully connected layers.

        Args:
            dropout (float): Dropout probability for regularization (default: None).

        Returns:
            nn.Sequential: Left fully connected layers.
        """
        return nn.Sequential(
            nn.Linear(in_features=4, out_features=16, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=16, out_features=8, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def right_fully_connected_layer(self, dropout=None):
        """
        Create the right fully connected layers.

        Args:
            dropout (float): Dropout probability for regularization (default: None).

        Returns:
            nn.Sequential: Right fully connected layers.
        """
        return nn.Sequential(
            nn.Linear(in_features=4, out_features=32, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=32, out_features=16, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def output_layer(self):
        """
        Create the output layer.

        Returns:
            nn.Sequential: Output layer for classification.
        """
        return nn.Sequential(
            nn.Linear(in_features=8 + 16, out_features=16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=16, out_features=3),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Model output.
        """
        # Pass input through the left and right fully connected layers
        left = self.left_layer(x)
        right = self.right_layer(x)

        # Concatenate the outputs of the left and right layers
        concat_layers = torch.cat((left, right), dim=1)

        # Pass the concatenated output through the output layer
        output_layer = self.out_layer(concat_layers)

        return output_layer

    def total_trainable_parameters(self):
        """
        Calculate and display the total number of trainable parameters in the model.

        Args:
            model (nn.Module): The PyTorch model for which to calculate the parameters.

        Returns:
            None
        """
        self.model = ANN()
        if self.model is None:
            raise Exception("Model is not found !")
        else:
            print("\nModel architecture\n".upper())
            print(self.model.parameters)
            print("\n", "_" * 50, "\n")

            total_params = []
            for layer_name, params in self.model.named_parameters():
                if params.requires_grad:
                    print(
                        "Layer # {} & trainable parameters # {} ".format(
                            layer_name, params.numel()
                        )
                    )
                    total_params.append(params.numel())
            print("\n", "_" * 50, "\n")
            total_trainable_parameters = sum(map(lambda x: x, total_params))
            print(
                "Total trainable parameters # {} ".format(
                    total_trainable_parameters
                ).upper(),
                "\n\n",
            )


if __name__ == "__main__":
    model = ANN()
    model.total_trainable_parameters()
