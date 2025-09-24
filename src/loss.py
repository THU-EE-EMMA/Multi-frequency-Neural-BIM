import torch
import torch.nn as nn

class RegLossFunc(nn.Module):
    """
    Regularization loss function based on gradient magnitudes.
    """
    def __init__(self):
        super().__init__()

    def forward(self, output: torch.Tensor) -> torch.Tensor:
        """
        Computes regularization loss as mean sqrt of squared gradients.

        Args:
            output: Input tensor [batch, channels, height, width].

        Returns:
            Scalar loss value.
        """
        height, width = output.shape[2:4]
        x_grad = output[:, :, 1:, :] - output[:, :, :-height + 1, :]
        y_grad = output[:, :, :, 1:] - output[:, :, :, :-width + 1]
        return torch.mean(torch.sqrt(x_grad.pow(2) + 1e-9)) + torch.mean(torch.sqrt(y_grad.pow(2) + 1e-9))
    

class MylossFunc(nn.Module):
    """
    Mean squared error loss function for evaluating model outputs.
    """
    def __init__(self):
        super().__init__()

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes mean squared error loss between predicted and target tensors.

        Args:
            predicted: Predicted tensor.
            target: Target tensor.

        Returns:
            Scalar MSE loss value.
        """
        return torch.mean((predicted - target).pow(2))