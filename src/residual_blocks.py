import torch
import torch.nn as nn
from .utils import conv3x3

class BaseResidualBlock(nn.Module):
    """
    Base class for residual blocks, providing shared operator application logic.
    """
    def apply_operator(self, contrast: torch.Tensor, operator: torch.Tensor, total_field: torch.Tensor) -> torch.Tensor:
        """
        Applies the operator to contrast and total field using Einstein summation.

        Args:
            contrast: Contrast tensor [batch, channels, height, width].
            operator: Operator tensor (e.g., GS or GD).
            total_field: Total field tensor.

        Returns:
            Resulting tensor after operator application.
        """
        field_channels = total_field.split(1, dim=1)
        chi_shape = contrast.shape
        chi_real = contrast[:, 0, :, :]  # Real part of contrast
        chi_imag = contrast[:, 1, :, :]  # Imaginary part
        chi_real_flat = chi_real.view(chi_shape[0], -1)
        chi_imag_flat = chi_imag.view(chi_shape[0], -1)
        chi_real_broadcast = chi_real_flat.unsqueeze(2).expand(-1, -1, field_channels[0].shape[3])
        chi_imag_broadcast = chi_imag_flat.unsqueeze(2).expand(-1, -1, field_channels[0].shape[3])

        multiplied = [
            chi_real_broadcast * field_channels[0].squeeze(1) - chi_imag_broadcast * field_channels[1].squeeze(1),
            chi_real_broadcast * field_channels[1].squeeze(1) + chi_imag_broadcast * field_channels[0].squeeze(1),
            chi_real_broadcast * field_channels[2].squeeze(1) - chi_imag_broadcast * field_channels[3].squeeze(1),
            chi_real_broadcast * field_channels[3].squeeze(1) + chi_imag_broadcast * field_channels[2].squeeze(1),
            chi_real_broadcast * field_channels[4].squeeze(1) - chi_imag_broadcast * field_channels[5].squeeze(1),
            chi_real_broadcast * field_channels[5].squeeze(1) + chi_imag_broadcast * field_channels[4].squeeze(1)
        ]

        xr1 = torch.einsum('abc,acd->abd', operator[:, 0, :, :], multiplied[0]) - \
              torch.einsum('abc,acd->abd', operator[:, 1, :, :], multiplied[1])
        xi1 = torch.einsum('abc,acd->abd', operator[:, 0, :, :], multiplied[1]) + \
              torch.einsum('abc,acd->abd', operator[:, 1, :, :], multiplied[0])
        xr2 = torch.einsum('abc,acd->abd', operator[:, 2, :, :], multiplied[2]) - \
              torch.einsum('abc,acd->abd', operator[:, 3, :, :], multiplied[3])
        xi2 = torch.einsum('abc,acd->abd', operator[:, 2, :, :], multiplied[3]) + \
              torch.einsum('abc,acd->abd', operator[:, 3, :, :], multiplied[2])
        xr3 = torch.einsum('abc,acd->abd', operator[:, 4, :, :], multiplied[4]) - \
              torch.einsum('abc,acd->abd', operator[:, 5, :, :], multiplied[5])
        xi3 = torch.einsum('abc,acd->abd', operator[:, 4, :, :], multiplied[5]) + \
              torch.einsum('abc,acd->abd', operator[:, 5, :, :], multiplied[4])

        return torch.stack((xr1, xi1, xr2, xi2, xr3, xi3), dim=1)

class FieldResidualBlock(BaseResidualBlock):
    """
    Residual block for updating the total field in NeuralBIM.

    Args:
        channel_sizes: List of channel counts for convolutional layers.
        stride: Convolution stride. Defaults to 1.
        downsample: Downsampling indices or None. Defaults to None.
    """
    def __init__(self, channel_sizes: list[int], stride: int = 1, downsample: torch.Tensor | None = None):
        super().__init__()
        self.conv0 = conv3x3(channel_sizes[0], channel_sizes[0], stride)
        self.conv1 = conv3x3(channel_sizes[1], channel_sizes[2], stride)
        self.bn1 = nn.BatchNorm2d(channel_sizes[2])
        self.tanh1 = nn.Tanh()
        self.conv2 = conv3x3(channel_sizes[2], channel_sizes[3])
        self.bn2 = nn.BatchNorm2d(channel_sizes[3])
        self.tanh2 = nn.Tanh()
        self.conv3 = conv3x3(channel_sizes[3], channel_sizes[4])
        self.bn3 = nn.BatchNorm2d(channel_sizes[4])
        self.tanh3 = nn.Tanh()
        self.conv4 = conv3x3(channel_sizes[4], channel_sizes[5])
        self.bn4 = nn.BatchNorm2d(channel_sizes[5])
        self.tanh4 = nn.Tanh()
        self.conv5 = conv3x3(channel_sizes[5], channel_sizes[6])
        self.bn5 = nn.BatchNorm2d(channel_sizes[6])
        self.tanh5 = nn.Tanh()
        self.downsample = downsample

    def forward(self, total_field: torch.Tensor, operator_GD: torch.Tensor, 
                contrast: torch.Tensor, incident_field: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Forward pass for field residual block.

        Args:
            total_field: Total field tensor.
            operator_GD: Operator tensor for (GD).
            contrast: Contrast tensor.
            incident_field: Incident field tensor.

        Returns:
            Tuple of updated total_field, operator_GD, contrast, incident_field, and residual.
        """
        operator_result = self.apply_operator(contrast, operator_GD, total_field)
        residual_input = incident_field + operator_result - total_field
        residual = total_field
        conv_output = self.conv0(residual_input)
        conv_input = torch.cat([conv_output, residual_input], dim=1)
        output = self.tanh1(self.bn1(self.conv1(conv_input)))
        output = self.tanh2(self.bn2(self.conv2(output)))
        output = self.tanh3(self.bn3(self.conv3(output)))
        output = self.tanh4(self.bn4(self.conv4(output)))
        output = self.tanh5(self.bn5(self.conv5(output)))
        output = output + residual
        return output, operator_GD, contrast, incident_field, residual_input

class ChiResidualBlock(BaseResidualBlock):
    """
    Residual block for updating contrast (chi) in NeuralBIM.

    Args:
        channel_sizes: List of channel counts for convolutional layers.
        stride: Convolution stride. Defaults to 1.
        downsample: Downsampling indices or None. Defaults to None.
    """
    def __init__(self, channel_sizes: list[int], stride: int = 1, downsample: torch.Tensor | None = None):
        super().__init__()
        self.conv0 = conv3x3(channel_sizes[0], channel_sizes[0], stride)
        self.conv1 = conv3x3(channel_sizes[1], channel_sizes[2], stride)
        self.bn1 = nn.BatchNorm2d(channel_sizes[2])
        self.tanh1 = nn.Tanh()
        self.conv2 = conv3x3(channel_sizes[2], channel_sizes[3])
        self.bn2 = nn.BatchNorm2d(channel_sizes[3])
        self.tanh2 = nn.Tanh()
        self.conv3 = conv3x3(channel_sizes[3], channel_sizes[4])
        self.bn3 = nn.BatchNorm2d(channel_sizes[4])
        self.tanh3 = nn.Tanh()
        self.conv4 = conv3x3(channel_sizes[4], channel_sizes[5])
        self.bn4 = nn.BatchNorm2d(channel_sizes[5])
        self.tanh4 = nn.Tanh()
        self.conv5 = conv3x3(channel_sizes[5], channel_sizes[6])
        self.bn5 = nn.BatchNorm2d(channel_sizes[6])
        self.tanh5 = nn.Tanh()
        self.downsample = downsample

    def forward(self, contrast: torch.Tensor, operator_GS: torch.Tensor, 
                total_field: torch.Tensor, scattered_field: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Forward pass for chi residual block.

        Args:
            contrast: Contrast tensor.
            operator_GS: Operator tensor for (GS).
            total_field: Total field tensor.
            scattered_field: Scattered field tensor.

        Returns:
            Tuple of updated contrast, operator_GS, total_field, scattered_field, and residual.
        """
        operator_result = self.apply_operator(contrast, operator_GS, total_field)
        residual_sca = scattered_field - operator_result
        conv_input = torch.cat([residual_sca, contrast], dim=1)
        conv_output = self.conv0(conv_input)
        conv_input = torch.cat([conv_output, conv_input], dim=1)
        output = self.tanh1(self.bn1(self.conv1(conv_input)))
        output = self.tanh2(self.bn2(self.conv2(output)))
        output = self.tanh3(self.bn3(self.conv3(output)))
        output = self.tanh4(self.bn4(self.conv4(output)))
        output = self.tanh5(self.bn5(self.conv5(output)))
        output = output + contrast
        return output, operator_GS, total_field, scattered_field, residual_sca