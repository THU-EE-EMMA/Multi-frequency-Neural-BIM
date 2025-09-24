import torch
import torch.nn as nn
from .residual_blocks import ChiResidualBlock, FieldResidualBlock

class MySequential(nn.Sequential):
    """
    Custom Sequential module that handles multiple inputs/outputs for residual blocks.
    """
    def forward(self, *inputs: torch.Tensor):
        for module in self._modules.values():
            inputs = module(*inputs)
        return inputs

class ResNet(nn.Module):
    """
    ResNet-like model with alternating chi and field residual layers.

    Args:
        chi_block: Class for chi residual blocks.
        field_block: Class for field residual blocks.
        num_blocks: Number of blocks per layer.
    """
    def __init__(self, chi_block: type, field_block: type, num_blocks: int):
        super().__init__()
        chi_channels = [8, 16, 128, 256, 256, 128, 2]  
        field_channels = [6, 12, 128, 256, 256, 128, 6]  
        self.chi_layers = nn.ModuleList([self._make_layer(chi_block, chi_channels, num_blocks) for _ in range(7)])
        self.field_layers = nn.ModuleList([self._make_layer(field_block, field_channels, num_blocks) for _ in range(7)])

    def _make_layer(self, block: type, channel_sizes: list[int], num_blocks: int, stride: int = 1) -> MySequential:
        layers = [block(channel_sizes, stride, None)]
        for _ in range(1, num_blocks):
            layers.append(block(channel_sizes, stride, None))
        return MySequential(*layers)

    def forward(self, initial_contrast: torch.Tensor, initial_total_field: torch.Tensor, 
                operator_GS: torch.Tensor, operator_GD: torch.Tensor, 
                incident_field: torch.Tensor, scattered_field: torch.Tensor):
        """
        Forward pass through alternating chi and field layers.

        Args:
            initial_contrast: Initial contrast tensor.
            initial_total_field: Initial total field tensor.
            operator_GS: Operator tensor for (GS).
            operator_GD: Operator tensor for (GD).
            incident_field: Incident field tensor.
            scattered_field: Scattered field tensor.

        Returns:
            Tuple of final contrast, total field, and stacked intermediate outputs.
        """
        contrast_outputs = []
        field_outputs = []
        contrast = initial_contrast
        total_field = initial_total_field

        for i in range(7):
            contrast, operator_GS, total_field, scattered_field, res_chi = \
                self.chi_layers[i](contrast, operator_GS, total_field, scattered_field)
            contrast_outputs.append(contrast)
            total_field, operator_GD, contrast, incident_field, res_field = \
                self.field_layers[i](total_field, operator_GD, contrast, incident_field)
            field_outputs.append(total_field)

        contrast_stack = torch.stack(contrast_outputs, dim=4)
        field_stack = torch.stack(field_outputs, dim=4)
        return contrast, total_field, contrast_stack, field_stack

class ResnetModel(nn.Module):
    """
    Full model integrating ResNet with loss computation.

    Args:
        resnet: ResNet class.
        chi_block: Chi residual block class.
        field_block: Field residual block class.
        regloss: Regularization loss class.
        num_blocks: Number of blocks per layer.
    """
    def __init__(self, resnet: type, chi_block: type, field_block: type, regloss: type, num_blocks: int):
        super().__init__()
        self.resnet = resnet(chi_block, field_block, num_blocks)
        self.regloss = regloss()
        self.log_sigma_square_1 = nn.Parameter(torch.Tensor(1).uniform_(1.0, 1.5))
        self.log_sigma_square_2 = nn.Parameter(torch.Tensor(1).uniform_(1.0, 1.5))
        self.log_sigma_square_3 = nn.Parameter(torch.Tensor(1).uniform_(1.0, 1.5))

    def forward(self, initial_contrast: torch.Tensor, initial_total_field: torch.Tensor, 
                operator_GS: torch.Tensor, operator_GD: torch.Tensor, 
                incident_field: torch.Tensor, scattered_field: torch.Tensor):
        """
        Forward pass computing predictions and losses.

        Args:
            initial_contrast: Initial contrast tensor.
            initial_total_field: Initial total field tensor.
            operator_GS: Operator tensor for (GS).
            operator_GD: Operator tensor for (GD).
            incident_field: Incident field tensor.
            scattered_field: Scattered field tensor.

        Returns:
            Tuple of total loss, final contrast, total field, intermediates, component losses, and sigmas.
        """
        from .residual_blocks import BaseResidualBlock
        contrast, total_field, contrast_stack, field_stack = self.resnet(
            initial_contrast, initial_total_field, operator_GS, 
            operator_GD, incident_field, scattered_field
        )

        # Field residual
        field_operator = BaseResidualBlock().apply_operator(contrast, operator_GD, total_field)
        residual_field = incident_field + field_operator - total_field
        loss_field_1 = torch.mean(residual_field[:, 0:2, :, :].pow(2))
        loss_field_2 = torch.mean(residual_field[:, 2:4, :, :].pow(2))
        loss_field_3 = torch.mean(residual_field[:, 4:6, :, :].pow(2))

        # Source residual
        source_operator = BaseResidualBlock().apply_operator(contrast, operator_GS, total_field)
        residual_source = scattered_field - source_operator
        loss_source_1 = torch.mean(residual_source[:, 0:2, :, :].pow(2))
        loss_source_2 = torch.mean(residual_source[:, 2:4, :, :].pow(2))
        loss_source_3 = torch.mean(residual_source[:, 4:6, :, :].pow(2))

        reg_loss = self.regloss(contrast)

        loss_component_1 = loss_field_1 + loss_source_1 + 1e-5 * reg_loss
        loss_component_2 = loss_field_2 + loss_source_2 + 1e-5 * reg_loss
        loss_component_3 = loss_field_3 + loss_source_3 + 1e-5 * reg_loss

        total_loss = (loss_component_1 / torch.exp(self.log_sigma_square_1) + self.log_sigma_square_1) + \
                     (loss_component_2 / torch.exp(self.log_sigma_square_2) + self.log_sigma_square_2) + \
                     (loss_component_3 / torch.exp(self.log_sigma_square_3) + self.log_sigma_square_3)

        return (total_loss, contrast, total_field, contrast_stack, field_stack,
                loss_component_1, loss_component_2, loss_component_3,
                self.log_sigma_square_1, self.log_sigma_square_2, self.log_sigma_square_3)