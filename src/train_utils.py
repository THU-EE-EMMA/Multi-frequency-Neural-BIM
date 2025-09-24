import torch
import os
import scipy.io as sio
import logging
import numpy as np
from torch.utils.data import DataLoader

def train_epoch(epoch: int, train_loader: DataLoader, optimizer: torch.optim.Optimizer, 
                model: torch.nn.Module, use_cuda: bool) -> dict[str, float]:
    """
    Train the model for one epoch.

    Args:
        epoch: Current epoch number.
        train_loader: DataLoader for training data.
        optimizer: Optimizer for model parameters.
        model: The neural network model.
        use_cuda: Whether to use CUDA.

    Returns:
        Dictionary of average losses (total, freq1, freq2, freq3).
    """
    model.train()
    running_losses = {'total': 0.0, 'freq1': 0.0, 'freq2': 0.0, 'freq3': 0.0}
    num_batches = 0

    for operator_GS, operator_GD, incident_field, scattered_field, total_field, _, initial_contrast in train_loader:
        if use_cuda:
            operator_GS = operator_GS.cuda()
            operator_GD = operator_GD.cuda()
            incident_field = incident_field.cuda()
            scattered_field = scattered_field.cuda()
            total_field = total_field.cuda()
            initial_contrast = initial_contrast.cuda()

        optimizer.zero_grad()
        outputs = model(initial_contrast, total_field, operator_GS, operator_GD, incident_field, scattered_field)
        loss, _, _, _, _, loss_freq1, loss_freq2, loss_freq3, _, _, _ = outputs
        loss = torch.mean(loss)
        loss_freq1 = torch.mean(loss_freq1)
        loss_freq2 = torch.mean(loss_freq2)
        loss_freq3 = torch.mean(loss_freq3)

        loss.backward()
        optimizer.step()

        running_losses['total'] += loss.item()
        running_losses['freq1'] += loss_freq1.item()
        running_losses['freq2'] += loss_freq2.item()
        running_losses['freq3'] += loss_freq3.item()
        num_batches += 1

    return {k: v / num_batches for k, v in running_losses.items()}

def evaluate_train_epoch(epoch: int, train_loader: DataLoader, model: torch.nn.Module, 
                        use_cuda: bool) -> tuple[dict[str, float], dict[str, float]]:
    """
    Evaluate the model on the training set for one epoch.

    Args:
        epoch: Current epoch number.
        train_loader: DataLoader for training data.
        model: The neural network model.
        use_cuda: Whether to use CUDA.

    Returns:
        Tuple of (average losses dictionary, sigma squares dictionary).
    """
    model.eval()
    running_losses = {'total': 0.0, 'freq1': 0.0, 'freq2': 0.0, 'freq3': 0.0}
    running_sigmas = {'freq1': 0.0, 'freq2': 0.0, 'freq3': 0.0}
    num_batches = 0

    with torch.no_grad():
        for operator_GS, operator_GD, incident_field, scattered_field, total_field, _, initial_contrast in train_loader:
            if use_cuda:
                operator_GS = operator_GS.cuda()
                operator_GD = operator_GD.cuda()
                incident_field = incident_field.cuda()
                scattered_field = scattered_field.cuda()
                total_field = total_field.cuda()
                initial_contrast = initial_contrast.cuda()

            outputs = model(initial_contrast, total_field, operator_GS, operator_GD, incident_field, scattered_field)
            loss, _, _, _, _, loss_freq1, loss_freq2, loss_freq3, sigma1, sigma2, sigma3 = outputs
            loss = torch.mean(loss)
            loss_freq1 = torch.mean(loss_freq1)
            loss_freq2 = torch.mean(loss_freq2)
            loss_freq3 = torch.mean(loss_freq3)
            sigma1 = torch.mean(sigma1)
            sigma2 = torch.mean(sigma2)
            sigma3 = torch.mean(sigma3)

            running_losses['total'] += loss.item()
            running_losses['freq1'] += loss_freq1.item()
            running_losses['freq2'] += loss_freq2.item()
            running_losses['freq3'] += loss_freq3.item()
            running_sigmas['freq1'] += sigma1.item()
            running_sigmas['freq2'] += sigma2.item()
            running_sigmas['freq3'] += sigma3.item()
            num_batches += 1

    return {k: v / num_batches for k, v in running_losses.items()}, {k: v / num_batches for k, v in running_sigmas.items()}

def evaluate_test_epoch(epoch: int, test_loader: DataLoader, model: torch.nn.Module, 
                       use_cuda: bool) -> dict[str, float]:
    """
    Evaluate the model on the test set for one epoch.

    Args:
        epoch: Current epoch number.
        test_loader: DataLoader for test data.
        model: The neural network model.
        use_cuda: Whether to use CUDA.

    Returns:
        Dictionary of average losses (total, freq1, freq2, freq3).
    """
    model.eval()
    running_losses = {'total': 0.0, 'freq1': 0.0, 'freq2': 0.0, 'freq3': 0.0}
    num_batches = 0

    with torch.no_grad():
        for operator_GS, operator_GD, incident_field, scattered_field, total_field, _, initial_contrast in test_loader:
            if use_cuda:
                operator_GS = operator_GS.cuda()
                operator_GD = operator_GD.cuda()
                incident_field = incident_field.cuda()
                scattered_field = scattered_field.cuda()
                total_field = total_field.cuda()
                initial_contrast = initial_contrast.cuda()

            outputs = model(initial_contrast, total_field, operator_GS, operator_GD, incident_field, scattered_field)
            loss, _, _, _, _, loss_freq1, loss_freq2, loss_freq3, _, _, _ = outputs
            loss = torch.mean(loss)
            loss_freq1 = torch.mean(loss_freq1)
            loss_freq2 = torch.mean(loss_freq2)
            loss_freq3 = torch.mean(loss_freq3)
            

            running_losses['total'] += loss.item()
            running_losses['freq1'] += loss_freq1.item()
            running_losses['freq2'] += loss_freq2.item()
            running_losses['freq3'] += loss_freq3.item()
            num_batches += 1

    return {k: v / num_batches for k, v in running_losses.items()}

def save_losses(train_losses: list, test_losses: list, train_freq_losses: dict, 
                test_freq_losses: dict, log_sigma_squares: dict, save_dir: str) -> None:
    """
    Save training and test losses to MAT files.

    Args:
        train_losses: List of total training losses.
        test_losses: List of total test losses.
        train_freq_losses: Dictionary of training frequency losses.
        test_freq_losses: Dictionary of test frequency losses.
        log_sigma_squares: Dictionary of log sigma square values.
        save_dir: Directory to save the loss files.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sio.savemat(os.path.join(save_dir, 'loss.mat'), 
                {'trainloss': np.array(train_losses), 'testloss': np.array(test_losses)})
    for freq in ['freq1', 'freq2', 'freq3']:
        sio.savemat(os.path.join(save_dir, f'{freq}_loss.mat'),
                    {f'train_{freq}_loss': np.array(train_freq_losses[freq]),
                     f'test_{freq}_loss': np.array(test_freq_losses[freq]),
                     f'log_sigma_square_{freq}': np.array(log_sigma_squares[freq])})

def save_results(epoch: int, model: torch.nn.Module, train_loader: DataLoader, 
                 test_loader: DataLoader, use_cuda: bool, output_dir: str) -> None:
    """
    Save model outputs for train and test sets to MAT files.

    Args:
        epoch: Current epoch number.
        model: The neural network model.
        train_loader: DataLoader for training data.
        test_loader: DataLoader for test data.
        use_cuda: Whether to use CUDA.
        output_dir: Directory to save results.
    """
    model.eval()
    train_result_dir = os.path.join(output_dir, f"epoch_{epoch}_train_results")
    test_result_dir = os.path.join(output_dir, f"epoch_{epoch}_test_results")
    os.makedirs(train_result_dir, exist_ok=True)
    os.makedirs(test_result_dir, exist_ok=True)

    with torch.no_grad():
        for i, (operator_GS, operator_GD, incident_field, scattered_field, total_field, target_contrast, initial_contrast) in enumerate(train_loader):
            if use_cuda:
                operator_GS = operator_GS.cuda()
                operator_GD = operator_GD.cuda()
                incident_field = incident_field.cuda()
                scattered_field = scattered_field.cuda()
                total_field = total_field.cuda()
                initial_contrast = initial_contrast.cuda()

            outputs = model(initial_contrast, total_field, operator_GS, operator_GD, incident_field, scattered_field)
            _, contrast, total_field_out, contrast_stack, total_field_stack, *_ = outputs

            sio.savemat(os.path.join(train_result_dir, f'train_{i}.mat'),
                        {
                            'contrast': contrast.cpu().numpy(),
                            'total_field': total_field_out.cpu().numpy(),
                            'incident_field': incident_field.cpu().numpy(),
                            'total_field_true': total_field.cpu().numpy(),
                            'contrast_true': target_contrast.cpu().numpy(),
                            'initial_contrast': initial_contrast.cpu().numpy(),
                            'scattered_field': scattered_field.cpu().numpy()
                        })
            sio.savemat(os.path.join(train_result_dir, f'train_op_{i}.mat'),
                        {
                            'contrast_stack': contrast_stack.cpu().numpy(),
                            'total_field_stack': total_field_stack.cpu().numpy()
                        })

        for i, (operator_GS, operator_GD, incident_field, scattered_field, total_field, target_contrast, initial_contrast) in enumerate(test_loader):
            if use_cuda:
                operator_GS = operator_GS.cuda()
                operator_GD = operator_GD.cuda()
                incident_field = incident_field.cuda()
                scattered_field = scattered_field.cuda()
                total_field = total_field.cuda()
                initial_contrast = initial_contrast.cuda()

            outputs = model(initial_contrast, total_field, operator_GS, operator_GD, incident_field, scattered_field)
            _, contrast, total_field_out, contrast_stack, total_field_stack, *_ = outputs

            sio.savemat(os.path.join(test_result_dir, f'test_{i}.mat'),
                        {
                            'contrast': contrast.cpu().numpy(),
                            'total_field': total_field_out.cpu().numpy(),
                            'incident_field': incident_field.cpu().numpy(),
                            'total_field_true': total_field.cpu().numpy(),
                            'contrast_true': target_contrast.cpu().numpy(),
                            'initial_contrast': initial_contrast.cpu().numpy(),
                            'scattered_field': scattered_field.cpu().numpy()
                        })
            sio.savemat(os.path.join(test_result_dir, f'test_op_{i}.mat'),
                        {
                            'contrast_stack': contrast_stack.cpu().numpy(),
                            'total_field_stack': total_field_stack.cpu().numpy()
                        })