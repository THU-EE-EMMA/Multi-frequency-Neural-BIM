import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import logging
import os
from torch.utils.data import DataLoader, random_split
from src import ResnetModel, ChiResidualBlock, FieldResidualBlock, RegLossFunc, ResNet
from src.dataset import ElectromagneticDataset
from src.train_utils import train_epoch, evaluate_train_epoch, evaluate_test_epoch, save_results, save_losses
from src.utils import set_seed, init_weights

def main(config: dict) -> None:
    """
    Main function to train and evaluate the electromagnetic ResNet model.

    Args:
        config: Configuration dictionary loaded from YAML.
    """
    os.makedirs(config['output_dir'], exist_ok=True)
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config['output_dir'], 'training.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Set random seed for reproducibility
    set_seed(config['seed'])

    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() and config['use_cuda'] else 'cpu')
    logger.info(f"Using device: {device}")

    # Initialize model
    model = ResnetModel(ResNet, ChiResidualBlock, FieldResidualBlock, RegLossFunc, num_blocks=config['num_blocks'])
    model = nn.DataParallel(model, device_ids=config['device_ids'])
    model.to(device)
    model.apply(init_weights)
    
    # Load pre-trained weights if specified
    if config['pretrained_model_path']:
        model.load_state_dict(torch.load(config['pretrained_model_path'], map_location=device))
        logger.info(f"Loaded pre-trained model from {config['pretrained_model_path']}")

    # Initialize dataset and dataloaders
    dataset = ElectromagneticDataset(
        height=config['height'],
        width=config['width'],
        num_receivers=config['num_receivers']
    )
    train_size = int(config['train_split'] * len(dataset))
    test_size = len(dataset) - train_size
    logger.info(f"Train size: {train_size}, Test size: {test_size}")
    
    train_data, test_data = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(config['seed']))
    train_loader = DataLoader(
        train_data, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'], 
        pin_memory=config['use_cuda'], 
        prefetch_factor=2
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=config['num_workers'], 
        pin_memory=config['use_cuda'], 
        prefetch_factor=2
    )

    # Initialize optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['scheduler_step_size'], gamma=config['scheduler_gamma'])

    # Training loop
    train_losses = []
    test_losses = []
    train_freq1_losses, train_freq2_losses, train_freq3_losses = [], [], []
    test_freq1_losses, test_freq2_losses, test_freq3_losses = [], [], []
    log_sigma_squares = {'freq1': [], 'freq2': [], 'freq3': []}

    for epoch in range(1, config['num_epochs'] + 1):
        # Train
        losses = train_epoch(epoch, train_loader, optimizer, model, config['use_cuda'])
        train_losses.append(losses['total'])
        train_freq1_losses.append(losses['freq1'])
        train_freq2_losses.append(losses['freq2'])
        train_freq3_losses.append(losses['freq3'])
        logger.info(f"Epoch {epoch} Train Loss: {losses['total']:.10f}, "
                    f"Freq1: {losses['freq1']:.10f}, Freq2: {losses['freq2']:.10f}, Freq3: {losses['freq3']:.10f}")

        # Evaluate on train set
        eval_losses, sigmas = evaluate_train_epoch(epoch, train_loader, model, config['use_cuda'])
        log_sigma_squares['freq1'].append(sigmas['freq1'])
        log_sigma_squares['freq2'].append(sigmas['freq2'])
        log_sigma_squares['freq3'].append(sigmas['freq3'])
        logger.info(f"Epoch {epoch} Train Eval Loss: {eval_losses['total']:.10f}, "
                    f"Freq1: {eval_losses['freq1']:.10f}, Freq2: {eval_losses['freq2']:.10f}, Freq3: {eval_losses['freq3']:.10f}, "
                    f"Log Sigma^2: {sigmas['freq1']:.10f}, {sigmas['freq2']:.10f}, {sigmas['freq3']:.10f}")

        # Evaluate on test set
        test_eval_losses = evaluate_test_epoch(epoch, test_loader, model, config['use_cuda'])
        test_losses.append(test_eval_losses['total'])
        test_freq1_losses.append(test_eval_losses['freq1'])
        test_freq2_losses.append(test_eval_losses['freq2'])
        test_freq3_losses.append(test_eval_losses['freq3'])
        logger.info(f"Epoch {epoch} Test Loss: {test_eval_losses['total']:.10f}, "
                    f"Freq1: {test_eval_losses['freq1']:.10f}, Freq2: {test_eval_losses['freq2']:.10f}, Freq3: {test_eval_losses['freq3']:.10f}")

        scheduler.step()

        # Save results and losses periodically
        if epoch == 1 or epoch % config['save_interval'] == 0:
            loss_dir = os.path.join(config['output_dir'], f"epoch_{epoch}_loss")
            save_losses(
                train_losses=train_losses, test_losses=test_losses,
                train_freq_losses={'freq1': train_freq1_losses, 'freq2': train_freq2_losses, 'freq3': train_freq3_losses},
                test_freq_losses={'freq1': test_freq1_losses, 'freq2': test_freq2_losses, 'freq3': test_freq3_losses},
                log_sigma_squares=log_sigma_squares,
                save_dir=loss_dir
            )
            save_results(epoch, model, train_loader, test_loader, config['use_cuda'], config['output_dir'])
            model_path = os.path.join(config['output_dir'], f"res_multiF_{epoch}.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved model checkpoint to {model_path}")

if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    main(config)