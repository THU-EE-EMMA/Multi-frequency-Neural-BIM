import torch
import h5py
import numpy as np
from torch.utils.data import Dataset

class ElectromagneticDataset(Dataset):
    """
    Dataset for loading electromagnetic simulation data from HDF5 files.

    Args:
        height: Height of the spatial grid.
        width: Width of the spatial grid.
        num_receivers: Number of receivers (Nre).
    """
    def __init__(self, height: int, width: int, num_receivers: int):
        self.height = height
        self.width = width
        self.num_receivers = num_receivers

        # Load operators
        self.operator_GD = self._load_operators("GD")
        self.operator_GS = self._load_operators("GS")

        # Load field data
        self.incident_field = self._load_field_data("pinc")
        self.total_field = self._load_field_data("ptot")
        self.scattered_field = self._load_field_data("psca")

        # Load contrast data
        self.target_contrast = self._load_contrast("chi_all")
        self.initial_contrast = self._load_contrast("chi0_all")

    def _load_operators(self, type: str):
        """
        Load operator data (real and imaginary parts) for GS or GD.

        Args:
            type: Either 'GS'  or 'GD' .

        Returns:
            Stacked tensor of shape [6, height*width, height*width or num_receivers].
        """
        prefix = f"Z{type}"
        real_data = [h5py.File(f"{prefix}{i}real.mat")[f"{prefix}{i}real"][:][:].T for i in range(1, 4)]
        imag_data = [h5py.File(f"{prefix}{i}imag.mat")[f"{prefix}{i}imag"][:][:].T for i in range(1, 4)]
        tensors = [torch.from_numpy(data).float() for data in real_data + imag_data]
        
        # Reshape based on type
        target_dim = self.num_receivers if type == "GS" else self.height * self.width
        for i in range(len(tensors)):
            tensors[i] = tensors[i].reshape(self.height * self.width, target_dim).T
        return torch.stack(tensors, dim=0)

    def _load_field_data(self, prefix: str) -> torch.Tensor:
        """
        Load field data (incident, total, or scattered) from HDF5 files.

        Args:
            prefix: Data type ('pinc', 'ptot', or 'psca').

        Returns:
            Stacked tensor of shape [num_samples, 6, height*width or num_receivers, 1].
        """
        real_data = [h5py.File(f"{prefix}{i}real.mat")[f"{prefix}{i}real"][:][:].T for i in range(1, 4)]
        imag_data = [h5py.File(f"{prefix}{i}imag.mat")[f"{prefix}{i}imag"][:][:].T for i in range(1, 4)]
        tensors = [torch.from_numpy(data).float() for data in real_data + imag_data]
        target_dim = self.num_receivers if prefix == "psca" else self.height * self.width
        for i in range(len(tensors)):
            tensors[i] = tensors[i].reshape(-1, self.num_receivers, target_dim).transpose(1, 2)
        return torch.stack(tensors, dim=1)

    def _load_contrast(self, prefix: str) -> torch.Tensor:
        """
        Load contrast data (real and imaginary parts).

        Args:
            prefix: Data type ('chi_all' or 'chi0_all').

        Returns:
            Stacked tensor of shape [num_samples, 2, height, width].
        """
        real_data = h5py.File(f"{prefix}_real.mat")[f"{prefix}_real"][:][:].T
        imag_data = h5py.File(f"{prefix}_imag.mat")[f"{prefix}_imag"][:][:].T
        real_tensor = torch.from_numpy(real_data).float().reshape(-1, self.height, self.height)
        imag_tensor = torch.from_numpy(imag_data).float().reshape(-1, self.height, self.height)
        return torch.stack([real_tensor, imag_tensor], dim=1)

    def __getitem__(self, index: int):
        """
        Get a single data sample.

        Args:
            index: Index of the sample.

        Returns:
            Tuple of (operator_GS, operator_GD, incident_field,
                      scattered_field, total_field, target_contrast, initial_contrast).
        """
        return (
            self.operator_GS,
            self.operator_GD,
            self.incident_field[index],
            self.scattered_field[index],
            self.total_field[index],
            self.target_contrast[index],
            self.initial_contrast[index]
        )

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            Number of samples.
        """
        return len(self.total_field)