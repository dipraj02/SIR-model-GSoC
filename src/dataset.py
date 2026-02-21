import torch
from torch.utils.data import Dataset
import numpy as np

class SIRDataset(Dataset):
    """
    Custom PyTorch Dataset for the SIR Model.
    """
    def __init__(self, x_path, y_path):
        """
        Args:
            x_path (str): Path to the input features (S, I, R curves) .npy file.
            y_path (str): Path to the target labels (beta, gamma) .npy file.
        """
        # Load data from .npy files
        self.X = np.load(x_path).astype(np.float32)
        self.Y = np.load(y_path).astype(np.float32)
        
        # Check if lengths match
        assert len(self.X) == len(self.Y), "Input and Target lengths do not match!"

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns a single sample at index `idx`.
        Output format: (Input Tensor, Target Tensor)
        """
        # Get the sequence (Shape: [Sequence_Length, Features])
        x_sample = self.X[idx] 
        
        # Get the parameters (Shape: [2]) -> beta, gamma
        y_sample = self.Y[idx]
        
        # Convert to PyTorch Tensors
        return torch.tensor(x_sample), torch.tensor(y_sample)