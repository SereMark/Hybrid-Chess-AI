import torch
import h5py
from torch.utils.data import Dataset

class H5Dataset(Dataset):
    def __init__(self, h5_file_path, indices):
        self.h5_file_path = h5_file_path
        self.indices = indices
        self.h5_file = None

        with h5py.File(self.h5_file_path, 'r') as f:
            self.input_shape = f['inputs'].shape[1:]
            self.policy_shape = f['policy_targets'].shape[1:] if len(f['policy_targets'].shape) > 1 else ()
            self.value_shape = f['value_targets'].shape[1:] if len(f['value_targets'].shape) > 1 else ()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Open the HDF5 file if it hasn't been opened yet
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_file_path, 'r')

        try:
            actual_idx = self.indices[idx]
            inp = self.h5_file['inputs'][actual_idx]
            pol = self.h5_file['policy_targets'][actual_idx]
            val = self.h5_file['value_targets'][actual_idx]

            # Validate the shapes of the data
            if inp.shape != self.input_shape:
                raise ValueError(f"Input shape mismatch at index {actual_idx}: {inp.shape} != {self.input_shape}")
            if pol.shape != self.policy_shape:
                raise ValueError(f"Policy target shape mismatch at index {actual_idx}: {pol.shape} != {self.policy_shape}")
            if val.shape != self.value_shape:
                raise ValueError(f"Value target shape mismatch at index {actual_idx}: {val.shape} != {self.value_shape}")

            # Convert data to tensors
            inp_t = torch.from_numpy(inp).float()
            pol_t = torch.tensor(pol).long()
            val_t = torch.tensor(val).float()
            return inp_t, pol_t, val_t
        except Exception as e:
            raise RuntimeError(f"Error loading data at index {idx}: {e}")

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()