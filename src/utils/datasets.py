import h5py
import torch
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
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_file_path, 'r')

        try:
            actual_idx = self.indices[idx]
            inp_t = torch.from_numpy(self.h5_file['inputs'][actual_idx]).float()
            pol_t = torch.tensor(self.h5_file['policy_targets'][actual_idx]).long()
            val_t = torch.tensor(self.h5_file['value_targets'][actual_idx]).float()
            return inp_t, pol_t, val_t
        except Exception as e:
            raise RuntimeError(f"Error loading data at index {idx}: {e}")

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()