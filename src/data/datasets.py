import torch
from torch.utils.data import Dataset


class H5Dataset(Dataset):
    def __init__(self, h5_file, indices):
        self.h5_file = h5_file
        self.indices = indices
        self.input_shape = h5_file['inputs'].shape[1:]
        self.policy_shape = h5_file['policy_targets'].shape[1:]
        self.value_shape = h5_file['value_targets'].shape[1:]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        try:
            actual_idx = self.indices[idx]
            input_tensor = self.h5_file['inputs'][actual_idx]
            policy_target = self.h5_file['policy_targets'][actual_idx]
            value_target = self.h5_file['value_targets'][actual_idx]

            if input_tensor.shape != self.input_shape:
                raise ValueError(f"Input shape mismatch at index {actual_idx}")
            if policy_target.shape != self.policy_shape:
                raise ValueError(f"Policy target shape mismatch at index {actual_idx}")
            if value_target.shape != self.value_shape:
                raise ValueError(f"Value target shape mismatch at index {actual_idx}")

            input_tensor = torch.from_numpy(input_tensor).float()
            policy_target = torch.tensor(policy_target).long()
            value_target = torch.tensor(value_target).float()
            return input_tensor, policy_target, value_target
        except Exception as e:
            raise RuntimeError(f"Error loading data at index {idx}: {str(e)}")