import h5py, torch
from torch.utils.data import Dataset
class H5Dataset(Dataset):
    def __init__(s, p, i): s.p, s.i, s.f = p, i, None
    def __len__(s): return len(s.i)
    def __getitem__(s, x): 
        s.f = s.f or h5py.File(s.p, 'r')
        return tuple(torch.tensor(s.f[k][s.i[x]], dtype=t) for k, t in zip(['inputs', 'policy_targets', 'value_targets'], [torch.float32, torch.long, torch.float32]))
    def __del__(s): s.f and s.f.close()