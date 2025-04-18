import os
import torch

class TPU:
    def __init__(self):
        self.available = False
        self.xm = None
        self.pl = None
        self.device = None
        self._init()
        
    def _init(self):
        try:
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.parallel_loader as pl
            self.device = xm.xla_device()
            self.xm = xm
            self.pl = pl
            self.available = True
            os.environ['XLA_USE_BF16'] = '1'
            os.environ['XLA_TENSOR_ALLOCATOR_MAXSIZE'] = '1000000000'
            print("TPU initialized")
        except ImportError:
            print("TPU not available")
            self.available = False
        except Exception as e:
            print(f"TPU init failed: {e}")
            self.available = False
    
    def get_device(self):
        return self.device if self.available else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_type(self):
        if self.available:
            return "tpu"
        return "gpu" if torch.cuda.is_available() else "cpu"
    
    def wrap_loader(self, dataloader):
        return self.pl.MpDeviceLoader(dataloader, self.device) if self.available else dataloader
    
    def step(self, optimizer):
        if self.available:
            self.xm.optimizer_step(optimizer)
        else:
            optimizer.step()
    
    def save(self, model, path):
        if self.available:
            cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(cpu_state_dict, path)
        else:
            torch.save(model.state_dict(), path)

    def load(self, path, map_location=None):
        try:
            if map_location is None:
                map_location = self.get_device()
            
            return torch.load(path, map_location=map_location)
        except RuntimeError as e:
            if "torch.storage.UntypedStorage (tagged with xla:0)" in str(e):
                print("Detected TPU storage location issue, attempting fix...")
                checkpoint = torch.load(path, map_location='cpu')
                return checkpoint
            else:
                raise

_tpu_instance = None

def get_tpu():
    global _tpu_instance
    if _tpu_instance is None:
        _tpu_instance = TPU()
    return _tpu_instance