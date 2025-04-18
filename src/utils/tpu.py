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
        try:
            cpu_state_dict = {}
            for k, v in model.state_dict().items():
                if isinstance(v, torch.Tensor):
                    cpu_state_dict[k] = v.detach().cpu()
                else:
                    cpu_state_dict[k] = v
                    
            if isinstance(model, dict) and 'model' in model:
                model_copy = model.copy()
                model_copy['model'] = cpu_state_dict
                torch.save(model_copy, path)
            else:
                torch.save(cpu_state_dict, path)
                
            print(f"Model saved to {path} (with CPU tensors)")
        except Exception as e:
            print(f"Error saving model: {e}")
            if self.available:
                self.xm.save(model.state_dict(), path)
            else:
                torch.save(model.state_dict(), path)

    def load(self, path, map_location=None):
        try:
            if map_location is None:
                map_location = 'cpu'
            
            checkpoint = torch.load(path, map_location=map_location)
            return checkpoint
        except RuntimeError as e:
            if "torch.storage.UntypedStorage (tagged with xla:0)" in str(e):
                print("Detected TPU storage location issue, attempting fix...")
                try:
                    checkpoint = torch.load(path, map_location='cpu')
                    return checkpoint
                except Exception as e2:
                    print(f"TPU compatibility fix failed: {e2}, trying alternative approach...")
                    try:
                        import io
                        with open(path, 'rb') as f:
                            buffer = io.BytesIO(f.read())
                        checkpoint = torch.load(buffer, map_location='cpu')
                        return checkpoint
                    except Exception as e3:
                        print(f"All loading methods failed: {e3}")
                        raise
            else:
                raise

_tpu_instance = None

def get_tpu():
    global _tpu_instance
    if _tpu_instance is None:
        _tpu_instance = TPU()
    return _tpu_instance