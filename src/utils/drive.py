import os
import shutil
from google.colab import drive
import glob

class Drive:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Drive, cls).__new__(cls)
            cls._instance._init()
        return cls._instance
    
    def _init(self):
        self.mount_path = '/content/drive'
        self.mounted = False
        self.project_dir = None
    
    def mount(self):
        if not self.mounted:
            drive.mount(self.mount_path)
            self.mounted = True
        return self.mount_path
    
    def setup(self, project_name):
        self.mount()
        self.project_dir = os.path.join(self.mount_path, 'MyDrive', project_name)
        os.makedirs(self.project_dir, exist_ok=True)
        
        for dir_name in ['data', 'models', 'logs', 'checkpoints']:
            os.makedirs(os.path.join(self.project_dir, dir_name), exist_ok=True)
            
        return self.project_dir
    
    def path(self, relative_path):
        if not self.project_dir:
            raise ValueError("Project directory not set")
        return os.path.join(self.project_dir, relative_path)
    
    def save(self, source_path, relative_destination):
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        dest_path = self.path(relative_destination)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        shutil.copy2(source_path, dest_path)
        return dest_path
    
    def load(self, relative_path, local_destination=None):
        source_path = self.path(relative_path)
        
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"File not found in Drive: {relative_path}")
        
        if local_destination:
            os.makedirs(os.path.dirname(local_destination), exist_ok=True)
            shutil.copy2(source_path, local_destination)
            return local_destination
        
        return source_path
    
    def list(self, relative_path, pattern=None):
        dir_path = self.path(relative_path)
        
        if not os.path.exists(dir_path):
            return []
        
        if pattern:
            files = glob.glob(os.path.join(dir_path, pattern))
            return [os.path.basename(f) for f in files]
        
        return os.listdir(dir_path)
    
    def get_dataset(self, relative_dataset_path, local_dir='/content/data'):
        drive_path = self.path(relative_dataset_path)
        
        if not os.path.exists(drive_path):
            raise FileNotFoundError(f"Dataset not found: {relative_dataset_path}")
        
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, os.path.basename(relative_dataset_path))
        
        if not os.path.exists(local_path):
            shutil.copy2(drive_path, local_path)
            
        return local_path

def get_drive():
    return Drive()