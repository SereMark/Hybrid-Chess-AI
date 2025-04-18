import os
import shutil
import time
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
    
    def mount(self, max_retries=3):
        if not self.mounted:
            for attempt in range(max_retries):
                try:
                    print(f"Attempting to mount Google Drive (attempt {attempt+1}/{max_retries})...")
                    drive.mount(self.mount_path)
                    self.mounted = True
                    print("Drive mounted successfully.")
                    break
                except Exception as e:
                    print(f"Mount attempt {attempt+1} failed: {e}")
                    if attempt < max_retries - 1:
                        print("Retrying in 2 seconds...")
                        time.sleep(2)
                    else:
                        raise RuntimeError(f"Failed to mount Google Drive after {max_retries} attempts") from e
        return self.mount_path
    
    def setup(self, project_name):
        self.mount()
        self.project_dir = os.path.join(self.mount_path, 'MyDrive', project_name)
        
        if not os.path.exists(self.project_dir):
            print(f"Project directory not found: {self.project_dir}")
            create = input("Create project directory? (y/n) [y]: ").strip().lower() or 'y'
            if create != 'y':
                raise FileNotFoundError(f"Project directory not found: {self.project_dir}")
        
        os.makedirs(self.project_dir, exist_ok=True)
        print(f"Project directory: {self.project_dir}")
        
        for dir_name in ['data', 'models', 'logs', 'checkpoints']:
            dir_path = os.path.join(self.project_dir, dir_name)
            os.makedirs(dir_path, exist_ok=True)
            print(f"Ensured directory exists: {dir_path}")
            
        return self.project_dir
    
    def path(self, relative_path):
        if not self.project_dir:
            raise ValueError("Project directory not set. Call setup() first.")
        return os.path.join(self.project_dir, relative_path)
    
    def save(self, source_path, relative_destination):
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        dest_path = self.path(relative_destination)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        shutil.copy2(source_path, dest_path)
        print(f"Saved file: {dest_path}")
        return dest_path
    
    def load(self, relative_path, local_destination=None):
        source_path = self.path(relative_path)
        
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"File not found in Drive: {relative_path}")
        
        if local_destination:
            os.makedirs(os.path.dirname(local_destination), exist_ok=True)
            shutil.copy2(source_path, local_destination)
            print(f"Loaded file to: {local_destination}")
            return local_destination
        
        print(f"Loaded file: {source_path}")
        return source_path
    
    def list(self, relative_path, pattern=None):
        dir_path = self.path(relative_path)
        
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
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
            print(f"Copying dataset from drive to local: {os.path.basename(drive_path)}")
            shutil.copy2(drive_path, local_path)
            print(f"Dataset copied to: {local_path}")
        else:
            print(f"Using cached dataset: {local_path}")
            
        return local_path

def get_drive():
    return Drive()