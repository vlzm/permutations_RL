import torch
import os

class DeviceManager:
    def __init__(self, gpu_id=None):
        """
        Initialize device manager
        
        Args:
            gpu_id (int, optional): Specific GPU ID to use. If None, will use CUDA_VISIBLE_DEVICES environment variable.
        """
        if gpu_id is not None:
            # Set specific GPU
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            self.device = torch.device('cuda:0')
        else:
            # Use default device selection
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
    def get_device(self):
        """Get the current device"""
        return self.device
    
    @staticmethod
    def get_available_gpus():
        """Get number of available GPUs"""
        return torch.cuda.device_count()
    
    @staticmethod
    def set_gpu_memory_fraction(fraction):
        """Set GPU memory fraction to use"""
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(fraction) 