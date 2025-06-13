import os
import sys
import subprocess
from src.utils.device_manager import DeviceManager

def run_instance(gpu_id, config_file=None):
    """
    Run a single instance of the solver on a specific GPU
    
    Args:
        gpu_id (int): GPU ID to use
        config_file (str, optional): Path to config file
    """
    # Set environment variable for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Build command
    cmd = [sys.executable, 'src/main.py']
    if config_file:
        cmd.extend(['--config', config_file])
    cmd.extend(['--gpu_id', str(gpu_id)])
    
    # Run process
    process = subprocess.Popen(cmd)
    return process

def main():
    # Get number of available GPUs
    n_gpus = DeviceManager.get_available_gpus()
    if n_gpus == 0:
        print("No GPUs available!")
        return
    
    print(f"Found {n_gpus} GPUs")
    
    # Run instances
    processes = []
    for gpu_id in range(n_gpus):
        print(f"Starting instance on GPU {gpu_id}")
        process = run_instance(gpu_id)
        processes.append(process)
    
    # Wait for all processes to complete
    for process in processes:
        process.wait()

if __name__ == '__main__':
    main() 