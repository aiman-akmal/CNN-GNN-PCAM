import time
import os
import psutil
import torch
import csv
from datetime import datetime

class ResourceTracker:
    def __init__(self, log_path="outputs/resource_metrics.csv"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.start_cpu_time = None
        
        # Initialize CSV header if file doesn't exist
        if not os.path.exists(self.log_path):
            with open(self.log_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "stage", "epoch", "step", 
                    "elapsed_sec", "cpu_percent", "ram_used_gb", 
                    "gpu_or_mps_allocated_gb"
                ])

    def start(self):
        self.start_time = time.perf_counter()
        self.start_cpu_time = self.process.cpu_times()
        self.process.cpu_percent(interval=None) # Reset baseline

    def log_step(self, stage: str, epoch: int = 0, step: int = 0):
        elapsed = time.perf_counter() - self.start_time
        cpu_usage = self.process.cpu_percent(interval=None)
        ram_gb = self.process.memory_info().rss / (1024 ** 3)
        
        # Check device allocation (MPS or CUDA)
        accel_gb = 0.0
        if torch.cuda.is_available():
            accel_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            accel_gb = torch.mps.current_allocated_memory() / (1024 ** 3)

        with open(self.log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                stage, epoch, step,
                f"{elapsed:.2f}",
                f"{cpu_usage:.1f}",
                f"{ram_gb:.3f}",
                f"{accel_gb:.3f}"
            ])
            
        print(f"[{stage} | Ep {epoch}] Time: {elapsed:.1f}s | CPU: {cpu_usage:.1f}% | RAM: {ram_gb:.2f}GB | Device RAM: {accel_gb:.2f}GB")