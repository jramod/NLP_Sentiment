# src/utils.py
import time
import random
import numpy as np
import torch
import platform
import psutil

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def epoch_timer(start_time, end_time):
    return end_time - start_time  # seconds

def get_hardware_report():
    return {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cpu_name": platform.processor(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2)
    }
