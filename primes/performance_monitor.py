import torch
import time
import psutil
from src.glastonbury_2048.cuda_optimizer import CUDAOptimizer

# Team Instruction: Implement performance monitor for GLASTONBURY 2048.
# Track CUDA core usage, IoT data throughput, and system metrics.
class PerformanceMonitor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = CUDAOptimizer()

    def monitor(self, data: torch.Tensor) -> dict:
        """Monitors CUDA and system performance during IoT data processing."""
        start_time = time.time()
        processed_data = self.optimizer.optimize(data)
        end_time = time.time()

        cuda_memory = torch.cuda.memory_allocated(self.device) / 1024**2  # MB
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent

        return {
            "processing_time": end_time - start_time,
            "cuda_memory_mb": cuda_memory,
            "cpu_usage_percent": cpu_usage,
            "memory_usage_percent": memory_usage,
            "data_shape": processed_data.shape
        }

# Example usage
if __name__ == "__main__":
    monitor = PerformanceMonitor()
    data = torch.randn(1000, 1000, device="cuda")
    metrics = monitor.monitor(data)
    print(f"Performance metrics: {metrics}")