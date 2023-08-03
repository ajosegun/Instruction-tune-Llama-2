import subprocess
import torch

def check_gpu_capacity():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.free,memory.used', '--format=csv,nounits,noheader'], encoding='utf-8')
        gpu_info = [line.split(', ') for line in output.strip().split('\n')]
        print("GPU Information:")
        for info in gpu_info:
            index, total_memory, free_memory, used_memory = info
            print(f"GPU {index}:")
            print(f"  Total Memory: {total_memory} MB")
            print(f"  Free Memory: {free_memory} MB")
            print(f"  Used Memory: {used_memory} MB")
    except Exception as e:
        print(f"Error: {e}")

# check_gpu_capacity()

def free_gpu_memory():
    torch.cuda.empty_cache()
    
    tensor = torch.randn((1000, 1000)).cuda()
    # Free up the GPU memory by setting the tensor to None
    tensor = None
    
def memory_stats():
    # Initialize a PyTorch tensor on GPU
    tensor = torch.randn((1000, 1000)).cuda()
    
    gpu_memory_used = torch.cuda.memory_allocated() / (1024 ** 3)  # Convert to GB
    gpu_memory_cached = torch.cuda.memory_reserved() / (1024 ** 3)  # Convert to GB
    
    print(f"GPU Memory Used: {gpu_memory_used:.2f} GB")
    print(f"GPU Memory Cached: {gpu_memory_cached:.2f} GB")
