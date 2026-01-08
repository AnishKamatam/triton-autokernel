import torch

def measure_runtime(kernel_fn, args, kwargs, n_warmup=10, n_repeat=100):
    # Warmup runs to ensure GPU is in steady state (caches warmed, etc.)
    for _ in range(n_warmup):
        kernel_fn(*args, **kwargs)
    
    # Use CUDA events for accurate GPU timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(n_repeat):
        kernel_fn(*args, **kwargs)
    end_event.record()
    
    # Synchronize to ensure all operations complete before measuring
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / n_repeat  # Average time in milliseconds
