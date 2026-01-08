import torch

def measure_runtime(kernel_fn, args, kwargs, n_warmup=10, n_repeat=100):
    for _ in range(n_warmup):
        kernel_fn(*args, **kwargs)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(n_repeat):
        kernel_fn(*args, **kwargs)
    end_event.record()
    
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / n_repeat
