"""Local camera/video-source runtime."""


def configure_runtime(cfg):
    """Bound CPU parallelism for batch-one inference; zero preserves PyTorch defaults.

    CLI-only: constructing a pipeline does not change global training settings.
    """
    threads = cfg.get('runtime', {}).get('cpu_threads', 0)
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 0:
        raise ValueError('runtime.cpu_threads must be a non-negative integer')
    if threads:
        import torch
        torch.set_num_threads(threads)
