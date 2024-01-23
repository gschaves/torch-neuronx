import os
import torch.distributed as dist

_ORIG_FUNC = dist._verify_params_across_processes


def _patched_verify_params_across_processes(process_group, tensors, logger=None):
    global _ORIG_FUNC
    """
    During neuron_parallel_compile we do a fake execution
    which could cause exchange of information between ranks during DDP.
    This patch is to work around the verify parameter shapes across processes
    during precompilation.
    """
    if os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None):
        return
    else:
        _ORIG_FUNC(process_group, tensors, logger)


dist._verify_params_across_processes = _patched_verify_params_across_processes
