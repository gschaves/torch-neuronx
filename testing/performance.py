# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
"""
Performance
===========
Minimal methods to record model performance statistics (e.g. throughput/latency)
"""
import time
import math
import warnings
import pathlib
import concurrent.futures
from typing import Tuple, Dict, Any, Optional, List, Union

import torch

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def percentile(data: List[float], q: int = 50) -> float:
    """
    Compute the q-th percentile of a collection of measurements.

    Arguments:
        data: The collection of values to compute the percentile on.
        q: The percentile to compute. Expected to be between 0 and 100.

    Returns:
        A single percentile float.
    """
    index = (len(data) - 1) * (q / 100)
    lower = math.floor(index)
    upper = math.ceil(index)
    alpha = index - lower
    data = sorted(data)
    return (data[lower] * (1 - alpha)) + (data[upper] * alpha)


def infer_batch_size(example: Tuple[torch.Tensor, ...]) -> int:
    """
    Attempts to find the common batch size in a collection of tensors.

    This method assumes that the 0th dimension is the batch dimension. If 0th
    dimension is inconsistent, then the resulting batch size will default to a
    value of 1.

    Arguments:
        example: An example model input.

    Returns:
        The batch size of the example tensors
    """
    batch_size = 0
    for tensor in example:
        if batch_size == 0:
            batch_size = tensor.shape[0]
        else:
            if tensor.shape[0] != batch_size:
                warnings.warn("Unable to infer batch size. Using default value of 1.")
                return 1
    return batch_size


# -----------------------------------------------------------------------------
# Benchmarking
# -----------------------------------------------------------------------------


def benchmark(
    filename: Union[str, pathlib.Path],
    example: Tuple[torch.Tensor, ...],
    n_models: int = 2,
    n_threads: int = 2,
    batches_per_thread: int = 1000,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Record performance statistics for a serialized model and its input example.

    WARNING: This function does not guarantee that any number of NeuronCores is
             available. This will naively load `n_models` models.

    Arguments:
        filename: The serialized torchscript model to load for benchmarking.
        example: An example model input.
        n_models: The number of models to load.
        n_threads: The number of simultaneous threads to infer with.
        batches_per_thread: The number of example batches to run per thread.
        batch_size: Used to compute throughput. Inferred from dimension 0
            by default.

    Returns:
        A dictionary of performance statisics.
    """
    assert isinstance(
        example, Tuple
    ), f'Argument "example" must be a tuple type but found {type(example)}'

    # Load models
    models = [torch.jit.load(filename) for _ in range(n_models)]

    # Warmup
    for _ in range(8):
        for model in models:
            model(*example)

    latencies = []

    # Thread task
    def task(model):
        for _ in range(batches_per_thread):
            start = time.time()
            model(*example)
            finish = time.time()
            latencies.append((finish - start) * 1000)

    # Submit tasks
    begin = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as pool:
        for i in range(n_threads):
            pool.submit(task, models[i % len(models)])
    end = time.time()

    # Compute metrics
    boundaries = [0, 50, 90, 95, 99, 100]
    percentiles = {}
    for boundary in boundaries:
        name = f"latency_p{boundary}"
        percentiles[name] = percentile(latencies, boundary)
    duration = end - begin
    if batch_size is None:
        batch_size = infer_batch_size(example)
    inferences = len(latencies) * batch_size
    throughput = inferences / duration

    # Metrics
    metrics = {
        "filename": str(filename),
        "batch_size": batch_size,
        "batches": len(latencies),
        "inferences": inferences,
        "threads": n_threads,
        "models": n_models,
        "duration": duration,
        "throughput": throughput,
        **percentiles,
    }

    display(metrics)
    return metrics


def display(metrics: Dict[str, Any]) -> None:
    """
    Display the metrics produced by `benchmark` function.

    Args:
        metrics: A dictionary of performance statisics.
    """
    pad = max(map(len, metrics)) + 1
    for key, value in metrics.items():
        parts = key.split("_")
        parts = list(map(str.title, parts))
        title = " ".join(parts) + ":"

        if isinstance(value, float):
            value = f"{value:0.3f}"

        print(f"{title :<{pad}} {value}")
