# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================
import copy
from distutils import version
import re
import logging
import warnings
import math
from concurrent.futures import ThreadPoolExecutor

import torch
from torch._utils import ExceptionWrapper

import torch_neuronx


log = logging.getLogger("Neuron")


def device_count():
    r"""
    Returns the number of avaiable NeuronCores.
    """
    runtime = torch.classes.neuron.Runtime()
    nc_count = runtime.get_nc_count()
    if nc_count == -1:
        raise RuntimeError(
            "Neuron runtime cannot be initialized; cannot determine the number of available NeuronCores"
        )
    return nc_count


def _get_all_neuroncore_indices():
    r"""
    Returns the indices of available NeuronCores.

    Calls device_count() and returns a list of integers for each NeuronCore
    """
    return list(range(device_count()))


def _get_neuroncore_index(neuroncore):
    r"""
    Gets the NeuronCore index from NeuronCore specification, which can be a `nc:#` or integer.

    Returns an integer value to use to load the NeuronCore
    """
    error_msg = "Expected `nc:#` or an integer, but got: {}".format(neuroncore)
    nc_idx = None

    if isinstance(neuroncore, str):
        try:
            nc_idx = int(re.findall(r"nc:(\d+)", neuroncore)[0])
        except Exception:
            raise ValueError(error_msg)
    elif isinstance(neuroncore, int):
        nc_idx = neuroncore

    if nc_idx is None:
        raise ValueError(error_msg)

    return nc_idx


def _is_namedtuple(obj):
    """
    Check if type was created from collections.namedtuple or a typing.NamedTuple.
    """
    return (
        isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")
    )


def _set_dynamic_batching(dim, module, set_dynamic_batching):
    if not set_dynamic_batching:
        log.info("Dynamic batching is disabled for torch_neuronx.DataParallel")
        torch_neuronx.dynamic_batch(module, False)
        return module
    elif dim != 0:
        log.info(
            "Dynamic batching is disabled for torch_neuronx.DataParallel because dim != 0"
        )
        torch_neuronx.dynamic_batch(module, False)
        return module
    else:
        # Enable dyamic batching for all input and output tensors
        torch_neuronx.dynamic_batch(module, True)
        return None


def _get_split_size(split_size, num_loaded_modules, dim, input):
    """
    Determine the input split size. If `split_size` is not given:
        `split_size` will be calculated such that it optimizes for
        full utilization of NeuronCores available.
        Remainders are kept track of, in cases where the number of
        NeuronCores `num_loaded_devices` is greater than
        batch size `input.shape[dim]`, and are distributed across devices
        as evenly as possible (handled in `scatter_map()`).

        For example,
        If num_loaded_modules = 4 and input.shape[dim] = 8: `split_size = 2`, `remainder = 0`
        If num_loaded_modules = 4 and input.shape[dim] = 5: `split_size = 1`, `remainder = 1`
        If num_loaded_modules = 4 and input.shape[dim] = 1: `split_size = 0`, `remainder = 1`
        If num_loaded_modules = 3 and input.shape[dim] = 8: `split_size = 2`, `remainder = 2`
        If num_loaded_modules = 3 and input.shape[dim] = 5: `split_size = 1`, `remainder = 2`
    """
    if split_size != -1:
        return split_size, 0
    else:
        num_batches = input.shape[dim]
        split_size, remainder = divmod(num_batches, num_loaded_modules)

        return split_size, remainder


class DataParallel(torch.nn.Module):
    r"""Implements data parallelism on torch_neuronx models.

    Applies data parallelism by duplicating the torch_neuronx module(s) generated from
    torch_neuronx.trace() on available NeuronCores and distributing data across the
    different cores for parallelized inference.

    By default, `torch_neuronx.DataParallel` will use all available NeuronCores
    in the current process for parallelism. `torch_neuronx.DataParallel` will
    apply parallelism on `dim=0` if `dim` is not specified.

    `torch_neuronx.DataParallel` automatically enables dynamic batching on
    eligible models.

    Args:
        module (`torch.jit.ScriptModule` or `ModelWrapper`): torch_neuronx module(s) to be
            parallelized, each generated from a torch_neuronx.trace() call.
            If a single torch_neuronx module is to be parallelized, provide it as a `torch.jit.ScriptModule`;
            If multiple torch_neuronx modules need to be parallelized, define a ModelWrapper class as follows:
            ```
            class ModelWrapper(torch.nn.Module):
            # ModelWrapper contains multiple neuron models (and multiple NEFFs)
            def __init__(self, model_1_neuron, model_2_neuron, model_3_neuron):
                # Each of model_1_neuron, model_2_neuron,
                # and model_3_neuron is generated from a call
                # to torch_neuronx.trace()

                super().__init__()
                self.model_1_neuron = model_1_neuron
                self.model_2_neuron = model_2_neuron
                self.model_3_neuron = model_3_neuron

            def forward(self, x):
                x = self.model_1_neuron(x)
                x = self.model_2_neuron(x)
                x = self.model_3_neuron(x)
                return x
            ```
            And pass the ModelWrapper as `module` to this DataParallel initializer.
            Alternatively, you can also use torch_neuronx.trace() to trace
            this ModelWrapper and pass the single, generated
            torchscript module as `module` device_ids
            (:obj:`list` of :obj:`int` or `nc:#`, optional): NeuronCores
            to use for parallelization (default: all NeuronCores).

        dim (:obj:`int`, optional): Dimension of input tensor that will be
            scattered across NeuronCores.

        set_dynamic_batching (:obj:`bool`, optional): Whether to enable
        dynamic batching for this DataParallel object (default: True).
        If dynamic batching is enabled, it can later be disabled via a
        call to self.disable_dynamic_batching()

    Attributes:
        num_workers (:obj:`int`, optional): Number of worker threads used for
            multithreaded inferece (default: 2 * number of NeuronCores).
        split_size (:obj:`int`, optional): Size of the input chunks
            (default: `split_size,remainder = divmod(input.shape[dim], number of NeuronCores)`)
            where the remainder is allocated to devices to make full use of NeuronCores.
        disable_dynamic_batching: Function that will disable automatic dynamic
            batching on the DataParallel module. Call as follows:
            >>> model_parallel = torch_neuronx.DataParallel(model_neuron)
            >>> model_parallel.disable_dynamic_batching()

    Examples:
        Default usage, in which all available NeuronCores are used:
        >>> model_parallel = torch_neuronx.DataParallel(model_neuron)
        >>> output = model_parallel(batched_input)

        Integer NeuronCore placement:
        >>> model_parallel = torch_neuronx.DataParallel(model_neuron, device_ids=[0, 1, 2])
        >>> output = model_parallel(batched_input)

        `nc:#` NeuronCore placement:
        >>> model_parallel = torch_neuronx.DataParallel(model_neuron, device_ids=['nc:0', 'nc:1', 'nc:2'])
        >>> output = model_parallel(batched_input)
    """

    def __init__(self, module, device_ids=None, dim=0, set_dynamic_batching=True):
        super().__init__()

        # get number of avaiable NeuronCores if device_ids not specified
        if device_ids is None:
            device_ids = _get_all_neuroncore_indices()

        self.module = module
        self.dim = dim

        self.set_dynamic_batching = set_dynamic_batching
        self.dynamic_batching_failed = False

        # Try to enable dynamic batching
        _set_dynamic_batching(self.dim, self.module, self.set_dynamic_batching)

        self.device_ids = [_get_neuroncore_index(device_id) for device_id in device_ids]

        self.num_workers = 0

        # Load the torch_neuronx module(s) on the NeuronCores
        self.loaded_modules = self._load_modules(self.module)

        self.split_size = -1

    def forward(self, *inputs):
        # Check whether dynamic batching needs to be disabled
        if self.dynamic_batching_failed and self.set_dynamic_batching:
            raise RuntimeError(
                "Automatic dynamic batching must be disabled before running inference. "
                "Please disable dynamic batching by calling `disable_dynamic_batching() "
                "on your DataParallel module."
            )

        if not self.device_ids:
            return self.module(*inputs)

        inputs = scatter_inputs(inputs, self.device_ids, self.split_size, self.dim)
        if not inputs:
            inputs = ((),)

        # Check if the NeuronCores are being underutilized
        if len(inputs) % len(self.device_ids) != 0:
            warnings.warn(
                "The NeuronCores are not being fully utilized because "
                "`inputs.shape[dim]` is not divisible by the number of NeuronCores given "
                "in `device_ids`. In order to get optimal performance, please try to "
                "ensure that the shape your inputs at `dim` is divisible by the number of NeuronCores "
                "that DataParallel is using, such that `input.shape[dim] % len(device_ids) == 0).`"
            )

        outputs = parallel_apply(
            modules=self.loaded_modules, inputs=inputs, num_workers=self.num_workers
        )
        return gather(outputs, self.dim)

    def _load_modules(self, module):
        try:
            is_device_ids_consecutive = True
            self.device_ids.sort()
            for i in range(len(self.device_ids) - 1):
                if self.device_ids[i + 1] != self.device_ids[i] + 1:
                    is_device_ids_consecutive = False
                    break

            loaded_modules = [module]
            if is_device_ids_consecutive:
                # If device_ids is consecutive, load onto all cores in one nrt_load call and avoid doing any deepcopy.
                torch_neuronx.experimental.placement.set_neuron_cores(
                    module, self.device_ids[0], len(self.device_ids)
                )
            else:
                # If device_ids is non-consecutive, perform deepcopy's and load onto each core independently.
                for i in range(len(self.device_ids) - 1):
                    loaded_modules.append(copy.deepcopy(module))
                for i, nc_index in enumerate(self.device_ids):
                    torch_neuronx.experimental.placement.set_neuron_cores(
                        loaded_modules[i], nc_index, 1
                    )

        except ValueError as err:
            self.dynamic_batching_failed = True
            log.warning(f"Automatic dynamic batching failed due to {err}.")
            log.warning(
                "Please disable dynamic batching by calling `disable_dynamic_batching()` "
                "on your DataParallel module."
            )
        self.num_workers = 2 * len(loaded_modules)
        return loaded_modules

    def disable_dynamic_batching(self):
        self.set_dynamic_batching = False
        module = _set_dynamic_batching(self.dim, self.module, self.set_dynamic_batching)
        # Reload the modules with dynamic batching disabled
        self.loaded_modules = self._load_modules(module)


def scatter(inputs, device_ids, split_size_, dim=0):
    r"""
    Breaks the `inputs` into a tuple of chunks of the input on `dim`.
    Duplicates references to objects in `inputs` that are not tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            num_devices = len(device_ids)
            num_batches = obj.shape[dim]
            split_size, remainder = _get_split_size(split_size_, num_devices, dim, obj)

            # optimal use of cores vs. use of user-defined split_size
            num_jobs = num_devices
            if split_size_ != -1:
                num_jobs = math.ceil(num_batches / split_size_)
            if split_size == 0:  # to prevent filling of jobs with no data
                num_jobs = remainder

            if dim != 0:
                return tuple(
                    torch.index_select(
                        obj,
                        dim=dim,
                        index=torch.linspace(
                            start=i * split_size + min(i, remainder),
                            end=(i + 1) * split_size + min(i + 1, remainder) - 1,
                            steps=split_size,
                            dtype=torch.int,
                        ),
                    )
                    for i in range(num_jobs)
                )
            else:
                return tuple(
                    obj[
                        i * split_size
                        + min(i, remainder):(i + 1) * split_size
                        + min(i + 1, remainder)
                    ]
                    for i in range(num_jobs)
                )

        if _is_namedtuple(obj):
            return [type(obj)(*args) for args in zip(*map(scatter_map, obj))]
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return [list(i) for i in zip(*map(scatter_map, obj))]
        if isinstance(obj, dict) and len(obj) > 0:
            return [
                type(obj)(zip(obj.keys(), i))
                for i in zip(*map(scatter_map, obj.values()))
            ]
        return [obj for _ in range(split_size)]

    try:
        res = scatter_map(inputs)
    finally:
        scatter_map = None
    return res


def scatter_inputs(inputs, device_ids, split_size, dim=0):
    inputs = scatter(inputs, device_ids, split_size, dim)
    inputs = tuple(inputs)
    return inputs


def parallel_apply(modules, inputs, num_workers):
    r"""
    Runs inference in parallel on the `inputs` using the loaded torch_neuronx
    `modules`. A `ThreadPoolExecutor` is used to parallelize the inference
    calls using `num_workers` threads.

    Args:
        module (`torch.jit.ScriptModule`): loaded torch_neuronx module(s) to be
            parallelized.
        inputs (`torch.tensor`): inputs to the `modules`.
        num_workers (:obj:`int`): number of ThreadPoolExecuter workers.
    """
    results = {}

    def _worker(i, module, input):
        try:
            if not isinstance(input, (list, tuple)):
                input = (input,)
            output = module(*input)
            results[i] = output
        except Exception:
            results[i] = ExceptionWrapper(where=f"on neuroncore {i}")

    with ThreadPoolExecutor(num_workers) as pool:
        for i, input in enumerate(inputs):
            module = modules[i % len(modules)]
            pool.submit(_worker, i, module, input)

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs


def gather(outputs, dim=0):
    r"""
    Gathers the tensors in `outputs` from different NeuronCores. Returns the
    combined outputs in the format of each individual NeuronCore output.
    """

    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            if all(t.dim() == 0 for t in outputs) and dim == 0:
                result = torch.cat(tuple(t.view(1) for t in outputs), 0)
                warnings.warn(
                    "The model returns scalar outputs, which cannot be gathered. "
                    "torch_neuronx.DataParallel will instead try to unsqueeze each scalar and "
                    "return a vector."
                )
                return result
            if dim != 0:
                return torch.cat(outputs, dim)
            else:
                # tensor slicing only written for dim=0
                output_batch_size = sum(
                    [outputs[i].shape[0] for i in range(len(outputs))]
                )
                res_shape = (
                    list(out.shape[:dim])
                    + [output_batch_size]
                    + list(out.shape[(dim + 1):])
                )
                result = torch.empty(res_shape, dtype=out.dtype)
                batch_idx = 0
                for output in outputs:
                    batch_size = output.shape[0]
                    result[batch_idx:(batch_idx + batch_size)] = output
                    batch_idx += batch_size
                return result
        if out is None:
            return None
        if isinstance(out, dict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError("All dicts must have the same number of keys")
            return type(out)(((k, gather_map([d[k] for d in outputs])) for k in out))
        if _is_namedtuple(out):
            return type(out)._make(map(gather_map, zip(*outputs)))
        return type(out)(map(gather_map, zip(*outputs)))

    try:
        res = gather_map(outputs)
    finally:
        gather_map = None
    return res
