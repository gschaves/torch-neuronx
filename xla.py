# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================

# TODO: import statements in this file are scattered everywhere temporarily.
# It is to make `from torch_neuronx import xla` work. When we have erased this
# pattern from all tests we can restore the import statements to a sane form.
import os
import logging
import libneuronxla
import subprocess

logging.basicConfig(format="%(name)s: %(message)s", level=logging.WARNING)
logger = logging.getLogger("torch_neuron")


def init():
    libneuronxla.configure_environment()
    configure_pjrt_environment()
    enable_torch_autocast_cuda_path()
    if os.environ.get("NEURON_INTERNAL_USE_VANILLA_TORCH_XLA", None) != "1":
        enable_neuron_custom_calls()
    if os.getenv("NEURON_USE_EAGER_DEBUG_MODE"):
        # XLA_USE_EAGER_DEBUG_MODE is torch-xla env variable for enabling
        # eager debug mode
        os.environ["XLA_USE_EAGER_DEBUG_MODE"] = "1"
        logger.warning(
            (
                "Eager debug mode is enabled. In this mode all operations would"
                " be executed eagerly. This will result in high execution times."
            )
        )


def enable_torch_autocast_cuda_path():
    import torch

    torch.cuda.is_bf16_supported = lambda: True


def enable_neuron_custom_calls(pattern=None):
    """
    Enable Neuron-specific HLO custom-call lowering.
    """
    # The reason to import hint/ops is that decorators @override/@lazy_override
    # must be executed before the enabler runs
    from torch_neuronx.xla_impl import base, hint, ops

    base.AwsNeuronCustomLoweringType.enable(pattern)
    import torch.optim.adamw as torch_adamw
    from torch_neuronx.optim.adamw import _single_tensor_adamw_

    torch_adamw._single_tensor_adamw = _single_tensor_adamw_


def configure_pjrt_environment():
    """
    Setting all necessary PJRT default environment variables.
    """
    # Path to load custom PjRT plugin
    if os.environ.get("PJRT_DEVICE") == "NEURON":
        os.environ["NEURON_LIBRARY_PATH"] = (
            subprocess.run(["libneuronpjrt-path"], stdout=subprocess.PIPE)
            .stdout.decode("utf-8")
            .strip()
        )

    # Set env variables if we don't use GSPMD and not using XRT
    if (
        os.environ.get("XLA_USE_SPMD") != "1"
        and os.environ.get("PJRT_DEVICE") == "NEURON"
    ):
        # Env variables that only need to be set once
        if "NEURON_PJRT_PROCESSES_NUM_DEVICES" not in os.environ:
            if "WORLD_SIZE" not in os.environ:
                logger.warning(
                    "WORLD_SIZE environment variable not set, defaulting to 1."
                )
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            processes_num_devices = ""
            for i in range(world_size):
                if i == (world_size - 1):
                    processes_num_devices += "1"
                else:
                    processes_num_devices += "1,"
            os.environ["NEURON_PJRT_PROCESSES_NUM_DEVICES"] = processes_num_devices
            if "LOCAL_WORLD_SIZE" not in os.environ:
                logger.warning(
                    "LOCAL_WORLD_SIZE environment variable not set, defaulting to 1."
                )
            os.environ["PJRT_LOCAL_PROCESS_COUNT"] = os.environ.get(
                "LOCAL_WORLD_SIZE", "1"
            )

        # Env variables that need to be set once per process
        os.environ["NEURON_RT_NUM_CORES"] = "1"
        if "RANK" not in os.environ:
            logger.warning("RANK environment variable is not set, defaulting to 0.")
        os.environ["NEURON_PJRT_PROCESS_INDEX"] = os.environ.get("RANK", "0")
        if "MASTER_ADDR" not in os.environ:
            logger.warning(
                "MASTER_ADDR environment variable is not set, defaulting to localhost."
            )
        os.environ["NEURON_RT_ROOT_COMM_ID"] = "{}:{}".format(
            os.environ.get("MASTER_ADDR", "localhost"), "62182"
        )

    # TODO: Add GSPMD defaults once ready
