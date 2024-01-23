# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================
from ._version import __version__
from . import xla
from . import distributed
from . import testing
from . import contexts
from . import experimental
from .xla_impl.trace import trace, move_trace_to_device
from .xla_impl.options import Options
from .xla_impl.torchscript import dynamic_batch, async_load, lazy_load
from .xla_impl.analyze import analyze
from .xla_impl.partitioner import PartitionerConfig, partition
from .xla_impl.data_parallel import DataParallel
import os
import glob


def _load_libtorchneuron():
    from . import libtorchneuron

    libtorchneuron.load()


def _add_lib_preload(lib_type):
    lib_find = False
    lib_set = False
    for item in os.getenv("LD_PRELOAD", "").split(":"):
        if item.endswith(f"lib{lib_type}.so"):
            lib_set = True
            break
    if not lib_set:
        lib_path = os.path.dirname(__file__) + "/lib"
        library_file = os.path.join(lib_path, f"lib{lib_type}.so")
        matches = glob.glob(library_file)
        if len(matches) > 0:
            ld_preloads = [f"{matches[0]}", os.getenv("LD_PRELOAD", "")]
            os.environ["LD_PRELOAD"] = os.pathsep.join(
                [p.strip(os.pathsep) for p in ld_preloads if p]
            )
            lib_find = True
    return lib_set or lib_find


_load_libtorchneuron()
# _add_lib_preload("jemalloc")
