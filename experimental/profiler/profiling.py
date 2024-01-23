# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================
from pathlib import Path
import shutil
import threading
import time
import os
import logging
import glob
import subprocess

import torch_xla.debug.profiler as xp
from torch_xla import _XLAC

# For logging setup
# import logging

logger = logging.getLogger("torch_neuron")
logger.setLevel(logging.INFO)

from .translate_to_tb_neuron import translate_xplane  # noqa: E501,E402


# Customer's will see use this as a scope like:
#
# with torch.neuron.experimental.profiler.profile():
#    # Neuron inference or training code
#
# NOTE: 'XLA_IR_DEBUG' and 'XLA_HLO_DEBUG' are currently automatically
#        enabled and then set back to prior values after run
class profile(object):
    def __init__(
        self,
        *,
        port=9012,
        ms_duration=60000,
        neuron_tensorboard_plugin_dir="logs/plugins/neuron",
        profile_type="operator",
        auto_start=True,
        delete_working=True,
        traced_only=False,
    ):
        """Profiler - experimental API is subject to change!
        'XLA_IR_DEBUG' and 'XLA_HLO_DEBUG' must be turned on
        at the time the model is first executed (i.e. compiled)
        to get symbols for profiling

        Args:

            port (int): Port to run the profiling GRPC server on

            ms_duration (int): Time to run profiler for
            (currently this must be set)

            neuron_tensorboard_plugin_dir (str): The directory the neuron
            tensorboard plugin will file write to
            (NB: Assumes that the tensorboard logdir="log/")

            delete_working (bool): If set to False turns off
            the deletion of temporary files (default True)
        """
        self.port_ = port
        self.ms_duration_ = ms_duration
        self.profile_type = profile_type
        self.delete_working = delete_working
        self.auto_start = auto_start
        self.traced_only = traced_only
        self.profiler_raw_dir = None
        self.profiler_temp_log_dir = "temp_profiler_logs"
        self.target_dir = neuron_tensorboard_plugin_dir
        self.trace_thread_ = None

    def _cleanup(self):
        # Cleanup temporary directory with xplane.pb file,
        # since we have extracted the data to the
        # neuron tensorboard plugin now
        if self.delete_working:
            logger.debug("Cleaning up temporary profiling files")
            shutil.rmtree(self.profiler_temp_log_dir, ignore_errors=True)

            # Delete if it exists
            shutil.rmtree(self.profiler_raw_dir, ignore_errors=True)
        else:
            logger.warning(
                "Temporary profiling files *not* deleted due to profile configuration"
            )

        # Turn off env variables if they were not on to start
        if self.ir_debug_off and "XLA_IR_DEBUG" in os.environ:
            del os.environ["XLA_IR_DEBUG"]

        if self.hlo_debug_off and "XLA_HLO_DEBUG" in os.environ:
            del os.environ["XLA_HLO_DEBUG"]

        # From 1.12 we need to force reinitialization to consume
        # the XLA_IR_DEBUG env variable
        _XLAC._map_xla_env_vars_to_lazy()

        self.profiler_raw_dir = None
        self.trace_thread_ = None

    def __enter__(self):
        if "NEURON_PROFILE" not in os.environ:
            os.environ["NEURON_PROFILE"] = "profile/"

        self.profiler_raw_dir = os.environ["NEURON_PROFILE"]
        os.makedirs(self.profiler_raw_dir, exist_ok=True)

        self.ir_debug_off = False
        if "XLA_IR_DEBUG" not in os.environ:
            self.ir_debug_off = True

        self.hlo_debug_off = False
        if "XLA_HLO_DEBUG" not in os.environ:
            self.hlo_debug_off = True

        if self.profile_type == "trace":
            os.environ["NEURON_PROFILE_TYPE"] = "high_level"
        elif self.profile_type == "operator":
            os.environ["NEURON_PROFILE_TYPE"] = "low_level"
        else:
            os.environ["NEURON_PROFILE_TYPE"] = "low_level"

        if not self.delete_working:
            os.environ["NEURON_PROFILE_NO_DELETE"] = "1"
        else:
            os.environ["NEURON_PROFILE_NO_DELETE"] = "0"

        if self.ir_debug_off:
            os.environ["XLA_IR_DEBUG"] = "1"

        if self.hlo_debug_off:
            os.environ["XLA_HLO_DEBUG"] = "1"

        # From 1.12 we need to force reinitialization to consume the
        # XLA_IR_DEBUG env variable
        _XLAC._map_xla_env_vars_to_lazy()

        if self.auto_start:
            self.start()

        return self

    def start(self):
        # This function connects the performs a tensorflow style
        # profile - connecting to the server created at construction
        # - do this in a 'thread' since it will block on response
        def profile_thread_function(port, ms_duration):
            xp.trace(
                f"127.0.0.1:{self.port_}",
                logdir=self.profiler_temp_log_dir,
                duration_ms=ms_duration,
            )

        # In a libtpu enabled torch-xla build this will result
        # in profiling calls on libtpu once a client invokes trace
        if not self.traced_only:
            logger.debug("Start profile server")
            self.server_ = xp.start_server(self.port_)

            if self.trace_thread_ is None:
                self.trace_thread_ = threading.Thread(
                    target=profile_thread_function,
                    kwargs={"port": self.port_, "ms_duration": self.ms_duration_},
                )
                self.start_time = time.time()
                self.trace_thread_.start()
            else:
                logger.warning(f"Profiling is already running {self.trace_thread_}")
        else:
            self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            # Join the trace thread as we exit the scope - this
            # ensures that any xplane.pb files have been output
            if self.trace_thread_ is not None:
                logger.info("Waiting for XLA profile completion ...")
                self.trace_thread_.join()
                self.trace_thread = None

            ts = time.localtime(self.start_time)
            target_dir = f"{self.target_dir}/{ts.tm_year:0>2d}_{ts.tm_mon:0>2d}_{ts.tm_mday:0>2d}_{ts.tm_hour:0>2d}_{ts.tm_min:0>2d}_{ts.tm_min:0>2d}_{ts.tm_sec:0>2d}"  # noqa: E501
            os.makedirs(target_dir, exist_ok=True)

            if not self.traced_only:
                logger.debug("Decoding xplane files from profiler")
                trace_file, json_files = translate_xplane(
                    self.profiler_temp_log_dir, self.profiler_temp_log_dir
                )

                logger.debug(f"Trace file: {trace_file}")
                logger.debug(f"Translated JSON files: {json_files}")
            else:
                neffs = dict()
                ntffs = dict()
                for file in os.listdir(self.profiler_raw_dir):
                    if file.endswith(".neff"):
                        fname = os.path.basename(file).split(".")
                        neffs[os.path.basename(fname[0])] = file
                    if file.endswith(".ntff"):
                        if "-" in str(file):
                            file_parts = str(os.path.basename(file)).split("-")
                            ntffs[file_parts[0]] = file

                # Iterate over the NEFF and NTFF combinations and compilet to target dir
                for key, val in ntffs.items():
                    command = [
                        "/opt/aws/neuron/bin/neuron-profile",
                        "analyze",
                        "-s",
                        f"{self.profiler_raw_dir}/{val}",
                        "-n",
                        f"{self.profiler_raw_dir}/{neffs[key]}",
                        "-d",
                        f"{self.profiler_raw_dir}/{val.split('.')[0]}",
                    ]
                    logger.info(f"Profile command = '{command}'")
                    subprocess.run(command)

                    json_files = glob.glob(
                        f"{self.profiler_raw_dir}/**/*.json", recursive=True
                    )

            logger.debug("Output processed JSON profiles")
            if self.profile_type == "trace":
                # One overall trace per context
                shutil.copy2(trace_file, f"{target_dir}/neuron_trace.json")
            else:
                for i, file in enumerate(json_files):
                    path_segments = file.split("/")
                    filename = path_segments[-1]
                    parent_path = path_segments[-2]
                    per_file_dir = f"{target_dir}_{parent_path}"
                    os.makedirs(per_file_dir, exist_ok=True)

                    if filename == "neuron_op_timeline_split.json":
                        # Add code here for manual post-processing
                        # this is intentionally preserved for this purpose
                        # It will be deleted with temporary files
                        pass
                    else:
                        shutil.copy2(file, f"{per_file_dir}/{filename}")

            self._cleanup()
        finally:
            # Make sure we cleanup even on failure
            self._cleanup()
