# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================
from torch_neuronx.pyhlo import xplane_pb2

import json
import os
import logging
from copy import copy
from datetime import datetime

from pathlib import Path

# For logging setup
# import logging

logger = logging.getLogger("torch_neuron")


# Find XSpace protobuf files under the target path and output under the
# target path (files for the tensorboard neuron plugin)
def translate_xplane(xs_pb_path, tb_neuron_path):
    output_neuron_op_json_files = []

    # Open the high level path irrespective since we may want to merge
    # sections of the xplane.pb file both into the high level "trace"
    # view
    high_level_path = tb_neuron_path + "/neuron_trace.json"

    cpu_epoch_start_ns = None
    neuron_epoch_start_ns = None

    with open(high_level_path, "w") as trace_f:
        trace_root = dict()
        cpu_host_events = set()

        for path in Path(xs_pb_path).rglob("*.xplane.pb"):
            logger.debug(f"translate_xplane: XPlane file: {path}")

            xspace = xplane_pb2.XSpace()

            with open(path, "rb") as f:
                xspace.ParseFromString(f.read())

            for plane in xspace.planes:
                logger.info(f"translate_xplane: Processing plane: '{plane.name}'")

                # Template for trace
                # NOTE: the process id = the plane name
                default_trace_entry = {
                    "args": None,
                    "cat": "",
                    "dur": 0,
                    "name": "",
                    "ph": "X",
                    "pid": plane.name,
                    "tid": "",
                    "ts": 0,
                    "tts": 0,
                }

                plane_name_array = plane.name.split("/")

                # TODO: Eliminate magic strings by loading from a shared object
                # NOTE: To allow merging (in tensorflow code) this would need to be
                #       "/host:TPU-runtime" here and in libtpu
                if plane.name.startswith("/host:Neuron-runtime"):
                    assert (
                        len(plane_name_array) >= 7
                    ), f"Plane path length for plane name {plane.name}, was unexpectedly short"  # noqa: E501

                    parent_dir = plane_name_array[-7]
                    source_file_name = plane_name_array[-1]

                    os.makedirs(f"{tb_neuron_path}/{parent_dir}", exist_ok=True)

                    if (
                        source_file_name == "neuron_framework_op.json"
                        or source_file_name == "neuron_hlo_op.json"
                    ):
                        low_level_path = (
                            f"{tb_neuron_path}/{parent_dir}/{source_file_name}"
                        )

                        logger.info(
                            f"translate_xplane: Writing plane: '{plane.name}' to '{low_level_path}'"  # noqa: E501
                        )

                        output = dict()
                        output["ops"] = []
                        output["total_time"] = 0

                        for stat in plane.stats:
                            name = plane.stat_metadata[stat.metadata_id].name
                            logger.debug(f"Read stat name = {name}")

                            if name.startswith("op_"):
                                output["ops"].append(json.loads(stat.str_value))
                            elif name == "total_time":
                                output["total_time"] = stat.uint64_value

                        with open(low_level_path, "w") as output_f:
                            json.dump(output, output_f, indent=2)

                        output_neuron_op_json_files.append(low_level_path)

                    elif (
                        source_file_name == "neuron_op_timeline_split.json"
                        or source_file_name == "neuron_op_timeline.json"
                    ):
                        low_level_path = (
                            f"{tb_neuron_path}/{parent_dir}/{source_file_name}"
                        )

                        logger.info(
                            f"translate_xplane: Writing plane: '{plane.name}' to '{low_level_path}'"  # noqa: E501
                        )

                        with open(low_level_path, "w") as output_f:
                            # Create a top level dictionary
                            root = dict()

                            for stat in plane.stats:
                                stat_name = plane.stat_metadata[stat.metadata_id].name

                                if stat_name == "displayTimeUnit":
                                    stat_value = stat.str_value
                                elif stat_name == "neuronStartTimestampNs":
                                    stat_value = stat.uint64_value

                                root[stat_name] = stat_value

                            traceEvents = []

                            for line in plane.lines:
                                for event in line.events:
                                    name = plane.event_metadata[event.metadata_id].name
                                    ts = event.offset_ps  # noqa: F841
                                    dur = event.duration_ps  # noqa: F841

                                    for stat in event.stats:
                                        # Read in the stat JSON (record) - in
                                        # case mutation is needed here
                                        # for some records
                                        record = json.loads(stat.str_value)
                                        assert isinstance(
                                            record, dict
                                        ), "Record was not a JSON dict"

                                        traceEvents.append(record)

                            root["traceEvents"] = traceEvents

                            json.dump(root, output_f, indent=2)
                        output_neuron_op_json_files.append(low_level_path)

                    elif source_file_name == "neuron_trace.json":
                        for stat in plane.stats:
                            stat_name = plane.stat_metadata[stat.metadata_id].name

                            if stat_name == "displayTimeUnit":
                                stat_value = stat.str_value
                            elif stat_name == "neuronStartTimestampNs":
                                stat_value = stat.uint64_value
                                if neuron_epoch_start_ns is None or (
                                    stat_value is not None
                                    and stat_value < neuron_epoch_start_ns
                                ):
                                    neuron_epoch_start_ns = stat_value

                            trace_root[stat_name] = stat_value

                        traceEvents = []

                        for line in plane.lines:
                            for event in line.events:
                                name = plane.event_metadata[event.metadata_id].name
                                ts = event.offset_ps  # noqa: F841
                                dur = event.duration_ps  # noqa: F841

                                for stat in event.stats:
                                    # Read in the stat JSON (record)
                                    # in case mutation is needed here
                                    # for some records
                                    record = json.loads(stat.str_value)
                                    assert isinstance(
                                        record, dict
                                    ), "Record was not a JSON dict"

                                    traceEvents.append(record)

                        if "traceEvents" not in trace_root:
                            trace_root["traceEvents"] = []
                        trace_root["traceEvents"].extend(traceEvents)

                # This is data from the torch_xla runtime on
                # processes it runs.  Here we translate
                # the timeline to ms
                # TODO: Confirm timeline scaling is correct
                elif plane.name.startswith("/host:CPU"):
                    # The absolute time offsets for lines in this file are *wrong*
                    # Read the directory name time and add to abs time
                    file_name = str(path).split("/")[-2]
                    date_parts = file_name.split("_")
                    abs_offset = 0  # noqa: F841

                    logger.info(f"XLA decode - Read filename {file_name}")
                    logger.info(f"XLA decode - Read date parts {date_parts}")

                    if len(date_parts) >= 6:
                        start_date = datetime(
                            year=int(date_parts[0]),
                            month=int(date_parts[1]),
                            day=int(date_parts[2]),
                            hour=int(date_parts[3]),
                            minute=int(date_parts[4]),
                            second=int(date_parts[5]),
                        )
                        logger.info(
                            f"XLA decode - Read start date {start_date} from directory stamp"  # noqa: E501
                        )

                    line_count = 1
                    for line in plane.lines:
                        entry = copy(default_trace_entry)

                        # Thread ID = line name
                        entry["tid"] = f"{line.name}#{line_count}"
                        start_ns = line.timestamp_ns

                        if cpu_epoch_start_ns is None or start_ns < cpu_epoch_start_ns:
                            cpu_epoch_start_ns = start_ns

                        for event in line.events:
                            # Content for display (changed from template)
                            entry["name"] = plane.event_metadata[event.metadata_id].name
                            entry["ts"] = (start_ns / 1000) + (
                                event.offset_ps / 1000000
                            )
                            entry["dur"] = event.duration_ps / 1000000

                            e = [(k, v) for k, v in entry.items()]
                            e.sort()
                            cpu_host_events.add(tuple(e))

                        line_count += 1

        cpu_host_events = [dict(x) for x in cpu_host_events]

        if "traceEvents" not in trace_root:
            trace_root["traceEvents"] = []

        if cpu_epoch_start_ns is not None and neuron_epoch_start_ns is not None:
            logger.info(
                f"XLA high level trace start = {cpu_epoch_start_ns} ns = {datetime.utcfromtimestamp(cpu_epoch_start_ns/1e9)}"  # noqa: E501
            )
            logger.info(
                f"Neuron high level trace start = {neuron_epoch_start_ns} ns = {datetime.utcfromtimestamp(neuron_epoch_start_ns/1e9)}"  # noqa: E501
            )

            neuron_offset_us = (cpu_epoch_start_ns - neuron_epoch_start_ns) / 1000
            logger.info(f"Delta from neuron to XLA events = {neuron_offset_us} us")

            cpu_epoch_start_us = cpu_epoch_start_ns / 1000

            for c in cpu_host_events:
                c["ts"] -= cpu_epoch_start_us

            # Align the first 'TPUExecute' with the first 'NeuronDevice Execution'
            # TODO: Improve this is fragile
            for t in trace_root["traceEvents"]:
                t["ts"] -= neuron_offset_us

            trace_root["traceEvents"].extend(cpu_host_events)

            trace_root["traceEvents"].sort(key=lambda x: x["ts"])
            min_ts = trace_root["traceEvents"][0]["ts"]

            # Make output JSON match in UX timestamps (remove min offset)
            for t in trace_root["traceEvents"]:
                t["ts"] -= min_ts

            logger.debug("Adding CPU traceEvent:")
            for r in cpu_host_events:
                logger.debug(r)

            json.dump(trace_root, trace_f, indent=2)

    return high_level_path, output_neuron_op_json_files
