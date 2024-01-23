# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# ==============================================================================


"""
This script is for starting the xrt_server. It also polls the PID and
checks if it exist. It would kill the server, when the process whose
PID it was tracking dies.
NOTE: This script should be used only by xrt_init.py and not anyone else.
"""
import os
import argparse
import psutil
import time
import signal
import multiprocessing
import torch_xla


def _polling(pids_to_track):
    def is_pid_alive(pid):
        # The idea behind this is: if the process doesn't exit,
        # getting a process status should throw an error.
        # If the process exist, then we check if it hasn't gone
        # into zombie state. This can happen when we run torchrun
        # from neuron_parallel_compile.
        try:
            return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        except Exception:
            return False

    while 1:
        for pid in pids_to_track:
            if not is_pid_alive(pid):
                return
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True)
    parser.add_argument("--pid_to_track", default=None)
    args = parser.parse_args()
    main_pid = int(args.pid_to_track)
    children = psutil.Process(main_pid).children(recursive=True)

    server_process = multiprocessing.Process(
        target=torch_xla._XLAC._run_xrt_local_service, args=(int(args.port),)
    )
    server_process.start()

    polling_process = multiprocessing.Process(
        target=_polling, args=([int(args.pid_to_track), server_process.pid],)
    )

    polling_process.start()
    polling_process.join()

    os.kill(server_process.pid, signal.SIGKILL)
    cur_pid = os.getpid()
    for child in children:
        try:
            if child.pid != cur_pid:
                child.terminate()
        except Exception:
            continue
    try:
        os.kill(int(main_pid), signal.SIGTERM)
    except Exception:
        pass
