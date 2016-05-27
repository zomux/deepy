#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This script is based on the launch of platoon.


from __future__ import print_function
import os
import shlex
import argparse
import subprocess
import logging
logging.basicConfig(level=logging.INFO)


def parse_arguments():
    ap = argparse.ArgumentParser(description="Launch a multi-GPU expreiment")
    ap.add_argument('worker_path', help='Path of worker')
    ap.add_argument('gpu_list', nargs='+', type=str, help='The list of Theano GPU ids (Ex: gpu0, cuda1) the script will use. 1 GPU id = 1 worker launched.')
    ap.add_argument("--port", type=int, default=5567)
    ap.add_argument("--easgd_alpha", default="auto")
    ap.add_argument('-w', '--workers-args', required=False, help='The arguments that will be passed to your workers. (Ex: -w="learning_rate=0.1")')
    return ap.parse_args()


def launch_process(is_server, args, device, path=""):
    print("Starting {0} on {1} ...".format("server" if is_server else "worker", device), end=' ')

    env = dict(os.environ)
    env['THEANO_FLAGS'] = '{},device={}'.format(env.get('THEANO_FLAGS', ''), device)
    if is_server:
        command = ["python",  "-u",  "-m", "deepy.multigpu.server"]
    else:
        command = ["python", "-u", path]
    if not args is None:
        command += args
    process = subprocess.Popen(command, bufsize=0, env=env)
    print("Done")
    return process

if __name__ == '__main__':
    args = parse_arguments()

    process_map = {}

    easgd_alpha = args.easgd_alpha
    if easgd_alpha == "auto":
        easgd_alpha = 1.0 / len(args.gpu_list)

    controller_args_str = "--port {} --easgd_alpha {}".format(
        args.port,
        easgd_alpha
    )
    p = launch_process(True, shlex.split(controller_args_str), "cpu")
    process_map[p.pid] = ('scheduling server', p)

    for device in args.gpu_list:
        worker_process = launch_process(False, shlex.split(args.workers_args or ''), device, args.worker_path)
        process_map[worker_process.pid] = ("worker_{}".format(device),
                                           worker_process)

    print("\n### Waiting on experiment to finish ...")

    # Silly error handling but that will do for now.
    while process_map:
        pid, returncode = os.wait()
        if pid not in process_map:
            print("Recieved status for unknown process {}".format(pid))

        name, p = process_map[pid]
        del process_map[pid]
        print("{} terminated with return code: {}.".format(name, returncode))
        if returncode != 0:
            print("\nWARNING! An error has occurred.")
            while process_map:
                for name, p in list(process_map.values()):
                    try:
                        p.kill()
                    except OSError:
                        pass
                    if p.poll() is not None:
                        del process_map[p.pid]
