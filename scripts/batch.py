#! /usr/bin/env python3
import argparse
import itertools
import multiprocessing
import os
import queue
import re
import shutil
import subprocess
import sys
import threading


def add_format_cc_argument(parser):
    parser.add_argument("file", type=str, nargs="+", help="Specify dir or file to format")
    parser.add_argument("--binary", type=str, default="clang-format", help="path to binary. default: clang-format")
    parser.add_argument("--flags", default="", help="Specify flags pass to binary")
    parser.add_argument(
        "--regex",
        type=str,
        default=r"^.*$",
        help="Specify regex for filename",
    )
    parser.add_argument(
        "--regex-cc",
        dest="regex",
        action="store_const",
        const=r".*\.((((c|C)(c|pp|xx|\+\+)?$)|((h|H)h?(pp|xx|\+\+)?$))|(ino|pde|proto|cu))$",
        help="Specify cc regex for filename: "
        r".*\.((((c|C)(c|pp|xx|\+\+)?$)|((h|H)h?(pp|xx|\+\+)?$))|(ino|pde|proto|cu))$",
    )
    parser.add_argument(
        "--regex-cmake",
        dest="regex",
        action="store_const",
        const=r"(^(./)?CMakeLists.txt$)|(^(./)?((ccsrc)|(tests))/.*/CMakeLists.txt$)|(^(./)?cmake/[^(third_party)].*cmake$)",
        help="Specify cc regex for filename: "
        r"(^(./)?CMakeLists.txt$)|(^(./)?((ccsrc)|(tests))/.*/CMakeLists.txt$)|(^(./)?cmake/[^(third_party)].*cmake$)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Do not actually execute",
    )
    parser.add_argument("-j", type=int, default=0, help="number of format instances to be run in parallel.")


def find_binary(arg, name):
    """Get the path for a binary or exit"""
    if arg:
        if shutil.which(arg):
            return arg
        else:
            raise SystemExit("error: passed binary '{}' was not found or is not executable".format(arg))

    binary = shutil.which(name)
    if binary:
        return binary
    else:
        raise SystemExit("error: failed to find {} in $PATH".format(name))


def get_binary_invocation(f, binary, flags):
    """Gets a command line for binary."""
    start = [binary]
    start.extend(flags.split())
    start.append(f)
    return start


def run_binary(args, binary, queue, lock, failed_files):
    """Takes filenames out of queue and runs binary on them."""
    while True:
        name = queue.get()
        invocation = get_binary_invocation(name, binary, args.flags)
        if args.dry_run is False:
            proc = subprocess.Popen(invocation, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, err = proc.communicate()
            if proc.returncode != 0:
                if proc.returncode < 0:
                    msg = "%s: terminated by signal %d\n" % (name, -proc.returncode)
                    err += msg.encode("utf-8")
                failed_files.append(name)
            with lock:
                sys.stdout.write(" ".join(invocation) + "\n" + output.decode("utf-8"))
                if len(err) > 0:
                    sys.stdout.flush()
                    sys.stderr.write(err.decode("utf-8"))
        else:
            with lock:
                print(" ".join(invocation), file=sys.stdout, flush=True)
        queue.task_done()


def main(argv):
    parser = argparse.ArgumentParser(description="Batch run command on files")
    add_format_cc_argument(parser)
    args = parser.parse_args(argv)

    binary = find_binary(args.binary, "clang-format")
    max_task = args.j
    if max_task == 0:
        max_task = multiprocessing.cpu_count()

    def file_or_dir(path):
        if os.path.isfile(path):
            yield path
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for name in files:
                    yield os.path.join(root, name)

    def walk_dir(paths):
        for path in paths:
            yield from file_or_dir(path)

    files = walk_dir(args.file)
    file_name_re = re.compile(args.regex)
    return_code = 0
    try:
        # Spin up a bunch of tidy-launching threads.
        task_queue = queue.Queue(max_task)
        # List of files with a non-zero return code.
        failed_files = []
        lock = threading.Lock()
        for _ in range(max_task):
            t = threading.Thread(target=run_binary, args=(args, binary, task_queue, lock, failed_files))
            t.daemon = True
            t.start()
        # Fill the queue with files.
        for name in files:
            if file_name_re.search(name):
                task_queue.put(name)

        # Wait for all threads to be done.
        task_queue.join()
        if len(failed_files):
            return_code = 1

    except KeyboardInterrupt:
        # This is a sad hack. Unfortunately subprocess goes
        # bonkers with ctrl-c and we start forking merrily.
        print("\nCtrl-C detected, goodbye.")
        os.kill(0, 9)
    sys.exit(return_code)


if __name__ == "__main__":
    main(sys.argv[1:])
