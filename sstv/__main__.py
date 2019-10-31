"""Main entry point for command line program"""

import signal
from sys import exit

import sstv


def handle_sigint(signal, frame):
    print()
    sstv.common.log_message("Received interrupt signal, exiting.")
    exit(0)


def main():
    signal.signal(signal.SIGINT, handle_sigint)
    with sstv.SSTVCommand() as prog:
        prog.start()


if __name__ == "__main__":
    main()
