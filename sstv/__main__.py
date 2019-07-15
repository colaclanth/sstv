"""Main entry point for command line program"""

import sstv


def main():
    with sstv.SSTVCommand() as prog:
        prog.start()


if __name__ == "__main__":
    main()
