#!/usr/bin/env python

import sstv


def main():
    with sstv.SSTVCommand() as prog:
        prog.start()


if __name__ == "__main__":
    main()
