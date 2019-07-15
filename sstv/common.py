"""Shared methods"""

from os import get_terminal_size
from sys import stderr, stdout, platform


def log_message(message="", show=True, err=False, recur=False, prefix=True):
    """Simple print wrapper"""

    if not show:
        return
    out = stdout
    if err:
        out = stderr
    end = '\n'
    if recur:
        end = '\r'
        cols = get_terminal_size().columns
        if cols < len(message):
            message = message[:cols]
    if prefix:
        message = ' '.join(["[SSTV]", message])

    print(message, file=out, end=end)


def progress_bar(progress, complete, message="", show=True):
    """Simple loading bar"""

    if not show:
        return

    message = ' '.join(["[SSTV]", message])
    cols = get_terminal_size().columns
    percent_on = True
    level = progress / complete
    bar_size = min(cols - len(message) - 10, 100)
    bar = ""

    if bar_size > 5:
        fill_size = round(bar_size * level)
        bar = "[{}]".format(''.join(['#' * fill_size,
                                     '.' * (bar_size - fill_size)]))
    elif bar_size < -3:
        percent_on = False

    percent = ""
    if percent_on:
        percent = "{:4.0f}%".format(level * 100)

    if platform == "win32":
        message = '\r' + message

    align = cols - len(message) - len(percent)
    not_end = not progress == complete
    log_message("{}{:>{width}}{}".format(message, bar, percent, width=align),
                recur=not_end, prefix=False)
