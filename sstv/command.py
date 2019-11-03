"""Parsing arguments and starting program from command line"""

import argparse
from sys import argv, exit

from PIL import Image
from soundfile import available_formats as available_audio_formats

from .common import log_message
from .decode import SSTVDecoder
from .spec import VIS_MAP


class SSTVCommand(object):
    """Main class to handle the command line features"""

    examples_of_use = """
examples:
  Decode local SSTV audio file named 'audio.ogg' to 'result.png':
    $ sstv -d audio.ogg

  Decode SSTV audio file in /tmp to './image.jpg':
    $ sstv -d /tmp/signal.wav -o ./image.jpg

  Start decoding SSTV signal at 50.5 seconds into the audio
    $ sstv -d audio.ogg -s 50.50"""

    def __init__(self, shell_args=None):
        """Handle command line arguments"""

        self._audio_file = None
        self._output_file = None

        if shell_args is None:
            self.args = self.parse_args(argv[1:])
        else:
            self.args = self.parse_args(shell_args)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        self.close()

    def __del__(self):
        self.close()

    def init_args(self):
        """Initialise argparse parser"""

        version = "sstv 0.1"

        parser = argparse.ArgumentParser(
            prog="sstv",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self.examples_of_use)

        parser.add_argument("-d", "--decode", type=argparse.FileType('rb'),
                            help="decode SSTV audio file", dest="audio_file")
        parser.add_argument("-o", "--output", type=str,
                            help="save output image to custom location",
                            default="result.png", dest="output_file")
        parser.add_argument("-s", "--skip", type=float,
                            help="time in seconds to start decoding signal at",
                            default=0.0, dest="skip")
        parser.add_argument("-V", "--version", action="version",
                            version=version)
        parser.add_argument("--list-modes", action="store_true",
                            dest="list_modes",
                            help="list supported SSTV modes")
        parser.add_argument("--list-audio-formats", action="store_true",
                            dest="list_audio_formats",
                            help="list supported audio file formats")
        parser.add_argument("--list-image-formats", action="store_true",
                            dest="list_image_formats",
                            help="list supported image file formats")
        return parser

    def parse_args(self, shell_args):
        """Parse command line arguments"""

        parser = self.init_args()
        args = parser.parse_args(shell_args)

        self._audio_file = args.audio_file
        self._output_file = args.output_file
        self._skip = args.skip

        if args.list_modes:
            self.list_supported_modes()
            exit(0)
        if args.list_audio_formats:
            self.list_supported_audio_formats()
            exit(0)
        if args.list_image_formats:
            self.list_supported_image_formats()
            exit(0)

        if self._audio_file is None:
            parser.print_help()
            exit(2)

        return args

    def start(self):
        """Start decoder"""

        with SSTVDecoder(self._audio_file) as sstv:
            img = sstv.decode(self._skip)
            if img is None:  # No SSTV signal found
                exit(2)

            try:
                img.save(self._output_file)
            except (KeyError, ValueError):
                log_message("Error saving file, saved to result.png instead",
                            err=True)
                img.save("result.png")

    def close(self):
        """Closes any input/output files if they exist"""

        if self._audio_file is not None and not self._audio_file.closed:
            self._audio_file.close()

    def list_supported_modes(self):
        modes = ', '.join([fmt.NAME for fmt in VIS_MAP.values()])
        print("Supported modes: {}".format(modes))

    def list_supported_audio_formats(self):
        audio_formats = ', '.join(available_audio_formats().keys())
        print("Supported audio formats: {}".format(audio_formats))

    def list_supported_image_formats(self):
        Image.init()
        image_formats = ', '.join(Image.SAVE.keys())
        print("Supported image formats: {}".format(image_formats))
