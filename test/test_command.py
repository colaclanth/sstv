"""Test cases for the CLI code"""

import sys
import unittest
from io import StringIO

from sstv.command import SSTVCommand


class SSTVCommandTestCase(unittest.TestCase):
    """Test SSTVCommand class"""

    def setUp(self):
        """Capture standard input/error using StringIO instance"""
        sys.stdout = StringIO()
        sys.stderr = StringIO()

    def tearDown(self):
        """Reset standard output/error"""
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    def test_arg_parser_output(self):
        """Test --list-modes flag outputs correctly"""
        with self.assertRaises(SystemExit):
            SSTVCommand(["--list-modes"])
        modes = "Supported modes: Robot 36, Robot 72, Martin 2, Martin 1, Scottie 2, Scottie 1, Scottie DX"
        self.assertEqual(sys.stdout.getvalue().strip(), modes, "List of modes not equal")

    def test_arg_parser_decode_error(self):
        """Test decode flag with no input"""
        with self.assertRaises(SystemExit):
            SSTVCommand(["-d"])
        self.assertIn("expected one argument", sys.stderr.getvalue().strip(),
                      "'Wrong argument' error message not present in output")

        with self.assertRaises(SystemExit):
            SSTVCommand(["--decode"])
        self.assertIn("expected one argument", sys.stderr.getvalue().strip(),
                      "'Wrong argument' error message not present in output")

        with self.assertRaises(SystemExit):
            SSTVCommand(["-d", "./test/data/abc123"])
        self.assertIn("No such file or directory", sys.stderr.getvalue().strip(),
                      "'No file' error message not present in output")

    def test_arg_parser_decode_success(self):
        """Test decode flag with no input"""
        args = SSTVCommand(["-d", "./test/data/m1.ogg"]).args
        self.assertTrue(hasattr(args, "audio_file"),
                        "audio_file attribute not set")
        self.assertEqual(args.audio_file.name, "./test/data/m1.ogg",
                         "Audio file name not set correctly")
        self.assertTrue(hasattr(args, "skip"),
                        "skip attribute not set")
        self.assertEqual(args.skip, 0.0,
                         "skip value not set to default value")

    def test_arg_parser_decode_set_skip(self):
        """Test setting the skip flag to a custom value"""
        args = SSTVCommand(["-d", "./test/data/m1.ogg", "-s", "15.50"]).args
        self.assertTrue(hasattr(args, "skip"),
                        "skip attribute not set")
        self.assertEqual(args.skip, 15.5,
                         "skip value not set correctly")
