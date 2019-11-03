"""Test cases for the decoder code"""

import sys
import unittest

from sstv.decode import barycentric_peak_interp, calc_lum, SSTVDecoder


class SSTVDecoderTestCase(unittest.TestCase):
    """Test SSTVDecoder class"""

    def test_calc_lum(self):
        self.assertEqual(calc_lum(1450), 0)
        self.assertEqual(calc_lum(2350), 255)
        self.assertEqual(calc_lum(1758.1531), 82)

    def test_barycentric_peak_interp(self):
        bins = [100, 50, 0, 25, 50, 75, 100, 200, 150, 100]
        # Left neighbour is higher, so result must be smaller/equal
        self.assertLess(barycentric_peak_interp(bins, 9), 9)
        # Right neighbour is smaller and no left, so it must be smaller
        self.assertLessEqual(barycentric_peak_interp(bins, 0), 0)
        # Right neighbour is larger than left, so x must increase
        self.assertGreaterEqual(barycentric_peak_interp(bins, 7), 7)

        bins = [1, 2, 2, 2, 1]
        # Centre 2 surrounded by 2s should result in no change
        self.assertEqual(barycentric_peak_interp(bins, 2), 2)
