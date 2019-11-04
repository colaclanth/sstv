"""Test cases for the decoder code"""

import unittest

from sstv.decode import barycentric_peak_interp, calc_lum, SSTVDecoder


class SSTVDecoderTestCase(unittest.TestCase):
    """Test SSTVDecoder class"""

    def test_calc_lum(self):
        """Test function that calculates pixel byte from frequency"""
        self.assertEqual(calc_lum(1450), 0)
        self.assertEqual(calc_lum(2350), 255)
        self.assertEqual(calc_lum(1758.1531), 82)

    def test_barycentric_peak_interp(self):
        """Test function to interpolate the x value from frequency bins"""
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

    def test_decoder_init(self):
        """Test SSTVDecoder init"""
        with open("./test/data/m1.ogg", 'rb') as fp:
            with SSTVDecoder(fp) as decoder:
                self.assertEqual(decoder._audio_file, fp)
                self.assertEqual(decoder._sample_rate, 44100)

    def test_decoder_freq_detect(self):
        """Test the peak frequency detection function"""
        with open("./test/data/220hz_sine.ogg", 'rb') as fp:
            with SSTVDecoder(fp) as decoder:
                # Test using all samples in sound file
                freq = round(decoder._peak_fft_freq(decoder._samples))
                self.assertEqual(freq, 220, "Incorrect frequency determined by peak detector using all samples")

                # Test using 1/4 of a second of samples
                freq = round(decoder._peak_fft_freq(decoder._samples[:11025]))
                self.assertEqual(freq, 220, "Incorrect frequency determined by peak detector using 1/4 second samples")

                # Test using 2000 samples
                freq = round(decoder._peak_fft_freq(decoder._samples[:1000]))
                self.assertEqual(freq, 220, "Incorrect frequency determined by peak detector using 1000 samples")
