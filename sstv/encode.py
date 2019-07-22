"""Class and methods to encode SSTV signal"""

from . import spec
from .common import log_message, progress_bar
import numpy as np


def calc_freq(lum):
    """Converts 0-255 luminance byte into SSTV pixel frequency range"""

    freq = (lum * 3.1372549) + 1500
    return 2300 if freq > 2300 else 1500 if freq < 1500 else freq


class SSTVEncoder(object):

    """Create an SSTV encoder that will encode a PIL image"""

    def __init__(self, image, mode=spec.M1):
        self.mode = mode
        self._sample_rate = 44100
        self._orig_image = image
        self._sig_phase = 0

    def encode(self):
        """Encodes image data into SSTV audio signal"""
        log_message("Encoding SSTV image with mode {}".format(self.mode.NAME))

        header = np.append(self.encode_header(), self.encode_vis())
        image_data = self.create_image_data()
        full_audio = np.append(header, self.encode_image_data(image_data))

        log_message("Done!")
        return full_audio, self._sample_rate

    def freq_to_samples(self, freq, length):
        """Creates an array of samples using a sine wave of given frequency"""

        sample_count = round(length * self._sample_rate)

        time_space = np.linspace(0, length, sample_count, endpoint=False)
        mult = freq * 2 * np.pi
        data = np.array(np.sin(time_space * mult + self._sig_phase),
                        dtype=np.float64)
        self._sig_phase += length * mult
        return data

    def add_tone_data(self, data, point, freq, length):
        """Adds sample data of given freq to array at specified point"""
        sig = self.freq_to_samples(freq, length)
        data[point:point+len(sig)] = sig

    def encode_header(self):
        """Returns audio data of the SSTV header"""
        return np.concatenate((self.freq_to_samples(1900, 0.300),
                               self.freq_to_samples(1200, 0.010),
                               self.freq_to_samples(1900, 0.300)))

    def encode_vis(self):
        """Encodes the VIS code into the correct frequencies"""
        vis_value = None
        for vis, mode in spec.VIS_MAP.items():
            if mode == self.mode:
                vis_value = vis
        if vis_value is None:
            raise ValueError("No vis code for mode: {}".format(self.mode.NAME))

        vis_bits = []
        for bit_idx in range(7):
            vis_bits.append((vis_value >> bit_idx) & 0x01)

        # Add even parity bit
        parity = sum(vis_bits) % 2
        vis_bits.append(parity)

        data = self.freq_to_samples(1200, 0.030)
        for bit in vis_bits:
            freq = 1100 if bit else 1300
            data = np.append(data, self.freq_to_samples(freq, 0.030))

        data = np.append(data, self.freq_to_samples(1200, 0.030))

        return data

    def create_image_data(self):
        """Transforms image data into correct format to be encoded"""
        log_message("Formatting image")

        width = self.mode.LINE_WIDTH
        height = self.mode.LINE_COUNT
        channels = self.mode.CHAN_COUNT

        image = self._orig_image.resize((width, height))
        pixel_data = image.load()

        image_data = []
        for y in range(height):
            image_data.append([])
            for x in range(width):
                if self.mode.COLOR == spec.COL_FMT.GBR:
                    pixel = (pixel_data[x, y][1],
                             pixel_data[x, y][2],
                             pixel_data[x, y][0])
                    image_data[y].append(pixel)

        return image_data

    def encode_image_data(self, image_data):
        """Encodes the actual image data into array of samples"""
        sync_length = round(self.mode.SYNC_PULSE * self._sample_rate)
        porch_length = round(self.mode.SYNC_PORCH * self._sample_rate)
        sep_length = round(self.mode.SEP_PULSE * self._sample_rate)

        total_time = round(self.mode.LINE_TIME * self.mode.LINE_COUNT
                           * self._sample_rate)

        data = np.zeros(total_time)
        data_ptr = 0

        height = self.mode.LINE_COUNT
        channels = self.mode.CHAN_COUNT
        width = self.mode.LINE_WIDTH

        pixel_time = self.mode.PIXEL_TIME

        for line in range(height):
            for chan in range(channels):

                if chan == self.mode.CHAN_SYNC:
                    self.add_tone_data(data, data_ptr, 1200,
                                       self.mode.SYNC_PULSE)
                    data_ptr += sync_length
                    self.add_tone_data(data, data_ptr, 1500,
                                       self.mode.SYNC_PORCH)
                    data_ptr += porch_length

                last_px_end = data_ptr
                for px in range(width):
                    px_pos = last_px_end
                    last_px_end = data_ptr + round((px + 1) * pixel_time
                                                   * self._sample_rate)
                    px_size = (last_px_end - px_pos) / self._sample_rate
                    freq = calc_freq(image_data[line][px][chan])
                    self.add_tone_data(data, px_pos, freq, px_size)

                data_ptr = last_px_end  # end of last pixel
                self.add_tone_data(data, data_ptr, 1500, self.mode.SEP_PULSE)
                data_ptr += sep_length

            progress_bar(line, height - 1, "Encoding image data...")

        return data
