"""Class and methods to decode SSTV signal"""

from . import spec
from .common import log_message, progress_bar
from PIL import Image
from scipy.signal.windows import hann
import numpy as np
import soundfile


def calc_lum(freq):
    """Converts SSTV pixel frequency range into 0-255 luminance byte"""

    lum = int(round((freq - 1500) / 3.1372549))
    if lum > 255:
        return 255
    elif lum < 0:
        return 0
    else:
        return lum


def barycentric_peak_interp(bins, x):
    """Interpolate between frequency bins to find x value of peak"""

    # Takes x as the index of the largest bin and interpolates the
    # x value of the peak using neighbours in the bins array

    # Make sure data is in bounds
    if x <= 0:
        y1 = bins[x]
    else:
        y1 = bins[x-1]

    if x + 1 >= len(bins):
        y3 = bins[x]
    else:
        y3 = bins[x+1]

    denom = y3 + bins[x] + y1
    if denom == 0:
        return 0  # erroneous

    return (y3 - y1) / denom + x


class SSTVDecoder(object):

    """Create an SSTV decoder for decoding audio data"""

    def __init__(self, audio_file):
        self.log_basic = True
        self.mode = None

        self._audio_file = audio_file

        self._samples, self._sample_rate = soundfile.read(self._audio_file)

        if self._samples.ndim > 1:  # convert to mono if stereo
            self._samples = self._samples.mean(axis=1)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        self.close()

    def __del__(self):
        self.close()

    def decode(self, skip=0.0):
        """Attempts to decode the audio data as an SSTV signal

        Returns a PIL image on success, and None if no SSTV signal was found
        """

        if skip > 0.0:
            self._samples = self._samples[round(skip * self._sample_rate):]

        header_end = self._find_header()

        if header_end is None:
            return None

        vis_end = header_end + round(spec.VIS_BIT_SIZE * 9 * self._sample_rate)
        vis_section = self._samples[header_end:vis_end]

        self.mode = self._decode_vis(vis_section)

        transmission_area = self._samples[vis_end:]
        image_data = self._decode_image_data(transmission_area)

        return self._draw_image(image_data)

    def close(self):
        """Closes any input files if they exist"""

        if self._audio_file is not None and not self._audio_file.closed:
            self._audio_file.close()

    def _peak_fft_freq(self, data):
        """Finds the peak frequency from a section of audio data"""

        windowed_data = data * hann(len(data))
        fft = np.abs(np.fft.rfft(windowed_data))

        # Get index of bin with highest magnitude
        x = np.argmax(fft)
        # Interpolated peak frequency
        peak = barycentric_peak_interp(fft, x)

        # Return frequency in hz
        return peak * self._sample_rate / len(windowed_data)

    def _find_header(self):
        """Finds the approx sample of the end of the calibration header"""

        header_size = round(spec.HDR_SIZE * self._sample_rate)
        window_size = round(spec.HDR_WINDOW_SIZE * self._sample_rate)

        leader_1_sample = 0
        leader_1_search = leader_1_sample + window_size

        break_sample = round(spec.BREAK_OFFSET * self._sample_rate)
        break_search = break_sample + window_size

        leader_2_sample = round(spec.LEADER_OFFSET * self._sample_rate)
        leader_2_search = leader_2_sample + window_size

        vis_start_sample = round(spec.VIS_START_OFFSET * self._sample_rate)
        vis_start_search = vis_start_sample + window_size

        jump_size = round(0.002 * self._sample_rate)  # check every 2ms

        # The margin of error created here will be negligible when decoding the
        # vis due to each bit having a length of 30ms. We fix this error margin
        # when decoding the image by aligning each sync pulse

        for current_sample in range(0, len(self._samples) - header_size,
                                    jump_size):
            if current_sample % (jump_size * 256) == 0:
                search_msg = "Searching for calibration header... {:.1f}s"
                progress = current_sample / self._sample_rate
                log_message(search_msg.format(progress), recur=True)

            search_end = current_sample + header_size
            search_area = self._samples[current_sample:search_end]

            leader_1_area = search_area[leader_1_sample:leader_1_search]
            break_area = search_area[break_sample:break_search]
            leader_2_area = search_area[leader_2_sample:leader_2_search]
            vis_start_area = search_area[vis_start_sample:vis_start_search]

            if (abs(self._peak_fft_freq(leader_1_area) - 1900) < 50
               and abs(self._peak_fft_freq(break_area) - 1200) < 50
               and abs(self._peak_fft_freq(leader_2_area) - 1900) < 50
               and abs(self._peak_fft_freq(vis_start_area) - 1200) < 50):

                stop_msg = "Searching for calibration header... Found!{:>4}"
                log_message(stop_msg.format(' '))
                return current_sample + header_size

        log_message()
        log_message("Couldn't find SSTV header in the given audio file",
                    err=True)
        return None

    def _decode_vis(self, vis_section):
        """Decodes the vis from the audio data and returns the SSTV mode"""

        bit_size = round(spec.VIS_BIT_SIZE * self._sample_rate)
        vis_bits = []

        for bit_idx in range(8):
            bit_offset = bit_idx * bit_size
            section = vis_section[bit_offset:bit_offset+bit_size]
            freq = self._peak_fft_freq(section)
            vis_bits.append(int(freq <= 1200))

        # check for even parity in last bit
        parity = sum(vis_bits) % 2 == 0
        if not parity:
            raise ValueError("error decoding VIS header (invalid parity bit)")

        # LSB first so we must reverse and ignore the parity bit
        vis_value = 0
        for bit in vis_bits[-2::-1]:
            vis_value = (vis_value << 1) | bit

        if vis_value not in spec.VIS_MAP:
            error = "SSTV mode is unsupported (VIS: {})"
            raise ValueError(error.format(vis_value))

        mode = spec.VIS_MAP[vis_value]
        log_message("Detected SSTV mode {}".format(mode.NAME))

        return mode

    def _align_sync(self, align_section, start_of_sync=True):
        """Returns sample where the beginning of the sync pulse was found"""

        # TODO - improve this

        sync_window = round(self.mode.SYNC_PULSE * 1.4 * self._sample_rate)
        search_end = len(align_section) - sync_window

        for current_sample in range(search_end):
            align_end = current_sample + sync_window
            search_section = align_section[current_sample:align_end]

            if self._peak_fft_freq(search_section) > 1350:
                break

        end_sync = current_sample + (sync_window // 2)

        if start_of_sync:
            return end_sync - round(self.mode.SYNC_PULSE * self._sample_rate)
        else:
            return end_sync

    def _decode_image_data(self, transmission):
        """Decodes image from the transmission section of an sstv signal"""

        window_factor = self.mode.WINDOW_FACTOR
        centre_window_time = (self.mode.PIXEL_TIME * window_factor) / 2
        pixel_window = round(centre_window_time * 2 * self._sample_rate)

        image_data = []

        seq_start = 0
        if self.mode.HAS_START_SYNC:
            # Start at the end of the initial sync pulse
            seq_start = self._align_sync(transmission, start_of_sync=False)

        for line in range(self.mode.LINE_COUNT):
            image_data.append([])

            if self.mode.CHAN_SYNC > 0 and line == 0:
                # Align seq_start to the beginning of the previous sync pulse
                sync_offset = self.mode.CHAN_OFFSETS[self.mode.CHAN_SYNC]
                seq_start -= round((sync_offset + self.mode.SCAN_TIME)
                                   * self._sample_rate)

            for chan in range(self.mode.CHAN_COUNT):
                image_data[line].append([])

                if chan == self.mode.CHAN_SYNC:
                    if line > 0 or chan > 0:
                        # Set base offset to the next line
                        seq_start += round(self.mode.LINE_TIME *
                                           self._sample_rate)

                    # Align to start of sync pulse
                    seq_start += self._align_sync(transmission[seq_start:])

                pixel_time = self.mode.PIXEL_TIME
                if self.mode.HAS_MERGE_SCAN:
                    # Robot mode has half-length second scan
                    if chan % 2 == 1:
                        pixel_time = self.mode.MERGE_PIXEL_TIME

                    centre_window_time = (pixel_time * window_factor) / 2
                    pixel_window = round(centre_window_time * 2 *
                                         self._sample_rate)

                for px in range(self.mode.LINE_WIDTH):

                    chan_offset = self.mode.CHAN_OFFSETS[chan]

                    px_sample = round(seq_start + (chan_offset + px *
                                      pixel_time - centre_window_time) *
                                      self._sample_rate)
                    pixel_area = transmission[px_sample:px_sample+pixel_window]
                    freq = self._peak_fft_freq(pixel_area)

                    image_data[line][chan].append(calc_lum(freq))

            progress_bar(line, self.mode.LINE_COUNT - 1, "Decoding image... ")

        return image_data

    def _draw_image(self, image_data):
        """Renders the image from the decoded sstv signal"""

        if self.mode.COLOR == spec.COL_FMT.YUV:
            col_mode = "YCbCr"
        else:
            col_mode = "RGB"

        width = self.mode.LINE_WIDTH
        height = self.mode.LINE_COUNT

        image = Image.new(col_mode, (width, height))
        pixel_data = image.load()

        log_message("Drawing image data...")

        for y in range(height):
            odd_line = y % 2

            for x in range(width):
                if self.mode.COLOR == spec.COL_FMT.GBR:
                    pixel = (image_data[y][2][x],
                             image_data[y][0][x],
                             image_data[y][1][x])
                elif self.mode.COLOR == spec.COL_FMT.YUV:
                    pixel = (image_data[y][0][x],
                             image_data[y-(odd_line-1)][1][x],
                             image_data[y-odd_line][1][x])
                else:
                    pixel = (image_data[y][0][x],
                             image_data[y][1][x],
                             image_data[y][2][x])

                pixel_data[x, y] = pixel

        if image.mode.COLOR != "RGB":
            image = image.convert("RGB")

        log_message("...Done!")
        return image
