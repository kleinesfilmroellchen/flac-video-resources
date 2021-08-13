# SPDX-License-Identifier: Apache-2.0

from manim import *
from math import *
from itertools import *
from manim.animation.animation import DEFAULT_ANIMATION_RUN_TIME
import scipy.io.wavfile
from scipy.fft import rfft, rfftfreq
from scipy import signal
import numpy as np

FONT = 'Candara'

# Experimentally-determined sweet spot, slightly lower values are also good.
FFT_SAMPLE_RATE = 10
FFT_BANDS = 60
LABEL_COUNT = 12
UPSCALE_RATIO = 8

MAX_SCALE = 2500
MAX_FREQ = 20_000
MIN_FREQ = 20
MAX_FREQ_LOG = log(MAX_FREQ)
MIN_FREQ_LOG = log(MIN_FREQ)

DYNAMIC_RANGE = 1000
LOG_B = log(DYNAMIC_RANGE)
LOG_A = 1/DYNAMIC_RANGE

EPSILON = 0.001


def show_waveform_for(manim: Scene, file: str, time: int = 10):
    rate, data = scipy.io.wavfile.read(file, mmap=True)
    # Average multi-channel WAVs
    if len(data.shape) > 1:
        mono_data = np.average(data, axis=1)
    else:
        mono_data = data

    # Short-time FFT over the entire file
    freqs, t, Zxx = signal.stft(
        mono_data[:rate*time], 1.0, nperseg=rate/FFT_SAMPLE_RATE)
    freqs *= rate
    abs_Zxx = np.abs(Zxx)
    # clipping... not needed here
    # p = np.percentile(abs_Zxx, 99)
    # np.clip(abs_Zxx, a_min=0, a_max=p, out=abs_Zxx)
    max = np.max(abs_Zxx)
    abs_Zxx /= max
    print(max, abs_Zxx.shape)

    baseline = Line(start=LEFT*5.1, end=RIGHT*5).set_color(BLUE).shift(DOWN*3)
    manim.play(GrowFromEdge(baseline, LEFT))

    log_freqs = np.arange(MIN_FREQ_LOG, MAX_FREQ_LOG,
                          (MAX_FREQ_LOG - MIN_FREQ_LOG) / FFT_BANDS)
    actual_freqs = np.exp(log_freqs)

    prev_lines = None
    labels = VGroup()
    printed_labels = False
    for spectrum in abs_Zxx.transpose():
        spectrum_by_freq = FrequencyIndexer(spectrum, freqs)
        spectrum_log_segmented = []

        # Compute sections of frequencies
        freq_start = actual_freqs[0]
        last_value = EPSILON
        for freq_end in actual_freqs[1:]:
            avg = abs(np.average(spectrum_by_freq[freq_start:freq_end]))
            if isnan(avg):
                spectrum_log_segmented.append(last_value)
            else:
                spectrum_log_segmented.append(avg)
                last_value = avg
            freq_start = freq_end

        # Draw frequency lines
        x_vals = np.arange(-5, 5, 10/FFT_BANDS)
        x_index = range(len(x_vals))
        freqlines = []
        for index, x, value, start_freq in zip(x_index, x_vals, spectrum_log_segmented, actual_freqs):
            x = RIGHT*x
            # doesn't give proper values, it'd be great if someone could fix this
            # att = max(log(value / LOG_B) / LOG_A, EPSILON)
            att = value**(1/4)

            # Print frequency labels if necessary
            if not printed_labels and index % (FFT_BANDS//LABEL_COUNT) == 0:
                ftext = int(round(abs(start_freq), -1))
                print(ftext)
                labels.add(Text('{}'.format(ftext),
                                font_size=0.25, font=FONT).shift(DOWN*3.5 + x))

            freqline = Line(start=x, end=4*UP*att + x).shift(DOWN*2.5)
            freqlines.append(freqline)

        if not printed_labels:
            manim.play(Write(labels))
            manim.wait()
            printed_labels = True

        lines = Group(*freqlines)
        if not prev_lines:
            manim.play(GrowFromEdge(lines, DOWN),
                       run_time=DEFAULT_ANIMATION_RUN_TIME / 2)
        else:
            manim.play(TransformMatchingShapes(prev_lines, lines,
                                               # As two segments overlap halfway, we need to speed up playback.
                                               run_time=1 / (FFT_SAMPLE_RATE*2)))

        prev_lines = lines

    manim.play(ShrinkToEdge(prev_lines, DOWN),
               run_time=DEFAULT_ANIMATION_RUN_TIME / 2)
    manim.play(Unwrite(labels))
    manim.play(ShrinkToEdge(baseline, LEFT))


def upsample_4x(data, rate):
    zeroes = []
    for i in range(UPSCALE_RATIO):
        zeroes.append(np.zeros(data.shape))
    data_4x = np.stack((data, *zeroes)).flatten()

    b, a = signal.butter(3, 22_000, fs=rate*UPSCALE_RATIO)
    signal.lfilter(b, a, data_4x)
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, data_4x, zi=zi*data_4x[0])
    return z


class FrequencyIndexer:
    '''Returns actual spectrum data slices from given frequency "indices".'''

    def __init__(self, data, frequencies):
        self.__data = data
        self.__frequencies = frequencies

    def __lowest_index_below(self, freq) -> int:
        '''Computes the lowest index that is below the frequency value.'''
        index = 0
        for possible_freq in self.__frequencies:
            if possible_freq >= freq:
                break
            index += 1
        return index

    def __getitem__(self, key):
        try:
            int_key = int(key)
            return self.__data[self.__lowest_index_below(int_key)]
        except TypeError:
            if key.step != 1 and key.step is not None:
                raise ValueError("Frequency index slices cannot use steps")
            start = self.__lowest_index_below(key.start)
            end = self.__lowest_index_below(key.stop)
            return self.__data[start:end]


class ShrinkToPoint(GrowFromPoint):
    def __init__(
        self, mobject: "Mobject", point: np.ndarray, point_color: str = None, **kwargs
    ) -> None:
        self.point = point
        self.point_color = point_color
        super().__init__(mobject, point, point_color, **kwargs)

    def create_starting_mobject(self) -> "Mobject":
        return self.mobject

    def create_target(self) -> "Mobject":
        start = super().create_starting_mobject()
        start.scale(0)
        start.move_to(self.point)
        if self.point_color:
            start.set_color(self.point_color)
        return start


# Mixin trickery so that GrowFromEdge uses ShrinkToPoint as its animation generator
class ShrinkToEdge(GrowFromEdge, ShrinkToPoint):
    pass
