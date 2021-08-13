# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Callable
from manim import *
from manim.animation.animation import DEFAULT_ANIMATION_RUN_TIME
from manim.animation.composition import DEFAULT_LAGGED_START_LAG_RATIO
from main import TRANSPARENT, font_args, small_font_args, mono_font_args, theming_colors
import numpy as np
from random import Random, sample
from math import cos, floor, sin, pi, asin
from functools import partial


class DigitalAudioIntro(MovingCameraScene):
    def construct(self):
        self.next_section()
        digital_audio = Text('Digital Audio', **font_args).shift(UP*2)
        self.play(Write(digital_audio))
        self.play(Unwrite(digital_audio))
        self.wait()

        explanation = Text('Air molecules (low pressure, high pressure)', **font_args, disable_ligatures=True, color=theming_colors[0],
                           t2c={'low pressure': theming_colors[1], 'high pressure': theming_colors[2]}).shift(DOWN*3.5)
        explanation.font_size = DEFAULT_FONT_SIZE*.5

        lparrow = Arrow(
            color=theming_colors[1], stroke_width=3, max_tip_length_to_length_ratio=0.3, buff=0, start=DOWN/6, end=UP/2).shift(LEFT*8, DOWN*3.1)
        hparrow = Arrow(
            color=theming_colors[2], stroke_width=3, max_tip_length_to_length_ratio=0.3, buff=0, start=DOWN/6, end=UP/2).shift(LEFT*8, DOWN*3.1)

        pf = self.make_pressure_field().shift(UP*.5)
        # Animating a fade-in on a point cloud is not possible right now.
        self.add(pf, lparrow, hparrow)
        self.play(Write(explanation))
        self.wait()
        ripple_count = 20
        self.play(LaggedStart(
            ApplyWave(pf,
                      direction=RIGHT,
                      rate_func=rate_functions.linear,
                      wave_func=rate_functions.ease_in_out_sine,
                      amplitude=1/4,
                      time_width=4,
                      run_time=ripple_count,
                      ripples=ripple_count
                      ),
            hparrow.animate(rate_func=rate_functions.linear,
                            run_time=4.8).shift(RIGHT*20),
            lparrow.animate(rate_func=rate_functions.linear,
                            run_time=4.8).shift(RIGHT*20),
            lag_ratio=0.17)
        )

        self.wait()
        self.remove(lparrow, hparrow, pf, explanation)
        self.next_section('Waves')

        coordinates = Axes(x_range=[-0.5, 20, 20], y_range=[-0.05, 1.2, 1], x_length=10, y_length=6, axis_config={
            'include_ticks': True,
            'include_tip': True,
            'tip_width': 0.001,
            'tip_height': 0.001,
            'include_numbers': False,
            'exclude_origin_tick': False,
        })
        pressure_text = Text('Pressure', **font_args)
        pressure_text.font_size = DEFAULT_FONT_SIZE*.6
        pressure = coordinates.get_y_axis_label(pressure_text)

        time_text = Text('Time', **font_args)
        time_text.font_size = DEFAULT_FONT_SIZE*.6
        time = coordinates.get_x_axis_label(time_text)

        self.play(Create(coordinates), Write(pressure), Write(time))

        sinwave = coordinates.plot(lambda x: sin(
            2*pi*x/20)/2+.5, x_range=(0, 20, 0.2), color=theming_colors[0])
        self.play(Create(sinwave))

        hparrow = coordinates.get_T_label(
            x_val=5, graph=sinwave, triangle_color=theming_colors[1], line_color=theming_colors[1])
        lparrow = coordinates.get_T_label(
            x_val=15, graph=sinwave, triangle_color=theming_colors[2], line_color=theming_colors[2])
        self.play(Write(hparrow))
        self.play(Write(lparrow))
        self.play(Unwrite(lparrow), Unwrite(hparrow))
        self.play(Uncreate(sinwave))

        self.wait()
        self.next_section('Frequency, range of hearing')

        mask = Rectangle(color=BLACK, height=5, width=4,
                         fill_opacity=1).shift(RIGHT*7.1, DOWN*.3)
        higher_amplitude_wave = coordinates.plot(lambda x: sin(
            2*pi*x/20)/2+.5, x_range=(0, 190, 0.2), color=theming_colors[0])

        self.add_foreground_mobjects(mask, time)
        self.play(Create(higher_amplitude_wave))
        self.play(higher_amplitude_wave.animate(run_time=3).apply_complex_function(
            lambda c: complex((c.real-coordinates.c2p(0, 0, 0)[0])/10+coordinates.c2p(0, 0, 0)[0], c.imag)))

        self.wait()

        frequency = Text('Frequency', **font_args).shift(UP*3)
        frequency.font_size = 0.9 * DEFAULT_FONT_SIZE
        frequency_formula = MathTex(
            r'f \overset{\wedge}{=} \frac{1}{s}\quad [f] = \text{ Hertz (Hz)}').shift(RIGHT*5, UP*3)

        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.shift(UP*.5, RIGHT))
        self.play(Write(frequency))
        self.play(Write(frequency_formula))

        self.wait()
        self.camera.frame.restore()
        self.remove(coordinates, frequency, frequency_formula,
                    mask, higher_amplitude_wave, pressure, time)
        self.next_section('Analog voltage and continuity')

        left_axis = Axes(x_range=[-0.5, 20, 20], y_range=[-0.05, 1.2, 1], x_length=5.5, y_length=4, axis_config={
            'include_ticks': True,
            'include_tip': True,
            'tip_width': 0.0005,
            'tip_height': 0.0005,
            'include_numbers': False,
            'exclude_origin_tick': False,
        }).shift(LEFT*4)
        right_axis = left_axis.copy().shift(RIGHT*8)

        pressure_text = Tex('Pressure', font_size=DEFAULT_FONT_SIZE*.4)
        pressure_left = left_axis.get_y_axis_label(pressure_text)

        time_text = Tex('Time', font_size=DEFAULT_FONT_SIZE*.4)
        time_left = left_axis.get_x_axis_label(time_text)

        voltage_text_right = Tex(
            '$U$ (Voltage)', font_size=DEFAULT_FONT_SIZE*.4)
        voltage_right = right_axis.get_y_axis_label(voltage_text_right)

        time_text_right = time_text.copy()
        time_right = right_axis.get_x_axis_label(time_text_right)

        self.play(Write(right_axis), Write(left_axis), Write(voltage_right), Write(
            time_right), Write(pressure_left), Write(time_left))
        self.wait()

        def weird_wave(unscaled_x, pos):
            x = (unscaled_x + pos) * 7
            y = (sin(2 * x - 2) + .2 * cos(6 * x) + .5 * sin(5 * x - 8) + .5 *
                 cos(2 * x - 4) + 3 * cos(.2 * x - 1) - 2 * sin(x))
            return y / 12 + .5

        pressure_wave = left_axis.plot(lambda x: weird_wave(
            x, 0), x_range=(0, 19, .02), color=theming_colors[0])
        voltage_wave = right_axis.plot(lambda x: weird_wave(
            x, 0), x_range=(0, 19, .005), color=theming_colors[1])
        arrow = Arrow(start=LEFT, end=RIGHT)

        lr = DEFAULT_LAGGED_START_LAG_RATIO*3
        self.play(
            LaggedStart(
                WavePlayback(pressure_wave, weird_wave,
                             (0, 200), left_axis, run_time=10*DEFAULT_ANIMATION_RUN_TIME, rate_func=linear),
                Write(arrow),
                FadeIn(voltage_wave, run_time=0.0001),
                WavePlayback(voltage_wave, weird_wave,
                             (0, 180), right_axis, run_time=(10-lr*2)*DEFAULT_ANIMATION_RUN_TIME, rate_func=linear),
                lag_ratio=lr))

        self.wait()

        self.play(self.camera.frame.animate.shift(
            RIGHT*4).scale(.6), Unwrite(arrow))

        self.camera.frame.save_state()
        xmark = right_axis.get_T_label(
            x_val=3.15, graph=voltage_wave, triangle_color=WHITE, line_color=WHITE, triangle_size=DEFAULT_ARROW_TIP_LENGTH*.4)
        xhmark = right_axis.get_T_label(
            x_val=3.2, graph=voltage_wave, triangle_color=WHITE, line_color=WHITE, triangle_size=DEFAULT_ARROW_TIP_LENGTH*.4)
        self.play(Write(xmark), Write(xhmark))
        xhmark.save_state()
        xmark.save_state()
        voltage_wave.save_state()
        self.play(self.camera.frame.animate.shift(LEFT*1.75, DOWN*1.25).scale(.03),
                  voltage_wave.animate.set_stroke(
                      width=DEFAULT_STROKE_WIDTH*.1),
                  xhmark.animate.set_stroke(width=DEFAULT_STROKE_WIDTH*.1),
                  xmark.animate.set_stroke(width=DEFAULT_STROKE_WIDTH*.1),
                  run_time=DEFAULT_ANIMATION_RUN_TIME*1)
        self.wait()
        self.play(Restore(self.camera.frame), Restore(xhmark),
                  Restore(xmark), Restore(voltage_wave))

        gain_descriptor = MathTex('0.00', font_size=0.6*DEFAULT_FONT_SIZE)
        gain_line = right_axis.get_horizontal_line(
            right_axis.c2p(20, 0.5), line_func=DashedLine, color=WHITE)

        def gain_updator(d):
            newd = MathTex(r'{:.5f}\dots'.format(
                right_axis.p2c(gain_line.get_start())[1]), font_size=0.6*DEFAULT_FONT_SIZE)
            newd.move_to(gain_line.get_start(), RIGHT).shift(
                LEFT*SMALL_BUFF)
            d.become(newd)
        gain_descriptor.add_updater(gain_updator)
        self.play(Create(gain_descriptor), Write(gain_line),
                  run_time=0.5*DEFAULT_ANIMATION_RUN_TIME)
        self.play(gain_line.animate.shift(UP*1.52278))
        self.wait(duration=DEFAULT_WAIT_TIME*.3)
        self.play(gain_line.animate.shift(DOWN*2.513))
        self.wait(duration=DEFAULT_WAIT_TIME*.3)
        self.play(gain_line.animate.shift(UP*0.913))
        self.wait(duration=DEFAULT_WAIT_TIME*.3)
        self.play(gain_line.animate.shift(UP*0.143))
        self.wait()

        self.play(Uncreate(gain_line), Uncreate(gain_descriptor), Unwrite(
            xhmark), Unwrite(xmark), self.camera.frame.animate.shift(LEFT*4).scale(1/.6))

        infinite_info = Text('Infinite information', **
                             font_args).shift(DOWN*2.7)
        infinite_info.font_size = 0.7*DEFAULT_FONT_SIZE
        right_arrow = Arrow(start=infinite_info.get_right(
        ), end=right_axis.get_bottom(), max_stroke_width_to_length_ratio=3).shift(RIGHT*SMALL_BUFF)
        left_arrow = Arrow(start=infinite_info.get_left(
        ), end=left_axis.get_bottom(), max_stroke_width_to_length_ratio=3).shift(LEFT*SMALL_BUFF)

        self.play(Write(infinite_info), Write(
            right_arrow), Write(left_arrow))
        self.wait()
        computer_info = Text('... and that\'s too much for a computer',
                             **font_args).align_to(infinite_info.get_bottom(), UP)
        computer_info.font_size = 0.4*DEFAULT_FONT_SIZE
        self.play(Write(computer_info))

        self.wait()

    def make_pressure_field(self):
        rng = Random(9902)

        cloud = PointCloudDot(
            color=theming_colors[0], stroke_width=7, density=20)
        cloud.apply_complex_function(
            lambda z: complex(rng.uniform(-8, 8), rng.uniform(-3, 4)))
        return cloud


class AnalogToDigital(MovingCameraScene):
    def construct(self):
        self.next_section(skip_animations=False)

        hearinglimit_title = Text(
            'Limits of human hearing', **font_args).shift(UP*1.5)
        freqrange = Tex(
            r'Frequency range: $f_\text{min} = 20\text{Hz},\;~ \; f_\text{max} = 20,\!000\text{Hz}$')
        pressrange = Tex(
            r'Pressure sensitivity: $I_0 = 10^{-12}\frac{W}{m^2}$\\ Pain threshold: $I_P = 10^{13}I_0$').shift(DOWN*1.5)
        self.play(Write(hearinglimit_title), Write(freqrange))
        self.play(Write(pressrange))

        self.wait()
        self.play(Unwrite(hearinglimit_title), Unwrite(
            freqrange), Unwrite(pressrange))

        coordinates = Axes(x_range=[-0.25, 8, 8], y_range=[-0.05, 1.2, 1], x_length=10, y_length=6, axis_config={
            'include_ticks': True,
            'include_tip': True,
            'tip_width': 0.001,
            'tip_height': 0.001,
            'include_numbers': False,
            'exclude_origin_tick': False,
        })
        pressure_text = Text('Pressure', **font_args)
        pressure_text.font_size = DEFAULT_FONT_SIZE*.4
        pressure = coordinates.get_y_axis_label(pressure_text)
        time_text = Text('Time', **font_args)
        time_text.font_size = DEFAULT_FONT_SIZE*.4
        time = coordinates.get_x_axis_label(time_text)

        def sampled_wave(x): return (
            (sin(1.1*x)+cos(.5*x)-.5*sin(3*x-8)))/(4)+.5
        wave = coordinates.plot(sampled_wave, x_range=[
                                0, 8, 0.001], color=theming_colors[0])
        self.play(Create(coordinates), Create(
            pressure), Create(time), Create(wave))

        # exactly one sample per 0.5 units, i.e. a 20kHz wave has a period of 1
        samplelines = coordinates.get_vertical_lines_to_graph(wave, x_range=[
            0.5, 7.5], num_lines=14, line_func=Line, color=theming_colors[1], stroke_width=3)
        sampledots = [Dot(line.get_end(), color=theming_colors[1])
                      for line in samplelines]
        sampledot_group = VGroup(*sampledots)
        self.play(Write(samplelines), Write(sampledot_group))

        self.play(FadeOut(samplelines), FadeOut(wave))
        self.wait()
        self.next_section('Nyquist-Shannon', skip_animations=False)

        nyquist = Text('Nyquist-Shannon Sampling Theorem',
                       **small_font_args).shift(DOWN*3.5)
        self.play(Write(nyquist))

        wave_20k = coordinates.plot(lambda x: sin(
            2*pi*x+.3)/2+.5, x_range=[0, 8, 0.001], color=theming_colors[2])
        descriptor_20k = MathTex(
            r'20,\!000\text{Hz}', color=theming_colors[2]).shift(RIGHT*5, UP*3)
        self.play(Succession(Create(wave_20k), Write(descriptor_20k)),
                  *[dot.animate.set_opacity(.5) for dot in sampledots])

        constant_1_helper = coordinates.plot(
            lambda x: 1, x_range=[-2, 20], color=TRANSPARENT)
        sampling_locations = coordinates.get_vertical_lines_to_graph(constant_1_helper, x_range=[
            0.5, 7.5], num_lines=14, line_func=Line, color=WHITE, stroke_width=2)
        self.play(Create(sampling_locations))

        sample_width_brace = BraceBetweenPoints(sampling_locations[4].get_end(
        ), sampling_locations[5].get_end(), direction=UP, sharpness=1.5)
        sample_width_text = MathTex(r'40,\!000\text{Hz}')
        sample_width_brace.put_at_tip(sample_width_text)
        self.play(Create(sample_width_brace), Write(sample_width_text))

        self.camera.frame.save_state()
        self.play(self.camera.frame.animate.shift(UP*6.1))

        common_samplerates = Text(
            'Common Sample Rates', **small_font_args).shift(UP*9)
        common_samplerates_values = MathTex(
            r'44,\!100\text{Hz (samples per second)} && \text{CD}\\48,\!000\text{Hz} && \text{DVD Audio}\\96,\!000\text{Hz} && \text{Some DAW Convolutions}\\192,\!000\text{Hz} && \text{Audiophiles}').shift(UP*7)
        self.play(Write(common_samplerates), Write(common_samplerates_values))
        self.wait()
        self.remove(nyquist, sampling_locations, wave_20k)
        for dot in sampledots:
            dot.set_opacity(1)
        self.play(Restore(self.camera.frame), FadeOut(
            sample_width_brace, sample_width_text, descriptor_20k))

        self.wait()
        self.next_section('Quantization', skip_animations=False)

        samplevalues = VGroup(*[MathTex(
            r'{:.6f}\dots'.format(coordinates.p2c(dot.get_end())[1])).scale(0.6) for dot in sampledots])
        samplevalues_quantized = VGroup(*[MathTex(
            r'{:05d}'.format(int(floor(coordinates.p2c(dot.get_end())[1]*(2**16))))).scale(0.6) for dot in sampledots])
        samplevalues.arrange_in_grid(rows=14, cols=1, buff=.2).shift(LEFT*2.5)
        samplevalues_quantized.arrange_in_grid(
            rows=14, cols=1, buff=.2)
        samplevalues_label = Text(
            'Value', **small_font_args).shift(LEFT*2.5, UP*3.5)
        quantized_label = Text(
            'Value (quantized)', **small_font_args).shift(UP*3.5)
        sampledots_label = Text(
            'Sample', **small_font_args).shift(LEFT*5, UP*3.5)
        samplevalues_label.font_size = DEFAULT_FONT_SIZE*0.6
        quantized_label.font_size = DEFAULT_FONT_SIZE*0.6
        sampledots_label.font_size = DEFAULT_FONT_SIZE*0.6

        self.play(FadeOut(coordinates, pressure, time),
                  sampledot_group.animate.arrange_in_grid(
                      rows=14, cols=1, buff=0.25).center().shift(LEFT*5),
                  Write(samplevalues), Write(samplevalues_label), Write(sampledots_label))
        self.play(Write(samplevalues_quantized), Write(quantized_label))

        self.play(FadeOut(sampledot_group, samplevalues, samplevalues_label,
                          samplevalues_quantized, sampledots_label, quantized_label))

        dash_size, dash_ratio = 0.2, 0.5
        dashline = DashedLine(start=ORIGIN, end=RIGHT*5, dash_length=dash_size,
                              dashed_ratio=dash_ratio, stroke_width=DEFAULT_STROKE_WIDTH*.5, color=GREY)
        six_values = MobjectTable(
            [[MathTex("2"), dashline.copy()],
             [MathTex("1"), dashline.copy()],
             [MathTex("0"), dashline.copy()],
             [MathTex("-1"), dashline.copy()],
             [MathTex("-2"), dashline.copy()],
             [MathTex("-3"), dashline.copy()]], line_config={'color': TRANSPARENT}, h_buff=0.3).shift(LEFT)
        self.play(Write(six_values))

        # FIXME: It would be nice if Manim could do this directly (aka. PolyLine)
        six_value_wave = VGroup()
        six_value_wave += Line(start=ORIGIN, end=RIGHT,
                               color=theming_colors[0])
        six_value_wave += Line(start=RIGHT, end=RIGHT +
                               UP, color=theming_colors[0])
        six_value_wave += Line(start=RIGHT+UP, end=RIGHT *
                               2+UP, color=theming_colors[0])
        six_value_wave += Line(start=RIGHT*2+UP,
                               end=RIGHT*2, color=theming_colors[0])
        six_value_wave += Line(start=RIGHT*2, end=RIGHT *
                               3, color=theming_colors[0])
        six_value_wave += Line(start=RIGHT*3, end=RIGHT *
                               3+UP, color=theming_colors[0])
        six_value_wave += Line(start=RIGHT*3+UP, end=RIGHT *
                               4+UP, color=theming_colors[0])
        six_value_wave += Line(start=RIGHT*4+UP,
                               end=RIGHT*4, color=theming_colors[0])
        six_value_wave.shift(DOWN*.49, LEFT*2.7).stretch(1.143, 1)

        volume_text_1 = Tex(r'Volume: 15\%').shift(RIGHT*4, UP*2)
        volume_text_2 = Tex(r'Volume: 66\%').shift(RIGHT*4, UP*2)
        volume_text_3 = Tex(r'Volume: 100\%').shift(RIGHT*4, UP*2)

        self.play(Create(six_value_wave), Write(volume_text_1))
        self.wait()
        self.play(six_value_wave.animate.stretch(3, 1),
                  volume_text_1.animate.become(volume_text_2))
        self.wait()
        self.play(six_value_wave.animate
                  .shift(DOWN*0.006).stretch(1.664, 1), volume_text_1.animate.become(volume_text_3))
        self.wait()

        sixteen_bit_values = MathTex(
            r'32767\\32766\\32765\\\vdots\\2\\1\\0\\-1\\\vdots\\-32766\\-32767\\-32768').scale(0.7).shift(LEFT*4)
        self.play(FadeOut(six_value_wave, volume_text_1),
                  Transform(six_values, sixteen_bit_values))
        self.remove(six_values)
        self.add(sixteen_bit_values)

        bit_depth = Text('Bit depth', **small_font_args).shift(RIGHT, UP)
        bit_depth_full = Text('Bit depth of 16', **
                              small_font_args).shift(RIGHT, UP)
        self.play(Write(bit_depth))
        _65536 = MathTex(r'65536').scale(1.2).shift(DOWN, RIGHT)
        power2 = MathTex(r'2^{16} = 65536').scale(1.2).shift(DOWN, RIGHT)
        self.play(Write(_65536))
        self.play(Transform(_65536, power2),
                  Transform(bit_depth, bit_depth_full))

        self.wait()


class Ending(Scene):
    def construct(self):
        self.next_section(skip_animations=False)

        def audio_function(i): return sin(i/1.3*PI)

        samples = list(map(audio_function, range(0, 12)))
        sample_coordsystem = Axes(
            x_range=(0, 11, 30), y_range=(-1, 1, 30), x_length=12, y_length=3, tips=False)
        sample_wave = sample_coordsystem.plot(
            audio_function, x_range=(0, 11, 0.01), color=theming_colors[0])

        samples_hex = Text(' '.join(map(lambda s: '{:04x}'.format(
            int(s*2**15) & 0xffff), samples)), color=theming_colors[0], **mono_font_args).shift(UP)
        samples_hex.font_size = DEFAULT_FONT_SIZE * 0.5
        sample_list_quantized = VGroup()
        sample_list = VGroup()
        sample_points = VGroup()
        sampling_lines = VGroup()
        dot_radius = DEFAULT_DOT_RADIUS*.8
        sample_spacing = 0.25 - dot_radius
        for i, sample in enumerate(samples):
            sample_list_quantized += MathTex(
                '{:05d}'.format(int(sample*2**15)), color=theming_colors[0]).scale(0.5)
            sample_list += MathTex('{:.3f}'.format(sample),
                                   color=theming_colors[0]).scale(0.6)
            sampling_lines += DashedLine(start=UP*3, end=DOWN*3,
                                         stroke_width=DEFAULT_STROKE_WIDTH*.5, color=GREY)
            sample_location = sample_coordsystem.c2p(i, sample)
            sample_points += Dot(sample_location,
                                 radius=dot_radius, color=theming_colors[0])
        for array in [sample_list, sample_list_quantized, sampling_lines]:
            for reference_element, element_to_move in zip(sample_points, array):
                element_to_move.move_to(reference_element, coor_mask=RIGHT)

        self.play(Write(samples_hex))
        self.wait()
        self.play(Transform(samples_hex, sample_list_quantized))
        self.remove(samples_hex)
        self.add(sample_list_quantized)
        self.wait()
        self.play(Transform(sample_list_quantized, sample_list))
        self.remove(sample_list_quantized)
        self.add(sample_list)
        self.wait()
        self.play(Transform(sample_list, sample_points),
                  Create(sampling_lines))
        self.remove(sample_list)
        self.add(sample_points)
        self.wait()
        self.play(FadeIn(sample_wave))
        self.wait()

        rate_brace = BraceBetweenPoints(sampling_lines[2].get_end(
        ), sampling_lines[3].get_end(), direction=DOWN, sharpness=1.5, buff=-SMALL_BUFF)
        rate_text = Text(r'Sample rate', **small_font_args)
        rate_text.font_size = 0.5*DEFAULT_FONT_SIZE
        rate_brace.put_at_tip(rate_text)
        self.play(Write(rate_text), Create(rate_brace))
        self.wait()

        pcm = Text('PCM (Pulse Code Modulation)', **small_font_args)
        self.play(Uncreate(rate_text), Uncreate(rate_brace), Write(pcm))
        self.wait()

        self.play(FadeOut(pcm, sample_wave, sample_points, sampling_lines))

        nextep = Text('Stay tuned for the other parts!',
                      **font_args).shift(UP*3)
        ep2 = Text('Episode 2:\nLossless Audio Compression',
                   **small_font_args).shift(LEFT*4)
        ep3 = Text('Episode 3:\nThe FLAC stream format',
                   **small_font_args).shift(RIGHT*4)
        ep3.font_size = 0.6 * DEFAULT_FONT_SIZE
        ep2.font_size = 0.6 * DEFAULT_FONT_SIZE

        self.play(Write(nextep, run_time=DEFAULT_ANIMATION_RUN_TIME*2),
                  Write(ep3, run_time=DEFAULT_ANIMATION_RUN_TIME*2),
                  Write(ep2, run_time=DEFAULT_ANIMATION_RUN_TIME*2))
        self.wait()


class WavePlayback(Animation):
    '''
    Plays a signal in a gain-time graph, progressing the signal through time. This is useful for e.g. showing an audio wave on a graph.

    Parameters
    ----------
    mobject
        The Mobject animated by this wave playback.
        Currently this animation only works for ParametricFunctions,
        as that has some known properties that make this animation work particularly well.
    function
        The function that is animated.
        This is not a normal x -> y function, but a (x, offset) -> y function which gets passed an offset.
        The x value is the one that was present in the original state of the Mobject, so it is the starting point.
        The offset is a time parameter, which increases as the function progresses.
        Simple implementations can just sum up `x + offset` as an input into a normal single-argument functions
        (this works well for trigonometric-like functions that mimic sound waves or radio transmissions),
        but for more complex implementations it is convenient to have specific knowledge about the actual position
        of the currently dealt-with point (x) versus the time progression (offset).
    function_range
        The range of the offset parameter, as a tuple of (start, end).
        This means that the function doesn't have to remap the offset parameter's range.
    coordinates
        A coordinate system in which the function is located.
        This is used to find out the "real" x and y values of points, which is what the function receives.
        You can also provide a pseudo-coordinate system that just provides the
        :meth:`manim.CoordinateSystem.p2c` and :meth:`manim.CoordinateSystem.c2p` conversion functions.
    '''

    def __init__(self, mobject: ParametricFunction, function: Callable[[float, float], float], function_range: Tuple[float, float], coordinates: CoordinateSystem, *args, **kwargs):
        self.name = 'WavePlayback'
        super().__init__(mobject, *args, **kwargs)
        self.internal_function = function
        self.function_start, self.function_end = function_range
        self.coordinates = coordinates

        if not isinstance(self.mobject, ParametricFunction):
            raise TypeError('WavePlayback only works on ParametricFunctions')

    def interpolate_submobject(self, submobject: Mobject, starting_submobject: Mobject, alpha: float):
        # Remap the alpha to the function's input parameter range
        function_input_value = alpha * \
            (self.function_end - self.function_start) + self.function_start

        def point_transformer(point: np.array) -> np.array:
            coordinate = self.coordinates.p2c(point)
            new_y = self.internal_function(coordinate[0], function_input_value)
            # FIXME: We could use the original point's z value here, but this animation is not intended for 3D anyways.
            transformed_point = self.coordinates.c2p(
                coordinate[0], new_y, 0)
            return transformed_point

        submobject.apply_function(point_transformer)
