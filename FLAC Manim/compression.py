# SPDX-License-Identifier: Apache-2.0

from itertools import chain, cycle
from pickle import FALSE
from tempfile import template
from typing import Tuple, Callable
from unittest import skip
from exp_golomb import alternate_encode_exp_golomb
from manim import *
from manim.animation.animation import DEFAULT_ANIMATION_RUN_TIME
from manim.animation.composition import DEFAULT_LAGGED_START_LAG_RATIO
from main import TRANSPARENT, font_args, small_font_args, mono_font_args, theming_colors
from exp_golomb import create_exp_golomb_table
import numpy as np
from random import Random, sample
from math import ceil, cos, factorial, floor, log2, sin, pi, asin, sqrt
from functools import partial
from scipy.io import wavfile
from scipy import fft
from cmath import phase as complex_angle

ARROW_BUFFER = MED_LARGE_BUFF*1.5


def hertz_labels(number: str | float, *args, **kwargs):
    number = r"{.2}\:\text{Hz}".format(number)
    return MathTex(number, *args, **kwargs)


class CustomParentScene(MovingCameraScene):
    def transform_between_equation_list(self, equations: list[MathTex], do_first_write: bool = True, transform_animation=TransformMatchingTex, additional_first_animation: list | None = None):
        '''Plays an animation that transforms between a series of equations, possibly writing the first equation. At the end only the last equation is visible.'''
        if additional_first_animation is None:
            additional_first_animation = []
        previous_equation: None | MathTex = None
        for equation in equations:
            if previous_equation is None:
                if do_first_write:
                    self.play(Write(equation), *additional_first_animation)
            else:
                self.play(transform_animation(previous_equation, equation))
            self.remove(previous_equation, equation)
            previous_equation = equation
            self.add(previous_equation)
            self.wait()


class Intro(CustomParentScene):
    def construct(self):
        terms = Paragraph(
            'PCM\nfrequencies & sound waves\nsamples & sampling\nbit depth', **small_font_args)
        self.play(Write(terms))
        self.wait()
        self.play(FadeOut(terms))

        from_last_episode = Text(
            'In the last episode...', **font_args).shift(UP*3)
        last_episode_rectangle = Rectangle(
            color=theming_colors[0], width=16, height=9).scale(0.5).shift(DOWN)
        self.play(Write(from_last_episode), Create(last_episode_rectangle))
        self.wait()


class LinearPredictiveCoding(CustomParentScene):
    def construct(self):
        template = TexTemplate()
        template.add_to_preamble(r'''\usepackage[english]{babel}
        \usepackage{csquotes}\usepackage{cancel}''')

        self.next_section(skip_animations=False)
        compression = Text('How Large Is Audio?', **font_args).shift(UP*2)
        self.play(Write(compression))
        wav = Text('WAV\n1.4Mb/s', **small_font_args).shift(DOWN*2)
        flac = Text('FLAC\n0.6Mb/s', **small_font_args).shift(DOWN*2)
        mp3 = Text('MP3\n0.15Mb/s', **small_font_args).shift(DOWN*2)
        textgroup = VGroup(wav, flac, mp3)
        textgroup.arrange(direction=RIGHT, buff=LARGE_BUFF)
        self.play(Write(wav))
        self.play(Write(flac))
        self.play(Write(mp3))
        self.wait()
        self.play(Unwrite(textgroup))
        self.wait()

        # FIXME: Paragraph has a bug where newline calculations are incorrect with ligatures
        sidebar_title = Paragraph(
            'The Journey of\nLossless Audio Compression', disable_ligatures=True, alignment='center', **small_font_args)
        # FIXME: OpenGLMobject can't handle multiple shift dimensions at once
        sidebar_title.shift(LEFT*5 + UP*3.5).scale(0.45)
        sidebar_separator = Rectangle(
            color=WHITE, height=15, width=8).shift(LEFT*7)
        sidebar_separator.set_fill(color=BLACK, opacity=0.6)
        self.play(Write(sidebar_separator), Write(sidebar_title),
                  compression.animate.shift(RIGHT*2).scale(0.8))
        self.wait()
        sidebar = VGroup(sidebar_separator, sidebar_title)
        self.play(FadeOut(compression), sidebar.animate.shift(LEFT*5))
        self.wait()
        self.next_section('Frequency space compression', skip_animations=False)

        (sample_rate, audio_data) = load_wav_to_f64("../yoshi_restored_6.wav")
        audio_points = []
        for i, sample in enumerate(audio_data[:1000]):
            audio_points.append(Dot(UP*sample*2 + LEFT*5 + RIGHT *
                                i*0.01, color=theming_colors[0], radius=0.03))
        audio_point_group = VGroup(*audio_points)
        description = Text("This is real 16-bit audio data",
                           **small_font_args).shift(DOWN*3)
        self.play(Write(audio_point_group), Write(description))
        self.wait()

        self.play(FadeOut(audio_point_group), FadeOut(description))
        self.wait()

        coordinates = Axes(x_range=[-0.5, 20, 20], y_range=[-1.1, 1.2, 1], x_length=10, y_length=6, axis_config={
            'include_ticks': False,
            'include_tip': True,
            'tip_width': 0.001,
            'tip_height': 0.001,
            'include_numbers': False,
            'exclude_origin_tick': False,
        })
        time_text = MathTex('t')
        time = coordinates.get_x_axis_label(time_text)
        sinwave = coordinates.plot(lambda x: sin(
            2*pi*x/20), x_range=(0, 20, 0.2), color=theming_colors[0])
        self.play(Create(coordinates), Create(sinwave), Create(time))
        self.wait()

        sin_formula = MathTex(r'{{f(t)}} = {{\text{sin} }} ( {{t}} )',
                              color=theming_colors[0]).shift(UP*3+RIGHT*2)
        self.play(Write(sin_formula))
        self.wait()
        general_sin_formula = MathTex(
            r'{{f(t)}} = a\cdot {{\text{sin} }} {{(}} 2\pi {{f}} \cdot {{t}} - {{\phi}} {{)}} {{+ o}}', color=theming_colors[0]).shift(UP*3+RIGHT*2.2)
        self.play(TransformMatchingTex(sin_formula, general_sin_formula))
        # FIXME: All the Transform animations have weird behavior in terms of which Mobjects stay visible. This is not documented in Manim, so for safety we manually ensure that the correct ones stay visible.
        self.remove(sin_formula, general_sin_formula)
        self.add(general_sin_formula)
        self.wait()

        explanation = Tex(r'''\item[$a$:] amplitude (\enquote{wave strength})
        \item[$f$:] frequency (wavelength is $\lambda = \frac{1}{f}$)
        \item[$\phi$:] phase offset (shift in x-direction)
        \item[$o$:] DC offset (shift in y-direction)''', tex_template=template, tex_environment=r'description').scale(0.6).next_to(general_sin_formula, DOWN).shift(RIGHT*0.5)
        self.play(Write(explanation))
        self.wait()

        section_size = 1000
        audiowave = coordinates.plot(lambda x: audio_data[int(
            x * (section_size/20))], x_range=(0, 20, 0.1), color=theming_colors[0])
        self.play(Transform(sinwave, audiowave), FadeOut(
            general_sin_formula), FadeOut(explanation))

        self.next_section('dft', skip_animations=False)
        # FIXME: There's no "frame" in OpenGL
        self.camera.frame.save_state()
        # DFT time!
        audiowave_freqs = fft.fft(audio_data[:section_size])
        audiowave_freqs /= np.max(audiowave_freqs) / 4
        freq_bins = fft.fftfreq(section_size, 1/sample_rate)
        # only use 6 evenly spaced frequencies
        selected_frequency_bins = np.array([0, 5, 24, 55, 100, 200])
        selected_frequencies = freq_bins[selected_frequency_bins]
        frequency_graphs = []
        for index, frequency_info in enumerate(zip(selected_frequency_bins, selected_frequencies)):
            frequency_bin_index, frequency = frequency_info
            complex_value: complex = audiowave_freqs[frequency_bin_index]
            phase, amplitude = complex_angle(complex_value), abs(complex_value)
            def function(x): return sin(2*pi*frequency *
                                        x/(section_size*10) + phase) * amplitude
            frequency_graph = Axes(x_range=[-0.1, 10, 1], y_range=[-1.1, 1.2, 1], x_length=5, y_length=1, axis_config={
                'include_ticks': True,
                'include_tip': False,
                'tip_width': 20,
                'tip_height': 20,
                'include_numbers': True,
                'exclude_origin_tick': False,
                'font_size': DEFAULT_FONT_SIZE*0.4,
            }).shift(UP*3 + DOWN*index*1.1 + RIGHT*12)
            frequency_function = frequency_graph.plot(function, x_range=(
                0, 10, 0.2), color=theming_colors[1])
            arrow = Arrow(start=coordinates.get_right()+DOWN*(index-len(selected_frequencies)/2)*.2,
                          end=frequency_graph.get_left(), buff=ARROW_BUFFER)
            frequency_graphs.append(frequency_graph)
            frequency_graphs.append(frequency_function)
            frequency_graphs.append(arrow)
        frequency_graphs = VGroup(*frequency_graphs)
        self.play(self.camera.frame.animate.shift(RIGHT*5).scale(1.5))
        self.play(Write(frequency_graphs))
        self.wait()

        # The above graphs transform into the full DFT graph
        dft_coordinates = Axes(x_range=[0, 350, 100], y_range=[0.01, 0.4, 1], x_length=7, y_length=2, axis_config={
            'include_ticks': True,
            'include_tip': True,
            'tip_width': 0.0004,
            'tip_height': 0.0004,
            'include_numbers': True,
            'label_constructor': hertz_labels,
            'exclude_origin_tick': False,
        }, y_axis_config={'include_numbers': False}).next_to(frequency_graphs[len(frequency_graphs)//2], direction=ORIGIN)
        # Interpolating this FFT graph for some reason yields infinite/NaN values, so just don't interpolate.
        dft_function = dft_coordinates.plot(lambda x: sqrt(sqrt(abs(audiowave_freqs[min(int(
            x), section_size-1)])+1))-0.9, x_range=(0, 350, 1), color=theming_colors[1], use_smoothing=False)
        dft_arrow = Arrow(start=coordinates.get_right(),
                          end=dft_coordinates.get_left(), buff=ARROW_BUFFER)
        dft_text = Text('Discrete Fourier Transform', **
                        small_font_args).next_to(dft_coordinates, direction=UP).shift(UP*1.5, LEFT)
        dft_group = VGroup(dft_coordinates, dft_function, dft_arrow)
        self.play(LaggedStart(
            Transform(frequency_graphs, dft_group), Write(dft_text)))
        self.remove(frequency_graphs, dft_group)
        self.add(dft_group)
        self.wait()

        self.bring_to_front(sidebar)
        idea1 = Paragraph(
            'Sine waves?\n—> MP3', disable_ligatures=True, alignment='center', **small_font_args).shift(LEFT*3).align_to(sidebar_title, direction=UP, alignment_vect=UP).shift(DOWN*.5).scale(0.4)
        self.play(sidebar.animate.shift(RIGHT*7))
        self.play(Write(idea1))
        sidebar.add(idea1)
        self.wait()

        too_many_freqs = Text('Over 44,000 frequencies!', **small_font_args).next_to(
            dft_coordinates, direction=DOWN).shift(DOWN)
        too_many_freqs_arrow = Arrow(start=too_many_freqs.get_top(),
                                     end=dft_coordinates.get_bottom(), stroke_width=DEFAULT_STROKE_WIDTH)
        self.play(Write(too_many_freqs), Write(too_many_freqs_arrow))
        self.wait()

        self.play(sidebar.animate.shift(LEFT*7))

        self.wait()
        self.play(LaggedStart(
            AnimationGroup(
                FadeOut(dft_group),
                FadeOut(dft_text),
                FadeOut(too_many_freqs),
                FadeOut(too_many_freqs_arrow)),
            self.camera.frame.animate.restore()))

        # EXCURSION 1 (see below)
        self.wait()
        self.remove(coordinates, audiowave, sinwave, time, time_text)

        self.next_section('discretization', skip_animations=False)

        functions = MathTex(
            r'\Gamma\\\text{sinc}\\\text{arctan}\\\text{ln}\\\zeta\\\text{Im}\\\vdots')
        self.play(Write(functions))
        self.wait()

        polynomials = MathTex(
            r'a_0x^0 + a_1x^1 + a_2x^2 + a_3x^3 + \dots').shift(RIGHT*3)
        self.play(LaggedStart(functions.animate.shift(
            LEFT*4), Write(polynomials), lag_ratio=0.6))
        self.wait()
        self.play(FadeOut(functions, polynomials))
        self.wait()

        coordinates = NumberPlane(
            x_range=(-2.2*PI, 2.2*PI, PI/2),
            y_range=(-4, 4, 1),
            background_line_style={
                "stroke_color": theming_colors[1],
                "stroke_opacity": 0.4
            })
        taylored_sine = coordinates.plot(
            sin, color=theming_colors[0])
        sine_function_name = MathTex(r'f(x) = \sin(x)').scale(
            0.9).shift(RIGHT*4.5 + DOWN*2)
        self.play(Create(
            coordinates, run_time=DEFAULT_ANIMATION_RUN_TIME*3))
        self.play(Create(taylored_sine), Write(sine_function_name))
        self.wait()

        x0 = PI/4
        x0_label = coordinates.get_T_label(
            x_val=x0, graph=taylored_sine, label=MathTex(r'x_0').scale(.7), triangle_size=0.15, line_color=theming_colors[-1])
        self.play(Create(x0_label))
        self.wait()

        taylor_terms = [
            r"f(x_0)",
            r"+ f'(x_0) (x - x_0)",
            r"+ \frac{f''(x_0)}{2} (x - x_0)^2",
            r"+ \frac{f'''(x_0)}{6} (x - x_0)^3",
        ]

        taylor_labels = [r'T(x) =']
        previous_taylor_term = previous_taylor_expansion_graph = previous_backing = None

        for degree, label in enumerate(taylor_terms):
            taylor_labels.append(label)
            taylor_expansion = partial(taylor_for_sin, degree+1, x0)
            taylor_expansion_graph = coordinates.plot(
                taylor_expansion, color=theming_colors[2])
            taylor_term = MathTex(
                *taylor_labels).scale(0.7).shift(UP*2.5).align_to(LEFT*2*PI, direction=LEFT)
            taylor_term_backing = BackgroundRectangle(
                taylor_term, fill_opacity=0.8)
            if previous_taylor_term is None:
                self.play(FadeIn(taylor_term_backing), Write(
                    taylor_term), Create(taylor_expansion_graph))
            else:
                self.play(TransformMatchingTex(previous_taylor_term, taylor_term), Transform(
                    previous_taylor_expansion_graph, taylor_expansion_graph), Transform(previous_backing, taylor_term_backing))
            self.wait()
            self.remove(taylor_term, taylor_expansion_graph,
                        previous_taylor_expansion_graph, previous_taylor_term)
            previous_backing, previous_taylor_term, previous_taylor_expansion_graph = taylor_term_backing, taylor_term, taylor_expansion_graph
            self.add(previous_backing, previous_taylor_term,
                     previous_taylor_expansion_graph)
        self.remove(previous_backing)

        definition_text = r"T(f, x_0, x)\; &= \sum_{n=0}^{\infty}\frac{f^{(n)}(x_0)}{n!}(x-x_0)^n \\&= f(x_0) + f'(x_0) (x - x_0) + \frac{f''(x_0)}{2} (x - x_0)^2 + \frac{f'''(x_0)}{6} (x - x_0)^3 + \dots"

        taylor_definition = MathTex(definition_text).scale(0.75)
        self.play(FadeOut(coordinates, sine_function_name, x0_label), Uncreate(
            taylored_sine), Uncreate(previous_taylor_expansion_graph), Transform(previous_taylor_term, taylor_definition))
        self.remove(previous_taylor_term, taylor_definition)
        self.add(taylor_definition)
        self.wait()

        sin_cos_taylor = MathTex(
            definition_text, r"\\\sin(x) \;&= \sum_{n=0}^{\infty} \frac{(-1)^n}{(2n+1)!}x^{2n+1}", r"\\\cos(x) \;&= \sum_{n=0}^{\infty} \frac{(-1)^n}{(2n)!}x^{2n}").scale(0.75)
        self.play(TransformMatchingTex(taylor_definition, sin_cos_taylor))
        self.remove(taylor_definition, sin_cos_taylor)
        self.add(sin_cos_taylor)
        self.wait()

        taylor_title = Text('Taylor Expansion', **font_args).shift(DOWN*3)
        self.play(Write(taylor_title))
        self.wait()

        self.bring_to_front(sidebar)
        idea2 = Paragraph(
            'Polynomial approximation', disable_ligatures=True, alignment='center', **small_font_args).shift(LEFT*5).align_to(idea1, direction=UP, alignment_vect=UP).shift(DOWN*.5).scale(0.4)
        self.play(sidebar.animate.shift(RIGHT*5))
        self.wait()
        self.play(Write(idea2))
        sidebar.add(idea2)
        self.wait()
        self.play(sidebar.animate.shift(LEFT*5),
                  FadeOut(taylor_title), FadeOut(sin_cos_taylor))
        self.wait()

        self.next_section('discretization - recursive', skip_animations=False)
        # NOTE: This section was removed late in editing as it is factually correct but irrelevant.
        # The section remains in the rendered animation to not throw off the timing of later parts; the DirectPolynomialProblemInsert scene replaces its contents.
        calculation_texts = [
            r'& 1000 \text{ samples}', r'\cdot (4 \text{ multiplications} + 5 \text{ additions})', r'\\&= 9,\!000 \text{ arithmetic ops}', r'\\&\text{for } 23 \text{ ms} / 1 \text{ channel!}']

        calculation_equations = [MathTex(
            *(calculation_texts[:i])).scale(0.9).align_to(LEFT*5, direction=LEFT) for i in range(1, len(calculation_texts)+1)]
        self.transform_between_equation_list(calculation_equations)

        # FIXME: The tex split points are chosen carefully so the entire equation renders correctly (Manim TeX splitting bug).
        functofunc = MathTex(
            r'p({{t}}) {{\approx}} f ({{t}}) \quad\quad {{p}} : \mathbb{R} {{\to}} \mathbb{R}').align_to(RIGHT*3, direction=RIGHT)
        signaltosignal = MathTex(
            r's[{{t}}{{]}} {{\approx}} {{f}}[{{t}}] \quad\quad {{s}} {{:}} \mathbb{Z} {{\to}} \mathbb{Z}').align_to(RIGHT*3, direction=RIGHT)
        signaltosignal_recursive = MathTex(
            r'{{s[}}{{t}}{{]}} = a_1 {{s}}[{{t}}-1] + a_2 s[t-2] + \dots')
        equations = [functofunc, signaltosignal, signaltosignal_recursive]
        self.play(LaggedStart(
            FadeOut(calculation_equations[-1]), Write(equations[0])))
        self.transform_between_equation_list(equations, do_first_write=False)

        self.bring_to_front(sidebar)
        idea3 = Paragraph(
            'Recursive signal definition', disable_ligatures=True, alignment='center', **small_font_args).shift(LEFT*5).align_to(idea2, direction=UP, alignment_vect=UP).shift(DOWN*.5).scale(0.4)
        self.play(sidebar.animate.shift(RIGHT*5))
        self.wait()
        self.play(Write(idea3))
        sidebar.add(idea3)
        self.wait()
        self.play(sidebar.animate.shift(LEFT*5))
        self.wait()

        predictor_formulas = MathTex(r's[t] &= s[t-1]\\',
                                     r's[t] &= 2s[t-1] - s[t-2]\\',
                                     r's[t] &= 3s[t-1] - 3s[t-2] + s[t-3]\\',
                                     r's[t] &= 4s[t-1] - 6s[t-2] + 4s[t-3] - s[t-4]').scale(0.9)
        self.play(Transform(equations[-1], predictor_formulas))
        self.remove(equations[-1], predictor_formulas)
        self.add(predictor_formulas)
        self.wait()

        order_2_predictor = MathTex(r's[t] &= 2s[t-1] - s[t-2]\\')
        self.play(TransformMatchingTex(predictor_formulas, order_2_predictor))
        self.remove(predictor_formulas, order_2_predictor)
        self.add(order_2_predictor)
        self.wait()

        self.play(Circumscribe(
            order_2_predictor[0][5], color=theming_colors[0], run_time=DEFAULT_ANIMATION_RUN_TIME*.5))
        self.play(Circumscribe(VGroup(
            *(order_2_predictor[0][6:12])), color=theming_colors[0], run_time=DEFAULT_ANIMATION_RUN_TIME*.5))
        self.play(Circumscribe(
            order_2_predictor[0][12], color=theming_colors[0], run_time=DEFAULT_ANIMATION_RUN_TIME*.5))
        self.play(Circumscribe(VGroup(
            *(order_2_predictor[0][13:])), color=theming_colors[0], run_time=DEFAULT_ANIMATION_RUN_TIME*.5))
        self.wait()

        rng = Random(1080)
        equation_template = r'{solution} &= 2 \cdot \left({s_1}\right) - \left({s_2}\right)'

        prediction_example_start_points = list(map(lambda i: np.floor(audio_data[i:i+2] * 2**15).astype(int), (
            rng.randint(0, 200) for _ in range(4))))
        prediction_example_equations = MathTex(r'\\'.join([equation_template.format(
            solution=2*points[1] - points[0], s_1=points[1], s_2=points[0]) for points in prediction_example_start_points]), color=theming_colors[0]
        ).scale(0.9).align_to(LEFT*4, direction=LEFT).shift(DOWN*2)
        self.play(Write(prediction_example_equations))

        prediction_example_functions = [
            {"a_0": points[0], "a_1": points[1] - points[0]} for points in prediction_example_start_points]
        linear_function_template = r's[t] &= {a_1}t + {a_0}'

        prediction_example_function_equations = MathTex(r'\\'.join([linear_function_template.format(
            **coefficients) for coefficients in prediction_example_functions]), color=theming_colors[0]
        ).scale(0.9).align_to(LEFT*4, direction=LEFT).shift(DOWN*2)
        self.play(Transform(prediction_example_equations,
                  prediction_example_function_equations))
        self.remove(prediction_example_equations,
                    prediction_example_function_equations)
        self.add(prediction_example_function_equations)
        self.wait()

        warmup_definition = MathTex(r's[0] &:= w_0\\s[1] &:= w_1\\\vdots').align_to(
            RIGHT*4, direction=RIGHT).shift(DOWN*2)
        self.play(Write(warmup_definition))
        self.wait()

        predictor_warmup_counts = MathTex(
            r's[0]\\s[0],s[1]\\s[0],s[1],s[2]\\s[0],s[1],s[2],s[3]').shift(RIGHT*5).scale(0.9)
        predictor_formulas.shift(LEFT*2)
        self.play(LaggedStart(AnimationGroup(FadeOut(warmup_definition), FadeOut(prediction_example_function_equations)),
                  TransformMatchingTex(order_2_predictor, predictor_formulas), Write(predictor_warmup_counts), lag_ratio=0.4))
        self.remove(order_2_predictor, predictor_formulas)
        self.add(predictor_formulas)
        self.wait()
        self.play(FadeOut(predictor_formulas),
                  FadeOut(predictor_warmup_counts))
        self.remove(predictor_formulas)
        self.wait()

        orders = MathTex(r'1\\2\\3\\4').scale(0.75)
        order_title = Text('Order', **small_font_args)
        # FIXME: We need to recreate the predictor formulas as it was deleted beforehand
        predictor_formulas = MathTex(r's[t] &= s[t-1]\\',
                                     r's[t] &= 2s[t-1] - s[t-2]\\',
                                     r's[t] &= 3s[t-1] - 3s[t-2] + s[t-3]\\',
                                     r's[t] &= 4s[t-1] - 6s[t-2] + 4s[t-3] - s[t-4]').scale(0.75)
        predictor_title = Text('Predictor', **small_font_args)
        warmup_title = Text('Warm-up samples', **small_font_args)
        order_title.font_size = predictor_title.font_size = warmup_title.font_size = DEFAULT_FONT_SIZE*.6
        warmup_sample_count = MathTex(
            r's[0]\\s[{0,1}]\\s[{0, 1, 2}]\\s[{0,1,2,3}]').scale(0.75)

        lpc_order_table = MobjectTable([[order_title, predictor_title, warmup_title],
                                        [orders, predictor_formulas, warmup_sample_count]],
                                       v_buff=0.4, h_buff=0.1, line_config={"color": TRANSPARENT}, add_background_rectangles_to_entries=False, include_background_rectangle=False)

        self.play(Write(orders), Write(order_title))
        self.wait()
        self.play(Write(predictor_formulas), Write(predictor_title))
        self.wait()
        self.play(Write(warmup_sample_count), Write(warmup_title))
        self.wait()

        lpc_text = Text('Linear Predictive Coding (LPC)',
                        **font_args).shift(DOWN*3)
        self.play(Write(lpc_text))

        self.bring_to_front(sidebar)
        self.bring_to_front(lpc_text)
        idea4 = Paragraph(
            'Linear Predictive Coding', disable_ligatures=True, alignment='center', **small_font_args).shift(LEFT*5).align_to(idea3, direction=UP, alignment_vect=UP).shift(DOWN*.5).scale(0.4)
        self.play(Transform(lpc_text, idea4), sidebar.animate.shift(RIGHT*5))
        self.remove(idea4, lpc_text)
        self.add(idea4)
        sidebar.add(idea4)
        self.wait()
        self.play(sidebar.animate.shift(
            LEFT*5))
        self.wait()

        # EXCURSION 2 (see below)
        self.next_section('end', skip_animations=False)
        generic_lpc = MathTex(
            r's[t] &= a_1s[t-1] + a_2s[t-2] + a_3s[t-3] + a_4s[t-4] + a_5s[t-5] + a_6s[t-6] + a_7s[t-7]',
            r' + \dots').scale(0.6).shift(DOWN)
        generic_lpc_expanded = MathTex(
            r's[t] &= a_1s[t-1] + a_2s[t-2] + a_3s[t-3] + a_4s[t-4] + a_5s[t-5] + a_6s[t-6] + a_7s[t-7]',
            r'\\&+ a_8s[t-8] + a_9s[t-9] + a_{10}s[t-10] + a_{11}s[t-11] + a_{12}s[t-12] + a_{13}s[t-13] + a_{14}s[t-14]',
            r'\\& + a_{15}s[t-15] + a_{16}s[t-16] + a_{17}s[t-17] + a_{18}s[t-18] + a_{19}s[t-19] + a_{20}s[t-20] + a_{21}s[t-21]',
            r'\\& + a_{22}s[t-22] + a_{23}s[t-23] + a_{24}s[t-24] + a_{25}s[t-25] + a_{26}s[t-26] + a_{27}s[t-27] + a_{28}s[t-28]',
            r'\\& + a_{29}s[t-29] + a_{30}s[t-30] + a_{31}s[t-31] + a_{32}s[t-32]',
        ).scale(0.6)
        self.play(Write(generic_lpc), lpc_order_table.animate.shift(UP*2))
        self.wait()
        self.play(
            TransformMatchingTex(generic_lpc, generic_lpc_expanded),
            FadeOut(order_title, predictor_title, warmup_title,
                    orders, predictor_formulas, warmup_sample_count))
        self.remove(generic_lpc)
        self.add(generic_lpc_expanded)
        self.wait()

        # identical RNG to the polynomial predictor excursion: same samples that are good for demonstration
        rng = Random(2)
        start_index = rng.randrange(len(audio_data)-5)
        samples = list(map(lambda s: int(s*2**15),
                           # hope that this doesn't index out of bounds :^)
                       audio_data[start_index-3:start_index+5]))
        sample_grid = NumberPlane(
            x_range=(-1, len(samples)+1, 1),
            y_range=(min(samples)-1500, max(samples)+10500, (2**15)/10),
            x_length=16,
            y_length=8.1,
            axis_config={
                'include_numbers': False,
                'stroke_width': 0,
                'stroke_color': theming_colors[1],
                'stroke_opacity': 0.4,
            },
            background_line_style={
                'stroke_color': theming_colors[1],
                'stroke_opacity': 0.4,
            },
        )
        sample_points = VGroup()
        for i, sample in enumerate(samples):
            sample_points.add(Dot(sample_grid.coords_to_point(
                i, sample), color=theming_colors[0]))

        def line_function(x: int | float):
            dy = samples[-1] - samples[-2]
            return dy * x - dy * (len(samples)-1)

        predictor_line = Line(color=theming_colors[2])
        predictor_line.put_start_and_end_on(
            start=sample_grid.coords_to_point(-1, line_function(-1)),
            end=sample_grid.coords_to_point(
                len(samples)+1, line_function(len(samples)+1))
        )
        self.play(
            Create(sample_grid),
            Create(sample_points),
            Create(predictor_line),
            FadeOut(generic_lpc_expanded))
        self.wait()

        predictor_subtraction = []
        for i, point in enumerate(sample_points):
            predictor_subtraction.append(point.animate.shift(
                DOWN*sample_grid.coords_to_point(0, line_function(i))[1]).set_stroke_color(theming_colors[4]).set_color(theming_colors[4]))
        predictor_subtraction = AnimationGroup(*predictor_subtraction)
        self.play(predictor_line.animate.put_start_and_end_on(
            start=sample_grid.coords_to_point(-1, 0),
            end=sample_grid.coords_to_point(len(samples)+1, 0)).set_opacity(0.5),
            predictor_subtraction,
        )
        self.wait()

        residual = Text('Residual', **font_args).shift(DOWN*2)
        self.play(Write(residual),
                  sample_points.animate.set_color(theming_colors[4]))
        self.wait()
        # smooth transition to Residual scene
        self.play(residual.animate.shift(UP*5),
                  FadeOut(sample_grid, sample_points, predictor_line))
        self.wait()


def taylor_for_sin(degree: int, x0: float, x: float) -> float:
    '''Implements Taylor polynomial expansion for sine up to a given degree with a starting point x0.'''
    derivatives = [sin, cos, lambda t: -sin(t), lambda t: -cos(t)]

    return sum(map(lambda n: derivatives[n % 4](x0) / factorial(n) * (x-x0)**n, range(degree)))


class DirectPolynomialProblemInsert(CustomParentScene):
    def construct(self):
        (_, audio_data) = load_wav_to_f64("../yoshi_restored_6.wav")
        rng = Random(337)
        start_index = rng.randrange(len(audio_data)-400)
        samples = np.array(list(map(lambda s: int(s*2**15),
                                    audio_data[start_index:start_index+400])))
        sample_grid = NumberPlane(
            x_range=(-10, len(samples)-1, 50),
            y_range=(min(samples)-150, max(samples)+500, (2**15)/10),
            x_length=16,
            y_length=8.1,
            axis_config={
                'include_numbers': False,
                'stroke_width': 0,
                'stroke_color': theming_colors[1],
                'stroke_opacity': 0.4,
            },
            background_line_style={
                'stroke_color': theming_colors[1],
                'stroke_opacity': 0.4,
            },
        )

        audio_wave = sample_grid.plot(
            lambda x: samples[int(x)], color=theming_colors[0])
        question_marks = Tex(
            r'$T$ with degree $>400$ !', color=theming_colors[2]).scale(1.2).shift(DOWN+RIGHT)
        self.play(Create(sample_grid))
        self.play(Create(audio_wave))
        self.wait()
        self.play(Write(question_marks))
        self.wait()


class ExcursionPolynomials(CustomParentScene):
    def construct(self):
        coordinates = NumberPlane(
            y_range=(-2, 8, 1),
            background_line_style={
                "stroke_color": theming_colors[1],
                "stroke_opacity": 0.4
            })
        self.add(coordinates)
        self.wait()

        monomial_range = range(0, 6)
        monomials = [coordinates.plot(
            lambda x: x**n, use_vectorized=True, color=theming_colors[0]) for n in monomial_range]
        monomial_labels = [MathTex(f'x^{{{n}}}').scale(
            2).shift(RIGHT*4 + UP*3) for n in monomial_range]

        previous_monomial = previous_label = None

        for monomial, label in zip(monomials, monomial_labels):
            if previous_monomial is None:
                self.play(Write(label), Create(monomial),
                          run_time=DEFAULT_ANIMATION_RUN_TIME*.5)
            else:
                # FIXME: Crashes in OpenGL with "list object has no attribute reshape"
                self.play(Transform(previous_label, label), Transform(
                    previous_monomial, monomial), run_time=DEFAULT_ANIMATION_RUN_TIME*.5)

            self.remove(monomial, label)
            # FIXME: OpenGL breaks if you try to remove a None from a scene, but Cairo just ignores it.
            if previous_monomial is not None:
                self.remove(previous_monomial, previous_label)
            previous_label, previous_monomial = label, monomial
            self.add(previous_monomial, previous_label)

        self.play(LaggedStart(Uncreate(previous_monomial, run_time=DEFAULT_ANIMATION_RUN_TIME*6), Unwrite(
            previous_label), run_time=DEFAULT_ANIMATION_RUN_TIME*6, lag_ratio=0.4))

        polynomials = [coordinates.plot(
            lambda x: 2*x**2 - 5*x+7, use_vectorized=True, color=theming_colors[0]), coordinates.plot(
            lambda x: -1*x**3 + 2*x**2 + x, use_vectorized=True, color=theming_colors[0]), coordinates.plot(
            lambda x: 1*x**5 + 2*x**4 - 3*x**3 - 5*x**2 + 3*x + 4, use_vectorized=True, color=theming_colors[0])]
        polynomial_labels = [
            MathTex(r'2x^2-5x+7'), MathTex(r'-x^3+2x^2+x'), MathTex('   1   ', r'x^', r'{5}', '+', '   2   ', 'x^4', '-3', 'x^3', '-5', 'x^2+', '   3   ', 'x', '^1', '+', '   4   ', 'x^0').shift(LEFT*1.89)]
        polynomial_labels[-1].set_color_by_tex(
            'x^0', TRANSPARENT).set_color_by_tex(
            '^1', TRANSPARENT).set_color_by_tex('   1   ', TRANSPARENT)

        previous_polynomial = previous_label = None

        for polynomial, label in zip(polynomials, polynomial_labels):
            label = label.scale(1.3).shift(RIGHT*4 + UP*3 + DOWN*0.35)
            if previous_polynomial is None:
                self.play(Write(label), Create(
                    polynomial, run_time=DEFAULT_ANIMATION_RUN_TIME*2))
            else:
                self.play(Transform(previous_label, label),
                          Transform(previous_polynomial, polynomial))

            self.remove(polynomial, previous_polynomial, label, previous_label)
            previous_label, previous_polynomial = label, polynomial
            self.add(previous_polynomial, previous_label)
            self.wait()

        self.play(previous_label.animate.set_color_by_tex(
            r'{5}', theming_colors[2]).set_color_by_tex('^1', WHITE))
        self.wait()
        self.play(previous_label.animate.set_color_by_tex('   1   ', theming_colors[3]).set_color_by_tex('   2   ', theming_colors[3]).set_color_by_tex('-3', theming_colors[3]).set_color_by_tex(
            '-5', theming_colors[3]).set_color_by_tex('   3   ', theming_colors[3]).set_color_by_tex('   4   ', theming_colors[3]))
        self.wait()
        self.play(previous_label.animate.set_color_by_tex('x^0', WHITE))
        self.wait()

        self.play(Unwrite(previous_label))
        self.wait()


class ExcursionPolynomialPrediction(CustomParentScene):
    def construct(self):
        template = TexTemplate()
        template.add_to_preamble(r'''\usepackage[english]{babel}
        \usepackage{csquotes}\usepackage{cancel}''')

        self.next_section(skip_animations=False)
        predictor_formulas = MathTex(r's[t] &= s[t-1]\\',
                                     r's[t] &= 2s[t-1] - s[t-2]\\',
                                     r's[t] &= 3s[t-1] - 3s[t-2] + s[t-3]\\',
                                     r's[t] &= 4s[t-1] - 6s[t-2] + 4s[t-3] - s[t-4]').scale(0.75)
        self.play(FadeIn(predictor_formulas))
        self.wait()

        lpc_implies_polynomial = MathTex(
            r's[t] = a_1 s[t-1] + a_2 s[t-2] + a_3 s[t-3] + \dots\\',
            r'\Downarrow\\',
            r'\text{Polynomial function}',
            tex_template=template, tex_environment='gather*').scale(0.9).shift(DOWN)
        lpc_implies_not_polynomial = MathTex(
            r's[t] = a_1 s[t-1] + a_2 s[t-2] + a_3 s[t-3] + \dots\\',
            r'\xcancel{ \Downarrow }\\',
            r'\text{Polynomial function}',
            tex_template=template, tex_environment='gather*').scale(0.9).shift(DOWN)
        self.play(predictor_formulas.animate.shift(
            UP*2), Write(lpc_implies_polynomial))
        self.wait()
        self.play(TransformMatchingTex(
            lpc_implies_polynomial, lpc_implies_not_polynomial))
        self.remove(lpc_implies_not_polynomial, lpc_implies_polynomial)
        self.add(lpc_implies_not_polynomial)
        self.wait()

        polynomials = MathTex(r's[t] &= a_0\\',
                              r's[t] &= a_1t + a_0\\',
                              r's[t] &= a_2t^2 + a_1t+a_0\\',
                              r's[t] &= a_3t^3 + a_2t^2 + a_1t+a_0').scale(0.75).shift(UP*2)
        self.play(Transform(predictor_formulas, polynomials))
        self.remove(predictor_formulas, polynomials)
        self.add(polynomials)
        self.wait()

        self.play(FadeOut(polynomials, lpc_implies_not_polynomial))
        self.wait()

        self.next_section("geometric intuition", skip_animations=False)

        (_, audio_data) = load_wav_to_f64("../yoshi_restored_6.wav")
        rng = Random(2)
        start_index = rng.randrange(len(audio_data)-5)
        samples = list(map(lambda s: int(s*2**15),
                       audio_data[start_index:start_index+5]))
        sample_texts = VGroup()
        for i, sample in enumerate(samples):
            sample_texts.add(
                MathTex(str(sample), color=theming_colors[0]).scale(0.6).shift(LEFT*5 + i*RIGHT*1.5))
        self.play(Write(sample_texts))
        self.wait()
        next_sample = MathTex(r's[t] = \: ?').scale(
            0.6).shift(LEFT*5 + RIGHT*1.5*len(samples))
        self.play(Write(next_sample))
        self.wait()

        # FIXME: This function gets passed digits individually. That is ridiculous but Manim.
        def the_t_label_constructor_for_an_axis(the_t_label_number: str | int, *args, **kwargs):
            # for some reason this happens with -1
            if the_t_label_number == '-':
                return MathTex(the_t_label_number, *args, **kwargs)
            the_t_label_number = int(the_t_label_number)
            if the_t_label_number == len(samples):
                text = 't'
            else:
                text = 't-{}'.format(len(samples)-the_t_label_number)
            return MathTex(text, *args, **kwargs)

        grid = NumberPlane(
            x_range=(-1, len(samples)+1, 1),
            y_range=(min(samples)-1500, max(samples)+10500, (2**15)/10),
            x_length=16,
            y_length=8.1,
            axis_config={
                'include_numbers': False,
                'stroke_width': 0,
                'stroke_color': theming_colors[1],
                'stroke_opacity': 0.4,
            },
            x_axis_config={
                'include_numbers': True,
                'label_constructor': the_t_label_constructor_for_an_axis,
            },
            background_line_style={
                'stroke_color': theming_colors[1],
                'stroke_opacity': 0.4,
            },
        )
        sample_points = VGroup()
        for i, sample in enumerate(samples):
            sample_points.add(Dot(grid.coords_to_point(
                i, sample), color=theming_colors[0]))
        next_sample_location_line = Line(
            start=grid.coords_to_point(5, -2**15),
            end=grid.coords_to_point(5, 2**15))
        self.play(Write(grid), Transform(sample_texts, sample_points),
                  Transform(next_sample, next_sample_location_line))
        self.remove(next_sample, next_sample_location_line, sample_texts)
        self.bring_to_back(grid)
        self.add(next_sample_location_line)
        self.add_foreground_mobjects(sample_points)
        self.wait()

        random_line_parameters = [
            (8000, -1000),
            (0, 1200),
            (-10000, 8000),
            (-9300, 3000),
            (-3400, -700),
            (5600, 150),
        ]

        predictor_line = Line(color=theming_colors[2])
        intersection_point = Cross(
            stroke_color=theming_colors[0], stroke_width=4, scale_factor=0.15)
        intersection_label = MathTex('s[t]').scale(0.6)
        b_tracker, a_tracker = ValueTracker(), ValueTracker()

        def line_function(x: float | int):
            # Simulates the line's function based on the current state of the above value trackers.
            return a_tracker.get_value() * x + b_tracker.get_value()

        def intersection_updater(point: Cross):
            point.move_to(grid.coords_to_point(
                len(samples), line_function(len(samples))))

        def intersection_label_updater(label: MathTex):
            label.move_to(grid.coords_to_point(
                len(samples), line_function(len(samples)))).shift(UP*.5+RIGHT*.5)

        def line_updater(line: Line):
            line.put_start_and_end_on(
                start=grid.coords_to_point(-1, line_function(-1)),
                end=grid.coords_to_point(
                    len(samples)+1, line_function(len(samples)+1))
            )

        intersection_point.add_updater(intersection_updater)
        intersection_label.add_updater(intersection_label_updater)
        predictor_line.add_updater(line_updater)

        self.play(Create(predictor_line), Create(
            intersection_point), Write(intersection_label))
        for b, a in random_line_parameters:
            self.play(a_tracker.animate.set_value(a),
                      b_tracker.animate.set_value(b))
            self.wait()

        line_formula = MathTex('ax+b').scale(0.9).shift(UP*3+LEFT*4)

        def line_formula_updater(formula: MathTex):
            formula.become(MathTex('{:.1f}x{:+.1f}'.format(a_tracker.get_value(),
                           b_tracker.get_value())).scale(0.9).shift(UP*3+LEFT*4))

        line_formula_updater(line_formula)
        self.play(Write(line_formula))
        self.wait()
        line_formula.add_updater(line_formula_updater)

        for b, a in random_line_parameters:
            self.play(a_tracker.animate.set_value(a),
                      b_tracker.animate.set_value(b))
        self.wait()

        line_formula.suspend_updating()
        self.play(Unwrite(line_formula), Uncreate(intersection_point),
                  Uncreate(predictor_line), FadeOut(intersection_label))
        line_formula.resume_updating()
        self.wait()

        predictor_step1 = MathTex(
            r'{{s[t] =}}{{s[t-1]}}').scale(0.9).shift(UP*3).align_to(LEFT*4, direction=LEFT)
        a_tracker.set_value(0)
        b_tracker.set_value(samples[-1])

        # FIXME: For some reason the intersection point can't be re-created once it has disappeared.
        intersection_point = Cross(
            stroke_color=theming_colors[0], stroke_width=4, scale_factor=0.15)
        intersection_point.add_updater(intersection_updater)
        self.play(Write(predictor_step1),
                  Flash(sample_points[-1], color=theming_colors[2]),
                  Create(predictor_line),
                  Create(intersection_point))
        self.wait()

        slope_end_point = grid.coords_to_point(len(samples)-1, samples[-1])
        slope_start_point = grid.coords_to_point(len(samples)-2, samples[-2])
        slope_corner = grid.coords_to_point(len(samples)-1, samples[-2])
        slope_lines = VMobject(stroke_color=theming_colors[3])
        slope_lines.start_new_path(slope_start_point).add_line_to(
            slope_corner).add_line_to(slope_end_point)
        self.play(Create(slope_lines))
        self.wait()

        slope_formula_general = MathTex(
            r'\frac{ \Delta y }{ \Delta x }').scale(0.9).shift(RIGHT*3+DOWN*2.9)
        slope_formula_specific = MathTex(
            r'\frac{ s[t-1] - s[t-2] }{ 1 }').scale(0.9).shift(RIGHT*3+DOWN*2.9)
        dx_text = MathTex('1', color=theming_colors[3]).scale(0.7).next_to(
            slope_lines, direction=DOWN, aligned_edge=DOWN)
        dy_text = MathTex('s[t-1] - s[t-2]',
                          color=theming_colors[3]).scale(0.7).next_to(slope_lines, direction=RIGHT, aligned_edge=RIGHT).shift(DOWN*0.2)
        dy_text.shift(RIGHT*dy_text.width*.5)
        dy_background = BackgroundRectangle(dy_text, fill_opacity=0.5)

        self.play(Write(slope_formula_general))
        self.wait()
        self.play(FadeIn(dy_background), Write(dx_text), Write(dy_text))
        self.wait()
        # FIXME: Again manual formula transformation because TransformMatchingTex is just terrible at everything.
        self.play(TransformMatchingShapes(dy_text, slope_formula_specific[0][:13]),
                  TransformMatchingShapes(
                      dx_text, slope_formula_specific[0][14:]),
                  Transform(slope_formula_general[0][2],
                            slope_formula_specific[0][13]),
                  ShrinkToCenter(slope_formula_general[0][:2]),
                  ShrinkToCenter(slope_formula_general[0][3:]),
                  FadeOut(dy_background))
        self.remove(slope_formula_general, *slope_formula_specific[0], slope_formula_general[0][2],
                    dx_text, dy_text)
        self.add(slope_formula_specific)
        self.wait()

        self.play(slope_lines.animate.move_to(
            sample_points[-1], aligned_edge=LEFT*DOWN).shift(UP*slope_lines.height/2 + RIGHT*slope_lines.width/2))

        predictor_step2 = MathTex(
            r'{{s[t] =}}{{s[t-1]}} + {{s[t-1]}} {{- s[t-2]}}').scale(0.9).shift(UP*3).align_to(LEFT*4, direction=LEFT)
        predictor_step_final = MathTex(
            r'{{s[t] =}} 2{{s[t-1]}} {{- s[t-2]}}').scale(0.9).shift(UP*3).align_to(LEFT*4, direction=LEFT)
        slope_formula_specific_clone = slope_formula_specific.copy()
        dy = samples[-1] - samples[-2]
        self.play(
            a_tracker.animate.set_value(dy),
            b_tracker.animate.set_value(-dy * (len(samples)-1)),
            TransformMatchingTex(predictor_step1, predictor_step2),
            TransformMatchingShapes(
                slope_formula_specific_clone[0][:13], predictor_step2[3:]),
        )
        self.remove(predictor_step1, slope_formula_specific_clone)
        self.add(predictor_step2)
        self.wait()
        self.play(TransformMatchingTex(predictor_step2, predictor_step_final))
        self.remove(predictor_step2)
        self.add(predictor_step_final)
        self.wait()

        distance_1_brace = BraceBetweenPoints(
            sample_points[1].get_center(),
            (sample_points[2].get_center()[0], sample_points[1].get_center()[1], 0))
        distance_1_label = MathTex('1').scale(0.9)
        distance_1_brace.put_at_tip(distance_1_label)
        distance_1_brace.add(distance_1_label)
        self.play(Create(distance_1_brace),
                  FadeOut(slope_formula_specific),
                  )
        self.wait()
        x_scale = grid.coords_to_point(1, 0)[0] - grid.coords_to_point(0, 0)[0]
        for i in range(2, len(sample_points)):
            self.play(ChangeSpeed(distance_1_brace.animate.shift(RIGHT * x_scale),
                                  speedinfo={0: 0, 1: 1}, rate_func=rate_functions.ease_in_out_cubic),
                      run_time=DEFAULT_ANIMATION_RUN_TIME*.8)
        self.wait()

        self.play(FadeOut(distance_1_brace, predictor_step_final, grid),
                  Uncreate(predictor_line),
                  Uncreate(intersection_point),
                  Uncreate(next_sample_location_line),
                  Uncreate(sample_points),
                  Uncreate(slope_lines))
        self.wait()

        self.next_section("formal taylor derivation", skip_animations=False)

        taylor = MathTex(r's[t]', r'&= \sum_{n=0}^{\infty}',
                         r'\frac{s^{(n)}(t_0)}{n!}', r'(t-t_0)^n').scale(0.9).shift(UP*2)
        self.play(Write(taylor))
        self.wait()

        self.play(Circumscribe(taylor[0], color=theming_colors[1]))
        self.wait()
        self.play(Circumscribe(taylor[2], color=theming_colors[1]))
        self.wait()

        interpolation_point = MathTex(
            r'\text{When predicting }t\!: \quad t_0 := t-1').scale(0.9)
        self.play(Write(interpolation_point))
        self.wait()

        s_tm1 = MathTex(
            r's^{(0)}(t_0) = s[t-1]', '= {}'.format(samples[-1])).scale(0.9).shift(DOWN*1.5)
        s_tm1[1].color = theming_colors[0]
        self.play(Write(s_tm1))
        self.wait()

        s_derivative = MathTex(
            r"s^{(1)}(t_0) = s'(t-1) =\; ???", color=theming_colors[1]).scale(0.9).shift(DOWN*3)
        self.play(Write(s_derivative))
        self.wait()

        self.play(FadeOut(s_tm1, s_derivative, interpolation_point, taylor))
        self.wait()

        self.next_section("discrete derivative", skip_animations=False)

        theorem = Tex(
            r'For the $n$-th derivative, at least $n+1$ points are required.').shift(UP*5)
        self.play(Write(theorem))
        self.wait()
        self.play(FadeOut(theorem))

        samples = MathTex(r's[t-1] = {} \quad s[t-2] = {}'.format(
            samples[-1], samples[-2]), substrings_to_isolate=['s[t-1]', "s[t-2]"]).scale(0.6).shift(UP*3)
        linear_combination_approach = MathTex(
            r"s'[t-1]= a\cdot s[t-1] + b\cdot s[t-2]", substrings_to_isolate=['s[t-1]', "s[t-2]", "s'[t-1]="]).scale(0.9).shift(UP*3)
        taylor_approximation_system = MathTex(r"s[t-1]", r"&=", r"s[t-1] &", r"\text{(trivial)}\\",
                                              r"s[t-2]", r"&=", r"s[t-1]-s'[t-1] &",
                                              r"=\frac{s^{(0)}[t-1]}{0!}(t-1 - (t-1))^0 + \frac{s^{(1)}[t-1]}{1!}(t - 2 - (t-1))^1").scale(0.9).shift(RIGHT*2+UP*1)
        for sidenote in [taylor_approximation_system[3], taylor_approximation_system[7]]:
            sidenote.set_opacity(0.8)
            sidenote.scale(0.5).align_to(RIGHT*6, direction=RIGHT)
        linear_combination = MathTex(
            r"{{s'[t-1]=}}a\big({{s[t-1]}}\big) +b\big({{s[t-1]-{{s'[t-1]}} }}\big) ").scale(0.9).shift(DOWN)
        linear_combination_explicit = MathTex(
            r"{{0\cdot}} s[t-1] + {{1\cdot}} {{s'[t-1]=}} ({{a+b}}) {{s[t-1]}} +({{-b}}) {{s'[t-1]}}").align_to(linear_combination).shift(DOWN)
        equation_system = MathTex(
            r"{{0}} & {{=}} {{a+b}}\\{{1}} & {{=}} {{-b}}").shift(DOWN*3)
        equation_system_solved = MathTex(
            r"{{a}} & {{=}} 1\\{{b}} & {{=}} -{{1}}").shift(DOWN*3)
        solved_discrete_derivative = MathTex(
            r"{{s'[t-1]=}}{{ \!\: }} s[t-1] {{-}} s[t-2]").scale(0.9).shift(UP*3)

        self.transform_between_equation_list(
            [samples, linear_combination_approach])
        self.play(Write(taylor_approximation_system))
        self.wait()

        # TransformMatchingTex is too primitive for this formula combination animation.
        lhs_source, a_source, _, b_source, _ = linear_combination_approach.copy()
        st1_rhs, st2_rhs = taylor_approximation_system[2].copy(
        ), taylor_approximation_system[6].copy()
        lhs_target, a_target, st1_target, b_target, st2_target, closing_brace = linear_combination
        # FIXME: Manim has problems matching up subtraction and addition symbols. Therefore we use a "dumb" transform, which here will mostly just move symbols around.
        self.play(Transform(st1_rhs, st1_target),
                  Transform(st2_rhs, st2_target),
                  Transform(lhs_source, lhs_target),
                  Transform(a_source, a_target),
                  Transform(b_source, b_target),
                  GrowFromCenter(closing_brace), run_time=DEFAULT_ANIMATION_RUN_TIME*2)
        self.remove(lhs_source, a_source, b_source, lhs_target, a_target,
                    st1_target, b_target, st2_target, closing_brace, st1_rhs, st2_rhs)
        self.add(linear_combination)
        self.wait()

        self.play(TransformMatchingTex(
            linear_combination, linear_combination_explicit))
        self.remove(linear_combination_explicit, linear_combination)
        self.add(linear_combination_explicit)
        self.wait()

        self.play(linear_combination_explicit.animate.set_color_by_tex_to_color_map({
            'a+b': theming_colors[0],
            '0\cdot': theming_colors[0],
            '-b': theming_colors[1],
            '1\cdot': theming_colors[1],
        }))
        self.wait()

        equation_system.set_color_by_tex_to_color_map({
            'a+b': theming_colors[0],
            '0': theming_colors[0],
            '-b': theming_colors[1],
            '1': theming_colors[1],
        })
        coeff_0_lhs, coeff_1_lhs, coeff_0_rhs, coeff_1_rhs = linear_combination_explicit[
            0].copy(), linear_combination_explicit[2].copy(), linear_combination_explicit[6].copy(), linear_combination_explicit[10].copy()
        coeff_0_lhs_target, _, eq1, _, coeff_0_rhs_target, _, coeff_1_lhs_target, _, eq2, _, coeff_1_rhs_target, *_ = equation_system
        self.play(TransformMatchingShapes(coeff_0_lhs, coeff_0_lhs_target),
                  TransformMatchingShapes(coeff_0_rhs, coeff_0_rhs_target),
                  TransformMatchingShapes(coeff_1_lhs, coeff_1_lhs_target),
                  TransformMatchingShapes(coeff_1_rhs, coeff_1_rhs_target),
                  GrowFromCenter(eq1), GrowFromCenter(eq2))
        self.remove(coeff_0_lhs, coeff_0_rhs, coeff_1_lhs,
                    coeff_1_rhs, *equation_system)
        self.add(equation_system)
        self.wait()

        self.play(TransformMatchingShapes(equation_system,
                  equation_system_solved, fade_transform_mismatches=True))
        self.remove(equation_system, equation_system_solved)
        self.add(equation_system_solved)
        self.wait()

        moving_equation_system = equation_system_solved.copy()
        self.play(
            TransformMatchingTex(linear_combination_approach,
                                 solved_discrete_derivative),
            Transform(moving_equation_system[:4],
                      solved_discrete_derivative[1]),
            Transform(moving_equation_system[4:], solved_discrete_derivative[3]))
        self.remove(linear_combination_approach,
                    moving_equation_system, solved_discrete_derivative)
        self.add(solved_discrete_derivative)
        self.wait()

        self.play(FadeOut(equation_system_solved,
                  taylor_approximation_system, linear_combination_explicit))
        self.remove(*self.mobjects)
        self.add(solved_discrete_derivative)
        self.wait()

        exercise = Text(
            "Exercise: Do this for the second derivative!", **small_font_args).shift(UP*1)
        second_derivative = MathTex(r"s''[t-1] = as[t-1]+bs[t-2]+cs[t-3]")
        self.play(Write(exercise), Write(second_derivative))
        self.wait()

        self.play(FadeOut(exercise, second_derivative))
        self.wait()

        final_combination = MathTex(r"s[t] &= s[t-1] + s'[t-1]",
                                    r"\\&= ", r"2s[t-1] + s[t-2]")
        final_combination[2].set_color(theming_colors[1])

        self.play(Write(final_combination))
        self.wait()

        other_approaches = MathTex(r"s[t] &= s[t-1]\\",
                                   r"s[t] &= s[t-1] + s'[t-1]", r"\\",
                                   r"s[t] &= s[t-1] + s'[t-1] + {{s''[t-1]}}", r"\\",
                                   r"s[t] &= s[t-1] + s'[t-1] + {{s''[t-1]}} + {{s'''[t-1]}}")
        self.remove(moving_equation_system, equation_system_solved)
        self.play(TransformMatchingTex(final_combination,
                  other_approaches), FadeOut(solved_discrete_derivative))
        self.remove(final_combination)
        self.add(other_approaches)
        self.wait()

        approaches_and_predictors = MathTex(r"s[t] &= s[t-1]\\",
                                            r"s[t] &= s[t-1] + s'[t-1]",
                                            r'&&= 2s[t-1] - s[t-2]\\',
                                            r"s[t] &= s[t-1] + s'[t-1] + ", r"s''[t-1]",
                                            r'&&= 3s[t-1] - 3s[t-2] + s[t-3]\\',
                                            r"s[t] &= s[t-1] + s'[t-1] + ", r"s''[t-1]", r" + ", r"s'''[t-1]",
                                            r'&&= 4s[t-1] - 6s[t-2] + 4s[t-3] - s[t-4]').scale(0.6)
        correct_taylor_version = MathTex(r"s[t] &= s[t-1]\\",
                                         r"s[t] &= s[t-1] + s'[t-1]",
                                         r'&&= 2s[t-1] - s[t-2]\\',
                                         r"s[t] &= s[t-1] + s'[t-1] + ", r"\frac{s''[t-1]}{2}",
                                         r'&&= 3s[t-1] - 3s[t-2] + s[t-3]\\',
                                         r"s[t] &= s[t-1] + s'[t-1] + ", r"\frac{ s''[t-1]}{2}", r" + ", r"\frac{ s'''[t-1] }{6}",
                                         r'&&= 4s[t-1] - 6s[t-2] + 4s[t-3] - s[t-4]').scale(0.6)
        self.transform_between_equation_list(
            [other_approaches, approaches_and_predictors, correct_taylor_version], do_first_write=False)

        self.play(correct_taylor_version.animate.set_color_by_tex(
            r'\frac', color=theming_colors[1]))
        self.wait()

        self.play(TransformMatchingTex(
            correct_taylor_version, approaches_and_predictors))
        self.remove(correct_taylor_version, approaches_and_predictors)
        self.add(approaches_and_predictors)
        self.wait()


class ResidualCoding(CustomParentScene):
    def construct(self):
        template = TexTemplate()
        template.add_to_preamble(r'''\usepackage[english]{babel}
        \usepackage{csquotes}\usepackage{cancel}\usepackage{tabularx}\usepackage{array}''')
        self.next_section(skip_animations=False)

        # pick up the transition from the LPC scene
        residual_title = Text('Residual', **font_args).shift(UP*3)
        self.add(residual_title)

        rng = Random(77)
        residual_numbers = np.array(
            [int(rng.gauss(mu=0, sigma=50)) for _ in range(17)]).T
        residuals = [MathTex(str(residual), color=theming_colors[4]).scale(
            0.6) for residual in residual_numbers]
        residuals = VGroup(*residuals).arrange_in_grid(
            cols=1, buff=MED_SMALL_BUFF).shift(LEFT*5)
        self.play(Write(residuals))
        self.wait()

        fixed_bit_depth = Tex('Choose bit depth 16').scale(0.9)
        self.play(FadeOut(residual_title), Write(fixed_bit_depth))
        self.wait()

        residuals_hex = [Text('{:04x}'.format(int(residual) & 0xffff), color=theming_colors[4], **mono_font_args)
                         for residual in residual_numbers]
        residuals_hex = VGroup(*residuals_hex).arrange_in_grid(
            cols=1, buff=MED_SMALL_BUFF).scale_to_fit_height(residuals.height).shift(LEFT*3.5)
        self.play(FadeIn(residuals_hex, shift=RIGHT*2),
                  fixed_bit_depth.animate.shift(RIGHT))
        self.wait()

        self.play(FadeOut(residuals, residuals_hex, fixed_bit_depth))
        counter_scenario_numbers = [1, 2, -4, -2, -1,
                                    0, 0, -1, -1, 1, 1, 0, 3, 0, 3, 3, -1390, 1581]
        counter_scenario_texts = [MathTex(str(n), color=theming_colors[4]).scale(0.9)
                                  for n in counter_scenario_numbers]
        counter_scenario_texts = VGroup(
            *counter_scenario_texts).arrange_in_grid(rows=1, buff=MED_SMALL_BUFF)
        original_counter_scenario_texts = counter_scenario_texts.copy()
        counter_scenario_bin = [Text(('{:012b}'.format(int(n) & 0xffff))[-12:], color=theming_colors[4] if abs(n) < 200 else theming_colors[1], **mono_font_args)
                                for n in counter_scenario_numbers]
        counter_scenario_bin = VGroup(*counter_scenario_bin).arrange_in_grid(
            rows=4, buff=MED_LARGE_BUFF).scale_to_fit_width(counter_scenario_texts.width).scale(1.2)
        self.play(Write(counter_scenario_texts))
        self.wait()

        small_bit_depth_brace = BraceBetweenPoints(
            counter_scenario_texts[0].get_left(), counter_scenario_texts[-3].get_right())
        brace_text = Text('Would fit in 3 bits', **small_font_args)
        brace_text.font_size = DEFAULT_FONT_SIZE*.5
        small_bit_depth_brace.put_at_tip(brace_text)
        small_bit_depth_brace.add(brace_text)
        self.play(Create(small_bit_depth_brace))
        self.wait()

        needs_13_bits = Text('... but needs 12 bits!', **
                             small_font_args).shift(DOWN*2.5)

        self.play(Transform(counter_scenario_texts, counter_scenario_bin),
                  Transform(small_bit_depth_brace, needs_13_bits))
        self.remove(counter_scenario_bin, counter_scenario_texts,
                    small_bit_depth_brace)
        self.add(counter_scenario_bin, needs_13_bits)
        self.wait()

        self.play(FadeOut(needs_13_bits),
                  Transform(counter_scenario_bin, original_counter_scenario_texts))
        self.remove(counter_scenario_bin, original_counter_scenario_texts)
        self.add(original_counter_scenario_texts)
        self.wait()

        prob_table = MathTable([[r'r\left[t\right]', r'\pm 1', r'\pm 2', r'\pm 4', r'\pm 8', r'\cdots'],
                                [r'p\left(r\left[t\right]\right) \approx', r'\frac{1}{2}', r'\frac{1}{8}', r'\frac{1}{32}', r'\frac{1}{128}', r'\cdots']],
                               line_config={'stroke_width': 1}).shift(DOWN)

        self.play(original_counter_scenario_texts.animate.shift(
            UP*3), prob_table.create())
        self.wait()
        self.play(FadeOut(original_counter_scenario_texts, prob_table))
        self.wait()

        self.next_section('elias gamma', skip_animations=False)

        variable_bit_depth = Text(
            'Variable bit depth?', **font_args).shift(UP*3)
        self.play(Write(variable_bit_depth))
        self.wait()

        example_number = 489
        example_number_text = MathTex(
            str(example_number), color=theming_colors[4]).scale(1.3)
        required_bit_depth = int(floor(log2(example_number)))
        example_number_bit_depth = MathTex(
            str(required_bit_depth), color=theming_colors[0]).scale(1.3).shift(LEFT*2)
        example_number_encoded = Text(('{:0' + str(required_bit_depth) + 'b}').format(example_number),
                                      color=theming_colors[2], **mono_font_args).scale(1.4).shift(RIGHT*2)
        description_bit_depth = Text(
            'Bit depth', **small_font_args).move_to(example_number_bit_depth).shift(DOWN*1.5)
        description_number_encoded = Text(
            'Encoded with bit depth', **small_font_args).move_to(example_number_encoded).shift(DOWN*1.5)
        description_bit_depth.font_size = description_number_encoded.font_size = DEFAULT_FONT_SIZE*.6

        self.play(Write(example_number_text))
        transformation_copy_1, transformation_copy_2 = example_number_text.copy(
        ), example_number_text.copy()
        self.play(Transform(transformation_copy_1, example_number_bit_depth),
                  Transform(transformation_copy_2, example_number_encoded),
                  example_number_text.animate.shift(UP*1.5),
                  Write(description_bit_depth),
                  Write(description_number_encoded))
        self.remove(transformation_copy_1, transformation_copy_2)
        self.add(example_number_bit_depth, example_number_encoded)
        self.wait()

        secondary_bit_depth = Text(
            '... at what bit depth?', **small_font_args).move_to(description_bit_depth).shift(DOWN*1)
        secondary_bit_depth.font_size = DEFAULT_FONT_SIZE*.6
        self.play(Write(secondary_bit_depth))
        self.wait()

        bit_depth_varsize = Text(
            'at variable bit depth', **small_font_args).move_to(secondary_bit_depth)
        bit_depth_varsize.font_size = DEFAULT_FONT_SIZE*.6
        self.play(Transform(secondary_bit_depth, bit_depth_varsize))
        self.remove(secondary_bit_depth)
        self.add(bit_depth_varsize)
        self.wait()

        self.play(FadeOut(bit_depth_varsize, description_bit_depth, description_number_encoded,
                  variable_bit_depth, example_number_text, example_number_encoded, example_number_bit_depth))

        unary = Text('Unary', **small_font_args).shift(UP*3)

        unary_example_numbers = [1, 3, 16, 20]
        previous_decimal = previous_unary = None
        for number in unary_example_numbers:
            decimal_number = MathTex(str(number)).shift(DOWN)
            unary_number = VGroup(*[
                Line(start=UP + (LEFT + RIGHT*rng.random())/2,
                     end=DOWN + (LEFT + RIGHT*rng.random())/2,
                     stroke_width=DEFAULT_STROKE_WIDTH*2)
                .shift(LEFT*4+RIGHT/1.9*i)
                for i in range(number)]).scale(0.8).center().shift(UP)
            if previous_decimal is None:
                self.play(Write(decimal_number), Write(
                    unary_number), Write(unary))
            else:
                self.play(Transform(previous_decimal, decimal_number),
                          Transform(previous_unary, unary_number))
                self.remove(previous_decimal, previous_unary)
            previous_unary, previous_decimal = unary_number, decimal_number
            self.add(previous_decimal, previous_unary)
            self.wait()

        unary_in_binary = Text(
            '0'*unary_example_numbers[-1] + '1', **mono_font_args).scale(1.4).shift(UP)
        self.play(Transform(previous_unary, unary_in_binary))
        self.remove(previous_unary)
        self.add(unary_in_binary)
        self.wait()

        self.play(
            FadeOut(unary_in_binary, unary, previous_decimal),
            FadeIn(example_number_text, example_number_bit_depth,
                   example_number_encoded)
        )
        example_number_bit_depth_unary = Text(
            '0'*required_bit_depth + '1', color=theming_colors[0], **mono_font_args).scale(1.4).move_to(example_number_bit_depth)
        self.play(Transform(example_number_bit_depth,
                  example_number_bit_depth_unary))
        self.remove(example_number_bit_depth, example_number_bit_depth_unary)
        self.add(example_number_bit_depth_unary)
        self.wait()

        combined_number = Text('{}1 {}'.format(
            '0'*required_bit_depth, bin(example_number)[3:]), color=theming_colors[4], **mono_font_args).scale(1.4)
        self.play(
            Transform(example_number_bit_depth_unary,
                      combined_number[:required_bit_depth+1]),
            Transform(example_number_encoded[1:],
                      combined_number[required_bit_depth+1:]),
            FadeOut(example_number_encoded[0], shift=LEFT))
        self.remove(example_number_bit_depth_unary,
                    example_number_encoded, *combined_number, combined_number, *example_number_encoded)
        self.add(combined_number)
        self.wait()

        very_combined_number = Text('{}{}'.format(
            '0'*required_bit_depth, bin(example_number)[2:]), color=theming_colors[4], **mono_font_args).scale(1.4)
        self.play(Transform(
            combined_number, very_combined_number))
        self.remove(combined_number, very_combined_number)
        self.add(very_combined_number)
        self.wait()

        self.play(FadeOut(very_combined_number, example_number_text))

        encode_paragraph = Tex(r'''To encode number:
        \begin{itemize}
            \item Count bits in number $\implies$ \# zeros to store
            \item Store number with natural bit depth
        \end{itemize}''', tex_environment='flushleft').scale(0.9).align_to(LEFT*5, direction=LEFT).shift(UP*2)
        decode_paragraph = Tex(r'''To decode number:
        \begin{itemize}
            \item Read zeros and terminating 1
            \item Read as many bits as zeros $\implies$ number
        \end{itemize}''', tex_environment='flushleft').scale(0.9).align_to(LEFT*5, direction=LEFT).shift(DOWN*2)
        self.play(Write(encode_paragraph))
        self.wait()
        self.play(Write(decode_paragraph))
        self.wait()

        elias_gamma_large = Text('Elias γ Coding', **font_args).shift(DOWN*3)
        self.play(
            Write(elias_gamma_large),
            encode_paragraph.animate.shift(UP*.5),
            decode_paragraph.animate.shift(UP*1.5))
        self.wait()

        sidebar_title = Paragraph(
            'The Journey of\nLossless Audio Compression', disable_ligatures=True, alignment='center', **small_font_args)
        sidebar_title.shift(UP*3.5).scale(0.45)
        sidebar_separator = Rectangle(
            color=WHITE, height=15, width=8).shift(LEFT*2)
        sidebar_separator.set_fill(color=BLACK, opacity=0.6)
        idea1 = Paragraph(
            'Sine waves?\n—> MP3', disable_ligatures=True, alignment='center', **small_font_args).align_to(sidebar_title, direction=UP, alignment_vect=UP).shift(DOWN*.5).scale(0.4)
        idea2 = Paragraph(
            'Polynomial approximation', disable_ligatures=True, alignment='center', **small_font_args).align_to(idea1, direction=UP, alignment_vect=UP).shift(DOWN*.5).scale(0.4)
        idea3 = Paragraph(
            'Recursive signal definition', disable_ligatures=True, alignment='center', **small_font_args).align_to(idea2, direction=UP, alignment_vect=UP).shift(DOWN*.5).scale(0.4)
        idea4 = Paragraph(
            'Linear Predictive Coding', disable_ligatures=True, alignment='center', **small_font_args).align_to(idea3, direction=UP, alignment_vect=UP).shift(DOWN*.5).scale(0.4)
        sidebar = VGroup(sidebar_separator, sidebar_title,
                         idea1, idea2, idea3, idea4).shift(LEFT*10)
        self.add(sidebar)
        self.bring_to_front(elias_gamma_large)
        self.play(sidebar.animate.shift(RIGHT*5))
        self.wait()

        idea5 = Paragraph(
            'Elias γ Coding', disable_ligatures=True, alignment='center', **small_font_args).shift(LEFT*5).align_to(idea4, direction=UP, alignment_vect=UP).shift(DOWN*.5).scale(0.4)
        self.play(Transform(elias_gamma_large, idea5))
        self.remove(elias_gamma_large)
        self.add(idea5)
        sidebar.add(idea5)
        self.wait()
        self.play(sidebar.animate.shift(LEFT*5))
        self.wait()

        self.next_section('exponential golomb', skip_animations=False)
        encode_paragraph_exp_golomb = Tex(r'''To encode number:
        \begin{itemize}
            \item Add 1
            \item Count bits in number $\implies$ \# zeros to store
            \item Store number with natural bit depth
        \end{itemize}''', tex_environment='flushleft').scale(0.6).shift(UP+LEFT*3)
        decode_paragraph_exp_golomb = Tex(r'''To decode number:
        \begin{itemize}
            \item Read zeros and terminating 1
            \item Read as many more bits as zeros
            \item Subtract 1
        \end{itemize}''', tex_environment='flushleft').scale(0.6).shift(UP+RIGHT*4)

        self.play(
            Transform(encode_paragraph, encode_paragraph_exp_golomb),
            Transform(decode_paragraph, decode_paragraph_exp_golomb),
        )
        self.remove(encode_paragraph, decode_paragraph)
        self.add(encode_paragraph_exp_golomb, decode_paragraph_exp_golomb)
        self.wait()

        exp_golomb = Text('Exponential Golomb Coding',
                          **font_args).shift(DOWN*2.5)
        self.play(Write(exp_golomb))
        self.wait()

        golomb_examples = [0, 1, 2, 5, 14, 31, example_number]
        golomb_example_numbers = VGroup()
        golomb_example_encode_1 = VGroup()
        golomb_example_encode_2 = VGroup()
        for number in golomb_examples:
            golomb_example_numbers.add(
                MathTex(str(number), color=theming_colors[4]))
            golomb_example_encode_1.add(
                MathTex(str(number+1), color=theming_colors[4]))
            bit_depth = int(floor(log2(number+1)))
            golomb_example_encode_2.add(
                Text('0'*bit_depth + ' {:b}'.format(number+1), color=theming_colors[4], **mono_font_args))
        golomb_example_numbers.arrange_in_grid(
            rows=1, buff=LARGE_BUFF).shift(DOWN*2)
        golomb_example_encode_1.arrange_in_grid(
            rows=1, buff=LARGE_BUFF).shift(DOWN*2)
        golomb_example_encode_2.arrange_in_grid(
            rows=2, buff=LARGE_BUFF).shift(DOWN*2)
        self.transform_between_equation_list(
            [golomb_example_numbers, golomb_example_encode_1, golomb_example_encode_2], transform_animation=Transform, additional_first_animation=[FadeOut(exp_golomb)])

        self.play(FadeOut(golomb_example_encode_2,
                  encode_paragraph_exp_golomb, decode_paragraph_exp_golomb))
        order_text = MathTex(r'k &\;\overset{\wedge}{=}\text{ Order}\\', r'k = 0 &\implies \text{ as before}\\',
                             r'k \text{ small} &\implies \text{ good for small } n\\', r'k \text{ large} &\implies \text{ good for large } n\\')
        for line in order_text:
            self.play(Write(line))
            self.wait()

        exp_golomb_formula = MathTex(r'{{n}} = {{q}} \cdot {{2^k}} + {{r}}')
        descriptor_n = Tex(r'number $\in \mathbb{N}_0$').scale(0.6)
        descriptor_q = Tex(r'quotient (larger part)').scale(0.6)
        descriptor_exponent = Tex(r'base').scale(0.6)
        descriptor_r = Tex(r'remainder (smaller part)').scale(0.6)
        descriptors = VGroup(*[descriptor_n, descriptor_q, descriptor_exponent,
                             descriptor_r]).arrange_in_grid(rows=1, buff=LARGE_BUFF).shift(DOWN*2)
        for descriptor, index in zip(descriptors, [0, 2, 4, 6]):
            descriptor_arrow = Arrow(start=descriptor.get_top(
            ), end=exp_golomb_formula[index].get_bottom(), stroke_width=3, max_tip_length_to_length_ratio=0.1)
            descriptor.add(descriptor_arrow)

        self.play(FadeOut(order_text), Write(exp_golomb_formula))
        self.wait()
        self.play(Write(descriptors))
        self.wait()

        example_k = 5
        example_binary = '{:b}'.format(example_number)
        example_number_text.move_to(ORIGIN)
        example_binary_whole = Text(
            example_binary, color=theming_colors[4], **mono_font_args).shift(DOWN*1)
        example_binary_separated = Text(example_binary[:-example_k] + ' ' + example_binary[-example_k:],
                                        color=theming_colors[4], **mono_font_args).move_to(example_binary_whole)
        example_k_text = MathTex('k = {}'.format(example_k)
                                 ).scale(0.9).shift(UP*1.5)

        self.play(FadeOut(descriptors, shift=UP*3),
                  exp_golomb_formula.animate.shift(UP*3),
                  Write(example_k_text))
        self.wait()
        self.play(Write(example_number_text), Write(example_binary_whole))
        self.wait()
        self.play(Transform(example_binary_whole, example_binary_separated))
        self.remove(example_binary_whole, example_binary_separated)
        self.add(example_binary_separated)
        self.wait()

        upper_number = example_number >> example_k
        upper_encoding_hint = Tex('encode this like {} with order 0'.format(
            upper_number)).scale(0.7).shift(DOWN*3+LEFT)
        arrow = Arrow(
            start=upper_encoding_hint.get_top(),
            end=example_binary_separated[-example_k*3//2].get_bottom(),
            stroke_width=3, max_tip_length_to_length_ratio=0.1)
        upper_encoding_hint.add(arrow)
        upper_encoded_bitdepth = int(floor(log2(upper_number+1)))
        upper_encoded = Text('0'*upper_encoded_bitdepth +
                             ' {:b}'.format(upper_number+1), color=theming_colors[4], **mono_font_args).move_to(example_binary_separated.get_left()).shift(LEFT*.5)

        self.play(Write(upper_encoding_hint))
        self.wait()
        self.remove(example_binary_separated)
        upper_section_example = VGroup(*example_binary_separated[:-example_k])
        lower_section_example = VGroup(*example_binary_separated[-example_k:])
        self.add(upper_section_example, lower_section_example)
        self.play(
            Transform(upper_section_example, upper_encoded))
        self.remove(upper_section_example, upper_encoded)
        self.add(upper_encoded)
        self.wait()

        unary_brace = Brace(upper_encoded[:upper_encoded_bitdepth])
        q_brace = Brace(upper_encoded[upper_encoded_bitdepth:])
        r_brace = Brace(example_binary_separated[-example_k:])
        label = Tex("$q$'s unary bit depth").scale(0.6)
        unary_brace.put_at_tip(label)
        label.shift(LEFT*.8)
        unary_brace.add(label)
        label = MathTex('q').scale(0.6)
        q_brace.put_at_tip(label)
        q_brace.add(label)
        label = MathTex('r').scale(0.6)
        r_brace.put_at_tip(label)
        r_brace.add(label)

        self.play(Create(unary_brace), Create(q_brace),
                  Create(r_brace), FadeOut(upper_encoding_hint))
        self.wait()

        # no need to include 0 as it is automatically added as the first column anyways.
        table_ks = [1, 2, 5, 8]
        table_numbers = list(range(0, 8, 1))
        table_numbers.extend(range(16, 256, 37))
        table_numbers.extend(range(256, 1300, 157))
        table_numbers.extend(range(1469, 10000, 1281))
        exp_golomb_table_data = create_exp_golomb_table(
            numbers_to_encode=table_numbers, ks=table_ks)
        exp_golomb_table = Table(
            exp_golomb_table_data,
            v_buff=0.12, h_buff=0.5,
            line_config={'stroke_opacity': 0, 'color': TRANSPARENT},
            element_to_mobject=Text,
            element_to_mobject_config={
                'color': theming_colors[4],
                **mono_font_args
            },
            arrange_in_grid_config={'cell_alignment': RIGHT},
            col_labels=[
                MathTex('n'),
                MathTex('k = 0'),
                *(MathTex('k = {}'.format(k)) for k in table_ks)
            ]).scale_to_fit_width(14)
        self.play(LaggedStart(
            exp_golomb_table.create(DEFAULT_ANIMATION_RUN_TIME*3),
            FadeOut(unary_brace, q_brace, r_brace, upper_encoded,
                    example_k_text, exp_golomb_formula, example_number_text, lower_section_example),
            lag_ratio=0.4,
        ))
        self.wait()

        self.play(Circumscribe(exp_golomb_table.get_columns()
                  [1], color=theming_colors[1]))
        self.wait()
        self.play(Circumscribe(exp_golomb_table.get_columns()
                  [-1], color=theming_colors[1]))
        self.wait()

        different_orders_text = Paragraph(
            'Up to 32,768 different orders\n(More in episode 3)', disable_ligatures=True, **small_font_args).shift(DOWN*3)
        self.play(exp_golomb_table.animate.shift(
            UP*1.5), Write(different_orders_text))
        self.wait()

        self.play(FadeOut(different_orders_text, exp_golomb_table))
        self.wait()

        self.next_section('final', skip_animations=False)
        problem = Text("Problem: Can’t encode negative numbers!",
                       **small_font_args).shift(UP*3)
        self.play(Write(problem))
        self.wait()

        positive_numbers = list(range(20))
        negative_numbers = list(
            chain(*zip(cycle(range(11)), range(-1, -11, -1))))
        number_list = MathTex(r'\begin{tabular}{*{20}{c }}' + '&'.join(('${}$'.format(number) for number in positive_numbers)) + r'\\' + '&'.join(
            ('${}$'.format(number) for number in negative_numbers)) + r'\end{tabular}', tex_environment='center', tex_template=template).scale_to_fit_width(13)

        # AGAIN, no way of doing this with split strings as the Manim tex renderer is a giant hack and the LaTeX would have unclosed environments.
        positive_number_list = VGroup(*number_list[0][:30])
        negative_number_list = VGroup(*number_list[0][30:])

        self.play(Write(positive_number_list))
        self.wait()
        self.play(FadeIn(negative_number_list, shift=DOWN*.5))
        self.wait()

        z_mapping_formula = MathTex(
            r'n = \begin{cases}2x & \text{ if } x \ge 0\\-2x-1 & \text{ if } x < 0\end{cases} \quad \rightsquigarrow \text{ normal Exponential Golomb}').shift(DOWN*2)
        self.play(Write(z_mapping_formula))
        self.wait()

        self.play(FadeOut(positive_number_list,
                  negative_number_list, problem, z_mapping_formula))
        self.wait()


class InterchannelDecorrelation(CustomParentScene):
    def construct(self):
        self.next_section(skip_animations=False)

        (_, audio_data) = load_wav_to_f64("../Mega Man Title 2.wav")
        assert audio_data.shape[1] == 2
        left_channel, right_channel = audio_data.T

        audio_section_start = 7400
        audio_section_length = 1000
        left_section = left_channel[audio_section_start:
                                    audio_section_start+audio_section_length]
        right_section = right_channel[audio_section_start:
                                      audio_section_start+audio_section_length]
        mid_section = (left_section + right_section)/2
        side_section = left_section - right_section

        left_coordinates = Axes(x_range=[-10, audio_section_length-1, 1], y_range=[-1.1, 1.2, 1], x_length=10, y_length=3, axis_config={
            'include_ticks': False,
            'include_tip': True,
            'tip_width': .15,
            'tip_height': .15,
            'include_numbers': False,
            'exclude_origin_tick': False,
        })
        right_coordinates = left_coordinates.copy().shift(DOWN*2)

        left_section_curve = left_coordinates.plot(
            lambda x: left_section[int(x)]*2, color=theming_colors[0])
        right_section_curve = right_coordinates.plot(
            lambda x: right_section[int(x)]*2, color=theming_colors[1])
        # will not be shifted around until coordinates are centered again
        mid_section_curve = left_coordinates.plot(
            lambda x: mid_section[int(x)]*2, color=theming_colors[5])
        side_section_curve = left_coordinates.plot(
            lambda x: side_section[int(x)]*2, color=theming_colors[3])

        self.play(Create(left_coordinates), Create(left_section_curve))
        self.pause()
        self.play(left_coordinates.animate.shift(UP*2), left_section_curve.animate.shift(
            UP*2), Create(right_coordinates), Create(right_section_curve))
        self.wait()

        interchannel_decorrelation = Text(
            'Interchannel Decorrelation', **small_font_args)
        self.play(Write(interchannel_decorrelation))
        self.wait()

        lr_labels = MarkupText(
            '<span foreground="' +
            theming_colors[0] + '">Left</span>    <span foreground="' +
            theming_colors[1] + '">Right</span>',
            **small_font_args).shift(UP*2)
        ms_labels = MarkupText(
            '<span foreground="' +
            theming_colors[5] + '">Mid</span>    <span foreground="' +
            theming_colors[3] + '">Side</span>',
            **small_font_args).move_to(lr_labels)

        self.play(FadeOut(interchannel_decorrelation), left_coordinates.animate.shift(DOWN*2),
                  right_coordinates.animate.shift(UP*2),
                  left_section_curve.animate.shift(DOWN*2),
                  right_section_curve.animate.shift(UP*2),
                  Write(lr_labels))
        self.remove(right_coordinates)
        self.wait()
        self.play(Transform(left_section_curve, mid_section_curve),
                  Transform(right_section_curve, side_section_curve),
                  Transform(lr_labels, ms_labels))
        self.remove(left_section_curve, right_section_curve, lr_labels)
        self.add(mid_section_curve, side_section_curve, ms_labels)
        self.wait()

        var_colors = {
            'l': theming_colors[0],
            'r': theming_colors[1],
            'm': theming_colors[5],
            's': theming_colors[3],
        }
        encode_formulas = MathTex(
            r'm&= \frac{l+r}{2}\\s&=l-r').shift(LEFT*3+DOWN*2.5)
        # AGAIN: because of frac it is impossible to split this normally
        encode_formulas[0][0].color = theming_colors[5]
        encode_formulas[0][7].color = theming_colors[3]
        encode_formulas[0][2].color = encode_formulas[0][9].color = theming_colors[0]
        encode_formulas[0][4].color = encode_formulas[0][11].color = theming_colors[1]
        decode_formulas = MathTex(r'l&=m+s \\ r&=m-s').shift(RIGHT*3+DOWN*2.5)
        decode_formulas[0][0].color = theming_colors[0]
        decode_formulas[0][5].color = theming_colors[1]
        decode_formulas[0][2].color = decode_formulas[0][7].color = theming_colors[5]
        decode_formulas[0][4].color = decode_formulas[0][9].color = theming_colors[3]
        self.play(Write(encode_formulas))
        self.wait()
        self.play(Write(decode_formulas))
        self.wait()

        self.play(side_section_curve.animate(
            run_time=DEFAULT_ANIMATION_RUN_TIME*.5).shift(UP*3.5))
        self.wait()
        self.play(side_section_curve.animate(
            run_time=DEFAULT_ANIMATION_RUN_TIME*.5).shift(DOWN*3.5))
        self.wait()


class Ending(CustomParentScene):
    def construct(self):
        from analysis import mid_residuals, side_residuals, subframe_warmups, blocksize, frame_index, subframe_partition_orders, subframe_orders

        (_, audio_data) = load_wav_to_f64("../Mega Man Title 2.wav")
        left_channel, right_channel = (audio_data.T * 2**15).astype(int)
        left_subblock = left_channel[frame_index *
                                     blocksize:(frame_index+1)*blocksize]
        right_subblock = right_channel[frame_index *
                                       blocksize:(frame_index+1)*blocksize]
        # will become subframe 0
        mid_subblock = (left_subblock + right_subblock)/2
        # subframe 1
        side_subblock = (left_subblock - right_subblock)
        mid_warmups = subframe_warmups[0]
        mid_predictor_function = np.array(
            mid_warmups + [2*mid_subblock[i-1] - mid_subblock[i-2] for i in range(subframe_orders[0], len(mid_subblock))])
        side_warmups = subframe_warmups[1]
        side_predictor_function = np.array(
            side_warmups + [side_subblock[i-1] for i in range(subframe_orders[1], len(mid_subblock))])

        coordinates = Axes(x_range=[0, blocksize-1, 10], y_range=[-2**14, 2**14, 100], x_length=12, y_length=6, axis_config={
            'include_ticks': False,
            'include_tip': True,
            'tip_width': .15,
            'tip_height': .15,
            'include_numbers': False,
            'exclude_origin_tick': False,
        })
        left_subblock_graph = coordinates.plot(
            lambda x: left_subblock[int(x)], color=theming_colors[0])
        right_subblock_graph = coordinates.plot(
            lambda x: right_subblock[int(x)], color=theming_colors[1])
        mid_subblock_graph = coordinates.plot(
            lambda x: mid_subblock[int(x)], color=theming_colors[5])
        side_subblock_graph = coordinates.plot(
            lambda x: side_subblock[int(x)], color=theming_colors[3])
        mid_predictor_graph = coordinates.plot(
            lambda x: mid_predictor_function[int(x)], color=theming_colors[2])
        side_predictor_graph = coordinates.plot(
            lambda x: side_predictor_function[int(x)], color=theming_colors[2])
        mid_residuals_graph = coordinates.plot(lambda x: mid_residuals[int(
            x)-subframe_partition_orders[0]], color=theming_colors[4])
        side_residuals_graph = coordinates.plot(lambda x: side_residuals[int(
            x)-subframe_partition_orders[1]], color=theming_colors[4])
        zeroed_graph = coordinates.plot(lambda _: 0, color=theming_colors[4])

        residual_count = 20
        def residual_converter(x): return 2*x if x >= 0 else -2*x - 1
        mid_residual_list = VGroup(
            *(MathTex(str(residual), color=theming_colors[4]).scale(0.6) for residual in mid_residuals[:residual_count])
        ).arrange_in_grid(cols=1).align_to(ORIGIN, direction=UP).shift(LEFT*3)
        mid_residual_list_positive = VGroup(
            *(MathTex(str(residual_converter(residual)), color=theming_colors[4]).scale(0.6) for residual in mid_residuals[:residual_count])
        ).arrange_in_grid(cols=1).align_to(ORIGIN, direction=UP).shift(LEFT*3)
        mid_residual_list_encoded = VGroup(
            *(Text(alternate_encode_exp_golomb(residual_converter(residual), k=subframe_partition_orders[0]), color=theming_colors[4], **mono_font_args).scale(0.8) for residual in mid_residuals[:residual_count])
        ).arrange_in_grid(cols=1).align_to(ORIGIN, direction=UP).shift(LEFT*3)

        side_residual_list = VGroup(
            *(MathTex(str(residual), color=theming_colors[4]).scale(0.6) for residual in side_residuals[:residual_count])
        ).arrange_in_grid(cols=1).align_to(ORIGIN, direction=UP).shift(RIGHT*3)
        side_residual_list_positive = VGroup(
            *(MathTex(str(residual_converter(residual)), color=theming_colors[4]).scale(0.6) for residual in side_residuals[:residual_count])
        ).arrange_in_grid(cols=1).align_to(ORIGIN, direction=UP).shift(RIGHT*3)
        side_residual_list_encoded = VGroup(
            *(Text(alternate_encode_exp_golomb(residual_converter(residual), k=subframe_partition_orders[0]), color=theming_colors[4], **mono_font_args).scale(0.8) for residual in side_residuals[:residual_count])
        ).arrange_in_grid(cols=1).align_to(ORIGIN, direction=UP).shift(RIGHT*3)

        mid_stored_text = Paragraph('Standard order 2\nwarmup: [ {:04x}, {:04x} ]\npredictor order {:x}'.format(
            mid_warmups[0] & 0xffff, mid_warmups[1] & 0xffff, subframe_partition_orders[0]), **mono_font_args).shift(UP*3+LEFT*3)
        side_stored_text = Paragraph('Standard order 1\nwarmup: [ {:04x} ]\npredictor order {:x}'.format(
            side_warmups[0] & 0xffff, subframe_partition_orders[1]), **mono_font_args).shift(UP*3+RIGHT*3)

        self.play(Create(coordinates), Create(
            left_subblock_graph), Create(right_subblock_graph))
        self.wait()

        decorrelation = Text('Interchannel Decorrelation',
                             **small_font_args).shift(DOWN*3)
        self.play(Transform(left_subblock_graph, mid_subblock_graph),
                  Transform(right_subblock_graph, side_subblock_graph),
                  Write(decorrelation))
        self.remove(left_subblock_graph, right_subblock_graph)
        self.add(mid_subblock_graph, side_subblock_graph)
        self.wait()

        lpc = Text('Linear Predictive Coding', **
                   small_font_args).move_to(decorrelation)
        self.play(Create(mid_predictor_graph), Transform(decorrelation, lpc))
        self.remove(decorrelation)
        self.add(lpc)
        self.play(Transform(mid_subblock_graph, mid_residuals_graph),
                  Transform(mid_predictor_graph, zeroed_graph))
        self.remove(mid_subblock_graph)
        self.add(mid_residuals_graph, zeroed_graph)
        self.play(FadeOut(zeroed_graph),
                  Transform(mid_predictor_graph, mid_stored_text[:2]),
                  mid_residuals_graph.animate.shift(UP*1.5))
        self.remove(mid_predictor_graph)
        self.add(mid_stored_text[:2])
        self.wait()

        self.play(Create(side_predictor_graph))
        self.play(Transform(side_subblock_graph, side_residuals_graph),
                  Transform(side_predictor_graph, zeroed_graph))
        self.remove(side_subblock_graph)
        self.add(zeroed_graph, side_residuals_graph)
        self.play(FadeOut(zeroed_graph, coordinates, lpc),
                  Transform(side_predictor_graph, side_stored_text[:2]),
                  side_residuals_graph.animate.shift(DOWN*1.5))
        self.remove(side_predictor_graph)
        self.add(side_stored_text[:2])
        self.wait()

        exp_golomb_coding = Text(
            'Exponential Golomb Coding', **small_font_args).shift(UP)
        self.play(Transform(side_residuals_graph, side_residual_list),
                  Transform(mid_residuals_graph, mid_residual_list))
        self.remove(side_residuals_graph, mid_residuals_graph)
        self.add(side_residual_list, mid_residual_list)
        self.wait()
        self.play(Transform(side_residual_list, side_residual_list_positive),
                  Transform(mid_residual_list, mid_residual_list_positive),
                  Write(exp_golomb_coding),
                  Write(side_stored_text[2:]),
                  Write(mid_stored_text[2:]))
        self.remove(side_residual_list, mid_residual_list)
        self.add(side_residual_list_positive, mid_residual_list_positive)
        self.wait()
        self.play(Transform(side_residual_list_positive, side_residual_list_encoded),
                  Transform(mid_residual_list_positive, mid_residual_list_encoded))
        self.remove(side_residual_list_positive, mid_residual_list_positive)
        self.add(side_residual_list_encoded, mid_residual_list_encoded)
        self.wait()

        explainer = Paragraph('This animation precisely represented an actual FLAC frame.\nSee the source code for more information.',
                              **small_font_args).scale_to_fit_width(10).shift(DOWN*3.5)
        explainer_bg = BackgroundRectangle(explainer)
        self.play(FadeIn(explainer_bg), Write(explainer))
        self.wait()

        self.play(FadeOut(*self.mobjects))
        self.wait()

        nextep = Text('Stay tuned for the final part!',
                      **font_args).shift(UP*3)
        ep2 = Text('Episode 1:\nWhat is digital audio?',
                   **small_font_args).shift(LEFT*4)
        ep3 = Text('Episode 3:\nThe FLAC stream format',
                   **small_font_args).shift(RIGHT*4)
        ep3.font_size = 0.6 * DEFAULT_FONT_SIZE
        ep2.font_size = 0.6 * DEFAULT_FONT_SIZE

        self.play(Write(nextep, run_time=DEFAULT_ANIMATION_RUN_TIME*2),
                  Write(ep3, run_time=DEFAULT_ANIMATION_RUN_TIME*2),
                  Write(ep2, run_time=DEFAULT_ANIMATION_RUN_TIME*2))
        self.wait()


class Thumbnail(Scene):
    def construct(self):
        interchannel_decorrelation = ImageMobject(
            '../Thumbnail InterchannelDecorrelation.png').set(width=5).shift(LEFT*4+UP*2)
        lpc = ImageMobject(
            '../Thumbnail LinearPredictiveCoding.png').set(width=5).shift(RIGHT*4+UP*2)
        residual = ImageMobject(
            '../Thumbnail Residual.png').set(width=5).shift(RIGHT*4+DOWN*2)
        interchannel_decorrelation_rect = Rectangle(
            color=theming_colors[0], width=interchannel_decorrelation.width, height=interchannel_decorrelation.height).move_to(interchannel_decorrelation)
        lpc_rect = Rectangle(
            color=theming_colors[0], width=lpc.width, height=lpc.height).move_to(lpc)
        residual_rect = Rectangle(
            color=theming_colors[0], width=residual.width, height=residual.height).move_to(residual)
        arrows = VGroup(
            Arrow(start=interchannel_decorrelation.get_right(), end=lpc.get_left(), stroke_width=6),
            Arrow(start=lpc.get_bottom(), end=residual.get_top(), stroke_width=6, max_stroke_width_to_length_ratio=8, max_tip_length_to_length_ratio=.45),
        )
        compression_text = Paragraph(
            'How to throw\naway 1,400,000\nbytes per second!', **font_args).shift(DOWN*2+LEFT*3)
        self.add(interchannel_decorrelation, lpc, residual, arrows,
                 interchannel_decorrelation_rect, lpc_rect, residual_rect, compression_text)


def load_wav_to_f64(filename: str) -> Tuple[int, np.array]:
    '''Loads a WAV file and converts it to 64-bit floats in the standard -1 to 1 range, if necessary.'''
    (rate, audio_data) = wavfile.read(
        filename, mmap=True)
    original_type = audio_data.dtype
    audio_data = audio_data.astype(np.float64)
    if original_type == np.int32:
        audio_data /= float(2**31)
    elif original_type == np.int16:
        audio_data /= float(2**15)
    elif original_type == np.uint8:
        audio_data /= float(2**7)
        audio_data -= 1.0
    return (rate, audio_data)
