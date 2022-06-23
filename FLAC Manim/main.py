# SPDX-License-Identifier: Apache-2.0

from itertools import repeat
from manim import *
from manim.animation.animation import DEFAULT_ANIMATION_RUN_TIME
from waveformshow import show_waveform_for
from random import Random
from functools import partial

FONT = 'Candara'
# This font is only available on Windows, choose something else for other systems.
MONOSPACE_FONT = 'Cascadia Code'

font_args = {
    'font': FONT,
    'font_size': 1.5 * DEFAULT_FONT_SIZE,
    'weight': LIGHT
}
small_font_args = font_args.copy()
small_font_args['font_size'] = DEFAULT_FONT_SIZE
mono_font_args = small_font_args.copy()
mono_font_args['font'] = MONOSPACE_FONT
mono_font_args['font_size'] = DEFAULT_FONT_SIZE * 0.6
mono_font_args['weight'] = NORMAL

# Theming colors are used in this order of priority, meaning that color 1 is generally used more often than color 2
# 0: Main accent color, (sample) data, functions, bit depth
# 1: Grids, highlight, sometimes functions
# 2: Secondary accent color, functions, approximators & construction functions
# 3: Highlight 1 (when grid is active)
# 4: Residuals, anything else
# 5: Highlight 2 (when grid is active and 3 is taken)
theming_colors = [BLUE, ORANGE, GREEN, YELLOW, TEAL, PURPLE]
TRANSPARENT = rgba_to_color([0, 0, 0, 0])


class FlacIntro(Scene):
    def construct(self):
        self.next_section()
        audio = Text(r'Audio', **font_args)
        audiophiles = MarkupText(
            r'Audiophiles  ', color=WHITE, font=FONT, font_size=1.5 * DEFAULT_FONT_SIZE).shift(DOWN)
        # The space in the markup matches the audiophiles text exactly for Candara.
        audiophiles_strike = MarkupText(
            r'<s>                                </s>', color=WHITE, font=FONT, font_size=1.5 * DEFAULT_FONT_SIZE).shift(DOWN)
        audiofiles = Text(r'audio files', font=FONT).shift(DOWN)
        self.play(Write(audio))
        self.play(ApplyMethod(audio.shift, UP*2))
        self.play(Write(audiophiles))
        self.play(GrowFromEdge(audiophiles_strike, edge=LEFT))
        audiophiles_entire = Group(audiophiles, audiophiles_strike)
        self.play(Transform(audiophiles_entire, audiofiles))
        self.wait()
        self.remove(audiophiles_entire)
        self.play(Unwrite(audiofiles))
        self.wait()

        wav, mp3, ogg, aac = Text('WAV', **font_args).shift(LEFT*4.5), Text('MP3', **font_args).shift(
            LEFT*1.5), Text('OGG', **font_args).shift(RIGHT*1.5), Text('AAC', **font_args).shift(RIGHT*4.5)
        audio_formats = [wav, mp3, ogg, aac]
        for audio_format in audio_formats:
            self.play(Write(audio_format))
        aformats_group = Group(*audio_formats, audio)
        self.wait()
        self.play(ApplyMethod(aformats_group.shift, UP))

        flac = Text('FLAC', **font_args).shift(DOWN)
        flac_full = Text('Free Lossless Audio Codec', **font_args).shift(DOWN)
        self.play(Write(flac))
        self.play(TransformMatchingShapes(flac, flac_full))

        self.wait()

        self.remove(flac)
        self.play(FadeOut(flac_full, aformats_group))

        whyflac = Text(r'But why FLAC?', **font_args).shift(UP*2.5)
        reasons = [
            Text(r'• Free and open-source standard+software', **font_args),
            Text(r'• Not well-covered', **font_args),
            Text(r'• Motivation to learn digital audio, data compression, …', **font_args),
            Text(r'• I have written a FLAC implementation.', **font_args),
        ]
        self.play(Write(whyflac))
        # helper object that is never actually shown
        alignment_point = Point(LEFT*6, color=BLACK)
        for index, reason in enumerate(reasons):
            reason.font_size = 0.8*DEFAULT_FONT_SIZE
            reason.align_to(alignment_point, LEFT).align_to(
                alignment_point, UP).shift(UP*0.9*(1.5 - index))
            self.play(Write(reason))
        self.play(FadeOut(*reasons))
        self.wait()

        self.next_section("Compression basics")

        flac = Text(r'FLAC', **font_args).shift(UP*2.5)
        self.play(TransformMatchingShapes(whyflac, flac))
        self.remove(flac, whyflac)
        self.add(flac)
        reasons = [
            Text(r'• 2001', **font_args),
            Text(r'• No DRM', **font_args),
            Text(r'• Fast decoding and encoding', **font_args),
            Text(r'• Ideal for CD', **font_args),
        ]
        for index, reason in enumerate(reasons):
            reason.font_size = 0.9*DEFAULT_FONT_SIZE
            reason.align_to(alignment_point, LEFT).align_to(
                alignment_point, UP).shift(UP*0.9*(1.5 - index))
            self.play(Write(reason))
        self.play(FadeOut(*reasons))
        self.wait()

        filetypes = Text(r'Three kinds of files', **font_args).shift(UP*2.5)
        self.play(Transform(flac, filetypes))
        self.remove(filetypes, flac)
        self.add(filetypes)
        types = [
            Text(r'Uncompressed', **small_font_args),
            Text(r'Lossless', **small_font_args),
            Text(r'Lossy', **small_font_args),
        ]
        alignment_point = Point(UP)
        for index, type_ in enumerate(types):
            type_.align_to(alignment_point, UP).shift(RIGHT*(index-1)*4.5)
            self.play(Write(type_))

        uncompressed_highlights = Rectangle(
            height=1.2, width=5).align_to(types[0], LEFT).align_to(types[0], UP).shift(LEFT*0.4, UP*0.4)
        uncompressed_highlights
        self.play(Create(uncompressed_highlights))

        uncompressed = Text(r'BMP, WAV, CDA*', **font_args).align_to(
            types[0], UP, alignment_vect=UP).align_to(types[0], LEFT, alignment_vect=LEFT).shift(DOWN*0.8, LEFT*2.4)
        uncompressed.font_size = DEFAULT_FONT_SIZE*.5
        self.play(Write(uncompressed))
        self.wait()
        self.next_section("Lossless Intermission end")

        lossless_highlights = Rectangle(height=1.2, width=3).align_to(
            types[1], LEFT).align_to(types[1], UP).shift(LEFT*0.4, UP*0.4)
        self.play(Transform(uncompressed_highlights, lossless_highlights))
        self.remove(uncompressed_highlights, lossless_highlights)
        self.add(lossless_highlights)
        lossless = MarkupText(r'PNG, <b>FLAC</b>, VP9', **font_args).align_to(
            types[1], UP, alignment_vect=UP).align_to(types[1], LEFT, alignment_vect=LEFT).shift(DOWN*0.8, LEFT*2.4)
        lossless.font_size = DEFAULT_FONT_SIZE*.5
        self.play(Write(lossless))

        lossy_highlights = Rectangle(height=1.2, width=2.4).align_to(
            types[2], LEFT).align_to(types[2], UP).shift(LEFT*0.4, UP*0.4)
        self.play(Transform(lossless_highlights, lossy_highlights))
        self.remove(lossless_highlights, lossy_highlights)
        self.add(lossy_highlights)
        lossy = MarkupText(r'JPEG, MP3, H.264', **font_args).align_to(
            types[2], UP, alignment_vect=UP).align_to(types[2], LEFT, alignment_vect=LEFT).shift(DOWN*0.8, LEFT*2.4)
        lossy.font_size = DEFAULT_FONT_SIZE*.5
        self.play(Write(lossy))

        self.play(FadeOut(lossy_highlights, uncompressed, lossy, lossless))
        compressed_group = Group(*types[1:])

        bracket_text = BraceText(compressed_group, r'Compressed', label_constructor=partial(
            Text, **font_args), font_size=small_font_args['font_size']).shift(DOWN*0.2)
        bracket_text.font_size = .8*DEFAULT_FONT_SIZE
        self.play(Write(bracket_text))


class LosslessIntermission(Scene):
    def construct(self):
        self.next_section()
        rows, cols = 10, 40
        repeats = 4
        rng = Random(29893)
        number_grid = VGroup()
        for _ in range(rows):
            row = ' '.join((str(rng.choice([0, 1])) for _ in range(cols)))
            number = Tex(row)
            number.font_size = DEFAULT_FONT_SIZE*0.8
            number_grid.add(number)
        number_grid.arrange_in_grid(
            rows=rows, cols=1, buff=(0.02, 0.2))
        self.play(Write(number_grid))

        regular_number_grid = VGroup()
        row_numbers = []
        for _ in range(rows):
            # Each row has a single random binary number repeated four times.
            row_number = ''.join(x + ' ' for x in ('{:0'+str(cols//repeats) +
                                                   'b}').format(rng.randint(0, 2**(cols//repeats))))
            row_numbers.append(row_number)
            row = row_number * repeats
            number = Tex(row)
            number.font_size = DEFAULT_FONT_SIZE*0.8
            regular_number_grid.add(number)
        regular_number_grid.arrange_in_grid(
            rows=rows, cols=1, buff=0.2)
        self.play(Transform(number_grid, regular_number_grid))
        # Transform does funky things, let's ensure that we have the correct bitstring object on screen.
        self.remove(regular_number_grid, number_grid)
        self.add(regular_number_grid)

        separated_number_grid = VGroup()
        for row_number in row_numbers:
            row_obj = VGroup()
            for i in range(repeats):
                number = Tex(row_number)
                number.font_size = DEFAULT_FONT_SIZE*0.8
                row_obj.add(number)
            row_obj.arrange_in_grid(rows=1, cols=repeats, buff=0.4)
            separated_number_grid.add(row_obj)
        separated_number_grid.arrange_in_grid(
            rows=rows, cols=1, buff=0.2)
        self.play(TransformMatchingShapes(
            regular_number_grid, separated_number_grid))
        self.remove(separated_number_grid, regular_number_grid)
        self.add(separated_number_grid)

        last_row = separated_number_grid.submobjects[-1]
        brace_focus = last_row.submobjects[0]
        column_bracket = BraceText(brace_focus, r'Repeated column', label_constructor=partial(
            Text, **font_args), font_size=small_font_args['font_size']*.7).shift(DOWN*0.2)
        self.play(Write(column_bracket))

        self.wait()
        self.play(FadeOut(column_bracket, separated_number_grid))
        self.next_section("Text compression")

        long_text = self.make_long_text(color=False)
        self.play(Write(long_text))
        colored_long_text = self.make_long_text(color=True)
        self.play(Transform(long_text, colored_long_text))
        self.remove(colored_long_text, long_text)
        self.add(colored_long_text)

        compression_text = Text(
            '“the” on [line] 2 [column] 11, 3 9, 4 21, 5 24, 5 34, …', **small_font_args).shift(DOWN*3)
        compression_text.font_size = small_font_args['font_size']*.5
        compression_box = Rectangle(width=8, height=1).shift(DOWN*3)
        self.play(FadeIn(compression_text, compression_box))

    def make_long_text(self, color: bool) -> Text:
        colors = {"the": theming_colors[0]} if color else {}
        long_text = Text('''For example, imagine a piece of English text.
The word “the” would be pretty common, right?
So what the text compressing format can do is
instead of storing “the” a thousand times,
it just stores: “well, the word “the”
should be at this position, that position,
and the other 998 positions”.
That will take up much less space.
In fact: it’s pretty much what the ZIP* does.''', **small_font_args, t2c=colors, line_spacing=1.05, disable_ligatures=True).shift(UP)
        long_text.font_size = small_font_args['font_size']*.65
        return long_text


class WhatIsAudio(Scene):
    def construct(self):
        show_waveform_for(self, "Bad Apple.wav", time=30)
