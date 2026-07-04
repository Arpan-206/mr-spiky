"""
Mr. Spiky pitch clip 1: "A neuron reads real senior code, and reacts like
your brain does." (~32s)

The code is a real, mildly-simplified version of `linecache.checkcache`
from the CPython stdlib — chosen because it has a natural "reviewer
flinch" moment: a bare `try/except (OSError, ValueError)` that silently
mutates a module-level cache. The eye scans top-to-bottom, the membrane
accumulates, and both the reviewer marker (❗) and the neuron spike land
on that specific line at the same moment.

DESIGN RULES:
  - Composition stays visible for the whole clip. Bottom banner strip
    handles all narration changes.
  - Every mobject inside safe bounds x∈[-6.8, 6.8], y∈[-3.8, 3.8].
  - Eye + membrane rise in visible sync. Break sync, break claim.
  - The specific line that spikes (line index 8 — the bare except) is
    the ONLY line that pushes membrane over threshold. Earlier lines
    accumulate meaningfully but stay sub-threshold.

Render: manim -qm lif_membrane_over_lines.py LIFMembraneOverLines
"""

from manim import (
    Scene, Code, Circle, Dot, Text, DashedLine, Axes,
    VGroup, VMobject, Rectangle, RoundedRectangle,
    Write, FadeIn, FadeOut, Flash, Create, Transform,
    UP, DOWN, LEFT, RIGHT, ORIGIN, WHITE, GRAY, YELLOW, ORANGE, RED,
    smooth,
)
import numpy as np


# ---------------------------------------------------------------------------
# Safe frame bounds. Default 14.222 × 8.0, safety margin 0.3 all around.
# ---------------------------------------------------------------------------
X_MAX =  6.8
X_MIN = -6.8
Y_MAX =  3.8
Y_MIN = -3.8

BG        = "#0f0f0f"
SPIKE     = "#ffd54a"
SPIKE_HOT = "#ff9a3a"
NEURON_C  = "#ffd54a"
EYE_C     = "#7fd6ff"
LINE_MID  = "#3a3a3f"
DIM       = "#8a8a92"
FLINCH    = "#ff6b6b"

MAX_BANNER_W = 12.0


# ---------------------------------------------------------------------------
# Real code: simplified linecache.checkcache from the CPython stdlib.
# 12 lines. The bare `except (OSError, ValueError)` and silent
# `cache.pop(...)` on that path is what a senior reviewer flinches at.
# ---------------------------------------------------------------------------
CODE_LINES = [
    "def checkcache(filename=None):",
    "    if filename is None:",
    "        filenames = cache.copy().keys()",
    "    else:",
    "        filenames = [filename]",
    "    for filename in filenames:",
    "        entry = cache.get(filename)",
    "        if entry is None or len(entry) == 1:",
    "            continue",
    "        size, mtime, lines, fullname = entry",
    "        try:",
    "            stat = os.stat(fullname)",
    "        except (OSError, ValueError):",
    "            cache.pop(filename, None)",
]

# Per-line "structural intensity." Only the bare-except line
# (index 12) crosses threshold. Earlier lines accumulate meaningfully.
# 14 values, one per line.
BUMPS = [
    0.05,  # def signature
    0.12,  # if filename is None:
    0.16,  # cache.copy().keys() — mutates external state
    0.08,  # else
    0.08,  # filenames = [filename]
    0.14,  # for loop
    0.18,  # entry = cache.get(...) — accesses global cache
    0.20,  # compound condition
    0.06,  # continue (leaks a bit)
    0.14,  # unpack tuple
    0.10,  # try:
    0.12,  # os.stat(...)
    0.62,  # except (OSError, ValueError): ← THIS is the flinch line
    0.30,  # cache.pop — silent mutation on error path
]

LEAK    = 0.85    # keep more between lines so context builds visibly
THRESH  = 1.00


def clamp_to_frame(mob, name=""):
    left, right, top, bot = (
        mob.get_left()[0], mob.get_right()[0],
        mob.get_top()[1],  mob.get_bottom()[1],
    )
    assert left  >= X_MIN - 0.05, f"{name or mob} clips left  ({left:.2f})"
    assert right <= X_MAX + 0.05, f"{name or mob} clips right ({right:.2f})"
    assert top   <= Y_MAX + 0.05, f"{name or mob} clips top   ({top:.2f})"
    assert bot   >= Y_MIN - 0.05, f"{name or mob} clips bot   ({bot:.2f})"


class LIFMembraneOverLines(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ==================================================================
        # LAYOUT
        # ==================================================================
        LEFT_CX  = -3.6     # left panel center x — code is wide, sits left
        RIGHT_CX =  3.6     # right panel center x — neuron + axes on right
        BANNER_Y = -3.2

        # ==================================================================
        # HEADINGS
        # ==================================================================
        heading_left = Text("a reviewer reading real senior code", font_size=20, color=DIM)
        heading_left.set(width=min(heading_left.width, 5.6))
        heading_left.move_to([LEFT_CX, 3.4, 0])
        clamp_to_frame(heading_left, "heading_left")

        heading_right = Text("a spiking neuron", font_size=20, color=DIM)
        heading_right.set(width=min(heading_right.width, 5.4))
        heading_right.move_to([RIGHT_CX, 3.4, 0])
        clamp_to_frame(heading_right, "heading_right")

        self.play(FadeIn(heading_left), FadeIn(heading_right), run_time=0.6)

        # ==================================================================
        # LEFT PANEL — code + eye + gut meter
        # ==================================================================
        code_block = Code(
            code_string="\n".join(CODE_LINES),
            language="python",
            background="window",
            tab_width=4,
            formatter_style="monokai",
        ).scale(0.42)
        # Position code near top of left panel so all 14 lines fit above the
        # gut meter.
        code_block.move_to([LEFT_CX, 0.55, 0])
        # Guarantee width fits.
        if code_block.width > 6.4:
            code_block.set(width=6.4)
            code_block.move_to([LEFT_CX, 0.55, 0])
        clamp_to_frame(code_block, "code_block")

        # Gut-reaction bar — vertical strip along the *right* edge of the
        # code block (inside the panel), matching code height.
        code_top_y = code_block.get_top()[1]
        code_bot_y = code_block.get_bottom()[1]
        gut_bar_x  = code_block.get_right()[0] + 0.28
        gut_frame  = Rectangle(
            width=0.18,
            height=code_top_y - code_bot_y,
            color=LINE_MID, stroke_width=1.5,
        ).move_to([gut_bar_x, (code_top_y + code_bot_y) / 2, 0])
        gut_label = Text("gut", font_size=13, color=DIM)
        gut_label.next_to(gut_frame, DOWN, buff=0.10)
        clamp_to_frame(gut_frame, "gut_frame")
        clamp_to_frame(gut_label, "gut_label")

        def make_gut_fill(fill_frac):
            h = max(0.001, (code_top_y - code_bot_y) * fill_frac)
            fill = Rectangle(
                width=0.16, height=h,
                color=EYE_C, fill_opacity=0.85, stroke_width=0,
            )
            fill.move_to([gut_bar_x, code_bot_y + h / 2, 0])
            return fill
        gut_fill = make_gut_fill(0.0)

        # Eye
        eye = Dot(radius=0.09, color=EYE_C)
        eye_glow = Circle(radius=0.15, color=EYE_C, fill_opacity=0.28, stroke_width=0)
        eye_group = VGroup(eye_glow, eye)
        first_line_y = list(code_block.code_lines)[0].get_center()[1]
        eye_group.move_to([code_block.get_left()[0] - 0.15, first_line_y, 0])

        self.play(
            Create(code_block),
            Create(gut_frame),
            FadeIn(gut_label),
            FadeIn(gut_fill),
            FadeIn(eye_group),
            run_time=1.2,
        )

        # ==================================================================
        # RIGHT PANEL — neuron + membrane axes
        # ==================================================================
        neuron = Circle(radius=0.34, color=NEURON_C, stroke_width=3).set_fill(NEURON_C, opacity=0.15)
        neuron.move_to([RIGHT_CX, 2.1, 0])
        neuron_label = Text("V", font_size=22, color=NEURON_C, slant="ITALIC").move_to(neuron.get_center())
        clamp_to_frame(neuron, "neuron")

        axes = Axes(
            x_range=[0, 14, 2],
            y_range=[0, 1.3, 1],
            x_length=4.6,
            y_length=2.4,
            tips=False,
            axis_config={"color": LINE_MID, "stroke_width": 1.5, "include_ticks": False},
        )
        axes.move_to([RIGHT_CX, -0.55, 0])
        clamp_to_frame(axes, "axes")

        thresh_line = DashedLine(
            axes.c2p(0, THRESH),
            axes.c2p(14, THRESH),
            color=DIM, stroke_width=1.2, dash_length=0.06,
        )
        thresh_label = Text("threshold", font_size=13, color=DIM)
        thresh_label.next_to(axes.c2p(0, THRESH), UP, buff=0.05).align_to(axes, LEFT).shift(RIGHT * 0.1)
        clamp_to_frame(thresh_label, "thresh_label")

        membrane_trace = VMobject(color=SPIKE, stroke_width=3)
        membrane_trace.set_points_as_corners([axes.c2p(0, 0), axes.c2p(0.001, 0)])

        self.play(
            Create(neuron), FadeIn(neuron_label),
            Create(axes),
            Create(thresh_line), FadeIn(thresh_label),
            Create(membrane_trace),
            run_time=1.2,
        )

        # ==================================================================
        # BOTTOM BANNER
        # ==================================================================
        banner_bg = RoundedRectangle(
            width=12.6, height=0.9,
            corner_radius=0.15,
            color=LINE_MID, stroke_width=1.0,
        ).set_fill("#141416", opacity=0.9)
        banner_bg.move_to([0, BANNER_Y, 0])
        clamp_to_frame(banner_bg, "banner_bg")

        banner_text = Text("", font_size=22, color=WHITE)
        banner_text.move_to(banner_bg.get_center())
        self.play(FadeIn(banner_bg), run_time=0.35)

        def set_banner(new_str: str, color=WHITE, run_time: float = 0.5):
            nonlocal banner_text
            new_text = Text(new_str, font_size=22, color=color)
            new_text.set(width=min(new_text.width, MAX_BANNER_W))
            new_text.move_to(banner_bg.get_center())
            self.play(Transform(banner_text, new_text), run_time=run_time)

        set_banner("watch it read a real function from the Python stdlib.", DIM, run_time=0.6)
        self.wait(0.9)

        # ==================================================================
        # THE READ — line by line, slower than before, longer holds
        # ==================================================================
        set_banner("its 'gut' rises with each line — just like yours does.", DIM, run_time=0.6)
        self.wait(0.6)

        v = 0.0
        trace_points = [axes.c2p(0, 0)]
        code_line_mobs = list(code_block.code_lines)
        FLINCH_IDX = 12    # the "except (OSError, ValueError):" line (0-based)

        for idx, (line_mob, bump) in enumerate(zip(code_line_mobs, BUMPS)):
            target_y = line_mob.get_center()[1]
            new_eye_pos = np.array([code_block.get_left()[0] - 0.15, target_y, 0.0])

            # Between-line leak (small backward slope).
            if idx > 0:
                v_pre = v * LEAK
                trace_points.append(axes.c2p(idx - 0.4, v_pre))
                partial_trace = VMobject(color=SPIKE, stroke_width=3)
                partial_trace.set_points_as_corners(trace_points)
                new_gut_pre = make_gut_fill(v_pre / 1.3)
                self.play(
                    eye_group.animate.move_to(new_eye_pos),
                    Transform(gut_fill, new_gut_pre),
                    Transform(membrane_trace, partial_trace),
                    run_time=0.35, rate_func=smooth,
                )
                v = v_pre

            # Highlight the current line as the eye lands.
            self.play(line_mob.animate.set_fill(color=WHITE, opacity=1.0), run_time=0.08)

            v_post = v + bump
            trace_points.append(axes.c2p(idx + 1, v_post))
            new_trace = VMobject(color=SPIKE, stroke_width=3)
            new_trace.set_points_as_corners(trace_points)
            new_gut = make_gut_fill(min(1.0, v_post / 1.3))

            # Is this the flinch line?
            is_flinch = idx == FLINCH_IDX and v < THRESH and v_post >= THRESH
            if is_flinch:
                # Recolor trace at crossing point.
                new_trace.set_color(SPIKE_HOT)
                # Put a "!" marker to the right of the flinch line.
                flinch_marker = Text("❗", font_size=26, color=FLINCH)
                flinch_marker.next_to(line_mob, RIGHT, buff=0.15)
                # If it would clip the right of the code block, tuck it inside.
                if flinch_marker.get_right()[0] > gut_frame.get_left()[0] - 0.1:
                    flinch_marker.move_to([gut_frame.get_left()[0] - 0.28, line_mob.get_center()[1], 0])
                self.play(
                    Transform(gut_fill, new_gut),
                    Transform(membrane_trace, new_trace),
                    neuron.animate.set_stroke(SPIKE_HOT, width=5).set_fill(SPIKE_HOT, opacity=0.5),
                    Flash(neuron, color=SPIKE_HOT, flash_radius=0.55, num_lines=14, line_length=0.15),
                    FadeIn(flinch_marker, scale=1.4),
                    Flash(eye_group, color=FLINCH, flash_radius=0.35, num_lines=10, line_length=0.12),
                    run_time=0.75,
                )
                # Hold for a beat so audience registers the sync.
                self.wait(0.8)
                set_banner("both flinched on the same line.", SPIKE, run_time=0.5)
                self.wait(1.4)
            else:
                self.play(
                    Transform(gut_fill, new_gut),
                    Transform(membrane_trace, new_trace),
                    run_time=0.35,
                )
                # Micro-pause after each line so the pacing feels human.
                self.wait(0.15)
            v = v_post

        self.wait(0.5)

        # ==================================================================
        # PUNCHLINE — the moment they see the parallel
        # ==================================================================
        set_banner("same accumulation. same threshold. same reaction.", SPIKE, run_time=0.6)
        self.wait(2.2)

        # ==================================================================
        # WHERE THE INTUITION CAME FROM
        # ==================================================================
        set_banner("its sensitivity came from 2,680 senior-authored functions.", WHITE, run_time=0.55)
        self.wait(1.3)
        set_banner("reading a lot of code — same as you.", SPIKE, run_time=0.5)
        self.wait(2.0)

        # Clean fade.
        self.play(
            FadeOut(heading_left), FadeOut(heading_right),
            FadeOut(code_block), FadeOut(gut_frame), FadeOut(gut_fill),
            FadeOut(gut_label), FadeOut(eye_group),
            FadeOut(neuron), FadeOut(neuron_label),
            FadeOut(axes), FadeOut(thresh_line), FadeOut(thresh_label),
            FadeOut(membrane_trace),
            FadeOut(banner_bg), FadeOut(banner_text),
            run_time=0.7,
        )


if __name__ == "__main__":
    print("Render with: manim -qm lif_membrane_over_lines.py LIFMembraneOverLines")
