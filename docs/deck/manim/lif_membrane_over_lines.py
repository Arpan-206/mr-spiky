"""
Mr. Spiky pitch clip 1: "A neuron reads code the way you do." (~22s)

DESIGN RULES (do not break):
  - The composition stays visible for the entire clip. No cuts to blank
    scenes. Punchlines land in a bottom banner strip, they don't replace
    the visual.
  - Every mobject must sit inside safe bounds:
        x in [-6.8, 6.8]
        y in [-3.8, 3.8]
    Manim's default frame is 14.222 wide × 8.0 tall (aspect 16:9), so
    those bounds leave ~0.3 units of safety. Wide Text is bounded by
    calling .set(width=W) after construction.
  - The reviewer's eye and the neuron's membrane trace rise in visible
    sync. That's the whole point of the clip. Break the sync, break
    the claim.

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
# Safe frame bounds. Manim's default frame is 14.222 × 8.0. Constrain to a
# rectangle 0.3 units inside that on every side so text never clips.
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

# Text that could get wide is bounded by .set(width=...) after construction.
MAX_HEADING_W = 5.4    # each panel is ~6.5 wide with 0.5 buff, so 5.4 fits comfortably
MAX_BANNER_W  = 12.0   # the bottom banner spans most of the frame


CODE_LINES = [
    "def process(items):",
    "    result = []",
    "    for item in items:",
    "        if item.valid and item.score > threshold:",
]

# Per-line "structural intensity." Only the last line pushes the membrane
# over threshold. Earlier lines contribute meaningful sub-threshold
# accumulation so the "context builds up" story reads.
BUMPS   = [0.22, 0.18, 0.38, 0.62]
LEAK    = 0.78    # fraction retained between line inputs
THRESH  = 1.00


def clamp_to_frame(mob):
    """Assert helper — verify a mobject stays inside safe bounds after we
    place it. Runs at construct time so a bad placement fails loudly
    instead of silently clipping in the rendered mp4."""
    left, right, top, bot = (
        mob.get_left()[0], mob.get_right()[0],
        mob.get_top()[1],  mob.get_bottom()[1],
    )
    assert left  >= X_MIN - 0.05, f"{mob} clips left  ({left:.2f} < {X_MIN})"
    assert right <= X_MAX + 0.05, f"{mob} clips right ({right:.2f} > {X_MAX})"
    assert top   <= Y_MAX + 0.05, f"{mob} clips top   ({top:.2f} > {Y_MAX})"
    assert bot   >= Y_MIN - 0.05, f"{mob} clips bot   ({bot:.2f} < {Y_MIN})"


class LIFMembraneOverLines(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ==================================================================
        # LAYOUT — Fixed regions, computed once, honored for the whole clip.
        # ==================================================================
        # Bottom banner strip lives at y in [-3.6, -2.8]. Everything else
        # lives above y = -2.6 so the banner never overlaps the composition.
        BANNER_Y = -3.2

        # Left panel occupies x in [-6.6, -0.2]; right panel x in [0.2, 6.6].
        LEFT_CX  = -3.4     # left panel center x
        RIGHT_CX =  3.4     # right panel center x

        # Vertical: headings at y=3.4, main content y=1.6 down to y=-2.4.

        # ==================================================================
        # HEADINGS — small, centered above each panel
        # ==================================================================
        heading_left = Text("a reviewer reading code", font_size=22, color=DIM)
        heading_left.set(width=min(heading_left.width, MAX_HEADING_W))
        heading_left.move_to([LEFT_CX, 3.4, 0])
        clamp_to_frame(heading_left)

        heading_right = Text("a spiking neuron", font_size=22, color=DIM)
        heading_right.set(width=min(heading_right.width, MAX_HEADING_W))
        heading_right.move_to([RIGHT_CX, 3.4, 0])
        clamp_to_frame(heading_right)

        self.play(FadeIn(heading_left), FadeIn(heading_right), run_time=0.5)

        # ==================================================================
        # LEFT PANEL — code + eye + gut meter
        # ==================================================================
        code_block = Code(
            code_string="\n".join(CODE_LINES),
            language="python",
            background="window",
            tab_width=4,
            formatter_style="monokai",
        ).scale(0.55)
        # Center code horizontally in the left panel, and position vertically.
        code_block.move_to([LEFT_CX, 1.1, 0])
        # Guarantee it fits — if the code renders wider than we expect, shrink.
        if code_block.width > 5.6:
            code_block.set(width=5.6)
        clamp_to_frame(code_block)

        # Gut-reaction bar — narrow vertical, sitting to the left of the code.
        gut_bar_x = code_block.get_left()[0] - 0.5
        gut_bar_top    = code_block.get_top()[1]
        gut_bar_bot    = code_block.get_bottom()[1]
        gut_frame = Rectangle(
            width=0.22,
            height=gut_bar_top - gut_bar_bot,
            color=LINE_MID, stroke_width=1.5,
        ).move_to([gut_bar_x, (gut_bar_top + gut_bar_bot) / 2, 0])
        gut_label = Text("gut", font_size=14, color=DIM)
        gut_label.next_to(gut_frame, DOWN, buff=0.12)
        clamp_to_frame(gut_frame)
        clamp_to_frame(gut_label)

        # Gut fill — a rectangle inside gut_frame; height animates.
        def make_gut_fill(fill_frac):
            h = max(0.001, (gut_bar_top - gut_bar_bot) * fill_frac)
            fill = Rectangle(
                width=0.20, height=h,
                color=EYE_C, fill_opacity=0.85, stroke_width=0,
            )
            fill.move_to([gut_bar_x, gut_bar_bot + h / 2, 0])
            return fill
        gut_fill = make_gut_fill(0.0)

        # The eye — a dot that lands on each code line in sequence.
        eye = Dot(radius=0.10, color=EYE_C)
        eye_glow = Circle(radius=0.16, color=EYE_C, fill_opacity=0.28, stroke_width=0)
        eye_group = VGroup(eye_glow, eye)
        # Position at first line initially.
        first_line_y = list(code_block.code_lines)[0].get_center()[1]
        eye_group.move_to([code_block.get_left()[0] - 0.15, first_line_y, 0])
        # But start invisible; fade in with the panel.

        self.play(
            Create(code_block),
            Create(gut_frame),
            FadeIn(gut_label),
            FadeIn(gut_fill),
            FadeIn(eye_group),
            run_time=1.0,
        )

        # ==================================================================
        # RIGHT PANEL — neuron + membrane axes
        # ==================================================================
        neuron = Circle(radius=0.32, color=NEURON_C, stroke_width=3).set_fill(NEURON_C, opacity=0.15)
        neuron.move_to([RIGHT_CX, 2.15, 0])
        neuron_label = Text("V", font_size=20, color=NEURON_C, slant="ITALIC").move_to(neuron.get_center())
        clamp_to_frame(neuron)

        # Membrane trace axes — sits below the neuron, safely inside bounds.
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 1.3, 1],
            x_length=4.4,
            y_length=2.2,
            tips=False,
            axis_config={"color": LINE_MID, "stroke_width": 1.5, "include_ticks": False},
        )
        axes.move_to([RIGHT_CX, -0.35, 0])
        clamp_to_frame(axes)

        thresh_line = DashedLine(
            axes.c2p(0, THRESH),
            axes.c2p(4, THRESH),
            color=DIM, stroke_width=1.2, dash_length=0.06,
        )
        thresh_label = Text("threshold", font_size=14, color=DIM)
        thresh_label.next_to(axes.c2p(0, THRESH), UP, buff=0.05).align_to(axes, LEFT).shift(RIGHT * 0.1)
        clamp_to_frame(thresh_label)

        membrane_trace = VMobject(color=SPIKE, stroke_width=3)
        membrane_trace.set_points_as_corners([axes.c2p(0, 0), axes.c2p(0.001, 0)])

        self.play(
            Create(neuron), FadeIn(neuron_label),
            Create(axes),
            Create(thresh_line), FadeIn(thresh_label),
            Create(membrane_trace),
            run_time=1.0,
        )

        # ==================================================================
        # BOTTOM BANNER — a persistent strip that changes text across beats
        # ==================================================================
        banner_bg = RoundedRectangle(
            width=12.6, height=0.9,
            corner_radius=0.15,
            color=LINE_MID, stroke_width=1.0,
        ).set_fill("#141416", opacity=0.9)
        banner_bg.move_to([0, BANNER_Y, 0])
        clamp_to_frame(banner_bg)

        banner_text = Text("", font_size=22, color=WHITE)
        banner_text.move_to(banner_bg.get_center())

        self.play(FadeIn(banner_bg), run_time=0.35)

        def set_banner(new_str: str, color=WHITE, run_time: float = 0.35):
            """Fade the banner's text to a new string, in place."""
            nonlocal banner_text
            new_text = Text(new_str, font_size=22, color=color)
            new_text.set(width=min(new_text.width, MAX_BANNER_W))
            new_text.move_to(banner_bg.get_center())
            self.play(Transform(banner_text, new_text), run_time=run_time)

        # Beat 1 banner.
        set_banner("watch these two rise together.", DIM)

        # ==================================================================
        # BEAT 1 — Synchronized reading (0..8s)
        # Eye moves down code lines. Membrane and gut fill rise in sync.
        # ==================================================================
        v = 0.0
        trace_points = [axes.c2p(0, 0)]
        code_line_mobs = list(code_block.code_lines)

        for idx, (line_mob, bump) in enumerate(zip(code_line_mobs, BUMPS)):
            target_y = line_mob.get_center()[1]
            new_eye_pos = np.array([code_block.get_left()[0] - 0.15, target_y, 0.0])

            # Between-line leak.
            if idx > 0:
                v_pre = v * LEAK
                trace_points.append(axes.c2p(idx - 0.5, v_pre))
                partial_trace = VMobject(color=SPIKE, stroke_width=3)
                partial_trace.set_points_as_corners(trace_points)
                new_gut_pre = make_gut_fill(v_pre / 1.3)  # normalize by y_max=1.3
                self.play(
                    eye_group.animate.move_to(new_eye_pos),
                    Transform(gut_fill, new_gut_pre),
                    Transform(membrane_trace, partial_trace),
                    run_time=0.45, rate_func=smooth,
                )
                v = v_pre

            # The line highlights as the eye lands.
            self.play(line_mob.animate.set_fill(color=WHITE, opacity=1.0), run_time=0.08)

            v_post = v + bump
            trace_points.append(axes.c2p(idx + 1, v_post))
            new_trace = VMobject(color=SPIKE, stroke_width=3)
            new_trace.set_points_as_corners(trace_points)
            new_gut = make_gut_fill(min(1.0, v_post / 1.3))

            if v_post >= THRESH and v < THRESH:
                # Threshold crossing — recolor + flash both sides.
                new_trace.set_color(SPIKE_HOT)
                self.play(
                    Transform(gut_fill, new_gut),
                    Transform(membrane_trace, new_trace),
                    neuron.animate.set_stroke(SPIKE_HOT, width=5).set_fill(SPIKE_HOT, opacity=0.5),
                    Flash(neuron, color=SPIKE_HOT, flash_radius=0.5, num_lines=14, line_length=0.15),
                    Flash(eye_group, color=RED, flash_radius=0.35, num_lines=10, line_length=0.12),
                    run_time=0.55,
                )
            else:
                self.play(
                    Transform(gut_fill, new_gut),
                    Transform(membrane_trace, new_trace),
                    run_time=0.45,
                )
            v = v_post

        self.wait(0.4)

        # ==================================================================
        # BEAT 2 — Punchline in the banner (no scene wipe) (8..14s)
        # ==================================================================
        set_banner("same accumulation. same threshold. same reaction.", SPIKE, run_time=0.55)
        self.wait(1.6)

        # ==================================================================
        # BEAT 3 — Where the intuition came from (14..22s)
        # Small yellow "past function" dots stream in from off-frame toward
        # the neuron. Composition still visible; banner changes.
        # ==================================================================
        set_banner("its sensitivity was trained on 2,680 senior-authored functions.", WHITE, run_time=0.5)

        # Corpus dots stream in from the right edge toward the neuron.
        rng = np.random.default_rng(11)
        corpus_dots = VGroup()
        for _ in range(24):
            # Enter from the right side of the right panel, at random y within safe bounds.
            enter_y = rng.uniform(-1.8, 3.0)
            enter_pt = np.array([X_MAX - 0.2, enter_y, 0.0])
            d = Dot(radius=0.055, color=SPIKE, fill_opacity=0.0).move_to(enter_pt)
            corpus_dots.add(d)
        self.add(corpus_dots)

        # Animate them: fade in at entry, drift toward the neuron, fade out.
        for k, d in enumerate(corpus_dots):
            neuron_pt = neuron.get_center()
            # Slight fan-in offset so they don't stack.
            offset = 0.02 * (k - 12)
            end_pt = neuron_pt + np.array([offset, offset * 0.3, 0.0])
            self.play(
                d.animate.set_opacity(0.85),
                run_time=0.05,
            )
            self.play(
                d.animate.move_to(end_pt).set_opacity(0.0),
                run_time=0.15,
            )

        self.wait(0.4)
        set_banner('reading a lot of code — same as you.', SPIKE, run_time=0.5)

        self.wait(1.8)

        # Clean fade so the closing frame is crisp.
        self.play(
            FadeOut(heading_left), FadeOut(heading_right),
            FadeOut(code_block), FadeOut(gut_frame), FadeOut(gut_fill),
            FadeOut(gut_label), FadeOut(eye_group),
            FadeOut(neuron), FadeOut(neuron_label),
            FadeOut(axes), FadeOut(thresh_line), FadeOut(thresh_label),
            FadeOut(membrane_trace),
            FadeOut(banner_bg), FadeOut(banner_text),
            run_time=0.6,
        )


if __name__ == "__main__":
    print("Render with: manim -qm lif_membrane_over_lines.py LIFMembraneOverLines")
