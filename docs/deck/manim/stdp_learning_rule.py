"""
Mr. Spiky pitch clip 2: "This is how your brain learns. Mr. Spiky learned
the same way." (~20s)

DESIGN RULES (do not break):
  - The two focal neurons + wire stay visible for the entire clip. No
    scene wipes. Punchlines land in a bottom banner strip below the
    composition.
  - Every mobject must sit inside safe bounds:
        x in [-6.8, 6.8]
        y in [-3.8, 3.8]
  - The wire thickness change MUST be visible each iteration. If subtle
    color-shift is enough, keep the stroke deltas linear so the eye
    tracks growth.

Render: manim -qm stdp_learning_rule.py STDPLearningRule
"""

from manim import (
    Scene, Circle, Text, Line, Rectangle, RoundedRectangle, VGroup,
    FadeIn, FadeOut, Flash, Create, Transform,
    UP, DOWN, LEFT, RIGHT, ORIGIN, WHITE, GRAY, YELLOW, GREEN, RED,
)
import numpy as np


X_MAX =  6.8
X_MIN = -6.8
Y_MAX =  3.8
Y_MIN = -3.8

BG        = "#0f0f0f"
SPIKE     = "#ffd54a"
DIM       = "#8a8a92"
NEURON_C  = "#ffd54a"
WIRE_C    = "#5a5a62"
WIRE_HOT  = "#ffd54a"

MAX_BANNER_W = 12.0


def clamp_to_frame(mob):
    left, right, top, bot = (
        mob.get_left()[0], mob.get_right()[0],
        mob.get_top()[1],  mob.get_bottom()[1],
    )
    assert left  >= X_MIN - 0.05, f"{mob} clips left  ({left:.2f})"
    assert right <= X_MAX + 0.05, f"{mob} clips right ({right:.2f})"
    assert top   <= Y_MAX + 0.05, f"{mob} clips top   ({top:.2f})"
    assert bot   >= Y_MIN - 0.05, f"{mob} clips bot   ({bot:.2f})"


def interpolate_hex(c1: str, c2: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    def hex_to_rgb(h): return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)
    r1, g1, b1 = hex_to_rgb(c1)
    r2, g2, b2 = hex_to_rgb(c2)
    r = round(r1 + (r2 - r1) * t)
    g = round(g1 + (g2 - g1) * t)
    b = round(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


class STDPLearningRule(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ==================================================================
        # LAYOUT
        # ==================================================================
        # Focal pair sits at y = 1.4. Banner at y = -3.2. Exposure log at
        # y = -1.2 (between them, and above the banner).
        NEURON_Y = 1.4
        BANNER_Y = -3.2
        LOG_Y    = -1.0

        # Ghost grid (small pairs behind the focal one) at y in [-2.4, 3.0].

        # ==================================================================
        # FOCAL PAIR — the anchor of the whole clip
        # ==================================================================
        neuron_a = Circle(radius=0.40, color=NEURON_C, stroke_width=3).set_fill(NEURON_C, opacity=0.15)
        neuron_a.move_to([-2.2, NEURON_Y, 0])
        label_a = Text("A", font_size=24, color=NEURON_C, weight="BOLD").move_to(neuron_a.get_center())

        neuron_b = Circle(radius=0.40, color=NEURON_C, stroke_width=3).set_fill(NEURON_C, opacity=0.15)
        neuron_b.move_to([2.2, NEURON_Y, 0])
        label_b = Text("B", font_size=24, color=NEURON_C, weight="BOLD").move_to(neuron_b.get_center())

        wire = Line(
            neuron_a.get_right(), neuron_b.get_left(),
            color=WIRE_C, stroke_width=2.5,
        )

        # Small caption above the pair.
        heading = Text("two neurons in Mr. Spiky", font_size=20, color=DIM)
        heading.set(width=min(heading.width, 6.0))
        heading.move_to([0, 3.2, 0])
        clamp_to_frame(heading)

        clamp_to_frame(neuron_a)
        clamp_to_frame(neuron_b)

        # ==================================================================
        # EXPOSURE LOG — a small strip that shows "senior code being fed in"
        # ==================================================================
        # A short horizontal row: the counter number, an arrow, and a "code
        # snippet" placeholder that changes as new exposures arrive.
        log_bg = RoundedRectangle(
            width=10.0, height=0.85,
            corner_radius=0.12,
            color=WIRE_C, stroke_width=1.0,
        ).set_fill("#141416", opacity=0.7)
        log_bg.move_to([0, LOG_Y, 0])
        clamp_to_frame(log_bg)

        log_caption_left = Text("exposure", font_size=15, color=DIM)
        log_caption_left.move_to([log_bg.get_left()[0] + 0.85, LOG_Y + 0.12, 0])

        log_counter = Text("0", font_size=28, color=SPIKE, weight="BOLD")
        log_counter.move_to([log_bg.get_left()[0] + 0.85, LOG_Y - 0.14, 0])

        log_arrow_x = log_bg.get_left()[0] + 1.7
        log_arrow = Text("→", font_size=24, color=DIM).move_to([log_arrow_x, LOG_Y, 0])

        log_snippet_placeholder = Text(
            "def parse(config):  # from django.utils",
            font_size=16, color=WHITE,
        )
        log_snippet_placeholder.move_to([log_arrow_x + 3.4, LOG_Y, 0])
        log_snippet_placeholder.set(width=min(log_snippet_placeholder.width, 6.4))
        clamp_to_frame(log_snippet_placeholder)

        # ==================================================================
        # BOTTOM BANNER
        # ==================================================================
        banner_bg = RoundedRectangle(
            width=12.6, height=0.9,
            corner_radius=0.15,
            color=WIRE_C, stroke_width=1.0,
        ).set_fill("#141416", opacity=0.9)
        banner_bg.move_to([0, BANNER_Y, 0])
        clamp_to_frame(banner_bg)

        banner_text = Text("", font_size=22, color=WHITE)
        banner_text.move_to(banner_bg.get_center())

        # ==================================================================
        # FADE THE COMPOSITION IN
        # ==================================================================
        self.play(
            FadeIn(heading),
            FadeIn(neuron_a), FadeIn(label_a),
            FadeIn(neuron_b), FadeIn(label_b),
            Create(wire),
            FadeIn(log_bg), FadeIn(log_caption_left), FadeIn(log_counter),
            FadeIn(log_arrow), FadeIn(log_snippet_placeholder),
            FadeIn(banner_bg),
            run_time=1.0,
        )

        # ==================================================================
        # Helper: replace the banner's text in place
        # ==================================================================
        def set_banner(new_str: str, color=WHITE, run_time: float = 0.4):
            nonlocal banner_text
            new_text = Text(new_str, font_size=22, color=color)
            new_text.set(width=min(new_text.width, MAX_BANNER_W))
            new_text.move_to(banner_bg.get_center())
            self.play(Transform(banner_text, new_text), run_time=run_time)

        def set_snippet(new_str: str):
            nonlocal log_snippet_placeholder
            new_snippet = Text(new_str, font_size=16, color=WHITE)
            new_snippet.move_to([log_arrow_x + 3.4, LOG_Y, 0])
            if new_snippet.width > 6.4:
                new_snippet.set(width=6.4)
            self.play(Transform(log_snippet_placeholder, new_snippet), run_time=0.18)

        def set_counter(n: int):
            nonlocal log_counter
            new_c = Text(str(n), font_size=28, color=SPIKE, weight="BOLD")
            new_c.move_to([log_bg.get_left()[0] + 0.85, LOG_Y - 0.14, 0])
            self.play(Transform(log_counter, new_c), run_time=0.12)

        # ==================================================================
        # BEAT 1 — Repeated exposures, wire visibly thickens (0..10s)
        # ==================================================================
        set_banner("A fires. B fires shortly after. Repeat.", DIM, run_time=0.5)

        NUM_EXPOSURES = 6
        # Realistic-ish snippet labels to sell "senior code being fed in."
        SNIPPETS = [
            "def parse(config):  # django.utils",
            "async def request(self, ...):  # httpx",
            "def compile(node):  # cpython/ast",
            "def _flatten(x):  # pydantic",
            "def dispatch(handler):  # flask",
            "class Session:  # sqlalchemy",
        ]

        for i in range(NUM_EXPOSURES):
            # Rotate the snippet so the log looks live.
            set_snippet(SNIPPETS[i])
            # A fires first.
            self.play(
                neuron_a.animate.set_stroke(SPIKE, width=6).set_fill(SPIKE, opacity=0.55),
                Flash(neuron_a, color=SPIKE, flash_radius=0.55, num_lines=10, line_length=0.15),
                run_time=0.20,
            )
            # A relaxes, B fires.
            self.play(
                neuron_a.animate.set_stroke(NEURON_C, width=3).set_fill(NEURON_C, opacity=0.15),
                neuron_b.animate.set_stroke(SPIKE, width=6).set_fill(SPIKE, opacity=0.55),
                Flash(neuron_b, color=SPIKE, flash_radius=0.55, num_lines=10, line_length=0.15),
                run_time=0.25,
            )
            # Wire thickens.
            fraction = (i + 1) / NUM_EXPOSURES
            new_width = 2.5 + fraction * 12  # 4.5 -> 14.5
            wire_color = interpolate_hex(WIRE_C, WIRE_HOT, fraction)
            new_wire = Line(
                neuron_a.get_right(), neuron_b.get_left(),
                color=wire_color, stroke_width=new_width,
            )
            self.play(
                neuron_b.animate.set_stroke(NEURON_C, width=3).set_fill(NEURON_C, opacity=0.15),
                Transform(wire, new_wire),
                run_time=0.20,
            )
            set_counter(i + 1)

        self.wait(0.5)

        # ==================================================================
        # BEAT 2 — Hebb quote in banner + subtle ghost grid (10..15s)
        # ==================================================================
        set_banner('"neurons that fire together, wire together." — Donald Hebb, 1949',
                   SPIKE, run_time=0.6)

        # A subtle grid of ghost-pairs appears BEHIND the focal pair — same
        # composition space, just dimmer, to convey "this happens in parallel
        # everywhere in the network." Only 6 ghosts so it doesn't crowd.
        ghosts = VGroup()
        ghost_positions = [
            (-5.0,  2.6), (-5.0, -0.3),
            ( 5.0,  2.6), ( 5.0, -0.3),
            (-5.0,  0.5), ( 5.0,  0.5),
        ]
        for gx, gy in ghost_positions:
            a = Circle(radius=0.12, color=NEURON_C, stroke_width=1.5).set_fill(NEURON_C, opacity=0.10).move_to([gx - 0.35, gy, 0])
            b = Circle(radius=0.12, color=NEURON_C, stroke_width=1.5).set_fill(NEURON_C, opacity=0.10).move_to([gx + 0.35, gy, 0])
            w = Line(a.get_right(), b.get_left(), color=WIRE_HOT, stroke_width=2.5).set_opacity(0.4)
            ghost = VGroup(a, b, w)
            ghost.set_opacity(0.0)
            ghosts.add(ghost)
        self.add(ghosts)
        # Fade all ghosts in gently.
        self.play(*[g.animate.set_opacity(0.55) for g in ghosts], run_time=0.5)

        self.wait(1.5)

        # ==================================================================
        # BEAT 3 — Punchline (15..20s)
        # ==================================================================
        set_banner("this is how your visual cortex learned to read.", WHITE, run_time=0.5)
        self.wait(1.2)
        set_banner("and it's how Mr. Spiky learned senior Python.", SPIKE, run_time=0.5)
        self.wait(1.6)

        # Small honest-methodology tag — moved INTO the banner instead of
        # below it, so it never fights the frame edge. Fades in beneath
        # the main banner text as a second, smaller line.
        tag = Text("no teacher. no labels. no gradient descent.", font_size=13, color=DIM)
        tag.move_to(banner_bg.get_center() + np.array([0, -0.28, 0]))
        # Shrink the main banner text upward slightly to make room.
        raised_banner = Text("and it's how Mr. Spiky learned senior Python.", font_size=22, color=SPIKE)
        raised_banner.set(width=min(raised_banner.width, MAX_BANNER_W))
        raised_banner.move_to(banner_bg.get_center() + np.array([0, 0.14, 0]))
        clamp_to_frame(tag)
        self.play(Transform(banner_text, raised_banner), FadeIn(tag), run_time=0.4)

        self.wait(1.5)

        # Clean fade.
        self.play(
            FadeOut(heading),
            FadeOut(neuron_a), FadeOut(label_a),
            FadeOut(neuron_b), FadeOut(label_b),
            FadeOut(wire),
            FadeOut(log_bg), FadeOut(log_caption_left), FadeOut(log_counter),
            FadeOut(log_arrow), FadeOut(log_snippet_placeholder),
            FadeOut(banner_bg), FadeOut(banner_text),
            FadeOut(tag), FadeOut(ghosts),
            run_time=0.6,
        )


if __name__ == "__main__":
    print("Render with: manim -qm stdp_learning_rule.py STDPLearningRule")
