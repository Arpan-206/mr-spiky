"""
Mr. Spiky pitch clip 2: "Different pairs specialize on different patterns —
same rule your brain uses." (~30s)

Three neurons on screen: A, B, C. A always fires when a senior function
starts. B fires on 'exception handling' patterns. C fires on 'dense math'
patterns. When A and B happen to co-fire (senior code with exceptions),
the A→B wire strengthens. When A and C happen to co-fire on a *different*
kind of function, THAT wire strengthens instead. After several exposures
mixed across patterns, A→B is strong and A→C is weak — visual proof of
specialization.

DESIGN RULES:
  - All three neurons visible for the entire clip.
  - Every mobject inside safe bounds x∈[-6.8, 6.8], y∈[-3.8, 3.8].
  - The two wires (A→B and A→C) must end up VISIBLY different — that's
    the whole payload of the clip.

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


def clamp_to_frame(mob, name=""):
    left, right, top, bot = (
        mob.get_left()[0], mob.get_right()[0],
        mob.get_top()[1],  mob.get_bottom()[1],
    )
    assert left  >= X_MIN - 0.05, f"{name or mob} clips left  ({left:.2f})"
    assert right <= X_MAX + 0.05, f"{name or mob} clips right ({right:.2f})"
    assert top   <= Y_MAX + 0.05, f"{name or mob} clips top   ({top:.2f})"
    assert bot   >= Y_MIN - 0.05, f"{name or mob} clips bot   ({bot:.2f})"


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
        # A on the left. B upper right, C lower right — so two wires spread
        # visibly from A.
        A_POS = np.array([-3.8, 1.4, 0])
        B_POS = np.array([ 2.6, 2.4, 0])
        C_POS = np.array([ 2.6, 0.4, 0])
        BANNER_Y = -3.2
        LOG_Y    = -1.0

        # ==================================================================
        # NEURONS + WIRES
        # ==================================================================
        neuron_a = Circle(radius=0.42, color=NEURON_C, stroke_width=3).set_fill(NEURON_C, opacity=0.15).move_to(A_POS)
        label_a  = Text("A", font_size=26, color=NEURON_C, weight="BOLD").move_to(A_POS)

        neuron_b = Circle(radius=0.42, color=NEURON_C, stroke_width=3).set_fill(NEURON_C, opacity=0.15).move_to(B_POS)
        label_b  = Text("B", font_size=26, color=NEURON_C, weight="BOLD").move_to(B_POS)

        neuron_c = Circle(radius=0.42, color=NEURON_C, stroke_width=3).set_fill(NEURON_C, opacity=0.15).move_to(C_POS)
        label_c  = Text("C", font_size=26, color=NEURON_C, weight="BOLD").move_to(C_POS)

        clamp_to_frame(neuron_a, "A"); clamp_to_frame(neuron_b, "B"); clamp_to_frame(neuron_c, "C")

        # Two wires. Both start thin/gray.
        wire_ab = Line(neuron_a.get_center(), neuron_b.get_center(), color=WIRE_C, stroke_width=2.5)
        wire_ac = Line(neuron_a.get_center(), neuron_c.get_center(), color=WIRE_C, stroke_width=2.5)

        # Small labels describing what each B/C responds to.
        b_hint = Text("responds to: exception handling", font_size=13, color=DIM)
        b_hint.next_to(neuron_b, RIGHT, buff=0.25)
        # If it clips, tuck under the neuron instead.
        if b_hint.get_right()[0] > X_MAX - 0.1:
            b_hint = Text("responds to: exception handling", font_size=12, color=DIM)
            b_hint.next_to(neuron_b, DOWN, buff=0.1)
        c_hint = Text("responds to: dense math", font_size=13, color=DIM)
        c_hint.next_to(neuron_c, RIGHT, buff=0.25)
        if c_hint.get_right()[0] > X_MAX - 0.1:
            c_hint = Text("responds to: dense math", font_size=12, color=DIM)
            c_hint.next_to(neuron_c, DOWN, buff=0.1)

        clamp_to_frame(b_hint, "b_hint"); clamp_to_frame(c_hint, "c_hint")

        heading = Text("three neurons in Mr. Spiky", font_size=20, color=DIM)
        heading.set(width=min(heading.width, 6.0))
        heading.move_to([0, 3.4, 0])
        clamp_to_frame(heading, "heading")

        # ==================================================================
        # EXPOSURE LOG (bottom half, above the banner)
        # ==================================================================
        log_bg = RoundedRectangle(
            width=10.6, height=0.85,
            corner_radius=0.12,
            color=WIRE_C, stroke_width=1.0,
        ).set_fill("#141416", opacity=0.7).move_to([0, LOG_Y, 0])
        clamp_to_frame(log_bg, "log_bg")

        log_caption_left = Text("exposure", font_size=14, color=DIM)
        log_caption_left.move_to([log_bg.get_left()[0] + 0.9, LOG_Y + 0.14, 0])
        log_counter = Text("0", font_size=26, color=SPIKE, weight="BOLD")
        log_counter.move_to([log_bg.get_left()[0] + 0.9, LOG_Y - 0.14, 0])

        log_arrow_x = log_bg.get_left()[0] + 1.85
        log_arrow = Text("→", font_size=22, color=DIM).move_to([log_arrow_x, LOG_Y, 0])

        log_snippet = Text("(waiting for input...)", font_size=15, color=DIM)
        log_snippet.move_to([log_arrow_x + 3.6, LOG_Y, 0])
        if log_snippet.width > 6.4:
            log_snippet.set(width=6.4)
        clamp_to_frame(log_snippet, "log_snippet")

        # ==================================================================
        # BOTTOM BANNER
        # ==================================================================
        banner_bg = RoundedRectangle(
            width=12.6, height=0.9,
            corner_radius=0.15,
            color=WIRE_C, stroke_width=1.0,
        ).set_fill("#141416", opacity=0.9).move_to([0, BANNER_Y, 0])
        clamp_to_frame(banner_bg, "banner_bg")

        banner_text = Text("", font_size=22, color=WHITE)
        banner_text.move_to(banner_bg.get_center())

        # ==================================================================
        # FADE IN THE COMPOSITION
        # ==================================================================
        self.play(
            FadeIn(heading),
            FadeIn(neuron_a), FadeIn(label_a),
            FadeIn(neuron_b), FadeIn(label_b),
            FadeIn(neuron_c), FadeIn(label_c),
            Create(wire_ab), Create(wire_ac),
            FadeIn(b_hint), FadeIn(c_hint),
            FadeIn(log_bg), FadeIn(log_caption_left), FadeIn(log_counter),
            FadeIn(log_arrow), FadeIn(log_snippet),
            FadeIn(banner_bg),
            run_time=1.4,
        )

        # ==================================================================
        # Helpers
        # ==================================================================
        def set_banner(new_str: str, color=WHITE, run_time: float = 0.5):
            nonlocal banner_text
            new_text = Text(new_str, font_size=22, color=color)
            new_text.set(width=min(new_text.width, MAX_BANNER_W))
            new_text.move_to(banner_bg.get_center())
            self.play(Transform(banner_text, new_text), run_time=run_time)

        def set_snippet(new_str: str):
            nonlocal log_snippet
            new = Text(new_str, font_size=15, color=WHITE)
            new.move_to([log_arrow_x + 3.6, LOG_Y, 0])
            if new.width > 6.4:
                new.set(width=6.4)
            self.play(Transform(log_snippet, new), run_time=0.2)

        def set_counter(n: int):
            nonlocal log_counter
            new = Text(str(n), font_size=26, color=SPIKE, weight="BOLD")
            new.move_to([log_bg.get_left()[0] + 0.9, LOG_Y - 0.14, 0])
            self.play(Transform(log_counter, new), run_time=0.15)

        # Track wire strengths so we can display final "specialization" state.
        ab_strength = 0.0  # 0..1
        ac_strength = 0.0

        def redraw_wires(rt=0.25):
            new_ab = Line(
                neuron_a.get_center(), neuron_b.get_center(),
                color=interpolate_hex(WIRE_C, WIRE_HOT, ab_strength),
                stroke_width=2.5 + ab_strength * 12,
            )
            new_ac = Line(
                neuron_a.get_center(), neuron_c.get_center(),
                color=interpolate_hex(WIRE_C, WIRE_HOT, ac_strength),
                stroke_width=2.5 + ac_strength * 12,
            )
            self.play(Transform(wire_ab, new_ab), Transform(wire_ac, new_ac), run_time=rt)

        def fire_pair(from_neuron, to_neuron, rt_each=0.22):
            """A fires first, then the other. Small pause between."""
            self.play(
                from_neuron.animate.set_stroke(SPIKE, width=6).set_fill(SPIKE, opacity=0.55),
                Flash(from_neuron, color=SPIKE, flash_radius=0.6, num_lines=10, line_length=0.15),
                run_time=rt_each,
            )
            self.play(
                from_neuron.animate.set_stroke(NEURON_C, width=3).set_fill(NEURON_C, opacity=0.15),
                to_neuron.animate.set_stroke(SPIKE, width=6).set_fill(SPIKE, opacity=0.55),
                Flash(to_neuron, color=SPIKE, flash_radius=0.6, num_lines=10, line_length=0.15),
                run_time=rt_each + 0.05,
            )
            self.play(
                to_neuron.animate.set_stroke(NEURON_C, width=3).set_fill(NEURON_C, opacity=0.15),
                run_time=0.15,
            )

        # ==================================================================
        # BEAT 1 — Setup: describe the three neurons (0..5s)
        # ==================================================================
        set_banner("A = 'a function starts'.   B & C wait for specific patterns.", DIM, run_time=0.6)
        self.wait(2.4)

        # ==================================================================
        # BEAT 2 — Different code, different wire strengthens (5..22s)
        # Each exposure shows a snippet; A fires (function starts); then
        # either B or C fires depending on the snippet's pattern.
        # ==================================================================
        set_banner("watch which wire strengthens with which pattern.", WHITE, run_time=0.5)
        self.wait(1.0)

        # 6 exposures. 4 hit B (exceptions), 2 hit C (dense math).
        exposures = [
            ("def parse(cfg):  # django (has try/except)",     "B", 0.20),
            ("def dot(a, b): return sum(x*y for x,y in ...)",  "C", 0.18),
            ("async def fetch(url):  # httpx (try/except)",    "B", 0.20),
            ("def compile(src):  # cpython (try/except)",      "B", 0.20),
            ("def _norm(v): return v / (v @ v)**0.5",          "C", 0.16),
            ("class Session:  # sqlalchemy (raise)",            "B", 0.20),
        ]

        for i, (snippet, target, delta) in enumerate(exposures):
            set_snippet(snippet)
            self.wait(0.2)
            if target == "B":
                fire_pair(neuron_a, neuron_b)
                ab_strength = min(1.0, ab_strength + delta)
            else:
                fire_pair(neuron_a, neuron_c)
                ac_strength = min(1.0, ac_strength + delta)
            redraw_wires(rt=0.25)
            set_counter(i + 1)
            self.wait(0.35)

        self.wait(1.0)

        # ==================================================================
        # BEAT 3 — The reveal: two wires now visibly different (22..27s)
        # ==================================================================
        set_banner("A→B is now strong. A→C stayed weak. Same rule; different pattern.",
                   SPIKE, run_time=0.7)
        self.wait(3.0)

        # ==================================================================
        # BEAT 4 — Punchline (27..30s)
        # ==================================================================
        # Fade the log so the banner + neurons dominate the final frame.
        self.play(
            FadeOut(log_bg), FadeOut(log_caption_left), FadeOut(log_counter),
            FadeOut(log_arrow), FadeOut(log_snippet),
            run_time=0.5,
        )

        # Combine the Hebb quote + Mr. Spiky punchline as two banner beats.
        set_banner('"neurons that fire together, wire together." — Hebb, 1949',
                   WHITE, run_time=0.55)
        self.wait(2.0)
        set_banner("this is how your visual cortex learned to read.", WHITE, run_time=0.5)
        self.wait(1.6)
        set_banner("and it's how Mr. Spiky learned senior Python.", SPIKE, run_time=0.55)

        # Small honest-methodology tag, inside the banner card.
        tag = Text("no teacher. no labels. no gradient descent.", font_size=13, color=DIM)
        tag.move_to(banner_bg.get_center() + np.array([0, -0.28, 0]))
        raised_banner = Text("and it's how Mr. Spiky learned senior Python.", font_size=22, color=SPIKE)
        raised_banner.set(width=min(raised_banner.width, MAX_BANNER_W))
        raised_banner.move_to(banner_bg.get_center() + np.array([0, 0.14, 0]))
        clamp_to_frame(tag, "tag")
        self.play(Transform(banner_text, raised_banner), FadeIn(tag), run_time=0.4)

        self.wait(2.2)

        self.play(
            FadeOut(heading),
            FadeOut(neuron_a), FadeOut(label_a),
            FadeOut(neuron_b), FadeOut(label_b),
            FadeOut(neuron_c), FadeOut(label_c),
            FadeOut(wire_ab), FadeOut(wire_ac),
            FadeOut(b_hint), FadeOut(c_hint),
            FadeOut(banner_bg), FadeOut(banner_text), FadeOut(tag),
            run_time=0.7,
        )


if __name__ == "__main__":
    print("Render with: manim -qm stdp_learning_rule.py STDPLearningRule")
