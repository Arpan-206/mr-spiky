"""
Mr. Spiky pitch clip 2: "This is how your brain learns. It's how Mr. Spiky
learned too." (~22s)

Three beats:

1. (0-7s) Two neurons. They co-fire on senior code, over and over. Each
   co-fire slightly thickens the wire between them. Do this 8 times.
   The audience watches wiring emerge without labels, without a teacher.

2. (7-14s) Zoom out. A grid of similar neuron pairs, all self-organizing
   the same way, in parallel. Text arrives: "Neurons that fire together
   wire together." — Donald Hebb, 1949. (The moment the analogy becomes
   concrete instead of metaphorical.)

3. (14-22s) Punchline: "this is how your visual cortex learned to read.
   and it's how Mr. Spiky learned senior Python." Optional small tag: "no
   teacher. no labels. no gradient descent."

Design invariants:
  * The wire's thickness change MUST be visible each iteration. If it's
    too subtle the "wiring emerges" story doesn't read.
  * The Hebb quote arrives ONLY in beat 2 after the audience has already
    seen the mechanism. Landing the words on the mechanism is the wow.
  * Beat 3's punchline is the whole payoff — hold the final frame longer
    (2s) so it sticks.

Render:
    manim -qm stdp_learning_rule.py STDPLearningRule
"""

from manim import (
    Scene,
    Circle,
    Arrow,
    Text,
    Line,
    DashedLine,
    Dot,
    VGroup,
    Rectangle,
    Write,
    FadeIn,
    FadeOut,
    Flash,
    Create,
    Transform,
    Indicate,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    ORIGIN,
    WHITE,
    GRAY,
    YELLOW,
    GREEN,
    RED,
)
import numpy as np


BG        = "#0f0f0f"
SPIKE     = "#ffd54a"
DIM       = "#8a8a92"
NEURON_C  = "#ffd54a"
WIRE_C    = "#5a5a62"
WIRE_HOT  = "#ffd54a"


class STDPLearningRule(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ==================================================================
        # BEAT 1 — Two neurons, repeated co-firing thickens the wire (0-7s)
        # ==================================================================

        # Two neurons, A on the left, B on the right.
        neuron_a = Circle(radius=0.36, color=NEURON_C, stroke_width=3).set_fill(NEURON_C, opacity=0.15)
        neuron_a.move_to(LEFT * 2.2)
        label_a = Text("A", font_size=22, color=NEURON_C, weight="BOLD").move_to(neuron_a.get_center())

        neuron_b = Circle(radius=0.36, color=NEURON_C, stroke_width=3).set_fill(NEURON_C, opacity=0.15)
        neuron_b.move_to(RIGHT * 2.2)
        label_b = Text("B", font_size=22, color=NEURON_C, weight="BOLD").move_to(neuron_b.get_center())

        # The connecting wire. Starts thin, gets thicker.
        wire = Line(
            neuron_a.get_right(), neuron_b.get_left(),
            color=WIRE_C, stroke_width=2,
        )

        # An "exposure counter" — small, unobtrusive, so audience sees
        # the number of senior-code exposures ticking up.
        counter_label = Text("exposures:", font_size=16, color=DIM).to_edge(DOWN, buff=0.6).shift(LEFT * 1.2)
        counter_val = Text("0", font_size=20, color=SPIKE).next_to(counter_label, RIGHT, buff=0.2)

        self.play(
            FadeIn(neuron_a), FadeIn(label_a),
            FadeIn(neuron_b), FadeIn(label_b),
            Create(wire),
            FadeIn(counter_label), FadeIn(counter_val),
            run_time=0.9,
        )

        # 8 repeated exposures: A fires, then B fires shortly after,
        # wire thickens each time. Small delta per iteration is visible
        # cumulatively.
        NUM_EXPOSURES = 8
        for i in range(NUM_EXPOSURES):
            # A fires first.
            self.play(
                neuron_a.animate.set_stroke(SPIKE, width=6).set_fill(SPIKE, opacity=0.55),
                Flash(neuron_a, color=SPIKE, flash_radius=0.55, num_lines=10, line_length=0.15),
                run_time=0.18,
            )
            # A relaxes as B fires.
            self.play(
                neuron_a.animate.set_stroke(NEURON_C, width=3).set_fill(NEURON_C, opacity=0.15),
                neuron_b.animate.set_stroke(SPIKE, width=6).set_fill(SPIKE, opacity=0.55),
                Flash(neuron_b, color=SPIKE, flash_radius=0.55, num_lines=10, line_length=0.15),
                run_time=0.22,
            )
            # Wire thickens. Also color-shifts slightly toward hot on strong exposures.
            new_width = 2 + (i + 1) * 1.5   # 3.5, 5, 6.5, ... 14
            fraction = (i + 1) / NUM_EXPOSURES
            wire_color = interpolate_hex(WIRE_C, WIRE_HOT, fraction)
            new_wire = Line(
                neuron_a.get_right(), neuron_b.get_left(),
                color=wire_color, stroke_width=new_width,
            )
            new_counter = Text(str(i + 1), font_size=20, color=SPIKE).next_to(counter_label, RIGHT, buff=0.2)
            self.play(
                neuron_b.animate.set_stroke(NEURON_C, width=3).set_fill(NEURON_C, opacity=0.15),
                Transform(wire, new_wire),
                Transform(counter_val, new_counter),
                run_time=0.18,
            )

        # Small pause so audience registers the emerged connection.
        self.wait(0.5)

        # ==================================================================
        # BEAT 2 — Zoom out to a grid of pairs + the Hebb quote (7-14s)
        # ==================================================================

        # Move the focal pair up so we have room for the grid + quote.
        focal_group = VGroup(neuron_a, label_a, neuron_b, label_b, wire)
        self.play(
            focal_group.animate.scale(0.55).shift(UP * 1.8 + LEFT * 3.0),
            FadeOut(counter_label), FadeOut(counter_val),
            run_time=0.7,
        )

        # Create a grid of smaller pair-plus-wire units around the focal
        # pair. Each pair fires on its own random rhythm, with wires
        # thickening a bit before fade-in completes — the point is "this
        # is happening in parallel, everywhere, on all pairs."
        rng = np.random.default_rng(7)
        pair_groups = []
        for row in range(3):
            for col in range(5):
                if row == 0 and col == 0:
                    continue   # skip focal position
                cx = -3.6 + col * 1.6
                cy = 1.0 - row * 1.4
                # Two dots, wire.
                a = Circle(radius=0.16, color=NEURON_C, stroke_width=2).set_fill(NEURON_C, opacity=0.15)
                a.move_to(np.array([cx - 0.4, cy, 0]))
                b = Circle(radius=0.16, color=NEURON_C, stroke_width=2).set_fill(NEURON_C, opacity=0.15)
                b.move_to(np.array([cx + 0.4, cy, 0]))
                # Wire thickness varies — some pairs have wired up more than others.
                thickness = 1.0 + rng.random() * 4.5
                colr = interpolate_hex(WIRE_C, WIRE_HOT, thickness / 5.5)
                w = Line(a.get_right(), b.get_left(), color=colr, stroke_width=thickness)
                pair_groups.append(VGroup(a, b, w))

        # Fade the grid in as a wave.
        for pg in pair_groups:
            self.play(FadeIn(pg), run_time=0.04)
        self.wait(0.3)

        # Trigger a few random pair-fires so the grid looks alive.
        for _ in range(6):
            pg = pair_groups[int(rng.integers(0, len(pair_groups)))]
            a_circle, b_circle, _ = pg
            self.play(
                a_circle.animate.set_stroke(SPIKE, width=3).set_fill(SPIKE, opacity=0.6),
                b_circle.animate.set_stroke(SPIKE, width=3).set_fill(SPIKE, opacity=0.6),
                run_time=0.1,
            )
            self.play(
                a_circle.animate.set_stroke(NEURON_C, width=2).set_fill(NEURON_C, opacity=0.15),
                b_circle.animate.set_stroke(NEURON_C, width=2).set_fill(NEURON_C, opacity=0.15),
                run_time=0.1,
            )

        # The Hebb quote. Small, quiet, arrives after mechanism is on screen.
        quote_line1 = Text(
            '"Neurons that fire together,',
            font_size=26, color=WHITE, slant="ITALIC",
        )
        quote_line2 = Text(
            'wire together."',
            font_size=26, color=SPIKE, slant="ITALIC",
        )
        quote_attrib = Text(
            "— Donald Hebb, 1949",
            font_size=16, color=DIM,
        )
        quote_group = VGroup(quote_line1, quote_line2, quote_attrib).arrange(DOWN, buff=0.15)
        quote_group.to_edge(DOWN, buff=0.7)

        self.play(FadeIn(quote_line1), run_time=0.4)
        self.play(FadeIn(quote_line2), run_time=0.4)
        self.play(FadeIn(quote_attrib), run_time=0.3)
        self.wait(1.2)

        # ==================================================================
        # BEAT 3 — Punchline (14-22s)
        # ==================================================================
        self.play(
            FadeOut(quote_group),
            *[FadeOut(pg) for pg in pair_groups],
            run_time=0.6,
        )
        # Move focal pair back to center for the final frame.
        self.play(
            focal_group.animate.scale(1.6).move_to(ORIGIN + UP * 0.2),
            run_time=0.6,
        )

        line1 = Text(
            "this is how your visual cortex",
            font_size=24, color=WHITE,
        )
        line2 = Text(
            "learned to read.",
            font_size=24, color=WHITE,
        )
        line3 = Text(
            "and it's how Mr. Spiky",
            font_size=24, color=WHITE,
        )
        line4 = Text(
            "learned senior Python.",
            font_size=26, color=SPIKE,
        )
        block = VGroup(line1, line2, line3, line4).arrange(DOWN, buff=0.15)
        block.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(line1), FadeIn(line2), run_time=0.5)
        self.wait(0.4)
        self.play(FadeIn(line3), FadeIn(line4), run_time=0.5)
        self.wait(0.6)

        # Micro-tag underneath, small.
        micro = Text(
            "no teacher. no labels. no gradient descent.",
            font_size=15, color=DIM,
        ).next_to(block, DOWN, buff=0.25)
        self.play(FadeIn(micro), run_time=0.4)

        self.wait(2.0)

        # Clean fade.
        self.play(
            FadeOut(block),
            FadeOut(micro),
            FadeOut(focal_group),
            run_time=0.8,
        )


# ---------------------------------------------------------------------------
# Helper — interpolate between two hex colors. Kept inline so this file has
# zero non-manim deps.
# ---------------------------------------------------------------------------
def interpolate_hex(c1: str, c2: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    def hex_to_rgb(h):
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r1, g1, b1 = hex_to_rgb(c1)
    r2, g2, b2 = hex_to_rgb(c2)
    r = round(r1 + (r2 - r1) * t)
    g = round(g1 + (g2 - g1) * t)
    b = round(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


if __name__ == "__main__":
    print("Render with: manim -qm stdp_learning_rule.py STDPLearningRule")
