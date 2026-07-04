"""
Mr. Spiky pitch clip 2: STDP — "neurons that fire together wire together" (~15s).

Beat to land: Spike-Timing-Dependent Plasticity is the biological learning rule
Mr. Spiky uses. If A fires BEFORE B, the A->B connection strengthens
(potentiation, dw > 0). If A fires AFTER B, the A->B connection weakens
(depression, dw < 0). Repeat this rule across thousands of senior-authored
functions and the network's weights become a structural fingerprint of good
code — no gradient descent, no labels.

Preserve these visual invariants when editing:
  1. The arrow thickness = the weight. It MUST visibly thicken in beat 1 and
     thin in beat 2. If a reviewer can't tell the difference, the point is
     lost. Do not swap this out for a numeric label alone.
  2. In beat 1 the ORDER is A-then-B; in beat 2 it is B-then-A. Getting this
     backwards inverts the sign of the rule — do not "clean up" by making
     both beats symmetric in the wrong direction.
  3. Implementation constants (A- = 0.020, A+ = 0.005, multiplicative
     depression) are deliberately NOT on screen. The pitch is about the
     concept, not the hyperparameters.

Render:
    manim -pqm stdp_learning_rule.py STDPLearningRule
"""

from manim import (
    Scene,
    Circle,
    Arrow,
    Text,
    MathTex,
    Line,
    DashedLine,
    Dot,
    VGroup,
    Write,
    FadeIn,
    FadeOut,
    Flash,
    Create,
    Transform,
    ReplacementTransform,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    ORIGIN,
    WHITE,
    GRAY,
    YELLOW,
    ORANGE,
    GREEN,
    RED,
    BLUE_B,
    BLACK,
    config,
)


config.background_color = "#0b0d12"


class STDPLearningRule(Scene):
    def construct(self) -> None:
        # ------------------------------------------------------------------
        # Faint background grid of neurons to hint at scale (used in beat 3).
        # Kept static; we don't animate all of them.
        # ------------------------------------------------------------------
        bg_neurons = VGroup()
        for row in range(-2, 3):
            for col in range(-5, 6):
                if row == 0 and col in (-1, 1):
                    # Leave gaps where the focal neurons will sit.
                    continue
                d = Dot(radius=0.09, color=BLUE_B).set_opacity(0.12)
                d.move_to(RIGHT * col * 1.1 + UP * row * 1.1)
                bg_neurons.add(d)
        # Don't add yet; we fade it in during beat 3.

        # ------------------------------------------------------------------
        # Focal neurons A and B with a weighted arrow between them.
        # ------------------------------------------------------------------
        neuron_a = Circle(radius=0.55, color=BLUE_B, fill_opacity=0.3)
        neuron_a.move_to(LEFT * 2.2)
        label_a = Text("A", font_size=32, color=WHITE).move_to(neuron_a.get_center())

        neuron_b = Circle(radius=0.55, color=BLUE_B, fill_opacity=0.3)
        neuron_b.move_to(RIGHT * 2.2)
        label_b = Text("B", font_size=32, color=WHITE).move_to(neuron_b.get_center())

        def make_arrow(thickness: float) -> Arrow:
            return Arrow(
                start=neuron_a.get_right(),
                end=neuron_b.get_left(),
                buff=0.05,
                color=WHITE,
                stroke_width=thickness,
                max_tip_length_to_length_ratio=0.08,
            )

        arrow = make_arrow(6.0)
        w_label = MathTex("w", font_size=32, color=WHITE)
        w_label.next_to(arrow, UP, buff=0.15)

        title = Text("STDP: spike-timing-dependent plasticity", font_size=26, color=GRAY)
        title.to_edge(UP, buff=0.4)

        self.play(FadeIn(title), run_time=0.4)
        self.play(
            Create(neuron_a),
            Create(neuron_b),
            FadeIn(label_a),
            FadeIn(label_b),
            run_time=0.6,
        )
        self.play(Create(arrow), FadeIn(w_label), run_time=0.4)

        # ------------------------------------------------------------------
        # Small timeline underneath for the delta-t marker.
        # ------------------------------------------------------------------
        timeline = Line(LEFT * 3.5, RIGHT * 3.5, color=GRAY, stroke_width=2)
        timeline.to_edge(DOWN, buff=1.8)
        t_label = Text("time", font_size=18, color=GRAY)
        t_label.next_to(timeline, RIGHT, buff=0.1)

        self.play(Create(timeline), FadeIn(t_label), run_time=0.3)

        # ==================================================================
        # BEAT 1 — Potentiation: A fires, then B fires. Arrow thickens.
        # ==================================================================
        beat1_caption = Text(
            "A fires, then B fires — connection strengthens.",
            font_size=24,
            color=WHITE,
        )
        beat1_caption.next_to(timeline, DOWN, buff=0.35)

        self.play(FadeIn(beat1_caption), run_time=0.3)

        # Spike A.
        tick_a1 = Line(UP * 0.15, DOWN * 0.15, color=YELLOW, stroke_width=3)
        tick_a1.move_to(timeline.point_from_proportion(0.35))
        tick_a1_label = Text("A", font_size=18, color=YELLOW).next_to(tick_a1, UP, buff=0.05)

        neuron_a.set_fill(YELLOW, opacity=0.9)
        self.play(
            Flash(neuron_a, color=YELLOW, flash_radius=0.85, num_lines=14),
            Create(tick_a1),
            FadeIn(tick_a1_label),
            run_time=0.4,
        )
        self.play(neuron_a.animate.set_fill(BLUE_B, opacity=0.3), run_time=0.2)

        self.wait(0.25)

        # Spike B (shortly after).
        tick_b1 = Line(UP * 0.15, DOWN * 0.15, color=YELLOW, stroke_width=3)
        tick_b1.move_to(timeline.point_from_proportion(0.55))
        tick_b1_label = Text("B", font_size=18, color=YELLOW).next_to(tick_b1, UP, buff=0.05)

        neuron_b.set_fill(YELLOW, opacity=0.9)
        self.play(
            Flash(neuron_b, color=YELLOW, flash_radius=0.85, num_lines=14),
            Create(tick_b1),
            FadeIn(tick_b1_label),
            run_time=0.4,
        )
        self.play(neuron_b.animate.set_fill(BLUE_B, opacity=0.3), run_time=0.2)

        # Delta-t bracket.
        dt_line = DashedLine(
            tick_a1.get_bottom() + DOWN * 0.15,
            tick_b1.get_bottom() + DOWN * 0.15,
            color=GRAY,
            stroke_width=2,
            dash_length=0.08,
        )
        dt_label = MathTex(r"\Delta t", font_size=26, color=GRAY)
        dt_label.next_to(dt_line, DOWN, buff=0.05)

        self.play(Create(dt_line), FadeIn(dt_label), run_time=0.3)

        # Thicken arrow: w increases.
        arrow_thick = make_arrow(12.0).set_color(GREEN)
        dw_pos = MathTex(r"\Delta w > 0", font_size=28, color=GREEN)
        dw_pos.next_to(arrow_thick, UP, buff=0.35)

        self.play(
            ReplacementTransform(arrow, arrow_thick),
            w_label.animate.set_color(GREEN),
            FadeIn(dw_pos, shift=UP * 0.1),
            run_time=0.7,
        )
        arrow = arrow_thick  # rebind for beat 2
        self.wait(0.6)

        # Clear beat 1 ticks/caption.
        self.play(
            FadeOut(tick_a1),
            FadeOut(tick_a1_label),
            FadeOut(tick_b1),
            FadeOut(tick_b1_label),
            FadeOut(dt_line),
            FadeOut(dt_label),
            FadeOut(dw_pos),
            FadeOut(beat1_caption),
            run_time=0.4,
        )

        # ==================================================================
        # BEAT 2 — Depression: B fires first, then A. Arrow thins.
        # ==================================================================
        beat2_caption = Text(
            "B fires, then A — connection weakens.",
            font_size=24,
            color=WHITE,
        )
        beat2_caption.next_to(timeline, DOWN, buff=0.35)

        self.play(FadeIn(beat2_caption), run_time=0.3)

        # Spike B first this time.
        tick_b2 = Line(UP * 0.15, DOWN * 0.15, color=YELLOW, stroke_width=3)
        tick_b2.move_to(timeline.point_from_proportion(0.35))
        tick_b2_label = Text("B", font_size=18, color=YELLOW).next_to(tick_b2, UP, buff=0.05)

        neuron_b.set_fill(YELLOW, opacity=0.9)
        self.play(
            Flash(neuron_b, color=YELLOW, flash_radius=0.85, num_lines=14),
            Create(tick_b2),
            FadeIn(tick_b2_label),
            run_time=0.4,
        )
        self.play(neuron_b.animate.set_fill(BLUE_B, opacity=0.3), run_time=0.2)

        self.wait(0.25)

        # Then A fires.
        tick_a2 = Line(UP * 0.15, DOWN * 0.15, color=YELLOW, stroke_width=3)
        tick_a2.move_to(timeline.point_from_proportion(0.55))
        tick_a2_label = Text("A", font_size=18, color=YELLOW).next_to(tick_a2, UP, buff=0.05)

        neuron_a.set_fill(YELLOW, opacity=0.9)
        self.play(
            Flash(neuron_a, color=YELLOW, flash_radius=0.85, num_lines=14),
            Create(tick_a2),
            FadeIn(tick_a2_label),
            run_time=0.4,
        )
        self.play(neuron_a.animate.set_fill(BLUE_B, opacity=0.3), run_time=0.2)

        # Thin arrow: w decreases.
        arrow_thin = make_arrow(3.0).set_color(RED)
        dw_neg = MathTex(r"\Delta w < 0", font_size=28, color=RED)
        dw_neg.next_to(arrow_thin, UP, buff=0.35)

        self.play(
            ReplacementTransform(arrow, arrow_thin),
            w_label.animate.set_color(RED),
            FadeIn(dw_neg, shift=UP * 0.1),
            run_time=0.7,
        )
        arrow = arrow_thin
        self.wait(0.6)

        # Clear beat 2 elements before the closing beat.
        self.play(
            FadeOut(tick_a2),
            FadeOut(tick_a2_label),
            FadeOut(tick_b2),
            FadeOut(tick_b2_label),
            FadeOut(dw_neg),
            FadeOut(beat2_caption),
            FadeOut(timeline),
            FadeOut(t_label),
            run_time=0.4,
        )

        # ==================================================================
        # BEAT 3 — Zoom out (visually): faint grid of many neurons + pitch.
        # ==================================================================
        # Bring in the background grid behind everything.
        self.add(bg_neurons)
        bg_neurons.set_opacity(0.0)
        self.play(
            bg_neurons.animate.set_opacity(0.18),
            # Shrink the focal pair a touch to imply "zoom out."
            VGroup(neuron_a, neuron_b, label_a, label_b, arrow, w_label)
            .animate.scale(0.75),
            run_time=0.8,
        )

        pitch = Text(
            "Repeat this on 2,680 senior-authored functions.\n"
            "No gradient descent. No labels.\n"
            "The network's weights become the shape of senior code.",
            font_size=26,
            color=WHITE,
            line_spacing=0.9,
        )
        pitch.to_edge(DOWN, buff=0.5)

        self.play(FadeIn(pitch, shift=UP * 0.2), run_time=0.8)
        self.wait(2.4)
        self.play(FadeOut(pitch), run_time=0.5)


if __name__ == "__main__":
    print("Render with: manim -pqm stdp_learning_rule.py STDPLearningRule")
