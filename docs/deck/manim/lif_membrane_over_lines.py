"""
Mr. Spiky pitch clip 1: "A neuron reads code the way you do." (~25s)

Wow-first. Three beats:

1. (0-8s) Side by side. Reviewer's eye scanning code + gut-reaction meter on
   the left; LIF neuron + membrane trace on the right. Both rise together.
   Both spike on the same gnarly line. No text yet — the visual is the
   claim.

2. (8-16s) The two panels *overlay* into one. Reveal: the neuron and the
   reviewer are doing the same thing. Punchline text: "same accumulation.
   same threshold. same reaction."

3. (16-25s) Zoom out. The neuron's sensitivity came from being exposed to
   2680 senior functions during training — direct bridge to clip 2.
   Punchline: "and its intuition came from the same place yours did —
   reading a lot of code."

Design invariants to preserve if editing:
  * Reviewer-eye trace and membrane trace MUST rise in visible sync. If
    they drift, the "same thing" claim reads as a coincidence, not a
    parallel.
  * Text is small and enters LATE in each beat. The presenter's voice
    carries the narration; text is the anchor, not the script.
  * Only the last line (compound `if`) crosses threshold. Earlier bumps
    are sub-threshold so the "context accumulates" story reads visually.
  * The leak between line inputs must stay visible — it's the "L" in LIF
    and drops the metaphor if removed.

Render:
    manim -qm lif_membrane_over_lines.py LIFMembraneOverLines
"""

from manim import (
    Scene,
    Code,
    Circle,
    Dot,
    Text,
    DashedLine,
    Axes,
    VGroup,
    VMobject,
    Rectangle,
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
    RED,
    BLUE,
    GREEN,
    linear,
    smooth,
)
import numpy as np


# ---------------------------------------------------------------------------
# Palette — match the deck's spike-yellow accent so the clip and the reveal.js
# slides feel like one piece.
# ---------------------------------------------------------------------------
BG        = "#0f0f0f"
SPIKE     = "#ffd54a"
SPIKE_HOT = "#ff9a3a"
LINE_MID  = "#3a3a3f"
DIM       = "#8a8a92"
NEURON_C  = "#ffd54a"
EYE_C     = "#7fd6ff"


CODE_LINES = [
    "def process(items):",
    "    result = []",
    "    for item in items:",
    "        if item.valid and item.score > threshold:",
]

# Per-line "structural intensity" — matches how the SNN would react.
# The last line (compound `if`) is the only one that pushes membrane over
# threshold. Earlier lines contribute meaningful sub-threshold accumulation.
BUMPS   = [0.22, 0.18, 0.38, 0.62]
LEAK    = 0.78   # membrane fraction retained per timestep between lines
THRESH  = 1.00


class LIFMembraneOverLines(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ==================================================================
        # BEAT 1 — Side-by-side (0-8s)
        # Left: reviewer eye + gut-reaction meter.
        # Right: LIF neuron + membrane trace.
        # ==================================================================
        left_label = Text(
            "a reviewer reading code",
            font_size=22, color=DIM,
        ).to_edge(LEFT, buff=0.6).to_edge(UP, buff=0.4)
        right_label = Text(
            "a spiking neuron reading the same code",
            font_size=22, color=DIM,
        ).to_edge(RIGHT, buff=0.6).to_edge(UP, buff=0.4)

        self.play(FadeIn(left_label), FadeIn(right_label), run_time=0.5)

        # --- LEFT: code + gut meter ------------------------------------
        code_block = Code(
            code_string="\n".join(CODE_LINES),
            language="python",
            background="window",
            tab_width=4,
            formatter_style="monokai",
        ).scale(0.5)
        code_block.next_to(left_label, DOWN, buff=0.35).align_to(left_label, LEFT)

        # The eye — a small blue dot that will step down through the lines.
        eye = Dot(radius=0.10, color=EYE_C)
        eye_glow = Circle(radius=0.16, color=EYE_C, fill_opacity=0.25, stroke_width=0)
        eye_group = VGroup(eye_glow, eye)
        eye_group.move_to(code_block.get_left() + LEFT * 0.2 + UP * 0.6)

        # Gut-reaction meter — a vertical bar next to the code that fills up.
        gut_frame = Rectangle(height=1.5, width=0.22, color=LINE_MID, stroke_width=1.5)
        gut_frame.next_to(code_block, DOWN, buff=0.3).align_to(code_block, LEFT)
        gut_fill = Rectangle(height=0.001, width=0.20, color=EYE_C, fill_opacity=0.85, stroke_width=0)
        gut_fill.next_to(gut_frame.get_bottom(), UP, buff=0.005).align_to(gut_frame, LEFT).shift(RIGHT * 0.01)
        gut_label = Text("gut", font_size=14, color=DIM).next_to(gut_frame, RIGHT, buff=0.15)
        gut_thresh = DashedLine(
            gut_frame.get_left() + UP * 0.55,
            gut_frame.get_right() + UP * 0.55,
            color=DIM, stroke_width=1.2, dash_length=0.05,
        )

        self.play(
            Create(code_block),
            Create(gut_frame),
            FadeIn(gut_label),
            Create(gut_thresh),
            FadeIn(eye_group),
            run_time=1.0,
        )

        # --- RIGHT: neuron + membrane trace ----------------------------
        neuron = Circle(radius=0.28, color=NEURON_C, stroke_width=3).set_fill(NEURON_C, opacity=0.15)
        neuron.next_to(right_label, DOWN, buff=0.35).shift(RIGHT * 1.5)
        neuron_label = Text("V", font_size=18, color=NEURON_C, slant="ITALIC")
        neuron_label.move_to(neuron.get_center())

        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 1.3, 0.5],
            x_length=3.6,
            y_length=1.6,
            tips=False,
            axis_config={"color": LINE_MID, "stroke_width": 1.5, "include_ticks": False},
        )
        axes.next_to(neuron, DOWN, buff=0.35).align_to(gut_frame, DOWN)
        thresh_line = DashedLine(
            axes.c2p(0, THRESH),
            axes.c2p(4, THRESH),
            color=DIM, stroke_width=1.2, dash_length=0.06,
        )
        thresh_label = Text("threshold", font_size=13, color=DIM)
        thresh_label.next_to(axes.c2p(4, THRESH), RIGHT, buff=0.05)

        membrane_trace = VMobject(color=SPIKE, stroke_width=2.8)
        membrane_trace.set_points_as_corners([axes.c2p(0, 0), axes.c2p(0.001, 0)])

        self.play(
            Create(neuron),
            FadeIn(neuron_label),
            Create(axes),
            Create(thresh_line),
            FadeIn(thresh_label),
            Create(membrane_trace),
            run_time=1.0,
        )

        # --- Sync animation: eye reads line by line, meter rises, membrane rises.
        v = 0.0
        trace_points = [axes.c2p(0, 0)]
        code_line_mobs = list(code_block.code_lines)

        for idx, (line_mob, bump) in enumerate(zip(code_line_mobs, BUMPS)):
            # Eye moves to this code line.
            target_eye_y = line_mob.get_center()[1]
            new_eye_pos = np.array([
                code_block.get_left()[0] - 0.2,
                target_eye_y,
                0.0,
            ])
            # Between-line leak (small backward slope on the trace).
            if idx > 0:
                v_pre = v * LEAK
                leak_x = idx - 0.5
                trace_points.append(axes.c2p(leak_x, v_pre))
                gut_pre_h = min(1.5, v_pre * 1.5)
                new_gut_pre = Rectangle(
                    height=max(gut_pre_h, 0.001), width=0.20,
                    color=EYE_C, fill_opacity=0.85, stroke_width=0,
                ).align_to(gut_frame, DOWN).align_to(gut_frame, LEFT).shift(RIGHT * 0.01)
                partial_trace = VMobject(color=SPIKE, stroke_width=2.8)
                partial_trace.set_points_as_corners(trace_points)
                self.play(
                    eye_group.animate.move_to(new_eye_pos),
                    Transform(gut_fill, new_gut_pre),
                    Transform(membrane_trace, partial_trace),
                    run_time=0.5,
                    rate_func=smooth,
                )
                v = v_pre

            # The line highlights as the eye lands on it.
            self.play(
                line_mob.animate.set_fill(color=WHITE, opacity=1),
                run_time=0.1,
            )

            # Bump: membrane rises, gut meter rises.
            v_post = v + bump
            trace_points.append(axes.c2p(idx + 1, v_post))
            gut_h = min(1.5, v_post * 1.5)
            new_gut = Rectangle(
                height=max(gut_h, 0.001), width=0.20,
                color=EYE_C, fill_opacity=0.85, stroke_width=0,
            ).align_to(gut_frame, DOWN).align_to(gut_frame, LEFT).shift(RIGHT * 0.01)
            new_trace = VMobject(color=SPIKE, stroke_width=2.8)
            new_trace.set_points_as_corners(trace_points)

            # If this line crosses threshold: flash + color shift.
            if v_post >= THRESH and v < THRESH:
                # Recolor the trace's crossing segment orange.
                new_trace.set_color(SPIKE_HOT)
                self.play(
                    Transform(gut_fill, new_gut),
                    Transform(membrane_trace, new_trace),
                    neuron.animate.set_stroke(SPIKE_HOT, width=5).set_fill(SPIKE_HOT, opacity=0.5),
                    Flash(neuron, color=SPIKE_HOT, flash_radius=0.5, num_lines=14),
                    run_time=0.6,
                )
                # The reviewer flinches — quick red pulse on the eye too.
                self.play(
                    Flash(eye_group, color=RED, flash_radius=0.35, num_lines=10),
                    run_time=0.3,
                )
            else:
                self.play(
                    Transform(gut_fill, new_gut),
                    Transform(membrane_trace, new_trace),
                    run_time=0.5,
                )
            v = v_post

        # Beat 1 tail: hold for a beat so audience registers the visual sync.
        self.wait(0.6)

        # ==================================================================
        # BEAT 2 — Overlay reveal (8-16s)
        # The two panels are doing the same thing. Punchline text.
        # ==================================================================
        # Dim non-essential elements to focus attention.
        beat2_dims = VGroup(
            left_label, right_label, code_block, gut_frame, gut_thresh,
            gut_label, thresh_label,
        )
        self.play(
            beat2_dims.animate.set_opacity(0.15),
            run_time=0.6,
        )

        # Punchline lines — small text, three fragments, timed to arrive
        # like the presenter's beats.
        punchline_lines = VGroup(
            Text("same accumulation.", font_size=28, color=WHITE),
            Text("same threshold.", font_size=28, color=WHITE),
            Text("same reaction.", font_size=28, color=SPIKE),
        ).arrange(DOWN, buff=0.25).move_to(ORIGIN).shift(DOWN * 0.4)

        for i, line in enumerate(punchline_lines):
            self.play(FadeIn(line, shift=UP * 0.15), run_time=0.55)
            self.wait(0.35 if i < 2 else 0.7)

        # ==================================================================
        # BEAT 3 — Where the intuition came from (16-25s)
        # Bridge to clip 2. Show that this neuron's sensitivity came from
        # exposure to 2680 senior-authored functions.
        # ==================================================================
        self.play(
            FadeOut(punchline_lines),
            FadeOut(beat2_dims),
            FadeOut(eye_group),
            FadeOut(gut_fill),
            FadeOut(axes),
            FadeOut(thresh_line),
            FadeOut(membrane_trace),
            neuron.animate.move_to(ORIGIN + UP * 0.8).set_stroke(NEURON_C, width=3).set_fill(NEURON_C, opacity=0.15),
            neuron_label.animate.move_to(ORIGIN + UP * 0.8),
            run_time=0.9,
        )

        # A cluster of small "past functions" appear around the neuron, each
        # flashes in briefly then fades — the training corpus visualized.
        rng = np.random.default_rng(42)
        corpus_dots = VGroup()
        for _ in range(28):
            r = 2.4 + rng.random() * 0.9
            theta = rng.random() * 2 * np.pi
            pos = np.array([r * np.cos(theta), r * np.sin(theta) * 0.55 + 0.4, 0.0])
            d = Dot(radius=0.05, color=SPIKE).set_opacity(0.0).move_to(pos)
            corpus_dots.add(d)

        # Fade the corpus in, in a wave, each dot briefly pulses toward the
        # neuron to convey "the neuron saw this."
        for k, d in enumerate(corpus_dots):
            self.play(d.animate.set_opacity(0.9), run_time=0.04)
        self.wait(0.2)

        exposure_text = Text(
            "2,680 senior-authored functions",
            font_size=22, color=DIM,
        ).move_to(ORIGIN + DOWN * 0.6)
        self.play(FadeIn(exposure_text), run_time=0.5)

        # Final punchline.
        final_line = Text(
            "its intuition came from the same place yours did —",
            font_size=22, color=WHITE,
        ).next_to(exposure_text, DOWN, buff=0.4)
        final_line2 = Text(
            "reading a lot of code.",
            font_size=24, color=SPIKE,
        ).next_to(final_line, DOWN, buff=0.15)

        self.play(FadeIn(final_line), run_time=0.5)
        self.play(FadeIn(final_line2), run_time=0.5)
        self.wait(1.6)

        # Clean fade to preserve the last frame in the audience's head.
        self.play(
            FadeOut(final_line),
            FadeOut(final_line2),
            FadeOut(exposure_text),
            FadeOut(corpus_dots),
            FadeOut(neuron),
            FadeOut(neuron_label),
            run_time=0.8,
        )


if __name__ == "__main__":
    print("Render with: manim -qm lif_membrane_over_lines.py LIFMembraneOverLines")
