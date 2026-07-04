"""
Mr. Spiky pitch clip 1: LIF membrane carries context across code lines (~20s).

Beat to land: a Leaky Integrate-and-Fire neuron accumulates evidence line by
line as Python code streams in. Each incoming line bumps the membrane; between
lines the membrane leaks slightly toward rest (the "L" in LIF). When a
structurally intense line (compound `if`) arrives, the membrane crosses
threshold and the neuron spikes — that line is what Mr. Spiky flags.

Preserve these visual invariants when editing:
  1. The membrane MUST visibly leak between line inputs (small downward slope).
     Removing the leak collapses the metaphor into a plain accumulator.
  2. Only the LAST line should push the membrane above threshold. The earlier
     bumps must be sub-threshold so the "context accumulates" story reads.
  3. The threshold line stays dashed gray; the trace goes yellow -> orange at
     crossing. Do not recolor these without a reason.

Render:
    manim -pqm lif_membrane_over_lines.py LIFMembraneOverLines
"""

from manim import (
    Scene,
    Code,
    Circle,
    Text,
    DashedLine,
    Axes,
    VGroup,
    VMobject,
    Write,
    FadeIn,
    FadeOut,
    Flash,
    Create,
    Indicate,
    UP,
    DOWN,
    LEFT,
    RIGHT,
    ORIGIN,
    WHITE,
    GRAY,
    YELLOW,
    ORANGE,
    BLUE_B,
    BLACK,
    config,
)


# Standard 16:9 canvas; keep defaults so `manim -pqm` works without flags.
config.background_color = "#0b0d12"


class LIFMembraneOverLines(Scene):
    def construct(self) -> None:
        # ------------------------------------------------------------------
        # Layout: code on the left half, neuron + trace on the right half.
        # ------------------------------------------------------------------
        code_lines = [
            "def process(items):",
            "    result = []",
            "    for item in items:",
            "        if item.valid and item.score > threshold:",
        ]

        # Per-line membrane bump amplitudes (in axis units). The last bump is
        # tuned to just cross threshold=1.0 after the previous accumulated
        # charge and leak. Earlier bumps must stay sub-threshold.
        bumps = [0.20, 0.18, 0.35, 0.55]

        # Leak amount applied between line inputs (fraction of current V).
        leak_factor = 0.75

        threshold_v = 1.0
        rest_v = 0.0

        # ---- Code panel (left) --------------------------------------------
        # Manim 0.19+ renamed `code=` to `code_string=`. Keeping the older
        # name here would 400 on newer installs.
        code_block = Code(
            code_string="\n".join(code_lines),
            language="python",
            background="window",
            tab_width=4,
            formatter_style="monokai",
        ).scale(0.6)
        code_block.to_edge(LEFT, buff=0.6).shift(UP * 0.2)

        # Hide code lines initially; we reveal them one at a time.
        # In Manim 0.19+ the highlighted lines live on `.code_lines`
        # (was `.code` in 0.18 and earlier).
        code_line_mobs = list(code_block.code_lines)
        for line in code_line_mobs:
            line.set_opacity(0.15)

        self.play(FadeIn(code_block, shift=UP * 0.2), run_time=0.6)

        # ---- Neuron (top right) -------------------------------------------
        neuron = Circle(radius=0.55, color=BLUE_B, fill_opacity=0.25)
        neuron.move_to(RIGHT * 4.2 + UP * 2.4)
        neuron_label = Text("LIF neuron", font_size=22, color=GRAY)
        neuron_label.next_to(neuron, UP, buff=0.15)

        self.play(Create(neuron), FadeIn(neuron_label), run_time=0.5)

        # ---- Membrane trace axes (bottom right) ---------------------------
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[-0.2, 1.4, 0.5],
            x_length=5.2,
            y_length=2.6,
            tips=False,
            axis_config={"color": GRAY, "stroke_width": 2},
        )
        axes.next_to(neuron, DOWN, buff=0.6).shift(LEFT * 0.1)

        y_label = Text("V(t)", font_size=20, color=GRAY)
        y_label.next_to(axes.y_axis, UP, buff=0.1)
        x_label = Text("line #", font_size=20, color=GRAY)
        x_label.next_to(axes.x_axis, RIGHT, buff=0.1)

        # Threshold marker.
        thr_start = axes.c2p(0, threshold_v)
        thr_end = axes.c2p(5, threshold_v)
        threshold_line = DashedLine(
            thr_start, thr_end, color=GRAY, stroke_width=2, dash_length=0.12
        )
        threshold_label = Text("threshold", font_size=18, color=GRAY)
        threshold_label.next_to(threshold_line, RIGHT, buff=0.1).shift(DOWN * 0.05)

        self.play(
            Create(axes),
            FadeIn(y_label),
            FadeIn(x_label),
            Create(threshold_line),
            FadeIn(threshold_label),
            run_time=0.8,
        )

        # ------------------------------------------------------------------
        # Line-by-line: reveal code line, bump membrane, then leak.
        # ------------------------------------------------------------------
        trace = VMobject(color=YELLOW, stroke_width=4)
        trace.set_points_as_corners([axes.c2p(0, rest_v), axes.c2p(0, rest_v)])
        self.add(trace)

        current_v = rest_v
        current_t = 0.0
        trace_points = [axes.c2p(current_t, current_v)]

        def redraw_trace() -> None:
            trace.set_points_as_corners(trace_points)

        for idx, (line_mob, bump) in enumerate(zip(code_line_mobs, bumps)):
            is_last = idx == len(code_lines) - 1

            # 1) Reveal the code line.
            self.play(line_mob.animate.set_opacity(1.0), run_time=0.35)
            self.play(Indicate(line_mob, scale_factor=1.05, color=YELLOW), run_time=0.35)

            # 2) Bump the membrane (integrate).
            target_v = current_v + bump
            steps = 8
            for s in range(1, steps + 1):
                v = current_v + bump * (s / steps)
                t = current_t + 0.4 * (s / steps)
                trace_points.append(axes.c2p(t, v))
            current_v = target_v
            current_t += 0.4
            redraw_trace()

            # Color shifts to orange once we're above threshold.
            if current_v >= threshold_v:
                trace.set_color(ORANGE)

            self.wait(0.15)

            if is_last and current_v >= threshold_v:
                # Threshold crossing: flash the neuron, drop membrane to rest.
                neuron.set_fill(ORANGE, opacity=0.9)
                self.play(
                    Flash(neuron, color=ORANGE, flash_radius=0.9, num_lines=16),
                    run_time=0.5,
                )
                spike_label = Text(
                    "spike — this line lit up", font_size=22, color=ORANGE
                )
                spike_label.next_to(neuron, RIGHT, buff=0.3)
                self.play(FadeIn(spike_label, shift=RIGHT * 0.2), run_time=0.4)

                # Reset membrane back to rest.
                reset_steps = 10
                for s in range(1, reset_steps + 1):
                    v = current_v + (rest_v - current_v) * (s / reset_steps)
                    t = current_t + 0.5 * (s / reset_steps)
                    trace_points.append(axes.c2p(t, v))
                current_v = rest_v
                current_t += 0.5
                redraw_trace()
                self.play(
                    neuron.animate.set_fill(BLUE_B, opacity=0.25), run_time=0.4
                )
                self.wait(0.4)
            else:
                # 3) Leak between line inputs (the "L" in LIF).
                leaked_v = max(rest_v, current_v * leak_factor)
                leak_steps = 6
                for s in range(1, leak_steps + 1):
                    v = current_v + (leaked_v - current_v) * (s / leak_steps)
                    t = current_t + 0.35 * (s / leak_steps)
                    trace_points.append(axes.c2p(t, v))
                current_v = leaked_v
                current_t += 0.35
                redraw_trace()
                self.wait(0.1)

        # ------------------------------------------------------------------
        # Final beat: pitch line overlay.
        # ------------------------------------------------------------------
        pitch = Text(
            "The network reads your code the way you do —\n"
            "line by line, carrying context.",
            font_size=28,
            color=WHITE,
            line_spacing=0.9,
        )
        pitch.to_edge(DOWN, buff=0.4)

        # Dim the working area a touch so the pitch line reads.
        self.play(FadeIn(pitch, shift=UP * 0.2), run_time=0.8)
        self.wait(2.2)
        self.play(FadeOut(pitch), run_time=0.5)


if __name__ == "__main__":
    # Convenience: allow `python lif_membrane_over_lines.py` to hint at usage.
    print(
        "Render with: manim -pqm lif_membrane_over_lines.py LIFMembraneOverLines"
    )
