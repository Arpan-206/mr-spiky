# Mr. Spiky pitch deck

Reveal.js deck for the 5-minute hackathon pitch. Zero-install — the whole
deck is one HTML file loading reveal.js from a CDN.

## Present

```bash
just present               # opens the deck in your browser
```

That serves `docs/deck/` on `:5555` and opens `index.html`. Press `s` in
the deck to open speaker notes in a second window. `Esc` for an
overview grid. `f` for fullscreen.

Alternatively, just double-click `index.html` — reveal.js works from the
`file://` protocol, but the Manim clips embedded as `<video>` might not
autoplay depending on your browser. Running via `just present` is more
reliable.

## Structure (10 slides, ~5 min)

| # | Slide | Time |
|---|---|---|
| 1 | Hook: real testbed PR comment | 30s |
| 2 | The prompt reframed | 20s |
| 3 | Three biological mechanisms | 45s |
| 4 | Manim clip: LIF membrane over lines | 30s |
| 5 | Manim clip: STDP learning rule | 20s |
| 6 | Six axes (the human-readable output) | 30s |
| 7 | Live demos — 5 testbed PRs | 45s |
| 8 | Honest validation table | 30s |
| 9 | Four product surfaces | 30s |
| 10 | Close + URLs | 30s |

Speaker notes on every slide. Read them.

## Assets you need to provide

The `assets/` directory has placeholders. On first run everything falls
back gracefully (stylized templates in place of missing images), but for
a real pitch these should be filled in:

- `assets/lif_membrane_over_lines.mp4` — render Manim clip 1
- `assets/stdp_learning_rule.mp4` — render Manim clip 2
- `assets/testbed-pr4-comment.png` — screenshot of the actual bot comment on
  [testbed PR #4](https://github.com/Arpan-206/mr-spiky-testbed/pull/4)

### Render the Manim clips

Requires Manim Community. One-time install:

```bash
pip install manim   # NOT via uv — Manim needs system-python + LaTeX
```

Render:

```bash
just deck-clips     # renders both clips into assets/*.mp4
```

Or manually:

```bash
cd docs/deck/manim
manim -pqm lif_membrane_over_lines.py LIFMembraneOverLines
manim -pqm stdp_learning_rule.py STDPLearningRule
# Copy the .mp4 outputs from ./media/videos/*/720p30/ into ../assets/
```

The `pqm` flags mean: `p` preview, `q` quality, `m` medium (720p, fast to
iterate). Bump to `qh` (high, 1080p) once you're happy with the timing.

### Screenshot the testbed PR

Open [PR #4](https://github.com/Arpan-206/mr-spiky-testbed/pull/4),
scroll to any of the inline bot comments (there are 4), and screenshot a
single comment card. Save as `assets/testbed-pr4-comment.png`.

## Editing

- Slide content: `index.html` — each slide is one `<section>`.
- Styles: `style.css` — colors live in `:root` at the top.
- Speaker notes: `<aside class="notes">` inside each `<section>`.

The theme extends reveal.js's `black` theme. Colors:

- Spike yellow (`#ffd54a`) — the accent color for headers, code, links.
- Carbon (`#0f0f0f`) — the slide background.
- Slate (`#1a1a1c`) — card / table backgrounds.

## Print to PDF

Append `?print-pdf` to the deck URL and use Chrome's Print → Save as PDF.
Details: [reveal.js PDF export](https://revealjs.com/pdf-export/).
