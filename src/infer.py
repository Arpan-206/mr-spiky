"""Inference entry point.

`analyze(code)` returns the fixed response schema regardless of whether the SNN
weights exist yet. If `models/snn_weights.pt` is missing, we fall back to
scoring each line from its normalized AST features (mock mode) so the frontend
can integrate against a live API before training finishes.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch

from .encode import DEFAULT_STEPS, encode_batch, encode_sequence
from .features import (
    AXIS_NAMES,
    FEATURE_NAMES,
    NUM_FEATURES,
    compute_axes,
    extract_function_features,
    extract_line_features,
    line_lineage,
    normalize,
)
from .model import SpikyNet, ecdf_rescale, per_timestep_attribution, temporal_spike_attribution

log = logging.getLogger("mrspiky.infer")

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_WEIGHTS_PATH = _MODELS_DIR / "snn_weights.pt"
_BASELINES_PATH = _MODELS_DIR / "snn_baselines.pt"
_ECDF_PATH = _MODELS_DIR / "snn_ecdf.pt"
_THRESHOLD_PATH = _MODELS_DIR / "threshold.json"

# Weights used to collapse normalized features into a single suspicion score
# in mock mode. Hand-picked, ordered to match FEATURE_NAMES in features.py.
# parse_error dominates — code that doesn't even parse should overpower
# every other signal. The three cross-function features are given moderate
# weight in mock: they're novel signal but often zero (only fire on lines
# with actual cross-scope reach), so they can't lead the score unless a
# real reach pattern is present.
_MOCK_WEIGHTS: tuple[float, ...] = (
    0.15,  # nesting_depth
    0.03,  # length
    0.05,  # token_entropy
    0.05,  # naming_entropy
    0.12,  # cyclomatic_proxy
    0.08,  # use_def_distance
    0.08,  # name_flow
    0.10,  # call_graph_shape
    0.05,  # exception_density
    0.50,  # parse_error (dominant)
    0.10,  # global_reach (new: cross-function reach)
    0.08,  # attr_reach (new: cross-method reach)
    0.06,  # call_graph_depth (new: same-file call chain depth)
)
assert len(_MOCK_WEIGHTS) == NUM_FEATURES, "mock weights out of sync with features"
_MOCK_THRESHOLD = 0.55

# Populated lazily on first call if weights exist.
_snn_state: dict[str, Any] | None = None


def _mock_score(vector: list[float]) -> float:
    norm = normalize(vector)
    s = sum(w * v for w, v in zip(_MOCK_WEIGHTS, norm))
    return max(0.0, min(1.0, s))


def describe_mode() -> dict[str, Any]:
    """Report the current scoring mode without running inference.

    - `mode`: "snn" when trained weights are loaded, "mock" when we're using
      the fallback linear-scoring path. The frontend can gate demo copy or
      confidence badges on this.
    - `threshold`: current flag cutoff (varies by mode).
    - When in SNN mode we also expose diagnostics from the loaded artifacts.
    """
    snn = _load_snn()
    if snn is None:
        return {
            "mode": "mock",
            "threshold": _MOCK_THRESHOLD,
            "reason": "models/snn_weights.pt not found; scoring from a linear combination of raw features",
        }
    hidden = snn.get("hidden_baseline")
    output = snn.get("output_baseline")
    ecdf = snn.get("ecdf_ref")
    n_distinct_hidden = (
        len({round(float(v), 4) for v in hidden.tolist()}) if hidden is not None else None
    )
    return {
        "mode": "snn",
        "threshold": snn["threshold"],
        "hidden_size": int(hidden.numel()) if hidden is not None else None,
        "output_size": int(output.numel()) if output is not None else None,
        "hidden_baselines_distinct": n_distinct_hidden,
        "ecdf_reference_size": int(ecdf.numel()) if ecdf is not None else None,
    }


def _load_snn() -> dict[str, Any] | None:
    """Lazy-load SNN weights + threshold. Returns None if not yet trained."""
    global _snn_state
    if _snn_state is not None:
        return _snn_state
    if not _WEIGHTS_PATH.exists():
        return None
    try:
        net = SpikyNet()
        net.load_state_dict(torch.load(_WEIGHTS_PATH, weights_only=True))
        net.eval()
        threshold = _MOCK_THRESHOLD
        if _THRESHOLD_PATH.exists():
            threshold = float(json.loads(_THRESHOLD_PATH.read_text())["threshold"])
        hidden_baseline = None
        output_baseline = None
        if _BASELINES_PATH.exists():
            b = torch.load(_BASELINES_PATH, weights_only=True)
            hidden_baseline = b["hidden"]
            output_baseline = b["output"]
            log.info(
                "loaded baselines: hidden mean=%.3f  output mean=%.3f",
                float(hidden_baseline.mean()), float(output_baseline.mean()),
            )
        ecdf_ref = None
        if _ECDF_PATH.exists():
            e = torch.load(_ECDF_PATH, weights_only=True)
            ecdf_ref = e["sorted_raw_scores"]
            log.info("loaded ECDF reference (n=%d)", int(ecdf_ref.numel()))
        _snn_state = {
            "net": net,
            "threshold": threshold,
            "hidden_baseline": hidden_baseline,
            "output_baseline": output_baseline,
            "ecdf_ref": ecdf_ref,
        }
        log.info("loaded SNN weights + threshold=%.4f", threshold)
        return _snn_state
    except Exception as e:  # noqa: BLE001 — hackathon: don't crash the API
        log.warning("failed to load SNN artifacts (%s), staying in mock mode", e)
        return None


def _snn_scores_isolated(net: SpikyNet, vectors: list[list[float]]) -> list[float]:
    """Line-independent rate-coded scoring. Each line rate-coded into 25
    steps, no membrane state shared between lines."""
    if not vectors:
        return []
    spikes = encode_batch(vectors, num_steps=DEFAULT_STEPS)
    with torch.no_grad():
        out = net(spikes)
    intensities = temporal_spike_attribution(out.hidden_spikes)
    return [max(0.0, min(1.0, float(v))) for v in intensities.tolist()]


def _snn_scores_sequence(
    net: SpikyNet,
    vectors: list[list[float]],
    hidden_baseline: torch.Tensor | None = None,
    output_baseline: torch.Tensor | None = None,
) -> list[float]:
    """Sequence-mode scoring with baseline-relative attribution.

    Feed line vectors as a temporal stream so each line's LIF membrane state
    carries forward. Per-line score = how much the SNN's firing *exceeds*
    its usual per-neuron firing rate on senior code. High score = 'this line
    lights up neurons that are usually quiet' — the SNN's version of a
    reviewer's gut reaction."""
    if not vectors:
        return []
    seq = encode_sequence(vectors)  # (T=lines, B=1, F)
    with torch.no_grad():
        out = net(seq)
    per_line = per_timestep_attribution(
        out.hidden_spikes, out.output_spikes,
        hidden_baseline=hidden_baseline,
        output_baseline=output_baseline,
        hidden_mem=out.hidden_mem,
        output_mem=out.output_mem,
    ).squeeze(1)
    return [max(0.0, min(1.0, float(v))) for v in per_line.tolist()]


# Toggle for the pitch: set to False to see per-line-isolated scoring.
_USE_SEQUENCE_MODE = True


def _snn_scores(
    net: SpikyNet,
    vectors: list[list[float]],
    hidden_baseline: torch.Tensor | None = None,
    output_baseline: torch.Tensor | None = None,
) -> list[float]:
    if _USE_SEQUENCE_MODE:
        return _snn_scores_sequence(net, vectors, hidden_baseline, output_baseline)
    return _snn_scores_isolated(net, vectors)


def _verdict(flagged_count: int, dominant_axis: str | None) -> str:
    if flagged_count == 0:
        return "no suspicious spikes detected"
    base = (
        "1 high-intensity spike detected"
        if flagged_count == 1
        else f"{flagged_count} high-intensity spikes detected"
    )
    if dominant_axis:
        return f"{base} — dominant axis: {dominant_axis}"
    return base


def _dominant_axis(axes_per_flagged_line: list[dict[str, float]]) -> str | None:
    """Across flagged lines, which axis has the highest mean value? That's what
    the SNN is 'objecting to' most across this snippet."""
    if not axes_per_flagged_line:
        return None
    means: dict[str, float] = {name: 0.0 for name in AXIS_NAMES}
    for a in axes_per_flagged_line:
        for name in AXIS_NAMES:
            means[name] += a.get(name, 0.0)
    return max(means, key=means.get)


# Templated phrasing per axis — plain-English fragments that read like a
# reviewer's gut reaction. Kept short so the concatenated reason fits in a UI
# tooltip.
_AXIS_PHRASE: dict[str, str] = {
    "complexity":         "deeply nested / branchy control flow",
    "tangled_state":      "variables reach across long distances",
    "hidden_calls":       "delegates to opaque calls",
    "exception_surface":  "heavy exception handling",
    "naming":             "unusual identifier density",
    "malformed":          "doesn't parse as valid Python — likely a typo",
}


def _reason_from_axes(axes: dict[str, float], top_k: int = 2, min_value: float = 0.35) -> str:
    """Templated one-sentence reason from the top-K axes above `min_value`.

    Reads like: 'high on complexity (deeply nested / branchy control flow) and
    hidden_calls (delegates to opaque calls) — the shape a reviewer would ask
    you to break up.'"""
    if not axes:
        return "SNN fired but no axis crossed the reporting threshold"
    sorted_axes = sorted(axes.items(), key=lambda kv: -kv[1])
    drivers = [(name, val) for name, val in sorted_axes[:top_k] if val >= min_value]
    if not drivers:
        name, val = sorted_axes[0]
        return f"broadly elevated across axes; leader is {name} ({val:.2f})"
    if len(drivers) == 1:
        name, val = drivers[0]
        return f"high on {name} ({val:.2f}): {_AXIS_PHRASE.get(name, name)}"
    return "high on " + " + ".join(f"{n} ({v:.2f})" for n, v in drivers) + \
        " — " + "; ".join(_AXIS_PHRASE.get(n, n) for n, _ in drivers)


def _reason_from_axes_and_lineage(
    axes: dict[str, float],
    lineage: list[dict[str, object]],
    line_no: int,
) -> str:
    """Extend the axis-only reason with concrete AST context, e.g.:

        high on complexity (1.00) + hidden_calls (1.00) — deeply nested /
        branchy control flow; delegates to opaque calls. Structurally: this
        line sits inside `if x > 0` at L14, inside `for i in range(x)` at
        L12, inside function `parse_config` at L5.

    Two-part reason means a reviewer can skim just the first clause (axes)
    or read the second (specific nodes) for the concrete "why."
    """
    base = _reason_from_axes(axes)
    if not lineage:
        return base
    parts = []
    for entry in lineage:
        label = entry.get("label", "")
        start = entry.get("line")
        parts.append(f"{label} at L{start}")
    joined = ", inside ".join(parts)
    return f"{base}. Structurally: this line sits inside {joined}."


def function_scores(code: str) -> dict[str, float]:
    """Score every top-level function in `code` and return {name: score}.

    Same computation as the `context.function_score` field in analyze(),
    but returned for *every* function regardless of whether any of its
    lines flagged. Used by the PR-review bot to compute before/after
    deltas — a function that was clean on base won't have any flagged
    lines to hang a context onto, so we need this direct entry point.
    """
    snn = _load_snn()
    if snn is None:
        return {}
    functions = extract_function_features(code)
    all_lines = extract_line_features(code)
    out: dict[str, float] = {}
    for fn in functions:
        fn_lines = [lf for lf in all_lines if fn.lineno <= lf.line <= fn.end_lineno]
        if not fn_lines:
            continue
        raw = _snn_scores_sequence(
            snn["net"],
            [lf.vector for lf in fn_lines],
            hidden_baseline=snn["hidden_baseline"],
            output_baseline=snn["output_baseline"],
        )
        if snn["ecdf_ref"] is not None:
            raw = ecdf_rescale(raw, snn["ecdf_ref"])
        if raw:
            out[fn.name] = round(max(raw), 4)
    return out


def _function_contexts(
    code: str,
    snn: dict[str, Any] | None,
    threshold: float,
) -> dict[int, dict[str, Any]]:
    """Map each line -> its enclosing function's context object.

    Also computes the function-level SNN score by running the function's own
    lines as a self-contained sequence. This gives a 'zoom-out' — the SNN's
    reaction to the function as a whole, useful when a single flagged line
    might look OK in isolation but sits inside a gnarly function."""
    contexts: dict[int, dict[str, Any]] = {}
    functions = extract_function_features(code)
    if not functions:
        return contexts

    for fn in functions:
        fn_line_feats = [
            lf for lf in extract_line_features(code)
            if fn.lineno <= lf.line <= fn.end_lineno
        ]
        fn_score: float | None = None
        if snn is not None and fn_line_feats:
            raw = _snn_scores_sequence(
                snn["net"],
                [lf.vector for lf in fn_line_feats],
                hidden_baseline=snn["hidden_baseline"],
                output_baseline=snn["output_baseline"],
            )
            if snn["ecdf_ref"] is not None:
                raw = ecdf_rescale(raw, snn["ecdf_ref"])
            fn_score = round(max(raw), 4) if raw else None

        ctx = {
            "function": fn.name,
            "span": [fn.lineno, fn.end_lineno],
            "function_score": fn_score,
        }
        for ln in range(fn.lineno, fn.end_lineno + 1):
            contexts[ln] = ctx
    return contexts


def _apply_axis_weights(
    score: float,
    axes: dict[str, float],
    weights: dict[str, float] | None,
) -> float:
    """Reshape `score` using per-axis weights so teams can tune what the
    SNN treats as suspicious.

    The score comes from the SNN (or the mock linear model) and already
    represents "how surprising this line is." Axis weights let a team say
    "when the SNN sees exception_surface fire, take it more seriously"
    (weight > 1.0) or "we don't care about naming density" (weight < 1.0).

    Concretely: compute a boost factor as the weighted mean of axis
    weights, using the current axis firing strengths as importance. If
    no axes are firing (all zero) or no custom weights are provided, boost
    is 1.0 (no-op) and we return the raw score unchanged.

    The result is clamped to [0, 1] so weight = 3.0 doesn't accidentally
    push flat scores over the threshold — a weight is a *reshape*, not a
    force-flag override.
    """
    if not weights or not axes:
        return score
    # Normalize to a boost centered around 1.0. Missing axes default to
    # weight 1.0 (no change) so a team only needs to specify what they
    # care to adjust — not the full six-axis table.
    total_strength = 0.0
    weighted_strength = 0.0
    for axis, strength in axes.items():
        w = float(weights.get(axis, 1.0))
        total_strength += strength
        weighted_strength += strength * w
    if total_strength <= 1e-6:
        return score
    boost = weighted_strength / total_strength
    return max(0.0, min(1.0, score * boost))


def analyze(
    code: str,
    axis_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Return the fixed JSON schema for a given code string.

    `axis_weights` (optional): per-axis multiplicative weights (default
    1.0 each) that let a team tune what the model treats as suspicious.
    E.g. `{"exception_surface": 1.5, "naming": 0.6}` boosts exception-heavy
    lines and quiets naming-driven flags. Only the axes you name are
    changed — the rest stay at 1.0. See `_apply_axis_weights`.
    """
    line_feats = extract_line_features(code)

    snn = _load_snn()
    if snn is None:
        log.warning("mock mode: models/snn_weights.pt missing — scoring from features only")
        threshold = _MOCK_THRESHOLD
        scores = [(lf.line, _mock_score(lf.vector)) for lf in line_feats]
    else:
        threshold = snn["threshold"]
        raw = _snn_scores(
            snn["net"],
            [lf.vector for lf in line_feats],
            hidden_baseline=snn["hidden_baseline"],
            output_baseline=snn["output_baseline"],
        )
        # ECDF rescale so raw scores (which collapse to a few discrete values
        # from LIF spike quantization) become percentile ranks vs the senior
        # corpus. Now the threshold in [0,1] means "top-(1-thr)% of senior code."
        if snn["ecdf_ref"] is not None:
            raw = ecdf_rescale(raw, snn["ecdf_ref"])
        scores = list(zip((lf.line for lf in line_feats), raw))

    # parse_error override: the SNN cannot have learned to react to parse
    # errors (they're absent from senior training code by definition), so
    # its score won't cross the flag threshold on a broken line. But we
    # still want that line surfaced — it's the strongest possible signal.
    # Force the score to a value that flags in either mode.
    _parse_error_idx = FEATURE_NAMES.index("parse_error")
    parse_error_lines = {
        lf.line for lf in line_feats if lf.vector[_parse_error_idx] >= 1.0
    }
    if parse_error_lines:
        scores = [
            (ln, max(sc, 1.0)) if ln in parse_error_lines else (ln, sc)
            for ln, sc in scores
        ]

    # Per-line axes: computed from the same normalized feature vector the SNN
    # consumed, so the axes explain *what the SNN saw*, not a parallel channel.
    axes_by_line = {lf.line: compute_axes(normalize(lf.vector)) for lf in line_feats}
    # Raw features for each line — enables richer frontend visualization.
    raw_features_by_line = {
        lf.line: {name: round(v, 4) for name, v in zip(FEATURE_NAMES, normalize(lf.vector))}
        for lf in line_feats
    }

    # Per-team axis weight reshape. Applied after parse_error override so
    # a syntax-broken line still saturates (weight 0.0 on `malformed` won't
    # silence it — that would defeat the point of a hard override).
    if axis_weights:
        scores = [
            (ln, sc if ln in parse_error_lines else _apply_axis_weights(sc, axes_by_line.get(ln, {}), axis_weights))
            for ln, sc in scores
        ]
    # Enclosing-function contexts (only computed for flagged lines to keep the
    # extra SNN passes proportional to the interesting work).
    contexts_by_line = _function_contexts(code, snn, threshold)

    flagged = [ln for ln, sc in scores if sc >= threshold]
    flagged_set = set(flagged)

    lines_out = []
    for ln, sc in scores:
        entry: dict[str, Any] = {
            "line": ln,
            "score": round(sc, 4),
            "flag": ln in flagged_set,
            "axes": axes_by_line.get(ln, {}),
        }
        # Enrich flagged lines with reason + context + raw features. Non-flagged
        # lines keep the lean shape to avoid bloating long-file responses.
        if ln in flagged_set:
            lineage = line_lineage(code, ln, max_depth=3)
            entry["reason"] = _reason_from_axes_and_lineage(
                axes_by_line.get(ln, {}), lineage, ln,
            )
            ctx = contexts_by_line.get(ln)
            if ctx is not None:
                # Copy so we don't mutate the shared dict (multiple lines
                # in the same function share the same context object).
                ctx = dict(ctx)
                ctx["lineage"] = lineage
                entry["context"] = ctx
            entry["raw_features"] = raw_features_by_line.get(ln, {})
        lines_out.append(entry)

    top_flagged = [
        ln for ln, _ in sorted(
            ((ln, sc) for ln, sc in scores if sc >= threshold),
            key=lambda pair: pair[1],
            reverse=True,
        )[:10]
    ]
    dom = _dominant_axis([axes_by_line[ln] for ln in flagged if ln in axes_by_line])

    return {
        "verdict": _verdict(len(flagged), dom),
        "dominant_axis": dom,
        "lines": lines_out,
        "top_flagged": top_flagged,
    }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    sample = sys.stdin.read() if not sys.stdin.isatty() else "def f(x):\n    if x:\n        for i in range(x):\n            print(i)\n"
    print(json.dumps(analyze(sample), indent=2))