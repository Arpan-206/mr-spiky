"""Stage 2 — calibrate a *structural-anomaly* threshold from CodeSearchNet stats.

Mr. Spiky's AST features (nesting depth, cyclomatic complexity, entropy) capture
structural complexity, not semantic correctness. Empirically, this can't tell
buggy Python apart from fixed Python (PyResBugs → ~51% accuracy). What it *can*
do is flag code that's structurally unusual for typical Python. So we calibrate
the threshold as `mean + K*std` of TSA intensities over the pretraining corpus:
anything above that is "unusually complex relative to normal code."

Fallback: if a labeled set (`data/mlcq_labeled.json`, historically named for
MLCQ but now populated by PyResBugs) exists, we *also* record its 95th-percentile
of label==1 intensities for provenance, but the primary threshold is the
anomaly cutoff. This is honest — the tool doesn't claim to detect bugs, it
claims to detect structural outliers.

Output: models/threshold.json
    {"threshold": float, "n_samples": int, "method": str, "stats": {...}}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from .encode import DEFAULT_STEPS, encode_batch, encode_sequence
from .features import extract_function_features, extract_line_features
from .model import SpikyNet, ecdf_rescale, per_timestep_attribution, temporal_spike_attribution

log = logging.getLogger("mrspiky.calibrate")

ROOT = Path(__file__).resolve().parent.parent
_SENIOR_CORPUS = ROOT / "data" / "senior_corpus.json"
_CSN_CORPUS = ROOT / "data" / "codesearchnet_python.json"
# "Normal" for calibration = the same corpus used for STDP pretraining.
PRETRAIN_PATH = _SENIOR_CORPUS if _SENIOR_CORPUS.exists() else _CSN_CORPUS
LABELED_PATH = ROOT / "data" / "mlcq_labeled.json"  # PyResBugs-populated
CODECOMPLEX_PATH = ROOT / "data" / "codecomplex_labeled.json"
ANNOTATIONS_PATH = ROOT / "data" / "annotations_labeled.json"
WEIGHTS_PATH = ROOT / "models" / "snn_weights.pt"
BASELINES_PATH = ROOT / "models" / "snn_baselines.pt"
ECDF_PATH = ROOT / "models" / "snn_ecdf.pt"
THRESHOLD_PATH = ROOT / "models" / "threshold.json"

# Anomaly cutoff = mean + K*std of TSA over the "normal" corpus.
# K=1.0 ≈ 85th percentile: flags visibly complex code while staying well above
# the median. K=1.5+ was too strict and only fired on extreme outliers.
ANOMALY_K = 1.0

# Hand-picked contrast pairs for when neither MLCQ nor a user-provided set is
# available. `label` is 1 = suspicious, 0 = normal.
_FALLBACK_LABELED: list[dict] = [
    {"label": 0, "code": "def add(a, b):\n    return a + b\n"},
    {"label": 0, "code": "def square(x):\n    return x * x\n"},
    {"label": 0, "code": "def greet(name):\n    return f'hi {name}'\n"},
    {
        "label": 1,
        "code": (
            "def tangled(a, b, c, d):\n"
            "    if a:\n"
            "        if b:\n"
            "            for i in range(c):\n"
            "                for j in range(d):\n"
            "                    if i * j > a:\n"
            "                        if b - i > 0:\n"
            "                            print(i, j)\n"
            "    return a\n"
        ),
    },
    {
        "label": 1,
        "code": (
            "def gnarly(x):\n"
            "    r = 0\n"
            "    for i in range(x):\n"
            "        for j in range(i):\n"
            "            for k in range(j):\n"
            "                if i + j + k > 100:\n"
            "                    if i * j > k:\n"
            "                        r += 1\n"
            "    return r\n"
        ),
    },
]


def _load_labeled() -> list[dict]:
    if LABELED_PATH.exists():
        data = json.loads(LABELED_PATH.read_text())
        log.info("loaded %d labeled samples from %s", len(data), LABELED_PATH)
        return data
    log.warning("no labeled set at %s — using built-in fallback contrast pairs", LABELED_PATH)
    return _FALLBACK_LABELED


def _load_pretrain_sources() -> list[str]:
    if PRETRAIN_PATH.exists():
        data = json.loads(PRETRAIN_PATH.read_text())
        log.info("loaded %d pretraining source samples from %s", len(data), PRETRAIN_PATH)
        return data
    log.warning("no pretraining corpus at %s — calibration will be less meaningful", PRETRAIN_PATH)
    return []


def _sequence_line_intensities(
    net: SpikyNet,
    src: str,
    hidden_baseline: torch.Tensor | None = None,
    output_baseline: torch.Tensor | None = None,
) -> list[float]:
    """Run one source through the SNN in *sequence mode*, return one score
    per line. Uses the same code path as infer.py so calibration and inference
    live on the same scale."""
    line_feats = extract_line_features(src)
    if not line_feats:
        return []
    seq = encode_sequence([lf.vector for lf in line_feats])
    with torch.no_grad():
        out = net(seq)
    per_line = per_timestep_attribution(
        out.hidden_spikes, out.output_spikes,
        hidden_baseline=hidden_baseline,
        output_baseline=output_baseline,
        hidden_mem=out.hidden_mem,
        output_mem=out.output_mem,
    ).squeeze(1)
    return [float(v) for v in per_line.tolist()]


def _compute_baselines(net: SpikyNet, sources: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean per-neuron firing rate across the corpus. Returns (hidden, output)
    both shape (H,) / (O,), values in [0, 1]."""
    h_sums: torch.Tensor | None = None
    o_sums: torch.Tensor | None = None
    total_steps = 0
    for src in sources:
        line_feats = extract_line_features(src)
        if not line_feats:
            continue
        seq = encode_sequence([lf.vector for lf in line_feats])
        with torch.no_grad():
            out = net(seq)
        # out.hidden_spikes: (T, 1, H). Sum over T; accumulate per-neuron.
        h = out.hidden_spikes.squeeze(1).sum(dim=0)  # (H,)
        o = out.output_spikes.squeeze(1).sum(dim=0)  # (O,)
        h_sums = h if h_sums is None else h_sums + h
        o_sums = o if o_sums is None else o_sums + o
        total_steps += out.hidden_spikes.shape[0]
    if total_steps == 0 or h_sums is None or o_sums is None:
        raise RuntimeError("no baseline data — check corpus")
    return h_sums / total_steps, o_sums / total_steps


def _intensity(
    net: SpikyNet,
    code: str,
    hidden_baseline: torch.Tensor | None = None,
    output_baseline: torch.Tensor | None = None,
) -> float:
    """Max per-line sequence-mode intensity for a snippet."""
    scores = _sequence_line_intensities(net, code, hidden_baseline, output_baseline)
    return max(scores) if scores else 0.0


def _batch_intensities(
    net: SpikyNet,
    sources: list[str],
    hidden_baseline: torch.Tensor | None = None,
    output_baseline: torch.Tensor | None = None,
) -> list[float]:
    out: list[float] = []
    for src in sources:
        out.extend(_sequence_line_intensities(net, src, hidden_baseline, output_baseline))
    return out


def _pick_threshold(scored: list[tuple[float, int]]) -> float:
    """Sweep candidate thresholds between observed scores and pick the one that
    maximizes accuracy. Ties broken by preferring the midpoint (more robust)."""
    if not scored:
        return 0.5
    sorted_scores = sorted({s for s, _ in scored})
    candidates = [
        (sorted_scores[i] + sorted_scores[i + 1]) / 2
        for i in range(len(sorted_scores) - 1)
    ] or [sorted_scores[0]]

    best_thr = candidates[0]
    best_acc = -1.0
    for thr in candidates:
        correct = sum(1 for s, y in scored if (s >= thr) == bool(y))
        acc = correct / len(scored)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    log.info("best threshold %.4f  accuracy %.2f on %d samples", best_thr, best_acc, len(scored))
    return best_thr


def _stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"n": 0, "mean": 0.0, "std": 0.0, "p50": 0.0, "p95": 0.0}
    t = torch.tensor(xs)
    return {
        "n": len(xs),
        "mean": float(t.mean()),
        "std": float(t.std(unbiased=False)),
        "p50": float(t.median()),
        "p95": float(torch.quantile(t, 0.95)),
    }


def calibrate() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not WEIGHTS_PATH.exists():
        raise SystemExit(f"no trained weights at {WEIGHTS_PATH} — run `just train` first")

    net = SpikyNet()
    net.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True))
    net.eval()

    # ---- Compute per-neuron baselines from the "normal" corpus ----
    # This is what turns the SNN into an intuition model: instead of scoring
    # raw firing (which saturates on typical senior-approved code because
    # senior code is genuinely complex), we score how much each line's firing
    # *exceeds* the neuron's usual firing rate on the training corpus. High
    # score means "this line lights up neurons that are usually quiet on
    # senior code" — the SNN's version of a reviewer's "wait, that's odd."
    pretrain_sources = _load_pretrain_sources()
    log.info("computing per-neuron baselines over %d sources…", len(pretrain_sources))
    hidden_baseline, output_baseline = _compute_baselines(net, pretrain_sources)
    log.info(
        "baselines: hidden mean=%.3f  min=%.3f max=%.3f  |  output mean=%.3f min=%.3f max=%.3f",
        float(hidden_baseline.mean()), float(hidden_baseline.min()), float(hidden_baseline.max()),
        float(output_baseline.mean()), float(output_baseline.min()), float(output_baseline.max()),
    )
    BASELINES_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"hidden": hidden_baseline, "output": output_baseline}, BASELINES_PATH)
    log.info("saved baselines -> %s", BASELINES_PATH)

    # ---- Primary path: anomaly threshold, now scored against baselines ----
    normal_intensities = _batch_intensities(net, pretrain_sources, hidden_baseline, output_baseline)
    result: dict = {}

    # ---- ECDF calibration ----
    # The raw SNN scores collapse to a few discrete values (e.g. {0.0, 0.4,
    # 0.5, 0.9, 1.0}) because sequence-mode LIF spikes are quantized. That
    # made the threshold saturate. Fix: save the *sorted* corpus scores and
    # at inference remap each raw score to its percentile in this
    # distribution — so a line at corpus median becomes 0.5, top 1% becomes
    # 0.99, etc. This spreads the score distribution across [0,1] and turns
    # the threshold from "did you saturate" into "top-X% relative to senior
    # code," which is what the pitch actually claims.
    if normal_intensities:
        sorted_scores = sorted(normal_intensities)
        ecdf_tensor = torch.tensor(sorted_scores, dtype=torch.float32)
        torch.save({"sorted_raw_scores": ecdf_tensor}, ECDF_PATH)
        log.info("saved ECDF (%d samples) -> %s", len(sorted_scores), ECDF_PATH)

    if normal_intensities:
        s = _stats(normal_intensities)
        t = torch.tensor(normal_intensities)
        # After ECDF rescale, the primary threshold in ECDF-space is just
        # the percentile cutoff. p90 = "top 10% most unusual for senior code."
        threshold = 0.90
        log.info(
            "anomaly threshold %.4f = p90 baseline-relative over n=%d  (mean=%.4f std=%.4f p95=%.4f)",
            threshold, s["n"], s["mean"], s["std"], s["p95"],
        )
        result = {
            "threshold": threshold,
            "n_samples": s["n"],
            "method": "anomaly: p90 of baseline-relative TSA over senior corpus (sequence-SNN)",
            "normal_stats": s,
        }
    else:
        log.warning("no pretraining stats available — falling back to labeled accuracy sweep")
        labeled = _load_labeled()
        scored = [(_intensity(net, i["code"], hidden_baseline, output_baseline), int(i["label"])) for i in labeled]
        threshold = _pick_threshold(scored)
        result = {
            "threshold": threshold,
            "n_samples": len(scored),
            "method": "labeled accuracy sweep (fallback)",
        }

    ecdf_ref = torch.tensor(sorted(normal_intensities), dtype=torch.float32) if normal_intensities else None

    def _intensity_ecdf(code: str) -> float:
        raw_scores = _sequence_line_intensities(net, code, hidden_baseline, output_baseline)
        if not raw_scores:
            return 0.0
        if ecdf_ref is not None:
            raw_scores = ecdf_rescale(raw_scores, ecdf_ref)
        return max(raw_scores)

    def _labeled_block(path: Path, name: str) -> dict | None:
        if not path.exists():
            return None
        labeled = json.loads(path.read_text())
        pos = [_intensity_ecdf(i["code"]) for i in labeled if int(i["label"]) == 1]
        neg = [_intensity_ecdf(i["code"]) for i in labeled if int(i["label"]) == 0]

        thr_primary = result["threshold"]
        flag_pos = sum(1 for x in pos if x >= thr_primary) / max(len(pos), 1)
        flag_neg = sum(1 for x in neg if x >= thr_primary) / max(len(neg), 1)
        bal_acc_primary = 0.5 * (flag_pos + (1 - flag_neg))

        # Also report the dataset's optimal threshold (sweep) — records the
        # *ceiling* accuracy the SNN can hit on this dataset, separate from the
        # anomaly-calibrated primary threshold that infer.py ships with. This
        # is diagnostic, not tuned into the primary.
        best_thr_local = thr_primary
        best_bal_acc_local = bal_acc_primary
        if pos and neg:
            candidates = sorted({round(x, 4) for x in pos + neg})
            for thr in candidates:
                fp = sum(1 for x in pos if x >= thr) / len(pos)
                fn_ = sum(1 for x in neg if x >= thr) / len(neg)
                ba = 0.5 * (fp + (1 - fn_))
                if ba > best_bal_acc_local:
                    best_bal_acc_local = ba
                    best_thr_local = thr

        log.info(
            "%s: n_pos=%d n_neg=%d  primary bal_acc=%.3f  optimal bal_acc=%.3f (thr=%.4f)",
            name, len(pos), len(neg), bal_acc_primary, best_bal_acc_local, best_thr_local,
        )
        return {
            "n_positive": len(pos),
            "n_negative": len(neg),
            "positive": _stats(pos),
            "negative": _stats(neg),
            "flag_rate_positive": flag_pos,
            "flag_rate_negative": flag_neg,
            "balanced_accuracy_at_primary_threshold": bal_acc_primary,
            "optimal_threshold": best_thr_local,
            "balanced_accuracy_at_optimal_threshold": best_bal_acc_local,
        }

    # PyResBugs (semantic bugs — expected NOT to separate; that's the finding)
    pyresbugs_block = _labeled_block(LABELED_PATH, "PyResBugs (semantic bugs)")
    if pyresbugs_block:
        result["pyresbugs_stats"] = pyresbugs_block

    # CodeComplex (structural complexity — expected to separate well)
    codecomplex_block = _labeled_block(CODECOMPLEX_PATH, "CodeComplex (complexity)")
    if codecomplex_block:
        result["codecomplex_stats"] = codecomplex_block

    # Annotations (real senior judgments: # noqa, # type: ignore, etc.)
    # — this is THE pitch validation: does the SNN's per-line score rise on
    # lines that senior devs themselves flagged as exceptions?
    annotations_block = _labeled_block(ANNOTATIONS_PATH, "Annotations (senior judgments)")
    if annotations_block:
        result["annotations_stats"] = annotations_block

    THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLD_PATH.write_text(json.dumps(result, indent=2))
    log.info("saved threshold -> %s", THRESHOLD_PATH)


if __name__ == "__main__":
    calibrate()