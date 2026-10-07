"""Small-window, explainable seam metrics. No generative model is loaded."""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np


LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def validate_sequence(images: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(images)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have IMAGE shape [frames, height, width, 3].")
    if array.shape[0] < 2 or min(array.shape[1:3]) < 2:
        raise ValueError(f"{name} needs at least two frames and a 2 × 2 image.")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{name} must be floating-point RGB in [0, 1].")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    if float(array.min()) < -1e-6 or float(array.max()) > 1.000001:
        raise ValueError(f"{name} must be RGB in [0, 1].")
    return array


def _small(images: np.ndarray, max_side: int = 128) -> np.ndarray:
    h, w = images.shape[1:3]
    scale = min(1.0, max_side / max(h, w))
    yi = np.linspace(0, h - 1, max(2, round(h * scale))).astype(int)
    xi = np.linspace(0, w - 1, max(2, round(w * scale))).astype(int)
    return images[:, yi][:, :, xi].astype(np.float32, copy=False)


def _pixel_change(a: np.ndarray, b: np.ndarray) -> float:
    """Exposure-normalized pixel change, explicitly not physical velocity."""
    ga, gb = a @ LUMA, b @ LUMA
    return float(np.abs((ga - ga.mean()) - (gb - gb.mean())).mean())


def _changes(images: np.ndarray) -> np.ndarray:
    return np.array([_pixel_change(a, b) for a, b in zip(images[:-1], images[1:])])


def detect_overlap(previous: np.ndarray, candidate: np.ndarray, max_trim: int,
                   tolerance: float) -> tuple[int, float | None]:
    """Only trim matching prefixes with evidence of motion, never static holds."""
    maximum = min(max_trim, len(previous), len(candidate) - 2)
    for count in range(maximum, 0, -1):
        old = previous[-count:]
        new = candidate[:count]
        errors = [float(np.abs(a - b).mean()) for a, b in zip(old, new)]
        if max(errors) > tolerance:
            continue
        if count > 1:
            varied = max(float(np.abs(a - b).mean()) for a, b in zip(old[:-1], old[1:]))
            confident = varied > max(tolerance * 4, 0.0005)
        else:
            entering = float(np.abs(previous[-1] - previous[-2]).mean())
            leaving = float(np.abs(candidate[1] - candidate[0]).mean())
            confident = min(entering, leaving) > max(tolerance * 4, 0.0005)
        if confident:
            return count, max(errors)
    return 0, None


def _flow_backend(mode: str):
    if mode not in {"auto", "pixel_change", "opencv"}:
        raise ValueError("motion_backend must be auto, pixel_change, or opencv.")
    if mode == "pixel_change":
        return None, "pixel_change", "Direction is unavailable; pixel change is not optical flow or physical speed."
    try:
        import cv2
        if not callable(getattr(cv2, "calcOpticalFlowFarneback", None)):
            raise ImportError("cv2.calcOpticalFlowFarneback is unavailable")
        return cv2, "opencv_farneback", "Direction is measured only when the flow field has a coherent resultant."
    except (ImportError, OSError) as error:
        if mode == "opencv":
            raise RuntimeError("OpenCV optical flow was requested but is unavailable. Install opencv-python-headless or choose pixel_change.") from error
        return None, "pixel_change", "OpenCV is unavailable: using exposure-normalized pixel change; direction and physical speed are unavailable."


def _flow_pair(a: np.ndarray, b: np.ndarray, cv2) -> dict:
    ga = np.rint(np.clip(a @ LUMA, 0, 1) * 255).astype(np.uint8)
    gb = np.rint(np.clip(b @ LUMA, 0, 1) * 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude = np.linalg.norm(flow, axis=-1)
    # Ignore quiet backgrounds when estimating direction, but retain their
    # contribution in the motion magnitude. Conflicting flows are not a direction.
    mask = magnitude >= max(0.1, float(np.percentile(magnitude, 70)))
    vector = flow[mask].mean(axis=0) if mask.any() else np.zeros(2)
    selected = float(magnitude[mask].mean()) if mask.any() else 0.0
    coherence = float(np.linalg.norm(vector) / max(selected, 1e-6))
    diagonal = math.hypot(*ga.shape)
    return {"magnitude": float(magnitude.mean()) / diagonal,
            "vector": vector.astype(float) / diagonal,
            "coherence": min(1.0, coherence)}


def _motion_details(previous: np.ndarray, candidate: np.ndarray, cv2) -> dict:
    if cv2 is None:
        return {"direction_change": None, "flow_speed_change": None,
                "direction_evaluated": False}
    tail = previous[-5:]
    prior = [_flow_pair(a, b, cv2) for a, b in zip(tail[:-1], tail[1:])]
    following = [_flow_pair(a, b, cv2) for a, b in zip(candidate[:4], candidate[1:5])]
    pv = np.mean([item["vector"] for item in prior], axis=0)
    cv = np.mean([item["vector"] for item in following], axis=0)
    pm = float(np.median([item["magnitude"] for item in prior]))
    cm = float(np.median([item["magnitude"] for item in following]))
    # Very small or conflicting fields cannot reliably establish direction.
    reliable = (min(np.linalg.norm(pv), np.linalg.norm(cv)) > 0.0005
                and min(np.mean([i["coherence"] for i in prior]),
                        np.mean([i["coherence"] for i in following])) > 0.35)
    direction = None
    if reliable:
        cosine = float(np.dot(pv, cv) / (np.linalg.norm(pv) * np.linalg.norm(cv)))
        direction = (1.0 - np.clip(cosine, -1, 1)) / 2.0
    speed_change = abs(math.log((cm + 0.001) / (pm + 0.001)))
    return {"direction_change": float(direction) if direction is not None else None,
            "flow_speed_change": speed_change, "direction_evaluated": bool(reliable),
            "previous_flow_magnitude": pm, "candidate_flow_magnitude": cm,
            "previous_flow_vector": pv.tolist(), "candidate_flow_vector": cv.tolist()}


def rank_candidates(previous: np.ndarray, candidates: Sequence[np.ndarray], *,
                    fps: float = 24.0, analysis_frames: int = 12, max_trim: int = 12,
                    duplicate_tolerance: float = 0.0001,
                    motion_backend: str = "auto", candidate_indices: Sequence[int] | None = None,
                    confirm_overlap: Callable[[int, int], float] | None = None) -> dict:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite number shared by every clip.")
    if not 2 <= analysis_frames <= 64 or not 0 <= max_trim <= 64:
        raise ValueError("analysis_frames must be 2–64 and max_trim must be 0–64.")
    if not math.isfinite(duplicate_tolerance) or not 0 <= duplicate_tolerance <= 0.02:
        raise ValueError("duplicate_tolerance must be between 0 and 0.02.")
    if not 1 <= len(candidates) <= 4:
        raise ValueError("Connect between one and four candidates.")
    indices = list(candidate_indices) if candidate_indices is not None else list(range(1, len(candidates) + 1))
    if len(indices) != len(candidates) or len(set(indices)) != len(indices):
        raise ValueError("candidate_indices must be distinct and match the candidate count.")
    previous = validate_sequence(previous, "previous")
    cv2, backend, backend_note = _flow_backend(motion_backend)
    results = []
    for index, raw_candidate in zip(indices, candidates):
        candidate = validate_sequence(raw_candidate, f"candidate_{index}")
        if candidate.shape[1:] != previous.shape[1:]:
            raise ValueError(f"candidate_{index} dimensions do not match previous; resize explicitly before ranking.")
        trim, overlap_error = detect_overlap(previous, candidate, max_trim, duplicate_tolerance)
        if trim and confirm_overlap is not None:
            original_error = confirm_overlap(int(index), trim)
            if not math.isfinite(original_error) or original_error > duplicate_tolerance:
                trim, overlap_error = 0, None
            else:
                overlap_error = original_error
        p = _small(previous[-analysis_frames:])
        c = _small(candidate[trim:trim + analysis_frames])
        pd, cd = _changes(p), _changes(c)
        prior_change = float(np.median(pd[-4:]))
        following_change = float(np.median(cd[:4]))
        typical_change = max(prior_change, following_change, 0.005)
        seam_change = _pixel_change(p[-1], c[0])
        natural_high = max(float(np.percentile(pd, 90)), float(np.percentile(cd, 90)), 0.005)
        seam_excess = max(0.0, seam_change - natural_high * 1.75) / typical_change
        pm = p.mean(axis=(1, 2))
        cm = c.mean(axis=(1, 2))
        brightness_step = abs(float((cm[0] - pm[-1]) @ LUMA))
        internal_luma = np.concatenate([np.abs(np.diff(pm @ LUMA)), np.abs(np.diff(cm @ LUMA))])
        natural_luma = float(np.percentile(internal_luma, 90))
        brightness_excess = max(0.0, brightness_step - 2 * natural_luma - 0.015) / 0.10
        pchroma = pm - (pm @ LUMA)[:, None]
        cchroma = cm - (cm @ LUMA)[:, None]
        color_step = float(np.abs(cchroma[0] - pchroma[-1]).mean())
        internal_color = np.concatenate([np.abs(np.diff(pchroma, axis=0)).mean(axis=1),
                                         np.abs(np.diff(cchroma, axis=0)).mean(axis=1)])
        color_excess = max(0.0, color_step - 2 * float(np.percentile(internal_color, 90)) - 0.01) / 0.10
        # Static footage on both sides is legitimate. Only report a possible
        # motion stop when the preceding clip has sustained recent movement.
        stopped_fraction = float(np.mean(cd[:min(6, len(cd))] <= 0.0005))
        motion_stop = stopped_fraction if prior_change > 0.002 and following_change < prior_change * 0.15 else 0.0
        motion = _motion_details(p, c, cv2)
        loss = (3 * min(brightness_excess, 4) + 2 * min(color_excess, 4)
                + 2 * min(seam_excess, 4) + 3 * motion_stop)
        if motion["direction_change"] is not None:
            loss += 1.5 * motion["direction_change"]
        if motion["flow_speed_change"] is not None:
            loss += 0.8 * min(motion["flow_speed_change"], 2)
        reasons = []
        if trim:
            reasons.append(f"Matching moving overlap: trim {trim} frame(s).")
        if brightness_excess > 0.5:
            reasons.append("Exposure step exceeds changes inside either clip.")
        if color_excess > 0.5:
            reasons.append("Color step exceeds changes inside either clip.")
        if seam_excess > 0.75:
            reasons.append("Structural seam change exceeds internal frame changes.")
        if motion_stop:
            reasons.append("Possible motion stop after a moving tail; review intentional holds.")
        if (motion["direction_change"] or 0) > 0.75:
            reasons.append("Coherent motion direction reverses near the seam.")
        if (motion["flow_speed_change"] or 0) > 1:
            reasons.append("Flow magnitude changes substantially near the seam.")
        if not reasons:
            reasons.append("No large seam anomaly in the inspected window.")
        results.append({"candidate_index": int(index), "trim_frames": trim,
                        "trim_seconds": trim / fps, "overlap_max_mean_error": overlap_error,
                        "input_frames": len(candidate), "remaining_frames": len(candidate) - trim,
                        "score": round(100 * math.exp(-loss / 5), 4),
                        "loss": float(loss), "reasons": reasons,
                        "metrics": {"exposure_step": brightness_step, "color_step": color_step,
                                    "exposure_excess": brightness_excess, "color_excess": color_excess,
                                    "seam_change": seam_change, "seam_change_excess": seam_excess,
                                    "previous_internal_change": prior_change,
                                    "candidate_internal_change": following_change,
                                    "possible_motion_stop": motion_stop,
                                    "repeated_pair_fraction": float(np.mean(cd <= 0.0005)), **motion}})
    results.sort(key=lambda item: (-item["score"], item["candidate_index"]))
    for position, result in enumerate(results, start=1):
        result["rank"] = position
    return {"schema_version": 1, "package_version": "0.1.1", "fps": float(fps),
            "motion_backend": backend, "motion_backend_note": backend_note,
            "analysis_frames": analysis_frames, "winner_index": results[0]["candidate_index"],
            "winner_trim_frames": results[0]["trim_frames"], "ranking": results,
            "limitations": ["Continuous shots at the same resolution and FPS only; FPS is a user declaration, not inferred from IMAGE.",
                            "Scores compare local seam evidence, not overall aesthetics, identity, narration, or lip sync.",
                            "Static matching holds are not trimmed because overlap is ambiguous.",
                            "No frames are synthesized, interpolated, or resampled by this node."]}
