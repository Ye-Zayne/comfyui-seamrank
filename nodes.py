"""ComfyUI V1 API wrapper. Uses standard IMAGE inputs and outputs."""

from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image, ImageDraw

from .core import rank_candidates


def _check_image(images: torch.Tensor, name: str) -> None:
    if not isinstance(images, torch.Tensor):
        raise ValueError(f"{name} must be a ComfyUI IMAGE tensor.")
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"{name} must be [frames, height, width, 3] RGB.")
    if images.shape[0] < 2 or min(images.shape[1:3]) < 2:
        raise ValueError(f"{name} needs at least two frames and a 2 × 2 image.")
    if not images.is_floating_point():
        raise ValueError(f"{name} must be floating-point RGB in [0, 1].")


def _as_numpy(images: torch.Tensor) -> np.ndarray:
    return images.detach().to(device="cpu", dtype=torch.float32).numpy()


def _analysis_numpy(images: torch.Tensor, name: str, max_side: int = 256) -> np.ndarray:
    low, high = torch.aminmax(images)
    if not bool(torch.isfinite(low) & torch.isfinite(high)):
        raise ValueError(f"{name} contains NaN or infinity.")
    if float(low) < -1e-6 or float(high) > 1.000001:
        raise ValueError(f"{name} must be RGB in [0, 1].")
    h, w = images.shape[1:3]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        images = torch.nn.functional.interpolate(
            images.permute(0, 3, 1, 2), size=(max(2, round(h * scale)), max(2, round(w * scale))),
            mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    return _as_numpy(images)


def make_contact_sheet(previous, candidates, ranking) -> torch.Tensor:
    """Two source frames plus two continuation frames for each ranked row."""
    thumb_w, thumb_h = 160, 120
    row_height, header = thumb_h + 46, 24
    sheet = Image.new("RGB", (thumb_w * 4, header + row_height * len(ranking)), (24, 27, 33))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 6), "Previous tail (left two)  |  Continuation after trim (right two)", fill="white")
    for row, item in enumerate(ranking):
        y = header + row * row_height
        index, trim = item["candidate_index"], item["trim_frames"]
        draw.text((8, y + 4), f"Rank {item['rank']}   Candidate {index}   Score {item['score']:.1f}   Trim {trim}", fill=(220, 229, 239))
        frames = [previous[-2], previous[-1], candidates[index][trim], candidates[index][trim + 1]]
        for column, frame in enumerate(frames):
            pixels = np.rint(np.clip(_as_numpy(frame), 0, 1) * 255).astype(np.uint8)
            thumbnail = Image.fromarray(pixels).copy()
            thumbnail.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            x = column * thumb_w + (thumb_w - thumbnail.width) // 2
            sheet.paste(thumbnail, (x, y + 23 + (thumb_h - thumbnail.height) // 2))
        reason = item["reasons"][0]
        draw.text((8, y + thumb_h + 26), reason[:96], fill=(190, 204, 219))
    return torch.from_numpy(np.asarray(sheet).copy()).to(torch.float32).div(255).unsqueeze(0)


class SeamRankCandidates:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "previous": ("IMAGE",),
                "candidate_1": ("IMAGE",),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 1000.0}),
                "analysis_frames": ("INT", {"default": 12, "min": 2, "max": 64}),
                "max_trim": ("INT", {"default": 12, "min": 0, "max": 64}),
                "duplicate_tolerance": ("FLOAT", {"default": 0.0001, "min": 0.0, "max": 0.02, "step": 0.0001}),
                "motion_backend": (["auto", "pixel_change", "opencv"],),
            },
            "optional": {"candidate_2": ("IMAGE",), "candidate_3": ("IMAGE",), "candidate_4": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING", "IMAGE", "INT")
    RETURN_NAMES = ("winner_after_trim", "trim_frames", "ranking_json", "contact_sheet", "winner_index")
    FUNCTION = "rank"
    CATEGORY = "SeamRank"
    DESCRIPTION = "Rank 1–4 continuation clips by local seam evidence. Same FPS and size, continuous shots only. Returns the selected continuation AFTER confirmed duplicate-prefix trim."

    def rank(self, previous, candidate_1, fps=24.0, analysis_frames=12,
             max_trim=12, duplicate_tolerance=0.0001, motion_backend="auto",
             candidate_2=None, candidate_3=None, candidate_4=None):
        clips = {index: clip for index, clip in enumerate(
            (candidate_1, candidate_2, candidate_3, candidate_4), start=1) if clip is not None}
        _check_image(previous, "previous")
        for index, clip in clips.items():
            _check_image(clip, f"candidate_{index}")
            if tuple(clip.shape[1:]) != tuple(previous.shape[1:]):
                raise ValueError(f"candidate_{index} dimensions do not match previous; resize explicitly before ranking.")
        # Only move inspected windows to CPU. The chosen output remains a view
        # of the caller's tensor, in its original dtype/device and original FPS.
        previous_window = previous[-max(analysis_frames, max_trim, 2):]
        limit = analysis_frames + max_trim + 2
        def confirm(index: int, trim: int) -> float:
            # Downsampling may conceal fine differences. Confirm a proposed
            # overlap against original full-resolution pixels, one pair at a time.
            candidate = clips[index]
            errors = [float(torch.mean(torch.abs(old - new), dtype=torch.float32))
                      for old, new in zip(previous[-trim:], candidate[:trim])]
            return max(errors)
        report = rank_candidates(
            _analysis_numpy(previous_window, "previous"),
            [_analysis_numpy(clip[:limit], f"candidate_{index}") for index, clip in clips.items()],
            fps=fps, analysis_frames=analysis_frames, max_trim=max_trim,
            duplicate_tolerance=duplicate_tolerance, motion_backend=motion_backend,
            candidate_indices=list(clips), confirm_overlap=confirm)
        for item in report["ranking"]:
            full_count = len(clips[item["candidate_index"]])
            item["input_frames"] = full_count
            item["remaining_frames"] = full_count - item["trim_frames"]
        report["previous_frames"] = len(previous)
        report["inspected_previous_frames"] = len(previous_window)
        report["source_dimensions"] = {"width": previous.shape[2], "height": previous.shape[1]}
        report["overlap_confirmation"] = "original_resolution"
        index, trim = report["winner_index"], report["winner_trim_frames"]
        sheet = make_contact_sheet(previous, clips, report["ranking"])
        return (clips[index][trim:], trim, json.dumps(report, ensure_ascii=False, allow_nan=False), sheet, index)
