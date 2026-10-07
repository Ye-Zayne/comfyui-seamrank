import builtins

import numpy as np
import pytest


def rank(core, previous, candidates, **kwargs):
    return core.rank_candidates(previous, candidates, motion_backend="pixel_change", **kwargs)


def test_exposure_jump_ranks_below_continuation(seam_core, moving_frames):
    previous, good = moving_frames(0), moving_frames(8)
    bad = np.clip(good + 0.18, 0, 1)
    report = rank(seam_core, previous, [bad, good])
    assert report["winner_index"] == 2
    assert report["ranking"][1]["metrics"]["exposure_step"] > 0.15
    assert any("Exposure" in r for r in report["ranking"][1]["reasons"])


def test_moving_tail_to_frozen_candidate_is_review_signal(seam_core, moving_frames):
    previous, good = moving_frames(0), moving_frames(8)
    frozen = np.repeat(good[:1], 8, axis=0)
    report = rank(seam_core, previous, [frozen, good])
    assert report["winner_index"] == 2
    assert report["ranking"][1]["metrics"]["possible_motion_stop"] == 1


def test_static_hold_is_neither_bad_nor_trimmed(seam_core):
    still = np.full((12, 32, 32, 3), 0.3, dtype=np.float32)
    item = rank(seam_core, still, [still])["ranking"][0]
    assert item["trim_frames"] == 0
    assert item["score"] == 100
    assert item["metrics"]["possible_motion_stop"] == 0


def test_exact_moving_overlap_is_trimmed(seam_core, moving_frames):
    previous, good = moving_frames(0), moving_frames(8)
    candidate = np.concatenate([previous[-3:], good])
    report = rank(seam_core, previous, [candidate])
    assert report["winner_trim_frames"] == 3
    assert report["ranking"][0]["remaining_frames"] == 8
    assert report["ranking"][0]["trim_seconds"] == 3 / 24


def test_single_moving_anchor_is_trimmed(seam_core, moving_frames):
    previous = moving_frames(0)
    candidate = np.concatenate([previous[-1:], moving_frames(8)])
    assert rank(seam_core, previous, [candidate])["winner_trim_frames"] == 1


def test_max_trim_zero_preserves_frames(seam_core, moving_frames):
    previous = moving_frames(0)
    candidate = np.concatenate([previous[-3:], moving_frames(8)])
    assert rank(seam_core, previous, [candidate], max_trim=0)["winner_trim_frames"] == 0


def test_constant_large_motion_uses_internal_baseline(seam_core):
    rng = np.random.default_rng(12)
    texture = rng.uniform(0.15, 0.8, (32, 64, 3)).astype(np.float32)
    frames = np.stack([np.roll(texture, 6 * i, axis=1) for i in range(16)])
    item = rank(seam_core, frames[:8], [frames[8:]])["ranking"][0]
    assert item["metrics"]["previous_internal_change"] > 0.1
    assert item["metrics"]["seam_change_excess"] == 0
    assert item["score"] == 100


def test_shape_mismatch_fails_clearly(seam_core, moving_frames):
    with pytest.raises(ValueError, match="dimensions do not match"):
        rank(seam_core, moving_frames(0), [moving_frames(8)[:, :32]])


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_non_finite_input_rejected(seam_core, moving_frames, bad_value):
    images = moving_frames(8)
    images[0, 0, 0, 0] = bad_value
    with pytest.raises(ValueError, match="NaN or infinity"):
        rank(seam_core, moving_frames(0), [images])


def test_sparse_candidate_indices_remain_socket_indices(seam_core, moving_frames):
    report = rank(seam_core, moving_frames(0), [moving_frames(8), moving_frames(8)],
                  candidate_indices=[1, 4])
    assert [i["candidate_index"] for i in report["ranking"]] == [1, 4]
    assert report["winner_index"] == 1


def test_four_candidates_include_every_score(seam_core, moving_frames):
    good = moving_frames(8)
    report = rank(seam_core, moving_frames(0), [np.clip(good + n, 0, 1) for n in (0.15, 0.1, 0.05, 0)])
    assert report["winner_index"] == 4
    assert len(report["ranking"]) == 4
    assert [i["rank"] for i in report["ranking"]] == [1, 2, 3, 4]


def test_missing_opencv_fallback_is_explicit(seam_core, moving_frames, monkeypatch):
    original = builtins.__import__
    def without_opencv(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_opencv)
    report = seam_core.rank_candidates(moving_frames(0), [moving_frames(8)], motion_backend="auto")
    assert report["motion_backend"] == "pixel_change"
    assert "unavailable" in report["motion_backend_note"]
    assert report["ranking"][0]["metrics"]["direction_change"] is None
    with pytest.raises(RuntimeError, match="requested but is unavailable"):
        seam_core.rank_candidates(moving_frames(0), [moving_frames(8)], motion_backend="opencv")


def test_optical_flow_reports_actual_backend(seam_core):
    pytest.importorskip("cv2")
    rng = np.random.default_rng(14)
    texture = rng.uniform(0.1, 0.9, (64, 64, 3)).astype(np.float32)
    frames = np.stack([np.roll(texture, i, axis=1) for i in range(16)])
    reverse = np.stack([np.roll(texture, 8 - i, axis=1) for i in range(8)])
    report = seam_core.rank_candidates(frames[:8], [reverse, frames[8:]], motion_backend="opencv")
    assert report["motion_backend"] == "opencv_farneback"
    assert report["winner_index"] == 2
    bad = next(i for i in report["ranking"] if i["candidate_index"] == 1)
    assert bad["metrics"]["direction_evaluated"]
    assert bad["metrics"]["direction_change"] > 0.8


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_invalid_fps_rejected(seam_core, moving_frames, fps):
    with pytest.raises(ValueError, match="fps"):
        rank(seam_core, moving_frames(0), [moving_frames(8)], fps=fps)
