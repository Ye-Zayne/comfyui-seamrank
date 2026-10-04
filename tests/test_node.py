import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


def load_nodes():
    directory = Path(__file__).parents[1]
    spec = importlib.util.spec_from_file_location("seamrank_test_package", directory / "__init__.py",
                                                 submodule_search_locations=[str(directory)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    return package


def test_node_outputs_trimmed_original_and_preview(moving_frames):
    package = load_nodes()
    previous = torch.from_numpy(moving_frames(0))
    continuation = np.concatenate([moving_frames(0)[-3:], moving_frames(8)])
    candidate = torch.from_numpy(continuation)
    winner, trim, text, sheet, index = package.NODE_CLASS_MAPPINGS["SeamRankCandidates"]().rank(
        previous, candidate, motion_backend="pixel_change")
    assert trim == 3 and index == 1
    assert torch.equal(winner, candidate[3:])
    assert winner.untyped_storage().data_ptr() == candidate.untyped_storage().data_ptr()
    assert sheet.ndim == 4 and sheet.shape[0] == 1 and sheet.shape[-1] == 3
    assert 0 <= float(sheet.min()) <= float(sheet.max()) <= 1
    assert json.loads(text)["winner_trim_frames"] == 3


def test_node_reports_full_long_candidate_count(moving_frames):
    node = load_nodes().NODE_CLASS_MAPPINGS["SeamRankCandidates"]()
    previous = torch.from_numpy(moving_frames(0))
    candidate = torch.from_numpy(np.concatenate([moving_frames(8)] * 10))
    _, _, text, _, _ = node.rank(previous, candidate, motion_backend="pixel_change")
    assert json.loads(text)["ranking"][0]["input_frames"] == 80


def test_high_resolution_output_is_untouched_and_overlap_verified(moving_frames):
    node = load_nodes().NODE_CLASS_MAPPINGS["SeamRankCandidates"]()
    previous = torch.from_numpy(moving_frames(0)).repeat_interleave(5, dim=1).repeat_interleave(5, dim=2)
    candidate = torch.cat([previous[-3:], torch.from_numpy(moving_frames(8)).repeat_interleave(5, dim=1).repeat_interleave(5, dim=2)])
    winner, trim, text, _, _ = node.rank(previous, candidate, motion_backend="pixel_change")
    assert trim == 3 and winner.shape[1:3] == (320, 320)
    assert torch.equal(winner, candidate[3:])
    assert json.loads(text)["overlap_confirmation"] == "original_resolution"


def test_invalid_full_resolution_pixel_not_hidden_by_resize(moving_frames):
    import pytest
    node = load_nodes().NODE_CLASS_MAPPINGS["SeamRankCandidates"]()
    previous = torch.from_numpy(moving_frames(0)).repeat_interleave(5, dim=1).repeat_interleave(5, dim=2)
    candidate = previous.clone()
    candidate[0, 51, 51, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or infinity"):
        node.rank(previous, candidate, motion_backend="pixel_change")


def test_downsampled_false_overlap_is_not_trimmed(moving_frames):
    node = load_nodes().NODE_CLASS_MAPPINGS["SeamRankCandidates"]()
    previous = torch.from_numpy(moving_frames(0)).repeat_interleave(8, dim=1).repeat_interleave(8, dim=2)
    continuation = torch.from_numpy(moving_frames(8)).repeat_interleave(8, dim=1).repeat_interleave(8, dim=2)
    false_overlap = previous[-3:].clone()
    row = torch.arange(512)[:, None]
    column = torch.arange(512)[None, :]
    checker = ((row + column) % 2 * 2 - 1).to(torch.float32).unsqueeze(-1) * 0.002
    false_overlap += checker  # averages away at 512 -> 256, but original pixels differ
    candidate = torch.cat([false_overlap, continuation])
    winner, trim, _, _, _ = node.rank(previous, candidate, motion_backend="pixel_change")
    assert trim == 0
    assert torch.equal(winner, candidate)
