import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def seam_core():
    path = Path(__file__).parents[1] / "core.py"
    spec = importlib.util.spec_from_file_location("seamrank_core_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def moving_frames():
    import numpy as np

    def generate(start, count=8, speed=2):
        images = np.full((count, 64, 64, 3), 0.12, dtype=np.float32)
        for i, frame in enumerate(images):
            x = 4 + (start + i) * speed
            frame[22:42, x:x + 10] = (0.7, 0.8, 0.65)
        return images
    return generate
