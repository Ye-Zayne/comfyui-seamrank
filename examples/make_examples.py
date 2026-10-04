"""Generate inspectable small Pillow sequences and run the actual node wrapper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageSequence
import torch


def moving_frames(start: int, count: int = 8) -> list[Image.Image]:
    frames = []
    for i in range(count):
        image = Image.new("RGB", (96, 64), (30, 35, 40))
        x = 6 + (start + i) * 3
        draw = ImageDraw.Draw(image)
        draw.rectangle((x, 22, x + 12, 42), fill=(175, 200, 165))
        draw.line((0, 54, 95, 54), fill=(65, 70, 75))
        frames.append(image)
    return frames


def load_gif(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.stack([np.asarray(frame.convert("RGB"), dtype=np.float32) / 255
                          for frame in ImageSequence.Iterator(image)])
    return torch.from_numpy(array)


def main():
    directory = Path(__file__).resolve().parents[1]
    output = Path(__file__).resolve().parent / "generated"
    output.mkdir(exist_ok=True)
    previous, good = moving_frames(0), moving_frames(8)
    bright = [Image.fromarray(np.clip(np.asarray(frame).astype(np.int16) + 40, 0, 255).astype(np.uint8))
              for frame in good]
    sequences = {"previous": previous, "bright": bright, "good_overlap": previous[-3:] + good}
    for name, frames in sequences.items():
        frames[0].save(output / f"seamrank_{name}.gif", save_all=True, append_images=frames[1:],
                       duration=40, loop=0, optimize=False, disposal=2)
        folder = output / name
        folder.mkdir(exist_ok=True)
        for index, frame in enumerate(frames):
            frame.save(folder / f"{index:03d}.png")
    spec = importlib.util.spec_from_file_location("seamrank_example", directory / "__init__.py",
                                                 submodule_search_locations=[str(directory)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    node = package.NODE_CLASS_MAPPINGS["SeamRankCandidates"]()
    winner, trim, report, contact, index = node.rank(
        load_gif(output / "seamrank_previous.gif"), load_gif(output / "seamrank_bright.gif"),
        candidate_2=load_gif(output / "seamrank_good_overlap.gif"), fps=25,
        motion_backend="pixel_change")
    Image.fromarray(np.rint(contact[0].numpy() * 255).astype(np.uint8)).save(output / "contact_sheet.png")
    (output / "ranking.json").write_text(json.dumps(json.loads(report), indent=2), encoding="utf-8")
    assert index == 2 and trim == 3 and len(winner) == 8
    print(f"Candidate {index}, trim {trim}; examples saved to {output}")


if __name__ == "__main__":
    main()
