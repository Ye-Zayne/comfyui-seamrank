# ComfyUI SeamRank

[简体中文](README.zh-CN.md)

Compare up to four continuation clips against the end of a previous clip. SeamRank ranks local seam evidence, shows the two frames on either side, and removes a confidently matching moving prefix from the selected continuation. It loads no generative model and works with standard ComfyUI `IMAGE` tensors from Wan, LTX, H3, VHS, or other workflows.

Version: **0.1.0** · Python **3.10+** · ComfyUI V1 custom-node API.

## Install

Copy this repository into `ComfyUI/custom_nodes/comfyui-seamrank` and restart ComfyUI. The baseline uses NumPy, PyTorch, and Pillow already used by ComfyUI. OpenCV is optional; install `opencv-python-headless` in the **same Python environment as ComfyUI** if you want optical flow. Do not install it alongside a conflicting OpenCV wheel.

Find **SeamRank · Rank Continuations** under **SeamRank**.

## Connect

`Previous clip IMAGE` + `1–4 continuation IMAGE batches` → `SeamRankCandidates` → review `contact_sheet` with Preview Image or Save Image.

The clips must have the same dimensions, the same FPS, and belong to a continuous shot. Put each complete candidate in its own input; a batch is one clip, not a collection of independent candidates. Each clip needs at least two RGB frames in `[0, 1]`.

| Input | Meaning |
|---|---|
| `previous` | The previous clip; only its tail is inspected. |
| `candidate_1` | Required continuation. |
| `candidate_2` … `candidate_4` | Optional continuations. |
| `fps` | Shared source FPS, default 24. IMAGE contains no FPS metadata; you must declare it correctly. |
| `analysis_frames` | Frames on each side used for local metrics, default 12, range 2–64. |
| `max_trim` | Largest prefix to test for duplication, default 12; 0 disables trimming. |
| `duplicate_tolerance` | Maximum mean RGB error **for every matching frame**, default 0.0001. Use 0 for exact matching. Re-encoded overlaps may need a higher tolerance, with higher false-match risk. |
| `motion_backend` | `auto`, `pixel_change`, or strict `opencv`. |

| Output slot | Name / type | Meaning |
|---|---|---|
| 0 | `winner_after_trim` / IMAGE | Full selected continuation **after** duplicate-prefix trimming, preserving input dtype/device and every remaining pixel. |
| 1 | `trim_frames` / INT | Prefix already removed from output 0. Do not trim it twice. |
| 2 | `ranking_json` / STRING | Every candidate's score, component measurements, reasons, actual motion backend, frame counts, and trim. |
| 3 | `contact_sheet` / IMAGE | Ranked rows: previous last two frames, then continuation first two frames after trim. |
| 4 | `winner_index` / INT | Original input socket number **1–4**, including sparse inputs. |

Append `winner_after_trim` to `previous` with your existing image-batch merge node. SeamRank returns the continuation, not the combined video. It does not route or trim audio. Set video encoding to the original FPS and handle any audio offset separately.

## What the score measures

Exposure and color steps are compared with their normal changes inside the clips. Exposure-normalized seam change is compared with internal frame changes, so uniformly large motion is not penalized just for being large. A repeated candidate following sustained motion is marked as a **possible motion stop**; an intentional hold may look the same. A static clip following another static clip is not penalized.

With OpenCV, Farneback estimates local flow magnitude and coherent direction. Direction is left unmeasured when the field is too weak or conflicting. `auto` falls back if OpenCV cannot be imported and explicitly records that in JSON. `pixel_change` measures appearance changes; **it does not measure direction or physical speed**. Strict `opencv` fails clearly when unavailable.

The score is a fixed heuristic from 0 to 100. Higher means fewer measured seam anomalies, and ties prefer the lower input index. There is no calibrated PASS/FAIL threshold. Inspect the images and reasons; this is not a guarantee of continuity or a quality model.

Duplicate trimming requires a matching moving sequence. Identical static holds are deliberately retained because a still overlap cannot be distinguished from an intended hold. At least two candidate frames always remain.

## Examples

Run `python examples/make_examples.py` from the repository with ComfyUI's Python to create tiny Pillow GIF sequences, individual PNG frames, a ranked contact sheet, and a JSON report in `examples/generated/`. The example needs no ComfyUI server or downloaded model.

Copy `seamrank_previous.gif`, `seamrank_bright.gif`, and `seamrank_good_overlap.gif` into ComfyUI's `input/` folder. The included [API prompt](examples/comfyui_api.json) uses native `LoadImage` to read those animated GIF frames, this node to compare them, and native `SaveImage` to save the contact sheet. Submit it with:

```sh
curl -X POST http://127.0.0.1:8188/prompt \
  -H 'Content-Type: application/json' \
  --data-binary @examples/comfyui_api.json
```

This is **API prompt format**, not a canvas workflow. For production videos, native `LoadVideo` → `GetVideoComponents` (images output) or VHS Load Video can replace the example image loaders. Native-node API references: [image nodes](https://github.com/Comfy-Org/ComfyUI/blob/master/nodes.py), [video nodes](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_video.py).

## Limits

- First release targets continuous shots at identical size and FPS. Hard cuts, flashes, camera transitions, occlusion, intentional stops, and creative color changes need human review.
- Only the local seam window is scored; flaws later in the candidate do not affect its ranking.
- Motion/appearance analysis uses a maximum 128-pixel image side. Small faces, text, or fine detail can be missed. Duplicate matching uses the original inspected pixels.
- It does not assess identity, anatomy, aesthetics, lip sync, narrative continuity, audio, or whether a scene follows its prompt.
- It does not synthesize, retime, interpolate, blend, or repair frames.
- Inputs may already occupy substantial RAM/VRAM. Inspected windows are resized to a maximum 256-pixel side before the CPU copy; each proposed overlap is then confirmed against original pixels one frame pair at a time. The returned winner is not resized.

## Development

```sh
python -m pytest tests -q --rootdir=.. --import-mode=importlib
```

Synthetic tests cover moving overlaps and trim, natural static holds, motion stops, exposure steps, internal-motion normalization, optical-flow direction reversal, missing-OpenCV fallback, tensor outputs, malformed inputs, sparse sockets, and four-candidate ranking. These are algorithm checks, not a real-footage accuracy benchmark.

Related projects already solve parts of video processing: [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite), [comfyUi-deflicker](https://github.com/karcsiha/comfyUi-deflicker), [H3 Project Suite](https://github.com/Adudeguyman/ComfyUI-H3-Project-Suite), and [comfyui-obvpm-timeline](https://github.com/chanon/comfyui-obvpm-timeline). SeamRank's scope is a small, model-independent comparison of continuation candidates; it makes no claim to be the first seam detector.
