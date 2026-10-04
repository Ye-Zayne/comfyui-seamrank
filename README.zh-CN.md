# ComfyUI SeamRank

[English](README.md)

把上一段视频的结尾与最多 4 个续写候选比较，给出接片排名、分项原因和接缝对照图。确定存在重复的运动片段时，会从选中候选的开头去掉该部分。无需生成模型，支持 Wan、LTX、H3、VHS 等工作流输出的标准 `IMAGE`。

版本 **0.1.0**，Python **3.10+**，ComfyUI V1 节点 API。

## 安装

放到 `ComfyUI/custom_nodes/comfyui-seamrank`，重启 ComfyUI。基础功能使用 ComfyUI 已有的 NumPy、PyTorch、Pillow。光流功能可选：在 ComfyUI 使用的同一 Python 环境安装 `opencv-python-headless`，避免和其他 OpenCV wheel 同时安装。

节点菜单：**SeamRank → SeamRank · Rank Continuations**。

## 连接方法

上一段 `IMAGE` → `previous`；每个完整续写视频分别连接 `candidate_1`～`candidate_4`；`contact_sheet` 连接 Preview Image 或 Save Image。

所有视频必须属于连续镜头，尺寸与 FPS 一致。一个候选输入中的 IMAGE batch 是一段视频，不能把几个候选混进同一个 batch。每段至少两帧，RGB 浮点值在 `[0, 1]`。

| 输入 | 说明 |
|---|---|
| `previous` | 上一段视频，仅检查尾部。 |
| `candidate_1` | 必填候选。 |
| `candidate_2`～`candidate_4` | 可选候选。 |
| `fps` | 所有视频共有的实际 FPS，默认 24。IMAGE 不带帧率元数据，由用户正确填写。 |
| `analysis_frames` | 接缝两侧用于分析的帧数，默认 12，范围 2～64。 |
| `max_trim` | 最多检查多少个重复前缀帧，默认 12；设 0 关闭裁切。 |
| `duplicate_tolerance` | 每一对匹配帧允许的最大 RGB 平均误差，默认 0.0001；0 表示精确相同。压缩视频提高此值会增加误匹配风险。 |
| `motion_backend` | 自动 `auto`、像素变化 `pixel_change`、必须使用光流 `opencv`。 |

| 输出位置 | 名称 / 类型 | 说明 |
|---|---|---|
| 0 | `winner_after_trim` / IMAGE | 已去掉重复前缀的完整胜出候选，保留剩余帧像素及原张量 dtype/device。 |
| 1 | `trim_frames` / INT | 输出 0 已经裁掉的帧数，不要再次裁切。 |
| 2 | `ranking_json` / STRING | 所有候选排名、分项测量、原因、实际算法、裁切量和帧数。 |
| 3 | `contact_sheet` / IMAGE | 每行按排名排列：左侧上一段末两帧，右侧候选裁切后前两帧。 |
| 4 | `winner_index` / INT | 原始候选接口编号 1～4，空接口不会改变编号。 |

用已有的图片批次合并节点，把 `winner_after_trim` 拼在 `previous` 后。输出 0 是续写候选，不是已经合并的视频。此节点不处理音频，合成视频使用原 FPS，音频偏移另行处理。

## 评分依据

曝光、颜色阶跃会和视频内部正常变化比较；接缝画面变化也会参考内部帧间变化，避免因为动作幅度大就直接扣分。上一段持续运动、候选突然重复静止时，会标记“可能的运动停止”；有意停顿也可能得到这个提示。两段本来都静止不会扣分。

OpenCV 可用时使用 Farneback 光流，估计运动量和足够一致的运动方向；运动太小或方向互相抵消时，不报告可靠方向。`auto` 在无法导入 OpenCV 时退回像素变化，JSON 会说明实际后端。`pixel_change` 只测量曝光归一后的像素变化，**不测运动方向或物理速度**。`opencv` 模式缺少 OpenCV 时明确报错。

0～100 分是固定启发式分数，用于比较接缝异常，尚无经过标定的通过/不通过阈值。同分优先接口编号较小的候选。需要检查缩略图和原因，不能把分数视为连续性保证。

重复裁切要求匹配片段中存在运动。完全相同的静止画面无法区分重复上下文与有意停留，因此不会自动裁切；至少保留候选的两帧。

## 示例和测试

在仓库目录运行 `python examples/make_examples.py`，会在 `examples/generated/` 生成 Pillow 动画 GIF、逐帧 PNG、接缝对照图和 JSON。无需 ComfyUI 服务或模型。

将 `seamrank_previous.gif`、`seamrank_bright.gif`、`seamrank_good_overlap.gif` 复制到 ComfyUI `input/`。英文 README 的 curl 示例可提交 [API prompt](examples/comfyui_api.json)，使用原生 LoadImage 读取 GIF 帧、SeamRank 排名、SaveImage 保存对照图。这是 API 格式，不是画布 workflow。真实视频可换成原生 `LoadVideo → GetVideoComponents` 或 VHS Load Video 的 images 输出。

```sh
python -m pytest tests -q --rootdir=.. --import-mode=importlib
```

合成测试覆盖曝光跳变、运动停止、静止保持、重复前缀和裁切、较大自然动作、光流方向反转、缺少 OpenCV 时的明确降级、尺寸错误、空帧/非有限值、稀疏接口和 4 个候选。合成测试不代表真实视频准确率基准。

## 边界

- 第一版面向同尺寸、同 FPS 的连续镜头；硬切、闪光、遮挡、创意调色和有意停止需要人工判断。
- 只评分接缝附近，候选后半段坏帧不会影响排名。
- 运动与画面变化分析最长边为 128 像素，可能漏掉小脸、文字和细节；重复前缀比较使用原始检查帧像素。
- 不评价人物身份、手脚、审美、口型、故事、提示词执行或音频。
- 不生成、补帧、改时长、混合或修复画面。
- 输入本身可能占用较多 RAM/VRAM；检查窗口先缩小到最长边 256 再复制到 CPU，重复前缀的裁切建议会逐帧回到原像素确认。胜出候选输出不会缩小。

已有 [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)、[comfyUi-deflicker](https://github.com/karcsiha/comfyUi-deflicker)、[H3 Project Suite](https://github.com/Adudeguyman/ComfyUI-H3-Project-Suite)、[comfyui-obvpm-timeline](https://github.com/chanon/comfyui-obvpm-timeline) 覆盖视频处理的部分能力。本包专注模型无关的续写候选比较，不声称接缝检测此前不存在。
