# Dataset Guide

本说明面向 `oi-mi` 数据集使用者，介绍数据目录、文件格式、实验范式、标签语义，以及使用这些数据时需要注意的事项。

## 数据根目录

典型目录如下：

```text
/mnt/dataset1/xkp/oi-mi/
├── records_storage/
│   └── S002/
│       ├── calibration/
│       │   └── 20260417_194441/
│       ├── test_mode/
│       │   ├── manifest.json
│       │   └── chunks/
│       └── realtime/
│           ├── manifest.json
│           └── chunks/
└── models_storage/
    └── S002/
        └── brainco/
            ├── eegnet.pt
            └── eegnet.metrics.yaml
```

说明：

- `records_storage/<subject_id>/calibration/<session_id>/`
  校准模式数据。`session_id` 为时间戳。
- `records_storage/<subject_id>/test_mode/`
  测试模式数据，按 chunk 分块保存。
- `records_storage/<subject_id>/realtime/`
  实时解码模式下保存的在线推理数据。
- `models_storage/<subject_id>/<device>/`
  被试个体模型及训练指标。

## 模式概览

本数据集包含三类会话：

1. `calibration`
   有 cue 的校准/训练采集，用于生成训练窗和个体模型。
2. `test_mode`
   有 cue 的在线测试，保存逐窗原始 EEG、真实标签和模型预测。
3. `realtime`
   无 cue 的实时解码，通常只有原始窗和模型输出，没有真实标签。

三分类标签固定为：

- `0`: `left`
- `1`: `right`
- `2`: `idle`

## 文件说明

### Calibration 目录

每个 `calibration/<session_id>/` 通常包含：

- `continuous_eeg.npy`
  原始连续 EEG，形状通常为 `(n_channels, n_samples)`，`float32`。
- `events.json`
  会话事件时间轴。
- `metadata.json`
  协议参数、trial 对齐信息、标签映射、训练摘要等。
- `training_windows_main.npz`
  训练主数据集。
- `training_windows_aux_1p5s.npz`
  可选的辅助短窗版本。

`training_windows_main.npz` 的主要字段：

- `raw_windows`
  从 `continuous_eeg.npy` 上切出来的原始窗，形状 `(n_windows, n_channels, n_times)`。
- `processed_windows`
  经过预处理后的模型输入，形状同上。
- `labels`
  逐窗标签，形状 `(n_windows,)`。
- `sfreq`
- `window_sec`
- `step_sec`

### Test Mode / Realtime 目录

`test_mode/` 与 `realtime/` 的保存格式类似，核心文件为：

- `manifest.json`
  会话级摘要。
- `chunks/chunk_*.npz`
  分块保存的逐窗数据。

每个 `chunk_*.npz` 的主要字段：

- `eeg_windows`
  原始 EEG 窗，形状 `(n_windows, n_channels, n_times)`。
- `labels_true`
  真实标签。
  在 `test_mode` 中为 `0/1/2`。
  在 `realtime` 中通常为 `-1`。
- `labels_pred`
  模型输出类别。
  若使用置信度阈值抑制，则低置信度窗可能为 `-1`。
- `confidences`
  每个窗的最大类别概率。

### 模型目录

- `eegnet.pt`
  PyTorch 权重。
- `eegnet.metrics.yaml`
  训练摘要，例如：
  `windows_collected`、`val_loss`、`val_acc`、`window_sec`、`step_sec`。

## 原始数据与派生数据

不是所有文件都是“原始 EEG”。

原始或接近原始的数据：

- `continuous_eeg.npy`
- `training_windows_main.npz` 中的 `raw_windows`
- `chunk_*.npz` 中的 `eeg_windows`
- `events.json`

派生或处理后的数据：

- `training_windows_main.npz` 中的 `processed_windows`
- `labels_true`
- `labels_pred`
- `confidences`
- `metadata.json`
- `manifest.json`
- `*.metrics.yaml`

建议：

- 如果你要自行重切窗或重做预处理，优先使用 `continuous_eeg.npy`。
- 如果你要复现项目默认训练输入，可直接使用 `processed_windows`。
- 如果你要分析在线表现，优先使用 `test_mode/chunks/chunk_*.npz`。

## Calibration 范式

Calibration 是“协议驱动的 cue 采集”。

默认时序来自 `metadata.json` 中的 `trial_timing`：

- `fixation_sec = 2.0`
- `cue_sec = 1.0`
- `control_sec = 5.0`
- `iti_sec = 2.0`

默认还包含：

- baseline：60s 睁眼注视中央十字
- 每次实验都从头校准，不区分新/老被试
- `4` 个 block
- 每个 block：每类 `5` 个 trial
- 三类：`left / right / idle`
- block 间休息：`20s`

因此正式采集时间为：

- baseline：`60s`
- 60 个 trial × `10s`：`600s`
- 3 次休息 × `20s`：`60s`
- 合计：`720s = 12min`

对训练真正生效的窗，不是整个 trial，而是 `control` 时段中的一部分：

- `control_window_range_sec = [0.5, 4.5]`

当 `window_sec=2.0`、`step_sec=0.5` 时，训练窗的起点通常为：

- `0.5`
- `1.0`
- `1.5`
- `2.0`
- `2.5`

对应的时间范围为：

- `0.5 - 2.5`
- `1.0 - 3.0`
- `1.5 - 3.5`
- `2.0 - 4.0`
- `2.5 - 4.5`

质量筛选前每个 trial 产生 5 个窗口，共 300 个窗口、每类 100 个。数据文件保存
`trial_ids`，训练/验证按 trial 分组拆分，避免同一 trial 的重叠窗口同时进入两边。

这意味着 calibration 训练数据来自较“干净”的 control 段，而不是整个 cue block。

## Test Mode 范式

`test_mode` 更接近在线滑窗解码。

基本流程：

1. 发出 cue 标签
2. 等待 `window_sec`
3. 随后每 `step_sec` 滑一次窗做预测
4. 在一个较长的 block 内连续保存多个窗口

因此：

- `test_mode` 的每个 cue 会产生多个连续窗
- 这些窗通常全部打成当前 cue 的标签
- 它比 calibration 更接近真实在线场景
- 它也更容易受到状态切换、疲劳、残留脑状态和动作不稳定的影响

## Calibration 与 Test Mode 的关键差异

两者的输入尺寸和预处理通常一致：

- 采样率相同
- 通道数相同
- `window_sec` 相同
- `step_sec` 相同
- 都使用同一个 `filter_and_transform(...)`

但它们的“标签语义”和“切窗时机”不完全一样：

- calibration 只取 control 段内的部分窗
- test mode 会在更长的 cue block 上连续滑窗
- calibration 的训练窗更干净
- test mode 更容易包含过渡态和后期漂移

因此：

- calibration 上表现好，不代表 test mode 一定同样好
- 若要做严格评估，建议明确记录每个 test 窗相对 cue 的时间偏移

## 预处理

默认预处理定义在 `utils/preprocessing.py`：

1. 检查非有限值、平直导联和相对异常高噪导联
2. 无 montage 坐标时，以健康导联逐点中位数稳健修复坏导
3. Common Average Reference
4. `0.3-40 Hz` 五阶 Butterworth SOS 零相位带通
5. 记录峰值、超幅比例和坏导比例
6. 为数值安全将模型输入限制到 `[-150, 150] uV`

对应函数：

- `common_average_reference`
- `bandpass_filter`
- `reject_artifacts`
- `preprocess_eeg_window`
- `filter_and_transform`

这里没有逐窗 z-score：CBraMod 的 MI 预处理也保留微伏幅值，而
ShallowConvNet 的运动想象特征依赖节律功率变化。`reject_artifacts` 只是兼容旧
调用的数值保护，真正的质量判定来自 `preprocess_eeg_window().quality`。校准和
NeuroOnline 只接收 `quality.accepted=True` 的窗口，实时落盘仍保留原始窗口和
质量字段，便于事后复核。

## Metadata 能看出什么

`metadata.json` 可以直接告诉你：

- 会话类型和协议名
- 被试模式：`new` 或 `old`
- 标签映射
- trial 时序参数
- baseline 配置
- 每个 trial 的标签、block 编号、trial 编号
- `control_on_sample / control_off_sample`
- 训练窗口范围

但 `metadata` 本身不能告诉你：

- 被试是否认真执行了该类想象
- 某个时间段是否疲劳、走神或有运动伪迹
- 在线窗是否正处于类别切换或状态残留

这些需要结合原始 EEG、逐窗标签、模型概率和额外标注来判断。

## 使用建议

### 如果你要做离线训练

优先使用：

- `training_windows_main.npz` 的 `processed_windows + labels`

若要自定义训练流程：

- 用 `continuous_eeg.npy + metadata.json + events.json` 自行重切窗

### 如果你要做协议分析

优先使用：

- `metadata.json`
- `events.json`

### 如果你要做在线解码评估

优先使用：

- `test_mode/chunks/chunk_*.npz`

建议额外统计：

- confusion matrix
- 各类 precision / recall
- 每类置信度分布
- 不同阈值下的 coverage / valid accuracy
- 窗口相对 cue 时间偏移与准确率的关系

### 如果你要做预训练 + 个体化

可以用 `transfer_personalize.py` 在一个或多个源被试上预训练，再用目标被试 calibration 微调，并直接回放目标被试 `test_mode`：

```bash
conda activate uni

python transfer_personalize.py \
  --records-dir /mnt/dataset1/xkp/oi-mi/records_storage \
  --source-subject S001 \
  --target-subject S002 \
  --test-chunk /mnt/dataset1/xkp/oi-mi/records_storage/S002/test_mode/chunks/chunk_000000.npz \
  --model eegnet \
  --pretrain-epochs 35 \
  --finetune-epochs 20 \
  --threshold 0.5 \
  --smooth 1 --smooth 3 --smooth 5 \
  --output-json /tmp/oi_mi_transfer/s001_to_s002_eegnet.json
```

若要做 pooled pretrain，也可以把目标被试 calibration 一起加入源域，然后再个体化微调：

```bash
python transfer_personalize.py \
  --records-dir /mnt/dataset1/xkp/oi-mi/records_storage \
  --source-subject S001 \
  --source-subject S002 \
  --target-subject S002 \
  --test-chunk /mnt/dataset1/xkp/oi-mi/records_storage/S002/test_mode/chunks/chunk_000000.npz \
  --model eegnet \
  --pretrain-epochs 35 \
  --finetune-epochs 15 \
  --learning-rate 0.001 \
  --finetune-learning-rate 0.0005 \
  --threshold 0.5 \
  --smooth 1 --smooth 3 --smooth 5
```

输出中需要重点看：

- `scratch`
  只用目标被试 calibration 从头训练。
- `transfer_full`
  加载源域预训练权重后，全模型微调。
- `transfer_head`
  加载源域预训练权重后，只微调分类头或最后一层。
- `argmax_acc`
  不做置信度抑制时的逐窗准确率。
- `coverage / valid_acc`
  使用阈值后保留下来的窗比例和保留窗准确率。

## 使用注意事项

1. `test_mode` 与 `calibration` 不是完全同分布。
   不要直接把 calibration 精度等同于在线测试精度。

2. 低置信度窗可能被抑制。
   在这种情况下，`labels_pred = -1` 表示“无有效输出”，不是某个真实类别。

3. `idle` 类通常最难。
   它容易和前一类残留状态混淆，尤其在长 block 或固定顺序 cue 中。

4. 若要比较不同被试，先确认：
   - 设备一致
   - 通道数一致
   - 采样率一致
   - `window_sec / step_sec` 一致
   - 标签映射一致

## 最小读取示例

读取 calibration 训练窗：

```python
import numpy as np

payload = np.load("training_windows_main.npz")
X = payload["processed_windows"]
y = payload["labels"]
```

读取 test_mode 原始窗：

```python
import numpy as np

payload = np.load("chunk_000000.npz")
X = payload["eeg_windows"]
y_true = payload["labels_true"]
y_pred = payload["labels_pred"]
conf = payload["confidences"]
```

读取 calibration metadata：

```python
import json

with open("metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(metadata["label_map"])
print(metadata["trial_timing"])
print(metadata["control_window_range_sec"])
```
