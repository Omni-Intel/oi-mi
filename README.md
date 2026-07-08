# oi-mi

`oi-mi` 是一个面向 Motor Imagery 的生产级 Python CLI 工程骨架，目标是支持真实 EEG 采集、个体校准与在线解码。

数据集使用说明见 [DATASET_GUIDE.md](./DATASET_GUIDE.md)。

- `oi-mi list-models`
- `oi-mi list-devices`
- `oi-mi calibrate --subject S001 --new|--old`
- `oi-mi run --subject S001`
- `oi-mi run --subject S001 --test-mode --test-duration 600`

当前真实设备侧支持两套链路：

- `neuracle`：基于 `collect` / Neuracle / JellyFish 转发
- `brainco`：基于 `bc-ecap-sdk` 的 BrainCo 32ch EEG Cap

## 目录结构

```text
oi-mi/
├── pyproject.toml
├── config.yaml
├── cli.py
├── acquisition/
├── models/
├── adaptation/
├── decoder/
├── utils/
├── models_storage/
├── README.md
├── LICENSE
└── tests/
```

## 安装

要求：

- Python 3.12.x（当前不声明支持 Python 3.13）
- 若使用 Neuracle 脑电帽JellyFish/Neuracle 数据转发服务已开启
- 如果使用硬件 trigger box，需要串口可访问

完整跨平台安装说明见 [INSTALL.md](./INSTALL.md)。虚拟环境不要跨系统复用，也不要提交到仓库。

Windows Git Bash：

```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
python -m pip install -U pip setuptools wheel
pip install -e .
python tools/check_environment.py
```

macOS / Linux：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -e .
python tools/check_environment.py
```

安装后可直接使用：

```bash
oi-mi list-models
oi-mi list-devices
```

## 命令示例

新被试校准：

```bash
oi-mi calibrate --subject S001 --new --duration 1800 --model riemann-mdm
```

老被试快速微调：

```bash
oi-mi calibrate --subject S001 --old --duration 300 --model eegnet
```

实时运行：

```bash
oi-mi run --subject S001 --model riemann-mdm --device neuracle
```

BrainCo 实时运行：

```bash
oi-mi run --subject S001 --model riemann-mdm --device brainco
```

测试模式（有 cue、保存 EEG/标签、输出准确率）：

```bash
oi-mi run --subject S001 --test-mode --test-duration 600
```

预下载数据集：

```bash
python download_datasets.py --dataset BNCI2014_001 --subject 1 --data-dir ./data/moabb
```

## 实时链路

- 采集：`acquisition`
- 预处理：`utils/preprocessing.py`
- 个体适配：`adaptation/calibrator.py`
- 在线解码：`decoder/real_time_decoder.py`
- 命令输出：LSL command stream

## 各模式 EEG 保存逻辑


| 模式                             | 是否保存 EEG | 保存根目录                                                           | 关键文件                                   | 内容说明                                                                    |
| ------------------------------ | -------- | --------------------------------------------------------------- | -------------------------------------- | ----------------------------------------------------------------------- |
| `calibrate --new/--old`（校准）    | 是        | `records_storage/<subject_id>/calibration/<session_timestamp>/` | `continuous_eeg.npy`                   | 会话期间连续增量 EEG 拼接后的原始连续数据（float32）。                                       |
| `calibrate --new/--old`（校准）    | 是        | 同上                                                              | `events.json` / `metadata.json`        | 事件时间轴、trial 元信息、协议参数等对齐信息。                                              |
| `calibrate --new/--old`（校准）    | 是        | 同上                                                              | `training_windows_main.npz`            | 切窗训练集：`raw_windows`（原始窗）+ `processed_windows`（预处理后）+ `labels`。          |
| `calibrate --new/--old`（校准，可选） | 是        | 同上                                                              | `training_windows_aux_1p5s.npz`        | 若 `protocol.export_window_sec` 非空，会额外导出辅助窗长版本。                          |
| `run --test-mode`（测试模式）        | 是        | `records_storage/<subject_id>/test_mode/`（内部按 chunk 写）          | `manifest.json` + `chunks/chunk_*.npz` | 每个推理步保存原始 `window`、`y_true`、`y_pred`、`confidence`；`manifest` 汇总窗口数和准确率。 |
| `run`（实时解码）+ `--record`        | 是        | `records_storage/<subject_id>/realtime/<session_timestamp>/`    | `manifest.json` + `chunks/chunk_*.npz` | 每个推理步保存原始 `window`、预测类别和置信度（`y_true=-1`）。                               |
| `run`（实时解码）不带 `--record`       | 否        | 不落盘                                                             | 无                                      | 只在线推理和输出控制命令，不写 EEG 文件。                                                 |


补充说明：

- 测试模式与实时模式的 `chunk_*.npz` 由后台 `StreamWriter` 异步分块写盘，避免阻塞解码循环。
- 在线推理会做 `filter_and_transform`，但落盘的 `window` 为采集器返回的原始窗（仅转为 `float32`）。

默认配置：

- 采样率：250 Hz
- 窗长：4.0 s
- 步长：0.5 s
- 三分类：左手 / 右手 / 静息

## 真实采集说明

当前真机采集支持两种后端：

- `acquisition/neuracle_acquirer.py` 中的 `NeuracleAcquirer` 复用了 `DataServerThread`
- `acquisition/brainco_acquirer.py` 中的 `BrainCoAcquirer` 复用了 `bc_ecap_sdk`
- `TriggerBoxMarkerBackend` 复用了 `TriggerBox`

因此请确保：

- JellyFish 数据转发端口与 `config.yaml` 一致
- BrainCo SDK 已安装，且设备可通过 mDNS 自动发现，或已手工填写 `brainco_addr` / `brainco_port`
- 若启用 trigger box，`trigger_serial_port` 配置正确

### `collect` 目录现在是做什么的

- `collect/neuracle_api.py`：和本地 JellyFish 数据转发服务建立 TCP 连接，并维护实时环形缓冲区。
- `collect/triggerBox.py`：向外部 trigger box 发送事件码，用于 cue 标记。
- `acquisition/neuracle_acquirer.py`：把 `collect` 的底层能力包装成统一采集接口，供 CLI 训练/推理调用。

### 命令行如何用真实设备链路

先确认 `config.yaml`：

- `device_type: neuracle`
- `device.neuracle_host` / `device.neuracle_port` 与 JellyFish 转发一致
- 若使用 trigger box，填写 `device.trigger_serial_port`

建议先做连通性探测：

```bash
oi-mi probe-device --device neuracle --duration 5
```

新被试校准（有 cue）：

```bash
oi-mi calibrate --subject S001 --new --duration 600 --model eegnet
```

老被试适配（有 cue）：

```bash
oi-mi calibrate --subject S001 --old --duration 600 --model eegnet
```

使用已保存的 calibration 数据重训：

```bash
oi-mi train-from-records --subject S001 --model eegnet
```

只用指定 session 重训：

```bash
oi-mi train-from-records --subject S001 --model eegnet \
  --session 20260413_224710
```

用 test_mode 保存的窗口做离线回放评估：

```bash
oi-mi replay-test-mode --subject S001 --model eegnet
```

如果数据和模型不在仓库默认目录，而是在外部目录 `/mnt/dataset1/xkp/oi-mi`，建议单独准备一个配置文件，例如 `config.dataset1.yaml`：

```yaml
subject_id: S001
model_name: eegnet
device_type: brainco
sfreq: 250
n_classes: 3
window_sec: 2.0
step_sec: 0.5
confidence_threshold: 0.7
mc_dropout_passes: 8
new_subject_epochs: 50
old_subject_epochs: 5
batch_size: 32
learning_rate: 0.001
early_stopping_patience: 40
buffer_sec: 60
storage:
  models_dir: /mnt/dataset1/xkp/oi-mi/models_storage
  records_dir: /mnt/dataset1/xkp/oi-mi/records_storage
```

用 `conda` 环境直接重训指定 calibration session：

```bash
conda run -n uni python cli.py --config config.dataset1.yaml \
  train-from-records --subject S001 --model eegnet --session 20260413_224710
```

用同一份配置做 test_mode 离线回放：

```bash
conda run -n uni python cli.py --config config.dataset1.yaml \
  replay-test-mode --subject S001 --model eegnet \
  --test-dir /mnt/dataset1/xkp/oi-mi/records_storage/S001/test_mode
```

实时运行（无 cue，自动输出）：

```bash
oi-mi run --subject S001 --model eegnet --device neuracle
```

测试模式（有 cue，保存 EEG/标签并输出准确率）：

```bash
oi-mi run --subject S001 --model eegnet --device neuracle --test-mode --test-duration 600
```

### 命令行如何用 BrainCo 32ch

先确认 `config.yaml`：

- `device_type: brainco`
- `sfreq` 为 BrainCo SDK 支持的采样率之一：`250 / 500 / 1000 / 2000`
- 优先使用 `device.brainco_auto_discover: true`
- 若自动发现失败，可手工填写 `device.brainco_addr` / `device.brainco_port`

建议先做连通性探测：

```bash
oi-mi probe-device --device brainco --duration 5
```

实时运行：

```bash
oi-mi run --subject S001 --model eegnet --device brainco
```

测试模式：

```bash
oi-mi run --subject S001 --model eegnet --device brainco --test-mode --test-duration 600
```

## 模型说明

当前可选模型：

- `riemann-mdm`
- `eegnet`
- `deepconvnet`
- `shallowconvnet`
- `s4d`

建议当前优先使用：

- `riemann-mdm` 作为首个稳定基线
- `eegnet` 作为轻量深度学习基线

## 测试

如果你要做离线 MOABB 预训练，建议先运行 `download_datasets.py` 预下载数据。

注意：`train_moabb.py` 对 `BNCI2014_001` 仍会使用数据集原生的 `left/right/feet` 三分类标签。
这只是离线数据集适配，不代表 `oi-mi` 真机校准和在线解码的第三类定义；在线链路统一按 `静息` 处理。

运行基础测试：

```bash
python -m unittest discover -s tests
```

运行静态编译检查：

```bash
python -m compileall .
```

## 打包

可编辑安装：

```bash
pip install -e .
```

可执行文件打包：

```bash
pyinstaller --onefile cli.py --name oi-mi
```

## 当前已知限制

- 尚未实现 Kappa 报告、混淆矩阵和完整 E2E 自动化测试
- 预处理当前采用轻量实时实现，尚未引入 ASR/ICA 的正式在线版本

## 下一步建议

- 把 `collect` 中的设备元数据、通道名、trigger 通道处理继续结构化
- 为 `neuracle` / `brainco` 加自动重连与采集健康检查
- 增加保存训练元数据、Kappa 和混淆矩阵
- 增加 `run` 的在线结果统计与会话级报告导出
