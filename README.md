# oi-mi

## 新设备快速开始（Windows）

项目要求 Python 3.12。每台新电脑都必须重新创建 `.venv`，不要复制其他电脑的虚拟环境。PowerShell 或 CMD 在仓库目录执行：

```powershell
git pull origin main
py -3.12 setup_local.py
.\.venv\Scripts\python.exe tools\check_environment.py
.\.venv\Scripts\python.exe cli.py gui
```

如果还没有 Python 3.12，先执行：

```powershell
winget install --id Python.Python.3.12 --exact --source winget --scope user
```

安装完成后关闭并重新打开终端，再运行上面的命令。不要在 PowerShell/CMD 中运行 `source`；也不要直接运行全局的 `streamlit run gui.py`，否则容易调用到缺少 `yaml`、PyTorch 等依赖的系统 Python。直接使用 `.venv\Scripts\python.exe` 不需要激活虚拟环境。

仓库已经包含经过协议校验的 Unity Windows 运行包。`setup_local.py` 会创建 `.venv`、安装项目及 PyYAML 等全部依赖，并校验随仓库提供的运行包；正常情况下不会再次下载：

```text
unity相关/ARPrototype3D-windows-x64/ARPrototype3D.exe
```

### 采集电脑强制更新（解决 `config.yaml` 冲突）

不需要删除或重装 `.venv`。先停止当前实验并关闭 GUI，然后在采集电脑的项目目录执行：

```powershell
cd D:\oi-mi
Copy-Item .\config.yaml .\config.local.backup.yaml -Force
git fetch origin
git reset --hard origin/main
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe tools\check_environment.py
.\.venv\Scripts\python.exe cli.py gui
```

`git reset --hard origin/main` 会丢弃采集电脑上所有尚未提交的项目代码和已跟踪配置修改，包括产生冲突的 `config.yaml`，并把 Unity 运行包同步到仓库中的正式版本；不会删除被 `.gitignore` 排除的 `.venv`、`models_storage` 和 `records_storage`。备份文件 `config.local.backup.yaml` 仅用于找回被试编号、设备地址等现场参数，不要把旧配置整份覆盖回来；正式 Neuracle 配置必须保留 `sfreq: 200` 和 `device.neuracle_source_sfreq: 250`。

如果采集电脑的仓库不在 `D:\oi-mi`，只需要把第一行 `cd` 改成实际项目路径。

现场电脑只运行仓库内的这个打包结果，不调用其他 Unity 源码仓库，也不需要安装 Unity Editor。运行包必须包含 `oi-mi-runtime.json`；启动前会校验协议版本、必要功能和关键文件 SHA-256，旧版或混装的 Unity 会直接报错，不会继续产生不可信标签。`tools/download_unity_build.py --force` 仅作为运行包损坏时的备用恢复方式，日常更新直接同步 Git 仓库即可。

点击网页左侧的“实时解码”或启动 dummy 测试时，如果 Unity 没有打开，程序会自动以窗口模式启动这个 exe，并等待 `127.0.0.1:5005` 可连接后再继续；随后直接进入小车场景。Windows 实验包会自动启用唯一支持的 Fixed Speed 控制模式，不依赖加载完成时刻不确定的菜单选择命令。实时解码运行期间关闭 Unity 窗口会让小车 TCP 连接断开，网页端实时解码也会停止。

启动成功后浏览器进入 `http://localhost:8501`。实验期间保持启动终端开启，不要重复启动第二个 GUI。

### 新设备第一次正式实验

1. 在“设置”页确认被试 ID、`shallowconvnet`、真实设备类型和设备地址，然后保存。
2. 在“连通检测”页确认 EEG 数据可读取；需要时再做“阻抗检查”。
3. 进入“校准”并点击“正式实验”；每次实验都会从头校准，不复用旧模型。
4. 校准采集结束后不要返回、刷新或关闭页面；等待离线训练完成并明确显示模型保存路径。
5. 模型和 CRM 应分别出现在 `models_storage/<被试>/<设备>/shallowconvnet.pt` 与 `shallowconvnet.pt.neuroonline.pt`。
6. 先进入“测试模式”验证模型和小车链路，再进入“实时解码”开始连续 NeuroOnline 实验。

真实实验前应确认 [config.yaml](./config.yaml) 中：

```yaml
hardware_dummy_mode: false
sfreq: 200
device:
  neuracle_source_sfreq: 250
  neuracle_transport_delay_sec: 0.0
  brainco_source_sfreq: 250
online_adaptation:
  enabled: true
  strategy: neuroonline
  simulation:
    enabled: false
  cued_labels:
    enabled: true
    scene_duration_sec: 5.0
    boundary_guard_sec: 0.5
```

如果 GUI 报 `No module named 'yaml'`，说明启动时用了系统 Python。回到仓库目录重新执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe cli.py gui
```

`oi-mi` 是一个面向 Motor Imagery 的生产级 Python CLI 工程骨架，目标是支持真实 EEG 采集、个体校准与在线解码。

数据集使用说明见 [DATASET_GUIDE.md](./DATASET_GUIDE.md)。

- `oi-mi list-models`
- `oi-mi list-devices`
- `oi-mi calibrate --subject S001`
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

Windows PowerShell / CMD（推荐，不需要激活环境）：

```powershell
py -3.12 setup_local.py
.\.venv\Scripts\python.exe cli.py gui
```

Windows Git Bash：

```bash
python setup_local.py
.venv/Scripts/python.exe cli.py gui
```

macOS / Linux：

```bash
python3.12 setup_local.py
.venv/bin/python cli.py gui
```

命令行操作也建议显式使用虚拟环境解释器，例如：

```powershell
.\.venv\Scripts\python.exe cli.py list-models
.\.venv\Scripts\python.exe cli.py list-devices
```

## 命令示例

统一从头校准：

```bash
oi-mi calibrate --subject S001 --model shallowconvnet
```

默认正式采集严格为约 12 分钟，不含可选教程、练习和后续训练时间：

- 60 秒睁眼 baseline
- 4 个 block，每个 block 包含 `LEFT/RIGHT/IDLE` 各 5 个 trial，共 60 trial
- 每 trial 为 `2s fixation + 1s cue + 5s control + 2s ITI`
- 3 次 block 间休息，每次 20 秒

每个 trial 通常产生 5 个 2 秒窗口，因此质量筛选前共有 300 个窗口、每类
100 个。训练/验证按 `trial_ids` 分组，同一 trial 的重叠窗口不会跨集合。后续
离线训练时间不计入这 12 分钟；训练上限为 50 epoch，验证 kappa 连续 6 个
epoch 不提升时提前停止。

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

### NeuroOnline 算法复现与小车在线应用

`config.yaml` 中将 `online_adaptation.strategy` 设为 `neuroonline` 后，实时解码采用严格的先预测、后更新流程。当前小车实验配置在累计 64 个有标签窗口后开始第一次更新，之后每新增 64 个有标签窗口更新一次；训练数据逐步增长到 320 个，此后始终使用最近 320 个窗口。每个样本在进入缓冲区时生成固定的时间遮挡和频率遮挡视图，更新目标为三个视图的分类损失加两项表示一致性损失，并联合更新 backbone、CRM 和分类头。

该模式仅支持 PyTorch 模型，不支持 `riemann-mdm`。CRM 以恒等门控初始化；更新后的 backbone 保存到原模型文件，CRM 状态保存到同目录的 `<model>.neuroonline.pt` sidecar 文件。将策略改回 `periodic_head` 可继续使用原有的十分钟候选分类头更新流程。

实时解码页面配套显示 NeuroOnline 遥测：累计 Scene、累计有标签窗口、距离下一次 64 窗口触发的进度、缓冲区类别覆盖、原始 argmax prequential accuracy/fixed-three-class balanced accuracy、实际控制覆盖率、选择性准确率、逐类准确率、累计混淆矩阵，以及每次更新的总损失、分类损失、一致性损失、CRM gate 和耗时曲线。所有在线性能只使用“先预测、后更新”时产生的预测，不会用更新后的模型回算历史样本。三类尚未全部出现时，未出现类别召回率固定按 0 计，避免早期 balanced accuracy 虚高。

在线更新在隔离的候选模型上后台执行，实时推理继续使用当前模型；候选训练完成后，在模型锁内一次性替换引用并递增 `model_revision`。每次成功更新都会保存主模型和 CRM sidecar。开启实时记录后，最终适配状态和完整更新历史写入 `manifest.json`，每个 chunk 额外保存原始 argmax 预测、阈值处理后的预测、模型 revision 和标签事件 ID。

### 论文级实验记录

正式 NeuroOnline 运行强制开启记录。GUI 必须勾选“保存实时脑波数据至本地记录”，CLI 会自动开启；真实 Neuracle 模式还必须填写 `storage.native_recording_id`，其值应与博瑞康采集软件保存的 BDF/NDF 文件名或会话编号一致。项目窗口记录不能替代放大器原生连续文件，归档时两者必须一起保存。

一次实时运行形成以下不可变实验包：

- `manifest.json`：schema/run ID、完整配置快照及 SHA-256、Git commit/dirty 状态、Python/依赖版本、初始模型哈希、预处理参数、最终统计、文件校验和及丢弃记录数。manifest 采用临时文件加原子替换更新。
- `chunks/chunk_*.npz`：逐窗原始 EEG、源时钟映射后的单调时钟与 Unix 起止时间、三类完整概率、原始 argmax、置信度阈值后的实际控制预测、置信度/不确定度、Scene ID/标签、模型 revision、质量判定/原因/坏导信息。chunk 先完整写入临时文件再原子替换。
- `events.jsonl`：session、Unity Scene ACK、固定边界结束、碰撞失败、更新开始、模型原子切换、更新完成和模型快照事件；每项同时保存 monotonic、Unix 和 UTC 时间。JSONL 即使异常中断也可恢复到最后一条完整事件。
- `model_revisions/revision_XXXX.pt` 及 CRM sidecar：revision 0 和每次在线切换后的实际模型，manifest 保存各版本哈希，使任一逐窗预测都能还原到对应权重。

停止运行后，程序从落盘 chunk 和事件日志独立重算并写入 `scientific_metrics`：

- `raw_window`：质量合格、有真值窗口的 test-then-train argmax Accuracy、固定三类 Balanced Accuracy、逐类召回和混淆矩阵；
- `operational_window`：置信度阈值后的覆盖率、拒识数、选择性准确率及把拒识视为错误的控制准确率；
- `scene_classification`：同一 Scene 内 prequential 概率取均值后再 argmax 的 Scene 级分类结果；
- `car_task`：以 `scene_end` 的碰撞/无碰撞结果计算避障成功率及 Wilson 95% 区间。

2 秒窗以 0.5 秒步长滑动，窗口之间并不独立。因此窗口级指标用于描述在线轨迹，论文显著性检验应以 Scene、block、session 或 subject 为统计单位，不能直接对所有重叠窗口使用独立二项假设。

实验结束、复制或归档后运行完整性检查：

```powershell
.venv\Scripts\python.exe tools\verify_experiment_bundle.py records_storage\S001\realtime\<session_timestamp>
```

只有输出 `"ok": true`、`integrity.status=complete`、`dropped_records=0` 且博瑞康原始文件编号对应时，才应将该会话纳入论文统计。

正式小车实验启用 `online_adaptation.cued_labels` 后采用连续统一场景协议。每个 scene epoch 只有一个真值：`LEFT` 表示左路为空、`RIGHT` 表示右路为空、`IDLE` 表示小车当前道路为空；Unity 会在同一帧、同一前向距离给另外两条路各布置一辆障碍车，并在小车 HUD 显示空路。碰撞只会炸掉被撞的障碍车，另一条路的障碍保持原位；Unity 上报 `SCENE_FAILED` 后，Python 只记录本 Scene 失败，不提前切换标签或障碍布局，到固定的 5 秒边界才同步下一 Scene。模型每 `step_sec` 持续预测并控制车辆，不再存在 fixation、cue、ITI 或面向被试的隐藏 `STOP`。正式 Scene 时长固定为 `5.0` 秒，Unity 按相对速度 `(6.0 - 3.2) m/s` 将障碍放在约 `14.0 m` 前方：走错车道会因车身碰撞半径在约 4.7 秒判定失败，但剩余约 0.3 秒仍属于同一 Scene；走对空路则在第 5 秒通过障碍横截面并判定成功。在线训练与校准采用相同的有效控制区间，只接受 Scene 内 `0.5-4.5` 秒范围中完整的 EEG 窗口。Python 收到 Unity 对 `SCENE_LEFT/RIGHT/IDLE` 的 ACK 后，才以 ACK 接收时刻作为该 Scene 的真实起点；跨越 ACK 或 5 秒切换边界的窗口均标为无标签。

Neuracle/JellyFish 原始数据以 `250 Hz` 转发。校准连续数据保留 250 Hz 源时间轴，切出完整源窗口后做带抗混叠滤波的 `250→200 Hz` 降采样；实时解码同样先取得 500 点的 2 秒源窗口，再降为模型使用的 400 点。后续预处理、模型输入和 NeuroOnline 更新统一使用 `200 Hz`。实时窗口不再把 `get_chunk()` 返回时刻当作脑电终点，而是利用 JellyFish 数据包的源时间戳，将样本时间映射到 Python 的单调时钟后再与 Unity Scene 对齐。`device.neuracle_transport_delay_sec` 用于补偿经 Trigger/回环实验测得的固定采集链路延迟；未测量前保持 `0.0`，系统仍会利用源时间戳消除排队和轮询抖动。

模型输入预处理参考 CBraMod 的运动想象数据管线，统一采用微伏输入、坏导稳健修复、Common Average Reference、`0.3-40 Hz` 五阶 Butterworth SOS 零相位滤波和 `[-150, 150] uV` 数值保护。校准、离线重建和实时解码调用同一个实现；预处理不会做逐窗 z-score，避免抹掉运动想象的绝对幅值变化。非有限值、坏导比例过高、峰值超过 `300 uV` 或超幅占比超过 1% 的窗口不会进入校准训练或 NeuroOnline 更新，但实时控制仍会产生预测，并在记录文件中保存质量标志。

200 Hz、2秒窗口对应模型输入长度为400点。旧的250 Hz/500点模型权重和 CRM sidecar
不能继续使用；切换采样率后必须重新完成正式校准。

连续模式没有“大轮完成”或固定 trial 停止点，GUI 显示累计 Scene、当前空路和 Unity 同步状态，NeuroOnline 只按累计有效窗口触发。`balance_pool_per_class: 32` 仅用于内部生成可复现的类别平衡场景池，不会清空缓存、重置计数或停止实验。运行持续到操作员切换页面、关闭 Unity 或停止 GUI。真实设备实验必须保持 `online_adaptation.simulation.enabled: false`。

## 各模式 EEG 保存逻辑


| 模式                             | 是否保存 EEG | 保存根目录                                                           | 关键文件                                   | 内容说明                                                                    |
| ------------------------------ | -------- | --------------------------------------------------------------- | -------------------------------------- | ----------------------------------------------------------------------- |
| `calibrate`（统一从头校准）    | 是        | `records_storage/<subject_id>/calibration/<session_timestamp>/` | `continuous_eeg.npy` / `continuous_sample_timestamps.npy` | 会话期间连续原始 EEG 及逐样本源时钟；metadata保存环境、配置、模型和文件哈希。 |
| `calibrate`（统一从头校准）    | 是        | 同上                                                              | `events.json` / `metadata.json`        | 事件时间轴、trial 元信息、协议参数等对齐信息。                                              |
| `calibrate`（统一从头校准）    | 是        | 同上                                                              | `training_windows_main.npz`            | 质量筛选后的训练集、`trial_ids` 和逐窗质量指标；按 trial 分组验证。          |
| `calibrate`（可选辅助窗） | 是        | 同上                                                              | `training_windows_aux_1p5s.npz`        | 若 `protocol.export_window_sec` 非空，会额外导出辅助窗长版本。                          |
| `run --test-mode`（测试模式）        | 是        | `records_storage/<subject_id>/test_mode/`（内部按 chunk 写）          | `manifest.json` + `chunks/chunk_*.npz` | 每个推理步保存原始 `window`、`y_true`、`y_pred`、`confidence`；`manifest` 汇总窗口数和准确率。 |
| `run`（实时解码）+ NeuroOnline        | 是（强制） | `records_storage/<subject_id>/realtime/<session_timestamp>/`    | `manifest.json` + `events.jsonl` + `chunks/` + `model_revisions/` | 保存完整逐窗概率/时间/质量/Scene/模型版本、事件、修订权重、独立重算指标与校验和。 |
| `run`（实时解码）不带 `--record`       | 否        | 不落盘                                                             | 无                                      | 只在线推理和输出控制命令，不写 EEG 文件。                                                 |


补充说明：

- 测试模式与实时模式的 `chunk_*.npz` 由后台 `StreamWriter` 异步分块写盘，避免阻塞解码循环。
- 在线推理会做 `preprocess_eeg_window`，但落盘的 `window` 为采集器返回的原始窗（仅转为 `float32`）。
- 实时 chunk 额外保存 `quality_accepted`、`quality_peak_abs_uv`、`quality_clip_fraction` 和 `quality_bad_channel_fraction`；质量不合格的窗不参与在线更新。

默认配置：

- 采样率：200 Hz
- 窗长：2.0 s
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
- BrainCo 硬件固定以 250 Hz 采集；实时窗口在进入预处理前统一抗混叠降采样到 200 Hz
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

统一从头校准（有 cue）：

```bash
oi-mi calibrate --subject S001 --model shallowconvnet
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
sfreq: 200
n_classes: 3
window_sec: 2.0
step_sec: 0.5
confidence_threshold: 0.7
mc_dropout_passes: 8
calibration_epochs: 50
batch_size: 32
learning_rate: 0.001
early_stopping_patience: 6
buffer_sec: 60
device:
  brainco_source_sfreq: 250
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

### 采集设备范围

Neuracle/JellyFish 与 BrainCo 设备端均按 250 Hz 采集。当前正式实验使用
Neuracle；采集器按完整时间窗做带抗混叠滤波的 `250→200 Hz` 降采样。降采样之后
的预处理、2 秒 400 点窗口、训练、解码和 NeuroOnline 在线更新仍全部按全局
`sfreq: 200` 执行。

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

- 预处理当前采用轻量实时实现，尚未引入 ASR/ICA 的正式在线版本
- Neuracle 转发接口当前没有提供可直接用于端到端对齐的放大器绝对时间戳；在线标签采用本机单调时钟并丢弃跨 Scene 窗口。正式论文数据仍建议用触发盒/原始文件事件通道做一次硬件延迟核验。
- 自动化测试覆盖 Python、Unity 场景协议与运行包黑盒链路，但真实 EEG 质量、被试执行度和现场网络/放大器稳定性仍需实验前人工检查。

## 下一步建议

- 把 `collect` 中的设备元数据、通道名、trigger 通道处理继续结构化
- 为 `neuracle` / `brainco` 加自动重连与采集健康检查
- 将 Unity Scene 事件同步写入放大器事件通道，建立在线与原始数据文件的硬件级时间对齐
- 增加正式会话级结果报告导出
