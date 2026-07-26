# Unity 小车运行包

现场实验只使用预先打包好的 Windows Player，不会调用 Unity 源码仓库，也不要求安装 Unity Editor。运行包安装在 Git 忽略的本地目录：

```text
unity相关/ARPrototype3D-windows-x64/ARPrototype3D.exe
```

## 新电脑安装

After cloning or pulling the repository, run:

```powershell
py -3.12 setup_local.py
.\.venv\Scripts\python.exe cli.py gui
```

`setup_local.py` 会创建 `.venv`、安装依赖、下载 Release 里的 Unity zip、原子替换本地运行包并执行环境检查。

## 运行包完整性

运行包必须包含：

```text
ARPrototype3D.exe
ARPrototype3D_Data/
ARPrototype3D_Data/Managed/ARPong.Runtime.dll
UnityPlayer.dll
UnityCrashHandler64.exe
MonoBleedingEdge/
oi-mi-runtime.json
```

`oi-mi-runtime.json` 声明 `continuous-scene-v4-dynamic-label` 协议，以及 `lane_state_ack`、`relative_action_truth`、`dynamic_action_truth`、`lane_settled_event`、`scene_ack`、`scene_failure_event` 等必要能力，并校验播放器、Unity 引擎和实际承载小车代码的 `ARPong.Runtime.dll`。Python 在每个 Scene 前查询实际车道；`scene_ack` 只有在 Unity 主线程按相对动作应用双障碍布局后，才连同起始车道、空车道和实际标签一起返回。Unity 在车辆真正完成换道后发送 `LANE_SETTLED`，Python 据此切换动态动作真值，跨越切换时刻的窗口不训练。碰撞只记录失败，到固定边界才推进下一 Scene。GUI 启动小车前会强制验证，旧构建、缺文件或混装版本都会停止运行。

当前验证包：

```text
build_id: 2026-07-26-dynamic-label-scene-v4-desktop
zip SHA-256: 9A3A899719103D2BF7E70C0D80685C74FE86E5B330E0D997860AD3A7B85AF9F1
```

重新安装当前 Release：

```powershell
.\.venv\Scripts\python.exe tools\download_unity_build.py --force
```

从本地 zip 安装：

```powershell
.\.venv\Scripts\python.exe tools\download_unity_build.py --force --from-local-zip C:\path\ARPrototype3D-windows-x64.zip
```

## 打包 Release

将已经验证的小车运行目录打包：

```powershell
.\.venv\Scripts\python.exe tools\package_unity_build.py --build-id 2026-07-26-dynamic-label-scene-v4-desktop
```

输出文件固定为：

```text
ARPrototype3D-windows-x64.zip
```

把它作为 GitHub Release asset 上传，名称不要修改。新电脑默认从以下地址下载：

```text
https://github.com/Omni-Intel/oi-mi/releases/latest/download/ARPrototype3D-windows-x64.zip
```

## 运行协议

- GUI 自动启动窗口模式 Player，并等待 `127.0.0.1:5005`。
- Player 直接进入小车的 Fixed Speed 模式，不初始化 MRTK/手势运行时，也不等待菜单选择命令。
- `SCENE_STATE` 返回小车实际车道；`SCENE_LEFT/RIGHT/IDLE` 表示相对当前车道的单步动作，Unity 必须返回含 `start_lane/safe_lane/applied_label` 的 ACK。
- `LANE_SETTLED` 只在小车实际完成换道后发送；Python 根据固定 `safe_lane` 动态生成 LEFT/RIGHT/IDLE 真值。
- `LEFT/RIGHT/STOP` 是车辆控制命令；模型每个解码步持续输出，协议内部不会插入隐藏停止阶段。
- 关闭 Unity 或场景 ACK 失败时，标签和在线更新会停止，避免静默记录错误真值。
