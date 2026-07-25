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

`oi-mi-runtime.json` 声明 `continuous-scene-v2` 协议、`scene_ack`、`scene_failure_event` 等必要能力，并校验播放器、Unity 引擎和实际承载小车代码的 `ARPong.Runtime.dll`。`scene_ack` 只在 Unity 主线程真正应用双障碍布局后返回；碰撞则通过 `scene_failure_event` 通知 Python 立即推进下一 Scene。GUI 启动小车前会强制验证，旧构建、缺文件或混装版本都会停止运行。

当前验证包：

```text
build_id: 2026-07-24-continuous-scene-5s-desktop
zip SHA-256: FE30B0068603EE39997E2A181B2304073544CF51A38F39B3F2BA01EE2752F70A
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
.\.venv\Scripts\python.exe tools\package_unity_build.py --build-id 2026-07-24-continuous-scene-5s-desktop
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
- `SCENE_LEFT/RIGHT/IDLE` 决定空路和障碍物布局，Unity 必须返回 ACK。
- `LEFT/RIGHT/STOP` 是车辆控制命令；模型每个解码步持续输出，协议内部不会插入隐藏停止阶段。
- 关闭 Unity 或场景 ACK 失败时，标签和在线更新会停止，避免静默记录错误真值。
