# Unity 小车运行包

现场实验只使用独立小车 Unity 工程生成的 Windows Player，不调用完整版 Unity 的源码、DLL、下载脚本或打包脚本。运行包位于：

```text
../oi-car-unity-src/Car_game/Builds/Windows/ARPrototype3D.exe
```

## Python 环境

After cloning or pulling the repository, run:

```powershell
py -3.12 setup_local.py
.\.venv\Scripts\python.exe cli.py gui
```

`setup_local.py` 只创建 `.venv`、安装 Python 依赖、下载 CBraMod 权重并执行环境检查，不处理 Unity 完整版运行包。

## 运行包完整性

运行包必须包含：

```text
ARPrototype3D.exe
ARPrototype3D_Data/
ARPrototype3D_Data/Managed/ARPrototype3D.Runtime.dll
UnityPlayer.dll
UnityCrashHandler64.exe
MonoBleedingEdge/
oi-mi-runtime.json
```

`oi-mi-runtime.json` 声明 `continuous-scene-v5-centered-single-decision` 协议，并校验播放器、Unity 引擎和实际承载小车代码的 `ARPrototype3D.Runtime.dll`。GUI 启动小车前会强制验证，旧构建、缺文件或混装版本都会停止运行。

## 构建独立小车

```powershell
Set-Location D:\Projects\ncc\oi-car-unity-src\Car_game
$env:CAR_WINDOWS_OUTPUT = "$PWD\Builds\Windows\ARPrototype3D.exe"
& "D:\UnityHub\Editors\2022.3.60f1c1\Editor\Unity.exe" `
  -batchmode -nographics -projectPath "$PWD" `
  -executeMethod ARPrototype3D.Editor.BuildCommand.BuildWindows64 -quit
```

独立构建命令会同时生成 `oi-mi-runtime.json`，写入协议能力和关键文件 SHA-256。

## 运行协议

- GUI 自动启动窗口模式 Player，并等待 `127.0.0.1:5005`。
- Player 直接进入小车 Fixed Speed 模式，不初始化完整版菜单、MRTK 或手势运行时。
- 每个 Scene 都把小车重置到中间车道；`SCENE_LEFT/RIGHT/IDLE` 同时确定障碍布局和唯一主训练标签。
- `SCENE_STATE` 返回当前 Scene 和实际车道；布局命令必须返回协议版本、Scene 编号、起始车道和安全车道。
- `LEFT/RIGHT/STOP` 是车辆控制命令；主决策窗完成前横向控制被门控，完成后立即释放。
- 等待 Scene 或同步失败时 Unity 不生成随机车辆；只有完整 Scene 命令成功后才同时生成两辆障碍车。
- 当前 Scene 未收到完整 ACK 时 Python 保持该 Scene 编号，不允许本地计时器跳过并建立后续 Scene。
- 关闭 Unity 或场景 ACK 失败时，标签和在线更新会停止，避免静默记录错误真值。
