# 安装与环境

`oi-mi` 当前按 Python 3.12.x 维护和验证。不要复用其他操作系统创建的虚拟环境，也不要把虚拟环境提交到仓库。

Python 虚拟环境是平台相关产物：

- Windows 使用 `.venv/Scripts/python.exe`
- macOS / Linux 使用 `.venv/bin/python`
- `torch`、`bc-ecap-sdk`、`brainflow`、`pyedflib` 等二进制依赖会按平台安装不同的原生文件

仓库里应该只保存源码和依赖声明。每台机器都应基于同一份依赖配置重新创建自己的本地虚拟环境。

## 环境自查

安装依赖后运行：

```bash
python tools/check_environment.py
```

这个脚本只使用 Python 标准库，会检查 Python 版本、是否启用虚拟环境，以及主要运行依赖是否可被发现。

## Windows Git Bash

先安装 Python 3.12，然后执行：

```bash
cd /e/Omni/oi-mi
py -3.12 -m venv .venv
source .venv/Scripts/activate
python -m pip install -U pip setuptools wheel
pip install -e .
python tools/check_environment.py
```

如果 `py -3.12` 不可用，先查看本机解释器：

```bash
py -0p
```

## macOS / Linux

先安装 Python 3.12，然后执行：

```bash
cd /path/to/oi-mi
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -e .
python tools/check_environment.py
```

## 依赖策略

`pyproject.toml` 是项目安装依赖的源头。当前将 Python 限制为 `>=3.12,<3.13`，因为脑电和科学计算相关依赖通常依赖平台 wheel，新 Python 版本刚发布时可能还没有完整支持。

如果实验室或部署机器需要完全复现环境，可以在确认环境可用后生成平台锁定文件，例如：

```bash
python -m pip freeze > requirements-win-py312.lock.txt
```

Windows 和 macOS/Linux 如需精确复现，建议分别维护独立 lock 文件。不要提交 `.venv`、`.venv-win` 或其他虚拟环境目录。

## 硬件说明

- BrainCo 依赖 `bc-ecap-sdk`，并需要设备发现或网络地址配置正常
- Neuracle 依赖 JellyFish/Neuracle 转发服务，并需要 `config.yaml` 中 host/port 正确
- trigger box 需要在 `config.yaml` 中配置可访问的串口
