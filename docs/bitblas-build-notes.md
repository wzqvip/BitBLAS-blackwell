# BitBLAS 本机构建记录（Jetson Thor，CUDA 13.0，Ubuntu 24.04）

本文记录在本机从源码成功编译运行 BitBLAS 的关键修改与步骤，便于后续复现。

## 关键修改

- `setup.py`：将 LLVM 版本从 `10.0.1` 升级到 `18.1.8`，并使用 `tar --no-same-owner` 解压，避免旧版 `libtinfo5` 依赖和解压权限问题。
  - 相关文件：`setup.py`

## 环境准备

```bash
# 创建并使用 Python 3.10 的隔离环境
conda create -y -n bitblas python=3.10
conda run -n bitblas python -m pip install --upgrade pip setuptools wheel
```

## 构建步骤

```bash
cd ~/BitBLAS

# 清理旧的第三方构建（如需）
rm -rf 3rdparty/clang+llvm-* 3rdparty/tvm/build 3rdparty/tilelang/build

# 在环境内安装（会自动下载 LLVM 18.1.8 并构建 TVM、tilelang）
conda run -n bitblas python -m pip install .
```

### 成功验证

```bash
conda run -n bitblas python -c "import bitblas; print('bitblas version:', bitblas.__version__)"
# 预期输出：bitblas version: 0.1.0
```

> 提示：首次导入会看到 tilelang 编译 Cython 适配器的日志，以及 Composable Kernel 未找到的警告（可忽略）。

## 使用建议

- 运行前激活环境：`conda activate bitblas311`（如需 3.10，可用 `bitblas`，但需自行编译匹配版本的 CUDA Torch）
- 保留 `~/.cache/tilelang` 可复用已编译的 jit 适配器，减少首次导入开销。
- 如果内置 target 数据库缺少 Blackwell (sm_110)，会自动回退到 Ada 调度；可选设置 `export TVM_TARGET="cuda -arch=sm_90"` 控制调度。

## 运行示例 (Thor / sm_110)

```bash
# 1) 使用带 CUDA 的 PyTorch 3.11 wheel（已预编译）
conda create -y -n bitblas311 python=3.11
conda run -n bitblas311 python -m pip install ~/jetson-pytorch-builder/wheels/py311/torch-*.whl

# 2) 安装 BitBLAS 源码（可编辑）
cd ~/BitBLAS
conda run -n bitblas311 python -m pip install -e .

# 3) （可选）指定调度目标，避免未知架构告警
export TVM_TARGET="cuda -arch=sm_90"

# 4) 运行示例
conda run -n bitblas311 python testing/1.py
```
