# BitBLAS aarch64 适配与 Blackwell 支持记录

记录本次在 aarch64 (Jetson Thor / CUDA 13.0 / Ubuntu 24.04) 上修复 TVM 构建问题并让 Blackwell (sm_110) 运行 BitBLAS 的关键改动和操作。

## 主要改动

1) **修复 aarch64 上 TVM 的 LLVM 依赖问题**  
   - 文件：`setup.py`  
   - 变更：将 `LLVM_VERSION` 从 `10.0.1` 升级到 `18.1.8`，并改用 `tar --no-same-owner` 解压，避免旧版 `libtinfo5` 依赖以及解压权限导致的 `Permission denied`。  
   - 影响：TVM 构建不再因为 `/lib/aarch64-linux-gnu/libtinfo.so.5` 缺失或解压权限报错而中断。

2) **显式支持 Blackwell 识别与调度**  
   - 文件：`bitblas/base/arch/cuda.py`，`bitblas/base/arch/__init__.py`  
   - 新增 `is_blackwell_arch`，并将 TensorCore 支持判定中把 Blackwell 视为 Hopper/Ada 支持的精度集合。  
   - `Matmul`/`MatmulDequantize` 调度逻辑中，Blackwell 不再抛 “Unsupported architecture”，直接走 Ampere/Ada 路径（可再调优）。

3) **尊重 TVM_TARGET 并自动选择 sm_110**  
   - 文件：`bitblas/utils/target_detector.py`  
   - 现在优先读取环境变量 `TVM_TARGET`；若未设置，会用 `nvidia-smi --query-gpu=compute_cap` 自动构造 arch（>=110 -> sm_110，>=90 -> sm_90，>=89 -> sm_89，>=80 -> sm_80）。  
   - 目的：在 Blackwell 上不再提示 “TVM target not found”，且可显式锁定 `TVM_TARGET="cuda -arch=sm_110"`。

## 操作步骤（本机已验证）

1) 环境与 Torch（PyTorch 2.4.0 Jetson 轮子）  
   ```bash
   conda create -y -n bitblas311 python=3.11
   conda run -n bitblas311 python -m pip install ~/jetson-pytorch-builder/wheels/py311/torch-*.whl
   ```

2) 安装 BitBLAS（含 TVM/tilelang 构建）  
   ```bash
   cd ~/BitBLAS
   conda run -n bitblas311 python -m pip install -e .
   ```

3) Blackwell 运行（可选先调优）  
   ```bash
   export TVM_TARGET="cuda -arch=sm_110"
   # 如需调优生成专用算子
   conda run -n bitblas311 python - <<'PY'
   import os, bitblas
   cfg = bitblas.MatmulConfig(M=1, N=2048, K=1024,
                              A_dtype="float16", W_dtype="int4",
                              accum_dtype="float16", out_dtype="float16",
                              layout="nt", with_bias=False)
   op = bitblas.Matmul(cfg, target=os.environ.get("TVM_TARGET", "cuda -arch=sm_110"),
                       enable_tuning=True, from_database=False)
   op.hardware_aware_finetune(topk=20, parallel_build=True)
   PY

   # 运行示例
   conda run -n bitblas311 TVM_TARGET="cuda -arch=sm_110" python testing/1.py
   ```

4) 运行现状  
   - 示例正常执行，Ref/BitBLAS 输出一致。  
   - 仍会有 cutlass `vector_types` 弃用 warning，可忽略。  
   - 如果不设 `TVM_TARGET`，将基于 compute_cap 自动落在 sm_110。

## 变更文件列表
- `setup.py`
- `bitblas/base/arch/cuda.py`
- `bitblas/base/arch/__init__.py`
- `bitblas/utils/target_detector.py`
