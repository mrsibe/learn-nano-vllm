# Learn Nano-vLLM

基于 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的学习版本，用来理解 LLM 推理引擎中的调度、KV Cache、Paged Attention 和张量并行。

## 环境

- Linux x86_64、Python 3.12（由 uv 自动管理）
- NVIDIA Ampere / Ada / Hopper GPU，例如 RTX 30 / 40 系列
- 支持 CUDA 12.8 的 NVIDIA 驱动，以及 [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv sync --locked
```

固定使用 PyTorch 2.8.0（cu128）与匹配的 FlashAttention 2.8.3 预编译包，不需要本地编译或安装 CUDA Toolkit。驱动支持 CUDA 13.x 也可以运行，不要单独升级 PyTorch，否则可能与 FlashAttention 不兼容。

检查 GPU 和依赖：

```bash
uv run python -c "import torch, flash_attn, nanovllm; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name())"
```

## 运行

下载模型并运行示例：

```bash
uv run hf download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B/
uv run python example.py
```

`example.py` 默认从上述目录加载模型。显存不足时，可在 `LLM(...)` 中调小 `max_num_batched_tokens`、`max_model_len` 或 `gpu_memory_utilization`。

## 代码检查与格式化

Ruff 随 `uv sync` 安装为开发依赖：

```bash
uv run ruff check .
uv run ruff format .
# 仅检查格式，不修改文件
uv run ruff format --check .
```

## 代码入口

- `example.py`：推理示例
- `bench.py`：性能测试
- `nanovllm/engine/`：调度、KV Cache 管理与模型执行
- `nanovllm/layers/`：注意力与其他算子
- `nanovllm/models/qwen3.py`：Qwen3 模型实现

原项目采用 MIT 许可证，见 [LICENSE](LICENSE)。
