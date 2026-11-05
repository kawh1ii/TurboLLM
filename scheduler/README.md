# TurboLLM v2：执行时间与 KV 预算联合调度

本项目对 vLLM 的单卡文本推理调度路径做源码级扩展。核心是：为每个请求规划当前步 token 配额，根据实测执行反馈更新耗时模型，在接纳新请求时核算其他在途请求未来需要的 KV 块。

**重写与验证已完成，实测未显示稳定超过原生 vLLM。** 先看 [重写说明与实测结论](RESULTS.md)，再审查代码；[简历表述建议](RESUME_BULLETS.md)只使用已经实现和验证的内容。

目标是理解并改善长短请求混合时的延迟、吞吐与重计算取舍。**不预设加速倍数，不声称提出了全新的调度算法，也不把上游已有的 chunked prefill、prefix caching、水位或完整输入准入检查作为自己的贡献。**

## 实现边界

- 支持：vLLM 0.32.0、单 GPU、单个完整注意力 KV 组、同步调度、BF16 文本生成；实测模型 Qwen2.5-3B/7B-Instruct。
- 自己实现：耗时特征与非负拟合、反馈修正、逐请求配额、等待补偿、未来块需求账本、前缀命中后准入判断、测量及回归工具。
- 复用 vLLM：模型执行、Paged KV 块表、缓存查找、物理块分配与引用计数、抢占和重计算、输出采样。
- 不支持：多卡、多租户隔离、远程 KV、异步调度、LoRA、多模态、混合注意力、推测解码、流式输入会话。普通流式输出支持。
- TTFT/ITL 是软优化目标，未证明硬实时保证或一般负载最优性。单卡结论不能外推至集群。

## 代码入口

| 文件 | 职责 |
|---|---|
| `src/turbollm/policy.py` | 纯 Python 请求视图、耗时模型、逐请求计划、KV 预算账本 |
| `src/turbollm/vllm_scheduler.py` | 原生追踪基线与自定义调度器；接入实际请求与块表 |
| `patches/vllm-0.32.0-scheduler.patch` | 小范围、显式、可撤销的上游扩展点 |
| `tools/patch_vllm.py` | 校验版本与源码 SHA-256 后应用补丁，拒绝未知版本 |
| `tools/benchmark.py` | 可重放的开环到达、SSE 计时、完整请求记录 |
| `tools/fit_cost.py` | 从原生基线执行记录拟合非负耗时模型 |
| `tools/run_matrix.py` | 同配置基线、调优基线、自定义策略与消融实验 |
| `tools/correctness.py` | 比较实际输出 token ID，差异时非零退出 |
| `tools/analyze.py` | 从原始数据复算 SLO goodput、延迟和机制计数 |

## 复现

在 NVIDIA GPU 的 Linux 机器上使用隔离环境。模型和下载缓存需要另留空间，不能只计算权重大小。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install '.[engine]'
.venv/bin/python tools/patch_vllm.py
.venv/bin/python tools/download_model.py --output /path/to/models/Qwen2.5-3B-Instruct
.venv/bin/python -m unittest discover -s tests -v
```

先建立原生调度基线并校准（以下是流程示例种子，最终实测的完整重跑命令见后文）：

```bash
.venv/bin/python tools/run_matrix.py \
  --model /path/to/models/Qwen2.5-3B-Instruct \
  --output results/calibration --modes native --seeds 41 \
  --count 64 --rate 8 --correctness
.venv/bin/python tools/fit_cost.py results/calibration/native.trace.jsonl \
  --output configs/calibrated.json
```

在不同种子上评估：

```bash
.venv/bin/python tools/run_matrix.py \
  --model /path/to/models/Qwen2.5-3B-Instruct \
  --output results/mixed --config configs/calibrated.json \
  --modes native,tuned,budget --seeds 43,44,45 --correctness
```

`tuned` 是明确的候选配置（5% KV 水位、256 token 阈值、开启上游自适应阈值），**未经校准轨迹上的搜索不能称作最优基线**。支持形如 `tuned-128-0.1` 的候选。调优选择只能用校准数据，最终评估不得重新挑选最佳配置。

`--kv-blocks` 可以构造受控 KV 压力；必须同时保留自然显存容量下的负对照。`--kind` 支持 `mixed`、`burst`、`prefix`、`low` 和 `cancel`。`no_reserve`、`no_time` 分别移除未来 KV 预留和耗时预算约束。

不要把不同 CUDA Graph / 编译模式的结果直接比较；`--eager` 只用于开发排错或单列实验，主对照双方必须一致。源码补丁只在自定义调度器实现相应接口时生效，原生基线不会采用自定义决策。

## 测量说明

1. 默认负载是长度和到达可控的合成 token 请求，不代表真实业务对话分布；共享前缀是单列负载。
2. 每次运行保存完整输入、到达偏移、输出 token ID、SSE 时间戳、失败和超时。预热明确单列。
3. goodput = 同时满足指定 TTFT 与 TPOT 目标的完成请求数 / 从到达开始至排空的墙钟时间。另报所有到达中的通过比例、失败与取消，不能只统计成功请求。
4. SSE 一次可能包含多个 token；TPOT 使用实际 token 数。网络分块间隔不是严格的逐 token GPU 时间。
5. `execution_ms` 是同步调度后到模型输出返回的墙钟时间，包含执行器与 IPC 等开销，**不是 CUDA kernel 时间**。
6. traces 含测量开销；正式结论应说明追踪是否开启，并用关闭写盘的独立运行检查开销。
7. GPU 硬件计数器在当前租用环境中受限；不声称直接测得带宽饱和率、occupancy 或 Tensor Core 利用率。

设计与取舍见 [docs/DESIGN.md](docs/DESIGN.md)。实际测量结果与状态以 `results/` 和最终实验报告为准。

## 输出一致性与完整验证

普通 BF16 模式不保证不同分块/批处理的逐 token 输出稳定；已做原生对原生控制。严格输出对照使用 `--batch-invariant`，不能将其性能与普通模式混为一谈。详细证据见 [docs/DEBUG_LOG.md](docs/DEBUG_LOG.md)。

完整验证脚本使用不重叠的校准、调优和评估种子。它顺序管理单张 GPU，保存阶段源码散列，拒绝把失败阶段或不同代码的旧结果作为已完成证据。下面重现最终使用的种子划分，输出到新的目录以保留附带证据：

```bash
.venv/bin/python tools/full_validation.py \
  --model /path/to/models/Qwen2.5-3B-Instruct \
  --model7b /path/to/models/Qwen2.5-7B-Instruct \
  --output results/reproduction \
  --calibration-seed 61 --tuning-seed 62 \
  --eval-seeds 63,64,65 --audit-seed 66
```

包括自然容量、受控 KV 压力、两项消融、突发、前缀复用、8K 提示词、低负载、取消恢复和同轨迹关闭 trace 的检查。7B 模型单独校准，不复用 3B 的耗时系数。[面试解释路线](docs/INTERVIEW.md)逐项区分了自己的实现与上游能力。

当前代码为 0.3.0。最终报告使用种子 61 校准、62 调优、63/64/65 评估；先前开发版本的结果与源码快照单列保存。`full_validation.py` 的 `--calibration-seed`、`--tuning-seed`、`--eval-seeds`、`--audit-seed` 可指定同一划分。

追踪基线采用原生决策加相同测量包装。主张端到端性能时，还必须查看 `production_compare.py` 的独立比较：它在原生/调优组恢复上游原始源码并直接使用原生 Scheduler，自定义组才应用扩展，双方关闭 trace 写盘。运行此工具时必须先停止其他实验服务；工具结束会恢复补丁状态。

```bash
.venv/bin/python tools/production_compare.py \
  --model /path/to/models/Qwen2.5-3B-Instruct \
  --config results/final/frozen-policy.json \
  --selection results/final/selection.json \
  --output results/reproduction-production-pressure \
  --kv-blocks 256 --seeds 63,64,65
```

不重跑 GPU 也能核对交付材料：`PYTHONPATH=src python tools/verify_results.py`。纯策略与测量测试可单独运行 `PYTHONPATH=src python -m unittest discover -s tests -v`。源码指纹的历史 macOS 元数据差异及可重建证据见 `results/final/source-hash-audit.json`。
