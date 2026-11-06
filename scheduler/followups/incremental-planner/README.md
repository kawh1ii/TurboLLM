# TurboLLM 规划器增量成本计算

GPU 压力实验曾记录规划阶段 P95 为 2.97 ms。该计时包含请求视图构建和队列排序。这里隔离纯 Python 的 Planner.plan，比较相同请求快照下的两种成本计算实现。

原版每试探一个 prefill 配额都会复制已选配额，再扫描整批请求来计算成本。线性成本的特征可以按请求相加。补丁先固定 decode 统计量，再累积已接纳 prefill 的 token 数和注意力项。每次试探只加入当前候选的贡献。排序保证等待最久的 prefill 位于 prefill 序列首位，因此重新计算强制推进配额时可以保持原有浮点累加次序。

在 Apple M5、Python 3.12.14 上使用冻结的 3B 成本系数与 4096 token 预算。每组 256 个合成输入，各运行 3 次，交替先测旧版或增量版。下表为每组全部 768 个样本的 P95，单位为 ms。

| 输入 | 请求数 | 原版 | 增量版 |
| --- | ---: | ---: | ---: |
| 混合 prefill / decode / 抢占后重算 | 16 | 0.082 | 0.038 |
| 混合 | 64 | 0.562 | 0.128 |
| 混合 | 256 | 6.577 | 0.494 |
| 混合 | 512 | 24.570 | 0.994 |
| 全 prefill | 16 | 0.105 | 0.055 |
| 全 prefill | 64 | 0.436 | 0.202 |
| 全 prefill | 256 | 1.903 | 0.831 |
| 全 prefill | 512 | 3.857 | 1.628 |

10,000 组随机输入和 2,048 组计时输入的配额值及插入顺序、候选次序、预测耗时和强制推进标记完全一致。随机检查覆盖等待时间并列、时间预算开关、等待补偿、多 token 重算和成本反馈缩放。补丁版通过原有 21 项 CPU 回归。

上表只计合成输入下的规划函数。实际 vLLM 热路径和端到端指标分别由下文的 GPU 实验衡量。

[补丁](incremental-planner.patch) · [复算脚本](benchmark.py) · [完整时间样本与源码散列](measurements.json)

## 复现

取 TurboLLM 仓库提交 5d130be12bafe3379397d755a310e965b4c012fa 的 scheduler 目录，复制两份分别放在 baseline/ 和 candidate/。在本说明所在目录运行以下命令，路径按需调整。实测使用 Python 3.12.14。

```sh
patch -d candidate -p1 < incremental-planner.patch
PYTHONPATH=candidate/src python -m unittest discover -s candidate/tests
python benchmark.py --baseline baseline --candidate candidate --output reproduced.json
```

## RTX 4090 严格对照，CUDA 12.9

在 RTX 4090、PyTorch 2.13.0+cu129、vLLM 0.32.0+cu129、Qwen2.5-3B-Instruct BF16 环境运行。使用 FlashAttention 2、eager 和 batch-invariant 模式，在 256 KV 块下回放种子 123 的 48 条相同突发请求。每组生成 8192 个 token，48 条请求的输出 token 序列逐条相等。两组请求和 KV 账本均排空，分配始终在配额内。

本组原版与增量版规划器 P95 分别为 2.260 和 0.911 ms，调度器 P95 分别为 3.413 和 1.952 ms。该组用于严格输出检查，采用单个种子和 trace，不以此组推断常规配置的端到端收益。

- [原始请求、输出和 trace](turbo-gpu-strict/)
- [本地独立复算](gpu-strict-audit.json)
- [CUDA 12.9 官方构建适配](cu129-compat.patch)。在原版和候选的实验副本都应用，放行准确版本后仍检查上游文件散列。

## RTX 4090 热路径对照，CUDA 13.0

使用同一张 RTX 4090、vLLM 0.32.0、PyTorch 2.13.0+cu130 和 Qwen2.5-3B-Instruct BF16。固定模型、策略系数、256 KV 块、2048 批 token 上限和每轮 48 条突发请求。种子 124、125、126 各跑一次原版和增量版，按原版先、增量版先、原版先交替。两版每轮均完成 48 条请求并生成 8192 个 token。上游 scheduler hook 一致，两份项目源码仅 `policy.py` 有差异。

| trace 指标 | 原版 | 增量版 |
| --- | ---: | ---: |
| 规划器 P95，ms | 3.715 | 1.589 |
| 调度器 P95，ms | 6.189 | 3.934 |
| 抢占次数 | 16 | 15 |
| 重算 token 数 | 4241 | 5382 |

表内各项为每轮结果的三种子中位数。逐步检查调度总 token 上限和逐请求配额，六次运行都恢复初始可用块数并清空请求和 KV 账本。时间变化会影响实时到达请求组成，实际轨迹中重算 token 数上升。常规 BF16 的完整输出 ID 一致数分别为 28/48、30/48、32/48。

- [CUDA 13.0 环境](gpu-cu130-environment.json)
- [原始 trace 与工作负载](turbo-gpu-traced/)
- [独立复算](gpu-traced-audit.json)

## 关闭 trace 的端到端对照

使用相同的三种子工作负载、交替顺序和配额配置。下表为各轮指标的三种子中位数。

| 指标 | 原版 | 增量版 |
| --- | ---: | ---: |
| 生成吞吐，token/s | 387.9 | 401.3 |
| TTFT P95，s | 14.777 | 13.709 |
| TPOT P95，ms | 24.347 | 21.150 |
| SLO 内请求数，每轮共 48 条 | 9 | 9 |

三组吞吐均上升，幅度分别为 8.9%、3.5% 和 0.8%。第三组 TTFT P95 从 14.315 升到 14.636 秒。各轮双门槛（TTFT ≤ 1 s、TPOT ≤ 40 ms）达标数均为 9/48，属于过载工况。关闭 trace 后，常规 BF16 完整输出 ID 一致数分别为 26/48、30/48、29/48。

- [原始请求、输出与日志](turbo-gpu-untraced/)
- [独立复算](gpu-untraced-audit.json)

## 常规 BF16 原版自重复

用种子 124 和关闭 trace 的同一份工作负载，在同机分别启动两次原版。源码散列、配置及请求文件字节一致，两轮均生成 8192 个 token，完整输出 ID 一致数为 24/48。常规模式的原版自身已出现漂移。这个单种子控制说明常规对照中的输出差异无法单独归因于补丁。

两轮原版吞吐分别为 398.3 和 387.0 token/s，差约 2.8%。三种子端到端吞吐的中位数变化约 3.5%，幅度与单组原版重复的运行波动接近。调度器和规划器的逐组 trace 改善更一致。

- [原版自重复原始记录](turbo-gpu-baseline-repeat/)
- [独立复算](gpu-baseline-repeat-audit.json)

## 同机严格对照

仍用种子 124 和同一份请求，开启 eager、batch-invariant 和 trace。原版和增量版的 48 条完整 token ID 序列逐条相同，均生成 8192 个 token。两次运行均满足配额检查并排空请求与 KV 账本。该控制与独立 CUDA 12.9 严格组分别报告。

- [同机严格组原始记录](turbo-gpu-strict-cu130/)
- [独立复算](gpu-strict-cu130-audit.json)

## GPU 复现

沿用前述两个源码副本与补丁。使用独立 venv、相同模型文件及仓库内冻结的策略配置。模型目录的 `download-manifest.json` 记录所用模型文件，实测文件散列见环境记录。CUDA 12.9 组还需给两个副本应用相同的 [兼容补丁](cu129-compat.patch)。以下以 CUDA 13.0 环境为例。

```sh
model=/path/to/Qwen2.5-3B-Instruct
python gpu_compare.py --baseline baseline --candidate candidate --model "$model" --output traced --seed 124 125 126 --trace
python gpu_compare.py --baseline baseline --candidate candidate --model "$model" --output untraced --seed 124 125 126
python gpu_compare.py --baseline baseline --candidate candidate --model "$model" --output repeat --seed 124 --repeat-baseline
python gpu_compare.py --baseline baseline --candidate candidate --model "$model" --output strict --seed 124 --trace --eager --batch-invariant
python baseline/tools/patch_vllm.py --restore
```
