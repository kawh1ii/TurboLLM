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

实测范围为合成输入下的规划函数。vLLM 等待队列改排、KV 准入及 GPU 端到端延迟需要单独重测。此前的 GPU 数据对应原版源码。

[补丁](incremental-planner.patch) · [复算脚本](benchmark.py) · [完整时间样本与源码散列](measurements.json)

## 复现

取 TurboLLM 仓库提交 5d130be12bafe3379397d755a310e965b4c012fa 的 scheduler 目录，复制两份分别放在 baseline/ 和 candidate/。在本说明所在目录运行以下命令，路径按需调整。实测使用 Python 3.12.14。

```sh
patch -d candidate -p1 < incremental-planner.patch
PYTHONPATH=candidate/src python -m unittest discover -s candidate/tests
python benchmark.py --baseline baseline --candidate candidate --output reproduced.json
```
