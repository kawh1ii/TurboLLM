# TurboLLM：推理调度与 KV 预算管理

扩展 vLLM 调度路径，按请求规划 token 配额，并追踪在途请求尚未分配的 KV 块预算。耗时模型在校准轨迹留出样本上的 MAE 为 0.96 ms。性能报告保留调度开销和高压退化，未观察到稳定的端到端提速。

- [实现与复现步骤](scheduler/README.md)
- [实验结果和适用边界](scheduler/RESULTS.md)
- [设计与实现细节](scheduler/docs/DESIGN.md)
- [源码](scheduler/src) · [测试](scheduler/tests) · [原始实验记录](scheduler/results)

当前实现及其测量证据在 `scheduler/`。旧实现与实验仍可从[此前的项目说明](README.previous.md)查看。
