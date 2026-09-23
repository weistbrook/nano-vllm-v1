# Scheduler 受控实验

这组实验比较同一仓库内的调度策略。模型、ModelRunner、Attention、KV 管理、Sampling、输入 token 和随机种子保持一致。三个核心实验固定 `enforce_eager=True`、`use_triton=False`、`use_triton_hidden_rmsnorm=False`，默认 TP=1。

## 阅读当前实现后的结论

- 新请求进入 `waiting`；分配 KV 后进入 `running`。`running` 同时包含未完成 Prefill 和 Decode。完成后释放 block；抢占时释放引用、重置计算进度并回到 `waiting`，保留已有输出 token，重新计算上下文。
- `num_cached_tokens` 是已有 KV 的 token 数，含命中的完整前缀；`num_new_tokens` 是本 step 的输入计算量。Prefill 完成才采样首个输出；普通 Decode 每 step 计算上一次输出并采样一个新 token。提交后只累计一次缓存进度。
- `block_table` 在 chunk 之间保留，下一 step 从 `num_cached_tokens` 继续写入；只对已计算的完整 block 发布 hash，partial block 不进入 Prefix Cache。
- 原仓库的 Scheduler 按 `running` deque 顺序处理，**running-first 不等于 Decode-first**。队首 partial Prefill 可以先耗尽预算；原来的 `chunked_prefill=False` 也不是这里的 legacy baseline。
- 当前 ModelRunner 已通过扁平输入、query/key 累积长度、位置、slot mapping、block table 和 logits 索引支持混合 step。只对本 step 到达上下文末尾的 sequence 采样。这里无需重新设计混合执行路径。
- CUDA Graph 只在存在 paged block table、所有 query 长度不超过 1、批量落在已捕获范围时可用；通常是纯 Decode。多 token Prefill 使用普通执行路径。核心实验关闭 Graph。
- 原 `serving_bench.py` 预生成指数到达间隔，在同步 step 之间接收所有已到达请求。因此延迟应从计划到达时间算起，包含 GPU 忙时的接收等待；无需 HTTP server。

## 两种策略与 baseline 边界

`chunked` 按明确的三个优先级消耗硬预算：

```text
budget = max_num_batched_tokens
1. Decode-ready running sequences：每条 1 token
2. running 中需要继续 Prefill / 重算上下文的 sequences
3. waiting 中的新请求 / 被抢占请求
```

后两类按剩余预算切片，同时检查 sequence slots 和 KV 空间。已选入本 step 的 sequence 不会再被抢占。

`legacy` 模拟 **Prefill 优先、Prefill/Decode 按 step 互斥**：能调度 Prefill 时整个 step 都不给 Decode；没有可执行 Prefill 时才执行 Decode。8192-token Prompt 在 1024 硬预算下同样分片。

这属于 **相同预算下的互斥 Prefill baseline**，不是 upstream 的无分片实现。否则只能突破预算或拒绝长 Prompt，无法满足本实验的控制变量要求。不要将结果描述为“原版 Nano-vLLM 原样复现”。

旧策略下，在线 Decode 必须等连续的 Prefill steps 结束；新策略每 step 先保证可运行的 Decode 获得一个 token 的计算份额，再填入 Prefill。后者减少调度饥饿，但同 step 的 Prefill 仍有计算成本，不保证 ITL 没有 spike，也不保证所有预算下都更快。

## 运行环境

需要项目支持的 Linux CUDA 环境、Python 3.10–3.12、PyTorch、FlashAttention、Triton，以及本地 Qwen3-4B 权重。使用现有可运行推理环境安装额外绘图依赖：

```bash
python -m pip install -e '.[benchmark]'
python -m pytest tests -q
```

`use_triton=False` 关闭可选 RMSNorm/MLP 融合开关；KV store 仍使用原有 Triton kernel，`torch.compile` 也可能生成 Triton。`enforce_eager=True` 关闭的是 CUDA Graph，不关闭这些编译装饰器。两种策略使用同一套算子代码，但 query 形状不同，混合 step 可能自然选择 varlen Attention 路径。这是该调度策略通过现有 ModelRunner 执行的结果，不能据此声称比较的是完全相同的 kernel launch 序列。

每个 case 在独立进程内加载模型，先做非计时结构检查，再完整排练该 workload，以预热实际执行形状；计时前清空前缀元数据、重新设置 Python/NumPy/PyTorch seed。运行时间不含模型加载、检查和预热。重复实验交替两种模式的执行顺序，保留每次结果。避免同一 GPU 同时运行其他任务。

现有 SamplingParams 明确不支持 temperature=0；本框架保留 Sampling 行为，采用正温度和结构正确性检查。同 seed、同输入不等于两种 batching 顺序下逐 token 输出相同，不据此判错。

## 实验 1：长 Prompt 干扰

```bash
python benchmarks/long_prompt_interference.py \
  --model ~/models/Qwen3-4B \
  --scheduler-mode both \
  --num-decode-requests 32 \
  --short-prompt-len 128 --short-output-len 256 \
  --long-prompt-len 8192 --long-output-len 64 \
  --inject-after-tokens 16 \
  --max-num-batched-tokens 1024 --max-num-seqs 64 \
  --tensor-parallel-size 1 --seed 100 --enforce-eager
```

也可用 `--scheduler-mode legacy` 或 `chunked` 单独运行；用 `both` 会直接共享一次生成并落盘的 workload。短请求同时到达，等每条至少生成 16 token 后注入长请求。比较短请求 ITL，特别是干扰窗口的 P50/P95/P99/Max，以及长请求 TTFT/latency。

图像包含以长请求到达为零点的 ITL timeline、Prefill 完成标记，以及干扰窗口 percentile 对比。图使用 matplotlib 默认配色，不预设改善幅度。

## 实验 2：长 Prompt 长度扫描

```bash
python benchmarks/long_prompt_sweep.py \
  --model ~/models/Qwen3-4B \
  --prompt-lengths 2048 4096 8192 \
  --repeats 3 \
  --max-num-batched-tokens 1024 --max-num-seqs 64 \
  --seed 100 --enforce-eager
```

输出 `results/long_prompt_sweep.csv`，以及长度对干扰窗口 P99 ITL、长请求 TTFT 的图。每个长度、每次重复都配对比较两种模式；不同重复使用可复现的 seed。可自行增加 16384。该实验验证干扰是否随输入长度扩大，以及 Decode 延迟与长请求首 token 等待之间的取舍。

## 实验 3：Token Budget 消融

```bash
python benchmarks/token_budget_ablation.py \
  --model ~/models/Qwen3-4B \
  --budgets 512 1024 2048 4096 \
  --repeats 3 --long-prompt-len 8192 \
  --max-num-seqs 64 --seed 100 --enforce-eager
```

只运行 chunked。输出 `results/token_budget_ablation.csv`，以及预算对 Decode P99 ITL、长请求 TTFT 和吞吐的图。JSON 还保留 step 的 Decode 数、Prefill 输入量和长 Prompt 分片数，帮助验证“预算 → chunk 大小 → step 负载 → 延迟”的关系。

预算必须大于短 Decode 请求数；小于该数量无法每步覆盖全部 Decode，等于它又无法给长 Prompt 留预算，均明确报错。并发上限必须容纳全部短请求及长请求。输出长度或注入阈值导致短请求提前退出时，也会报错或将干扰窗口标为无效，不能混入有效比较。

## 指标口径

- 每次 `engine.step()` 前后比较 `num_completion_tokens`，只接受增量 0 或 1。部分 Prefill 不记输出；完成 Prefill 的首个输出和 finished sequence 的最后一个输出均记录。保留 sequence 引用避免漏掉已从 running 移除的请求。
- token timestamp 是 `perf_counter()` 的 engine 返回时刻，同 batch 输出共享时间戳。真实输出通过 runner 的 GPU tensor `.tolist()` 返回 CPU 后才可见，不在每个 operation 加 synchronize。
- TTFT = 首 timestamp − arrival；ITL = 相邻 timestamp 差；TPOT = 首尾 timestamp 差 / (输出数量 − 1)；latency = 最后 timestamp − arrival。输出一个 token 时没有 ITL，TPOT 留空。
- CSV/JSON 时间统一为 **秒**，图和终端的延迟显示使用 **毫秒**。`throughput` 是输入加输出 token/s，另存 `output_throughput`，避免与只算输出的吞吐混淆。
- ITL 分位数汇总全部短请求的 token 间隔，按 token 加权；TPOT 是逐请求汇总。不同重复的图是每次统计量的聚合，不将不同重复的原始 ITL 混成一个分位数。
- 干扰窗口从长请求实际注入开始，到首 token 返回为止，作为 engine 可观测的 Prefill 完成边界。**ITL 只要与该窗口重叠就归入干扰**，使用完整间隔长度，不切分、不重复统计。legacy 的阻塞间隔通常在 Prefill 完成之后才结束，仅筛选落在窗口内的 token 会漏掉最关键的 spike。
- `step_records` 的时间是主机返回时间，且包含相邻 step 之间的框架开销；没有输出的纯 Prefill chunk 可能尚未完成 GPU 工作。它用于核对调度量，不是独立 GPU kernel/step 耗时。`decode_bench.py` 的前后 synchronize 则是另一种单 step wall-clock 测量。

保存的 workload、SHA256、每 token 时间戳、环境版本、GPU、有效 KV 容量、参数和检查结果用于复核控制变量。A/B 配对发现 GPU、有效 KV 容量或配置不同会拒绝作为受控对比。没有有效样本的分位数留空，不填 0，不补造数据。

## 正确性与结果解释

CPU 回归覆盖调度顺序、step 互斥、硬预算、抢占重算、Prefix Cache 完整 block 发布、引用计数及最终释放。GPU case 在计时前检查输出计数、有限步完成、部分 block、共享前缀复用、KV 索引范围和 block 释放；完整排练还验证正式 workload 可以完成。

结构检查不能代替 GPU 数值等价验证，也不声称已验证所有 TP 拓扑。性能结论必须从真实 CSV/图得出；若没有可用推理环境，框架明确失败，不会产生替代性能数据。

更严谨的项目描述：

> 在支持混合 Prefill/Decode 的 ModelRunner 上完善 Token Budget 驱动的 Chunked Prefill 调度，明确 Decode、已有分片 Prefill 和新请求的优先级；通过分页 KV Cache 跨 step 维护增量上下文，并在同一推理栈内建立 Prefill/Decode step 互斥 baseline，以逐 token ITL、长请求 TTFT 和吞吐评估调度策略的取舍。

当前版本可以支持上述描述。“重构 ModelRunner”属于该仓库既有实现的历史工作，这次修改主要完善 Scheduler 和实验框架；在跑出实际结果前，不宜写出具体 ITL 降幅或宣称已经消除长 Prompt 干扰。
