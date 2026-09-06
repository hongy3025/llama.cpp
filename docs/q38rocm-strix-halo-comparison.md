# q38rocm 与当前分支的 Strix Halo 优化对比报告

> 研究快照：2026-09-06  
> 比较对象：Qwen 3.8 27B、ROCmFP4/ROCmFP4_FAST、AMD Strix Halo（gfx1151）  
> 性能口径：用户报告当前分支在很小实际上下文下约为 35 tok/s；本报告不重复运行该基准

## 1. 结论

当前约 35 tok/s 与 q38rocm 宣传的最高 36.04 tok/s，作为小上下文 MTP 用户可见吞吐的峰值/爆发档位，可以判定为**同一性能档**。若要把这一档位判断升级为严格 A/B 结论，还必须对齐以下条件：

- 很小的实际上下文；
- 单流；
- greedy 或 near-greedy；
- embedded/separate MTP speculative decode；
- 以用户可见输出吞吐为指标；
- 高可预测的代码、JSON 或类似内容。

两者相差 1.04 tok/s，相对 36.04 tok/s 低 2.89%。从数值定位看，当前 35 tok/s 还比 q38rocm 文档给出的默认单流 sustained sweet spot 33.80 tok/s 高 1.20 tok/s，即 3.55%。但前者是用户报告的小上下文结果，后者是对方的 sustained 声明，这一比较只说明数值已经进入该区间，不证明当前 sustained 吞吐更高。1.04 tok/s 的峰值差距小于 MTP acceptance、输出内容、draft depth、采样器、时钟状态和生成长度通常造成的波动。

这个结论不能扩展为“两个运行栈整体性能完全等价”。当前结果没有证明以下项目：

| 维度 | 判断 |
|---|---|
| 短上下文、单流、greedy/near-greedy MTP 峰值 | 基本同一性能档 |
| 相对 q38rocm 默认 33.80 tok/s sustained 档 | 数值已进入并略高于该区间；当前 sustained 尚未严格证明 |
| target-only 裸解码 `tg` | 尚未证明同档 |
| 长上下文 sustained decode | 尚未证明；当前缺少 TurboQuant KV |
| Prefill 和 TTFT | 尚未证明 |
| ROCmFP4 Vulkan | 当前明确未实现 |
| Prompt-cache 复用 | 功能集合各有强项，需要按冷/热请求实测 |
| 多 slot 并发和 aggregate throughput | 尚未证明 |
| 长期频率、OOM、GPU reset 和恢复稳定性 | 尚未证明 |

## 2. 比较范围与来源

### 2.1 当前仓库

- 仓库：`hongy3025_llama.cpp.devel`
- 分支：`feat/rocmfpx`
- 研究时源码 HEAD：`fb1d5d468fc68b4216d58edd871620721f41f75c`
- 本地二进制：`0.4.0-dev`，build 10871，commit `cc76f036b`
- 本地构建目标：`gfx1151`
- 本地后端：HIP/ROCm 开启，Vulkan 关闭

源码 HEAD 比本地二进制多一个服务器脚本提交；该差异不涉及核心推理内核。

### 2.2 q38rocm wrapper

- 仓库：`https://github.com/julianmb/q38rocm`
- 研究时本地 HEAD：`868c7bee2ea02d8ee491e71e46f7294d0ad9d08a`
- 角色：构建、启动、缓存、系统调优、benchmark 和 NPU 实验包装层

q38rocm 本身不是独立的矩阵乘或 attention 内核实现。它固定构建下述 ROCmFPX engine，并围绕该 engine 提供部署 profile。

### 2.3 q38rocm 使用的 engine

- 仓库：`https://github.com/ROCmFPX/ROCmFPX`
- 固定 commit：`75e67a92b2d230849aec2d6c1f7b1d1fd624e0e0`
- 本地二进制：`0.3.0-dev`，build 11474，commit `75e67a92b`
- 本地构建目标：`gfx1151`
- 本地后端：HIP/ROCm 开启，Vulkan 关闭

q38rocm 的 `build_engine.sh` 默认设计为 HIP+Vulkan 双后端，但研究时磁盘上的 `engine/src/build-static/CMakeCache.txt` 显示 `GGML_VULKAN=OFF`。因此该本地二进制不能直接复现 README 所称的 Vulkan0 36.04 tok/s。

README 仍把 engine pin 写成 `0fc9568e...`，而实际 `build_engine.sh`、`engine/BUILD_INFO.txt` 和二进制都已经是 `75e67a92...`。本报告在发生冲突时以源码、CMake cache、BUILD_INFO 和原始 JSON 为准。

## 3. 总体架构判断

q38rocm 可以拆成四层：

1. **ROCmFPX engine**：ROCmFP4/FAST、其他 ROCmFPX 格式、HIP/Vulkan kernels、MTP、TurboQuant 和可选插件。
2. **启动 profile**：backend、KV 类型、MTP depth、slot、batch、RAM checkpoint 等参数。
3. **系统调优**：GPU DPM、GTT/TTM、THP、ROCm/RADV 环境变量。
4. **实验组件**：NPU prefix handoff、替代 DFlash engine、Hyperloom 等开发工具。

只有第一层中的 Qwen3.8 实际热路径和第二层的运行参数能够直接解释 27B 持续解码吞吐。GPU 调频、GTT、THP 等主要影响频率波动、容量和稳定性；NPU prefix handoff 主要影响 TTFT。它们不能被等同为新的 llama.cpp 推理内核。

## 4. 优化逐项对照

### 4.1 汇总表

| 优化或能力 | q38rocm / pinned engine | 当前仓库 | 状态 | 与 36.04 tok/s 的关系 |
|---|---|---|---|---|
| ROCmFP4、ROCmFP4_FAST GGUF 类型 | 有 | 有 | 已实现 | 核心前提 |
| CPU quant/dequant/vector-dot | 有 | 有 | 已实现 | 转换与 fallback，非 GPU 峰值核心 |
| HIP MMVQ、MMQ、dequant、`MUL_MAT_ID` | 有 | 有 | 已实现 | 核心 HIP 路径 |
| RDNA3.5/gfx1151 MMQ 配置 | 有 | 有同类实现 | 部分等价 | 核心，但具体 layout/参数不同 |
| ROCmFP4 专用 codebook 和未对齐 load | 有 | 无独立专用实现 | 缺少微优化 | 可能影响 HIP 热路径 |
| gfx1151 ROCmFP4 专用 MMVQ knobs | 有 | 主要复用通用 RDNA2 参数表 | 缺少调优 | 可能影响 HIP MMVQ |
| ROCmFP4 Vulkan MMVQ/MMQ/dequant/FA | 有 | 无 | 明确缺失 | 复现文档 Vulkan 峰值的最大功能缺口 |
| 通用 Strix Vulkan batched mat-vec | 有 | 有 | 已继承 | 当前 ROCmFP4 尚不能使用该 Vulkan 路径 |
| Qwen 大词表 radix TOP_K | 有 | 有 | 已继承 | 减少 sampler 开销，非独有优势 |
| embedded/separate MTP 基础设施 | 有 | 有 | 同代能力 | 36.04 的核心放大器 |
| MTP 内部 `top_k=10` | 有 | 有 | 已实现 | 两边 draft 候选逻辑同类 |
| `--spec-mtp-strict-qwen` | 有 | 无 | 明确缺失 | 正确性和边界安全，非峰值必要项 |
| empty `data_spec` restore 和 MTP reset | 有 | 无 | 明确缺失 | Cache/MTP 恢复稳定性 |
| TurboQuant 3/4 KV | 有 | 无 | 明确缺失 | 长上下文内存和带宽的重要差异 |
| RAM prompt cache/checkpoints | 有 | 有 | 已实现 | 热请求和分支复用 |
| GGSD SSD prompt cache | 无 | 有 | 当前额外能力 | 跨进程/磁盘缓存优势，非短上下文峰值 |
| DFlash2、ngram-map-k4v | 有/文档另有 fork 路径 | 有 | 当前已有 | 不属于 q38rocm 的 36.04 embedded-MTP 路径 |
| DFlash encoder/KV injection 融合 | 有 | 有 | 已继承 | 只作用 DFlash workload |
| ROCmFP2/3/5/6/7/8、ROCmI4 | 有 | 无 | 格式覆盖较少 | 与 ROCmFP4_FAST 36.04 弱相关 |
| ROCmFPX plugin ABI/Charlie Vulkan plugin | 有，可选且默认关闭 | 无 | 可选能力缺失 | 默认 36.04 路径不依赖 |
| Qwen4Exp PLE 模型支持 | 有 | 有 | 双方均有 | 不属于 Qwen3.8 27B |
| AMD Mamba-2 SSD prefill | 有 | 当前 SSD 只允许 NVIDIA | 部分缺失 | 不属于 Qwen3.8 27B |
| NPU prefix handoff | 有实验脚本 | 无 | 实验能力缺失 | 只改善 TTFT，不改善持续 decode |
| DPM/GTT/THP 自动调优脚本 | 有 | 无同等自动化 | 部署层差异 | 影响波动、容量和稳态，不是内核能力 |

### 4.2 当前已经完整具备的 ROCmFP4 基础

当前仓库已经不是“只有 ROCmFP4 文件格式”。它实现了完整的基本推理链路：

- `GGML_TYPE_Q4_0_ROCMFP4 = 100`
- `GGML_TYPE_Q4_0_ROCMFP4_FAST = 101`
- CPU 量化、反量化和 vector dot
- CUDA/HIP f32/f16/bf16 反量化
- CUDA/HIP MMVQ 和 MMQ
- `MUL_MAT_ID`/MoE 类型分派
- RDNA3、RDNA3.5 和其他架构的 MMQ 配置
- `llama-quantize` 和模型加载支持
- FAST、LEAN、COHERENT、STRIX、STRIX_LEAN 等量化策略

当前关键实现包括：

- `ggml/src/ggml-cuda/vecdotq.cuh`
- `ggml/src/ggml-cuda/mmq-load-tiles.cuh`
- `ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh`
- `ggml/src/ggml-cuda/mmvq.cu`
- `docs/rocmfp4.md`
- `docs/rocmfpx-merge-changes.md`

因此当前 35 tok/s 不是依靠 fallback 或临时兼容层得到的；ROCmFP4 已实际进入 HIP 计算路径。

### 4.3 HIP 热路径仍不完全相同

Pinned engine 在 `ggml/rocmfp4/rocmfp4_hip_codebook.cuh` 中提供 ROCmFP4 私有实现：

- `rocmfp4_get_qs_i32()`
- `rocmfp4_get_int_from_codebook_16()`
- `rocmfp4_get_low_int_from_codebook_16()`
- 编译期常量化的 AMD `amdgcn_perm` codebook 展开
- `GGML_ROCMFP4_UNALIGNED_QS_DWORD_LOAD`

当前 ROCmFP4 热路径仍主要调用通用的 `get_int_b4()` 和 `get_int_from_table_16()`。当前通用 helper 同样可以使用 AMD byte permute，因此这不是“当前完全没有对应能力”，而是缺少 pinned engine 的 ROCmFP4 私有常量化和可单独开关的未对齐 dword-load 实现。最终生成指令和收益必须通过 kernel A/B 或反汇编确认，不能仅凭源码名字断言有固定百分比损失。

Pinned engine 的 `mmvq.cu` 还包含 gfx1151/RDNA3.5 ROCmFP4 专用参数，例如：

- `GGML_ROCMFP4_RDNA35_NWARPS`
- `GGML_ROCMFP4_RDNA35_NWARPS_MAX_NCOLS`
- `GGML_ROCMFP4_RDNA35_MMID_MAX_BATCH`
- `GGML_ROCMFP4_RDNA35_RPB_WIDE*`
- `GGML_ROCMFP4_MOE_MMVQ_ROWS_PER_BLOCK`

当前 RDNA3.5 MMQ 已有同类配置，但 MMVQ 仍主要复用通用 RDNA2 parameter table。两边属于“计算能力同类，具体热路径和调参不完全一致”。

### 4.4 Vulkan 是明确的后端功能差距

Pinned engine 的普通 Vulkan backend 包含 ROCmFP4/FAST 的：

- MMVQ
- MMQ
- `MUL_MAT_ID`
- dequant
- copy/set-rows
- FlashAttention shader 支持

当前 `ggml/src/ggml-vulkan` 中没有 `ROCMFP4` 或 `Q4_0_ROCMFP4` 分派。因此即使打开 `GGML_VULKAN=ON`，也不能认为当前已经拥有对方的 ROCmFP4 Vulkan 路径。

当前已经继承通用 Strix Halo Vulkan batched mat-vec 调优，但缺少 ROCmFP4 类型接入。这两件事必须分开判断。

### 4.5 MTP 基础同代，但恢复和严格模式有差距

两边都有：

- `common_speculative_impl_draft_mtp`
- target/draft context
- recurrent state
- checkpoint/speculative state 基础设施
- MTP sampler `top_k=10`

Pinned engine 额外包含：

1. `--spec-mtp-strict-qwen`
2. 单 slot 限制和 rollback 深度校验
3. 256-token padded KV block 边界限制
4. exact prompt-cache hit 时的 cold fallback
5. empty `data_spec` restore 容忍
6. 清空 `pending_h`、`verify_h`、`verify_h_rows`、`i_last`、`chain_h`
7. restore 后把 `drafting` 重置为 `false`

这些改动首先是 greedy-exact、边界安全和缓存恢复修复，不是 q38rocm `n7/p0.35` 峰值所必需。缺少它们不会否定当前小上下文 35 tok/s，但会影响复杂 prompt-cache/MTP 场景的等价性。

### 4.6 TurboQuant 主要决定长上下文等价性

Pinned engine 支持 `GGML_TYPE_TURBO3_0`、`GGML_TYPE_TURBO4_0` 及相应 HIP FlashAttention 路径。q38rocm 默认 speed profile 使用：

- K KV：`q8_0`
- V KV：`turbo4`

当前没有 Turbo3/Turbo4。当前检入的 `start-server.sh` 只显式把 draft KV 配成 `q8_0/q8_0`，主 KV 没有显式指定，因而仍使用默认 `f16/f16`。

在 512 个实际 token 时，q38rocm canonical 表中的 KV 只有约 0.04 GiB，因此缺 TurboQuant 不能合理解释 35 与 36.04 之间的 1.04 tok/s 差异。随着上下文增长，KV 的内存占用和每 token 带宽不断增加，这一差异会变得重要。因此：

- 短上下文峰值：TurboQuant 不是必要条件；
- 长上下文内存效率和持续解码：当前与 q38rocm 明确不等价。

### 4.7 Cache 能力不是简单的强弱关系

双方都有：

- `--cache-ram`
- `--cache-prompt`
- `--ctx-checkpoints`

当前额外包含 GGSD SSD 自动缓存：

- `--prompt-cache-ssd`
- `--prompt-cache-ssd-max-mib`
- `--prompt-cache-ssd-min-prefix`
- `--prompt-cache-ssd-margin`

Pinned engine 则在 MTP checkpoint restore、empty-state reset 和 strict boundary handling 上更完整。结论应是“功能集合各有强项”，而不是当前整体缓存能力更弱。

### 4.8 DFlash、PLE、SSD、插件和扩展格式的相关性

以下能力不应包装成 Qwen3.8 27B ROCmFP4_FAST 36.04 tok/s 的核心原因：

- ROCmFP2/3/5/6/7/8 和 ROCmI4：不同权重量化格式；
- Qwen4Exp PLE：不同模型架构，当前仓库也已有基础模型支持；
- AMD State Space Duality：用于 Mamba-2 长 prefill，不是 Qwen3.8 27B 主路径；
- Hyperloom：开发工具，不由常规 `llama-cli`/`llama-server` 推理加载；
- ROCmFPX plugin ABI：扩展接口，默认没有启用；
- Charlie Vulkan plugin：可选独立 backend，不是普通 `Vulkan0`；
- DFlash2 高吞吐：属于另一条 speculative 路径，q38rocm 的 36.04 使用 embedded MTP。

当前已继承若干与 pinned engine 同源的 upstream 优化，包括 Strix Vulkan batched mat-vec、Qwen radix top-k、RDNA3 MMQ 和 DFlash encoder/KV injection 融合。因此这些不能作为 q38rocm 相对当前的独有优势重复计算。

## 5. 部署和 NPU 调优的定位

### 5.1 环境与硬件脚本

q38rocm `setup_env.sh` 设置：

- `HSA_OVERRIDE_GFX_VERSION=11.5.1`
- `GGML_HIP_ENABLE_UNIFIED_MEMORY=1`
- `HIP_VISIBLE_DEVICES=0`
- `ROCM_FLUSH_ACCEPT=1`
- `AMD_VULKAN_ICD=RADV`
- `RADV_PERFTEST=gpl,sam,nggc`

`apply_hardware_tweaks.sh` 还会尝试：

- 将 GPU DPM 锁为 `high`
- 把 THP 设置为 `madvise`
- 在 128 GiB 主机上把 TTM/GTT limit 调到约 120 GiB
- 在 64 GiB 主机上调到约 56 GiB

这些设置可能减少降频和内存不足，使长时间结果更稳定，但没有增加新的 ggml 计算内核。公平 A/B 必须统一这些设置，而不是把脚本本身计作内核优化。

### 5.2 NPU pipeline

q38rocm 的 `scripts/run_pipeline.py` 使用两阶段 prefix handoff：

1. NPU 上的 0.8B 模型先流式生成约 24 token；
2. iGPU 27B 模型从已经发送给用户的 prefix 继续生成。

其文档明确记录：

| 架构 | Decode | 长 prompt TTFT |
|---|---:|---:|
| iGPU + embedded MTP | 33.8 tok/s | 1587 ms |
| NPU burst → iGPU | 33.8 tok/s | 870 ms |

NPU 把 TTFT 改善约 1.8 倍，但不改善持续 decode。NPU 和 iGPU 同时进行 sustained co-decode 还会受到共享 DRAM contention 影响。因此当前没有 NPU pipeline 不构成 35 与 36.04 的核心差距。

## 6. Benchmark 证据审查

### 6.1 36.04 的实际口径

q38rocm README 和 `benchmarks/canonical.json` 把 36.04 tok/s 定义为：

- ROCmFP4_FAST
- MTP deep spec `n7/p0.35`
- temperature 约为 0
- JSON/code 等高可预测内容
- 很短实际上下文
- peak/burst，而非默认持续 profile

q38rocm 默认推荐的 `n4/p0.0` 单流 sustained sweet spot 是 33.80 tok/s。

### 6.2 Context scaling

Canonical 表给出的数据如下：

| 实际上下文 | KV RAM | Prefill | Raw `tg` | MTP decode |
|---:|---:|---:|---:|---:|
| 512 | 0.04 GiB | 382.21 tok/s | 14.06 tok/s | 34.82–36.04 tok/s |
| 2,048 | 0.15 GiB | 356.85 tok/s | 14.04 tok/s | 32.40–34.82 tok/s |
| 4,096 | 0.31 GiB | 339.73 tok/s | 14.01 tok/s | 30.56–32.24 tok/s |
| 8,192 | 0.62 GiB | 311.76 tok/s | 13.98 tok/s | 29.73 tok/s |
| 16,384 | 1.23 GiB | 266.57 Vulkan / 329.86 ROCm | 13.85 tok/s | 28.02 tok/s |
| 32,768 | 2.45 GiB | 约 245 tok/s | 13.62 tok/s | 26.85 tok/s |

用户报告当前约 35 tok/s 时实际上下文很小，所以它只支持对第一行附近进行判断，不能外推到 8K、16K 或 32K。

### 6.3 原始 artifact 与 canonical 的区别

检入的 `benchmarks/benchmark_20260816_043425.json` 记录：

- 平均 decode：26.85 tok/s
- 四个 prompt token 数：71、90、87、74
- 单项 decode：32.24、29.73、28.02、17.41 tok/s
- 平均 MTP acceptance：73.7%

对整个 `benchmarks/` 搜索 36.04，只命中整理后的 `canonical.json`，没有与 36.04 对应的请求级 raw JSON。

这不证明 36.04 虚假。准确结论是：

> 36.04 有 README、canonical 和外部复现声明，但仓库内没有对应的请求级原始 artifact，其仓库内证据强度低于检入的 raw JSON；它也不能被当作默认持续性能。

### 6.4 Benchmark harness 的复现限制

`scripts/benchmark.py` 固定 `temperature: 0.0`，并从 server response 的 `timings` 读取：

- `predicted_per_second`
- `draft_n`
- `draft_n_accepted`

但 raw report 没有完整记录：

- server 命令
- backend/device
- engine commit
- GGUF SHA256
- seed
- 完整 sampler chain
- reasoning 状态
- 输出内容

`canonical.json` 补充了 GGUF SHA256、硬件、Mesa 和默认 speed profile，但仍不是 36.04 对应的请求级产物。公平重放需要额外固定这些变量。

## 7. 为什么短上下文可判同档，但裸内核尚不能

Speculative decode 的用户可见速度可近似理解为 target verification 成本、draft 成本和 acceptance 的组合。相同 target kernel 下，输出越容易预测，单次 target verification 接受的 draft token 越多，用户可见 tok/s 就越高。

反过来，即使两个 target backend 的裸 `tg` 不完全相同，也可能因为不同 acceptance 得到接近的最终 MTP tok/s。因此：

- 35 与 36.04 接近，足以证明当前完整 MTP 路径没有明显落后一个性能级别；
- 它不足以证明当前 HIP ROCmFP4 kernel 与 pinned engine 或 Vulkan kernel 完全等速；
- 裸内核必须关闭 speculation，比较相同 prompt depth 下的 target-only `tg`；
- MTP 比较必须同时记录 drafted、accepted 和 acceptance，而不能只看最终 tok/s。

q38rocm 自己的 raw artifact 在约 71–90 token prompt 上从 17.41 到 32.24 tok/s，说明内容和 acceptance 的影响远大于 1.04 tok/s 的差值。

## 8. 两个 launcher 默认配置不等价

| 项目 | 当前 `start-server.sh` | q38rocm `run_server.sh` 默认 speed profile |
|---|---|---|
| 权重模型 | `ROCMFP4_STRIX` | `ROCmFP4_FAST` |
| MTP 模型 | 独立 `-md` GGUF | embedded MTP |
| Slot | `-np 4` | 默认 1 |
| 配置 context | 131072 | 131072 |
| 实际 benchmark context | 取决于请求 | 36.04 属于很短实际上下文 |
| Speculator | `ngram-map-k4v,draft-mtp` | `draft-mtp` |
| Draft depth | `n_max=3` | 默认 `n_max=4`；peak 使用 7 |
| Draft probability cutoff | 使用当前脚本/默认配置 | 默认 0.0；peak 为 0.35 |
| Main KV | 未显式设置，默认 F16/F16 | q8_0/turbo4 |
| Draft KV | q8_0/q8_0 | 由 embedded context/profile 决定 |
| RAM cache | 8192 MiB | 128 GiB 主机默认 32768 MiB |
| Checkpoints | 8 | 128 GiB 主机默认 64 |
| SSD cache | GGSD 自动缓存 | 无 GGSD |

因此不能直接运行两个 launcher 的默认值，然后把结果解释成纯 engine A/B。模型量化 preset、slot、speculator、KV 类型和 cache 配置都已经变化。

## 9. 最小公平 A/B 矩阵

### 9.1 固定条件

必须固定：

1. 同一台 Strix Halo APU；
2. 相同 GPU governor、频率和功耗状态；
3. 相同 ROCm、Mesa、内核和环境变量；
4. 同一 GGUF SHA256；
5. 同一 backend，`ROCm0` 与 `Vulkan0` 分开报告；
6. 同一实际 prompt token 深度，而不只是相同 `-c`；
7. 同一 completion 长度、seed、temperature 和 reasoning 状态；
8. 同一 sampler、repeat/presence penalty；
9. 同一 slot 数、batch、ubatch；
10. 同一 KV K/V 类型；
11. 同一 MTP `n_max`、`p_min` 和 strict 状态；
12. 冷缓存与热缓存分开。

由于当前没有 ROCmFP4 Vulkan 和 TurboQuant，第一轮公平基线应使用双方都支持的共同路径：

- backend：`ROCm0`
- 单 slot
- 同一个 ROCmFP4_FAST GGUF
- main/draft KV：共同支持的 `q8_0/q8_0`
- 只启用 `draft-mtp`，禁用 ngram
- strict 关闭

随后才能单独测 q38rocm 的 `q8_0/turbo4` 增量。Vulkan0 应作为另一个“当前缺失功能”测试，而不是与当前 ROCm0 直接混比。

### 9.2 推荐顺序

| 顺序 | 场景 | 目的 |
|---:|---|---|
| 1 | 512、8K、32K，no-spec | 隔离 target-only ROCmFP4 `tg` |
| 2 | 相同深度，MTP `n4/p0` | 比较默认 sustained MTP |
| 3 | 512，MTP `n7/p0.35` | 只作为 peak probe |
| 4 | code、prose、JSON 各生成 512–1024 token | 控制 acceptance 的内容依赖 |
| 5 | 4K、32K、128K cold prefill | 比较 prefill 和 TTFT |
| 6 | 32K、64K warm turn | 比较 prompt-cache/checkpoint 恢复 |
| 7 | q8_0/q8_0 对 q8_0/turbo4 | 隔离 TurboQuant 收益 |
| 8 | 4 slot 并发 | 比较 aggregate 和 per-slot 吞吐 |
| 9 | 长时间 soak | 检查频率、OOM、GPU reset 和缓存增长 |

### 9.3 每次必须记录

- target-only raw `tg`
- prompt processing `pp`
- TTFT
- drafted token 数
- accepted token 数
- acceptance
- 用户可见 tok/s
- completion token 数
- 实际 prompt/context token 数
- RSS、KV cache 和 GPU memory
- backend/device
- engine commit
- GGUF SHA256
- 完整命令和 sampler 参数
- 长时间运行中的频率、OOM、GPU reset

每个场景至少运行三次并报告中位数。Peak/burst 与 sustained 必须分列，不能只摘最高一次。

## 10. 建议的实现优先级

如果目标是补齐真正影响 Strix Halo Qwen3.8 的差距，建议按以下顺序：

1. **先建立共同 ROCm0 no-spec/MTP guard。** 在没有同条件基线前，不应直接移植所有下游代码。
2. **评估 ROCmFP4 私有 codebook、未对齐 dword load 和 gfx1151 MMVQ knobs。** 每项用 kernel microbenchmark 和真实 27B sustained guard 决定是否保留；当前已经达到 35 tok/s，编译器可能已生成接近的指令，不能只按源码差异推断收益。
3. **补 TurboQuant KV 及相应 HIP FlashAttention。** 这是长上下文容量和带宽等价性的主要缺口。
4. **补 strict Qwen MTP 与 empty `data_spec` 状态复位。** 目标是正确性、缓存边界和长期稳定性。
5. **需要 Vulkan 部署时，再补完整 ROCmFP4 Vulkan backend。** 这是最大的 backend 能力缺口，但不是当前 ROCm0 35 tok/s 的必要条件。
6. **部署脚本单独管理。** GPU DPM、GTT/TTM、THP 和 RADV 参数应视为机器 profile，不与核心 kernel 提交混在一起。

扩展 ROCmFPX 格式、插件 ABI、Mamba SSD、NPU pipeline 和 Hyperloom 不应进入“追平 Qwen3.8 36.04 tok/s”的首要范围。

## 11. 最终判断

可以使用的准确表述：

> 当前分支在很小上下文达到约 35 tok/s，已经与 q38rocm 的 36.04 tok/s ROCmFP4_FAST+MTP 峰值处于同一性能档；从数值上也进入并略高于其默认 33.80 tok/s sustained 区间，但当前 sustained 吞吐仍需同条件 A/B 证明。这个结果证明当前 ROCmFP4 HIP+MTP 主链路已经具有竞争力，但不证明裸 kernel、Vulkan、长上下文、prefill、缓存复用、并发和长期稳定性完全等价。当前明确缺少 ROCmFP4 Vulkan、TurboQuant KV、strict Qwen MTP、empty-state restore，以及若干 ROCmFP4/gfx1151 专用 HIP 微调。

不应使用的表述：

- “已经实现 q38rocm 的全部优化”；
- “35 与 36 证明所有上下文长度完全等速”；
- “NPU、GTT 或 THP 是 36 tok/s 的核心推理内核”；
- “README 中的 36.04 已由仓库内 raw artifact 完整重放”；
- “缺少 TurboQuant 会直接导致 512-token 峰值少约 1 tok/s”。

## 12. 关键证据索引

当前仓库：

- `docs/rocmfp4.md`
- `docs/rocmfpx-merge-changes.md`
- `build_rocm.sh`
- `start-server.sh`
- `ggml/src/ggml-cuda/vecdotq.cuh`
- `ggml/src/ggml-cuda/mmq-load-tiles.cuh`
- `ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh`
- `ggml/src/ggml-cuda/mmvq.cu`
- `common/speculative.cpp`
- `tools/server/server-context.cpp`

q38rocm wrapper：

- `README.md`
- `build_engine.sh`
- `run_server.sh`
- `setup_env.sh`
- `apply_hardware_tweaks.sh`
- `scripts/benchmark.py`
- `scripts/run_pipeline.py`
- `benchmarks/canonical.json`
- `benchmarks/benchmark_20260816_043425.json`
- `docs/NPU_INTEGRATION.md`
- `docs/UPSTREAM_TRACKING.md`

Pinned ROCmFPX engine：

- `ggml/rocmfp4/rocmfp4_hip_codebook.cuh`
- `ggml/src/ggml-cuda/mmvq.cu`
- `ggml/src/ggml-cuda/mmq-config-rdna3-5.cuh`
- `ggml/src/ggml-cuda/fattn*.cuh`
- `ggml/src/ggml-cuda/ssm-scan.cu`
- `ggml/src/ggml-vulkan/`
- `common/arg.cpp`
- `common/speculative.cpp`
- `tools/server/server-context.cpp`

当前分支已包含的同源 upstream 提交：

- `2cdae802e`：Strix Halo Vulkan batched mat-vec
- `daef7b687`：Qwen 大词表 radix TOP_K
- `0b5be7e4a`：RDNA3 MMQ 调优
- `662a0b012`：DFlash encoder/KV injection 融合

Pinned engine 的相关提交：

- `61f2f2d7b`：避免 TurboQuant FlashAttention 全缓存 staging
- `e391458a1`：RDNA3.5 MMQ workgroup geometry
- `8aee61eca`：`--spec-mtp-strict-qwen`
- `0ef57fb81`：empty `data_spec` checkpoint restore
- `725c725d6`：two-scale ROCmFPX MMQ dispatch
- `e3cee8264`：ROCmFPX Vulkan `MUL_MAT_ID` geometry
