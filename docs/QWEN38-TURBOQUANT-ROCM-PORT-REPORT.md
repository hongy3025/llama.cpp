# Qwen3.8 TurboQuant ROCm 移植报告

分支：`feat/turboquant`

## 1. 项目目标

本分支将 Qwen3.8-27B 的 ROCmFP4 Strix 权重、独立 MTP draft、ROCm0 GPU 推理、GPU KV-cache 和 TurboQuant FlashAttention 路径接通，并将运行 profile 固化到 `start-server.sh`。

目标运行组合：

| 项目 | 配置 |
| --- | --- |
| Target 权重 | `/models/qwen3.8/Qwen3.8-27B-Q4_0_ROCMFP4_STRIX.gguf` |
| Draft 权重 | `/models/qwen3.8/mtp-Qwen3.8-27B-Q4_0.gguf` |
| 后端 | ROCm0 / AMD Radeon 8060S Graphics |
| 模型权重 | ROCmFP4，保持原始 packed 权重，不转换为 F16 或 CPU 路径 |
| Target KV | K=`q8_0`，V=`turbo4` |
| Draft KV | K=`q8_0`，V=`turbo4` |
| MTP | `draft-mtp`，严格 Qwen 边界验证 |
| FlashAttention | HIP packed Turbo VEC，eligible batch 直接执行 |

本次工作不修改机器级 DPM、THP、GTT 或 TTM 设置，也不进行远程提交、推送或其他 remote 操作。

## 2. 迁移内容

### 2.1 ROCmFP4 模型加载

- 修正 ROCmFP4 ftype 映射，使 `Q4_0_ROCMFP4` 和相关权重类型被正确识别。
- 保留未知类型告警，避免将未知 GGUF 类型静默映射为错误的模型格式。
- 保持目标模型的 ROCmFP4 / ROCmFP4 fast packed 权重，不引入运行时权重重排替代方案。
- 最终服务日志确认目标模型 `65/65` 层、draft 模型 `66/66` 层全部 offload 到 ROCm0。

### 2.2 GPU KV-cache 放置

`src/llama-kv-cache.cpp` 根据实际模型层设备选择 KV buffer type 和 device。这样 Turbo KV-cache 不再退回 CPU buffer，而是跟随对应 layer 放置在 ROCm0。

最终目标服务的显存放置：

| Buffer | 大小 |
| --- | ---: |
| Target ROCm0 model | `13071.19 MiB` |
| Target ROCm0 KV | `6656.00 MiB` |
| Target K q8_0 | `4352.00 MiB` |
| Target V turbo4 | `2304.00 MiB` |
| Target recurrent state | `748.12 MiB` |
| Target compute | `122.00 MiB` |
| Draft ROCm0 model | `909.96 MiB` |
| Draft ROCm0 KV | `416.00 MiB` |
| Draft compute | `142.00 MiB` |

### 2.3 TurboQuant SET_ROWS

`GGML_OP_SET_ROWS` 增加 Turbo3/Turbo4 GPU 支持：

- 每个 128-element chunk 使用 128-thread block。
- 计算 L2 norm 并进行归一化。
- 执行归一化 128 点 FWHT。
- 使用 Turbo3/Turbo4 codebook 查找量化 index。
- 直接写入 packed Turbo block。
- Turbo4 nearest-codebook 搜索使用有序 binary decision tree，减少每个元素的比较次数。

该路径输出的 layout 与 HIP FlashAttention 的 Turbo K/V 消费 layout 一致。

### 2.4 HIP packed FlashAttention

增加并编译 HIP mixed Turbo VEC template instances，覆盖：

- Turbo3/Turbo3
- Turbo3/Turbo4
- Turbo4/Turbo3
- Turbo4/Turbo4
- q8_0/Turbo3
- q8_0/Turbo4
- Turbo3/q8_0
- Turbo4/q8_0

Turbo VEC eligible 条件包括：

- K/V pair 有对应的 HIP template。
- head dimension 为 128 或 256。
- Q、K、V 的 head dimension 匹配。
- K 的 sequence stride 满足 FlashAttention stride 要求。
- element stride 和输出 contiguous 条件满足 kernel 假设。

满足条件时，所有 batch 宽度都直接走 packed Turbo VEC，不再因为 prompt batch 宽度较大而进入未处理 packed 数据的 MMA 路径。

### 2.5 FWHT 变换边界

TurboQuant KV 的编码空间为归一化 FWHT 空间：

1. Turbo-K 进入 FlashAttention 前，对 Q 执行一次归一化 FWHT。
2. FlashAttention 在 Turbo-K 空间中完成 K·Q。
3. Turbo-V 的加权结果离开 FlashAttention 后执行一次逆 FWHT。
4. 128 点归一化 FWHT 是自逆变换。

Q 的旋转只由 VEC FlashAttention 内部负责一次；外层 dispatch 不重复旋转 Q。此前的重复旋转会导致整数序列出现 `1 2 3 3 4 ...` 等错误。

### 2.6 严格 Qwen MTP

`start-server.sh` 固定启用：

```text
--spec-type draft-mtp
--spec-mtp-strict-qwen
```

严格模式使用边界安全验证和有界 recurrent rollback，保证本次确定性 greedy 请求的输出序列保持正确。

## 3. 性能问题定位

### 3.1 对照配置

使用相同模型、draft、batch、context、采样参数和严格 MTP 配置进行 q8/q8 对照。

q8/q8 实测基线：

| 指标 | q8/q8 |
| --- | ---: |
| Prompt | `116.27 tok/s` |
| Generation | `46.08 tok/s` |
| Prompt tokens | `23` |
| MTP accepted/generated | `405/409` |

Turbo4 初始实现的真实服务测量曾出现：

| 指标 | 初始 q8/Turbo4 |
| --- | ---: |
| Prompt | `90.00 tok/s` |
| Generation | `46.14 tok/s` |
| Prompt tokens | `23` |

因此主要异常集中在 23-token prompt 阶段，而不是 MTP decode 主循环。

### 3.2 ROCm kernel profile

使用 `rocprofv3` 采集 HIP kernel dispatch 统计。q8/q8 与初始 q8/Turbo4 的 FlashAttention kernel 对比如下：

| Kernel | q8/q8 | Turbo4/q8 | 增量 |
| --- | ---: | ---: | ---: |
| FlashAttention VEC 全部变体 | `117.93 ms` | `140.14 ms` | `+22.21 ms` |
| prompt 单列变体 | `9.49 ms` | `14.99 ms` | `+5.50 ms` |

混合 KV profile 将原因拆分为两部分：

| KV 组合 | FlashAttention VEC 时间 |
| --- | ---: |
| q8-K / q8-V | `117.93 ms` |
| q8-K / Turbo4-V | `124.52 ms` |
| Turbo4-K / q8-V | `126.27 ms` |

`k_set_rows_turbo4` 约 `8.5 ms`，Turbo4 逆 FWHT 约 `6.6 ms`，各自只占总 kernel 时间约百分之零点零几，不是主要瓶颈。

### 3.3 根因

Turbo4 VEC 的旧实现与 q8 VEC 的硬件友好路径不对称：

- Turbo4 K dot 每个 pair 调用一次标量解码函数。
- 同一个 Turbo4 block 的 FP16 norm 被重复读取和转换。
- packed nibble 没有在 block 级别复用。
- Turbo4 V 解码逐 pair 产生临时 `float2`，随后逐标量写入目标临时数组。
- q8 路径可以连续读取量化值，并使用成熟的 half2 / q8 dot 处理。

因此 Turbo4 的压缩存储节省了 KV 显存，但旧 VEC 解码开销抵消了部分计算收益，尤其在 prompt 的短 sequence、低算术强度场景中更明显。

## 4. 性能优化

`ggml/src/ggml-cuda/fattn-common.cuh` 做了以下优化：

- Turbo4 K dot 按 block 计算并复用 norm。
- 直接读取 packed byte，拆分两个 nibble 并查 Turbo4 codebook。
- Turbo4 V 对 block-aligned、四元素访问走快速路径。
- V norm 每次 block 只转换一次。
- V 输出使用 `half2` 或 `float2` 写入。
- 保留跨 block 的通用路径，避免破坏其他调用者的边界行为。

优化后的 profile 中，Turbo4/q8 FlashAttention 从约 `140.14 ms` 降至约 `132.50 ms`。

## 5. 最终服务结果

构建命令：

```bash
cmake --build build-rocm --target llama-server -j 16
```

最终服务由以下 launcher 启动：

```bash
HOST=127.0.0.1 PORT=8096 ./start-server.sh -lv 4
```

运行时环境：

```text
HSA_OVERRIDE_GFX_VERSION=11.5.1
GGML_HIP_ENABLE_UNIFIED_MEMORY=1
ROCM_FLUSH_ACCEPT=1
```

固定请求：

```bash
curl -sS --max-time 180 http://127.0.0.1:8096/completion \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt":"只输出从 1 开始、以单个空格分隔的递增整数。不要解释，不要换行。",
    "n_predict":512,
    "temperature":0,
    "top_p":0.95,
    "top_k":20,
    "seed":123,
    "cache_prompt":false
  }'
```

最终一次重建后的真实服务结果：

| 指标 | 最终值 |
| --- | ---: |
| Prompt | `200.87 ms / 23 tokens`，`114.50 tok/s` |
| Generation | `11000.76 ms / 512 tokens`，`46.45 tok/s` |
| Total | `11201.63 ms / 535 tokens` |
| Graphs reused | `97` |
| MTP accepted/generated | `406/408` |
| Draft acceptance | `99.510%` |
| Mean draft length | `4.90` |

优化后的两次服务测量中，prompt 速度为 `114.50–117.30 tok/s`，与 q8/q8 的 `116.27 tok/s` 同一量级；generation 为 `45.27–46.45 tok/s`，达到并超过要求的 `45 tok/s` 基线。

响应内容保持正确递增整数序列，512 个预测 token 对应输出 `1` 到 `155`，没有出现 NaN、重复数字或非整数污染。

## 6. 验证结果

已执行：

```bash
./build-rocm-tests/bin/test-turboquant
```

结果：

```text
7/7 tests passed
```

验证覆盖：

- FWHT 自逆误差。
- Turbo3/Turbo4 round-trip 误差。
- Turbo3/Turbo4 pack determinism。
- Turbo3/Turbo4 dequantize finite/nonzero。
- Turbo4-K/q8-V 短请求输出 `1..8`。
- Turbo4-K/Turbo4-V 短请求输出 `1..8`。
- q8-K/Turbo4-V 完整 512-token 请求输出正确。
- 严格 Qwen MTP 启动告警和 accepted/generated 统计。
- Target/draft 全层 GPU 放置。
- `git diff --check` 通过。

所有测试服务均已停止，端口 `8096` 最终检查为未监听。

## 7. 运行边界与后续维护

- 原始目标配置命中 packed Turbo VEC；不会因为目标 shape eligible 而转换到 CPU、q8 KV 或 F16 KV。
- 非 eligible shape 保留安全兼容路径，但不作为本目标 profile 的执行路径。
- 当前 HIP toolchain 不提供 `__float22half2`，Turbo4 K dot 保持标量 float 累加实现，避免引入不兼容 intrinsic。
- `HSA_OVERRIDE_GFX_VERSION=11.5.1`、统一内存开关和 `ROCM_FLUSH_ACCEPT=1` 是当前 Strix Halo 实测 profile 的运行环境组成部分。
- `.rocprofv3/` 为本地 profiler 产物，已加入 `.gitignore`，不会进入提交。
- 不提交 `/tmp/rocprof-*` 等临时 profile 输出，也不保留临时 device `printf` 诊断。
