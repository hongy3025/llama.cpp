# q38rocm ROCmFPX 合并报告

日期：2026-09-06  
分支：`feat/q38rocm`

## 目标与边界

将 q38rocm 相对原版 llama.cpp 的 engine 修改合并到从 `hongy_main`
创建的 `feat/q38rocm`，保留原 `feat/rocmfpx` 的状态，并在 ROCm0 上以指定的
Qwen3.8 target 与独立 NextN MTP draft 做实际服务验证。

本次只合并 q38rocm 的 engine 源码。包装层的部署/benchmark 脚本、NPU pipeline、
以及会修改 DPM、THP、GTT/TTM 的主机级调优没有带入本仓库或自动执行。

## 提交与来源

| 项目 | 值 |
| --- | --- |
| 基线 | `10974376c` (`hongy_main`) |
| q38rocm merge 源仓库 | <https://github.com/julianmb/q38rocm> |
| q38rocm engine 固定源 | `75e67a92b2d230849aec2d6c1f7b1d1fd624e0e0` |
| 共同 merge-base | `8e53fcefd2c01ff70434ab41866bfc2eca31fe90` |
| 合并提交 | `a6c89be50` `merge: integrate q38rocm ROCmFPX engine` |
| 服务 profile | `7bb60641f` `scripts: use q38rocm dense Qwen MTP profile` |
| 响应解析修正 | `924fd44f4` `scripts: parse disabled reasoning traces` |
| 保留的原分支 | `feat/rocmfpx` @ `7dadb66da` |

本次 merge 的上游仓库为 [julianmb/q38rocm](https://github.com/julianmb/q38rocm)；
实际合入的是该仓库 `engine/src` 对应的固定 engine source commit，而不是 wrapper
根目录中的部署脚本、NPU pipeline 或主机调优脚本。

q38rocm engine 以本地 remote `q38rocm-engine` 固定到上述源提交后合并。合并相对
`hongy_main` 覆盖 485 个文件；报告只列出影响本目标的类别，而不是重复完整 diff。

## 合并内容

### ROCmFPX 与后端

- ROCmFPX quant 格式、tensor/格式注册、参考路径以及 HIP 加速路径。
- ROCmFPX plugin/sidecar 相关的构建和运行时接线。
- 面向 gfx1151 的 HIP 构建配置与 ROCmFPX kernel 集成。
- Qwen 模型加载中的 PLE tensor metadata 读取和严格 range 验证。
- Qwen MTP 的边界安全严格验证选项，见[风险与严格 MTP](#风险与严格-mtp)。

### 冲突处理

| 文件 | 保留的最终行为 |
| --- | --- |
| `.gitignore` | 同时保留本仓库的 `/saves/` 和 q38rocm 的 benchmark/local-helper 忽略项。 |
| `src/CMakeLists.txt` | 同时编译 GGSD 源文件和 `rocmfpx-plugin.cpp`。 |
| `src/models/qwen4exp.cpp` | 使用实际 PLE tensor metadata，并校验 head range；移除合并造成的重复 `ple_name`。 |
| `tests/test-llama-archs.cpp` | 使用紧凑 Q3 PLE fixture；对 Meta 的 typed PLE metadata 不支持路径跳过 split。 |

## 运行 profile

`start-server.sh` 是本次实际运行的唯一 launcher。调用时不附加额外参数，使用：

```text
MODEL=/models/qwen3.8/Qwen3.8-27B-Q4_0_ROCMFP4_STRIX.gguf
DRAFT=/models/qwen3.8/mtp-Qwen3.8-27B-Q4_0.gguf
```

| 模型 | 大小 |
| --- | ---: |
| target | 14,760,069,472 bytes |
| MTP draft | 1,680,271,648 bytes |

launcher 导出以下运行时环境，而不改变机器级设置：

```bash
HSA_OVERRIDE_GFX_VERSION=11.5.1
GGML_HIP_ENABLE_UNIFIED_MEMORY=1
ROCM_FLUSH_ACCEPT=1
```

完整的 llama-server 参数如下；末尾的 `"$@"` 只用于调用方显式追加参数，本报告的测试未
追加任何参数。

```text
-m "$MODEL" -md "$DRAFT"
-dev ROCm0 --spec-draft-device ROCm0
--n-gpu-layers all --spec-draft-ngl all
--host "$HOST" --port "$PORT" -np 1 --ctx-size 262144
-b 512 -ub 512 -t 16 -tb 32 --poll 100
--flash-attn on --mmap
--cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 8192 --ctx-checkpoints 0
--jinja --reasoning off --reasoning-format deepseek --reasoning-budget -1
--no-context-shift --no-mmproj
--temp 0 --top-p 0.95 --top-k 20 --seed 123
--spec-type draft-mtp
--spec-draft-type-k q4_0 --spec-draft-type-v q4_0
--spec-draft-n-max 4 --spec-draft-n-min 0
--spec-draft-p-min 0.75 --spec-draft-p-split 0.10
--spec-draft-threads 16 --spec-draft-threads-batch 32
--no-spec-draft-backend-sampling
--spec-draft-poll 1 --spec-draft-poll-batch 1
--metrics
```

这不是 q38rocm wrapper 的 embedded-MTP `FAST`/TurboQuant profile。指定 target 是
`ROCMFP4_STRIX`，draft 是独立的 MTP GGUF；因此采用 capability profile 建议的 target
`q8_0/q8_0` KV 与 draft `q4_0/q4_0` KV、`n_max=4`、`p_min=0.75`、单 ROCm slot。

`--reasoning-format deepseek` 是一次 API 合规修正。`--reasoning-format none` 会将模型
产生的空 `<think>` 标记留在 `message.content`，使“仅返回 `PASS`”的测试失败；改为
`deepseek` 后会解析该标记，实际响应内容为精确的 `PASS`。

## 构建与服务验证

### 构建

```bash
./build_rocm.sh
```

构建完成，产物为 `build-rocm/bin/llama-server`。二进制报告：

```text
version: 0.4.0-dev (build 11626, commit 7bb60641f)
ROCm0: AMD Radeon 8060S Graphics (122880 MiB, 54492 MiB free)
```

构建是 HIP-only、`AMDGPU_TARGETS=gfx1151` 配置；没有声称或测试 Vulkan 支持。

### 启动

服务由 `start-server.sh` 在端口 8080 启动。初始监督器的 ready 日志模式写成了
`server is listening`，而 llama-server 实际输出为 `listening on http`；这会把已可用的
服务误标为 `starting`。将 ready 模式改为实际日志并设置 20 秒上限后，最终一次实测：

```text
model loaded / listening on http://0.0.0.0:8080: 5.018 s
```

启动日志确认了指定 target、独立 MTP draft、一个 slot 和 `n_ctx_slot = 262144`。

### 正确性请求

```json
{
  "messages": [{"role": "user", "content": "Reply with exactly PASS and nothing else."}],
  "temperature": 0,
  "max_tokens": 8,
  "stream": false
}
```

| 字段 | 实测值 |
| --- | --- |
| HTTP status | 200 |
| finish reason | `stop` |
| content | `PASS` |
| prompt / completion tokens | 20 / 2 |
| prompt speed | 102.85 tok/s |
| generation speed | 9.13 tok/s |
| draft generated / accepted | 4 / 4 |

这证明了该确定性 API 请求正常完成并满足精确内容约束；它不是 MTP 与 no-spec
逐 token 等价的证明。

### non-strict MTP 吞吐基线

固定请求要求模型以单空格持续生成递增整数，`temperature=0`、`top_p=0.95`、
`top_k=20`、`max_tokens=512`、不流式返回。响应以长度上限结束，数字序列保持递增。

| 字段 | 实测值 |
| --- | ---: |
| HTTP status / finish reason | 200 / `length` |
| prompt / completion / total tokens | 37 / 512 / 549 |
| cached prompt tokens | 0 |
| prompt 时间 / 速度 | 180.432 ms / 205.06 tok/s |
| generation 时间 / 速度 | 11,076.337 ms / **46.13 tok/s** |
| 请求 wall time | 11.289 s |
| drafted / accepted tokens | 410 / 408 |
| draft acceptance | **99.512%** |
| mean draft length | 4.96 |

这是 **ROCm0、STRIX target、外置 MTP、non-strict MTP** 的单次可复现基线。它不能与
q38rocm 文档中的不同模型、embedded MTP、不同 KV 格式或 Vulkan 测量直接比较，也没有
no-MTP 对照，因此不声明 MTP 加速倍数。

## 风险与严格 MTP

### 风险含义

普通 `draft-mtp` 并非“直接相信 draft”。draft token 会经过 target 验证，首个失配之后的
draft 会被拒绝。不过，对 Qwen3.5/Qwen3.5MoE 的 recurrent MTP，通用路径没有保证下列
状态与关闭 speculative 后的逐 token greedy 路径完全等价：

- padded KV block 边界；
- MTP recurrent state 的 rollback；
- prompt cache 的精确命中。

因此即使 `temperature=0`，non-strict MTP 也可能在这些边界生成与 no-MTP 不同的 token
序列。这是确定性/审计风险，不是“draft 未经 target 验证”或已观察到的生成错误。

启动时服务明确报告：

```text
Qwen MTP strict verification is disabled; greedy output may diverge from no-spec decoding
```

### 合入前与当前默认行为

`hongy_main` 在合入前已有通用 `draft-mtp`，但没有 `--spec-mtp-strict-qwen`、
`mtp_strict_qwen` 状态字段或对应的 Qwen boundary/rollback 逻辑。合并新增了：

- 参数定义和默认 `false` 状态；
- 一 slot/sequence、目标架构和完整 rollback 能力的启动期检查；
- 单 padded KV block 内的 draft 长度限制；
- prompt-cache 精确命中时的 cold fallback，以重建 MTP 边界状态。

所以当前 launcher 未加 strict flag 时延续的是合入前的普通 MTP 语义；合入的价值是提供了
一个此前不存在的 exact-greedy 保护模式并暴露风险，而不是把先前严格模式降级。

### 规避方案

对要求 no-spec greedy 等价的服务，应把：

```text
--spec-mtp-strict-qwen
```

加入 launcher，并把吞吐数字重新标记为 **strict MTP baseline**。该模式的实现契约是
“boundary-safe verification for exact greedy Qwen35/Qwen35MoE MTP output”。它必须使用
`-np 1`；当前 launcher 已满足。启动还会验证目标具备覆盖完整 `n_max` draft 的 bounded
recurrent rollback，不满足则拒绝启动。

严格模式的代价是 KV block 边界处可能缩短或关闭 draft，且 prompt-cache 精确命中会重建
MTP 状态；吞吐可能低于本报告的 non-strict 数字。

推荐的升级验证顺序：

1. 以相同模型、`temp=0` 和固定 seed 分别运行 no-MTP、strict MTP、non-strict MTP。
2. 比较 no-MTP 与 strict MTP 的完整 token 序列，而不是只比较渲染文本。
3. 覆盖 256-token KV block 边界、长生成以及 prompt-cache 命中请求。
4. 仅在 strict MTP 与 no-MTP 完全一致后，将 strict profile 用于确定性/审计流量。
5. non-strict MTP 仅作为明确标注的吞吐优先模式；如果严格模式不可用或成本不能接受，对
   确定性请求使用 `--spec-type none`。

## 当前状态

- `feat/q38rocm` 包含合并、launcher profile 和响应解析修正。
- 原 `feat/rocmfpx` 分支已保留。
- 已完成 HIP Release 构建、真实模型启动、精确响应检查和 512-token non-strict MTP 基线。
- 服务已停止；报告未将 non-strict 吞吐表述为 strict/no-spec 等价结果。
