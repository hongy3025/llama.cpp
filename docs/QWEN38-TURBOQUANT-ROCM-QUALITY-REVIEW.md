# Qwen3.8 TurboQuant ROCm 质量审查

## 1. 范围与结论

审查对象：当前分支 `feat/turboquant`，HEAD `c1a31668abbf569e4e57671f5fc180a3c0cb2568`。

直接基线：`feat/q38rocm`，提交 `e800ffc7242ba8ede13e08a18839d2d9e5a4027c`。当前分支相对直接基线包含 3 个提交、14 个文件、665 行新增和 93 行删除：

```text
c99776263 feat: complete TurboQuant KV cache integration
ff73bd1da test: use TurboQuant KV in server launcher
c1a31668a feat: optimize Qwen3.8 TurboQuant ROCm path
```

审查清单：scope/quick-reject、security、general、ggml/backend、server、approach/design。审查只新增本报告，没有修改被审查的实现。

**最终验收：拒绝合入。** 本结论不要求 TurboQuant 提速：该功能的首要目标是节省 KV cache 内存，速度与 q8/q8 持平即可通过性能验收，性能不提升不构成拒绝理由。当前拒绝结论仅来自下列正确性、兼容性和验证阻塞：

1. 非 eligible FlashAttention fallback 的 Turbo K/V 数值错误，NMSE 约 `1.85-1.95`，测试为 `0/4`。
2. CPU/no-KV-offload 的 Turbo `SET_ROWS` 调用空 `from_float` 函数指针，进程以 `SIGSEGV` 退出，退出码 `139`。
3. ROCm `SET_ROWS` support gate 接受 `ne00=96`，执行 kernel 却要求 `ne00 % 128 == 0` 并触发断言。
4. Q2 ROCmFPX 的既有 model-loader ftype 映射被删除，构成兼容性回归。
5. 正式 GPU 测试矩阵没有 Turbo case，已有移植报告把 CPU reference 测试描述成覆盖 GPU、服务、MTP 和 FlashAttention，验收证据不成立。

按此验收原则，正向结果应明确计入：

- **内存目标通过**：服务总 KV cache 从 `9248 MiB` 降至 `7072 MiB`，节省 `2176 MiB`，即 `23.529%`。
- **在线 generation 速度通过**：Turbo 中位数 `45.9352 tok/s`，q8/q8 中位数 `45.4298 tok/s`，差异 `+1.1125%`，视为持平。
- **不要求单 kernel 加速**：MTP acceptance 不同只表示不能把 `+1.1125%` 宣称为 Turbo kernel 加速，不影响“内存下降且服务速度持平”的验收。
- review-only `llama-bench` 的 512/2048-token prompt 分别回退 `11.18%`/`13.52%`，不属于持平，但在本报告中只作为非阻塞性能风险，不参与拒绝合入结论。

## 2. 环境与可复现对象

硬件和软件：

```text
CPU: AMD RYZEN AI MAX+ 395 w/ Radeon 8060S
GPU: AMD Radeon 8060S Graphics, gfx1151, 40 CU, wavefront 32
统一显存池: rocminfo 约 122880 MiB
HIP: 7.15.26333-0000000
AMD clang: 23.0.0git
模型: /models/qwen3.8/Qwen3.8-27B-Q4_0_ROCMFP4_STRIX.gguf
模型大小: 14749075456 bytes
MTP: /models/qwen3.8/mtp-Qwen3.8-27B-Q4_0.gguf
```

HEAD 构建配置：

```text
build type: Release
GGML_HIP=ON
GGML_CUDA_FA_ALL_QUANTS=ON
GGML_CUDA_GRAPHS=ON
LLAMA_BUILD_TESTS=OFF (build-rocm)
LLAMA_BUILD_TESTS=ON  (build-rocm-tests)
shared libraries: ON
```

构建命令和版本：

```bash
cmake --build build-rocm --target llama-server -j16
cmake --build build-rocm-tests --target test-backend-ops test-turboquant -j16
cmake --build build-rocm --target llama-bench -j16
./build-rocm/bin/llama-server --version
```

构建成功；版本输出为：

```text
version: 0.4.0-dev (build 11632, commit c1a31668a)
```

## 3. Quick reject / scope gate

### 3.1 新增 ggml type 的维护门槛未满足

`GGML_TYPE_TURBO3_0=105` 和 `GGML_TYPE_TURBO4_0=106` 位于 `ggml/include/ggml.h` 的下游保留区。新增量化类型同时改动 CPU traits、CUDA/HIP dequant、FlashAttention、KV cache、SET_ROWS 和 server launcher，维护面远大于单一模型适配。

本次变更和现有移植报告没有提供完整的维护门槛证据：GGUF 样本上传、相对 FP16/BF16 和相近大小量化的困惑度、KL divergence、CPU 性能以及长期维护归属。新类型不应在这些材料缺失时进入通用 ggml 类型集合。

### 3.2 首次支持同时修改多个后端

变更同时触及 `ggml-cpu`、CUDA/HIP 共用代码和 `ggml-hip`。项目贡献规则对首次 backend 支持建议先做 CPU-only，再以独立变更添加其他后端；当前提交把跨后端类型接通、GPU kernel、服务默认配置和优化混在一起，增加了审查和回归隔离成本。

### 3.3 缺少关联 issue/discussion 证据

在本次审查的提交信息和变更报告中没有看到关联 issue 或设计讨论链接。对新增量化类型、跨后端 FlashAttention 和默认 launcher 行为，这不满足先讨论需求和维护责任的 quick-reject 要求。

## 4. Blocking findings

### B1. 非 eligible fallback 破坏 TurboQuant 元素布局，FlashAttention 数值错误

位置：

- `ggml/src/ggml-cuda/fattn.cu:595-619`
- `ggml/src/ggml-cuda/convert.cu:28-39,896-899`
- `ggml/src/ggml-cuda/dequantize.cuh:308-354`

`ggml_cuda_fattn_pre_dequantize()` 使用 `ggml_get_to_fp16_nc_cuda()` 将 Turbo K/V 转成 F16。该 converter 把 Turbo3/4 映射到通用 `dequantize_block_cuda<..., QR=2, ...>`。通用 converter 对 `qr != 1` 使用：

```cpp
const int64_t iqs = (i00%qk)/qr;
const int64_t y_offset = qr == 1 ? 1 : qk/2;
y[iy0 + 0]        = v.x;
y[iy0 + y_offset] = v.y;
```

但 Turbo decoder 的 `v.x`、`v.y` 是相邻逻辑元素，即 `(2*iqs, 2*iqs+1)`。因此通用 converter 对每个 32-element block 写出类似：

```text
x0, x2, x4, ..., x30, x1, x3, ..., x31
```

后续 Turbo FWHT 和普通 FlashAttention 读取的是原始连续布局，无法恢复该排列。

复现实测：

```bash
/tmp/test-backend-ops-turbo-isolation test -b ROCm0 -o FLASH_ATTN_EXT -p 'kv=113.*type_V=turbo'
```

结果：`0/4 tests passed`。`kv=113` 强制走非 eligible fallback，4 个 case 的 NMSE 为：

```text
Turbo4/Turbo4, D128: 1.867570358
q8_0/Turbo4, D256: 1.954714780
q8_0/Turbo4, D128: 1.864138120
q8_0/Turbo3, D128: 1.850922169
```

阈值为 `0.0005`。这不是量化误差，而是输出布局错误。正常 profile 的 K/V 行数经常是 256 对齐，不能掩盖公共 fallback 对非对齐 KV、view 或其他合法 shape 的错误。

修复要求：增加 Turbo-specific NC converter，按 decoder 的相邻元素顺序写回并正确处理所有 strides；或者复用一个能完成正确 Turbo row decode 和逆 FWHT 的路径，再删除当前错误的通用 converter 映射。修复后必须覆盖 Turbo3/Turbo4、对称和混合 K/V、q8/Turbo、D128/D256、非 256 倍数 KV、permuted/view 以及 mask/sink case，并在正式 `test-backend-ops` 中保留这些 case。

### B2. CPU `SET_ROWS` 对 Turbo 调用空函数指针，触发 SIGSEGV

位置：

- `ggml/src/ggml-cpu/ggml-cpu.c:525-533`
- `ggml/src/ggml-cpu/ggml-cpu.cpp:443-452`
- `ggml/src/ggml-cpu/ops.cpp:5288,5301-5304`
- `common/arg.cpp:305-316`

Turbo3/4 的 CPU traits 只设置了 `.vec_dot` 和 `.vec_dot_type`，没有 `.from_float`。CPU `supports_op()` 的 `CPY/SET_ROWS` gate 只排除了若干 IQ 类型，没有排除 Turbo，因此返回 supported。执行端随后直接取并调用：

```cpp
ggml_from_float_t const from_float = ggml_get_type_traits_cpu(dst->type)->from_float;
from_float(...);
```

Turbo cache type 已通过公共 CLI 暴露；因此 CPU-only 或 `--no-kv-offload` 不再是不可达内部状态。

复现实测：

```bash
/tmp/test-backend-ops-turbo-review test -b CPU -o SET_ROWS -p 'type_src=f32,type_dst=turbo3'
```

结果：退出码 `139`。同一个 case 在 `ROCm0` 也以 `139` 退出；instrumentation 显示 tensor 初始化完成后才崩溃，和空 `from_float` 路径一致。

修复要求二选一：

- 实现安全的 CPU Turbo `from_float`，每行按独立 128-element chunk 执行 FWHT、归一化和打包；或
- 在 CPU capability gate 明确拒绝 Turbo `SET_ROWS`，并在 context/CLI 创建阶段给出清晰错误。

无论采用哪种方案，都不能让 `supports_op=true` 与空函数指针共存。必须复验 CPU-only 和 `--no-kv-offload`。

### B3. `SET_ROWS` support contract 接受非法 Turbo row shape

位置：

- `ggml/src/ggml-cuda/ggml-cuda.cu:5335-5348`
- `ggml/src/ggml-cuda/set-rows.cu:457-467,496-506`

CUDA/HIP support gate 只检查类型和输入类型，没有检查 Turbo kernel 的 128-element chunk 前置条件。实际 Turbo3/Turbo4 launch 都有：

```cpp
GGML_ASSERT(ne00 % TURBO_HEAD_DIM_SR == 0);
```

其中 `TURBO_HEAD_DIM_SR == 128`。

复现实测：

```bash
/tmp/test-backend-ops-turbo-isolation support -b ROCm0 -o SET_ROWS -p 'type_dst=turbo4.*ne=\[96'
```

结果：

```text
SET_ROWS(type_src=f32,type_dst=turbo4,...,ne=[96,3,1,1],...): SUPPORTED
```

该 shape 后续执行会触发 `GGML_ASSERT(ne00 % 128 == 0)`。这是 capability contract 与执行前置条件不一致，外部 graph 选择 backend 后会被进程终止。

修复要求：support gate 和实际 kernel 必须采用同一契约，至少要求 `ne00 % 128 == 0`；如果要接受任意 32 的倍数，必须提供真正正确的 fallback，而不是在执行阶段 assert。CPU gate 也必须和最终支持范围一致。

### B4. Q2 ROCmFPX ftype 映射回归

位置：`src/llama-model-loader.cpp:769-804`。

直接基线已有：

```cpp
case GGML_TYPE_Q2_0_ROCMFPX:
    ftype = LLAMA_FTYPE_MOSTLY_Q2_0_ROCMFPX;
    break;
```

当前分支在加入 Q4 ROCmFP4/FAST 映射时删除了该 case。没有 `general.file_type` metadata 时，`type_max == GGML_TYPE_Q2_0_ROCMFPX` 会落入 `default`，记录 `unknown type` 并猜测为 `LLAMA_FTYPE_ALL_F32`。这会让已有 Q2 ROCmFPX GGUF 的类型识别退化，并可能影响后续量化兼容性和显示信息。

修复要求：恢复基线映射，并增加一个不依赖 `general.file_type` 的 Q2 ROCmFPX loader 验证。当前未把没有该 metadata 的 Q2 GGUF 误报成已实测；这是基于基线 diff 的静态回归判断。

### B5. 正式测试和已有验收报告不能证明 GPU Turbo 正确性

位置：

- `tests/test-backend-ops.cpp:8645-8669`
- `tests/test-backend-ops.cpp:10374-10517`
- `tests/test-turboquant.cpp:1-15`
- `docs/QWEN38-TURBOQUANT-ROCM-PORT-REPORT.md:232-257`

正式 `all_types[]` 没有 `GGML_TYPE_TURBO3_0` 或 `GGML_TYPE_TURBO4_0`，正式 FlashAttention case 也没有 Turbo pair。`test-turboquant.cpp` 是 15 行压缩的 CPU reference 测试，只覆盖 FWHT 自逆、CPU round-trip、确定性 pack 和 finite/nonzero。

正式测试命令：

```bash
./build-rocm-tests/bin/test-turboquant
```

结果：

```text
7/7 tests passed
```

但该结果不执行 GPU kernel、backend dispatch、GPU SET_ROWS、服务、MTP、offload 或 FlashAttention。现有移植报告却在“验证覆盖”中列出 GPU 放置、MTP accepted/generated、服务输出和多个 FlashAttention case，并把这些内容与 `7/7` 放在同一验证结论下。源码和命令不支持这些覆盖声明。

为取得证据，本次审查在 `/tmp` 编译了 review-only backend harness，基于当前 `test-backend-ops.cpp` 补入 Turbo case；没有修改仓库测试。当前 HEAD 上 eligible case 的实测为：

```bash
/tmp/test-backend-ops-turbo-review test -b ROCm0 -o FLASH_ATTN_EXT -p 'type_V=turbo'
```

结果：`10/10 tests passed`，包括 Turbo3/4 对称和混合、q8/Turbo、D128/D256、GQA、permuted KV、batch、KV 1280、mask、sink 和 logit softcap。这个结果只证明 eligible VEC 路径的所选 case，不能替代正式测试，也不能抵消 B1 fallback 失败。

修复要求：将相同的 observable cases 正式放入 `test-backend-ops.cpp`，并保留非 eligible fallback 和非法 shape case；更新测试结果说明，使报告中的命令、源码覆盖和结论一一对应。

## 5. Will slow the review

### W1. `llama-bench` 无法解析 Turbo type，性能验收必须使用临时工具

位置：`tools/llama-bench/llama-bench.cpp:494-520`。

该文件有自己的 `ggml_type_from_name()` 白名单，只包含 `f16`、`bf16`、q4/q5/q8 和 `iq4_nl`，没有 `turbo3`/`turbo4`。因此原始正式工具命令：

```bash
./build-rocm/bin/llama-bench \
  -m /models/qwen3.8/Qwen3.8-27B-Q4_0_ROCMFP4_STRIX.gguf \
  -ngl 999 -fa on -ctk q8_0 -ctv q8_0,turbo4 \
  -p 23,512,2048 -n 128 -b 512 -ub 512 -t 16 -r 5 --delay 1 -o json
```

直接返回：

```text
error: invalid parameter for argument: -ctv
```

本次只能用 `/tmp/llama-bench-turbo-review` 临时加入两项 parser 映射。长期修复应更新正式 benchmark parser，并为 `turbo3`、`turbo4` 保留可重复的 benchmark 记录，而不是依赖 review-only 工具。

### W2. eligible Turbo batch 无条件强制 VEC，声称的 bounded-slice TILE/MFMA 路径默认不可达

位置：

- `ggml/src/ggml-cuda/fattn.cu:775-805`
- `ggml/src/ggml-hip/fattn-kv-batched.cu:167-299`
- `ggml/src/ggml-cuda/fattn-common.cuh:1606-1643`

当前 dispatch 对所有满足 Turbo VEC eligibility 的 batch 返回 `BEST_FATTN_KERNEL_VEC`：

```cpp
if (ggml_cuda_fattn_turbo_vec_fused_eligible(dst) && can_use_vector_kernel) {
    return BEST_FATTN_KERNEL_VEC;
}
```

但 HIP 的 bounded-slice 实现通过 `launch_fattn(..., allow_kv_batching=true)` 才能调用 `ggml_cuda_fattn_kv_batched()`。因此默认 eligible Turbo prompt 不进入该实现；非 eligible Turbo 先被 F16 staging 后也不再满足该 helper 的 Turbo type 检查。该 299 行实现基本成为不可达或仅在另一路径偶然可达的维护负担。

本机同一 HEAD、同一模型、`-b 512 -ub 512 -r 5` 的 review-only benchmark（仅为 parser 临时加入 Turbo type，核心 benchmark 逻辑未改）显示，Turbo VEC 在长 prompt 上明显落后 q8/q8：

| workload | q8/q8 平均 tok/s | q8/Turbo4 平均 tok/s | 差异 |
| --- | ---: | ---: | ---: |
| prompt 23 | 129.697705 | 125.932209 | -2.90% |
| prompt 512 | 366.485672 | 325.512670 | -11.18% |
| prompt 2048 | 352.347713 | 304.727863 | -13.52% |
| generation 128 | 13.791509 | 13.782457 | -0.07% |

这是当前实现的非阻塞性能风险，不是纯 Turbo kernel 隔离结果；q8/Turbo4 还包含混合类型 dispatch 和转换成本。512/2048-token prompt 的 `-11.18%`/`-13.52%` 不能称为持平，也否定了“长 prompt 仍自动走更快 TILE/MMA”的注释性结论；但按照内存优先、速度持平即可的原则，此项不作为拒绝合入依据。

本次还构建了一个只修改 dispatch 的 review-only 变体，把 eligible batch 大于 8 路由到 TILE。该变体的 q8/Turbo4 结果为：

```text
prompt 512: 345.616176 tok/s, samples [331.946, 351.071, 348.262, 348.300, 348.503]
prompt 2048: 340.484912 tok/s, samples [343.223, 341.455, 339.560, 339.656, 338.531]
```

但同一变体的 Turbo backend harness 在原本 eligible 的 KV 1280 case 失败，说明不能只改 dispatch 就安全启用 bounded-slice 路径。应先修正该 helper 的类型、布局、mask/sink、view 和多批次合并契约，再以正式 case 和 perf 数据决定 VEC/TILE/MFMA 阈值。该问题保留为非阻塞性能和维护风险；最终拒绝结论不依赖此项。

### W3. 服务 A/B 达到内存目标，generation 速度持平；不应宣称单 kernel 加速

HEAD Turbo launcher 实测：

```bash
HOST=127.0.0.1 PORT=8096 ./start-server.sh -lv 4
```

target 和 draft 均完整 offload 到 ROCm0，MTP 初始化成功；target/draft 均为 K=q8、V=Turbo4。固定 512 `n_predict` 请求 3 次，输出均为严格递增整数 `1 ... 155`，没有重复、跳号、非整数或 NaN。

同一 HEAD、模型、draft、上下文、采样和 MTP 配置的 q8/q8 对照为：

| profile | target/draft KV | prompt 样本 tok/s | prompt 中位数 | generation 样本 tok/s | generation 中位数 |
| --- | --- | --- | ---: | --- | ---: |
| q8/q8 | 9248 MiB | 85.8715, 129.8547, 129.0438 | 129.0438 | 45.4298, 45.3066, 45.6143 | 45.4298 |
| q8/Turbo4 | 7072 MiB | 110.7553, 125.2635, 130.0853 | 125.2635 | 45.8374, 45.9352, 46.1172 | 45.9352 |

Turbo 节省 `2176 MiB`，即相对 q8/q8 的 `23.529%`；target 节省 `2048 MiB`，draft 节省 `128 MiB`，核心内存目标通过。服务 generation 中位数差异为 `+1.1125%`，按验收原则视为持平并通过；3 次短 prompt 的中位数差异为 `-2.9295%`，且首轮波动较大。MTP acceptance 不同：Turbo 为 `406/408`，q8/q8 为 `405/409`，因此不能把 generation 的 `+1.1125%` 归因于单个 FA kernel，但本目标也不要求 kernel 提速。这组性能结果不构成拒绝合入理由。

### W4. hidden boundary-layer 环境开关缺乏契约和专门测试

位置：`src/llama-kv-cache.cpp:25-37,185-188,264-267`。

`llama_env_u32()` 只检查 `strtoul()` 是否消费了零字符：

- 不检查 `errno`；
- 不拒绝尾随垃圾，例如 `12junk`；
- 负数经过 `strtoul` 后再转 `uint32_t`；
- 超过 `UINT32_MAX` 时直接截断；
- `il + turbo_boundary_layers` 是 `uint32_t` 加法，极端环境值可溢出并错误识别边界层。

这些变量还改变 cache 类型而没有正式 CLI schema、范围错误和测试。应使用带完整字符串、errno、上下界检查的 parser，并用无符号安全比较表达边界条件；为 K/V 独立开关添加专门测试，或删除未证明必要的隐藏控制面。

### W5. 生成文档和 ops 文档未同步

`tools/server/README.md:71-72,116-117` 和 `tools/cli/README.md:53-54,99-100` 仍列出旧的 cache type 集合，没有 `turbo3`/`turbo4`。本次改变了 `SET_ROWS` 和 FlashAttention backend 能力，但没有同步 `docs/ops.md` 及相关 ops 表格。用户可从 CLI 得到支持，却从生成帮助文档得到相反信息。

### W6. 测试文件被压缩成不可维护的单行实现，断言质量弱

位置：`tests/test-turboquant.cpp:1-15`。

整个 reference 测试被压成少量超长行，违反周围 4-space 风格并提高审查和后续修改成本。MSE 测试要求结果落在固定区间，可能把合法的算法改进判成失败；determinism 和 finite/nonzero 只提供弱信号，不能证明 GPU layout、shape、stride、fallback 或 SET_ROWS contract。应复用 `test-backend-ops`，保留少量真正防回归的 reference 边界测试，而不是以 `7/7` 作为 GPU 验收替代品。

## 6. Nits

1. `src/llama-kv-cache.cpp` 用 `<cstdlib>` 替换基线的 `<cstring>`，但 `:125` 仍直接使用 `strcmp`，`:1871` 和 `:1880` 使用 `memcpy`。应显式保留所需标准头，避免依赖传递 include。
2. `ggml/src/ggml-cuda/fattn.cu:408` 的注释使用 Unicode `K·Q`。代码和注释规范要求 ASCII，应写成 `K*Q` 或普通 ASCII 文字。
3. `ggml/src/ggml-hip/fattn-kv-batched.cu` 的 Turbo staging、FWHT、online softmax 合并逻辑较重；在其 dispatch 可达性和 correctness 没有正式证明前，不应继续扩展更多特殊分支。
4. `start-server.sh` 把 TurboQuant 作为默认 launcher 行为，意味着脚本消费者会自动进入尚有 fallback 和 CPU contract 缺陷的格式。至少应在默认化前完成 B1-B3 和正式 GPU 回归矩阵。

## 7. 原始验证记录

### 7.1 正式 CPU reference

```bash
./build-rocm-tests/bin/test-turboquant
```

```text
7/7 tests passed
```

覆盖内容仅为：FWHT self-inverse、Turbo3/Turbo4 CPU round-trip MSE、deterministic pack、finite/nonzero。

### 7.2 正式 backend 对照

```bash
./build-rocm-tests/bin/test-backend-ops test -b ROCm0 -o SET_ROWS -p 'type_dst=q8_0'
```

结果：ROCm q8_0 可执行矩阵 `13/13 tests passed`。正式 `all_types[]` 没有 Turbo，因此该命令没有执行新增 Turbo kernel。

### 7.3 Review-only Turbo eligible matrix

```bash
/tmp/test-backend-ops-turbo-review test -b ROCm0 -o FLASH_ATTN_EXT -p 'type_V=turbo'
```

HEAD 结果：`10/10 tests passed`。这是临时 harness，不是仓库永久测试。

### 7.4 Review-only Turbo fallback matrix

```bash
/tmp/test-backend-ops-turbo-isolation test -b ROCm0 -o FLASH_ATTN_EXT -p 'kv=113.*type_V=turbo'
```

结果：`0/4 tests passed`，NMSE `1.850922169-1.954714780`，见 B1。

### 7.5 CPU SET_ROWS crash

```bash
/tmp/test-backend-ops-turbo-review test -b CPU -o SET_ROWS -p 'type_src=f32,type_dst=turbo3'
```

结果：退出码 `139`，见 B2。

### 7.6 support gate mismatch

```bash
/tmp/test-backend-ops-turbo-isolation support -b ROCm0 -o SET_ROWS -p 'type_dst=turbo4.*ne=\[96'
```

结果：`ne=[96,3,1,1]` 被报告 `SUPPORTED`，见 B3。

### 7.7 正式 benchmark parser 失败

```bash
./build-rocm/bin/llama-bench \
  -m /models/qwen3.8/Qwen3.8-27B-Q4_0_ROCMFP4_STRIX.gguf \
  -ngl 999 -fa on -ctk q8_0 -ctv q8_0,turbo4 \
  -p 23,512,2048 -n 128 -b 512 -ub 512 -t 16 -r 5 --delay 1 -o json
```

结果：`error: invalid parameter for argument: -ctv`，见 W1。

### 7.8 benchmark A/B 原始样本

临时 parser 工具命令：

```bash
/tmp/llama-bench-turbo-review \
  -m /models/qwen3.8/Qwen3.8-27B-Q4_0_ROCMFP4_STRIX.gguf \
  -ngl 999 -fa on -ctk q8_0 -ctv q8_0,turbo4 \
  -p 23,512,2048 -n 128 -b 512 -ub 512 -t 16 \
  -r 5 --delay 1 -o json
```

HEAD commit `c1a31668a`、build `11632` 的原始样本（单位 tok/s）：

```text
q8/q8 p=23:   [100.108, 137.322, 137.087, 136.892, 137.080]
q8/q8 p=512:  [358.444, 370.313, 368.460, 368.503, 366.708]
q8/q8 p=2048: [356.970, 353.813, 352.301, 350.438, 348.216]
q8/q8 tg=128: [13.7718, 13.8002, 13.7988, 13.7932, 13.7934]

q8/Turbo4 p=23:   [92.6919, 134.638, 134.335, 134.228, 133.768]
q8/Turbo4 p=512:  [317.489, 329.436, 326.047, 326.240, 328.352]
q8/Turbo4 p=2048: [306.084, 306.009, 304.714, 303.635, 303.197]
q8/Turbo4 tg=128: [13.7652, 13.7940, 13.7870, 13.7838, 13.7823]
```

对应 `llama-bench` 报告的均值和标准差：

```text
q8/q8       p=23:   129.697705 +/- 16.541751
q8/q8       p=512:  366.485672 +/-  4.672551
q8/q8       p=2048: 352.347713 +/-  3.325597
q8/q8       tg=128:  13.791509 +/-  0.011447

q8/Turbo4   p=23:   125.932209 +/- 18.584519
q8/Turbo4   p=512:  325.512670 +/-  4.707515
q8/Turbo4   p=2048: 304.727863 +/-  1.324507
q8/Turbo4   tg=128:  13.782457 +/-  0.010664
```

## 8. 合入前必须满足的复验条件

1. 修复 B1，并在正式 `test-backend-ops` 覆盖所有 Turbo pair、非 eligible KV、views/permutation、mask/sinks、D128/D256 和 mixed q8 cases；所有 case 数值误差必须回到项目阈值内。
2. 修复 B2，CPU `supports_op` 与 `from_float` 能力完全一致；CPU-only 和 `--no-kv-offload` 启动、KV 写入及后续 attention 均不得崩溃。
3. 修复 B3，非法 `ne00=96` 不得报告 supported；合法 Turbo row shape 必须执行完整量化并有 backend case。
4. 恢复 B4 的 Q2 ROCmFPX 映射，并用缺少 `general.file_type` 的样本验证。
5. 把临时 harness 的高价值 case正式合入 `test-backend-ops.cpp`，同时修正移植报告的覆盖声明。
6. 修复正式 `llama-bench` parser 后，在同一硬件、同一模型、同一构建参数下重复记录 q8/q8、q8/Turbo3、q8/Turbo4 和 Turbo/Turbo 的 prompt/decode 数据，并报告原始样本、方差和 warmup 规则。
7. 重新评估 W2 的 VEC/TILE/MFMA dispatch；如果 bounded-slice helper 保留，必须先通过其长 KV、mixed、mask、sink、view 和 graph capture 测试，再用真实 batch 数据证明收益。
8. 同步 server/CLI/ops 文档，移除无契约的边界环境变量或补齐 parser、范围和测试。

内存节省 `23.529%` 和服务 generation 速度持平已经通过目标验收；它们不抵消 B1-B5 的正确性、兼容性和验证阻塞。最终拒绝合入仅基于这些阻塞项，不基于 TurboQuant 没有提速。
