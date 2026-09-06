# ROCmFPX 合并功能变更说明

本文记录 `feat/rocmfpx` 从 `hongy_main` 合并
`../QingYis_llama-7900xtx-qwen3.8-27b` 后引入的功能修改。

合并范围是源仓库基线 `89e531a` 之后的 9 个提交，而不是直接合并源仓库
较旧的完整分支树。这样可以只引入 ROCmFP4 和多模态 speculative decoding
相关修改，同时保留 `hongy_main` 已有的 GGSD 功能。

## 1. 新增 ROCmFP4 GGUF 量化类型

来源提交：`df9738db4 ggml: add Q4_0_ROCMFP4 / Q4_0_ROCMFP4_FAST quantization types`

新增两个 tensor 类型，并使用固定的 GGUF 类型编号：

| 类型 | 编号 | 格式特点 |
|---|---:|---|
| `GGML_TYPE_Q4_0_ROCMFP4` | 100 | 每块 32 个值、18 字节；两个 16 值半块各有一个 UE4M3 half-scale |
| `GGML_TYPE_Q4_0_ROCMFP4_FAST` | 101 | 每块 32 个值、17 字节；整块共用一个 UE4M3 scale |

具体格式行为如下：

- 4-bit 值使用独立的 ROCmFP4 Codebook10 查找表，量化级别为
  `0, 1, 2, 3, 4, 6, 8, 10` 及其负值。
- 16 个字节采用 split-nibble 布局：每个字节的低半字节保存前 16 个值，
  高半字节保存后 16 个值；它不是普通 Q4_0 的连续双值布局。
- 标准 unsigned UE4M3 scale 在 ROCmFP4 中采用 half-scale 语义；非法的
  inf/NaN 编码按格式规则归零。
- 新增 CPU 侧量化、反量化、向量点积及批量量化 API，并把类型纳入
  `ggml_type`、类型大小/块大小和通用类型分发表。

涉及的核心文件包括：

- `ggml/include/ggml.h`
- `ggml/src/ggml-common.h`
- `ggml/src/ggml-quants.c`、`ggml/src/ggml-quants.h`
- `ggml/src/ggml-cpu/ggml-cpu.c`
- `ggml/src/ggml-cpu/quants.c`、`ggml/src/ggml-cpu/quants.h`
- `ggml/src/ggml.c`

## 2. CUDA/HIP ROCmFP4 解码与矩阵乘内核

来源提交：`ae9784dbd cuda: add MMQ, MMVQ and convert kernels for ROCmFP4`

ROCmFP4 不再只作为文件格式存在，而是接入 CUDA/HIP 后端的实际推理路径：

- 新增 f32/f16/bf16 反量化支持。
- 新增 MMVQ 向量矩阵乘和 MMQ tiled GEMM 的 ROCmFP4 分派。
- 新增 `Q4_0_ROCMFP4` 与 `Q4_0_ROCMFP4_FAST` 的 CUDA template instances。
- MMQ tile loader 按 split-nibble 规则读取数据，避免按普通 Q4_0 布局解码造成静默错误。
- 为 Ampere、CDNA、Pascal/DP4A、RDNA2、RDNA3、RDNA3.5 和 RDNA4
  添加相应的 MMQ 配置项。
- RDNA3 配置包含 WMMA/MFMA 路径所需的不同 `J`、tile 尺寸和 fallback 组合。
- `MUL_MAT_ID` 等需要类型分派的 CUDA/HIP 路径也识别这两个新类型。
- 更新实例生成脚本和向量点积宏，使构建系统会编译 ROCmFP4 专用实例。

主要涉及：

- `ggml/src/ggml-cuda/common.cuh`
- `ggml/src/ggml-cuda/convert.cu`
- `ggml/src/ggml-cuda/mmq.cu`、`mmq.cuh`、`mmvq.cu`
- `ggml/src/ggml-cuda/mmq-load-tiles.cuh`
- `ggml/src/ggml-cuda/mmq-config-*.cuh`
- `ggml/src/ggml-cuda/vecdotq.cuh`
- `ggml/src/ggml-cuda/template-instances/`

## 3. 模型加载与量化工具支持

来源提交：`832c176f2 llama: support loading and quantizing ROCmFP4 models`

llama 层现在可以识别和生成 ROCmFP4 模型：

- `llama_model_loader` 根据 GGUF tensor 类型识别
  `Q4_0_ROCMFP4` 和 `Q4_0_ROCMFP4_FAST`。
- 增加 ROCmFP4 的 ftype 名称显示，包括普通、lean、coherent、fast 和
  STRIX 方案对应的固定 ftype 编号。
- `llama_ftype_get_default_type()` 将不同 ROCmFP4 ftype 映射到对应的
  `GGML_TYPE_Q4_0_ROCMFP4` 或 `GGML_TYPE_Q4_0_ROCMFP4_FAST`。
- `llama-quantize` 可以使用 ROCmFP4 类型生成量化模型；模型文件中的
  token embedding 等特殊层仍可通过 lean/coherent 方案使用 Q5_K 或 Q6_K
  组合，以平衡体积和质量。

涉及文件：

- `include/llama.h`
- `src/llama-model-loader.cpp`
- `src/llama-quant.cpp`

## 4. ROCmFP4 后端测试与真实模型形状覆盖

来源提交：`4668cf4c4 tests: add ROCmFP4 backend coverage, document the format`

`tests/test-backend-ops.cpp` 新增：

- 两种 ROCmFP4 类型进入全类型测试集合。
- Qwen3.8-27B 相关 hidden size 测试，包括 `k=5120` 和 `k=96`。
- `nrow=1/64/16/128` 等 MMVQ/MMQ 形状覆盖。
- RDNA MMQ 的 `J=16` 回归测试，以及 NVFP4/Q4_0 的相关控制样例。
- 数值误差失败时打印前若干个元素，便于定位 split-nibble 或 scale 解码错误。

## 5. DFlash2 多模态 speculative decoding 绕过机制

来源提交：

- `6d3c47cbd speculative: bypass drafting for multimodal (vision) requests`
- `873997802 speculative: detect real media chunks for the vision bypass`

多模态请求现在可以按请求粒度关闭 speculative decoding：

- `common_speculative` 新增每个 sequence 的 `vision_skip` 标志。
- 新增公开函数
  `common_speculative_set_vision_skip(spec, seq_id, skip)`，用于在生成开始时
  设置或重置该标志。
- `common_speculative_process()` 在 vision 请求上直接绕过 speculative
  processing。
- `common_speculative_draft()` 对 vision sequence 不再生成 draft token。
- DFlash draft 实现跳过 embedding batch，避免把 mtmd 图像 embedding 注入
  draft KV cache 后破坏连续位置约束。
- server 不再仅依据“是否加载了 mtmd pipeline”判断请求类型，而是通过
  `find_next_media_chunk()` 检测当前请求是否真的包含媒体块。
- 含图片/媒体的请求按 target model 速度运行；纯文本请求继续使用原有
  speculative decoding。

涉及文件：

- `common/speculative.cpp`
- `common/speculative.h`
- `tools/server/server-context.cpp`

这样处理的原因是 DFlash draft KV 注入无法可靠跟踪 mtmd image-chunk 的
位置；如果继续 drafting，可能触发 draft cache 初始化失败或位置不连续。

## 6. 启动脚本与运行配置

来源提交：`6f993ba83 docs: reference serving script and full configuration rationale`

新增 `scripts/serve-dflash2.sh`，提供 RX 7900 XTX/gfx1100 的参考启动方式，
覆盖：

- target/draft 模型完整 offload；
- q4 KV cache；
- ngram-map 与 DFlash2 叠加 speculative decoding；
- 256K context、batch size、checkpoint、cache RAM 等稳定性参数；
- `LLAMA_ATTN_ROT_DISABLE=1` 等 HIP/hybrid 模型相关环境设置。

同时新增 `docs/rocmfp4.md`，说明格式布局、量化方案、启动参数和验证结果。

## 7. README 与双语文档

来源提交：

- `2e3737976 docs: bilingual README (en/zh-CN) with language switcher`
- `01467b7e5 docs: add complete annotated launch command to both READMEs`

文档层面新增或更新：

- `README.zh-CN.md` 中文 README；
- 中英文 README 的语言切换链接；
- ROCmFP4、DFlash2、服务启动脚本和完整参数示例；
- Qwen3.8-27B / RX 7900 XTX 的参考运行说明。

这些提交主要是使用与分发支持，不改变核心推理接口。

## 8. 本次合并中的兼容处理

源仓库的 Pascal 配置文件名是 `mmq-config-pascal.cuh`，而
`hongy_main` 使用的是 `mmq-config-pascal-dp4a.cuh`。合并时将 ROCmFP4
对应的 Pascal 配置变更映射到当前分支的 `mmq-config-pascal-dp4a.cuh`，
并保留 `hongy_main` 中已有的：

- GGSD `synth_probs` 状态；
- 现有 GGSD cache/autoload/hybrid split 等功能；
- RDNA3 原有的 NVFP4 tile 配置。

因此当前 `feat/rocmfpx` 同时包含 ROCmFP4 支持、DFlash2 多模态绕过和
`hongy_main` 的 GGSD 功能。

## 9. 验证状态

- 合并提交：`85bf60f39c905faedb5af9980cf04d3dfce37c37`
- 分支：`feat/rocmfpx`
- 使用 `GGML_HIP=ON`、`GGML_HIP_MMQ_MFMA=ON` 配置的独立 ROCm 构建成功，
  构建命令退出码为 0。
- 构建覆盖了 HIP backend、ROCmFP4 MMQ 实例、`server-context`、
  `test-backend-ops` 和 `llama-server`。
