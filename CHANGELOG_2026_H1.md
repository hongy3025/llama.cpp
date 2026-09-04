# llama.cpp 2026 上半年重大变化总结

> 统计周期: 2026-01-01 至 2026-06-10
> 总提交数: ~1988 commits

---

## 一、新增模型支持

### 大语言模型

| 模型 | 说明 |
|------|------|
| **Gemma 4** | Google Gemma4 全系列，含 Gemma4ForCausalLM、Gemma4_26B_A4B_NVFP4、Gemma4 MTP |
| **Qwen3.5 / Qwen3.6** | Qwen3.5 dense 和 MoE 架构，含 NVFP4 量化权重支持 |
| **DeepSeek V3.2** | 含 DeepSeek Sparse Attention (DSA) 通用实现 |
| **DeepSeekOCR / DeepSeekOCR 2** | OCR 专用模型 |
| **Granite 4.0 / 4.1** | IBM Granite 系列，含 Granite4 Vision、Granite Speech、Granite multilingual embeddings R2 |
| **Mellum** | 新架构支持 |
| **EXAONE 4.5 / EXAONE MoE** | LG 模型系列 |
| **Step3.5-Flash / Step3.7-Flash** | StepFun 系列 |
| **Nemotron 3 Super / Nemotron-H / Nemotron Nano 3 Omni** | NVIDIA 系列 |
| **Mistral 3 / Mistral Small 4 / Mistral-Medium-3.5-128B** | Mistral 系列 |
| **GLM 4.7 / GLM MoE DSA / GLM-OCR** | 智谱系列 |
| **MiniCPM5 / MiniCPM-o 4.5 / MiniCPM-V 4.6** | 面壁系列 |
| **LFM2 / LFM2.5 / LFM2-VL / LFM2-Audio / LFM2-ColBert** | Liquid 系列 |
| **MiMo v2.5** | 新架构 |
| **Reka Edge 2603** | Reka 系列 |
| **Sarvam MoE** | 多语言 MoE |
| **Solar Open / Solar Open 100B** | Upstage 系列 |
| **Kimi-Linear / Kimi-K2.5** | Moonshot 线性注意力 |
| **Phi4ForCausalLMV** | Microsoft Phi-4 |
| **HunyuanOCR / HunyuanVL** | 腾讯混元 OCR/VL |
| **PaddleOCR-VL** | 百度 OCR-VL |
| **GigaChatV3/3.1** | Sber 系列 |
| **Talkie-1930-13B** | 新模型 |
| **Sarashina2.2-Vision-3B** | 日文 VL 模型 |
| **RuGPT3XL** | 俄语模型 |
| **Jina Embeddings v5 Nano / JinaBertModel** | Jina 嵌入/BERT |
| **Modern BERT** | 完整 ModernBERT 支持 |
| **Qwen3VL Reranker** | Qwen 重排序模型 |
| **Plamo2** | PKSHA 系列 |
| **Carbon-3B (HybridDNA)** | 混合 DNA tokenizer |

### 多模态 (mtmd)

- **Gemma 4 Unified** — 统一视觉+音频多模态
- **Qwen3 Omni / Qwen3 ASR** — 全模态和语音识别
- **Granite Speech** — IBM 语音模型
- **Gemma 4 Audio Conformer** — 音频编码器
- **MERaLiON-2 Audio** — 多模态音频
- **DeepSeekOCR / DeepSeekOCR 2** — OCR 支持
- **HunyuanOCR / HunyuanVL** — 混元 OCR/VL
- **PaddleOCR-VL** — 百度 OCR-VL
- **GLM-OCR** — 智谱 OCR
- **Dots OCR** — 通用 OCR
- **MiniCPM-o 4.5 / MiniCPM-V 4.6** — 视觉支持
- **LFM2-VL / LFM2-Audio** — Liquid 多模态
- **InternVL** — 动态高分辨率图像预处理
- **Gemma3n** — MobileNetV5 视觉编码器
- **Reka Edge** — 多模态
- **Nemotron Nano 12B v2 VL** — NVIDIA VL
- **Step3-VL** — StepFun VL
- **MiMo v2.5 Vision** — 视觉支持
- **Sarashina2.2-Vision** — 日文 VL
- **Frame Merge (Qwen-VL)** — 视频帧合并

---

## 二、核心框架增强

### 2.1 新 GGML 算子

- **GATED_DELTA_NET** — Delta-Net 门控算子，支撑 Qwen3.5/Next、Kimi-Linear 等新架构
- **Q1_0 1-bit 量化** — 极致压缩，CPU/Metal/CUDA/Vulkan/WebGPU 全后端支持
- **NVFP4 量化类型** — NVIDIA Blackwell 原生 FP4 支持，含 MMQ/DP4A 内核
- **Fast Walsh-Hadamard Transform (FWHT)** — KV cache 旋转，CUDA/Vulkan/CPU(SVE) 均实现
- **SOLVE_TRI / TRI** — 三角矩阵求解
- **FILL / CUMSUM / DIAG** — 新基础算子
- **CONV_3D** — 3D 卷积 (Metal)
- **XIELU / SOFTPLUS / FLOOR / CEIL / ROUND / TRUNC** — 新一元算子
- **ROLL** — 张量滚动 (Metal)

### 2.2 张量并行 (Tensor Parallelism, TP)

- **后端无关的张量并行** (实验性) — 支持多 GPU 分布式推理
- TP 量化 KV cache 支持
- TP 多 GPU 粒度修复 (Qwen 3.5/3.6 + 3 GPUs)
- TP AllReduce 延迟修复、零大小切片处理
- 内部 AllReduce CUDA kernel

### 2.3 KV Cache 增强

- **统一 KV Cache (unified KV)** — 跨 slot 共享 KV cache
- KV cache 量化支持扩展
- SWA (Sliding Window Attention) checkpoint 优化
- 异构 iSWA 注意力旋转支持
- KV cell 共享与复制优化
- V-less cache 支持
- M-RoPE checkpoint 修复

### 2.4 量化进展

- **NVFP4** — Blackwell 原生 FP4，支持 Gemma4/Mistral3/Qwen3.5/Nemotron-H
- **Q1_0** — 1-bit 量化，全后端覆盖
- **MXFP4** — 微缩格式 FP4 (OpenCL Adreno MoE 优化)
- **IQ4_NL / IQ4_XS** — 新 i-quant 变体
- **BF16** — 广泛的后端 BF16 支持 (CUDA FA、SYCL、OpenCL、CANN、Metal)
- **FP8** — FP8 到 Q8 转换、KV-cache FP8 scale
- **AVX512-FP16** — x86 原生 FP16 运算
- **KleidiAI SME FP16** — ARM SVE2 SME FP16 Q4_0 GEMM
- **RVV 量化向量点积** — RISC-V 128-bit/更高 VLEN 量化内核
- **PowerPC FP16 MMA** — Q4/Q8 矩阵乘法

---

## 三、后端重大更新

### 3.1 Vulkan (97 commits)

- **Cooperative Matrix 2 (coopmat2)** — 利用 `GL_NV_cooperative_matrix_decode_vector` 加速矩阵乘法
- **Flash Attention BF16 KV cache** — BF16 KV cache 支持
- **Flash Attention 量化 KV cache** — DP4A shader 量化 KV
- **FWHT 快速路径** — Intel GPU walsh-hadamard 变换
- **算子融合** — Snake 激活融合、SSM_CONV+BIAS+SILU 融合
- **Q1_0 / NVFP4 支持**
- **非对称 Flash Attention** — scalar/mmq/coopmat1/coopmat2 路径
- **IM2COL 优化** — 卷积性能提升
- **Intel Xe2 BF16 性能修复** — Windows 回归修复
- **AMD UMA 传输队列优化**

### 3.2 CUDA (64 commits)

- **Programmatic Dependent Launch (PDL)** — Hopper+ GPU 性能提升
- **Blackwell NVFP4 原生支持** — DP4A/MMQ 内核
- **Fast Walsh-Hadamard Transform** — KV 旋转加速
- **Flash Attention 增强** — DKQ=320/DV=256 支持、GQA 非 2 次幂加速、MLA K 数据复用
- **RDNA3 Q6_K MMVQ 调优** — AMD GPU 优化
- **算子融合** — Snake、SSM_CONV+BIAS+SILU、ReLU+SQR
- **CUDA Graph LRU 淘汰** — 内存管理改进
- **MoE topk 重构** — 支持更多模型 (GLM 4.7, Nemotron)
- **BF16 Flash Attention** — 原生 BF16 FA
- **量化 FA kernel 选择逻辑优化**

### 3.3 Metal (45 commits)

- **Q1_0 后端支持**
- **Flash Attention 扩展** — HSK=512/HSV=512、HSK=320/HSV=256、MLA heads FA
- **CONV_3D 支持**
- **算子扩展** — XIELU、FLOOR/CEIL/ROUND/TRUNC、ROLL
- **性能优化** — concat/pad/cpy 优化、MUL_MAT Tensor API 优化
- **macOS GPU 看门狗规避**
- **Apple device ID 支持**
- **GLU 模板化** — f16/f32 双精度

### 3.4 WebGPU (45 commits)

- **Flash Attention 重构** — 标准化量化支持、subgroup 感知 vec 路径
- **Prefill 性能提升** — K-quant 预填充加速
- **MMVQ 路径** — Q4/Q8/Q2_K/Q4_K 矩阵-向量量化
- **新算子** — SSM_SCAN、GATED_DELTA_NET、IM2COL、Layer Norm、Upscale
- **Q1_0 支持**
- **NVIDIA 自托管 CI** — 自动化测试
- **GPT-OSS-20B 支持**
- **RMS_NORM+MUL 融合**

### 3.5 Hexagon / Qualcomm (51 commits)

- **HMX Flash Attention** — 高通 HMX 加速器 FA
- **HMX 量化矩阵乘法重构**
- **GATED_DELTA_NET HTP kernel**
- **算子融合** — RMS_NORM+MUL 融合
- **Q4_1 / IQ4_NL / MXFP4 支持**
- **MROPE / IMROPE 支持**
- **GDN 优化** — 最新模型适配
- **DMA 优化** — 性能回归修复

### 3.6 OpenCL / Adreno (49 commits)

- **Adreno MoE 全面支持** — Q4_0/Q4_1/Q5_0/Q5_1/Q4_K/Q5_K/Q6_K MoE GEMM
- **MXFP4 Adreno MoE 优化**
- **BF16 支持** — 转换为 F16
- **GATED_DELTA_NET 支持**
- **Q5_0/Q5_1 基础支持**
- **批量性能分析** — 速度提升 + 内存泄漏修复

### 3.7 SYCL / Intel (19 commits)

- **BF16 DMMV kernel** — Intel Arc ~4x token generation 加速
- **VMM 内存池** — 虚拟内存管理
- **MoE prefill 吞吐提升**
- **Flash Attention Q4_1/Q5_0/Q5_1 支持**
- **Battlemage AOT 构建**
- **Q8_0 reorder 优化** — ~3x tg 加速

### 3.8 其他后端

- **OpenVINO** — 全新后端，含 NPU 优化
- **ZenDNN** — Q8_0 量化、MoE MUL_MAT_ID、自适应 CPU fallback
- **CANN (昇腾)** — BF16、Flash Attention、算子融合、量化 MoE
- **KleidiAI** — ARM SME/Neon 混合执行、动态分块调度、v1.24.0
- **SpacemiT IME2** — RISC-V 向量扩展指令支持
- **RISC-V** — RVV 量化向量点积、128-bit RVV、SIMD GEMM kernel
- **PowerPC** — FP16 MMA、BF16 向量点积优化
- **s390x** — BF16 向量点积、乘加扩展指令优化

---

## 四、Server & WebUI

### 4.1 Server 重大更新

- **统一可执行文件 (llama unified binary)** — 单一二进制包含所有子命令
- **Router 模式增强** — 子命令注入、子模型信息暴露、form-data 转发
- **RPC 后端** — 原生 RDMA 传输 (RoCEv2)、图缓存优化、Windows 修复
- **ETag / HTTP 缓存** — 条件请求支持
- **SSE Ping 间隔** — 长连接保活
- **实时推理中断** — control endpoint 支持 reasoning 中断
- **Vertex AI 兼容 API**
- **内置工具后端** — get_datetime 等服务器端工具
- **模型热重载** — `/models?reload=1`
- **Speculative checkpointing** — 推测解码检查点
- **Prompt 日志记录** — 目录持久化
- **超时提升至 3600s**

### 4.2 WebUI 重大更新

- **Agentic Loop + MCP Client** — 工具/资源/提示完整支持
- **Pinned Conversations** — 对话固定
- **Mermaid 图表** — 聊天内图表渲染
- **Thinking Mode** — 推理努力级别切换
- **视频文件输入** — 多媒体附件
- **自定义 CSS 注入**
- **单行推理预览**
- **run_javascript 前端工具** (opt-in)
- **MCP CORS Proxy** — 诊断和改进
- **无障碍 (a11y)** — 键盘导航修复
- **Tailwind v4 迁移**
- **KaTeX 更新**
- **仓库重构** — `tools/ui` 目录、统一命名

---

## 五、推测解码 (Speculative Decoding)

- **MTP (Multi-Token Prediction)** — 完整 MTP 支持，含 Gemma4 MTP、StepFun 3.5 MTP
- **Self-Speculative Decoding** — 无需 draft 模型的自推测解码
- **Ngram-Map / Ngram-Mod** — 增强的 ngram 推测方法
- **Parallel Drafting** — 并行 draft 支持
- **后端采样 MTP** — draft 路径移至后端采样
- **Server 推测检查点** — 服务端推测解码状态保存
- **Server-Bench 推测基准** — 专用性能测试工具

---

## 六、Chat Template & Parser

- **全新 Jinja 模板引擎** — 替代 minja，完整 Jinja2 兼容
- **Autoparser 完全重构** — 真正流式解析、参数重排
- **DeepSeek V3.2 专用 parser**
- **Gemma 4 专用 parser** — 含 reasoning budget
- **Granite 4.0/4.1 chat template** — 含 tool_call 角色映射
- **LFM2/LFM2.5 parser** — PEG 解析器
- **GigaChatV3/3.1 parser**
- **GPT-OSS parser 重构**
- **Solar Open parser**
- **Qwen3.5 非回溯 tokenizer handler**
- **MiniCPM5 tokenizer**
- **HybridDNA tokenizer (Carbon-3B)**
- **Jina Embeddings v2 tokenizer**
- **Falcon-H1 FIM tokens**

---

## 七、重大性能提升

### 后端级

| 领域 | 提升 |
|------|------|
| CUDA PDL (Hopper+) | 新一代 GPU 显著加速 |
| CUDA FWHT | KV 旋转加速 |
| Vulkan coopmat2 decode_vector | 矩阵乘法加速 |
| Vulkan FWHT (Intel) | Intel GPU KV 旋转加速 |
| SYCL BF16 DMMV | Intel Arc ~4x token generation |
| SYCL Q8_0 reorder | ~3x token generation |
| SYCL MoE prefill | 吞吐提升 |
| Metal concat/pad/cpy | 通用性能优化 |
| WebGPU prefill K-quant | 预填充加速 |
| Hexagon HMX FA/MM | 高通平台大幅优化 |
| OpenCL Adreno MoE | 全量化类型 MoE 加速 |
| HIP RDNA3 MMA FA | AMD GPU FA 加速 |
| GGML-CPU AVX2 Q6_K | x86 量化优化 |
| GGML-CPU SVE Q8_0 GEMM | ARM SVE 优化 |
| GGML-CPU RVV 量化 | RISC-V 向量加速 |
| GGML-CPU FA GEMM microkernel | CPU FA 加速 |
| KleidiAI SME FP16 | ARM SME Q4_0 GEMM |
| PowerPC FP16 MMA | Q4/Q8 矩阵乘法 |

### 框架级

- **Fast matmul i-quants** — 快速 i-quant 矩阵乘法
- **Fast mat-vec i-quants** — 快速 i-quant 矩阵-向量乘法
- **CUDA graph LRU 淘汰** — 减少显存占用
- **GGML 量化 LUT 并行初始化**
- **KV cache 量化** — 节省显存
- **统一 KV cache** — 多 slot 共享减少显存

---

## 八、基础设施

### 构建系统

- **统一可执行文件** — `llama` 单一二进制
- **UI 子目录重构** — `tools/ui` 独立目录
- **cpp-httplib 持续更新** — 0.30.1 -> 0.46.1
- **OpenSSL 替代 libcurl** — 移除 libcurl 依赖
- **Nix 构建支持** — Web UI 构建设施

### CI/CD

- **WebGPU NVIDIA 自托管 CI**
- **ARM 自托管 CI**
- **RISC-V 默认 runner 迁移**
- **s390x 发布**
- **KleidiAI Server CI**
- **IQ9 IoT 设备 CI**
- **SpacemiT 工具链更新**
- **ROCm 7.2 构建目标**
- **CANN Docker 8.5.0**
- **Snapdragon 工具链 v0.6/v0.7**

### 开发者体验

- **AGENTS.md 重写** — 明确项目价值观和 AI 使用准则
- **PR 模板 Requirements 章节**
- **Issue 模板改进** — 后端标签、log-file 提示
- **UI Git Hooks 改进**
- **代码所有者 (CODEOWNERS)** — ZenDNN、autoparser

---

## 九、统计概览

| 类别 | 提交数 (约) |
|------|------------|
| Vulkan 后端 | 97 |
| CI/构建 | 91 |
| Common 库 | 73 |
| GGML 核心 | 70 |
| Server | 65 |
| CUDA 后端 | 64 |
| WebUI | 54 |
| Hexagon 后端 | 51 |
| 模型转换 | 51 |
| OpenCL 后端 | 49 |
| Metal 后端 | 45 |
| WebGPU 后端 | 45 |
| 多模态 (mtmd) | 42 |
| llama 核心 | 34 |
| GGML-CPU | 31 |
| Vendor 更新 | 29 |
| SYCL 后端 | 19 |
| Jinja 引擎 | 17 |

---

*报告生成日期: 2026-06-10*
*数据来源: `git log --since="2026-01-01"`*
