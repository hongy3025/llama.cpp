# GGSD 整改计划(Review Remediation)

日期:2026-09-02
状态:已执行(见各条目状态)
范围:针对 2026-09-02 对 `feat/ggsd` 分支静态评审发现的全部问题。设计 spec 为 `docs/superpowers/specs/2026-07-31-ggsd-incremental-slot-storage-design.md`,以下 R 编号在整改后的 spec 中均有对应修订。

## 评审结论(背景)

方向成立:KV cache 层位置范围切片 + 缓存层内容寻址 hash 链段,是段级前缀共享的正确架构。问题集中在名实关系、manifest 表达力、未声明边界三方面,均为 spec/语义层修补,不动架构。

## R1 — G2 名实不符:跨 slot 前缀复用未实现【已修,选"池语义"方案】

原 restore 只匹配显式传入的单个 session 文件,"新请求与哪个历史会话共享前缀"正是客户端不知道的,G2 未兑现。

**决策:采用"目录即池"(segment pool)语义,而不是降级 G2。** 理由:hash 链自认证(段 hash 覆盖 model_id/kv_params/prev_hash/tokens),按 prompt token 现算 hash 链、逐段探测文件存在性即可走链,session 文件对 restore 是多余的;实现反而变简单。

- `state_seq_load_incr`:删除 session 文件读取与 `n_segments` 上限;`n_seg = n_prompt_tokens / 1024`,直接按文件存在性走链。目录中**任意** session 的段文件都可被匹配 → G2 兑现。
- session 文件职责收缩为 save 侧的链尾提示(见 R2)。

## R2 — manifest 表达力不足:KV 部分驱逐导致链被静默缩短【已修】

原实现 fork 探测要求"段文件存在 **且** KV 仍有该段 cells",KV 驱逐即触发链缩短,丢失已持久化段。

**决策:fork 探测只看文件存在性(KV cells 仅约束能否写新段);不做 manifest v2 全 hash 链扩展。** 理由:

- 文件存在 ⇒ hash 链前缀一致 ⇒ 内容与链匹配(段 hash 覆盖 prev_hash),KV 是否还在不影响已持久化数据的有效性。
- 驱逐后 save:文件全在 → 链不变、零写入;分叉 + cells 不足 → 只写到 cells 断点,断点之后的旧链段因 hash 链分叉本就失效。
- 池语义下 restore 不依赖 manifest(R1),manifest 缩短不再造成不可达;save 侧 fork 探测基于文件存在性,可自行恢复更长链。
- manifest 仍记 `tail_hash + n_segments`(v1 格式不变),作为 save 的"上次链长"提示与跳过重写的判据。

新 save 流程:

```
guards(见 R4)
hashes[0..n_seg)
k0 = 第一个 seg_<hash> 文件不存在的 k(上限 min(n_segments, n_seg))
for k = k0..n_seg-1:              // 写新段
    if count_cells_range != 1024: break        // KV 断点,停止
    if 文件不存在: 写段(tmp + rename)          // 已存在则跳过(内容等价)
n_chain = 从 0 起按文件存在性的最长连续链(≤ n_seg)  // 最终裁决
if n_seg > 0 && n_chain == 0: head 缺失错误(-1)
if n_chain != n_segments: 重写 session 文件
return n_chain
```

## R3 — spec 自相矛盾:驱逐缩短 vs head 缺失报错【已修,统一规则】

"early cells evicted → chain shortened" 与 "KV head missing → save fails" 是同一场景(context shift 驱逐头部)的两行矛盾。统一为单规则:

- **head 缺失** = `n_seg > 0` 且段 0 既无文件又写不出来(`n_chain == 0`)→ 报错。
- 其余一切部分 KV 情况 → 尽力而为,链以文件为准(R2),不报错、不虚构。

spec 边界表已重写为该规则。

## R4 — 未声明边界:SWA、pos 基址、n_pos_per_embd【已修,显式拒绝】

- **SWA**:SWA cache 下旧 cell 被回收,逐段完整性永不成立,原实现会报误导性的 "KV head missing"。改为进入 save/load 即检查 `kv->swa_type != LLAMA_SWA_TYPE_NONE` → 明确报 "not supported"。
- **pos 基址**:段换算 `pos = token_index * n_pos_per_embd` 隐含"位置从 0 连续"。save 在**需要写头段时**检查 `seq_pos_min(seq_id) != 0` → 报错(评审 M2 后收窄作用域:所有段已在盘时跳过检查,驱逐/空缓存不报错,与 R2/R3 一致);把隐式假设变成显式错误条件。

## R5 — mtmd 序列未拦截【已修,server 侧拒绝】

`check_no_mtmd` 在仓库中不存在被删除,但替代检查没有补上:`get_tokens()` 对多模态序列返回含 `LLAMA_TOKEN_NULL` 占位符的裸 token,会存出"看似合法"的段。在 server `SLOT_SAVE_INCR` 分支增加 `slot->prompt.tokens.has_mtmd` 检查并报错。restore 侧无需检查(纯文本 tokenize 与含媒体序列天然不匹配,优雅退化为全量 prefill)。

## R6 — FP 非确定性假设未声明【已修,spec 声明】

段文件名只含 token 链,KV payload 不参与内容寻址。命中即跳写意味着恢复出的 KV 可能来自不同 batch 组成/线程数的计算(同 token ⇒ 不同 FP 结果)。功能上正确(与现有跨 slot prompt cache 同一假设),但 spec 必须显式声明该假设及其适用性。

## R7 — 文档修正【已修】

- C API 注释:save 返回值澄清为"save 后 session 链覆盖的段数"(非本次写入数);load 注释改为池语义;两处补充 R4 拒绝条件。(`llama.h`)
- spec "Reusable pieces" 声称复用 `ggml_sha256` —— 该符号在基线 ggml 中不存在(仅 opencl/hexagon 私有实现),vendor `src/llama-sha256.{h,c}` 是必要的。spec 已修正表述。
- spec 补充:单写者假设(session/段目录不支持多进程并发写,server 内部由任务队列串行保证);`.tmp + rename` 原子性说明。
- 测试 Test 8(partial KV)预期更新为新语义:尾部驱逐后 save 链保持不变、restore 恢复全部已存段;head 缺失报错改用全新 session 触发。Test 5/6/7 预期在新语义下不变(fork 测试中"orphaned chain 仍可 restore"恰好被池语义正式化)。
- spec 与实现对齐的其余出入:段文件内容布局补齐实际字段(`payload_size` u64 dummy 写实测、`payload_hash` 16 字节写后回填读侧校验);`prev_hash`/`tail_hash` 澄清为 32 个 hex 字符、链首 prev_hash 为 16 个零字节;C API 小节恢复完整函数签名与 load 池语义描述,实现位置更正为 `src/llama-state-incr.cpp`(声明在 `llama-context.h`);Architecture Overview 补充"目录即段池"语义;Server Endpoints 响应字段澄清(n_segments/n_tokens 为链覆盖量)。

## 不改的事项

- 逐段两次遍历 cells(dummy 测大小 + 实写):正确且 dummy 不触 GPU 拷贝,代价可接受。
- orphan 段无 GC:维持 non-goal。
- 跨进程并发写:维持单写者假设(R7)。

## R8 — 独立 review 修复(2026-09-02,GgsdReviewer 报告)【已修】

- **M1** load 段损坏回滚后仍返回非零恢复 token 数,返回值与缓存状态矛盾 → 删除回滚,保留已验证前缀并如实返回(与 spec "restore shorter prefix" 一致);异常路径仍清空并抛出(C API 返回 0)。
- **M2** `seq_pos_min` 守卫无条件生效,头段驱逐/序列清空即使链已持久化也报错,违反 R2/R3 → 守卫收窄为"仅当需要写头段时"(k0 < n_seg);并在代码注释/spec/llama.h 声明正确性依赖"调用方传入的 tokens 与序列 cells 的位置 0..n-1 对应"。
- **m1** Test 8 的 -1 归因错误(实为 pos 守卫)且 R3 零覆盖 → 重构:整头驱逐 + 原 session 断言链保持(R2);部分头驱逐 [512,1024) + 全新 session 断言 -1(真 R3 路径)。
- **m2** 等长 re-fork 后 session tail_hash 陈旧 → 恢复重写条件中的 tail_hash 比较。
- **m3** restore_incr 后 `has_mtmd` 残留,纯文本序列被 save_incr 误拒 → RESTORE_INCR 用 `server_tokens(..., false)` 重建 prompt。
- **d1** handoff "唯一报错条件" 表述与 R4 拒绝条件矛盾 → 改为"除显式拒绝外"。
