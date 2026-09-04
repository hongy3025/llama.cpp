# GGSD Prompt Cache SSD(自动填充与自动落盘)- 使用手册与技术原理

GGSD Prompt Cache SSD 让服务端形成 KV 复用的自动闭环:completion 结束时自动把槽位状态落盘到磁盘段池(autosave);分配槽位时,当槽位内存缓存都帮不上忙,自动从段池找回与新请求前缀匹配的 KV(autoload),只对剩余部分做 prefill。

设计文档:`docs/superpowers/specs/2026-09-03-ggsd-autoload-design.md`
基础 GGSD 手册:`docs/ggsd-guide.md`(段格式、哈希链、保存/恢复语义);hybrid split 模式设计:`docs/superpowers/specs/2026-09-03-ggsd-hybrid-split-mode-design.md`

---

## 第一部分:使用者视角

### 1.1 解决什么问题

开启本特性之前,磁盘段池只是一个"手动检查点":要靠调用方显式调 `save_incr` 写入、`restore_incr` 读回,进程重启后即使池里有匹配前缀也会从零 prefill。

开启之后,服务端把磁盘段池当作第三级 KV 复用来源,自动写入、自动读回,与另外两级竞争:

```
第 1 级  槽位 KV(内存,请求间留存)      - 现有 LCP 相似度逻辑
第 2 级  RAM prompt cache(--cache-ram) - 现有 LRU 逻辑
第 3 级  GGSD 段池(磁盘,跨进程持久)   - 本特性新增
```

三者择优,**只执行胜者**。典型受益场景:服务重启后重放同一长对话;多个客户端共享相同的长 system prompt / 文档前缀;进程内 RAM cache 被淘汰后磁盘上仍有备份。

### 1.2 配置与参数

一个开关同时启用读与写两个方向,**默认关闭**:

```
--prompt-cache-ssd
```

前置条件:`--slot-save-path` 必须已设置(段池目录),否则启动时打警告并将开关视为关闭。示例:

```
llama-server -m model.gguf --slot-save-path saves --prompt-cache-ssd
```

两个阈值参数(默认 1024/256,可用命令行调整):

| 参数 | 默认 | 含义 |
|---|---|---|
| `--prompt-cache-ssd-min-prefix N` | 1024 | 至少能复用这么多 token 才触发磁盘恢复/落盘(建议为段大小 1024 的倍数) |
| `--prompt-cache-ssd-margin N` | 256 | GGSD 必须比次优来源多出至少这么多 token 才胜出 |

Margin 的作用是防抖:GGSD 恢复一次约 65ms/段的磁盘 IO,如果只比 RAM cache 多赚几个 token,不值得。关闭开关时,服务端行为与不装此特性**逐字节一致**。

### 1.3 使用上有什么变化

**没有新的 API、没有新的请求字段**。你照常发 `/completion`;区别只在服务端内部:

- **自动落盘(autosave)**:服务端在用户消息边界保存可分叉前缀,并在 completion 结束、槽位释放时保存完整序列(含生成部分)到共享 session `__autosave__`;不足 1024 token 或内容已存在时是近零成本 no-op;失败(磁盘满等)只打日志,不影响请求;
- **自动恢复(autoload)**:命中时服务端日志出现 `__GGSD__ autoload: restored <N> tokens (slot X, cache Y)`,响应里 `timings.prompt_n` 只覆盖恢复前缀之后的 token 数;
- 未命中时:与今天完全一样,槽位续用或全量 prefill。

实测证据(135M 模型,约 2000 token 的 prompt:第一次 completion 触发自动落盘,重启服务端后重发同一请求):

```
关闭开关:  prompt_n = 2001   (全量 prefill)
开启开关:  prompt_n = 977    (前 1024 token 从磁盘自动恢复)
          日志: slot autoload_ggs: id  3 | task -1 | __GGSD__ autoload: restored 1024 tokens (slot 0, cache 0)
```

手动端点 `POST /slots/{id}?action=restore_incr` 的语义不变,两者互不干扰:手动调用显式指定目标槽位与 min_prefix;自动路径只在槽位分配瞬间、按统一阈值决策。

### 1.4 行为语义表

| 场景 | 行为 |
|---|---|
| 槽位/RAM cache 可复用长度 >= 1024,且 GGSD 帮不上更多 | 不碰磁盘,走现有路径 |
| GGSD 预估命中 >= 1024 且超过次优 256 | 自动从磁盘恢复,剩余部分 prefill |
| 槽位已有与新请求共享的 KV,GGSD 命中更长 | 差量恢复:槽位截断到段对齐边界,只回放缺失的段 |
| GGSD 恢复中途段损坏/缺失 | 保留已验证前缀(M1),失败时清槽兜底,回退全量 prefill |
| 恢复尝试前预估就不足 | 什么都不碰(槽位此前已被抢救进 RAM cache) |
| 含多模态(mtmd)的请求 | 跳过 GGSD 分支(哈希链无法表达媒体占位) |
| 跨模型/跨 KV 配置 | 哈希链自然失配,GGSD 计 0,不触发 |
| completion 结束 | 自动落盘到共享 session `__autosave__`(无新段边界时近零成本 no-op);失败仅日志,不影响请求 |

### 1.5 什么值得开启

- 长对话检查点 + 服务经常重启:磁盘池跨进程持久,重启后第一发请求就能省下整段 prefill;
- 多客户端共享长前缀(固定 system prompt、RAG 文档):段池按内容寻址,谁保存的都能复用;
- 开环变闭环:无需任何手动 `save_incr`,对话在多次请求/多次重启间自动接力。

不建议开启的场景:请求前缀高度随机(命中率低,虽然 miss 路径只有哈希计算 + 每段一次 stat,成本可忽略,但也没有收益);磁盘 IO 受严格 SLO 约束的环境(命中路径有真实读盘,尽管 OS page cache 会吸收重复读)。

---

## 第二部分:技术原理

### 2.1 三方仲裁:单点决策、单路径执行

决策挂在 `get_available_slot` 现有的 `update_cache` 块内(`server-context.cpp`),即槽位分配、prompt cache 存取的既有位置:

```
n_slot  = LCP(slot.prompt.tokens, task.tokens)              // 现有槽位复用
n_cache = prompt_cache->peek(task.tokens, slot.prompt.tokens) // RAM cache 最优候选
n_ggsd  = ctx->state_seq_load_incr_estimate(session, tokens)  // 零 IO 预估

best = max(n_slot, n_cache, n_ggsd)
best < 1024            -> 现行路径(cache 候选有则消费,无则清槽全量 prefill)
ggsd 胜出(超次优 256) -> GGSD 自动恢复
cache 胜出             -> consume 候选条目并 set_data(即现行 load)
slot 胜出              -> 什么都不做
```

三个数量纲一致:都是"可复用前缀 token 数"。进入仲裁的前置条件不变(槽位由 LRU 或 f_keep<0.5 相似度路径选出、`--cache-ram` 开启、completion 任务),且 `prompt_save` 抢救逻辑照旧先执行 - 所以 GGSD 胜出后截断掉的槽位尾部,已经被抢救进 RAM cache,不丢上下文。

关键设计:**peek 与 consume 分离**。原 `server_prompt_cache::load` 是"评分+灌入+删除"一体的;拆成 `peek`(只评分,不消耗条目)和 `consume`(set_data + 移动 prompt + erase)之后,仲裁才能在"不付出恢复成本"的前提下给 RAM cache 报价。消耗语义(erase)原封不动保留在 consume 里。

为什么两源不能叠加:GGSD 哈希链从位置 0 的链头自认证,段只在"从头匹配的链"上有意义;RAM cache 条目灌入的是任意前缀状态,两者无法在链中段接续。所以是竞争关系 - 若 cache 命中 500 而 GGSD 命中 2048,先灌 cache 再追加重放既会产生同位置单元冲突,又浪费了整块 restore。

### 2.2 零 IO 预估:为什么决策本身几乎免费

```cpp
size_t state_seq_load_incr_estimate(session_path, seq_id, tokens, n, min_prefix) const;
```

实现 = 对 `tokens` 逐段计算哈希链(纯 SHA-256,无磁盘读取)+ 沿链对每个 `seg/<hh>/<hash>` 做一次 `stat` 判存在。万级 prompt 只是十几次 stat,微秒级。段池的"文件存在即有效"性质(基础手册 2.1:段不可变、自认证)使**不读内容就能精确知道能恢复多少**。SWA 直接返回 0;标准 cache 下 M-RoPE 也已放开(`cell_ext` 随追加式恢复还原)。hybrid 模型走独立的 rec 头部扫描(见 2.8)。

因此自动路径的 miss 成本可以忽略,不需要命中率学习或缓存决策结果。

### 2.3 差量回放:只读缺失的段

当槽位已持有请求前缀 `[0, m)` 的有效 KV(比如它就是上一轮对话留下的),从链头全量重放是浪费。`llama_state_seq_load_incr` 因此增加第 7 个参数:

```c
size_t llama_state_seq_load_incr(..., size_t min_prefix_tokens,
                                     size_t n_prefix_valid);   // 新增
```

- `n_prefix_valid == 0`:现行语义,从链头全量重放(内部先清序列),手动端点走这条;
- `n_prefix_valid == m > 0`:声明槽位已持有 `[0, m)` 的有效 KV。实现计算 `k0 = m / 1024`:
  - 哈希链仍从头计算(身份自认证不变),但**跳过的段 `[0, k0)` 连 stat 都不做**;
  - 存在性检查、payload 哈希校验、`state_read_append` 重放都只对 `k >= k0`;
  - 截断由 load 自己完成:`seq_rm(seq_id, k0*1024, -1)`(丢弃不足一段的尾部,最多 1023 个 token 的 KV,这些本就在"差异部分"里);
  - 返回值 = `(已重放段数 + k0) * 1024`,始终与槽位实际覆盖一致(M1 语义)。

服务端侧的差量触发条件:`n_slot >= 1024` 且槽位 token 恰好是请求 token 的前缀(LCP 完整,否则跳过的段与槽位 KV 不对应)。对齐公式 `m_aligned = floor(min(n_slot, n_ggsd)/1024)*1024` 保证不重叠。

hybrid split 模式不走差量回放:rec 状态钉死覆盖量,`n_prefix_valid` 被忽略,始终全量重放(见 2.8)。

强验证(已入测试 Test 11):decode 2500 → save 2 段 → **删掉段 0 文件** → 用 `n_prefix_valid=1024` 恢复,仍得 2048 且生成与参考一致 - 证明跳过的段从未被读;同一删除下 `n_prefix_valid=0` 则恢复 0。

### 2.4 失败一致性:为什么失败路径必须清槽

`load_incr` 的拒绝路径(参数守卫、min_prefix 不足)在截断**之前**返回;但一旦进入重放,截断已经发生 - 中途段损坏会使已截断的槽位与陈旧的 `slot.prompt.tokens` 不一致。若此时回退到 prompt cache 路径且 cache 无候选(返回"无事可做"),槽位就会拿着错误上下文继续解码。

因此 `autoload_ggsd` 的规则:任何**发起了 load 尝试**后的失败(返回 < MIN_PREFIX),无条件清序列 + 清 prompt,让请求落到干净的全量 prefill;而**尝试之前**的预估不足不碰任何状态(槽位此前已被 `prompt_save` 抢救)。RAM cache 的抢救保证被清掉的上下文大概率还在内存里。

### 2.5 与基础 GGSD 语义的关系

- 段格式、哈希链、save/load 的核心流程不变;本特性新增了段池的自动消费(autoload)与自动写入(autosave),写入仍走原有 `save_incr` 路径。
- FP 非确定性假设不变:槽位前缀 KV 与请求 token 的对应关系靠 `prompt.tokens` 记录信任(payload 不在哈希内),与 prompt cache 同一信任面。手动端点面对空槽没有这个问题,自动路径显式承担了它。
- 单写者假设不变;预估与实际加载之间段池若被并发修改,由 load 的逐段校验兜底,返回值如实。

### 2.6 实现足迹

- `src/llama-state-incr.cpp`:`ggsd_hash_chain` helper(收敛 save/load 的重复链计算)、load 的 `k0` 差量逻辑、`state_seq_load_incr_estimate`;顺带修复既有栈溢出(payload_hash 16 字节缓冲写入 32 字节摘要,磁盘格式不变)。
- `src/llama-context.h`:`state_seq_load_incr` 加参、声明 estimate(const 成员)。
- `tools/server/server-context.cpp`:`--prompt-cache-ssd` 前置检查、`autoload_ggsd` 仲裁方法、`slot_autosave` 自动落盘、`update_cache` 块改造。
- `common/common.h`、`common/arg.cpp`:开关定义与解析。
- 测试:`tests/test-save-load-state.cpp` Test 11(差量 + 删头强验证 + 生成等价)。

### 2.7 自动落盘(autosave)

触发点有两个:处理用户消息前的 context checkpoint,以及 completion 结束、槽位释放处(生成停止的两个路径)。两处都调用 `llama_state_seq_save_incr`,session 文件固定为段池目录下的 `session___autosave__.bin`(注意:C API 的 `session_path` 参数是 session **文件**路径,段池是其父目录)。用户边界快照保存 system/tools 等可分叉前缀;completion 快照支持同一对话继续。共享单链的意义:多个槽位交替保存不同链时,分叉探测只看文件存在性,提示文件过期无害;段按内容寻址,跨槽共享不受 session 名影响。

安全边界:fire-and-forget - save 失败(磁盘满、context shift 后的 R4 位置偏移拒绝等)只打 `SRV_WRN`,绝不影响请求结果;非 completion 任务、mtmd 序列、不足一段的 prompt 直接跳过。

### 2.8 hybrid 模型(Qwen3.5 家族):split 模式下的自动闭环

`--prompt-cache-ssd` 现在同时覆盖两类 memory:`llama_kv_cache` 走上述段模式(行为不变);`llama_memory_hybrid` 走 split 模式(基础手册 1.7:共享 attn 段 + 每份保存状态一个 `rec/<hh>/<hash>`);其余 cache 类(`llama_kv_cache_iswa`、纯 recurrent、hybrid-iswa)照旧拒绝。SWA 混合被拒绝,`--swa-full` 与此无关。

对自动路径的影响:

- **仲裁接口不变**。hybrid 的预估改为扫描 `rec/<hh>/` 下的文件头部(只读头部,不读 token 数组,每头几 KB):命中条件 = 请求 token 前缀的链哈希等于文件的 `chain_hash`,并且用与 load 完全相同的覆盖规则校验段存在性 - 该 rec 自身链长 `(n_tokens - n_tail) / 1024` 所需的段必须都在盘上;池中来自其他会话的更长链不影响命中(load 只重放 rec 自身链长的段,再接 tail,不会与更长的盘上链冲突)。报价 = 该文件的 `n_tokens`。`min_prefix` / margin 规则不变。
- **匹配更严**:请求必须精确延长某个已保存的 rec 快照(rec 状态无法从中间截断,段文件不单独扩大覆盖)。服务端在用户消息边界保存可分叉前缀,因此连续对话和共享 system/tools 前缀的新对话都能命中。
- **成本**:hybrid 命中路径没有差量回放(见 2.3),整段重放;每份会话状态的磁盘占用 = rec 状态(固定,几 MB)+ 非对齐 attn 尾部(最多 1023 token)。
- **保留存盘点**:父 rec 必须保留供兄弟分支恢复,不能由更长的子 rec 替代。rec 与段池均无自动 GC,手动删。
- autosave 失败兜底、`--cache-ram` 共存、mtmd 跳过等语义与标准模型完全一致。
