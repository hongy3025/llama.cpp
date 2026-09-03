# GGSD:增量式槽位存储 - 使用手册与技术原理

GGSD(增量式槽位存储)为 llama.cpp 增加了内容寻址的增量式 KV cache 保存/恢复能力。长对话可以反复保存而代价接近于零;只要 token 前缀匹配,恢复时可以复用*任意*先前 session 保存的 KV 状态。

设计文档:`docs/superpowers/specs/2026-07-31-ggsd-incremental-slot-storage-design.md`

---

## 第一部分:使用者视角

### 1.1 解决什么问题

现有的 `/slots/{id}/save` / `restore` 端点(GGSQ v2)每次调用都会写入整个槽位状态的完整快照:对长对话来说,即使只有几个 token 变化,每次保存也要写数百 MB。

GGSD 改为把 KV 状态切分成固定 1024-token 的段(segment),每段只写一次、永不重写:

- **连续保存很便宜。** 只要没有跨过 1024-token 边界,生成几个 token 后再保存只需微秒级。
- **跨 session、跨槽位的前缀复用。** 恢复时,从磁盘重放 prompt 已保存的前缀,只对剩余部分做 prefill。产生该前缀的 session 无关紧要 - 保存在同一目录下的任何 session 都可复用。

### 1.2 新增配置与参数

**服务端参数:** GGSD 复用现有的 `--slot-save-path PATH` 目录(与经典槽位保存共用同一目录)。该目录必须已存在;未提供此参数则功能禁用:

```
llama-server -m model.gguf --slot-save-path saves
```

新增可选开关:

- `--prompt-cache-ssd` - 默认关闭;需要 `--slot-save-path`。开启后形成闭环:completion 结束时自动把槽位序列落盘到共享 session `__autosave__`(autosave);为请求挑选槽位时,自动把段池中匹配的 prompt 前缀恢复进槽位(autoload),客户端无需显式调用 `save_incr` / `restore_incr`(详见 `docs/ggsd-autoload-guide.md`)。

自动恢复与自动落盘使用两个命令行可调参数:

- `--prompt-cache-ssd-min-prefix N`(默认 1024)- 触发自动恢复/落盘的最小可复用前缀 token 数(建议为段大小 1024 的倍数);
- `--prompt-cache-ssd-margin N`(默认 256)- GGSD 候选必须领先次优来源(槽内 KV cache / RAM prompt cache)至少这么多 token 才会胜出。

GGSD 的所有产物都在这一个目录里:

```
saves/
  seg_<hash>.bin          # 每 1024 token 一个 KV 段
  session_<name>.bin      # 极小的保存侧提示文件(链尾哈希,44 字节)
```

### 1.3 HTTP API

在现有 slots 路由上新增两个 action,均为同步调用。

若启用 `--prompt-cache-ssd`(见 1.2),服务端会在槽位选择阶段自动恢复与请求 prompt 匹配的 GGSD 磁盘前缀,无需客户端调用。下面两个 action 的语义不受该开关影响,`restore_incr` 仍按原样工作。

#### 保存

```
POST /slots/{id_slot}?action=save_incr
{ "filename": "my-session" }
```

响应:

```json
{ "id_slot": 0, "filename": "my-session",
  "n_segments": 3, "n_tokens": 3072, "t_ms": 12.4 }
```

- `n_segments` / `n_tokens` 描述的是**本次保存之后磁盘上整条链的覆盖量**,不是本次调用写入的量。没有跨出新的完整段时,重复保存返回相同数字且 `t_ms` 接近零。
- `filename` 经过服务端文件名校验,对应磁盘上的 `session_<filename>.bin`。

#### 恢复

```
POST /slots/{id_slot}?action=restore_incr
{ "filename": "my-session", "prompt": "...", "min_prefix": 64 }
```

响应:

```json
{ "id_slot": 0, "filename": "my-session",
  "n_tokens_restored": 1024, "t_ms": 65.5 }
```

- `prompt` 按纯文本分词。其哈希链与保存目录中的 `seg_<hash>.bin` 文件进行匹配;`filename` 仅用于定位目录 - session 文件本身永远不会被读取。
- 恢复的 token 数向下对齐到 1024。槽位的 prompt 被置为已恢复前缀,下一次 `/completion` 从这里继续;只有前缀之后的 token 需要重新 prefill。
- `min_prefix`(可选,默认 64)是最小恢复长度。匹配不足时不恢复任何内容,返回 `n_tokens_restored: 0`,prompt 走完整 prefill。把 `min_prefix` 设得高于预期匹配长度,即可表达"值得恢复才恢复"。

#### 典型工作流

```sh
# 1. 跑一段长生成
curl http://localhost:8080/completion -d '{"prompt": "<long text>", "n_predict": 512}'

# 2. 给槽位做检查点(想调多频繁就调多频繁)
curl http://localhost:8080/slots/0?action=save_incr -d '{"filename": "story"}'

# 3. 稍后 / 在另一个进程里:恢复前缀并继续
curl http://localhost:8080/slots/0?action=restore_incr \
     -d '{"filename": "story", "prompt": "<same long text>", "min_prefix": 1024}'
```

跨 session 复用:保存 session `A` 后,用 `filename: "B"`(从未保存过)和 A 的 prompt 调 `restore_incr` - 前缀照样能恢复,因为段池按内容匹配,与 session 名无关。

### 1.4 C API

对内嵌 llama.cpp 的应用(`llama.h` 中的 `llama_state_seq_save_incr` / `llama_state_seq_load_incr`,实现位于 `src/llama-state-incr.cpp`):

```c
// 返回保存之后的链长(段数,而非本次新写的段数),出错返回 -1。
int32_t llama_state_seq_save_incr(
        struct llama_context * ctx,
        const char * session_path,   // session 文件路径;其父目录即段池
        llama_seq_id   seq_id,
        const llama_token * tokens,  // 序列的完整 token 列表,
        size_t   n_token_count);     // 位置必须从 0 开始

// 返回恢复的 token 数(向下对齐到 1024),无匹配或出错返回 0。
size_t llama_state_seq_load_incr(
        struct llama_context * ctx,
        const char * session_path,
        llama_seq_id   seq_id,
        const llama_token * prompt_tokens,
        size_t   n_prompt_tokens,
        size_t   min_prefix_tokens);
```

两者的同步要求与现有 `llama_state_seq_*` API 相同(照常在同一线程上调用 `llama_synchronize()` / decode)。

### 1.5 磁盘占用

每个段文件(1024 token):

```
header      ~110 字节    magic、version、seg_index、prev_hash、payload
                         size、payload hash、model_id、kv_params
tokens      4 KiB        该 1024 个 token id(i32)
payload     视模型而定   这 1024 个 token 对应的 KV 单元
```

payload 占大头。每段约等于:

```
n_layer x n_embd_kv x 2 (K 与 V) x sizeof(type_k/v) x n_streams
```

示例(K/V 为 f16,单流):

| 模型                      | layers x n_embd_kv | 每段 payload   |
|---------------------------|--------------------|----------------|
| stories15M (6 x 288)      | 6 x 288 x 2 x 2 B  | ~6.8 KiB       |
| 7B 级 (32 x 1024)         | 32 x 1024 x 2 x 2 B| ~128 KiB       |
| 70B 级 GQA (80 x 1024)    | 80 x 1024 x 2 x 2 B| ~320 KiB       |

session 文件每个固定 44 字节。

**增长与清理。** 段是内容寻址且共享的:分叉与改写(例如上下文回滚后继续生成)会创建*新*段并保留旧段,因此目录单调增长。**没有自动垃圾回收** - 孤儿段(没有任何 session 指向的链)需要手动删除。安全做法:先停服务,再删掉不再需要恢复的 `seg_*.bin`;即使删错了段,一切也只是优雅降级(恢复回退到更短前缀,或完整 prefill)。

### 1.6 语义、限制与错误行为

| 场景 | 行为 |
|---|---|
| 保存,内容全部已在磁盘 | 零 IO,链不变 |
| 保存,距上次保存不足 1024 个新 token | 不写任何内容(不完整的段永不落盘) |
| 保存,prompt 与已存链分叉 | 从分叉点写新分叉段;旧段仍可恢复 |
| 保存,KV cache 部分被驱逐 | 磁盘上的链保持不变;只有 cache 仍连续覆盖的段可写;不报错 |
| 保存,头段在 cache 和磁盘中都不存在 | 报错(`-1` / HTTP 500) |
| 恢复,匹配前缀 < min_prefix | 无操作,恢复 0 token |
| 恢复,链中途缺段或损坏 | 就此停止;已验证前缀保持已恢复状态 |
| 跨模型 / 跨 KV 配置恢复 | 自然不匹配(身份包含 arch 与 KV 类型) |
| 自动恢复开启,前缀匹配已保存的段 | 槽位选择阶段自动恢复,prefill 只处理剩余部分 |
| 自动恢复开启,GGSD 胜出但中途缺段 | 已验证前缀保持已恢复状态,剩余部分正常 prefill |

显式拒绝的场景:

- **SWA(滑窗注意力)cache** - 保存与恢复都直接报错拒绝;SWA 的单元驱逐使"每段完整"无法保证。
- **`n_pos_per_embd != 1`**(如 M-RoPE)- 拒绝;哈希链定义在文本 token 位置上。
- **多模态(mtmd)序列** - 服务端对含媒体的槽位拒绝 `save_incr`(哈希链只覆盖文本 token id)。`restore_incr` 总是按纯文本分词,天然不会匹配含媒体的序列。
- **位置偏移**(context shift)- 仅当需要写头段时拒绝保存;已持久化的链不受影响。

并发:每个保存目录单写者。服务端内部由任务队列串行;C API 本身不防两个进程写同一目录。

---

## 第二部分:技术原理

### 2.1 内容寻址的段链

一段对话是 1024-token 段组成的链表。每段的名字由其身份的截断 SHA-256 哈希导出:

```
hash_0 = sha256(model_id || kv_params || 16 zero bytes || tokens_0)   # 链头
hash_k = sha256(model_id || kv_params || hash_{k-1}     || tokens_k)

model_id  = llm_arch_name(model.arch)
kv_params = "<type_k>|<type_v>|<n_pos_per_embd>"      # 例如 "f16|f16|1"
```

关键性质:

- **不可变。** KV payload *不*参与哈希。段文件只写一次、永不修改,因此"文件是否存在"成为可靠的分叉探测信号。
- **自认证。** 给定 token 列表,任何人都能零磁盘读取地重算整条哈希链。恢复不需要索引、不需要 manifest、也不需要 session 文件:对 prompt 求哈希,检查哪些 `seg_<hash>.bin` 存在即可。这正是跨 session(G2)复用得以免费实现的原因。
- **链式绑定。** 每个哈希绑定其前驱,所以一段只在某条特定 token 链的特定位置上有效 - 无法把不同对话的段拼接起来。
- **配置绑定。** `model_id` 与 `kv_params` 在哈希之内,所以换模型或换 cache 类型恢复时会静默失配,退化为正常的完整 prefill。

### 2.2 段文件格式(小端)

```
u32  magic "GGSD"
u32  version = 1
u32  seg_index
[32] prev_hash                 # 十六进制 ASCII,链头全 '0'
u32  n_tokens = 1024
u64  payload_size
[16] payload_hash              # payload 的截断 sha256
str  model_id
str  kv_params
i32  tokens[1024]
     payload
```

payload 是标准的 `llama_kv_cache::state_write` 单元序列化(与 GGSQ v2 同格式),限定在该段的位置区间内。payload 哈希在流式写出时计算并回填进头部;恢复时校验,首个失配即停止(损坏防护:坏数据只能缩短恢复,绝不能污染已恢复的前缀)。

session 文件是 44 字节的提示(magic、version、链尾哈希、链长),仅供保存侧使用,用于跳过多余重写并记住上次链长。恢复永远不读它。

### 2.3 保存流程

1. 守卫:仅标准 KV cache、非 SWA、`n_pos_per_embd == 1`。
2. 读 session 文件(可选;损坏或缺失就当新链开始)。
3. 对 `tokens[0 .. n/1024)` 计算哈希链。
4. **分叉探测只看文件是否存在**(R2):沿链找到第一个 `seg_<hash>.bin` 缺失的哈希。这里有意不查 KV cache - 已持久化的段即使其单元被驱逐也依然有效。
5. 从分叉点开始写段,只要 `count_cells_range` 显示 cache 仍连续覆盖该区间;遇到第一个空洞即停(部分 KV 尽力而为,绝不缩短链)。已存在的文件直接跳过(内容寻址使重写变得多余,见 2.5)。
6. 链长从磁盘重新导出(最长的存在前缀)。若为 0 - 头段既不在磁盘也写不出 - 保存失败(R3,唯一的头段缺失规则)。位置守卫(`seq_pos_min == 0`)仅在必须写头段时生效(M2)。
7. 链长或链尾哈希变化时重写 44 字节的 session 文件。

推论(G1):没有跨出新的 1024-token 边界时,保存完全不触碰磁盘 - 实测 ~0.07 ms,对比真实写段 ~67 ms。

### 2.4 恢复流程

1. 守卫:非 SWA、`n_pos_per_embd == 1`。
2. 计算 prompt 的哈希链(纯算术,零磁盘读取)。
3. 对每个哈希,从 `session_path` 所在目录打开 `seg_<hash>.bin`(段池语义,R1)。第一个文件缺失即停止。
4. 对每个匹配段:校验头部(magic、version、model_id、kv_params、prev_hash 与链一致),边读边校验 payload 哈希,并通过 `llama_kv_cache::state_read_append` 把 payload 重放进 KV cache(不清空序列、直接追加,使多段可以连续恢复)。
5. 向下对齐到 1024,再套用 `min_prefix`;低于阈值则返回 0 且 cache 保持原样。链中途损坏时,已验证前缀保持已恢复状态,返回值如实反映该长度(M1:返回值始终与 cache 状态一致)。
6. 服务端把槽位 prompt 置为已恢复前缀;正常 prompt 处理从 `n_past = n_tokens_restored` 继续。

### 2.5 已接受的浮点非确定性

段的身份不包含 KV payload。同一 token 链两次 prefill 可能产生逐位不同的 KV 数据(batch 组成、线程数、后端都可能有影响)。保存发现段文件已存在时会跳过写盘,于是之后的恢复可能重放的是在不同配置下算出的 KV。这与现有跨槽位 prompt cache 的等价性假设相同,在此显式接受。若将来需要逐位确定性,就必须把 payload 哈希纳入内容地址。

### 2.6 实现足迹

- `src/llama-sha256.{h,c}`:内联的公有领域 SHA-256(ggml 没有公开实现)。
- `src/llama-state-incr.cpp`:两个入口函数、哈希、文件 IO(复用 `llama_file`、`llama_io_write_i`/`llama_io_read_i`)。
- `src/llama-kv-cache.cpp`:`state_write_range`(按位置区间切分单元)、`state_read_append`(不清空、追加式读取)、`count_cells_range`。现有 `state_write`/`state_read` 改为委托给它们,外部行为不变。
- `tools/server`:两个任务类型与两个槽位 action;路由不变(`POST /slots/{id}?action=save_incr|restore_incr`)。

GGSQ v2 格式及其代码路径完全未动;两种机制可以在同一服务端共存(只共享保存目录)。
