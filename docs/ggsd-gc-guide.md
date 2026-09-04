# GGSD Cache GC - 配额、回收与运维手册

GGSD Cache GC 为 GGSD 磁盘段池提供同步、依赖感知的自动回收。它限制
`seg`、`rec` 和写入中的临时文件占用，避免长时间运行的服务因分支、自动保存
和服务重启持续积累缓存文件。

相关手册:

- 基础格式与手动保存/恢复:`docs/ggsd-guide.md`
- 自动加载与自动保存:`docs/ggsd-autoload-guide.md`
- 设计说明:`docs/superpowers/specs/2026-09-04-ggsd-cache-auto-gc-design.md`

---

## 第一部分:快速配置

### 1.1 启用磁盘配额

服务端参数:

```sh
llama-server \
  -m model.gguf \
  --slot-save-path saves \
  --prompt-cache-ssd-max-mib 4096
```

`--prompt-cache-ssd-max-mib N` 的语义:

| 值 | 行为 |
|---|---|
| `0` | 默认值。不启用 quota/GC，保持原有无限制行为 |
| 正数 | 将 GGSD 管理字节限制为 `N * 1024 * 1024` |
| 负数、非十进制数字、乘法溢出 | 配置错误，服务启动失败 |

也可以使用环境变量:

```sh
export LLAMA_ARG_PROMPT_CACHE_SSD_MAX_MIB=4096
```

正值必须同时提供已存在的 `--slot-save-path`。不要求启用
`--prompt-cache-ssd`；因此下面的配置也会管理手动 `save_incr` /
`restore_incr` 操作:

```sh
llama-server \
  -m model.gguf \
  --slot-save-path saves \
  --prompt-cache-ssd-max-mib 4096
```

`--prompt-cache-ssd` 仍然独立控制自动加载和自动保存。

### 1.2 配额到底统计什么

配额统计以下文件的**逻辑文件大小**:

```text
saves/
  seg/<hh>/<32 位小写十六进制 hash>
  rec/<hh>/<32 位小写十六进制 hash>
  seg/<hh>/<32 位小写十六进制 hash>.tmp
  rec/<hh>/<32 位小写十六进制 hash>.tmp
```

不统计:

- `session_*.bin` 等 GGSD session 提示文件;
- 传统完整 slot save 文件;
- 目录和目录项的分配开销;
- 文件系统稀疏块、元数据开销;
- 保存目录中的其它用户文件;
- 不符合 GGSD 命名规则的文件。

因此该参数限制的是 GGSD 内容缓存，而不是整个
`--slot-save-path` 的 apparent size。

---

## 第二部分:回收策略

### 2.1 高水位与低水位

设:

```text
M = --prompt-cache-ssd-max-mib 转换后的字节上限
L = floor(90% * M)
U = 当前已提交的 GGSD 字节
R = 当前写入或 grouped write 已预留的字节
```

如果 `U + R <= M`，直接允许写入。如果即将越过高水位，GC 在创建
`.tmp` 文件前同步执行，并将目标设置为:

```text
target = min(L, M - R)
```

这会在通常情况下回收到 90% 水位，为下一次写入留下空间。单个新对象
本身可以大于低水位，但永远不能使已预留或已提交字节越过高水位。

### 2.2 Leaf-LRU，而不是简单按文件删除

segment 不是相互独立的文件:

```text
A <- B <- C
       \- D
```

删除 `B` 会使 `C`、`D` 都失去可恢复性。因此 GC 只会删除:

1. 未保护的 `rec`，按最近使用时间从冷到热排序;
2. 没有子 segment、也没有 `rec` 依赖的 segment 叶子。

删除叶子后，父节点可能成为新的叶子；删除 rec 后，其尾段可能成为新的
候选。GC 每次删除后重新检查依赖关系，逐层剥离冷分支，不破坏保留下来的
前缀闭包。

候选排序为:

1. 无效对象优先;
2. mtime 更旧优先;
3. mtime 相同则文件更大者优先;
4. 仍相同则按 hash 稳定排序。

共享前缀只保留一份。只要仍有一个有效后代或 rec 依赖它，前缀就不会被删。

### 2.3 操作保护

保存触发 GC 时，当前保存链上最深的已有对象及其全部祖先会被保护。
因此 GC 不会为了给当前链追加新对象而删除这条链的前缀。

hybrid 保存会在能够完整测量待写 segment 和 rec 大小时，尝试一次性预留整个
写入组。group reservation 成功后，segment 与 rec 共用这份预留；测量不可用时
才退回逐对象 reservation。quota 拒绝只影响缓存写入，不影响模型推理。

---

## 第三部分:启动、写入与失败语义

### 3.1 服务启动时

启用正 quota 后，服务在接受请求前同步执行一次 pool reconciliation:

1. 规范化并检查 pool 路径;
2. 只枚举 `seg/` 和 `rec/` 下的规范 shard;
3. 统计合法对象与临时文件;
4. 删除遗留 `.tmp` 文件;
5. 解析并校验对象 header、内容 hash 和依赖关系;
6. 对超限池执行 GC，目标为 90% 水位。

软链接不会被跟随到 pool 外部。无法建立可信字节统计，或无法把超限池
回收到目标时，quota 配置失败，服务应拒绝启动，而不是静默降级为无限制。

### 3.2 写入中的临时文件

在创建 `.tmp` 之前，写入路径必须先取得精确序列化大小并完成 reservation。
reservation 计入硬上限，因此并发的临时对象不会把实际占用推过 quota。

- rename 成功:预留字节转为 committed bytes;
- 写入异常:释放未消费 reservation，并尽力删除临时文件;
- 临时文件删除失败:实际临时文件继续计入统计，写入失败;
- 文件逻辑大小与预期不符:不提交对象，缓存 admission 失败。

GGSD GC、quota 和磁盘 I/O 失败都只影响缓存覆盖，不影响模型推理。请求会
回退到正常 prefill 或保持已有的内存 KV 状态。

### 3.3 访问时间

GC 不使用文件系统 atime，因为 `noatime` 和 `relatime` 会使它不可靠。成功
使用对象后更新 mtime:

- 标准 restore:触碰最深的 restored segment;
- hybrid restore:触碰选中的 rec;
- 成功保存且对象已存在:触碰该保存可恢复链的最深对象;
- estimate、miss、被拒绝的 restore、损坏候选:不触碰。

同一个对象十分钟内最多触碰一次。触碰失败不会撤销成功的保存或恢复，
但会增加 `touch_failures` 并记录告警。

---

## 第四部分:监控与排障

启用 metrics 后，服务始终输出以下九个稳定的 Prometheus series；未配置
quota 时其值为零:

| 类型 | 指标 | 含义 |
|---|---|---|
| counter | `llamacpp:ggsd_gc_runs_total` | GC 执行次数 |
| counter | `llamacpp:ggsd_gc_deleted_bytes_total` | GC 删除的累计字节 |
| counter | `llamacpp:ggsd_gc_failures_total` | 未达到目标或其它 GC 失败次数 |
| counter | `llamacpp:ggsd_cache_writes_rejected_total` | 因 quota 拒绝的写入次数 |
| counter | `llamacpp:ggsd_cache_touch_failures_total` | mtime 更新失败次数 |
| gauge | `llamacpp:ggsd_cache_bytes` | 当前 GGSD 管理字节 |
| gauge | `llamacpp:ggsd_cache_limit_bytes` | 当前硬上限 |
| gauge | `llamacpp:ggsd_cache_segments` | 当前 segment 数量 |
| gauge | `llamacpp:ggsd_cache_rec_snapshots` | 当前 rec 数量 |

基本检查:

```sh
curl -s http://127.0.0.1:8080/metrics | grep ggsd
```

手动核对 managed bytes 时，只统计合法的 `seg/*/*`、`rec/*/*` 和匹配的
`.tmp` 文件；不要把 session 文件或传统 slot 文件加进去。

### 4.1 常见现象

**`ggsd_cache_writes_rejected_total` 增加**

可能原因:

- 单个对象或 grouped write 大于 quota;
- 没有足够的可回收叶子;
- 当前链被操作保护，不能删除;
- 文件删除失败;
- 启动扫描发现大量无效对象但无法清理。

这不是推理错误。先检查 `ggsd_cache_bytes`、
`ggsd_gc_failures_total` 和 quota/GC 相关日志。当前实现不会为每个删除对象
单独记录日志；应以 metrics 和 pool 文件状态为准。

**缓存字节低于预期但对象数量仍多**

GC 只删除可安全删除的叶子。共享前缀、仍被 rec 引用的尾段和当前操作
保护链都必须保留。

**重启后 quota 配置失败**

检查:

- pool 是否仍是原来的目录;
- `seg/`、`rec/` 和 shard 是否被替换成不可读对象;
- 是否有无法删除的 `.tmp`;
- 是否存在损坏对象或越界 header;
- 运行用户是否有读取、删除和更新 mtime 的权限。

不要手动删除仍被正在运行的 writer 使用的对象。GGSD 保持单进程/单 context
writer 约束，不提供跨进程锁。

---
## 第五部分:验证与当前限制

当前仓库提供 model-free 的 `test-ggsd-cache`，覆盖空 pool、segment
hash 链、父子依赖计数、quota 初始化和超限 reservation 拒绝:

```sh
cmake --build build-linux --target test-ggsd-cache -j
ctest --test-dir build-linux -R '^test-ggsd-cache$' --output-on-failure
```

参数解析和现有标准 GGSD 状态回归可以分别运行:

```sh
cmake --build build-linux --target test-arg-parser test-save-load-state -j
ctest --test-dir build-linux \
  -R '^(test-arg-parser|test-save-load-state)$' \
  --output-on-failure
```

`test-ggsd-cache --bench N` 规模 benchmark 当前尚未提供。当前实现也没有
为每次 GC 输出统一的 `__GGSD__ gc:` summary 日志；排障应使用 metrics、
服务错误/告警日志和 pool 文件扫描。

全量 CTest 可能依赖额外的 Python 包和外部 vocab 测试资源。失败时应区分
GGSD 测试失败与环境问题，不要仅根据全量 CTest 的总状态判断 GGSD pool
行为。

---

## 第六部分:边界与安全约束


- GC 在现有串行保存路径同步运行，没有后台 collector thread;
- 不提供运行时 quota 修改接口;
- 不支持多进程同时写同一个 pool;
- 不添加数据库、journal、持久化索引、pin 或 TTL;
- 无 quota 的用户保持原有无限制行为，且不会因为 GC 产生启动扫描;
- `max_bytes == 0` 关闭 quota 和 GC;
- 识别对象时严格要求 32 位小写十六进制 hash 与正确 shard，未知文件忽略;
- 保留 prefix closure:有效 segment 的祖先必须存在，rec 的 required tail 必须存在。

GC 的目标是控制 GGSD 缓存占用，而不是保证所有历史恢复点永久存在。需要
长期保留的状态应使用传统 slot save 或独立备份机制，不应依赖无 pin 的 LRU
缓存。
