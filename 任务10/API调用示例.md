# 医学知识 RAG 问答服务 · API 调用示例

> 部署与配置见 [`部署文档.md`](部署文档.md)；机器可读的接口定义见 [`openapi.json`](openapi.json)
> 与在线 <http://127.0.0.1:8000/docs>；可直接导入的请求集合见
> [`medrag_api.postman_collection.json`](medrag_api.postman_collection.json)。
>
> 下面所有响应都是**真跑出来的**，不是手写的示意。

---

## 零、两条贯穿全篇的约定

**1. 统一响应体**——成功与失败**同一个形状**，靠 `code` 区分，`code == 0` 即成功：

```json
{
  "code": 0, "message": "ok", "data": { }, "detail": null,
  "request_id": "req-a8a52fe9", "timestamp": "2026-08-14 15:43:56", "elapsed_ms": 12.3
}
```

`request_id` 同时出现在响应体、`X-Request-Id` 响应头、日志行首、SQLite 调用记录里——**四处一致**，
出问题时报这一个值就够。

**2. Windows 上用 `curl.exe`，且命令行里的 query 一律写英文。**
PowerShell 的 `curl` 是 `Invoke-WebRequest` 的别名，参数完全不同；而中文内联进 JSON 会被终端
弄坏编码，服务端收到的是坏 JSON——**返回的是「请求体格式错误」而不是你想看的那个结果**。
要问中文，用下面的 PowerShell 或 Python 写法（它们显式指定 UTF-8）。

---

## 一、健康检查

```powershell
curl.exe -s http://127.0.0.1:8000/health
curl.exe -s http://127.0.0.1:8000/health/ready
```

`/health` 只答"进程还在不在"，**绝不碰下游**；`/health/ready` 会探 Ollama、stat 向量库与两个
SQLite。就绪探针的 `data.components`（六个组件，含任务书点名的 LLM / 向量库 / 数据库）：

```json
{"name": "vector_db",  "ok": true, "critical": false,
 "detail": "E:\\rag\\data\\chroma_db_4m｜64.6 GB／6 文件｜检索器未加载"}
{"name": "doc_catalog","ok": true, "critical": false, "detail": "2,274,167 篇，建于 2026-08-14 15:26:55"}
{"name": "llm:ollama", "ok": true, "critical": true,  "detail": "模型 qwen3:8b 已就绪", "latency_ms": 3.1}
```

关键组件挂了 → HTTP **503** + 业务码 **5002**；非关键的挂了 → 仍 200，`status=degraded`。

---

## 二、同步问答

```powershell
$body = @{ query = "法布里病有哪些酶替代治疗药物？"; top_k = 8 } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/qa/ask -Method Post `
  -ContentType "application/json; charset=utf-8" `
  -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 300 | ConvertTo-Json -Depth 6
```

⚠ **一次要 30~100 秒**（四段链 + 层 D 校验），客户端超时**必须设到 300 秒以上**，否则你看到的
会是客户端超时而不是服务端的问题。

`data` 里值得先看的几个字段：

| 字段 | 意思 |
|---|---|
| `answer` | 最终答案，每个事实句带 `[S#]` 出处 + 参考文献 + 免责声明 |
| `sources[]` | `[S#]` → PMCID / PMID / 期刊 / 年份 / 章节 / PMC 链接 |
| `retrieval_query` | **真正送进检索层的那句英文**。中文问题在这里被译过去 |
| `refused` / `partial` | 完全拒答 / 部分作答（互斥）。⚠ 拒答是**正确行为**，走 200 + code 0 |
| `citation_check` | `used / available / fabricated`——编造的编号会被揪出来 |
| `constraint_check` | 阶段九层 D 的合规判定与逐条违规 |
| `metrics` | 各段耗时、token、模型调用次数 |

> **排障要点**：怀疑"库里没有证据"时，先看 `retrieval_query`。如果它还是中文，
> 说明中译英降级了（见部署文档第七节），此时的拒答**不是**因为库里没有。

---

## 三、流式问答（SSE）

```powershell
curl.exe -N -s -X POST http://127.0.0.1:8000/api/v1/qa/stream `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"enzyme replacement therapy for Fabry disease\",\"top_k\":8}"
```

`-N` 关掉缓冲，否则看不到"流"。浏览器端用 `EventSource` 时只能发 GET，用 GET 版：

```javascript
const es = new EventSource("http://127.0.0.1:8000/api/v1/qa/stream?query=" + encodeURIComponent(q));
es.addEventListener("meta",    e => showAccepted(JSON.parse(e.data)));    // 受理即到
es.addEventListener("sources", e => renderSources(JSON.parse(e.data)));   // 0.03 秒就到
es.addEventListener("delta",   e => append(JSON.parse(e.data).text));     // 首字要 7.7~8.0 秒
es.addEventListener("done",    e => { finalize(JSON.parse(e.data)); es.close(); });
es.addEventListener("error",   e => { showError(JSON.parse(e.data)); es.close(); });
```

七种事件：`meta`（受理信息）/ `stage`（某段开始或结束）/ `sources`（出处清单）/
`delta`（文本增量）/ `check`（层 D 校验与修正）/ `done`（最终结果）/ `error`。
心跳不是事件，是 SSE 注释行（以 `:` 开头），客户端应当忽略它。

⚠ **最终答案以 `done` 为准**，`delta` 只是过程量（层 D 可能在收尾时改写答案）。
⚠ 流一旦开始，HTTP 状态行已经发出，**再也没法表达 HTTP 层的失败**——所以鉴权、限流、参数校验
都在开流之前做完，开流之后的错误只能以 `error` 事件出现。

---

## 四、会话（多轮对话）

```powershell
# 1) 建会话（不传 body 也行，服务端发 id）
curl.exe -s -X POST http://127.0.0.1:8000/api/v1/sessions `
  -H "Content-Type: application/json" -d "{\"title\":\"Fabry 咨询\"}"
# → {"code":0,"data":{"session_id":"sess-3f2a1c9b4d5e","turns":0,...}}

# 2) 带着 session_id 问，追问会被自动改写成可独立检索的问题
#    （"它们的不良反应呢？" → "这些酶替代治疗药物的不良反应是什么？"）

# 3) 看历史消息列表
curl.exe -s "http://127.0.0.1:8000/api/v1/sessions/sess-3f2a1c9b4d5e?turns=20"

# 4) 会话列表（分页）
curl.exe -s "http://127.0.0.1:8000/api/v1/sessions?page=1&page_size=20"

# 5) 删除
curl.exe -s -X DELETE http://127.0.0.1:8000/api/v1/sessions/sess-3f2a1c9b4d5e
```

**"添加消息"没有单独的接口**——按任务书写的，它由问答接口自动调用：每次 `/qa/ask` 或
`/qa/stream` 结束时写入一问一答两条轮次。所以 `turns` 是偶数。

⚠ 两个刻意的不对称，别当成 bug：
- **问答时传一个不存在的 `session_id` 会自动建会话**（聊天接口要求先建再问是多一次往返）；
- 但 `GET /api/v1/sessions/{id}` 上不存在就是 **3002**。一个是写入语义，一个是查询语义。

⚠ **历史绝不进证据区**。它只用来把追问改写成能独立检索的问题；把上一轮问答原文塞进上下文，
等于给模型一段"没有 `[S#]` 出处的事实材料"，阶段九压下去的编造会从这个口子回来。

---

## 五、文档管理

### 5.1 列表（过滤 + 游标分页）

```powershell
curl.exe -s "http://127.0.0.1:8000/api/v1/documents?journal=Nature&pub_year=2021&limit=2"
```

```json
{
  "code": 0,
  "data": {
    "items": [
      {"doc_id": "PMC7612144", "pmcid": "PMC7612144", "pmid": "34040253",
       "title": "MARK4 controls ischaemic heart failure through microtubule detyrosination",
       "journal": "Nature", "pub_year": 2021, "abstract": null,
       "total_chunks": 50, "indexed_chunks": 2, "sections": ["methods", "other"]}
    ],
    "limit": 2, "has_more": true, "next_cursor": "PMC7616976", "total": null,
    "filters": {"journal": "Nature", "pub_year": 2021}, "elapsed_ms": 1.01
  }
}
```

翻下一页：把 `next_cursor` 原样传回去。

```powershell
curl.exe -s "http://127.0.0.1:8000/api/v1/documents?journal=Nature&pub_year=2021&limit=2&cursor=PMC7616976"
```

支持的过滤：`journal`（精确）、`pub_year`、`year_from` / `year_to`、`title`（包含，大小写不敏感）。

**三件必须知道的事**：

1. **没有 `page` 参数**，只有 `cursor`。227 万行上 `LIMIT 20 OFFSET 100000` 要先扫掉前十万行，
   页码越深越慢。游标是上一页最后一条的 `pmcid`，翻到第几页都是常数代价（实测 0~1 ms）。
2. **`total` 默认是 `null`**。要总数就传 `with_total=true`：无过滤约 24 ms，
   配上标题模糊搜最坏约 1.5 s。翻页并不需要它，所以不默认付这笔钱。
3. **`total_chunks` 与 `indexed_chunks` 是两回事**：前者是原文切块数（平均 28.6），
   后者是真正进了 4M 向量库的条数（平均 1.76）。上面那篇原文 50 块，库里只有 2 块。

### 5.2 按 id 查

```powershell
curl.exe -s http://127.0.0.1:8000/api/v1/documents/PMC212698
```

查不到 → **3001 / HTTP 404**：

```json
{"code": 3001, "message": "文档不存在：PMC7000001", "data": null,
 "detail": {"doc_id": "PMC7000001",
            "hint": "该 ID 不在本地 4M 索引的文献目录中；本索引是 oa_comm 全量的抽样子集，查不到不代表 PMC 上没有这篇文献"}}
```

⚠ `abstract` 只有 **7.7%** 的文献有值（175,664 / 2,274,167），其余为 `null`——
4M 索引是按**块**抽样的，多数文献的摘要块没被抽中。这不是数据丢了，是抽样的必然结果。
⚠ 目录没建时返 **3004 / HTTP 503**，并在 `detail.fix` 里给出该跑哪条命令，
**不会返回一个空列表**——"查得到但空空如也"和"库没建"必须分得开。

---

## 六、运营统计

```powershell
curl.exe -s "http://127.0.0.1:8000/api/v1/qa/stats"
curl.exe -s "http://127.0.0.1:8000/api/v1/qa/stats?since_hours=24"
```

分两段：

**调用侧**（随请求变）：`total` 问答次数、`elapsed_ms.avg/p50/p95/max` 耗时、`success_rate` 成功率、
`refusal_rate` 拒答率、`compliant_rate` 层 D 合规率、`by_status` / `by_code` / `by_mode`、`tokens`。

**知识库侧**（`index` 段，只随重建索引变）：

```json
"index": {
  "available": true,
  "total_documents": 2274167,
  "total_chunks": 3998000,
  "documents_note": "被抽中至少一块的文献数。4M 索引是从 92,432,502 块里按**块**分层抽样得到的…",
  "index_size_bytes": 74021683071, "index_size_human": "68.9 GB",
  "incremental_updates": 0,
  "incremental_updates_note": "当前语料是 2026-06-18 冻结快照，一次全量构建，没有增量更新机制…",
  "last_index_built_at": {"vector_db": "2026-07-16T17:02:42",
                          "bm25_index": "2026-07-22T14:09:58",
                          "doc_catalog": "2026-08-14 15:26:55"}
}
```

⚠ 三个容易读错的地方：
- **`total_documents` ≠ 227 万篇完整文献在库里**。口径是"被抽中至少一块"，平均每篇只有 1.76 块
  进了库（原文平均 28.6 块），54.2% 的文献只有 1 块。所以这个字段自带 `documents_note`。
- **`incremental_updates: 0` 是如实报的**，不是"还没实现所以填 0"：语料是冻结快照、一次全量构建。
- **建库时间有三个**（向量库 / BM25 / 文献目录），不合成一个。
- 统计文件缺失时这些字段是 **`null` 而不是 `0`**，并且 `available=false`——
  0 是个看起来正常的假数字。

调用记录本身也能查：

```powershell
curl.exe -s "http://127.0.0.1:8000/api/v1/qa/logs?page=1&page_size=20&status=ok"
curl.exe -s http://127.0.0.1:8000/api/v1/qa/logs/req-a8a52fe9
```

---

## 七、元信息与鉴权

```powershell
curl.exe -s http://127.0.0.1:8000/                    # 服务概览 + 接口清单
curl.exe -s http://127.0.0.1:8000/api/v1/errors       # 21 个错误码，带 HTTP 状态映射
curl.exe -s http://127.0.0.1:8000/api/v1/config       # 生效配置（已脱敏）+ 每项来源
```

开了鉴权（`--api-key` 非空）之后，两种带法都认：

```powershell
curl.exe -s http://127.0.0.1:8000/api/v1/qa/stats -H "X-API-Key: 你的密钥"
curl.exe -s http://127.0.0.1:8000/api/v1/qa/stats -H "Authorization: Bearer 你的密钥"
```

缺 Key → **2001 / 401**；超过 `--rate-limit` → **2004 / 429**。

---

## 八、Python 客户端（问中文推荐用这个）

```python
import json, requests            # 或 httpx

BASE = "http://127.0.0.1:8000/api/v1"

sid = requests.post(f"{BASE}/sessions", json={"title": "Fabry 咨询"}).json()["data"]["session_id"]

r = requests.post(f"{BASE}/qa/ask",
                  json={"query": "法布里病有哪些酶替代治疗药物？", "top_k": 8,
                        "session_id": sid},
                  timeout=300)                     # ⚠ 必须放宽超时
b = r.json()
if b["code"] != 0:
    raise RuntimeError(f'{b["code"]} {b["message"]} {b["detail"]}')

d = b["data"]
print("检索式：", d["retrieval_query"])            # 中文问题被译成了什么
print("拒答：", d["refused"], "｜部分作答：", d["partial"])
print(d["answer"])
for s in d["sources"]:
    print(f'  {s["marker"]} {s["pmcid"]} {s.get("journal")} {s.get("pub_year")}')

# 追问：不用自己拼上下文，传同一个 session_id 即可
r2 = requests.post(f"{BASE}/qa/ask",
                   json={"query": "它们的不良反应呢？", "session_id": sid}, timeout=300)
print(r2.json()["data"]["resolved_query"])         # 改写后的独立问题
```

流式：

```python
import httpx, json
with httpx.stream("POST", f"{BASE}/qa/stream", json={"query": "..."}, timeout=300) as r:
    event = None
    for line in r.iter_lines():
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: "):
            data = json.loads(line[6:])
            if event == "sources":
                print("出处已到：", len(data["sources"]))     # 0.03 秒
            elif event == "delta":
                print(data["text"], end="", flush=True)
            elif event == "done":
                # ⚠ done 的载荷是**完整的统一响应体**（含 code/message），
                #    业务数据在 data["data"] 里，与同步接口的 data 字段结构一致
                final = data["data"]                          # ← 以这个为准
```

---

## 九、Postman

导入 [`medrag_api.postman_collection.json`](medrag_api.postman_collection.json)，
把集合变量 `baseUrl` 指向你的服务（默认 `http://127.0.0.1:8000`），开了鉴权就再填 `apiKey`。

- **44 条请求，覆盖全部 18 个端点**（100%），按验证脚本的分组归类。
- 每条的断言（HTTP 状态 + 业务码 + 统一响应体字段）**取自验证脚本那一轮的真实响应**，
  不是手写的期望值——集合是**从真跑里导出来的**，不是另写的一套测试。
- 顺序跑：建会话、问答、文献列表那三条会把 `sessionId` / `requestId` / `docId` 写进集合变量，
  后面几条才有得用（`docId` 有个默认值 `PMC212698`，单独跑详情那条也不会空）。
- ⚠ `qa/ask` 与 `qa/stream` 真跑一次 30~100 秒，Postman 的请求超时要调大。
- ⚠ 集合只覆盖走 HTTP 的那部分。验证脚本 223 项里还有一大批**进程内断言**
  （响应模型边界、错误码表、SQLite 存储、参数校验在模型层就被拦下），它们不经过 HTTP，
  **集合全绿 ≠ 那 223 项全绿**。两者是互补关系。
