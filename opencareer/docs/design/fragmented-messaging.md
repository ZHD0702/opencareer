# 碎片化消息发送 — 设计文档

> 状态：待实现
> 方案：B — LLM 标记驱动 + 后端流式解析
> 日期：2026-05-07

---

## 1. 目标

让 AI 回复模拟微信聊天式的"碎片化发送"——一个完整的 AI 回复被拆成多条短消息，每条短消息是独立气泡，依次出现。用户在视觉和节奏上感觉像是在和一个真人聊天。

**当前行为**：整个 AI 回复 = 一个大气泡，token 逐个追加。
**目标行为**：AI 回复自动拆分为多个短气泡，每泡 1-3 句话，泡间有自然停顿。

---

## 2. 核心设计决策

| 决策 | 选项 | 结论 |
|------|------|------|
| 断点由谁判断 | 前端规则 / 后端规则 / LLM 标记 | **LLM 标记**（语义最自然） |
| 标记格式 | `<split/>` / `⏎` / `[BR]` | **`<split/>`**（不易与正常对话混淆） |
| 泡内流式 | 有（token 逐个出现）/ 无（整泡出现） | **有**（保持 AI 打字的体验感） |
| 泡间延迟 | 前端控制 / 后端控制 | **前端控制**（随机延迟更自然） |
| 无标记 fallback | 回退单泡 / 后端智能断句 | **回退单泡**（保持兼容，后续可升级到方案 C） |
| 单次回复最大片段数 | 无限制 / Prompt 约束 / Backend 硬限制 | **Prompt 约束 + Backend 硬限制**（双重保障） |
| 片段间信息独立性 | 同一个回复多话题 / 一个回复聚焦一个主题 | **一个回复聚焦一个主题**（Prompt 约束） |

---

## 3. 信息过载防护

碎片化的初衷是降低用户的认知负荷，模拟真人聊天的节奏感。但如果一次回复被拆成过多片段，或不同片段包含互不相关的信息点，反而会适得其反——用户需要在短时间内处理多个气泡中的不同信息。

### 3.1 设计原则

```
一个 AI 回复 = 一个核心主题
一个片段     = 该主题的一个信息点
多个片段     = 信息点的递进展开（而非并列堆砌）
```

**反面案例**（应禁止）：
```
气泡1: "面试紧张是很正常的"
气泡2: "你的简历需要加上量化成果"      ← 切换到简历话题
气泡3: "薪资谈判时要注意这三点"         ← 又跳到薪资
气泡4: "另外你还可以看看Boss直聘"       ← 再跳到求职渠道
```
问题：4 个气泡、4 个不同话题，用户不知道该回应哪个。

**正面案例**（期望）：
```
气泡1: "面试紧张是很正常的，很多人都这样"              ← 共情
气泡2: "这里有两个帮你缓解紧张的方法："                ← 过渡到方法
气泡3: "第一，提前准备常见问题；第二，把面试当成双向交流" ← 核心建议
气泡4: "你觉得这两个方法适合你吗？"                    ← 收尾引导
```
4 个气泡、1 个主题，层层递进，用户只需回应最后一个问题。

### 3.2 双层防护

**第一层 — Prompt 约束（主防线）**：

在 system prompt 中明确：
- 一次回复最多使用 2-5 个 `<split/>`（即 3-6 个气泡）
- 一个回复只讨论一个核心主题，不要在不同片段中切换话题
- 如果确实需要涉及多个主题，选择最重要的一个先回复，其余留到后续对话

**第二层 — Backend 硬限制（兜底）**：

`FragmentStreamer` 内部维护片段计数器，超过上限（默认 6）后忽略后续 `<split/>` 标记，将它们视为普通文本继续追加到当前气泡。这确保即使 Prompt 约束失效，最坏情况下也不会出现 10+ 个气泡的情况。

---

## 4. 整体数据流

```
User 输入
  │
  ▼
Frontend: sendMessage()
  ├─ addMessage("user", content)
  ├─ addMessage("ai")          ← 乐观创建第一个 AI 气泡
  └─ ws.send({type:"chat"})
        │
        ▼
Backend: AgentPipeline.process_message()
  ├─ Brain 分析（不变）
  ├─ ws {type:"dialogue"}     ← brain 响应（不变）
  ├─ ws {type:"demand_analysis"}（不变）
  ├─ ws {type:"status"}       ← 切换到下游 agent（不变）
  │
  └─ _stream_work_agent / _stream_emotion_agent  ← 核心改动
        │
        ▼
     FragmentStreamer（新增）
        │  缓冲 token，检测 <split/> 标记
        │  剥离标记，发送干净的 token
        │  检测到标记时发送 fragment_break
        │
        ├─ ws {type:"token", content:"今天面试还不错"}
        ├─ ws {type:"token", content:"，面试官问"}
        ├─ ... （检测到 <split/>）
        ├─ ws {type:"fragment_break"}        ← 新增消息类型
        ├─ ws {type:"token", content:"面试官问了几个算法题"}
        ├─ ... （检测到 <split/>）
        ├─ ws {type:"fragment_break"}
        ├─ ws {type:"token", content:"基本都答上来了"}
        │
        └─ ws {type:"done"}                  ← 最后一个 fragment 结束
              │
              ▼
Frontend: useWebSocket 消息处理
  ├─ "token"          → appendToken()        ← 追加到最后一个 isStreaming 气泡
  ├─ "fragment_break" → breakFragment()      ← 封口当前泡，创建新泡（带延迟）
  └─ "done"           → finishStreaming()    ← 标记最后泡为完成
```

---

## 5. 改动范围

### 5.1 System Prompt — 新增碎片化指令

**文件**：`demo/opencareer/prompts/personas/base.yaml`
**位置**：`communication_principles` 和 `response_structure` 两个 section

在 `communication_principles` 末尾追加：

```yaml
  8. **碎片化分段**：你的回复会通过 <split/> 标记自动拆分为多条短消息，
     就像微信聊天一样。你需要在自然的话锋转换处（思路切换、话题微调、
     从共情转到建议等）插入 <split/> 标记。每条短消息应控制在 1-3 句话。
     不要在句子中间插入标记。

  9. **信息密度控制**：一次回复最多使用 3-4 个 <split/>（即 4-5 个气泡），
     一个回复只围绕一个核心主题展开。不要在一次回复中切换多个不相关话题。
     如果用户提出了多个问题，选择最重要的一个先回复，
     其余留到后续对话中处理。记住：碎片化的目的是让用户感觉在和真人聊天，
     而不是被信息轰炸。
```

在 `response_structure` 末尾追加：

```yaml
  ### 碎片化分段规则

  你的完整回复将被拆分成多条短消息发送给用户。请在以下位置插入 <split/> 标记：

  - **共情确认 → 核心内容** 之间：先表达理解，<split/>，再进入建议
  - **话题微调** 时：从一个建议点切换到另一个建议点时
  - **信息量较大** 时：避免单条消息超过 3 句话
  - **反问引导** 前：核心内容说完，<split/>，然后提问

  **数量限制**：
  - 一次回复最多使用 3-4 个 <split/>（即 4-5 个气泡）
  - 如果内容确实需要更多分段，选择最重要的 4-5 个信息点发送，其余后续再聊
  - 一个回复只围绕一个核心主题，不要在不同气泡中讨论不同话题

  示例格式（4 个气泡，围绕"面试紧张"一个主题）：
  "我能理解你的感受，面试紧张是很正常的<split/>这里有两个可以帮助你的方法：<split/>第一，提前准备常见问题的回答框架；第二，把面试当成双向交流而非考试<split/>你觉得这些对你有帮助吗？"

  注意：标记前后不要加空格或换行，直接拼接在文字后面即可。
```

### 5.2 后端 — FragmentStreamer 类

**文件**：`demo/web_api/agent_pipeline.py`（或新建 `demo/web_api/fragment_streamer.py`）

新增 `FragmentStreamer` 类，负责：

```
输入：LLM 的 token 流 + send_json 回调
输出：剥离 <split/> 后的干净 token + fragment_break 信号

内部状态机：
┌──────────────────────────────────────────────────────────┐
│  buffer = ""                                             │
│  fragment_count = 0                                      │
│                                                           │
│  async for token in llm_stream:                          │
│    buffer += token                                        │
│                                                           │
│    if "<split/>" in buffer:                              │
│      if fragment_count >= MAX_FRAGMENTS:                 │
│        flush safe prefix (忽略标记，作为普通文本)          │
│        continue                                          │
│      before, after = buffer.split("<split/>", 1)         │
│      if before: send_json({type:"token", before})         │
│      send_json({type:"fragment_break"})                  │
│      fragment_count += 1                                 │
│      buffer = after                                       │
│                                                           │
│    else if buffer has no partial marker match:           │
│      flush all                                            │
│    else:                                                  │
│      flush safe prefix (can't be start of <split/>)      │
│                                                           │
│  on stream end: flush remaining buffer                   │
└──────────────────────────────────────────────────────────┘
```

**核心逻辑 — 部分标记匹配检测**：

标记 `<split/>` 共 8 个字符。Token 可能把标记切碎（如 `"<spli"` + `"t/>"`），所以不能简单用 `"<split/>" in buffer` 然后立即 flush。

采用**前缀匹配缓冲区**策略：

```python
MARKER = "<split/>"

def _safe_prefix_len(self, text: str) -> int:
    """返回 text 中可以安全发送的长度（不可能是 MARKER 前缀的部分）"""
    # 找到 text 的最长后缀，恰好是 MARKER 的某个前缀
    for i in range(1, len(MARKER)):
        prefix = MARKER[:i]       # "<", "<s", "<sp", "<spl", "<spli", "<split", "<split/"
        if text.endswith(prefix):
            return len(text) - len(prefix)
    return len(text)  # 没有部分匹配，全部安全
```

这确保了即使 token 边界切在标记中间，也不会把标记的一部分发给前端。

**完整接口**：

```python
class FragmentStreamer:
    MARKER = "<split/>"
    MAX_FRAGMENTS = 6  # 最多允许 6 个片段（即 5 个 <split/> 标记）

    def __init__(self, send_json: Callable[[dict], Any]):
        self.send_json = send_json
        self.buffer = ""
        self._fragment_count = 0

    async def feed(self, token: str) -> None:
        """喂入一个 token，自动处理标记检测和发送"""
        self.buffer += token

        if self.MARKER in self.buffer:
            if self._fragment_count >= self.MAX_FRAGMENTS:
                # 已达片段上限，忽略后续 <split/> 标记，作为普通文本处理
                safe_len = self._safe_prefix_len(self.buffer)
                if safe_len > 0:
                    safe = self.buffer[:safe_len]
                    await self.send_json({"type": "token", "content": safe})
                    self.buffer = self.buffer[safe_len:]
                return

            before, after = self.buffer.split(self.MARKER, 1)
            if before:
                await self.send_json({"type": "token", "content": before})
            await self.send_json({"type": "fragment_break"})
            self._fragment_count += 1
            self.buffer = after
        else:
            safe_len = self._safe_prefix_len(self.buffer)
            if safe_len > 0:
                safe = self.buffer[:safe_len]
                await self.send_json({"type": "token", "content": safe})
                self.buffer = self.buffer[safe_len:]

    async def flush(self) -> None:
        """流结束时发送剩余缓冲区"""
        if self.buffer:
            await self.send_json({"type": "token", "content": self.buffer})
            self.buffer = ""

    def _safe_prefix_len(self, text: str) -> int:
        for i in range(1, len(self.MARKER)):
            if text.endswith(self.MARKER[:i]):
                return len(text) - i
        return len(text)
```

**集成到 AgentPipeline**：

`_stream_work_agent` 和 `_stream_emotion_agent` 中各创建一个 `FragmentStreamer` 实例，替换原有的直接 `send_json({"type": "token", ...})`：

```python
# 改前
async for token in self.llm_client.stream_chat(...):
    full_response += token
    await send_json({"type": "token", "content": token})

# 改后
streamer = FragmentStreamer(send_json)
async for token in self.llm_client.stream_chat(...):
    full_response += token
    await streamer.feed(token)
await streamer.flush()
```

### 5.3 WebSocket 协议 — 新增消息类型

**文件**：`demo/web_api/models.py`

新增服务端 → 客户端消息类型：

```python
# 在 WsMessage 或相关文档中新增
{"type": "fragment_break"}  # 表示当前气泡封口，下一个 token 属于新气泡
```

无需新增客户端 → 服务端消息类型。

### 5.4 前端 Store — 新增 breakFragment action

**文件**：`demo/web/src/stores/chatStore.ts`

新增 action：

```typescript
breakFragment: () => {
  set((s) => {
    const msgs = [...s.messages]
    const last = msgs[msgs.length - 1]
    if (last && last.isStreaming) {
      // 封口当前气泡
      msgs[msgs.length - 1] = { ...last, isStreaming: false }
    }
    // 立即创建新的空气泡，准备接收下一个 fragment 的 token
    const id = String(nextId++)
    const newMsg: Message = {
      id,
      role: "ai",
      content: "",
      isStreaming: true,
      timestamp: Date.now(),
    }
    return { messages: [...msgs, newMsg] }
  })
}
```

**与现有 action 的交互**：

- `appendToken`：现有逻辑不变——始终追加到 `messages` 中最后一个 `isStreaming === true` 的消息。`breakFragment` 创建的新气泡天然成为 append 目标。
- `finishStreaming`：现有逻辑不变——封口最后一个气泡，设置 `isStreaming: false`。
- `addMessage("ai")`：当 `sendMessage` 被调用时，乐观创建第一个 AI 气泡。这个气泡是第一个 fragment 的容器。

### 5.5 前端 WebSocket Hook — 处理新消息类型

**文件**：`demo/web/src/hooks/useWebSocket.ts`

在 `ws.onmessage` 的 switch 中新增：

```typescript
case "fragment_break":
  breakFragment()
  break
```

`breakFragment` 从 chatStore 中解构获取。

### 5.6 前端 MessageList — 碎片气泡的视觉时序

**文件**：`demo/web/src/components/MessageList.tsx`

每个新 fragment 气泡在刚创建时 `content` 为空，只显示 `BreathingDot`。随后 token 到达，内容开始填充。这天然形成了"对方正在输入..."的视觉效果。

**改动**：当前 `BreathingDot` 在 `msg.isStreaming &&` 条件下显示。无需额外改动——`breakFragment` 创建的新气泡 `isStreaming: true`，自动显示呼吸点。

**可选增强**（后续迭代）：
- 新 fragment 气泡出现时带 `motion.div` 入场动画
- 每个 fragment 气泡之间添加肉眼可见的短暂间隔（通过 CSS transition-delay）

---

## 6. 异常场景处理

### 6.1 LLM 未生成任何 `<split/>` 标记

`FragmentStreamer` 的行为等于普通 streaming——所有 token 直接透传，流结束时 `flush()` 发送剩余内容。最终效果和改动前一致：一个完整气泡。

→ **自动降级，无需额外处理。**

### 6.2 LLM 过度使用 `<split/>`（每条只有几个字）

Frontend 层面不做限制（保持简洁）。如果体验不好，后续可在 system prompt 中加约束（如"每条至少 15 个字"），或在后端添加最短 fragment 合并逻辑。

### 6.3 片段数超过上限（信息过载）

当 LLM 在一个回复中使用过多 `<split/>` 标记（超过 `MAX_FRAGMENTS`，默认 6），`FragmentStreamer` 忽略后续标记，将其视为普通文本追加到最后一个气泡。同时发送一个 `fragment_limit_reached` 日志事件供调试。

```python
# FragmentStreamer 内部
MAX_FRAGMENTS = 6

async def feed(self, token: str) -> None:
    self.buffer += token

    if self.MARKER in self.buffer:
        if self._fragment_count >= self.MAX_FRAGMENTS:
            # 已达上限，忽略标记，作为普通文本处理
            safe_len = self._safe_prefix_len(self.buffer)
            if safe_len > 0:
                safe = self.buffer[:safe_len]
                await self.send_json({"type": "token", "content": safe})
                self.buffer = self.buffer[safe_len:]
            return

        before, after = self.buffer.split(self.MARKER, 1)
        if before:
            await self.send_json({"type": "token", "content": before})
        await self.send_json({"type": "fragment_break"})
        self._fragment_count += 1
        self.buffer = after
    else:
        # ... 原有安全前缀逻辑
```

这确保即使在最坏情况下（Prompt 约束完全失效），一次回复也不会超过 6 个气泡。

### 6.4 用户快速连续发送消息

当前 `isStreaming` 全局标志阻止发送新消息，逻辑不变。在 fragment 之间（旧泡已封口、新泡尚未完成），`isStreaming` 仍为 `true`，阻止用户输入。这是正确的——一个完整的 AI 回复（包含多个 fragment）期间不应被打断。

### 6.5 重新生成（regenerate）

现有 `regenerate` 逻辑：移除最后一条 AI 消息和它前面的用户消息，重新发送。对于碎片化回复，`removeLastMessage()` 只移除最后一个 AI 气泡。需要改为：移除所有属于最后一次回复的 AI 气泡。

**改动**：

```typescript
// chatStore.ts - removeLastMessage 改为移除连续的后缀 AI 消息
removeLastMessage: () => {
  set((s) => {
    const msgs = [...s.messages]
    // 移除所有后缀的 AI 消息（对应碎片化回复的多个气泡）
    while (msgs.length > 0 && msgs[msgs.length - 1].role === "ai") {
      msgs.pop()
    }
    // 移除触发这次回复的用户消息
    if (msgs.length > 0 && msgs[msgs.length - 1].role === "user") {
      msgs.pop()
    }
    return { messages: msgs, isStreaming: false }
  })
}
```

### 6.6 复制按钮行为

当前 `MessageBubble` 的复制按钮只复制单条 `msg.content`。碎片化后每个气泡独立可复制，符合预期——不需要改。

---

## 7. 测试要点

### 7.1 单元测试 — FragmentStreamer

| 用例 | 输入 token 序列 | 期望输出 |
|------|----------------|---------|
| 正常分段 | `["今天不错", "<split/>", "继续努力"]` | `token:"今天不错"`, `fragment_break`, `token:"继续努力"` |
| 标记被切碎 | `["今天不错<spli", "t/>继续努力"]` | `token:"今天不错"`, `fragment_break`, `token:"继续努力"` |
| 无标记 | `["今天不错，继续努力"]` | `token:"今天不错，继续努力"` |
| 连续标记 | `["a<split/><split/>b"]` | `token:"a"`, `fragment_break`, `fragment_break`, `token:"b"` |
| 标记在开头 | `["<split/>你好"]` | `fragment_break`, `token:"你好"` |
| 标记在结尾 | `["你好<split/>"]` | `token:"你好"`, `fragment_break` |
| 空输入 | `[]` | flush 后无输出 |
| 超过最大片段数 | `["a<split/>b<split/>c<split/>d<split/>e<split/>f<split/>g<split/>h"]` | 前 6 个 fragment_break 正常，后续 `<split/>` 作为普通文本出现在第 6 个气泡中 |

### 7.2 集成测试

1. 启动后端 + 前端，发送一条普通消息
2. 确认 AI 回复按照 system prompt 规则分段
3. 确认每个气泡独立显示，无 `<split/>` 文本泄漏到 UI
4. 确认"重新生成"能正确清除所有 fragment 气泡
5. 确认 copy 按钮独立工作
6. 确认 dark mode 下气泡样式正常
7. 构造一条会被拆成 10+ 个气泡的回复，确认后端硬限制生效（不超过 6 个气泡）

---

## 8. 文件变更清单

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `demo/opencareer/prompts/personas/base.yaml` | 修改 | 新增碎片化分段规则到 communication_principles 和 response_structure |
| `demo/web_api/agent_pipeline.py` | 修改 | 在 `_stream_work_agent` 和 `_stream_emotion_agent` 中集成 FragmentStreamer |
| `demo/web_api/models.py` | 修改 | 文档化 `fragment_break` 消息类型（如适用） |
| `demo/web/src/stores/chatStore.ts` | 修改 | 新增 `breakFragment` action，修改 `removeLastMessage` |
| `demo/web/src/hooks/useWebSocket.ts` | 修改 | 处理 `fragment_break` 消息，解构 `breakFragment` |

**不改动的文件**：
- `llm_client.py` — 流式 API 不变
- `MessageList.tsx` — 气泡渲染逻辑无需改动
- `ChatInput.tsx` — 输入框逻辑不变
- `ChatArea.tsx` — 编排逻辑不变

---

## 9. 后续演进路径

1. **方案 C 升级**：当 LLM 频繁漏加 `<split/>` 时，在 `FragmentStreamer` 中叠加后端智能断句（检测中文标点 `。！？` 作为 fallback 断点）
2. **自适应延迟**：根据 fragment 长度动态调整泡间延迟（长消息后停顿更久，模拟"打字时间"）
3. **用户偏好开关**：允许用户在设置中关闭碎片化模式，回到单泡模式
4. **语音消息模拟**：类似碎片化，但以"语音气泡"形式随机出现（模拟微信语音消息的体验）
