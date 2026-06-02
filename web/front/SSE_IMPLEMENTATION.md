# SSE 实现说明

## 从 WebSocket 迁移到 SSE

项目已从 WebSocket 迁移到 Server-Sent Events (SSE) 架构。

### 主要变更

1. **移除文件**：`src/hooks/useWebSocket.ts`
2. **新增文件**：`src/hooks/useSSEChat.ts`
3. **修改文件**：`src/components/ChatArea.tsx`

### 工作原理

#### SSE vs WebSocket

- **WebSocket**：双向通信，服务器和客户端都可以主动发送消息
- **SSE**：单向通信，只有服务器可以主动发送消息给客户端

#### SSE 的优势

1. 更简单的实现（标准 HTTP 请求）
2. 自动重连机制
3. 更好的浏览器原生支持
4. 符合 HTTP 协议规范

#### 当前实现

1. 客户端发送 POST 请求到 `/api/chat/{sessionId}`
2. 服务器返回 SSE 流式响应
3. 客户端解析事件流并更新 UI

### API 协议

#### SSE 事件类型

```typescript
// 事件类型
type SSEEvent = 
  | { type: "start" }
  | { type: "think_status", phase: string, status?: string }
  | { type: "think_complete", metadata?: object }
  | { type: "sentence", content: string, emotion?: string, isLast?: boolean }
  | { type: "token", content: string }
  | { type: "dialogue", content: string }
  | { type: "demand_analysis", data?: object }
  | { type: "fragment_break" }
  | { type: "done" }
  | { type: "error", message: string }
  | { type: "status", phase?: string, agent?: string }
```

### 使用说明

#### 发送消息

```typescript
import { useSSEChat } from "../hooks/useSSEChat"

function MyComponent() {
  const { sendMessage, regenerate, isConnected } = useSSEChat(sessionId)
  
  // 发送消息
  sendMessage("你好")
  
  // 重新生成
  regenerate()
}
```

### 后端对应实现

后端需要实现 `/api/chat/{sessionId}` 接口：

- 方法：POST
- Content-Type：application/json
- 响应：SSE 流式响应

请求体：
```json
{
  "message": "用户输入内容"
}
```

响应格式：
```
data: {"type": "start"}

data: {"type": "think_status", "phase": "analyzing"}

data: {"type": "sentence", "content": "理解你的问题", "isLast": false}

data: {"type": "sentence", "content": "让我来帮你。", "isLast": true}

data: {"type": "done"}
```

### 注意事项

1. 发送消息时会先关闭上一个连接
2. SSE 连接在 `done` 事件后会自动关闭
3. 错误时会自动完成流并记录日志
4. 使用原生 fetch API 解析 SSE 格式
