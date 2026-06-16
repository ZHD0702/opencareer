---
name: Frontend bugfixes 2026-05-10
description: 前端点击事件和气泡显示问题的排查与修复记录
type: project
---

## 修复内容

### 修复 1：欢迎页建议点击无效（竞态条件）
- **文件**: `src/hooks/useWebSocket.ts`
- **根因**: session 创建是异步的，用户点击欢迎页建议时 WebSocket 尚未连接，`sendMessage()` 直接 return 丢弃消息
- **修复**: 添加 `messageQueueRef` 消息队列，`sendMessage()` 先将消息写入 store（用户立即看到气泡），再推入队列；`ws.onopen` 时自动冲刷队列

### 修复 2：AI 气泡全宽显示（CSS 冲突）
- **文件**: `src/components/MessageList.tsx:93`
- **根因**: `tailwind-merge` 按"后者胜出"合并，AI 分支中 `max-w-none`（来自 prose）出现在 `max-w-[75%]` 之后，导致无宽度限制
- **修复**: `max-w-none` 后紧跟响应式断点 `max-w-[90%] md:max-w-[80%] lg:max-w-[75%]`，按窗口大小自适应

### 非问题：空流式气泡
- 零宽空格 + 呼吸点动画 + padding 已足够，无需额外处理
