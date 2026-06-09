import { useState, useRef, useCallback } from 'react'
import { useChatStore } from '../stores/chatStore'

export function useSSEChat(sessionId: string | null) {
  const [isConnected, setIsConnected] = useState(false)
  const { addMessage, appendToken, setDialogue, breakFragment, finishStreaming, setStatus, setDemandAnalysis, isStreaming } = useChatStore()
  const isStreamingRef = useRef(isStreaming)
  isStreamingRef.current = isStreaming

  const processSSEEvent = useCallback((data: any) => {
    try {
      const msg = JSON.parse(data)
      console.log('[SSE] 收到事件:', msg)
      switch (msg.type) {
        case 'dialogue':
          setDialogue(msg.content)
          break
        case 'demand_analysis':
          setDemandAnalysis(msg.data || null)
          break
        case 'token':
          appendToken(msg.content)
          break
        case 'sentence':
          if (msg.content) {
            console.log('[SSE] 处理句子:', msg.content)
            for (const char of msg.content) {
              appendToken(char)
            }
            // 如果不是最后一个句子，创建新气泡
            if (!msg.is_last) {
              console.log('[SSE] 创建新气泡')
              breakFragment()
            }
          }
          break
        case 'fragment_break':
          breakFragment()
          break
        case 'done':
          console.log('[SSE] 完成')
          finishStreaming()
          setIsConnected(false)
          break
        case 'error':
          console.error('[SSE] Error:', msg.message)
          finishStreaming()
          setIsConnected(false)
          break
        case 'status':
          setStatus(msg.phase || '', msg.agent || null)
          break
        case 'think_status':
          setStatus(msg.phase || 'thinking', null)
          break
        case 'start':
          console.log('[SSE] 开始连接')
          setIsConnected(true)
          break
      }
    } catch (e) {
      console.error('[SSE] 解析消息失败:', data, e)
    }
  }, [appendToken, setDialogue, breakFragment, finishStreaming, setStatus, setDemandAnalysis])

  const sendMessage = useCallback(async (content: string) => {
    if (!sessionId) return
    console.log('[SSE] 发送消息:', content)

    addMessage('user', content)
    addMessage('ai')

    try {
      const url = `/api/chat/${sessionId}`
      console.log('[SSE] 请求 URL:', url)
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: content }),
      })

      if (!response.ok) {
        console.error('[SSE] HTTP错误:', response.status)
        throw new Error(`HTTP ${response.status}`)
      }

      const reader = response.body?.getReader()
      if (!reader) return

      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const events = buffer.split('\n\n')
        buffer = events.pop() || ''

        for (const event of events) {
          if (!event.trim()) continue

          const dataMatch = event.match(/^data:\s*(.+)$/m)
          if (dataMatch) {
            processSSEEvent(dataMatch[1])
          }
        }
      }
    } catch (e) {
      console.error('[SSE] Error:', e)
      finishStreaming()
    }
  }, [sessionId, addMessage, processSSEEvent, finishStreaming])

  const regenerate = useCallback(async () => {
    if (!sessionId) return
    if (isStreamingRef.current) return

    const { messages } = useChatStore.getState()
    const lastUser = [...messages].reverse().find((m) => m.role === 'user')
    if (!lastUser) return

    useChatStore.getState().removeLastMessage()
    addMessage('user', lastUser.content)
    addMessage('ai')

    sendMessage(lastUser.content)
  }, [sessionId, addMessage, sendMessage])

  return { sendMessage, regenerate, isConnected }
}
