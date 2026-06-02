import { useState, useRef, useCallback } from 'react'
import { useChatStore } from '../stores/chatStore'

export function useSSEChat(session_id: string | null) {
  const [isConnected, setIsConnected] = useState(false)
  const { addMessage, setTyping, addFragment, finishStreaming, setStatus, setDemandAnalysis } = useChatStore()
  const eventSourceRef = useRef<EventSource | null>(null)

  const sendMessage = useCallback(async (message: string) => {
    if (!session_id) return
    try {
      // 先添加用户消息
      addMessage('user', message)
      
      // 开始请求
      const response = await fetch(`/api/chat/${session_id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
      })

      const reader = response.body?.getReader()
      if (!reader) return

      const decoder = new TextDecoder()
      let buffer = ''
      setIsConnected(true)

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.trim()) continue
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6).trim())
              switch (data.type) {
                case 'typing_start':
                  setTyping(true)
                  break
                case 'typing_end':
                  setTyping(false)
                  break
                case 'fragment':
                  addFragment(data.content, data.emotion)
                  break
                case 'done':
                  finishStreaming()
                  setIsConnected(false)
                  break
                case 'status':
                  setStatus(data.phase, data.agent)
                  break
                case 'demand_analysis':
                  setDemandAnalysis(data.data)
                  break
                case 'error':
                  console.error('SSE error:', data.message)
                  finishStreaming()
                  setIsConnected(false)
                  break
              }
            } catch (e) {
              console.error('Failed to parse SSE data:', e)
            }
          }
        }
      }
    } catch (error) {
      console.error('SSE error:', error)
      finishStreaming()
      setIsConnected(false)
    }
  }, [session_id, addMessage, setTyping, addFragment, finishStreaming, setStatus, setDemandAnalysis])

  const disconnect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }
  }, [])

  return { sendMessage, isConnected, disconnect }
}
