import { useState, useRef, useCallback } from 'react'
import { useChatStore } from '../stores/chatStore'

export function useSSEChat(session_id: string | null) {
  const [isConnected, setIsConnected] = useState(false)
  const {
    addMessage,
    beginStreaming,
    setTyping,
    addFragment,
    finishStreaming,
    setStatus,
    setDemandAnalysis,
    setEmotionAnalysis,
    setResumeUpdate,
    setResumePdf,
    attachAction,
  } = useChatStore()
  const eventSourceRef = useRef<EventSource | null>(null)

  const sendMessage = useCallback(async (message: string, sessionIdOverride?: string) => {
    const activeSessionId = sessionIdOverride || session_id
    if (!activeSessionId) return

    const controller = new AbortController()
    let inactivityTimer: ReturnType<typeof setTimeout> | null = null
    let receivedTerminalEvent = false
    const resetInactivityTimer = () => {
      if (inactivityTimer) clearTimeout(inactivityTimer)
      inactivityTimer = setTimeout(() => controller.abort(), 65000)
    }

    try {
      // Show the outgoing message and typing bubble immediately.
      addMessage('user', message)
      beginStreaming()

      const response = await fetch(`/api/chat/${activeSessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
        signal: controller.signal,
      })

      if (!response.ok) {
        throw new Error(`Chat request failed: ${response.status}`)
      }

      const reader = response.body?.getReader()
      if (!reader) {
        setTyping(false)
        return
      }

      const decoder = new TextDecoder()
      let buffer = ''
      setIsConnected(true)
      resetInactivityTimer()

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        resetInactivityTimer()

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (!line.trim()) continue
          if (!line.startsWith('data: ')) continue

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
                receivedTerminalEvent = true
                finishStreaming()
                setTyping(false)
                setIsConnected(false)
                break
              case 'status':
                setStatus(data.phase, data.agent)
                break
              case 'demand_analysis':
                setDemandAnalysis(data.data)
                break
              case 'emotion_analysis':
                setEmotionAnalysis(data.data)
                break
              case 'resume_update':
                setResumeUpdate(data.data)
                break
              case 'resume_pdf':
                setResumePdf(data.data, true)
                break
              case 'assistant_action':
                attachAction(data.data)
                break
              case 'error':
                receivedTerminalEvent = true
                console.error('SSE error:', data.message)
                addFragment(data.message || '刚才这条消息没有处理成功，请再试一次。', 'serious')
                finishStreaming()
                setTyping(false)
                setIsConnected(false)
                break
            }
          } catch (e) {
            console.error('Failed to parse SSE data:', e)
          }
        }
      }
    } catch (error) {
      console.error('SSE error:', error)
      if (!receivedTerminalEvent) {
        const message = controller.signal.aborted
          ? '这次回复等待太久，已经自动停止了。请再发一次，我会接着当前对话继续。'
          : '刚才这条消息没有处理成功。你可以稍等一下再发，我会接着当前对话继续。'
        addFragment(message, 'serious')
      }
    } finally {
      if (inactivityTimer) clearTimeout(inactivityTimer)
      finishStreaming()
      setTyping(false)
      setIsConnected(false)
    }
  }, [session_id, addMessage, beginStreaming, setTyping, addFragment, finishStreaming, setStatus, setDemandAnalysis, setEmotionAnalysis, setResumeUpdate, setResumePdf, attachAction])

  const disconnect = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }
  }, [])

  return { sendMessage, isConnected, disconnect }
}
