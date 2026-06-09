import { useEffect, useRef, useCallback, useState } from "react"
import { useChatStore } from "../stores/chatStore"

export function useWebSocket(sessionId: string | null) {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>()
  const messageQueueRef = useRef<string[]>([])
  const [isConnected, setIsConnected] = useState(false)
  const { addMessage, appendToken, setDialogue, breakFragment, finishStreaming, setStatus, setDemandAnalysis, isStreaming } = useChatStore()
  const isStreamingRef = useRef(isStreaming)
  isStreamingRef.current = isStreaming

  const connect = useCallback(() => {
    if (!sessionId) return

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:"
    const host = window.location.hostname
    const url = `${protocol}//${host}:8080/ws/${sessionId}`

    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      console.log("[WS] Connected:", sessionId)
      setIsConnected(true)
      // Flush queued messages
      const queue = messageQueueRef.current
      messageQueueRef.current = []
      for (const content of queue) {
        ws.send(JSON.stringify({ type: "chat", content }))
      }
    }

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        switch (msg.type) {
          case "dialogue":
            setDialogue(msg.content)
            break
          case "demand_analysis":
            setDemandAnalysis(msg.data || null)
            break
          case "token":
            appendToken(msg.content)
            break
          case "fragment_break":
            breakFragment()
            break
          case "done":
            finishStreaming()
            break
          case "error":
            console.error("[WS] Error:", msg.message)
            finishStreaming()
            break
          case "status":
            setStatus(msg.phase, msg.agent || null)
            break
          case "pong":
            break
        }
      } catch {
        console.error("[WS] Failed to parse message:", event.data)
      }
    }

    ws.onclose = () => {
      console.log("[WS] Disconnected")
      setIsConnected(false)
      wsRef.current = null
      // Reconnect after 2s
      reconnectTimer.current = setTimeout(connect, 2000)
    }

    ws.onerror = (err) => {
      console.error("[WS] Error:", err)
      ws.close()
    }
  }, [sessionId, addMessage, appendToken, setDialogue, breakFragment, finishStreaming, setStatus, setDemandAnalysis])

  useEffect(() => {
    connect()
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      wsRef.current?.close()
    }
  }, [connect])

  const sendMessage = useCallback((content: string) => {
    addMessage("user", content)
    addMessage("ai")
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      // Queue message for when WebSocket connects
      messageQueueRef.current.push(content)
      return
    }
    wsRef.current.send(JSON.stringify({ type: "chat", content }))
  }, [addMessage])

  const regenerate = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return
    if (isStreamingRef.current) return
    const { messages } = useChatStore.getState()
    // Find the last user message
    const lastUser = [...messages].reverse().find((m) => m.role === "user")
    if (!lastUser) return
    // Remove last AI message and add new empty one
    useChatStore.getState().removeLastMessage()
    addMessage("user", lastUser.content)
    addMessage("ai")
    wsRef.current.send(JSON.stringify({ type: "chat", content: lastUser.content }))
  }, [addMessage])

  return { sendMessage, regenerate, isConnected }
}
