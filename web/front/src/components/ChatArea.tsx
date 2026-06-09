import { useCallback, useEffect, useRef } from "react"
import { useSSEChat } from "../hooks/useSSEChat"
import { useChatStore } from "../stores/chatStore"
import { useSessionStore } from "../stores/sessionStore"
import { MessageList } from "./MessageList"
import { ChatInput } from "./ChatInput"

export function ChatArea() {
  const sessionId = useSessionStore((s) => s.sessionId)
  const setSessionId = useSessionStore((s) => s.setSessionId)
  const hydrateMessages = useChatStore((s) => s.hydrateMessages)
  const { sendMessage } = useSSEChat(sessionId)
  const creatingRef = useRef<Promise<string | null> | null>(null)
  const restoredSessionRef = useRef<string | null>(null)

  const createSession = useCallback(async () => {
    if (creatingRef.current) return creatingRef.current

    creatingRef.current = fetch("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: "default_user" }),
    })
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Failed to create session: ${res.status}`)
        }
        return res.json()
      })
      .then((data) => {
        const nextSessionId = data.session_id as string
        setSessionId(nextSessionId)
        return nextSessionId
      })
      .catch((err) => {
        console.error("Failed to create session:", err)
        return null
      })
      .finally(() => {
        creatingRef.current = null
      })

    return creatingRef.current
  }, [setSessionId])

  const handleSend = useCallback(async (content: string) => {
    const message = content.trim()
    if (!message) return

    const activeSessionId = sessionId || await createSession()
    if (!activeSessionId) return

    sendMessage(message, activeSessionId)
  }, [sessionId, createSession, sendMessage])

  // Listen for send-message events from welcome page suggestions
  useEffect(() => {
    const handler = (e: Event) => {
      const content = (e as CustomEvent<string>).detail
      handleSend(content)
    }
    window.addEventListener("send-message", handler)
    return () => window.removeEventListener("send-message", handler)
  }, [handleSend])

  useEffect(() => {
    if (!sessionId) {
      restoredSessionRef.current = null
      hydrateMessages([])
      return
    }

    if (restoredSessionRef.current === sessionId) return
    restoredSessionRef.current = sessionId

    fetch(`/api/sessions/${sessionId}/messages`)
      .then((res) => {
        if (res.status === 404) {
          setSessionId(null)
          restoredSessionRef.current = null
          return null
        }
        if (!res.ok) {
          throw new Error(`Failed to restore messages: ${res.status}`)
        }
        return res.json()
      })
      .then((data) => {
        if (!data) return
        hydrateMessages(
          data.messages.map((message: { id: string; role: "user" | "ai"; content: string; created_at: string }) => ({
            id: message.id,
            role: message.role,
            content: message.content,
            timestamp: new Date(message.created_at).getTime() || Date.now(),
          }))
        )
      })
      .catch((err) => {
        console.error("Failed to restore session messages:", err)
      })
  }, [sessionId, setSessionId, hydrateMessages])

  return (
    <main className="flex-1 flex flex-col min-w-0">
      <div className="flex-1 overflow-hidden">
        <MessageList onRegenerate={() => {}} />
      </div>
      <ChatInput onSend={handleSend} />
    </main>
  )
}
