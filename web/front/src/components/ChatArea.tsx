import { useEffect, useRef } from "react"
import { useSSEChat } from "../hooks/useSSEChat"
import { useSessionStore } from "../stores/sessionStore"
import { MessageList } from "./MessageList"
import { ChatInput } from "./ChatInput"

export function ChatArea() {
  const sessionId = useSessionStore((s) => s.sessionId)
  const setSessionId = useSessionStore((s) => s.setSessionId)
  const { sendMessage } = useSSEChat(sessionId)
  const creatingRef = useRef(false)

  // Listen for send-message events from welcome page suggestions
  useEffect(() => {
    const handler = (e: Event) => {
      const content = (e as CustomEvent<string>).detail
      sendMessage(content)
    }
    window.addEventListener("send-message", handler)
    return () => window.removeEventListener("send-message", handler)
  }, [sendMessage])

  useEffect(() => {
    if (!sessionId && !creatingRef.current) {
      creatingRef.current = true
      fetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: "default_user" }),
      })
        .then((res) => res.json())
        .then((data) => setSessionId(data.session_id))
        .catch((err) => {
          console.error("Failed to create session:", err)
          creatingRef.current = false
        })
    }
  }, [sessionId, setSessionId])

  return (
    <main className="flex-1 flex flex-col min-w-0">
      <div className="flex-1 overflow-hidden">
        <MessageList onRegenerate={() => {}} />
      </div>
      <ChatInput onSend={sendMessage} />
    </main>
  )
}
