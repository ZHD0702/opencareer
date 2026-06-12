import { useCallback, useEffect, useRef } from "react"
import { motion } from "framer-motion"
import { FileText } from "lucide-react"
import { useSSEChat } from "../hooks/useSSEChat"
import { useChatStore } from "../stores/chatStore"
import { useSessionStore } from "../stores/sessionStore"
import { MessageList } from "./MessageList"
import { ChatInput } from "./ChatInput"

export function ChatArea() {
  const sessionId = useSessionStore((s) => s.sessionId)
  const setSessionId = useSessionStore((s) => s.setSessionId)
  const hydrateMessages = useChatStore((s) => s.hydrateMessages)
  const setResumePdf = useChatStore((s) => s.setResumePdf)
  const resumePdf = useChatStore((s) => s.resumePdf)
  const resumePanelOpen = useChatStore((s) => s.resumePanelOpen)
  const setResumePanelOpen = useChatStore((s) => s.setResumePanelOpen)
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
      setResumePdf(null)
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
  }, [sessionId, setSessionId, hydrateMessages, setResumePdf])

  useEffect(() => {
    if (!sessionId) return

    setResumePdf(null)
    fetch(`/api/resume/${sessionId}/pdf`)
      .then((res) => {
        if (res.status === 404) return null
        if (!res.ok) throw new Error(`Failed to load resume PDF: ${res.status}`)
        return res.json()
      })
      .then((document) => {
        if (document) setResumePdf(document, true)
      })
      .catch((err) => console.error("Failed to restore resume PDF:", err))
  }, [sessionId, setResumePdf])

  return (
    <main className="relative flex min-w-0 flex-1 flex-col">
      {resumePdf && !resumePanelOpen && (
        <motion.button
          type="button"
          initial={{ opacity: 0, x: 8 }}
          animate={{ opacity: 1, x: 0 }}
          onClick={() => setResumePanelOpen(true)}
          className="absolute right-4 top-4 z-20 grid h-9 w-9 place-items-center rounded-md border border-border bg-background text-muted-foreground shadow-sm transition-colors hover:border-primary/40 hover:bg-primary/10 hover:text-primary"
          title="打开简历预览"
        >
          <FileText className="h-4 w-4" />
        </motion.button>
      )}
      <div className="flex-1 overflow-hidden">
        <MessageList onRegenerate={() => {}} />
      </div>
      <ChatInput onSend={handleSend} />
    </main>
  )
}
