import { useRef, useEffect, useState, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { User, Bot, Copy, RotateCcw, ThumbsUp, ThumbsDown, Check } from "lucide-react"
import ReactMarkdown from "react-markdown"
import { useChatStore, type Message, type FeedbackType } from "../stores/chatStore"
import { cn } from "../lib/utils"

const suggestions = [
  "怎么优化我的简历？",
  "面试前应该准备什么？",
  "如何谈薪资？",
  "没有实习经历怎么办？",
  "如何规划职业发展路径？",
]

function formatTime(ts: number) {
  const d = new Date(ts)
  return d.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })
}

// ── Breathing dot for streaming ──────────────────────────────────────────
function BreathingDot() {
  return (
    <motion.span
      className="inline-block w-2 h-2 rounded-full bg-primary ml-1 align-middle"
      animate={{ scale: [0.8, 1.3, 0.8], opacity: [0.5, 1, 0.5] }}
      transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
    />
  )
}

// ── Message bubble ───────────────────────────────────────────────────────
function MessageBubble({
  msg,
  onRegenerate,
}: {
  msg: Message
  onRegenerate: () => void
}) {
  const isUser = msg.role === "user"
  const feedback = useChatStore((s) => s.messageFeedback[msg.id])
  const setFeedback = useChatStore((s) => s.setFeedback)
  const isStreaming = useChatStore((s) => s.isStreaming)
  const [copied, setCopied] = useState(false)
  const [showActions, setShowActions] = useState(false)

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(msg.content)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // fallback
    }
  }, [msg.content])

  const handleFeedback = (type: FeedbackType) => {
    setFeedback(msg.id, feedback === type ? null : type)
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.2 }}
      className={cn("flex gap-2.5 group", isUser ? "flex-row-reverse" : "flex-row")}
      onMouseEnter={() => setShowActions(true)}
      onMouseLeave={() => setShowActions(false)}
    >
      {/* Avatar */}
      <div
        className={cn(
          "w-7 h-7 rounded-full flex items-center justify-center shrink-0",
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-muted text-muted-foreground"
        )}
      >
        {isUser ? (
          <User className="w-3.5 h-3.5" />
        ) : (
          <Bot className="w-3.5 h-3.5" />
        )}
      </div>

      <div className={cn("flex flex-col min-w-0", isUser ? "items-end" : "items-start")}>
        {/* Bubble */}
        <div
          className={cn(
            "rounded-2xl px-4 py-2.5 text-sm leading-relaxed",
            isUser
              ? "max-w-[75%] bg-chat-bubble-user text-chat-bubble-user-text rounded-br-md whitespace-pre-wrap"
              : "bg-chat-bubble-ai text-chat-bubble-ai-text rounded-bl-md prose prose-sm prose-stone dark:prose-invert max-w-none max-w-[90%] md:max-w-[80%] lg:max-w-[75%]"
          )}
        >
          {msg.content ? (
            isUser ? (
              msg.content
            ) : (
              <ReactMarkdown>{msg.content}</ReactMarkdown>
            )
          ) : (
            msg.isStreaming && "\u200b"
          )}
          {msg.isStreaming && <BreathingDot />}
        </div>

        {/* Timestamp */}
        <span className="text-[10px] text-muted-foreground mt-1 px-1 opacity-0 group-hover:opacity-100 transition-opacity">
          {formatTime(msg.timestamp)}
        </span>

        {/* Actions (AI messages only, when not streaming) */}
        {!isUser && !msg.isStreaming && msg.content && (
          <motion.div
            initial={false}
            animate={{ opacity: showActions ? 1 : 0, height: showActions ? "auto" : 0 }}
            className="flex items-center gap-1 mt-1 overflow-hidden"
          >
            <motion.button
              type="button"
              whileTap={{ scale: 0.85 }}
              onClick={handleCopy}
              className="p-1 rounded hover:bg-muted transition-colors text-muted-foreground hover:text-foreground"
              title="复制"
            >
              <motion.span
                key={copied ? "check" : "copy"}
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ type: "spring", stiffness: 400, damping: 10 }}
              >
                {copied ? <Check className="w-3 h-3 text-green-500" /> : <Copy className="w-3 h-3" />}
              </motion.span>
            </motion.button>

            <motion.button
              type="button"
              whileTap={{ scale: 0.85 }}
              onClick={onRegenerate}
              disabled={isStreaming}
              className="p-1 rounded hover:bg-muted transition-colors text-muted-foreground hover:text-foreground disabled:opacity-30"
              title="重新生成"
            >
              <RotateCcw className="w-3 h-3" />
            </motion.button>

            <motion.button
              type="button"
              whileTap={{ scale: 0.85 }}
              onClick={() => handleFeedback("liked")}
              className={cn(
                "p-1 rounded transition-colors",
                feedback === "liked"
                  ? "text-primary"
                  : "text-muted-foreground hover:text-foreground"
              )}
              title="赞同"
            >
              <motion.span
                key={feedback === "liked" ? "liked" : "unliked"}
                initial={{ scale: 0.8 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", stiffness: 400, damping: 10 }}
              >
                <ThumbsUp className="w-3 h-3" />
              </motion.span>
            </motion.button>

            <motion.button
              type="button"
              whileTap={{ scale: 0.85 }}
              onClick={() => handleFeedback("disliked")}
              className={cn(
                "p-1 rounded transition-colors",
                feedback === "disliked"
                  ? "text-primary"
                  : "text-muted-foreground hover:text-foreground"
              )}
              title="不赞同"
            >
              <motion.span
                key={feedback === "disliked" ? "disliked" : "undisliked"}
                initial={{ scale: 0.8 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", stiffness: 400, damping: 10 }}
              >
                <ThumbsDown className="w-3 h-3" />
              </motion.span>
            </motion.button>
          </motion.div>
        )}
      </div>
    </motion.div>
  )
}

// ── Welcome page ─────────────────────────────────────────────────────────
function WelcomePage({ onSend }: { onSend: (content: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-full px-4">
      <motion.div
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5 }}
        className="text-center"
      >
        <div className="text-4xl mb-3">🧡</div>
        <h2 className="text-xl font-semibold text-foreground mb-1">
          OpenCareer
        </h2>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.1, duration: 0.4 }}
          className="text-sm text-muted-foreground"
        >
          你的职业发展助手
        </motion.p>
      </motion.div>

      <div className="mt-8 grid gap-2 w-full max-w-sm">
        {suggestions.map((text, i) => (
          <motion.button
            key={text}
            type="button"
            initial={{ y: 12, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.2 + i * 0.08, type: "spring", stiffness: 300, damping: 25 }}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.97 }}
            onClick={() => onSend(text)}
            className="w-full text-left px-4 py-2.5 rounded-xl border border-border bg-background
                       text-sm text-muted-foreground hover:text-foreground hover:border-primary/30
                       hover:bg-muted/50 transition-colors"
          >
            💡 {text}
          </motion.button>
        ))}
      </div>
    </div>
  )
}

// ── MessageList ──────────────────────────────────────────────────────────
interface MessageListProps {
  onRegenerate: () => void
}

export function MessageList({ onRegenerate }: MessageListProps) {
  const messages = useChatStore((s) => s.messages)
  const currentPhase = useChatStore((s) => s.currentPhase)
  const currentAgent = useChatStore((s) => s.currentAgent)
  const isStreaming = useChatStore((s) => s.isStreaming)
  const bottomRef = useRef<HTMLDivElement>(null)
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const userScrolledUpRef = useRef(false)

  // Track whether user has manually scrolled up
  const handleScroll = useCallback(() => {
    const el = scrollContainerRef.current
    if (!el) return
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 100
    userScrolledUpRef.current = !atBottom
  }, [])

  // Auto-scroll when new content arrives (if user is near bottom)
  useEffect(() => {
    if (!userScrolledUpRef.current) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" })
    }
  }, [messages, isStreaming, currentPhase])

  // Scroll to bottom on first load
  useEffect(() => {
    bottomRef.current?.scrollIntoView()
  }, [])

  if (messages.length === 0) {
    return (
      <WelcomePage
        onSend={(content) => {
          // Called via the welcome page suggestions — we use the same mechanism
          // as ChatInput by dispatching a custom event that ChatArea listens to
          window.dispatchEvent(new CustomEvent("send-message", { detail: content }))
        }}
      />
    )
  }

  return (
    <div
      ref={scrollContainerRef}
      onScroll={handleScroll}
      className="px-4 py-6 space-y-4 overflow-y-auto h-full"
    >
      <AnimatePresence>
        {messages.map((msg) => (
          <MessageBubble key={msg.id} msg={msg} onRegenerate={onRegenerate} />
        ))}
      </AnimatePresence>

      {currentPhase === "analyzing" && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex items-center gap-2 text-xs text-muted-foreground px-1"
        >
          <BreathingDot />
          {currentAgent ? `${currentAgent} 分析中...` : "分析中..."}
        </motion.div>
      )}

      <div ref={bottomRef} />
    </div>
  )
}
