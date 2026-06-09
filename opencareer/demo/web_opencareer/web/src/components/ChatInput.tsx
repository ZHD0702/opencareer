import { useState, useRef, useCallback, type FormEvent, type KeyboardEvent } from "react"
import { motion } from "framer-motion"
import { Send } from "lucide-react"
import { useChatStore } from "../stores/chatStore"
import { cn } from "../lib/utils"

const quickSuggestions = ["帮我看简历", "模拟面试", "薪资谈判", "职业规划"]

interface ChatInputProps {
  onSend: (content: string) => void
}

export function ChatInput({ onSend }: ChatInputProps) {
  const [input, setInput] = useState("")
  const [focused, setFocused] = useState(false)
  const isStreaming = useChatStore((s) => s.isStreaming)
  const currentPhase = useChatStore((s) => s.currentPhase)
  const currentAgent = useChatStore((s) => s.currentAgent)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const autoGrow = useCallback(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = "auto"
    el.style.height = Math.min(el.scrollHeight, 160) + "px"
  }, [])

  const handleChange = (value: string) => {
    setInput(value)
    requestAnimationFrame(autoGrow)
  }

  const handleSubmit = (e?: FormEvent) => {
    e?.preventDefault()
    if (!input.trim() || isStreaming) return
    onSend(input.trim())
    setInput("")
    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleSuggestionClick = (text: string) => {
    setInput((prev) => (prev.trim() ? prev + " " + text : text))
    requestAnimationFrame(() => {
      textareaRef.current?.focus()
      autoGrow()
    })
  }

  const phaseLabel = currentPhase === "analyzing"
    ? currentAgent
      ? `${currentAgent} 分析中...`
      : "分析中..."
    : null

  return (
    <form
      onSubmit={handleSubmit}
      className="border-t border-border p-4 bg-background"
    >
      <div className="max-w-3xl mx-auto">
        {/* Quick suggestions */}
        <div className="flex gap-2 mb-2 flex-wrap">
          {quickSuggestions.map((text) => (
            <motion.button
              key={text}
              type="button"
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              onClick={() => handleSuggestionClick(text)}
              disabled={isStreaming}
              className="text-xs px-2.5 py-1 rounded-full border border-border
                         text-muted-foreground hover:text-foreground hover:border-primary/30
                         bg-background transition-colors disabled:opacity-50"
            >
              {text}
            </motion.button>
          ))}
        </div>

        {/* Input row */}
        <motion.div
          className={cn(
            "flex gap-2 items-end rounded-xl border bg-background px-3 py-2 transition-colors",
            focused
              ? "border-primary/40 shadow-[0_0_0_3px_rgba(234,88,12,0.12)]"
              : "border-border"
          )}
          animate={{
            scale: isStreaming ? 0.99 : 1,
            opacity: isStreaming ? 0.7 : 1,
          }}
        >
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => handleChange(e.target.value)}
            onKeyDown={handleKeyDown}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            placeholder={isStreaming ? "AI 正在回复中..." : "输入你的问题... (Enter 发送, Shift+Enter 换行)"}
            disabled={isStreaming}
            rows={1}
            className="flex-1 resize-none bg-transparent text-sm py-1.5
                       placeholder:text-muted-foreground focus:outline-none
                       disabled:opacity-50"
          />

          <motion.button
            type="submit"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.93 }}
            disabled={!input.trim() || isStreaming}
            className={cn(
              "rounded-lg p-2 shrink-0 transition-colors",
              input.trim() && !isStreaming
                ? "bg-primary text-primary-foreground hover:bg-primary/90"
                : "bg-muted text-muted-foreground cursor-not-allowed"
            )}
          >
            <Send className="w-4 h-4" />
          </motion.button>
        </motion.div>

        {/* Processing status */}
        {phaseLabel && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="flex items-center gap-1.5 mt-1.5 text-xs text-muted-foreground"
          >
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
            {phaseLabel}
          </motion.div>
        )}
      </div>
    </form>
  )
}
