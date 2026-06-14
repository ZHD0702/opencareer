import { useRef, useEffect, useState, useCallback } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { AlertCircle, Bot, BriefcaseBusiness, Building2, Check, Copy, ExternalLink, GraduationCap, Loader2, MapPin, RotateCcw, ThumbsDown, ThumbsUp, User } from "lucide-react"
import ReactMarkdown from "react-markdown"
import { useChatStore, type BrowserTaskState, type Message, type FeedbackType } from "../stores/chatStore"
import { useSessionStore } from "../stores/sessionStore"
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

function cleanAssistantContent(content: string) {
  return content.replace(/\s*\[调用工具\s*[:：][^\]]+\]\s*/g, "\n").trim()
}

function readApiError(data: unknown) {
  if (!data || typeof data !== "object") return { message: "岗位搜索暂时失败" }
  const payload = data as Record<string, unknown>
  const detail = payload.detail && typeof payload.detail === "object"
    ? payload.detail as Record<string, unknown>
    : payload.message && typeof payload.message === "object"
      ? payload.message as Record<string, unknown>
      : payload
  return {
    message: typeof detail.message === "string"
      ? detail.message
      : typeof payload.message === "string"
        ? payload.message
        : "岗位搜索暂时失败",
    code: typeof detail.code === "string" ? detail.code : undefined,
    queryPlan: detail.query_plan && typeof detail.query_plan === "object"
      ? detail.query_plan as Record<string, unknown>
      : undefined,
  }
}

// ── Typing dots animation ───────────────────────────────────────────────
function TypingDots() {
  return (
    <span className="flex gap-0.5">
      <motion.span
        className="w-1.5 h-1.5 bg-muted-foreground rounded-full"
        animate={{ opacity: [0.4, 1, 0.4] }}
        transition={{ duration: 1.4, repeat: Infinity, ease: "easeInOut" }}
      />
      <motion.span
        className="w-1.5 h-1.5 bg-muted-foreground rounded-full"
        animate={{ opacity: [0.4, 1, 0.4] }}
        transition={{ duration: 1.4, repeat: Infinity, ease: "easeInOut", delay: 0.2 }}
      />
      <motion.span
        className="w-1.5 h-1.5 bg-muted-foreground rounded-full"
        animate={{ opacity: [0.4, 1, 0.4] }}
        transition={{ duration: 1.4, repeat: Infinity, ease: "easeInOut", delay: 0.4 }}
      />
    </span>
  )
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

interface JobMatch {
  id: string
  title: string
  company: string
  salary: string
  education: string
  experience: string
  city: string
  district: string
  url: string
  match_score: number
  match_level: string
  match_reasons: string[]
  concerns: string[]
  matched_skills: string[]
  published_at: string
  work_type: string
}

function formatPublishedAt(value: string) {
  if (!value) return ""
  const date = new Date(value.replace(" ", "T"))
  if (Number.isNaN(date.getTime())) return value.slice(0, 10)
  return date.toLocaleDateString("zh-CN", { month: "numeric", day: "numeric" })
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
  const setBrowserTask = useChatStore((s) => s.setBrowserTask)
  const isStreaming = useChatStore((s) => s.isStreaming)
  const [copied, setCopied] = useState(false)
  const [showActions, setShowActions] = useState(false)
  const [startingMatch, setStartingMatch] = useState(false)
  const [jobMatches, setJobMatches] = useState<JobMatch[] | null>(null)
  const [matchError, setMatchError] = useState<string | null>(null)
  const [matchNotice, setMatchNotice] = useState<string | null>(null)
  const [candidateCount, setCandidateCount] = useState(0)
  const browserPollTokenRef = useRef(0)
  const displayContent = isUser ? msg.content : cleanAssistantContent(msg.content)

  const bubbleMaxW = "min(36rem, 64vw)"

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(displayContent)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // fallback
    }
  }, [displayContent])

  const handleFeedback = (type: FeedbackType) => {
    setFeedback(msg.id, feedback === type ? null : type)
  }

  useEffect(() => () => {
    browserPollTokenRef.current += 1
  }, [])

  const pollBrowserMatches = async (taskId: string, token: number) => {
    for (let attempt = 0; attempt < 150 && browserPollTokenRef.current === token; attempt += 1) {
      await new Promise((resolve) => window.setTimeout(resolve, 1200))
      const response = await fetch(`/api/browser-sessions/${taskId}`, { cache: "no-store" })
      if (!response.ok) continue
      const task = await response.json() as BrowserTaskState
      setBrowserTask(task, false)
      const matches = Array.isArray(task.matches) ? task.matches as unknown as JobMatch[] : []
      if (matches.length > 0) {
        setJobMatches(matches.slice(0, 5))
        setCandidateCount(task.total_candidates || matches.length)
        setMatchError(null)
        setMatchNotice(null)
        return
      }
      if (task.status === "needs_user") {
        setMatchNotice("智联需要登录或安全验证，请在右侧完成后点击“登录后继续匹配”。")
      } else if (task.status === "error" || task.status === "stopped") {
        setMatchNotice(null)
        setMatchError(task.error || task.status_text || "智联岗位搜索失败")
        return
      } else {
        setMatchNotice(task.status_text || "正在从智联获取实时岗位…")
      }
    }
  }

  const startJobMatching = async (action: NonNullable<Message["actions"]>[number]) => {
    browserPollTokenRef.current += 1
    const pollToken = browserPollTokenRef.current
    setStartingMatch(true)
    setMatchError(null)
    setMatchNotice(null)
    try {
      const sessionId = useSessionStore.getState().sessionId
      if (!sessionId) return
      const response = await fetch("/api/job-search/match", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, limit: 5, ...action.query_plan }),
      })
      const data = await response.json()
      if (!response.ok) {
        const apiError = readApiError(data)
        if (apiError.code === "zhaopin_verification_required") {
          const queryPlan = { ...action.query_plan, ...apiError.queryPlan }
          const browserResponse = await fetch("/api/browser-sessions", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId, ...queryPlan }),
          })
          const browserData = await browserResponse.json()
          if (!browserResponse.ok) throw new Error(readApiError(browserData).message)
          const task = browserData as BrowserTaskState
          setBrowserTask(task, true)
          setMatchNotice("正在通过智联页面获取实时岗位，若出现验证请在右侧完成。")
          void pollBrowserMatches(task.id, pollToken)
          return
        }
        throw new Error(apiError.message)
      }
      setJobMatches(data.matches || [])
      setCandidateCount(data.total_candidates || 0)
    } catch (error) {
      console.error(error)
      setMatchError(error instanceof Error ? error.message : "岗位搜索暂时失败")
    } finally {
      setStartingMatch(false)
    }
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
          "w-8 h-8 rounded-full flex items-center justify-center shrink-0",
          isUser
            ? "bg-primary text-primary-foreground"
            : "bg-muted text-muted-foreground"
        )}
      >
        {isUser ? (
          <User className="w-4 h-4" />
        ) : (
          <Bot className="w-4 h-4" />
        )}
      </div>

      <div className={cn("flex flex-col min-w-0", isUser ? "items-end" : "items-start")}>
        {/* Bubble */}
        <div
          style={{ maxWidth: bubbleMaxW }}
          className={cn(
            "w-fit whitespace-pre-wrap break-words px-5 py-3 text-base leading-relaxed rounded-2xl",
            isUser
              ? "bg-[var(--color-chat-bubble-user)] text-[var(--color-chat-bubble-user-text)] rounded-tr-sm shadow-lg"
              : "bg-[var(--color-chat-bubble-ai)] text-[var(--color-chat-bubble-ai-text)] rounded-tl-sm"
          )}
        >
          {displayContent ? (
            isUser ? (
              displayContent
            ) : (
              <div className="prose prose-base prose-stone dark:prose-invert prose-p:mb-0 prose-p:mt-0 prose:max-w-none">
                <ReactMarkdown>{displayContent}</ReactMarkdown>
              </div>
            )
          ) : (
            msg.isStreaming && "\u200b"
          )}
          {msg.isStreaming && <BreathingDot />}
        </div>

        {!isUser && !msg.isStreaming && msg.actions?.map((action) => (
          <button key={action.action} type="button" onClick={() => startJobMatching(action)} disabled={startingMatch} className="mt-2 flex h-9 items-center gap-2 rounded-md border border-primary/30 bg-primary/5 px-3 text-sm font-medium text-primary transition-colors hover:bg-primary/10 disabled:opacity-60">
            {startingMatch ? <Loader2 className="h-4 w-4 animate-spin" /> : <BriefcaseBusiness className="h-4 w-4" />}
            {jobMatches ? "重新匹配岗位" : action.label}
          </button>
        ))}

        {matchError && <div className="mt-2 flex max-w-xl items-start gap-2 rounded-md border border-red-300/60 bg-red-50 px-3 py-2 text-sm text-red-700 dark:bg-red-950/20 dark:text-red-300"><AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />{matchError}</div>}
        {matchNotice && <div className="mt-2 flex max-w-xl items-start gap-2 rounded-md border border-amber-300/60 bg-amber-50 px-3 py-2 text-sm text-amber-800 dark:bg-amber-950/20 dark:text-amber-200"><Loader2 className="mt-0.5 h-4 w-4 shrink-0 animate-spin" />{matchNotice}</div>}

        {jobMatches && <div className="mt-3 w-[min(36rem,64vw)] space-y-2">
          <div className="flex items-center justify-between px-1 text-xs text-muted-foreground"><span>实时匹配 · 从 {candidateCount} 个真实职位中筛选</span><span>智联招聘</span></div>
          {jobMatches.map((job, index) => <article key={job.id} className="rounded-md border border-border bg-background p-3 shadow-sm">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <a href={job.url} target="_blank" rel="noreferrer" className="inline-flex max-w-full items-center gap-1.5 font-medium text-foreground hover:text-primary"><span className="shrink-0 text-xs text-muted-foreground">{index + 1}.</span><span className="truncate">{job.title}</span><ExternalLink className="h-3.5 w-3.5 shrink-0" /></a>
                <div className="mt-1 flex items-center gap-1.5 text-xs text-muted-foreground"><Building2 className="h-3.5 w-3.5" /><span className="truncate">{job.company}</span></div>
              </div>
              <div className="shrink-0 text-right"><div className="text-base font-semibold text-primary">{job.match_score}</div><div className="text-[10px] text-muted-foreground">{job.match_level}</div></div>
            </div>
            <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
              <span className="font-medium text-primary">{job.salary}</span>
              <span className="inline-flex items-center gap-1"><MapPin className="h-3.5 w-3.5" />{job.city}{job.district ? ` · ${job.district}` : ""}</span>
              {job.education && <span className="inline-flex items-center gap-1"><GraduationCap className="h-3.5 w-3.5" />{job.education}</span>}
              {job.experience && <span>{job.experience}</span>}
              {job.work_type && <span>{job.work_type}</span>}
              {job.published_at && <span>发布于 {formatPublishedAt(job.published_at)}</span>}
            </div>
            {job.matched_skills.length > 0 && <div className="mt-2 flex flex-wrap gap-1">{job.matched_skills.slice(0, 4).map((skill) => <span key={skill} className="rounded border border-primary/20 bg-primary/5 px-1.5 py-0.5 text-[11px] text-primary">{skill}</span>)}</div>}
            {job.match_reasons.length > 0 && <div className="mt-2 text-xs leading-5 text-foreground/80">{job.match_reasons.slice(0, 2).join("；")}</div>}
            {job.concerns.length > 0 && <div className="mt-1 text-xs leading-5 text-amber-700 dark:text-amber-300">注意：{job.concerns[0]}</div>}
          </article>)}
          {jobMatches.length === 0 && <div className="rounded-md border border-dashed border-border px-4 py-5 text-center text-sm text-muted-foreground">没有找到达到当前匹配条件的岗位</div>}
        </div>}

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

// ── Typing indicator bubble (AI) ─────────────────────────────────────────
function TypingBubble() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      className="flex gap-2.5"
    >
      {/* Avatar */}
      <div className="w-8 h-8 rounded-full flex items-center justify-center shrink-0 bg-muted text-muted-foreground">
        <Bot className="w-4 h-4" />
      </div>

      <div className="flex flex-col min-w-0 items-start">
        {/* Bubble */}
        <div className="px-5 py-3 text-base leading-relaxed rounded-2xl bg-[var(--color-chat-bubble-ai)] text-[var(--color-chat-bubble-ai-text)] rounded-tl-sm">
          <div className="flex items-center gap-1.5">
            <span>对方正在输入</span>
            <TypingDots />
          </div>
        </div>
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
  const isTyping = useChatStore((s) => s.isTyping)
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
  }, [messages, isStreaming, isTyping, currentPhase])

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
      className="px-10 md:px-16 lg:px-20 py-6 space-y-4 overflow-y-auto h-full"
    >
      <AnimatePresence>
        {messages.map((msg) => (
          <MessageBubble key={msg.id} msg={msg} onRegenerate={onRegenerate} />
        ))}
        
        {/* Typing bubble (shows when isTyping is true) */}
        {isTyping && <TypingBubble />}
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
