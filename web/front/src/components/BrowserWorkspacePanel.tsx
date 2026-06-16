import { useCallback, useEffect, useRef, useState, type MouseEvent } from "react"
import { motion } from "framer-motion"
import { ExternalLink, Keyboard, Loader2, MousePointer2, Play, RefreshCw, Square, X } from "lucide-react"
import { useChatStore, type BrowserTaskState } from "../stores/chatStore"

export function BrowserWorkspacePanel() {
  const task = useChatStore((state) => state.browserTask)
  const setTask = useChatStore((state) => state.setBrowserTask)
  const setOpen = useChatStore((state) => state.setBrowserPanelOpen)
  const [frameUrl, setFrameUrl] = useState<string | null>(null)
  const [input, setInput] = useState("")
  const [loadingFrame, setLoadingFrame] = useState(false)
  const [resuming, setResuming] = useState(false)
  const frameRef = useRef<HTMLImageElement>(null)

  const refreshStatus = useCallback(async () => {
    if (!task) return
    const response = await fetch(`/api/browser-sessions/${task.id}`, { cache: "no-store" })
    if (response.ok) setTask({ ...(await response.json()) as BrowserTaskState }, false)
  }, [task, setTask])

  const refreshFrame = useCallback(async () => {
    if (!task) return
    setLoadingFrame(true)
    try {
      const response = await fetch(`/api/browser-sessions/${task.id}/screenshot?t=${Date.now()}`, { cache: "no-store" })
      if (!response.ok) return
      const next = URL.createObjectURL(await response.blob())
      setFrameUrl((current) => {
        if (current) URL.revokeObjectURL(current)
        return next
      })
    } finally {
      setLoadingFrame(false)
    }
  }, [task])

  useEffect(() => {
    const timer = window.setInterval(() => {
      refreshStatus()
      refreshFrame()
    }, 1200)
    refreshStatus()
    refreshFrame()
    return () => window.clearInterval(timer)
  }, [refreshFrame, refreshStatus])

  useEffect(() => () => { if (frameUrl) URL.revokeObjectURL(frameUrl) }, [frameUrl])
  if (!task) return null

  const interact = async (payload: Record<string, unknown>) => {
    await fetch(`/api/browser-sessions/${task.id}/interact`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
    window.setTimeout(refreshFrame, 250)
  }

  const clickFrame = (event: MouseEvent<HTMLImageElement>) => {
    const image = frameRef.current
    if (!image) return
    const rect = image.getBoundingClientRect()
    interact({ type: "click", x: (event.clientX - rect.left) * (1280 / rect.width), y: (event.clientY - rect.top) * (900 / rect.height) })
  }

  const stop = async () => {
    await fetch(`/api/browser-sessions/${task.id}`, { method: "DELETE" })
    await refreshStatus()
  }

  const resume = async () => {
    setResuming(true)
    try {
      const response = await fetch(`/api/browser-sessions/${task.id}/resume`, { method: "POST" })
      if (response.ok) setTask({ ...(await response.json()) as BrowserTaskState }, false)
      await refreshFrame()
    } finally {
      setResuming(false)
    }
  }

  return <motion.aside initial={{ width: 0, opacity: 0, x: 28 }} animate={{ width: "min(48vw, 760px)", opacity: 1, x: 0 }} exit={{ width: 0, opacity: 0, x: 28 }} transition={{ type: "spring", stiffness: 320, damping: 34 }} className="max-w-[760px] shrink-0 overflow-hidden border-l border-border bg-background">
    <div className="flex h-full min-w-[430px] flex-col">
      <header className="flex h-14 shrink-0 items-center justify-between border-b border-border px-4">
        <div className="min-w-0"><div className="flex items-center gap-2 text-sm font-medium"><MousePointer2 className="h-4 w-4 text-primary" />智联实时匹配</div><div className="truncate text-[11px] text-muted-foreground">{task.query_plan.role} · {task.query_plan.city}</div></div>
        <div className="flex items-center gap-1">
          {task.current_url && <a href={task.current_url} target="_blank" rel="noreferrer" className="grid h-8 w-8 place-items-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground" title="在浏览器打开"><ExternalLink className="h-4 w-4" /></a>}
          <button type="button" onClick={refreshFrame} className="grid h-8 w-8 place-items-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground" title="刷新画面"><RefreshCw className={`h-4 w-4 ${loadingFrame ? "animate-spin" : ""}`} /></button>
          <button type="button" onClick={stop} className="grid h-8 w-8 place-items-center rounded-md text-muted-foreground hover:bg-muted hover:text-red-500" title="停止"><Square className="h-3.5 w-3.5" /></button>
          <button type="button" onClick={() => setOpen(false)} className="grid h-8 w-8 place-items-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground" title="关闭"><X className="h-4 w-4" /></button>
        </div>
      </header>

      <div className="flex min-h-10 shrink-0 items-center justify-between gap-3 border-b border-border px-4 py-2 text-xs">
        <div className="flex min-w-0 items-center gap-2"><span className={`h-2 w-2 shrink-0 rounded-full ${task.status === "ready" ? "bg-emerald-500" : task.status === "error" ? "bg-red-500" : "bg-amber-500"}`} /><span className="min-w-0 truncate">{task.status === "error" && task.error ? task.error : task.status_text}</span></div>
        {task.status === "needs_user" && <button type="button" onClick={resume} disabled={resuming} className="flex h-8 shrink-0 items-center gap-1.5 rounded-md border border-primary/35 px-2.5 font-medium text-primary hover:bg-primary/5 disabled:opacity-60" title="保留当前登录状态并继续搜索岗位">{resuming ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}登录后继续匹配</button>}
      </div>

      <div className="relative min-h-0 flex-1 overflow-auto bg-[#202020] p-2" onWheel={(event) => { event.preventDefault(); interact({ type: "scroll", delta_y: event.deltaY }) }}>
        {frameUrl ? <img ref={frameRef} src={frameUrl} onClick={clickFrame} className="block w-full cursor-crosshair select-none bg-white" draggable={false} alt="智联招聘实时浏览器画面" /> : <div className="grid h-full place-items-center text-sm text-neutral-300">{task.status === "error" ? task.error : <span className="flex items-center gap-2"><Loader2 className="h-4 w-4 animate-spin" />正在等待浏览器画面</span>}</div>}
      </div>

      <footer className="flex shrink-0 items-center gap-2 border-t border-border p-3">
        <Keyboard className="h-4 w-4 shrink-0 text-muted-foreground" />
        <input value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && input) { interact({ type: "type", text: input }); setInput("") } }} placeholder="需要人工接管时，在此输入并按 Enter" className="h-9 min-w-0 flex-1 rounded-md border border-border bg-background px-3 text-sm outline-none focus:border-primary/40" />
        <button type="button" onClick={() => interact({ type: "key", key: "Enter" })} className="h-9 rounded-md border border-border px-3 text-xs text-muted-foreground hover:text-foreground">回车</button>
      </footer>
    </div>
  </motion.aside>
}
