import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Clock, MessageSquare, Trash2, Search, ChevronRight, Loader2, WifiOff } from "lucide-react"
import { useSessionStore } from "../stores/sessionStore"
import { useSessionHistory } from "../hooks/useSessionHistory"

function formatDate(iso: string): string {
  const d = new Date(iso)
  const now = new Date()
  const diffMs = now.getTime() - d.getTime()
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24))

  if (diffDays === 0) {
    return d.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })
  } else if (diffDays === 1) {
    return "昨天 " + d.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })
  } else if (diffDays < 7) {
    return `${diffDays}天前`
  }
  return d.toLocaleDateString("zh-CN", { month: "short", day: "numeric" })
}

export function SessionHistoryTab() {
  const sessionId = useSessionStore((s) => s.sessionId)
  const setSessionId = useSessionStore((s) => s.setSessionId)
  const [search, setSearch] = useState("")
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)

  const { data: sessions, isLoading, isError, deleteMutation } = useSessionHistory()

  const activeSessionId = sessionId

  const filtered = (sessions || []).filter(
    (s) =>
      s.title.toLowerCase().includes(search.toLowerCase()) ||
      s.preview.toLowerCase().includes(search.toLowerCase())
  )

  const handleSwitchSession = (id: string) => {
    setSessionId(id)
  }

  const handleDelete = (id: string) => {
    deleteMutation.mutate(id, {
      onSuccess: (result) => {
        if (!result.deleted) {
          console.warn("Session not found on server:", id)
        }
        // If we deleted the active session, clear it
        if (id === activeSessionId) {
          setSessionId(null)
        }
      },
    })
    setConfirmDelete(null)
  }

  // Loading state
  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader2 className="w-5 h-5 text-sidebar-muted animate-spin" />
      </div>
    )
  }

  // Error state
  if (isError) {
    return (
      <div className="text-center py-8">
        <WifiOff className="w-8 h-8 text-sidebar-muted/40 mx-auto mb-2" />
        <p className="text-xs text-sidebar-muted">无法加载会话列表</p>
        <button
          type="button"
          onClick={() => window.location.reload()}
          className="mt-2 text-[11px] text-primary hover:underline"
        >
          重试
        </button>
      </div>
    )
  }

  return (
    <div className="text-sm">
      {/* Search */}
      <div className="relative mb-3">
        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-sidebar-muted" />
        <input
          className="w-full bg-muted/50 border border-sidebar-border rounded-lg pl-8 pr-3 py-1.5 text-xs
                     placeholder:text-sidebar-muted focus:outline-none focus:border-primary/40
                     transition-colors"
          placeholder="搜索历史会话..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      {/* Session list */}
      <AnimatePresence>
        {filtered.length === 0 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-8"
          >
            <Clock className="w-8 h-8 text-sidebar-muted/40 mx-auto mb-2" />
            <p className="text-xs text-sidebar-muted">
              {search ? "无匹配结果" : sessions?.length === 0 ? "暂无历史会话，开始对话吧" : "暂无历史会话"}
            </p>
          </motion.div>
        ) : (
          <div className="space-y-1">
            {filtered.map((session, i) => (
              <motion.div
                key={session.session_id}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.03 }}
                className="group/session"
              >
                {/* Delete confirmation */}
                {confirmDelete === session.session_id ? (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="rounded-lg border border-red-200 dark:border-red-800 bg-red-50/50 dark:bg-red-950/20 px-3 py-2"
                  >
                    <p className="text-[11px] text-red-600 dark:text-red-400 mb-2">
                      确定删除会话「{session.title}」？
                    </p>
                    <div className="flex justify-end gap-1.5">
                      <button
                        type="button"
                        onClick={() => setConfirmDelete(null)}
                        className="px-2 py-0.5 text-[11px] rounded hover:bg-muted transition-colors text-sidebar-muted"
                      >
                        取消
                      </button>
                      <button
                        type="button"
                        onClick={() => handleDelete(session.session_id)}
                        disabled={deleteMutation.isPending}
                        className="px-2 py-0.5 text-[11px] rounded bg-red-500 text-white hover:bg-red-600 transition-colors disabled:opacity-50"
                      >
                        {deleteMutation.isPending ? "删除中..." : "删除"}
                      </button>
                    </div>
                  </motion.div>
                ) : (
                  <button
                    type="button"
                    onClick={() => handleSwitchSession(session.session_id)}
                    className={`w-full text-left rounded-lg border px-3 py-2.5 transition-colors
                      ${session.session_id === activeSessionId
                        ? "border-primary/40 bg-primary/5"
                        : "border-sidebar-border hover:bg-muted/50"
                      }`}
                  >
                    <div className="flex items-center justify-between gap-1">
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-1.5">
                          <span className="text-xs font-medium text-sidebar-foreground truncate">
                            {session.title}
                          </span>
                          {session.session_id === activeSessionId && (
                            <span className="text-[9px] px-1 py-px rounded bg-primary/10 text-primary shrink-0">
                              当前
                            </span>
                          )}
                        </div>
                        <p className="text-[11px] text-sidebar-muted truncate mt-0.5">
                          {session.preview || "暂无消息"}
                        </p>
                      </div>

                      <ChevronRight className="w-3.5 h-3.5 text-sidebar-muted shrink-0" />
                    </div>

                    <div className="flex items-center gap-3 mt-1.5 text-[10px] text-sidebar-muted">
                      <div className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {formatDate(session.created_at)}
                      </div>
                      <div className="flex items-center gap-1">
                        <MessageSquare className="w-3 h-3" />
                        {session.turn_count} 条消息
                      </div>

                      {/* Delete button */}
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation()
                          setConfirmDelete(session.session_id)
                        }}
                        className="ml-auto p-0.5 rounded hover:bg-red-50 dark:hover:bg-red-950/30 transition-colors opacity-0 group-hover/session:opacity-100"
                        title="删除"
                      >
                        <Trash2 className="w-3 h-3 text-red-400" />
                      </button>
                    </div>
                  </button>
                )}
              </motion.div>
            ))}
          </div>
        )}
      </AnimatePresence>

      {filtered.length > 0 && (
        <p className="text-[10px] text-sidebar-muted text-center mt-3">
          {filtered.length} 个会话
        </p>
      )}
    </div>
  )
}
