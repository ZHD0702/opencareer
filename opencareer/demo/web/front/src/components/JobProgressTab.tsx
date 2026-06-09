import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Briefcase, Calendar, Building2, Plus, Trash2, ChevronRight, Loader2, WifiOff } from "lucide-react"
import { useSessionStore } from "../stores/sessionStore"
import { useJobProgress, type JobCard } from "../hooks/useJobProgress"

type Stage = "applied" | "interviewing" | "offered" | "rejected"

const stageOrder: Stage[] = ["applied", "interviewing", "offered", "rejected"]

const stageConfig: Record<Stage, { label: string; color: string; bg: string }> = {
  applied: { label: "已投递", color: "text-blue-600 dark:text-blue-400", bg: "bg-blue-50 dark:bg-blue-950/30" },
  interviewing: { label: "面试中", color: "text-amber-600 dark:text-amber-400", bg: "bg-amber-50 dark:bg-amber-950/30" },
  offered: { label: "已Offer", color: "text-emerald-600 dark:text-emerald-400", bg: "bg-emerald-50 dark:bg-emerald-950/30" },
  rejected: { label: "已拒绝", color: "text-red-400/60", bg: "bg-red-50/50 dark:bg-red-950/20" },
}

export function JobProgressTab() {
  const sessionId = useSessionStore((s) => s.sessionId)
  const { data, isLoading, isError, createMutation, moveMutation, deleteMutation } = useJobProgress(sessionId || null)

  const [showAdd, setShowAdd] = useState<Stage | null>(null)
  const [newCompany, setNewCompany] = useState("")
  const [newRole, setNewRole] = useState("")

  const stages = data?.stages ?? {}
  const cards = (stage: Stage): JobCard[] => stages[stage] ?? []

  const totalApplied = Object.values(stages).flat().length
  const activeCount = (stages.applied?.length ?? 0) + (stages.interviewing?.length ?? 0)
  const offerCount = stages.offered?.length ?? 0

  const moveCard = (card: JobCard, to: Stage) => {
    moveMutation.mutate({ cardId: card.id, toStage: to })
  }

  const deleteCard = (card: JobCard) => {
    deleteMutation.mutate(card.id)
  }

  const addCard = (stage: Stage) => {
    if (!newCompany.trim() || !newRole.trim()) return
    createMutation.mutate({
      stage,
      company: newCompany.trim(),
      role: newRole.trim(),
      date: new Date().toISOString().slice(0, 10),
    })
    setNewCompany("")
    setNewRole("")
    setShowAdd(null)
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
        <p className="text-xs text-sidebar-muted">无法加载求职进度</p>
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
      {/* Stats bar */}
      <div className="flex gap-2 mb-4">
        <div className="flex-1 rounded-lg bg-muted/60 p-2 text-center">
          <p className="text-lg font-semibold text-sidebar-foreground">{totalApplied}</p>
          <p className="text-[10px] text-sidebar-muted">总计</p>
        </div>
        <div className="flex-1 rounded-lg bg-blue-50 dark:bg-blue-950/20 p-2 text-center">
          <p className="text-lg font-semibold text-blue-600 dark:text-blue-400">{activeCount}</p>
          <p className="text-[10px] text-blue-500">进行中</p>
        </div>
        <div className="flex-1 rounded-lg bg-emerald-50 dark:bg-emerald-950/20 p-2 text-center">
          <p className="text-lg font-semibold text-emerald-600 dark:text-emerald-400">{offerCount}</p>
          <p className="text-[10px] text-emerald-500">Offer</p>
        </div>
      </div>

      {/* Pipeline columns */}
      <div className="space-y-3">
        {stageOrder.map((stage) => {
          const cfg = stageConfig[stage]
          const stageCards = cards(stage)
          return (
            <div key={stage} className="rounded-lg border border-sidebar-border overflow-hidden">
              {/* Stage header */}
              <div className={cfg.bg}>
                <div className="flex items-center justify-between px-3 py-1.5">
                  <div className="flex items-center gap-1.5">
                    <div className={`w-2 h-2 rounded-full ${cfg.color.replace("text-", "bg-")}`} />
                    <span className={`text-xs font-medium ${cfg.color}`}>{cfg.label}</span>
                    <span className="text-[10px] text-sidebar-muted">({stageCards.length})</span>
                  </div>
                  <button
                    type="button"
                    onClick={() => setShowAdd(showAdd === stage ? null : stage)}
                    className="p-0.5 rounded hover:bg-black/10 dark:hover:bg-white/10 transition-colors"
                  >
                    <Plus className="w-3.5 h-3.5 text-sidebar-muted" />
                  </button>
                </div>
              </div>

              {/* Cards */}
              <AnimatePresence>
                {stageCards.length === 0 ? (
                  <p className="text-[11px] text-sidebar-muted text-center py-3">暂无记录</p>
                ) : (
                  <div className="px-2 py-1.5 space-y-1">
                    {stageCards.map((card) => (
                      <motion.div
                        key={card.id}
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        className="group/card bg-background rounded-md border border-sidebar-border px-2.5 py-2"
                      >
                        <div className="flex items-start justify-between gap-1">
                          <div className="min-w-0">
                            <p className="text-xs font-medium text-sidebar-foreground truncate">
                              {card.role}
                            </p>
                            <div className="flex items-center gap-1 mt-0.5">
                              <Building2 className="w-3 h-3 text-sidebar-muted shrink-0" />
                              <span className="text-[11px] text-sidebar-muted truncate">{card.company}</span>
                            </div>
                            <div className="flex items-center gap-1 mt-0.5">
                              <Calendar className="w-3 h-3 text-sidebar-muted shrink-0" />
                              <span className="text-[10px] text-sidebar-muted">{card.date}</span>
                            </div>
                            {card.note && (
                              <p className="text-[10px] text-sidebar-muted mt-1 italic">{card.note}</p>
                            )}
                          </div>

                          {/* Actions (hover) */}
                          <div className="flex flex-col gap-0.5 opacity-0 group-hover/card:opacity-100 transition-opacity">
                            {stage !== "offered" && (
                              <button
                                type="button"
                                onClick={() => {
                                  const currentIdx = stageOrder.indexOf(stage)
                                  const nextStage = stageOrder[currentIdx + 1]
                                  if (nextStage) moveCard(card, nextStage)
                                }}
                                className="p-0.5 rounded hover:bg-muted transition-colors"
                                title="推进到下一阶段"
                              >
                                <ChevronRight className="w-3 h-3 text-sidebar-muted" />
                              </button>
                            )}
                            <button
                              type="button"
                              onClick={() => deleteCard(card)}
                              className="p-0.5 rounded hover:bg-red-50 dark:hover:bg-red-950/30 transition-colors"
                              title="删除"
                            >
                              <Trash2 className="w-3 h-3 text-red-400" />
                            </button>
                          </div>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                )}
              </AnimatePresence>

              {/* Add form */}
              <AnimatePresence>
                {showAdd === stage && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="px-3 pb-2 space-y-1.5"
                  >
                    <input
                      className="w-full bg-background border border-sidebar-border rounded px-2 py-1 text-xs"
                      placeholder="公司名称"
                      value={newCompany}
                      onChange={(e) => setNewCompany(e.target.value)}
                    />
                    <input
                      className="w-full bg-background border border-sidebar-border rounded px-2 py-1 text-xs"
                      placeholder="岗位名称"
                      value={newRole}
                      onChange={(e) => setNewRole(e.target.value)}
                    />
                    <div className="flex justify-end gap-1.5">
                      <button
                        type="button"
                        onClick={() => setShowAdd(null)}
                        className="px-2 py-0.5 text-[11px] rounded hover:bg-muted transition-colors text-sidebar-muted"
                      >
                        取消
                      </button>
                      <button
                        type="button"
                        onClick={() => addCard(stage)}
                        disabled={!newCompany.trim() || !newRole.trim() || createMutation.isPending}
                        className="px-2 py-0.5 text-[11px] rounded bg-sidebar-foreground text-sidebar disabled:opacity-40"
                      >
                        {createMutation.isPending ? "添加中..." : "添加"}
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )
        })}
      </div>
    </div>
  )
}
