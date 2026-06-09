import { useState } from "react"
import { motion } from "framer-motion"
import { Sparkles, TrendingUp, AlertCircle, Target, Loader2, WifiOff, Edit3, Check, X } from "lucide-react"
import { useSessionStore } from "../stores/sessionStore"
import { useSkillAssessment } from "../hooks/useSkillAssessment"

const categoryConfig: Record<string, string> = {
  "编程语言": "bg-blue-50 dark:bg-blue-950/30 text-blue-700 dark:text-blue-400",
  "框架": "bg-purple-50 dark:bg-purple-950/30 text-purple-700 dark:text-purple-400",
  "数据库": "bg-emerald-50 dark:bg-emerald-950/30 text-emerald-700 dark:text-emerald-400",
  "中间件": "bg-amber-50 dark:bg-amber-950/30 text-amber-700 dark:text-amber-400",
  "DevOps": "bg-cyan-50 dark:bg-cyan-950/30 text-cyan-700 dark:text-cyan-400",
  "基础": "bg-rose-50 dark:bg-rose-950/30 text-rose-700 dark:text-rose-400",
  "工具": "bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400",
}

function getBarColor(level: number, required: number): string {
  const ratio = level / required
  if (ratio >= 1) return "bg-emerald-500"
  if (ratio >= 0.8) return "bg-amber-500"
  if (ratio >= 0.5) return "bg-orange-500"
  return "bg-red-500"
}

export function SkillAssessmentTab() {
  const sessionId = useSessionStore((s) => s.sessionId)
  const { data, isLoading, isError, mutation } = useSkillAssessment(sessionId || null)

  const [editingRole, setEditingRole] = useState(false)
  const [roleDraft, setRoleDraft] = useState("")

  const skills = data?.skills ?? []
  const gaps = data?.gaps ?? []
  const targetRole = data?.target_role ?? ""
  const matchRate = data?.match_rate ?? 0

  // Group skills by category
  const categories = [...new Set(skills.map((s) => s.category).filter(Boolean))]

  const handleEditRole = () => {
    setRoleDraft(targetRole)
    setEditingRole(true)
  }

  const handleSaveRole = () => {
    mutation.mutate({ target_role: roleDraft.trim() })
    setEditingRole(false)
  }

  const handleCancelRole = () => {
    setEditingRole(false)
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
        <p className="text-xs text-sidebar-muted">无法加载技能评估</p>
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
    <div className="text-sm space-y-4">
      {/* Summary header */}
      <div className="bg-muted/50 rounded-lg p-3">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-1.5 min-w-0">
            <Target className="w-4 h-4 text-primary shrink-0" />
            <span className="text-xs font-medium text-sidebar-foreground shrink-0">目标岗位</span>
            {editingRole ? (
              <div className="flex items-center gap-1 min-w-0">
                <input
                  className="flex-1 bg-background border border-sidebar-border rounded px-1.5 py-0.5 text-[11px] min-w-0"
                  value={roleDraft}
                  onChange={(e) => setRoleDraft(e.target.value)}
                  autoFocus
                />
                <button
                  type="button"
                  onClick={handleSaveRole}
                  disabled={mutation.isPending}
                  className="p-0.5 rounded hover:bg-emerald-100 dark:hover:bg-emerald-950/30 text-emerald-600"
                >
                  <Check className="w-3 h-3" />
                </button>
                <button
                  type="button"
                  onClick={handleCancelRole}
                  className="p-0.5 rounded hover:bg-red-50 dark:hover:bg-red-950/30 text-red-400"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ) : (
              <div className="flex items-center gap-1 min-w-0">
                <span className="text-[11px] text-sidebar-muted truncate">
                  {targetRole || "未设置"}
                </span>
                <button
                  type="button"
                  onClick={handleEditRole}
                  className="p-0.5 rounded hover:bg-muted transition-colors shrink-0"
                >
                  <Edit3 className="w-3 h-3 text-sidebar-muted" />
                </button>
              </div>
            )}
          </div>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-[10px] text-sidebar-muted">综合匹配度</span>
          <div className="flex items-center gap-1">
            <div className="w-16 h-2 bg-muted rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${Math.min(matchRate, 100)}%` }}
                transition={{ duration: 0.8, ease: "easeOut" }}
                className="h-full bg-primary rounded-full"
              />
            </div>
            <span className="text-xs font-semibold text-sidebar-foreground">{matchRate}%</span>
          </div>
        </div>
      </div>

      {/* Empty state for no skills */}
      {skills.length === 0 ? (
        <div className="text-center py-8">
          <Sparkles className="w-8 h-8 text-sidebar-muted/40 mx-auto mb-2" />
          <p className="text-xs text-sidebar-muted">暂无技能评估数据</p>
          <p className="text-[10px] text-sidebar-muted/60 mt-1">
            与 AI 助手对话可自动分析技能
          </p>
        </div>
      ) : (
        <>
          {/* Skills by category */}
          <div>
            <h3 className="text-xs font-medium text-sidebar-foreground mb-2 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-sidebar-muted" />
              技能清单
            </h3>

            <div className="space-y-3">
              {categories.map((cat) => {
                const catSkills = skills.filter((s) => s.category === cat)
                return (
                  <div key={cat}>
                    <span className={`text-[10px] px-2 py-0.5 rounded-full ${categoryConfig[cat] || ""}`}>
                      {cat}
                    </span>
                    <div className="mt-1.5 space-y-1.5">
                      {catSkills.map((skill) => (
                        <div key={skill.name} className="flex items-center gap-2 text-xs">
                          <span className="w-20 text-sidebar-foreground truncate shrink-0" title={skill.name}>
                            {skill.name}
                          </span>
                          <div className="flex-1 flex items-center gap-1">
                            <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${skill.level}%` }}
                                transition={{ duration: 0.6, delay: 0.1, ease: "easeOut" }}
                                className={`h-full ${getBarColor(skill.level, skill.required)} rounded-full`}
                              />
                            </div>
                            <span className="text-[10px] text-sidebar-muted w-7 text-right">
                              {skill.level}%
                            </span>
                          </div>
                          {/* Requirement marker */}
                          <div
                            className="relative w-1 h-3 shrink-0"
                            title={`岗位要求: ${skill.required}%`}
                          >
                            <div
                              className="absolute top-0 left-0 w-full bg-sidebar-foreground/30 rounded-full"
                              style={{ height: `${skill.required}%` }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Gap analysis */}
          {gaps.length > 0 && (
            <div>
              <h3 className="text-xs font-medium text-sidebar-foreground mb-2 flex items-center gap-1.5">
                <AlertCircle className="w-3.5 h-3.5 text-amber-500" />
                技能差距
              </h3>

              <div className="space-y-2">
                {gaps.map((gap, i) => (
                  <motion.div
                    key={gap.skill}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.1 + i * 0.08 }}
                    className="rounded-lg border border-sidebar-border p-2.5"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-medium text-sidebar-foreground">{gap.skill}</span>
                      <span className="text-[10px] text-red-500 font-medium">差距 {gap.gap}%</span>
                    </div>
                    <p className="text-[11px] text-sidebar-muted leading-relaxed">
                      {gap.suggestion || "建议重点提升该领域能力"}
                    </p>
                  </motion.div>
                ))}
              </div>
            </div>
          )}

          {/* Recommendations */}
          <div className="rounded-lg bg-primary/5 border border-primary/20 p-3">
            <h3 className="text-xs font-medium text-primary mb-1.5 flex items-center gap-1.5">
              <TrendingUp className="w-3.5 h-3.5" />
              提升建议
            </h3>
            <ul className="space-y-1">
              {gaps.slice(0, 3).map((gap, i) => (
                <li key={gap.skill} className="text-[11px] text-sidebar-foreground flex items-start gap-1">
                  <span className="text-primary mt-0.5">{i + 1}.</span>
                  优先补强{gap.skill}能力
                  {gap.gap >= 20 ? "，这是最大短板" : ""}
                </li>
              ))}
              {gaps.length === 0 && (
                <li className="text-[11px] text-sidebar-muted">所有技能均达到或超过岗位要求，继续保持！</li>
              )}
            </ul>
          </div>
        </>
      )}
    </div>
  )
}
