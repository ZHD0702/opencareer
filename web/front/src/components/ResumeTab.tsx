import { useState } from "react"
import { useResume, type ResumeData } from "../hooks/useResume"
import { useSessionStore } from "../stores/sessionStore"
import { AlertCircle, Check, ChevronDown, ChevronRight, FileText, Pencil, Sparkles, X } from "lucide-react"

type SectionKey = "basic" | "job" | "skills" | "concerns" | "summary"

const jobStageLabels: Record<string, string> = {
  starting: "准备开始",
  applying: "投递中",
  interviewing: "面试中",
  waiting: "等待结果",
  negotiating: "薪资谈判",
  accepted: "已接受offer",
}

const resumeStageLabels: Record<string, string> = {
  basic_info: "基础信息",
  experience_discovery: "经历发现",
  experience_deepening: "STAR 深挖",
  soft_skill_assessment: "软技能评估",
  resume_preview: "片段预览",
}

function formatStage(stage: string | null): string {
  if (!stage) return ""
  return jobStageLabels[stage] || stage
}

function formatTime(iso: string): string {
  if (!iso) return ""
  try {
    const d = new Date(iso)
    return d.toLocaleString("zh-CN", {
      month: "numeric",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    })
  } catch {
    return ""
  }
}

function hasSectionData(data: ResumeData, section: SectionKey): boolean {
  switch (section) {
    case "basic":
      return !!(data.grade_level || data.major || data.school)
    case "job":
      return !!(data.target_role || data.job_search_stage)
    case "skills":
      return data.skill_focus.length > 0
    case "concerns":
      return data.common_concerns.length > 0
    case "summary":
      return !!data.background_summary
  }
}

export function ResumeTab() {
  const sessionId = useSessionStore((s) => s.sessionId)
  const { data, isLoading, isError, mutation } = useResume(sessionId)

  const [expanded, setExpanded] = useState<Record<SectionKey, boolean>>({
    basic: true,
    job: false,
    skills: false,
    concerns: false,
    summary: false,
  })
  const [editing, setEditing] = useState<SectionKey | null>(null)
  const [editValues, setEditValues] = useState<Partial<ResumeData>>({})

  if (!sessionId) {
    return (
      <div className="text-sm text-sidebar-muted">
        <p className="text-center mt-8">等待会话创建...</p>
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="text-sm text-sidebar-muted animate-pulse">
        <div className="h-4 bg-muted rounded w-3/4 mx-auto mt-8" />
        <div className="h-16 bg-muted rounded mt-4" />
        <div className="h-12 bg-muted rounded mt-3" />
        <div className="h-4 bg-muted rounded w-1/2 mx-auto mt-3" />
      </div>
    )
  }

  if (isError || !data) {
    return (
      <div className="text-sm text-sidebar-muted">
        <p className="text-center mt-8 text-red-500">无法加载简历数据</p>
        <p className="text-center text-xs mt-2">请检查后端服务是否运行</p>
      </div>
    )
  }

  const resumeData = data.data
  const resumeState = resumeData.resume_state

  const toggleSection = (section: SectionKey) => {
    setExpanded((prev) => ({ ...prev, [section]: !prev[section] }))
    if (editing === section) {
      setEditing(null)
    }
  }

  const startEditing = (section: SectionKey) => {
    setEditing(section)
    setEditValues({
      grade_level: resumeData.grade_level,
      major: resumeData.major,
      school: resumeData.school,
      target_role: resumeData.target_role,
      job_search_stage: resumeData.job_search_stage,
      skill_focus: [...resumeData.skill_focus],
      common_concerns: [...resumeData.common_concerns],
      background_summary: resumeData.background_summary,
    })
  }

  const cancelEditing = () => {
    setEditing(null)
    setEditValues({})
  }

  const saveEditing = () => {
    if (editing) {
      // Build payload — only include fields relevant to this section
      const payload: Partial<ResumeData> = {}
      switch (editing) {
        case "basic":
          payload.grade_level = editValues.grade_level
          payload.major = editValues.major
          payload.school = editValues.school
          break
        case "job":
          payload.target_role = editValues.target_role
          payload.job_search_stage = editValues.job_search_stage
          break
        case "skills":
          payload.skill_focus = editValues.skill_focus
          break
        case "concerns":
          payload.common_concerns = editValues.common_concerns
          break
        case "summary":
          payload.background_summary = editValues.background_summary
          break
      }
      mutation.mutate(payload)
    }
    setEditing(null)
    setEditValues({})
  }

  const hasAnyData = (
    hasSectionData(resumeData, "basic") ||
    hasSectionData(resumeData, "job") ||
    hasSectionData(resumeData, "skills") ||
    hasSectionData(resumeData, "concerns") ||
    hasSectionData(resumeData, "summary")
  )

  if (!hasAnyData) {
    return (
      <div className="text-sm">
        <ResumeBuilderPanel state={resumeState} />
        <p className="text-center mt-5 text-sidebar-muted">暂无简历数据</p>
        <p className="text-center text-xs mt-2 text-sidebar-muted">对话中提及的经历会被自动提取成结构化片段</p>
      </div>
    )
  }

  return (
    <div className="text-sm">
      <ResumeBuilderPanel state={resumeState} />

      {/* Last updated */}
      {data.last_updated && (
        <p className="text-[11px] text-sidebar-muted mb-3">
          更新于 {formatTime(data.last_updated)}
        </p>
      )}

      {/* Basic info section */}
      <SectionCard
        title="基本信息"
        section="basic"
        expanded={expanded}
        editing={editing}
        hasData={hasSectionData(resumeData, "basic")}
        onToggle={toggleSection}
        onEdit={startEditing}
        onCancel={cancelEditing}
        onSave={saveEditing}
        isSaving={mutation.isPending}
      >
        {editing === "basic" ? (
          <div className="space-y-2">
            <FieldRow label="年级">
              <input
                className="w-full bg-background border border-sidebar-border rounded px-2 py-1 text-xs"
                value={editValues.grade_level || ""}
                placeholder="如：大三、研一"
                onChange={(e) => setEditValues({ ...editValues, grade_level: e.target.value || null })}
              />
            </FieldRow>
            <FieldRow label="专业">
              <input
                className="w-full bg-background border border-sidebar-border rounded px-2 py-1 text-xs"
                value={editValues.major || ""}
                placeholder="如：软件工程"
                onChange={(e) => setEditValues({ ...editValues, major: e.target.value || null })}
              />
            </FieldRow>
            <FieldRow label="学校">
              <input
                className="w-full bg-background border border-sidebar-border rounded px-2 py-1 text-xs"
                value={editValues.school || ""}
                placeholder="如：北京大学"
                onChange={(e) => setEditValues({ ...editValues, school: e.target.value || null })}
              />
            </FieldRow>
          </div>
        ) : (
          <div className="space-y-1.5">
            <FieldValue label="年级" value={resumeData.grade_level} />
            <FieldValue label="专业" value={resumeData.major} />
            <FieldValue label="学校" value={resumeData.school} />
          </div>
        )}
      </SectionCard>

      {/* Job search section */}
      <SectionCard
        title="求职意向"
        section="job"
        expanded={expanded}
        editing={editing}
        hasData={hasSectionData(resumeData, "job")}
        onToggle={toggleSection}
        onEdit={startEditing}
        onCancel={cancelEditing}
        onSave={saveEditing}
        isSaving={mutation.isPending}
      >
        {editing === "job" ? (
          <div className="space-y-2">
            <FieldRow label="目标岗位">
              <input
                className="w-full bg-background border border-sidebar-border rounded px-2 py-1 text-xs"
                value={editValues.target_role || ""}
                placeholder="如：后端开发工程师"
                onChange={(e) => setEditValues({ ...editValues, target_role: e.target.value || null })}
              />
            </FieldRow>
            <FieldRow label="求职阶段">
              <select
                className="w-full bg-background border border-sidebar-border rounded px-2 py-1 text-xs"
                value={editValues.job_search_stage || ""}
                onChange={(e) => setEditValues({ ...editValues, job_search_stage: e.target.value || null })}
              >
                <option value="">未知</option>
                <option value="starting">准备开始</option>
                <option value="applying">投递中</option>
                <option value="interviewing">面试中</option>
                <option value="waiting">等待结果</option>
                <option value="negotiating">薪资谈判</option>
                <option value="accepted">已接受offer</option>
              </select>
            </FieldRow>
          </div>
        ) : (
          <div className="space-y-1.5">
            <FieldValue label="目标岗位" value={resumeData.target_role} />
            <FieldValue label="求职阶段" value={formatStage(resumeData.job_search_stage)} />
          </div>
        )}
      </SectionCard>

      {/* Skills section */}
      <SectionCard
        title="技能方向"
        section="skills"
        expanded={expanded}
        editing={editing}
        hasData={hasSectionData(resumeData, "skills")}
        onToggle={toggleSection}
        onEdit={startEditing}
        onCancel={cancelEditing}
        onSave={saveEditing}
        isSaving={mutation.isPending}
      >
        {editing === "skills" ? (
          <div className="space-y-2">
            <textarea
              className="w-full bg-background border border-sidebar-border rounded px-2 py-1 text-xs resize-none"
              rows={3}
              placeholder="每行一个技能"
              value={(editValues.skill_focus || []).join("\n")}
              onChange={(e) =>
                setEditValues({
                  ...editValues,
                  skill_focus: e.target.value
                    .split("\n")
                    .map((s) => s.trim())
                    .filter(Boolean),
                })
              }
            />
          </div>
        ) : (
          <div className="flex flex-wrap gap-1">
            {resumeData.skill_focus.map((skill, i) => (
              <span
                key={i}
                className="px-2 py-0.5 rounded-full bg-blue-50 dark:bg-blue-950/30 text-blue-700 dark:text-blue-400 text-xs"
              >
                {skill}
              </span>
            ))}
          </div>
        )}
      </SectionCard>

      {/* Concerns section */}
      <SectionCard
        title="常见顾虑"
        section="concerns"
        expanded={expanded}
        editing={editing}
        hasData={hasSectionData(resumeData, "concerns")}
        onToggle={toggleSection}
        onEdit={startEditing}
        onCancel={cancelEditing}
        onSave={saveEditing}
        isSaving={mutation.isPending}
      >
        {editing === "concerns" ? (
          <div className="space-y-2">
            <textarea
              className="w-full bg-background border border-sidebar-border rounded px-2 py-1 text-xs resize-none"
              rows={3}
              placeholder="每行一个顾虑"
              value={(editValues.common_concerns || []).join("\n")}
              onChange={(e) =>
                setEditValues({
                  ...editValues,
                  common_concerns: e.target.value
                    .split("\n")
                    .map((s) => s.trim())
                    .filter(Boolean),
                })
              }
            />
          </div>
        ) : (
          <div className="flex flex-wrap gap-1">
            {resumeData.common_concerns.map((concern, i) => (
              <span
                key={i}
                className="px-2 py-0.5 rounded-full bg-amber-50 dark:bg-amber-950/30 text-amber-700 dark:text-amber-400 text-xs"
              >
                {concern}
              </span>
            ))}
          </div>
        )}
      </SectionCard>

      {/* Background summary section */}
      <SectionCard
        title="背景摘要"
        section="summary"
        expanded={expanded}
        editing={editing}
        hasData={hasSectionData(resumeData, "summary")}
        onToggle={toggleSection}
        onEdit={startEditing}
        onCancel={cancelEditing}
        onSave={saveEditing}
        isSaving={mutation.isPending}
      >
        {editing === "summary" ? (
          <div className="space-y-2">
            <textarea
              className="w-full bg-background border border-sidebar-border rounded px-2 py-1 text-xs resize-none"
              rows={4}
              placeholder="简要描述用户背景..."
              value={editValues.background_summary || ""}
              onChange={(e) =>
                setEditValues({
                  ...editValues,
                  background_summary: e.target.value || null,
                })
              }
            />
          </div>
        ) : (
          <p className="text-xs text-sidebar-foreground leading-relaxed">
            {resumeData.background_summary}
          </p>
        )}
      </SectionCard>

      {/* Mutation error */}
      {mutation.isError && (
        <p className="text-xs text-red-500 mt-2">保存失败，请重试</p>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ */
/* Internal sub-components                                            */
/* ------------------------------------------------------------------ */

function SectionCard({
  title,
  section,
  expanded,
  editing,
  hasData,
  onToggle,
  onEdit,
  onCancel,
  onSave,
  isSaving,
  children,
}: {
  title: string
  section: SectionKey
  expanded: Record<SectionKey, boolean>
  editing: SectionKey | null
  hasData: boolean
  onToggle: (s: SectionKey) => void
  onEdit: (s: SectionKey) => void
  onCancel: () => void
  onSave: () => void
  isSaving: boolean
  children: React.ReactNode
}) {
  const isOpen = expanded[section]
  const isEditing = editing === section

  return (
    <div className="mb-2 rounded-lg border border-sidebar-border overflow-hidden">
      {/* Header */}
      <button
        type="button"
        onClick={() => onToggle(section)}
        className="w-full flex items-center justify-between px-3 py-2 hover:bg-muted/50 transition-colors text-left"
      >
        <span className="text-xs font-medium text-sidebar-foreground flex items-center gap-1.5">
          {isOpen ? (
            <ChevronDown className="w-3.5 h-3.5 text-sidebar-muted" />
          ) : (
            <ChevronRight className="w-3.5 h-3.5 text-sidebar-muted" />
          )}
          {title}
        </span>
        {isOpen && !isEditing && hasData && (
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation()
              onEdit(section)
            }}
            className="p-0.5 rounded hover:bg-muted transition-colors"
            title="编辑"
          >
            <Pencil className="w-3 h-3 text-sidebar-muted" />
          </button>
        )}
      </button>

      {/* Body */}
      {isOpen && (
        <div className="px-3 pb-3">
          {children}

          {/* Edit action bar */}
          {isEditing && (
            <div className="flex justify-end gap-1.5 mt-2">
              <button
                type="button"
                onClick={onCancel}
                disabled={isSaving}
                className="flex items-center gap-1 px-2 py-1 text-xs rounded hover:bg-muted transition-colors"
              >
                <X className="w-3 h-3" />
                取消
              </button>
              <button
                type="button"
                onClick={onSave}
                disabled={isSaving}
                className="flex items-center gap-1 px-2 py-1 text-xs rounded bg-sidebar-foreground text-sidebar hover:opacity-90 transition-opacity disabled:opacity-50"
              >
                <Check className="w-3 h-3" />
                {isSaving ? "保存中..." : "保存"}
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function FieldRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <span className="text-[10px] text-sidebar-muted block mb-0.5">{label}</span>
      {children}
    </div>
  )
}

function FieldValue({ label, value }: { label: string; value: string | null | undefined }) {
  if (!value) return null
  return (
    <div className="flex justify-between items-baseline gap-2">
      <span className="text-[10px] text-sidebar-muted shrink-0">{label}</span>
      <span className="text-xs text-sidebar-foreground truncate text-right">{value}</span>
    </div>
  )
}

function ResumeBuilderPanel({ state }: { state: ResumeData["resume_state"] }) {
  const completion = state?.completion ?? 0
  const stage = state?.stage || "basic_info"
  const latestPreview = state?.previews?.[0]
  const questions = state?.unresolved_questions || []
  const conflicts = state?.conflicts || []
  const hardSkills = state?.skills?.hard || []
  const softSkills = state?.skills?.soft || []
  const industryMetrics = state?.industry_insights?.preferred_metrics || []
  const llmInsights = state?.llm_insights

  return (
    <div className="mb-3 rounded-lg border border-sidebar-border bg-muted/35 p-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5 text-xs font-medium text-sidebar-foreground">
            <FileText className="h-3.5 w-3.5 text-primary" />
            简历采集
          </div>
          <p className="mt-1 text-[11px] text-sidebar-muted">
            {resumeStageLabels[stage] || stage}
          </p>
        </div>
        <div className="text-right">
          <div className="text-sm font-semibold text-sidebar-foreground">{completion}%</div>
          <div className="text-[10px] text-sidebar-muted">完成度</div>
        </div>
      </div>

      <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-background">
        <div
          className="h-full rounded-full bg-primary transition-all"
          style={{ width: `${Math.max(4, completion)}%` }}
        />
      </div>

      {latestPreview && (
        <div className="mt-3 rounded-md border border-sidebar-border bg-sidebar/70 p-2">
          <div className="mb-1 flex items-center gap-1.5 text-[11px] font-medium text-sidebar-muted">
            <Sparkles className="h-3.5 w-3.5" />
            实时片段预览
          </div>
          <p className="text-xs leading-relaxed text-sidebar-foreground">
            {latestPreview.content}
          </p>
          {latestPreview.needs_confirmation && (
            <p className="mt-1 text-[11px] text-amber-600">
              这条还缺量化结果，建议继续追问确认。
            </p>
          )}
        </div>
      )}

      {(hardSkills.length > 0 || softSkills.length > 0) && (
        <div className="mt-3 flex flex-wrap gap-1">
          {[...hardSkills.slice(0, 5), ...softSkills.slice(0, 3)].map((skill) => (
            <span
              key={skill}
              className="rounded-full bg-background px-2 py-0.5 text-[11px] text-sidebar-foreground"
            >
              {skill}
            </span>
          ))}
        </div>
      )}

      {industryMetrics.length > 0 && (
        <div className="mt-3">
          <p className="mb-1 text-[11px] font-medium text-sidebar-muted">行业优先指标</p>
          <div className="flex flex-wrap gap-1">
            {industryMetrics.slice(0, 6).map((metric) => (
              <span
                key={metric}
                className="rounded-full border border-sidebar-border bg-sidebar px-2 py-0.5 text-[11px] text-sidebar-foreground"
              >
                {metric}
              </span>
            ))}
          </div>
        </div>
      )}

      {llmInsights?.last_applied && (
        <p className="mt-2 text-[11px] text-sidebar-muted">
          LLM 已补强 STAR 与行业表达，置信度 {Math.round((llmInsights.confidence || 0) * 100)}%
        </p>
      )}

      {questions.length > 0 && (
        <div className="mt-3">
          <p className="mb-1 text-[11px] font-medium text-sidebar-muted">下一步可以问</p>
          <div className="space-y-1">
            {questions.map((question) => (
              <p key={question} className="rounded-md bg-background px-2 py-1.5 text-xs text-sidebar-foreground">
                {question}
              </p>
            ))}
          </div>
        </div>
      )}

      {conflicts.length > 0 && (
        <div className="mt-3 space-y-1">
          {conflicts.map((conflict) => (
            <div
              key={`${conflict.type}-${conflict.message}`}
              className="flex gap-1.5 rounded-md border border-amber-200 bg-amber-50 px-2 py-1.5 text-xs text-amber-700 dark:border-amber-900 dark:bg-amber-950/30 dark:text-amber-300"
            >
              <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
              <span>{conflict.message}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
