import { BadgeCheck, ChevronRight, CircleAlert, Loader2, MessageCircleQuestion, Sparkles } from "lucide-react"
import { useSkillAssessment, type SkillEvidence, type SkillStatus } from "../hooks/useSkillAssessment"
import { useChatStore } from "../stores/chatStore"
import { useSessionStore } from "../stores/sessionStore"

const groups: Array<{ status: SkillStatus; label: string; icon: typeof BadgeCheck; color: string }> = [
  { status: "proven", label: "已证明", icon: BadgeCheck, color: "text-emerald-600 dark:text-emerald-400" },
  { status: "mentioned", label: "提到但缺证据", icon: MessageCircleQuestion, color: "text-amber-600 dark:text-amber-400" },
  { status: "gap", label: "岗位缺口", icon: CircleAlert, color: "text-red-500 dark:text-red-400" },
]

export function SkillAssessmentTab() {
  const sessionId = useSessionStore((state) => state.sessionId)
  const openWorkspace = useChatStore((state) => state.openCareerWorkspace)
  const { data, isLoading } = useSkillAssessment(sessionId)
  if (!sessionId) return <div className="py-8 text-center text-xs text-sidebar-muted">开始对话后分析技能证据</div>
  if (isLoading) return <div className="grid place-items-center py-10"><Loader2 className="h-5 w-5 animate-spin text-sidebar-muted" /></div>
  const skills = data?.skills || []
  return <div className="space-y-3 text-sm">
    <button type="button" onClick={() => openWorkspace({ type: "skills" })} className="flex w-full items-center justify-between rounded-md border border-sidebar-border bg-background/50 p-3 text-left hover:border-primary/30">
      <div><div className="flex items-center gap-1.5 text-xs font-medium text-sidebar-foreground"><Sparkles className="h-3.5 w-3.5 text-primary" />{data?.target_role || "通用技能画像"}</div><div className="mt-1 text-[10px] text-sidebar-muted">基于简历、对话与岗位要求</div></div><ChevronRight className="h-4 w-4 text-sidebar-muted" />
    </button>
    {groups.map((group) => {
      const items = skills.filter((skill) => skill.status === group.status)
      return <section key={group.status}>
        <div className={`mb-1.5 flex items-center gap-1.5 text-xs font-medium ${group.color}`}><group.icon className="h-3.5 w-3.5" />{group.label}<span className="text-[10px] text-sidebar-muted">{items.length}</span></div>
        {items.length ? <div className="space-y-1">{items.slice(0, 4).map((skill) => <SkillRow key={skill.id} skill={skill} onOpen={() => openWorkspace({ type: "skills", itemId: String(skill.id) })} />)}</div> : <div className="rounded-md border border-dashed border-sidebar-border px-2 py-2 text-center text-[10px] text-sidebar-muted">暂无</div>}
      </section>
    })}
  </div>
}

function SkillRow({ skill, onOpen }: { skill: SkillEvidence; onOpen: () => void }) { return <button type="button" onClick={onOpen} className="flex w-full items-center justify-between rounded-md border border-sidebar-border bg-background/40 px-2.5 py-2 text-left hover:border-primary/25"><div className="min-w-0"><div className="truncate text-xs text-sidebar-foreground">{skill.skill_name}</div><div className="truncate text-[10px] text-sidebar-muted">{skill.evidence || skill.suggestion || skill.category}</div></div><ChevronRight className="h-3.5 w-3.5 shrink-0 text-sidebar-muted" /></button> }
