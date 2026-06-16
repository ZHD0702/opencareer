import { useState } from "react"
import { BriefcaseBusiness, CalendarClock, ChevronRight, CirclePlus, Loader2 } from "lucide-react"
import { useJobProgress, type JobCard } from "../hooks/useJobProgress"
import { useChatStore } from "../stores/chatStore"
import { useSessionStore } from "../stores/sessionStore"

const stages = [
  ["saved", "收藏"], ["applied", "已投递"], ["test", "笔试"],
  ["interview", "面试"], ["offered", "Offer"], ["closed", "已结束"],
] as const

export function JobProgressTab() {
  const sessionId = useSessionStore((state) => state.sessionId)
  const openWorkspace = useChatStore((state) => state.openCareerWorkspace)
  const { data, isLoading, createMutation } = useJobProgress(sessionId)
  const [adding, setAdding] = useState(false)
  const [company, setCompany] = useState("")
  const [role, setRole] = useState("")

  if (!sessionId) return <Empty text="开始对话后管理求职岗位" />
  if (isLoading) return <div className="grid place-items-center py-10"><Loader2 className="h-5 w-5 animate-spin text-sidebar-muted" /></div>

  const allCards = stages.flatMap(([stage]) => data?.stages[stage] || [])
  const active = allCards.filter((card) => !["offered", "closed"].includes(card.stage))
  const urgent = active.filter((card) => card.deadline).slice(0, 3)

  const add = () => {
    if (!company.trim() || !role.trim()) return
    createMutation.mutate({ company: company.trim(), role: role.trim(), stage: "saved" })
    setCompany(""); setRole(""); setAdding(false)
  }

  return (
    <div className="space-y-3 text-sm">
      <div className="grid grid-cols-3 gap-2">
        <Stat value={allCards.length} label="全部" />
        <Stat value={active.length} label="进行中" />
        <Stat value={(data?.stages.offered || []).length} label="Offer" />
      </div>

      {urgent.length > 0 && (
        <div className="rounded-md border border-amber-200 bg-amber-50/70 p-2.5 dark:border-amber-900 dark:bg-amber-950/20">
          <div className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-amber-700 dark:text-amber-300">
            <CalendarClock className="h-3.5 w-3.5" />近期行动
          </div>
          {urgent.map((card) => <JobRow key={card.id} card={card} onOpen={() => openWorkspace({ type: "job", itemId: card.id })} compact />)}
        </div>
      )}

      <div className="space-y-2">
        {stages.map(([stage, label]) => {
          const cards = data?.stages[stage] || []
          if (!cards.length) return null
          return <section key={stage}>
            <div className="mb-1 flex items-center justify-between text-[11px] text-sidebar-muted"><span>{label}</span><span>{cards.length}</span></div>
            <div className="space-y-1">{cards.map((card) => <JobRow key={card.id} card={card} onOpen={() => openWorkspace({ type: "job", itemId: card.id })} />)}</div>
          </section>
        })}
      </div>

      {allCards.length === 0 && <Empty text="暂无岗位，添加一个准备跟进的机会" />}

      {adding ? <div className="space-y-1.5 rounded-md border border-sidebar-border p-2">
        <input value={company} onChange={(event) => setCompany(event.target.value)} placeholder="公司" className="w-full rounded border border-sidebar-border bg-background px-2 py-1.5 text-xs" />
        <input value={role} onChange={(event) => setRole(event.target.value)} placeholder="岗位" className="w-full rounded border border-sidebar-border bg-background px-2 py-1.5 text-xs" />
        <div className="flex justify-end gap-2 text-xs"><button onClick={() => setAdding(false)}>取消</button><button onClick={add} className="text-primary">添加</button></div>
      </div> : <button type="button" onClick={() => setAdding(true)} className="flex w-full items-center justify-center gap-1.5 rounded-md border border-dashed border-sidebar-border py-2 text-xs text-sidebar-muted transition-colors hover:border-primary/40 hover:text-primary"><CirclePlus className="h-3.5 w-3.5" />添加岗位</button>}
    </div>
  )
}

function JobRow({ card, onOpen, compact = false }: { card: JobCard; onOpen: () => void; compact?: boolean }) {
  return <button type="button" onClick={onOpen} className="flex w-full items-center gap-2 rounded-md border border-sidebar-border bg-background/50 px-2.5 py-2 text-left hover:border-primary/30">
    <BriefcaseBusiness className="h-3.5 w-3.5 shrink-0 text-primary" />
    <div className="min-w-0 flex-1"><div className="truncate text-xs font-medium text-sidebar-foreground">{card.role}</div>{!compact && <div className="truncate text-[10px] text-sidebar-muted">{card.company}{card.next_action ? ` · ${card.next_action}` : ""}</div>}</div>
    <ChevronRight className="h-3.5 w-3.5 text-sidebar-muted" />
  </button>
}

function Stat({ value, label }: { value: number; label: string }) { return <div className="rounded-md border border-sidebar-border bg-background/50 p-2 text-center"><div className="text-base font-semibold text-sidebar-foreground">{value}</div><div className="text-[10px] text-sidebar-muted">{label}</div></div> }
function Empty({ text }: { text: string }) { return <div className="py-8 text-center text-xs text-sidebar-muted">{text}</div> }
