import { useEffect, useMemo, useState } from "react"
import { motion } from "framer-motion"
import { BadgeCheck, BriefcaseBusiness, CircleAlert, MessageCircleQuestion, Save, Send, Trash2, X } from "lucide-react"
import { useJobProgress, type JobCard } from "../hooks/useJobProgress"
import { useSkillAssessment } from "../hooks/useSkillAssessment"
import { useChatStore } from "../stores/chatStore"
import { useSessionStore } from "../stores/sessionStore"

const stageOptions = [
  ["saved", "收藏"], ["applied", "已投递"], ["test", "笔试"],
  ["interview", "面试"], ["offered", "Offer"], ["closed", "已结束"],
] as const

export function CareerWorkspacePanel() {
  const workspace = useChatStore((state) => state.careerWorkspace)
  const close = useChatStore((state) => state.closeCareerWorkspace)
  if (!workspace) return null
  return <motion.aside initial={{ width: 0, opacity: 0, x: 24 }} animate={{ width: "min(44vw, 680px)", opacity: 1, x: 0 }} exit={{ width: 0, opacity: 0, x: 24 }} transition={{ type: "spring", stiffness: 320, damping: 34 }} className="max-w-[680px] shrink-0 overflow-hidden border-l border-border bg-background">
    <div className="flex h-full min-w-[390px] flex-col">
      <header className="flex h-14 shrink-0 items-center justify-between border-b border-border px-4">
        <div className="flex items-center gap-2 text-sm font-medium"><BriefcaseBusiness className="h-4 w-4 text-primary" />{workspace.type === "job" ? "岗位详情" : "技能证据"}</div>
        <button type="button" onClick={close} className="grid h-8 w-8 place-items-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground" title="关闭"><X className="h-4 w-4" /></button>
      </header>
      <div className="min-h-0 flex-1 overflow-y-auto p-5">{workspace.type === "job" ? <JobWorkspace itemId={workspace.itemId} /> : <SkillsWorkspace selectedId={workspace.itemId} />}</div>
    </div>
  </motion.aside>
}

function JobWorkspace({ itemId }: { itemId: string }) {
  const sessionId = useSessionStore((state) => state.sessionId)
  const close = useChatStore((state) => state.closeCareerWorkspace)
  const { data, updateMutation, deleteMutation } = useJobProgress(sessionId)
  const card = useMemo(() => Object.values(data?.stages || {}).flat().find((item) => item.id === itemId), [data, itemId])
  const [draft, setDraft] = useState<Partial<JobCard>>({})
  useEffect(() => { if (card) setDraft(card) }, [card])
  if (!card) return <p className="text-sm text-muted-foreground">岗位记录不存在或正在加载。</p>
  const field = (key: keyof JobCard, value: string) => setDraft((current) => ({ ...current, [key]: value }))
  return <div className="space-y-5">
    <div className="grid grid-cols-2 gap-3"><Input label="公司" value={draft.company || ""} onChange={(value) => field("company", value)} /><Input label="岗位" value={draft.role || ""} onChange={(value) => field("role", value)} /></div>
    <label className="block"><span className="mb-1 block text-xs text-muted-foreground">当前阶段</span><select value={draft.stage || "saved"} onChange={(event) => field("stage", event.target.value)} className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm">{stageOptions.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
    <div className="grid grid-cols-2 gap-3"><Input label="下一步行动" value={draft.next_action || ""} onChange={(value) => field("next_action", value)} /><Input label="截止日期" type="date" value={draft.deadline || ""} onChange={(value) => field("deadline", value)} /></div>
    <Textarea label="岗位 JD" value={draft.jd_text || ""} onChange={(value) => field("jd_text", value)} rows={9} placeholder="粘贴岗位职责和要求，技能评估会据此生成岗位缺口。" />
    <Textarea label="备注与复盘" value={draft.note || ""} onChange={(value) => field("note", value)} rows={5} />
    <div className="flex items-center justify-between border-t border-border pt-4"><button type="button" onClick={() => deleteMutation.mutate(card.id, { onSuccess: close })} className="flex items-center gap-1.5 text-xs text-red-500"><Trash2 className="h-3.5 w-3.5" />删除岗位</button><button type="button" onClick={() => updateMutation.mutate({ cardId: card.id, payload: draft })} className="flex items-center gap-1.5 rounded-md bg-primary px-3 py-2 text-xs font-medium text-primary-foreground"><Save className="h-3.5 w-3.5" />{updateMutation.isPending ? "保存中" : "保存"}</button></div>
  </div>
}

function SkillsWorkspace({ selectedId }: { selectedId?: string }) {
  const sessionId = useSessionStore((state) => state.sessionId)
  const { data } = useSkillAssessment(sessionId)
  const skills = data?.skills || []
  return <div className="space-y-6">
    <div><h2 className="text-lg font-semibold">{data?.target_role || "技能证据画像"}</h2><p className="mt-1 text-sm text-muted-foreground">只有能够由经历、项目或面试表现支撑的技能才标记为“已证明”。</p></div>
    {["proven", "mentioned", "gap"].map((status) => {
      const items = skills.filter((item) => item.status === status)
      const config = status === "proven"
        ? { label: "已证明", Icon: BadgeCheck, color: "text-emerald-600" }
        : status === "mentioned"
          ? { label: "提到但缺证据", Icon: MessageCircleQuestion, color: "text-amber-600" }
          : { label: "岗位缺口", Icon: CircleAlert, color: "text-red-500" }
      const Icon = config.Icon
      return <section key={status}><div className={`mb-2 flex items-center gap-2 text-sm font-medium ${config.color}`}><Icon className="h-4 w-4" />{config.label} <span className="text-xs text-muted-foreground">{items.length}</span></div><div className="space-y-2">{items.map((item) => <SkillEvidenceCard key={item.id} item={item} selected={String(item.id) === selectedId} />)}</div></section>
    })}
  </div>
}

function SkillEvidenceCard({ item, selected }: { item: import("../hooks/useSkillAssessment").SkillEvidence; selected: boolean }) {
  const chains = item.evidence_chain || []
  const prompt = item.suggestion || `我想继续补充 ${item.skill_name} 的技能证据`
  const continueInChat = () => window.dispatchEvent(new CustomEvent("prefill-chat", { detail: `${prompt}\n` }))
  return <article className={`rounded-md border p-3 ${selected ? "border-primary/50 bg-primary/5" : "border-border"}`}>
    <div className="flex items-center justify-between gap-3"><div><strong className="text-sm">{item.skill_name}</strong><span className="ml-2 text-[11px] text-muted-foreground">{item.category}</span></div><span className="text-[11px] text-muted-foreground">{chains.length} 条证据</span></div>
    {item.requirement && <p className="mt-2 text-sm"><span className="text-xs text-muted-foreground">岗位要求：</span>{item.requirement}</p>}
    {chains.length > 0 ? <div className="mt-3 space-y-3">{chains.map((evidence, index) => <div key={evidence.id} className="border-l-2 border-primary/25 pl-3"><div className="flex items-center justify-between text-xs"><span className="font-medium">证据 {index + 1}</span><span className="text-muted-foreground">{levelLabels[evidence.level]} · {evidence.completeness}%</span></div><EvidenceLine label="场景" value={evidence.scenario} /><EvidenceLine label="行动" value={evidence.action} /><EvidenceLine label="结果" value={evidence.result} /><EvidenceLine label="量化" value={evidence.metric} /><EvidenceLine label="角色" value={evidence.user_role} />{evidence.missing_fields.length > 0 && <p className="mt-2 text-xs text-amber-600 dark:text-amber-400">待补充：{evidence.missing_fields.map((field) => fieldLabels[field] || field).join("、")}</p>}</div>)}</div> : item.evidence && <p className="mt-2 text-sm leading-relaxed"><span className="text-xs text-muted-foreground">现有线索：</span>{item.evidence}</p>}
    {(item.suggestion || item.status !== "proven") && <button type="button" onClick={continueInChat} className="mt-3 flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1.5 text-xs text-muted-foreground hover:border-primary/30 hover:text-foreground"><Send className="h-3.5 w-3.5" />在对话中补充</button>}
  </article>
}

const levelLabels = { mentioned: "仅提到", used: "应用过", proven: "已证明", strong: "强证据" }
const fieldLabels: Record<string, string> = { scenario: "使用场景", action: "具体行动", result: "实际结果", metric: "量化数据", user_role: "个人角色" }
function EvidenceLine({ label, value }: { label: string; value?: string | null }) { return value ? <p className="mt-1 text-sm leading-relaxed"><span className="mr-1 text-xs text-muted-foreground">{label}</span>{value}</p> : null }

function Input({ label, value, onChange, type = "text" }: { label: string; value: string; onChange: (value: string) => void; type?: string }) { return <label className="block"><span className="mb-1 block text-xs text-muted-foreground">{label}</span><input type={type} value={value} onChange={(event) => onChange(event.target.value)} className="w-full rounded-md border border-border bg-background px-3 py-2 text-sm" /></label> }
function Textarea({ label, value, onChange, rows, placeholder }: { label: string; value: string; onChange: (value: string) => void; rows: number; placeholder?: string }) { return <label className="block"><span className="mb-1 block text-xs text-muted-foreground">{label}</span><textarea value={value} onChange={(event) => onChange(event.target.value)} rows={rows} placeholder={placeholder} className="w-full resize-none rounded-md border border-border bg-background px-3 py-2 text-sm leading-relaxed" /></label> }
