import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useChatStore } from "../stores/chatStore"

export type SkillStatus = "proven" | "mentioned" | "gap"

export type EvidenceLevel = "mentioned" | "used" | "proven" | "strong"

export interface SkillEvidenceItem {
  id: number
  skill_name: string
  context_key?: string | null
  scenario?: string | null
  task?: string | null
  action?: string | null
  result?: string | null
  metric?: string | null
  user_role?: string | null
  used_at?: string | null
  raw_text?: string | null
  source?: string | null
  confidence?: number | null
  level: EvidenceLevel
  completeness: number
  missing_fields: string[]
}

export interface SkillEvidence {
  id: number
  skill_name: string
  status: SkillStatus
  category: string
  evidence?: string | null
  requirement?: string | null
  suggestion?: string | null
  source?: string | null
  evidence_chain?: SkillEvidenceItem[]
}

export interface SkillAssessmentData {
  session_id: string
  target_role: string
  counts: Record<SkillStatus, number>
  skills: SkillEvidence[]
  gaps: SkillEvidence[]
  pending_follow_up?: {
    id: number
    evidence_id?: number | null
    skill_name: string
    missing_field: string
    question: string
  } | null
}

async function fetchAssessment(sessionId: string): Promise<SkillAssessmentData> {
  const res = await fetch(`/api/skill-assessment/${sessionId}`)
  if (!res.ok) throw new Error(`Failed to fetch skill assessment: ${res.status}`)
  return res.json()
}

async function patchAssessment(sessionId: string, payload: { target_role?: string; skills?: Partial<SkillEvidence>[] }) {
  const res = await fetch(`/api/skill-assessment/${sessionId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error(`Failed to update skill assessment: ${res.status}`)
  return res.json()
}

export function useSkillAssessment(sessionId: string | null) {
  const queryClient = useQueryClient()
  const streamFinishCount = useChatStore((state) => state.streamFinishCount)
  const query = useQuery({
    queryKey: ["skill-assessment", sessionId, streamFinishCount],
    queryFn: () => fetchAssessment(sessionId!),
    enabled: !!sessionId,
    staleTime: 5_000,
    refetchOnWindowFocus: true,
  })
  const mutation = useMutation({
    mutationFn: (payload: { target_role?: string; skills?: Partial<SkillEvidence>[] }) => patchAssessment(sessionId!, payload),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["skill-assessment", sessionId] }),
  })
  return { ...query, mutation }
}
