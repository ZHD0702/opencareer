import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { useChatStore } from "../stores/chatStore"

export interface SkillItem {
  name: string
  level: number
  required: number
  category: string
}

export interface GapItem {
  skill: string
  gap: number
  suggestion: string
}

export interface SkillAssessmentData {
  session_id: string
  target_role: string
  match_rate: number
  skills: SkillItem[]
  gaps: GapItem[]
}

async function fetchSkillAssessment(sessionId: string): Promise<SkillAssessmentData> {
  const res = await fetch(`http://localhost:8080/api/skill-assessment/${sessionId}`)
  if (!res.ok) {
    throw new Error(`Failed to fetch skill assessment: ${res.status}`)
  }
  return res.json()
}

async function patchSkillAssessment(
  sessionId: string,
  payload: { target_role?: string; skills?: SkillItem[] },
): Promise<SkillAssessmentData> {
  const res = await fetch(`http://localhost:8080/api/skill-assessment/${sessionId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    throw new Error(`Failed to update skill assessment: ${res.status}`)
  }
  return res.json()
}

export function useSkillAssessment(sessionId: string | null) {
  const queryClient = useQueryClient()
  const streamFinishCount = useChatStore((s) => s.streamFinishCount)

  const query = useQuery({
    queryKey: ["skill-assessment", sessionId, streamFinishCount],
    queryFn: () => fetchSkillAssessment(sessionId!),
    enabled: !!sessionId,
    staleTime: 10_000,
    refetchOnWindowFocus: false,
  })

  const mutation = useMutation({
    mutationFn: (payload: { target_role?: string; skills?: SkillItem[] }) =>
      patchSkillAssessment(sessionId!, payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["skill-assessment", sessionId] })
    },
  })

  return { ...query, mutation }
}
