import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { useChatStore } from "../stores/chatStore"

export interface ResumeData {
  grade_level: string | null
  major: string | null
  school: string | null
  target_role: string | null
  job_search_stage: string | null
  skill_focus: string[]
  common_concerns: string[]
  background_summary: string | null
  resume_state?: ResumeState | null
}

export interface ResumeState {
  basics: Record<string, string | null>
  target: Record<string, string | null>
  education: Array<Record<string, unknown>>
  experiences: ResumeExperience[]
  projects: Array<Record<string, unknown>>
  skills: {
    hard: string[]
    soft: string[]
  }
  preferences: Record<string, string | null>
  previews: ResumePreview[]
  industry_insights?: {
    keywords: string[]
    preferred_metrics: string[]
    notes: string[]
  }
  llm_insights?: {
    last_applied: boolean
    confidence: number
    reason: string | null
  }
  conflicts: ResumeConflict[]
  unresolved_questions: string[]
  stage: string
  completion: number
  last_update_summary: string | null
}

export interface ResumeExperience {
  id: string
  type: string
  raw: string
  star: Record<string, string | null>
  metrics: Array<{ value: string; unit: string }>
  bullets: string[]
  created_at: string
}

export interface ResumePreview {
  source: string
  content: string
  type: string
  created_at: string
  needs_confirmation: boolean
}

export interface ResumeConflict {
  type: string
  message: string
}

export interface ResumeResponse {
  session_id: string
  data: ResumeData
  last_updated: string
}

async function fetchResume(sessionId: string): Promise<ResumeResponse> {
  const res = await fetch(`/api/resume/${sessionId}`)
  if (!res.ok) {
    throw new Error(`Failed to fetch resume: ${res.status}`)
  }
  return res.json()
}

async function patchResume(
  sessionId: string,
  fields: Partial<ResumeData>,
): Promise<ResumeResponse> {
  const res = await fetch(`/api/resume/${sessionId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(fields),
  })
  if (!res.ok) {
    throw new Error(`Failed to update resume: ${res.status}`)
  }
  return res.json()
}

export function useResume(sessionId: string | null) {
  const queryClient = useQueryClient()
  const streamFinishCount = useChatStore((s) => s.streamFinishCount)
  const resumeUpdateCount = useChatStore((s) => s.resumeUpdateCount)

  const query = useQuery({
    queryKey: ["resume", sessionId, streamFinishCount, resumeUpdateCount],
    queryFn: () => fetchResume(sessionId!),
    enabled: !!sessionId,
    staleTime: 10_000,
    refetchOnWindowFocus: false,
  })

  const mutation = useMutation({
    mutationFn: (fields: Partial<ResumeData>) => patchResume(sessionId!, fields),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["resume", sessionId] })
    },
  })

  return { ...query, mutation }
}
