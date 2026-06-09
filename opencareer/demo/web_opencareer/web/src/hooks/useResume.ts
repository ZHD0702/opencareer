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
}

export interface ResumeResponse {
  session_id: string
  data: ResumeData
  last_updated: string
}

async function fetchResume(sessionId: string): Promise<ResumeResponse> {
  const res = await fetch(`http://localhost:8080/api/resume/${sessionId}`)
  if (!res.ok) {
    throw new Error(`Failed to fetch resume: ${res.status}`)
  }
  return res.json()
}

async function patchResume(
  sessionId: string,
  fields: Partial<ResumeData>,
): Promise<ResumeResponse> {
  const res = await fetch(`http://localhost:8080/api/resume/${sessionId}`, {
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

  const query = useQuery({
    queryKey: ["resume", sessionId, streamFinishCount],
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
