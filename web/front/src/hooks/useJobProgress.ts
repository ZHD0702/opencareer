import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { useChatStore } from "../stores/chatStore"

export interface JobCard {
  id: string
  company: string
  role: string
  date: string
  note?: string | null
  stage: string
  next_action?: string | null
  deadline?: string | null
  jd_text?: string | null
  source?: string | null
}

export interface JobProgressData {
  session_id: string
  stages: Record<string, JobCard[]>
}

async function fetchJobProgress(sessionId: string): Promise<JobProgressData> {
  const res = await fetch(`/api/job-progress/${sessionId}`)
  if (!res.ok) {
    throw new Error(`Failed to fetch job progress: ${res.status}`)
  }
  return res.json()
}

async function createJobCard(
  sessionId: string,
  payload: Partial<JobCard> & { stage: string; company: string; role: string },
): Promise<JobCard> {
  const res = await fetch(`/api/job-progress/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    throw new Error(`Failed to create job card: ${res.status}`)
  }
  return res.json()
}

async function moveJobCard(
  sessionId: string,
  cardId: string,
  toStage: string,
): Promise<JobCard> {
  const res = await fetch(`/api/job-progress/${sessionId}/${cardId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ to_stage: toStage }),
  })
  if (!res.ok) {
    throw new Error(`Failed to move job card: ${res.status}`)
  }
  return res.json()
}

async function updateJobCard(sessionId: string, cardId: string, payload: Partial<JobCard>): Promise<JobCard> {
  const res = await fetch(`/api/job-progress/${sessionId}/${cardId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error(`Failed to update job card: ${res.status}`)
  return res.json()
}

async function deleteJobCard(
  sessionId: string,
  cardId: string,
): Promise<{ deleted: boolean }> {
  const res = await fetch(`/api/job-progress/${sessionId}/${cardId}`, {
    method: "DELETE",
  })
  if (!res.ok) {
    throw new Error(`Failed to delete job card: ${res.status}`)
  }
  return res.json()
}

export function useJobProgress(sessionId: string | null) {
  const queryClient = useQueryClient()
  const streamFinishCount = useChatStore((s) => s.streamFinishCount)

  const query = useQuery({
    queryKey: ["job-progress", sessionId, streamFinishCount],
    queryFn: () => fetchJobProgress(sessionId!),
    enabled: !!sessionId,
    staleTime: 10_000,
    refetchOnWindowFocus: false,
  })

  const createMutation = useMutation({
    mutationFn: (payload: Partial<JobCard> & { stage: string; company: string; role: string }) =>
      createJobCard(sessionId!, payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["job-progress", sessionId] })
    },
  })

  const moveMutation = useMutation({
    mutationFn: ({ cardId, toStage }: { cardId: string; toStage: string }) =>
      moveJobCard(sessionId!, cardId, toStage),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["job-progress", sessionId] })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (cardId: string) => deleteJobCard(sessionId!, cardId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["job-progress", sessionId] })
    },
  })

  const updateMutation = useMutation({
    mutationFn: ({ cardId, payload }: { cardId: string; payload: Partial<JobCard> }) =>
      updateJobCard(sessionId!, cardId, payload),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["job-progress", sessionId] }),
  })

  return { ...query, createMutation, moveMutation, updateMutation, deleteMutation }
}
