import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { useChatStore } from "../stores/chatStore"

export interface SessionInfo {
  session_id: string
  user_id: string
  title: string
  preview: string
  turn_count: number
  current_agent: string | null
  created_at: string
}

async function fetchSessions(): Promise<SessionInfo[]> {
  const res = await fetch("/api/sessions")
  if (!res.ok) {
    throw new Error(`Failed to fetch sessions: ${res.status}`)
  }
  return res.json()
}

async function deleteSession(sessionId: string): Promise<{ deleted: boolean }> {
  const res = await fetch(`/api/sessions/${sessionId}`, {
    method: "DELETE",
  })
  if (!res.ok) {
    throw new Error(`Failed to delete session: ${res.status}`)
  }
  return res.json()
}

export function useSessionHistory() {
  const queryClient = useQueryClient()
  const streamFinishCount = useChatStore((s) => s.streamFinishCount)
  const resumeUpdateCount = useChatStore((s) => s.resumeUpdateCount)

  const query = useQuery({
    queryKey: ["sessions", streamFinishCount, resumeUpdateCount],
    queryFn: fetchSessions,
    staleTime: 10_000,
    retry: 10,
    retryDelay: (attempt) => Math.min(1000 * (attempt + 1), 3000),
    refetchOnWindowFocus: true,
    refetchOnReconnect: true,
  })

  const deleteMutation = useMutation({
    mutationFn: deleteSession,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sessions"] })
    },
  })

  return { ...query, deleteMutation }
}
