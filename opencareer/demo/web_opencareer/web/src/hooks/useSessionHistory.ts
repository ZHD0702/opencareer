import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"

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
  const res = await fetch("http://localhost:8080/api/sessions")
  if (!res.ok) {
    throw new Error(`Failed to fetch sessions: ${res.status}`)
  }
  return res.json()
}

async function deleteSession(sessionId: string): Promise<{ deleted: boolean }> {
  const res = await fetch(`http://localhost:8080/api/sessions/${sessionId}`, {
    method: "DELETE",
  })
  if (!res.ok) {
    throw new Error(`Failed to delete session: ${res.status}`)
  }
  return res.json()
}

export function useSessionHistory() {
  const queryClient = useQueryClient()

  const query = useQuery({
    queryKey: ["sessions"],
    queryFn: fetchSessions,
    staleTime: 10_000,
    refetchOnWindowFocus: false,
  })

  const deleteMutation = useMutation({
    mutationFn: deleteSession,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sessions"] })
    },
  })

  return { ...query, deleteMutation }
}
