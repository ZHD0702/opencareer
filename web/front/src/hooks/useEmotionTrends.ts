import { useQuery } from "@tanstack/react-query"
import { useChatStore } from "../stores/chatStore"

export interface EmotionTrendItem {
  emotions: string[]
  overall_state: string
  current_mood: string
  support_intensity: string
  demand_type: string
  confidence: number
  created_at?: string
  timestamp: string
}

export interface EmotionTrends {
  session_id: string
  current_mood: string | null
  current_overall_state: string
  current_emotions: string[]
  confidence: number
  support_intensity: string
  suggested_action: string
  trend: "improving" | "declining" | "stable" | "insufficient_data"
  consecutive_negative: number
  negative_ratio: number
  needs_intervention: boolean
  reason: string
  history: EmotionTrendItem[]
}

async function fetchEmotionTrends(sessionId: string): Promise<EmotionTrends> {
  const res = await fetch(`/api/emotion/trends/${sessionId}`)
  if (!res.ok) {
    throw new Error(`Failed to fetch emotion trends: ${res.status}`)
  }
  return res.json()
}

export function useEmotionTrends(sessionId: string | null) {
  const streamFinishCount = useChatStore((s) => s.streamFinishCount)

  return useQuery({
    queryKey: ["emotion-trends", sessionId, streamFinishCount],
    queryFn: () => fetchEmotionTrends(sessionId!),
    enabled: !!sessionId,
    staleTime: 10_000,
    refetchOnWindowFocus: false,
  })
}
