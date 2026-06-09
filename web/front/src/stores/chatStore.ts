import { create } from "zustand"

export interface Message {
  id: string
  role: "user" | "ai"
  content: string
  isStreaming: boolean
  timestamp: number
  emotion?: string
}

export type FeedbackType = "liked" | "disliked"

export interface EmotionSnapshot {
  session_id?: string
  overall_state?: string
  current_mood?: string
  emotions?: string[]
  confidence?: number
  suggested_action?: string
  support_intensity?: string
  should_intervene?: boolean
  reason?: string
  trend?: string
  matched_keywords?: string[]
}

export interface ResumeSnapshot {
  session_id: string
  state: Record<string, unknown>
  changes: string[]
  stage: string
  completion: number
  next_questions: string[]
  latest_preview: Record<string, unknown> | null
  conflicts: Array<Record<string, string>>
}

interface ChatState {
  messages: Message[]
  isStreaming: boolean
  isTyping: boolean
  currentPhase: string | null
  currentAgent: string | null
  demandAnalysis: Record<string, unknown> | null
  emotionAnalysis: EmotionSnapshot | null
  resumeUpdate: ResumeSnapshot | null
  resumeUpdateCount: number
  streamFinishCount: number
  messageFeedback: Record<string, FeedbackType>

  addMessage: (role: "user" | "ai", content?: string) => string
  appendToken: (token: string) => void
  setDialogue: (content: string) => void
  breakFragment: () => void
  finishStreaming: () => void
  setTyping: (typing: boolean) => void
  addFragment: (content: string, emotion?: string) => void
  setStatus: (phase: string | null, agent: string | null) => void
  setDemandAnalysis: (data: Record<string, unknown> | null) => void
  setEmotionAnalysis: (data: EmotionSnapshot | null) => void
  setResumeUpdate: (data: ResumeSnapshot | null) => void
  setFeedback: (id: string, type: FeedbackType | null) => void
  removeLastMessage: () => void
}

let nextId = 1

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  isStreaming: false,
  isTyping: false,
  currentPhase: null,
  currentAgent: null,
  demandAnalysis: null,
  emotionAnalysis: null,
  resumeUpdate: null,
  resumeUpdateCount: 0,
  streamFinishCount: 0,
  messageFeedback: {},

  addMessage: (role, content = "") => {
    const id = String(nextId++)
    const msg: Message = {
      id,
      role,
      content,
      isStreaming: role === "ai",
      timestamp: Date.now(),
    }
    set((s) => ({
      messages: [...s.messages, msg],
      isStreaming: role === "ai",
    }))
    return id
  },

  appendToken: (token) => {
    set((s) => {
      const msgs = [...s.messages]
      const last = msgs[msgs.length - 1]
      if (last && last.isStreaming) {
        msgs[msgs.length - 1] = { ...last, content: last.content + token }
      }
      return { messages: msgs }
    })
  },

  setDialogue: (content) => {
    set((s) => {
      const msgs = [...s.messages]
      const last = msgs[msgs.length - 1]
      if (last && last.isStreaming) {
        msgs[msgs.length - 1] = { ...last, content }
      }
      return { messages: msgs }
    })
  },

  breakFragment: () => {
    set((s) => {
      const msgs = [...s.messages]
      const last = msgs[msgs.length - 1]
      if (last && last.isStreaming) {
        // Seal the current streaming bubble
        msgs[msgs.length - 1] = { ...last, isStreaming: false }
      }
      // Create a new empty streaming bubble for the next fragment
      const id = String(nextId++)
      const newMsg: Message = {
        id,
        role: "ai",
        content: "",
        isStreaming: true,
        timestamp: Date.now(),
      }
      return { messages: [...msgs, newMsg] }
    })
  },

  finishStreaming: () => {
    set((s) => {
      const msgs = [...s.messages]
      const last = msgs[msgs.length - 1]
      if (last && last.isStreaming) {
        msgs[msgs.length - 1] = { ...last, isStreaming: false }
      }
      return { messages: msgs, isStreaming: false, currentPhase: null, currentAgent: null, streamFinishCount: s.streamFinishCount + 1 }
    })
  },

  setTyping: (typing) => {
    set({ isTyping: typing })
  },

  addFragment: (content, emotion) => {
    set((s) => {
      const msgs = [...s.messages]
      const last = msgs[msgs.length - 1]
      
      // 如果最后一个消息还在流式传输中，先停止它
      if (last && last.isStreaming) {
        msgs[msgs.length - 1] = { ...last, isStreaming: false }
      }
      
      // 创建新的气泡来显示碎片
      const id = String(nextId++)
      const newMsg: Message = {
        id,
        role: "ai",
        content,
        isStreaming: false,
        timestamp: Date.now(),
        emotion: emotion || 'neutral',
      }
      
      return { messages: [...msgs, newMsg] }
    })
  },

  setStatus: (phase, agent) => {
    set({ currentPhase: phase, currentAgent: agent })
  },

  setDemandAnalysis: (data) => {
    set({ demandAnalysis: data })
  },

  setEmotionAnalysis: (data) => {
    set({ emotionAnalysis: data })
  },

  setResumeUpdate: (data) => {
    set((s) => ({ resumeUpdate: data, resumeUpdateCount: s.resumeUpdateCount + 1 }))
  },

  setFeedback: (id, type) => {
    set((s) => ({
      messageFeedback: type
        ? { ...s.messageFeedback, [id]: type }
        : (() => {
            const next = { ...s.messageFeedback }
            delete next[id]
            return next
          })(),
    }))
  },

  removeLastMessage: () => {
    set((s) => {
      const msgs = [...s.messages]
      // Remove ALL consecutive trailing AI messages (handles fragmented replies)
      while (msgs.length > 0 && msgs[msgs.length - 1].role === "ai") {
        msgs.pop()
      }
      // Remove the user message that triggered this AI response
      if (msgs.length > 0 && msgs[msgs.length - 1].role === "user") {
        msgs.pop()
      }
      return { messages: msgs, isStreaming: false, isTyping: false }
    })
  },
}))
