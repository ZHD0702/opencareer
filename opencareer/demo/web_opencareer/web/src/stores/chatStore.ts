import { create } from "zustand"

export interface Message {
  id: string
  role: "user" | "ai"
  content: string
  isStreaming: boolean
  timestamp: number
}

export type FeedbackType = "liked" | "disliked"

interface ChatState {
  messages: Message[]
  isStreaming: boolean
  currentPhase: string | null
  currentAgent: string | null
  demandAnalysis: Record<string, unknown> | null
  streamFinishCount: number
  messageFeedback: Record<string, FeedbackType>

  addMessage: (role: "user" | "ai", content?: string) => string
  appendToken: (token: string) => void
  setDialogue: (content: string) => void
  breakFragment: () => void
  finishStreaming: () => void
  setStatus: (phase: string | null, agent: string | null) => void
  setDemandAnalysis: (data: Record<string, unknown> | null) => void
  setFeedback: (id: string, type: FeedbackType | null) => void
  removeLastMessage: () => void
}

let nextId = 1

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  isStreaming: false,
  currentPhase: null,
  currentAgent: null,
  demandAnalysis: null,
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

  setStatus: (phase, agent) => {
    set({ currentPhase: phase, currentAgent: agent })
  },

  setDemandAnalysis: (data) => {
    set({ demandAnalysis: data })
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
      return { messages: msgs, isStreaming: false }
    })
  },
}))
