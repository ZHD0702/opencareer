import { create } from "zustand"

export interface Message {
  id: string
  role: "user" | "ai"
  content: string
  isStreaming: boolean
  timestamp: number
  emotion?: string
  actions?: AssistantAction[]
}

export interface JobSearchPlan {
  role: string
  city: string
  city_code: string
  salary?: string | null
  skills?: string[]
  employment_type?: string | null
  major?: string | null
  industry?: string | null
}

export interface AssistantAction {
  action: "start_job_matching"
  label: string
  query_plan: JobSearchPlan
}

export interface BrowserTaskState {
  id: string
  session_id: string
  query_plan: JobSearchPlan
  status: string
  status_text: string
  current_url: string
  error?: string | null
  matches?: Array<Record<string, unknown>>
  total_candidates?: number
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

export interface ResumePdfDocument {
  id: number
  session_id: string
  filename: string
  created_at: string | null
  preview_url: string
  download_url: string
}

export type CareerWorkspace =
  | { type: "job"; itemId: string }
  | { type: "skills"; itemId?: string }

interface ChatState {
  messages: Message[]
  isStreaming: boolean
  isTyping: boolean
  currentPhase: string | null
  currentAgent: string | null
  demandAnalysis: Record<string, unknown> | null
  emotionAnalysis: EmotionSnapshot | null
  resumeUpdate: ResumeSnapshot | null
  resumePdf: ResumePdfDocument | null
  resumePanelOpen: boolean
  careerWorkspace: CareerWorkspace | null
  browserTask: BrowserTaskState | null
  browserPanelOpen: boolean
  resumeUpdateCount: number
  streamFinishCount: number
  messageFeedback: Record<string, FeedbackType>

  addMessage: (role: "user" | "ai", content?: string) => string
  beginStreaming: () => void
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
  setResumePdf: (data: ResumePdfDocument | null, openPanel?: boolean) => void
  setResumePanelOpen: (open: boolean) => void
  openCareerWorkspace: (workspace: CareerWorkspace) => void
  closeCareerWorkspace: () => void
  attachAction: (action: AssistantAction) => void
  setBrowserTask: (task: BrowserTaskState | null, openPanel?: boolean) => void
  setBrowserPanelOpen: (open: boolean) => void
  hydrateMessages: (messages: Array<Pick<Message, "id" | "role" | "content" | "timestamp" | "actions">>) => void
  resetConversation: () => void
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
  resumePdf: null,
  resumePanelOpen: false,
  careerWorkspace: null,
  browserTask: null,
  browserPanelOpen: false,
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

  beginStreaming: () => set({ isStreaming: true, isTyping: true }),

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

  setResumePdf: (data, openPanel = true) => {
    set({ resumePdf: data, resumePanelOpen: Boolean(data) && openPanel, careerWorkspace: null, browserPanelOpen: false })
  },

  setResumePanelOpen: (open) => {
    set((state) => ({ resumePanelOpen: Boolean(state.resumePdf) && open, browserPanelOpen: open ? false : state.browserPanelOpen }))
  },

  openCareerWorkspace: (workspace) => {
    set({ careerWorkspace: workspace, resumePanelOpen: false, browserPanelOpen: false })
  },

  closeCareerWorkspace: () => set({ careerWorkspace: null }),

  attachAction: (action) => set((state) => {
    const messages = [...state.messages]
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      if (messages[index].role === "ai") {
        messages[index] = { ...messages[index], actions: [...(messages[index].actions || []), action] }
        break
      }
    }
    return { messages }
  }),

  setBrowserTask: (task, openPanel = true) => set((state) => ({
    browserTask: task,
    browserPanelOpen: openPanel ? Boolean(task) : state.browserPanelOpen,
    resumePanelOpen: openPanel && task ? false : state.resumePanelOpen,
    careerWorkspace: openPanel && task ? null : state.careerWorkspace,
  })),

  setBrowserPanelOpen: (open) => set((state) => ({ browserPanelOpen: Boolean(state.browserTask) && open })),

  hydrateMessages: (messages) => {
    const maxNumericId = messages.reduce((max, message) => {
      const numeric = Number(message.id)
      return Number.isFinite(numeric) ? Math.max(max, numeric) : max
    }, 0)
    nextId = Math.max(nextId, maxNumericId + 1)

    set({
      messages: messages.map((message) => ({
        ...message,
        isStreaming: false,
      })),
      isStreaming: false,
      isTyping: false,
      currentPhase: null,
      currentAgent: null,
    })
  },

  resetConversation: () => {
    set({
      messages: [],
      isStreaming: false,
      isTyping: false,
      currentPhase: null,
      currentAgent: null,
      demandAnalysis: null,
      emotionAnalysis: null,
      resumeUpdate: null,
      resumePdf: null,
      resumePanelOpen: false,
      careerWorkspace: null,
      browserTask: null,
      browserPanelOpen: false,
      messageFeedback: {},
    })
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
