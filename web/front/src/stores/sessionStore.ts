import { create } from "zustand"

export type SidebarTab =
  | "emotion"
  | "resume"
  | "job-progress"
  | "skill-assessment"
  | "session-history"

function getStoredCollapsed(): boolean {
  try {
    const v = localStorage.getItem("oc-sidebar-collapsed")
    return v === "true"
  } catch {
    return false
  }
}

function getStoredSessionId(): string | null {
  try {
    return localStorage.getItem("oc-current-session-id")
  } catch {
    return null
  }
}

interface SessionState {
  sessionId: string | null
  sidebarTab: SidebarTab
  sidebarCollapsed: boolean
  setSessionId: (id: string | null) => void
  setSidebarTab: (tab: SidebarTab) => void
  setSidebarCollapsed: (collapsed: boolean) => void
}

export const useSessionStore = create<SessionState>((set) => ({
  sessionId: getStoredSessionId(),
  sidebarTab: "session-history",
  sidebarCollapsed: getStoredCollapsed(),
  setSessionId: (id) => {
    if (id) {
      localStorage.setItem("oc-current-session-id", id)
      set({ sessionId: id })
    } else {
      localStorage.removeItem("oc-current-session-id")
      set({ sessionId: null })
    }
  },
  setSidebarTab: (tab) => set({ sidebarTab: tab }),
  setSidebarCollapsed: (collapsed) => {
    localStorage.setItem("oc-sidebar-collapsed", String(collapsed))
    set({ sidebarCollapsed: collapsed })
  },
}))
