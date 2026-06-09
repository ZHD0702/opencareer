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

interface SessionState {
  sessionId: string | null
  sidebarTab: SidebarTab
  sidebarCollapsed: boolean
  setSessionId: (id: string) => void
  setSidebarTab: (tab: SidebarTab) => void
  setSidebarCollapsed: (collapsed: boolean) => void
}

export const useSessionStore = create<SessionState>((set) => ({
  sessionId: null,
  sidebarTab: "emotion",
  sidebarCollapsed: getStoredCollapsed(),
  setSessionId: (id) => set({ sessionId: id }),
  setSidebarTab: (tab) => set({ sidebarTab: tab }),
  setSidebarCollapsed: (collapsed) => {
    localStorage.setItem("oc-sidebar-collapsed", String(collapsed))
    set({ sidebarCollapsed: collapsed })
  },
}))
