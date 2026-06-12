import { Sidebar } from "./Sidebar"
import { ChatArea } from "./ChatArea"
import { Header } from "./Header"
import { useSessionStore } from "../stores/sessionStore"
import { useChatStore } from "../stores/chatStore"
import { AnimatePresence } from "framer-motion"
import { ResumePreviewPanel } from "./ResumePreviewPanel"
import { CareerWorkspacePanel } from "./CareerWorkspacePanel"

export function Layout() {
  const sidebarCollapsed = useSessionStore((s) => s.sidebarCollapsed)
  const setSidebarCollapsed = useSessionStore((s) => s.setSidebarCollapsed)
  const resumePanelOpen = useChatStore((s) => s.resumePanelOpen)
  const careerWorkspace = useChatStore((s) => s.careerWorkspace)

  return (
    <div className="flex flex-col h-screen bg-background text-foreground">
      <Header
        sidebarCollapsed={sidebarCollapsed}
        onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)}
      />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <ChatArea />
        <AnimatePresence initial={false}>
          {resumePanelOpen && <ResumePreviewPanel />}
          {careerWorkspace && <CareerWorkspacePanel />}
        </AnimatePresence>
      </div>
    </div>
  )
}
