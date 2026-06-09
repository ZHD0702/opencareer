import { Sidebar } from "./Sidebar"
import { ChatArea } from "./ChatArea"
import { Header } from "./Header"
import { useSessionStore } from "../stores/sessionStore"

export function Layout() {
  const sidebarCollapsed = useSessionStore((s) => s.sidebarCollapsed)
  const setSidebarCollapsed = useSessionStore((s) => s.setSidebarCollapsed)

  return (
    <div className="flex flex-col h-screen bg-background text-foreground">
      <Header
        sidebarCollapsed={sidebarCollapsed}
        onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)}
      />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <ChatArea />
      </div>
    </div>
  )
}
