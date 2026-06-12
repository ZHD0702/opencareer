import { Download, Eye, FileText, Loader2 } from "lucide-react"
import { useResumeDocuments } from "../hooks/useResume"
import { useChatStore, type ResumePdfDocument } from "../stores/chatStore"
import { useSessionStore } from "../stores/sessionStore"

function formatTime(iso: string | null): string {
  if (!iso) return ""
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return ""
  return date.toLocaleString("zh-CN", {
    month: "numeric",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  })
}

export function ResumeTab() {
  const sessionId = useSessionStore((state) => state.sessionId)
  const activeResumePdf = useChatStore((state) => state.resumePdf)
  const setResumePdf = useChatStore((state) => state.setResumePdf)
  const { data: documents = [], isLoading, isError, refetch } = useResumeDocuments(sessionId)

  if (!sessionId) {
    return (
      <div className="pt-8 text-center text-xs text-sidebar-muted">
        开始对话后，生成的简历会保存在这里
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-10">
        <Loader2 className="h-5 w-5 animate-spin text-sidebar-muted" />
      </div>
    )
  }

  if (isError) {
    return (
      <div className="py-8 text-center">
        <p className="text-xs text-sidebar-muted">无法加载简历列表</p>
        <button
          type="button"
          onClick={() => refetch()}
          className="mt-2 text-xs text-primary hover:underline"
        >
          重试
        </button>
      </div>
    )
  }

  if (documents.length === 0) {
    return (
      <div className="py-10 text-center">
        <FileText className="mx-auto mb-2 h-8 w-8 text-sidebar-muted/40" />
        <p className="text-xs text-sidebar-muted">暂未生成 PDF 简历</p>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between px-0.5">
        <span className="text-xs font-medium text-sidebar-foreground">已生成简历</span>
        <span className="text-[10px] text-sidebar-muted">{documents.length} 个版本</span>
      </div>

      {documents.map((document, index) => (
        <ResumeDocumentRow
          key={document.id}
          document={document}
          latest={index === 0}
          active={activeResumePdf?.id === document.id}
          onOpen={() => setResumePdf(document, true)}
        />
      ))}
    </div>
  )
}

function ResumeDocumentRow({
  document,
  latest,
  active,
  onOpen,
}: {
  document: ResumePdfDocument
  latest: boolean
  active: boolean
  onOpen: () => void
}) {
  return (
    <div
      className={`flex items-center gap-2 rounded-md border px-2.5 py-2.5 transition-colors ${
        active
          ? "border-primary/40 bg-primary/10"
          : "border-sidebar-border bg-background/50 hover:border-primary/25"
      }`}
    >
      <button type="button" onClick={onOpen} className="min-w-0 flex-1 text-left">
        <div className="flex items-center gap-1.5">
          <FileText className="h-3.5 w-3.5 shrink-0 text-primary" />
          <span className="truncate text-xs font-medium text-sidebar-foreground">
            {document.filename}
          </span>
          {latest && (
            <span className="shrink-0 rounded bg-primary/10 px-1 py-px text-[9px] text-primary">
              最新
            </span>
          )}
        </div>
        <span className="mt-1 block pl-5 text-[10px] text-sidebar-muted">
          {formatTime(document.created_at)}
        </span>
      </button>

      <button
        type="button"
        onClick={onOpen}
        className="grid h-7 w-7 shrink-0 place-items-center rounded text-sidebar-muted transition-colors hover:bg-muted hover:text-sidebar-foreground"
        title="查看简历"
      >
        <Eye className="h-3.5 w-3.5" />
      </button>
      <a
        href={document.download_url}
        className="grid h-7 w-7 shrink-0 place-items-center rounded text-sidebar-muted transition-colors hover:bg-muted hover:text-sidebar-foreground"
        title="下载 PDF"
      >
        <Download className="h-3.5 w-3.5" />
      </a>
    </div>
  )
}
