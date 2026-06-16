import { useEffect, useState } from "react"
import { motion } from "framer-motion"
import { Download, FileText, RefreshCw, X } from "lucide-react"
import { useChatStore } from "../stores/chatStore"

export function ResumePreviewPanel() {
  const document = useChatStore((s) => s.resumePdf)
  const setResumePanelOpen = useChatStore((s) => s.setResumePanelOpen)
  const [reloadKey, setReloadKey] = useState(0)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [previewError, setPreviewError] = useState("")

  useEffect(() => {
    if (!document) return

    const controller = new AbortController()
    let objectUrl = ""
    setPreviewUrl(null)
    setPreviewError("")

    fetch(document.preview_url, { cache: "no-store", signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("简历预览加载失败")
        return response.blob()
      })
      .then((blob) => {
        if (controller.signal.aborted) return
        objectUrl = URL.createObjectURL(new Blob([blob], { type: "application/pdf" }))
        setPreviewUrl(objectUrl)
      })
      .catch((error) => {
        if (error instanceof DOMException && error.name === "AbortError") return
        setPreviewError(error instanceof Error ? error.message : "简历预览加载失败")
      })

    return () => {
      controller.abort()
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [document, reloadKey])

  if (!document) return null

  return (
    <motion.aside
      initial={{ width: 0, opacity: 0, x: 24 }}
      animate={{ width: "min(46vw, 720px)", opacity: 1, x: 0 }}
      exit={{ width: 0, opacity: 0, x: 24 }}
      transition={{ type: "spring", stiffness: 320, damping: 34 }}
      className="max-w-[720px] shrink-0 overflow-hidden border-l border-border bg-background"
      aria-label="简历 PDF 预览"
    >
      <div className="flex h-full min-w-[380px] flex-col">
        <div className="flex h-14 shrink-0 items-center justify-between border-b border-border px-4">
          <div className="flex min-w-0 items-center gap-2.5">
            <div className="grid h-8 w-8 shrink-0 place-items-center rounded-md bg-primary/10 text-primary">
              <FileText className="h-4 w-4" />
            </div>
            <div className="min-w-0">
              <p className="truncate text-sm font-medium text-foreground">{document.filename}</p>
              <p className="text-[11px] text-muted-foreground">PDF 简历预览</p>
            </div>
          </div>

          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => setReloadKey((value) => value + 1)}
              className="grid h-8 w-8 place-items-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
              title="重新加载"
            >
              <RefreshCw className="h-4 w-4" />
            </button>
            <a
              href={document.download_url}
              download={document.filename}
              className="grid h-8 w-8 place-items-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
              title="下载 PDF"
            >
              <Download className="h-4 w-4" />
            </a>
            <button
              type="button"
              onClick={() => setResumePanelOpen(false)}
              className="grid h-8 w-8 place-items-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
              title="关闭预览"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>

        <div className="min-h-0 flex-1 bg-muted/35 p-2">
          {previewUrl ? (
            <iframe
              key={previewUrl}
              src={previewUrl}
              title={`${document.filename} 预览`}
              className="h-full w-full border-0 bg-white"
            />
          ) : (
            <div className="grid h-full place-items-center text-sm text-muted-foreground">
              {previewError || "正在加载简历预览"}
            </div>
          )}
        </div>
      </div>
    </motion.aside>
  )
}
