import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  type SortingState,
  useReactTable,
} from "@tanstack/react-table"
import { useVirtualizer } from "@tanstack/react-virtual"
import { FileUp, LoaderCircle, RefreshCw, Trash2, Upload } from "lucide-react"
import { useMemo, useRef, useState } from "react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { api } from "@/lib/api"
import { cn, formatBytes, formatTime } from "@/lib/utils"
import type { DocumentItem } from "@/types/api"

const STATUS_STYLE: Record<string, string> = {
  completed: "border-emerald-200 bg-emerald-50 text-emerald-700",
  processing: "border-neutral-300 bg-neutral-100 text-neutral-700",
  failed: "border-rose-200 bg-rose-50 text-rose-700",
}

export function DocumentsTable() {
  const queryClient = useQueryClient()
  const parentRef = useRef<HTMLDivElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const [sorting, setSorting] = useState<SortingState>([])
  const [knowledgeSpaceId, setKnowledgeSpaceId] = useState("")
  const [uploadHint, setUploadHint] = useState<string | null>(null)
  const [uploadSuccess, setUploadSuccess] = useState(false)

  const documentsQuery = useQuery({
    queryKey: ["documents"],
    queryFn: async () => {
      const result = await api.listDocuments()
      if (result.error) {
        throw new Error(result.error)
      }
      return result.data
    },
    // 若有处理中文档，则渐进退避轮询（3s→5s→10s→20s→30s），
    // 避免后端处理慢时无限快速轮询；无处理中文档则停止轮询
    refetchInterval: (query) => {
      const docs = query.state.data?.documents
      if (!docs?.some((doc) => doc.status === "processing")) {
        return false
      }
      const updates = query.state.dataUpdateCount
      if (updates <= 5) return 3000
      if (updates <= 15) return 5000
      if (updates <= 30) return 10000
      if (updates <= 60) return 20000
      return 30000
    },
  })

  // 知识空间列表：上传时下拉选择，留空走后端默认策略
  const spacesQuery = useQuery({
    queryKey: ["knowledge-spaces"],
    queryFn: async () => {
      const result = await api.listKnowledgeSpaces()
      if (result.error) {
        throw new Error(result.error)
      }
      // 后端可能返回 {spaces: [...]} 或直接是数组，兼容两种
      const data = result.data
      const spaces = Array.isArray(data) ? data : data?.spaces ?? []
      return spaces
    },
    retry: 0,
  })

  const uploadMutation = useMutation({
    mutationFn: async (file: File) => {
      const result = await api.uploadDocument(file, knowledgeSpaceId)
      if (result.error) {
        throw new Error(result.error)
      }
      return result.data
    },
    onSuccess: async (data) => {
      setUploadSuccess(true)
      setUploadHint(data?.message || "文件上传成功，正在后台处理。")
      console.log("[文档上传] 成功:", data)
      await queryClient.invalidateQueries({ queryKey: ["documents"] })
    },
    onError: (error: Error) => {
      setUploadSuccess(false)
      setUploadHint(error.message)
      console.error("[文档上传] 失败:", error.message)
    },
  })

  const data = useMemo(() => documentsQuery.data?.documents ?? [], [documentsQuery.data])

  const deleteMutation = useMutation({
    mutationFn: async (docId: string) => {
      const result = await api.deleteDocument(docId)
      if (result.error) {
        throw new Error(result.error)
      }
      return result.data
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["documents"] })
    },
    onError: (error: Error) => {
      setUploadSuccess(false)
      setUploadHint(`删除失败：${error.message}`)
    },
  })

  const columns = useMemo(
    () => [
      {
        accessorKey: "title",
        header: "Title",
        cell: ({ row }: { row: { original: DocumentItem } }) => (
          <div>
            <div className="font-medium text-slate-950">{row.original.title}</div>
            <div className="text-xs uppercase tracking-wide text-slate-500">
              {row.original.file_type}
            </div>
          </div>
        ),
      },
      {
        accessorKey: "status",
        header: "Status",
        cell: ({ row }: { row: { original: DocumentItem } }) => (
          <Badge className={cn("capitalize", STATUS_STYLE[row.original.status] ?? "")}>
            {row.original.status}
          </Badge>
        ),
      },
      {
        accessorKey: "progress_percentage",
        header: "Progress",
        cell: ({ row }: { row: { original: DocumentItem } }) => {
          const progress = row.original.progress_percentage ?? 0
          return (
            <div className="min-w-[140px]">
              <div className="mb-1 flex justify-between text-xs text-slate-500">
                <span className="truncate">{row.original.current_stage || "queued"}</span>
                <span>{progress}%</span>
              </div>
              <div className="h-2 rounded-full bg-neutral-100">
                <div
                  className="h-2 rounded-full bg-black transition-all"
                  style={{ width: `${progress}%` }}
                />
              </div>
            </div>
          )
        },
      },
      {
        accessorKey: "file_size",
        header: "Size",
        cell: ({ row }: { row: { original: DocumentItem } }) => (
          <span className="tabular-nums">{formatBytes(row.original.file_size)}</span>
        ),
      },
      {
        accessorKey: "created_at",
        header: "Created",
        cell: ({ row }: { row: { original: DocumentItem } }) => (
          <span className="tabular-nums">{formatTime(row.original.created_at)}</span>
        ),
      },
      {
        id: "actions",
        header: "操作",
        cell: ({ row }: { row: { original: DocumentItem } }) => (
          <button
            className="inline-flex size-7 items-center justify-center rounded-lg text-slate-400 transition-colors hover:bg-rose-50 hover:text-rose-600"
            title="删除文档"
            type="button"
            onClick={(event) => {
              event.stopPropagation()
              const doc = row.original
              if (window.confirm(`确认删除文档「${doc.title}」？该操作不可恢复。`)) {
                deleteMutation.mutate(doc.id)
              }
            }}
          >
            <Trash2 className="size-3.5" />
          </button>
        ),
      },
    ],
    [deleteMutation],
  )

  // TanStack Table 实例；按官方用法保持非 memo
  // eslint-disable-next-line react-hooks/incompatible-library
  const table = useReactTable({
    data,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  })

  const rows = table.getRowModel().rows

  const rowVirtualizer = useVirtualizer({
    count: rows.length,
    estimateSize: () => 84,
    getScrollElement: () => parentRef.current,
    overscan: 8,
  })

  const errors = [
    documentsQuery.error instanceof Error ? `文档列表加载失败：${documentsQuery.error.message}` : null,
    uploadMutation.error instanceof Error ? `上传失败：${uploadMutation.error.message}` : null,
  ].filter(Boolean) as string[]

  const handlePickFile = () => {
    fileRef.current?.click()
  }

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setUploadHint(null)
      setUploadSuccess(false)
      uploadMutation.mutate(file)
    }
    // 清空 input，便于重复选择同一文件
    event.target.value = ""
  }

  return (
    <div className="grid h-full gap-4 p-4 xl:grid-cols-[320px_minmax(0,1fr)]">
      <Card className="flex flex-col overflow-hidden">
        <CardHeader className="shrink-0 pb-3">
          <CardTitle className="text-sm">上传</CardTitle>
          <CardDescription className="text-xs">multipart/form-data</CardDescription>
        </CardHeader>
        <CardContent className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto">
          {errors.length > 0 ? (
            <div className="rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700">
              {errors.map((error) => (
                <div key={error}>{error}</div>
              ))}
            </div>
          ) : null}

          <div className="space-y-2">
            <label className="text-xs font-medium uppercase tracking-[0.16em] text-slate-500">
              知识空间（可选，可输入自定义名称）
            </label>
            <input
              className="h-10 w-full rounded-xl border border-neutral-200 bg-white px-3 text-sm text-neutral-800 outline-none focus:border-neutral-400 focus:ring-2 focus:ring-neutral-200"
              list="knowledge-spaces-options"
              placeholder="留空走默认策略，或输入/选择知识空间"
              value={knowledgeSpaceId}
              onChange={(event) => setKnowledgeSpaceId(event.target.value)}
            />
            <datalist id="knowledge-spaces-options">
              {(spacesQuery.data ?? []).map((space) => {
                const value = space.id ?? space.name
                return (
                  <option key={value} value={value}>
                    {space.name}
                    {space.document_count ? `（${space.document_count} 篇）` : ""}
                  </option>
                )
              })}
            </datalist>
            {spacesQuery.isError ? (
              <p className="text-[11px] leading-5 text-amber-600">
                知识空间列表加载失败，可手动输入或留空使用默认策略。
              </p>
            ) : (
              <p className="text-[11px] leading-5 text-slate-500">
                可从下拉选择已有知识空间，或手动输入新名称；留空走默认策略。
              </p>
            )}
          </div>

          <div className="space-y-3 rounded-2xl border border-dashed border-slate-300 bg-white p-4">
            <input
              className="hidden"
              onChange={handleFileChange}
              ref={fileRef}
              type="file"
            />
            <Button
              className="w-full"
              onClick={handlePickFile}
              variant="outline"
              disabled={uploadMutation.isPending}
            >
              {uploadMutation.isPending ? (
                <LoaderCircle className="size-4 animate-spin" />
              ) : (
                <FileUp className="size-4" />
              )}
              选择文件上传
            </Button>
            <div className="text-xs leading-5 text-slate-500">
              支持 PDF / Word / Markdown / TXT / 图片等。上传后自动开始解析、分块与向量化。
            </div>
          </div>

          {uploadHint ? (
            <div
              className={cn(
                "rounded-xl border px-3 py-2 text-xs",
                uploadSuccess
                  ? "border-emerald-200 bg-emerald-50 text-emerald-800"
                  : "border-rose-200 bg-rose-50 text-rose-800",
              )}
            >
              {uploadHint}
            </div>
          ) : null}
        </CardContent>
      </Card>

      <Card className="flex min-h-0 flex-col overflow-hidden">
        <CardHeader className="shrink-0 border-b border-[var(--blue-line)] pb-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <CardTitle className="text-sm">文档列表</CardTitle>
              <CardDescription className="text-xs">虚拟滚动表格</CardDescription>
            </div>
            <div className="flex gap-2">
              <Badge>{rows.length} 篇</Badge>
              <Button
                onClick={() => queryClient.invalidateQueries({ queryKey: ["documents"] })}
                size="sm"
                variant="ghost"
              >
                <RefreshCw className={cn("size-3.5", documentsQuery.isFetching && "animate-spin")} />
                刷新
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="flex min-h-0 flex-1 flex-col overflow-hidden p-0">
          <div className="grid shrink-0 grid-cols-[2fr_1fr_1.5fr_1fr_1fr_auto] gap-4 border-b border-slate-200 bg-slate-50 px-4 py-3 text-xs uppercase tracking-[0.16em] text-slate-500">
            {table.getFlatHeaders().map((header) => (
              <button
                key={header.id}
                className="text-left transition-colors hover:text-slate-900 focus-visible:outline-none focus-visible:text-slate-900"
                onClick={header.column.getToggleSortingHandler()}
                type="button"
              >
                {flexRender(header.column.columnDef.header, header.getContext())}
              </button>
            ))}
          </div>

          <div className="min-h-0 flex-1 overflow-auto" ref={parentRef}>
            {rows.length === 0 ? (
              <div className="flex h-full items-center justify-center text-sm text-slate-400">
                {documentsQuery.isPending ? "加载中…" : "暂无文档，上传一个试试。"}
              </div>
            ) : (
              <div
                style={{
                  height: `${rowVirtualizer.getTotalSize()}px`,
                  position: "relative",
                }}
              >
                {rowVirtualizer.getVirtualItems().map((virtualRow) => {
                  const row = rows[virtualRow.index]
                  return (
                    <div
                      key={row.id}
                      className="grid grid-cols-[2fr_1fr_1.5fr_1fr_1fr_auto] gap-4 border-b border-slate-100 px-4 py-4 text-sm text-slate-700"
                      style={{
                        position: "absolute",
                        top: 0,
                        left: 0,
                        width: "100%",
                        transform: `translateY(${virtualRow.start}px)`,
                      }}
                    >
                      {row.getVisibleCells().map((cell) => (
                        <div key={cell.id} className="flex items-center">
                          {flexRender(cell.column.columnDef.cell, cell.getContext())}
                        </div>
                      ))}
                    </div>
                  )
                })}
              </div>
            )}
          </div>

          <div className="shrink-0 flex items-center gap-2 border-t border-[var(--blue-line)] px-4 py-2 text-xs text-slate-500">
            <Upload className="size-3.5" />
            <span>处理中文档自动每 3 秒刷新进度</span>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
