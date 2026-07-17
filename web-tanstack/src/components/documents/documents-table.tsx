import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  type SortingState,
  useReactTable,
} from "@tanstack/react-table"
import { useVirtualizer } from "@tanstack/react-virtual"
import { CheckCircle2, FileUp, FolderUp, LoaderCircle, RefreshCw, Trash2, Upload, XCircle } from "lucide-react"
import { useCallback, useMemo, useRef, useState } from "react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { api } from "@/lib/api"
import { cn, formatBytes, formatTime } from "@/lib/utils"
import type { DocumentItem, DocumentUploadBatchResponse } from "@/types/api"

const STATUS_STYLE: Record<string, string> = {
  completed: "border-emerald-200 bg-emerald-50 text-emerald-700",
  processing: "border-neutral-300 bg-neutral-100 text-neutral-700",
  failed: "border-rose-200 bg-rose-50 text-rose-700",
}

export function DocumentsTable() {
  const queryClient = useQueryClient()
  const parentRef = useRef<HTMLDivElement>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const folderRef = useRef<HTMLInputElement>(null)
  const [sorting, setSorting] = useState<SortingState>([])
  const [knowledgeSpaceId, setKnowledgeSpaceId] = useState("")
  const [filterSpaceId, setFilterSpaceId] = useState("")
  const [uploadHint, setUploadHint] = useState<string | null>(null)
  const [uploadSuccess, setUploadSuccess] = useState(false)
  const [uploadQueue, setUploadQueue] = useState<{ filename: string; status: "uploading" | "success" | "failed" | "duplicated"; error?: string }[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set())
  const [showBatchDeleteConfirm, setShowBatchDeleteConfirm] = useState(false)
  const [batchDeleteProgress, setBatchDeleteProgress] = useState<{ current: number; total: number } | null>(null)

  const documentsQuery = useQuery({
    queryKey: ["documents", filterSpaceId],
    queryFn: async () => {
      const result = await api.listDocuments(0, 100, filterSpaceId || undefined)
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
    mutationFn: async (files: File[]) => {
      const result = await api.uploadDocumentsBatch(files, knowledgeSpaceId)
      if (result.error || !result.data) {
        throw new Error(result.error || "上传失败")
      }
      return result.data
    },
    onMutate: (files: File[]) => {
      setUploadQueue(files.map((f) => ({ filename: f.name, status: "uploading" as const })))
      setUploadHint(null)
      setUploadSuccess(false)
    },
    onSuccess: async (data: DocumentUploadBatchResponse) => {
      setUploadQueue(data.results.map((r) => ({ filename: r.filename, status: r.status, error: r.error })))
      setUploadSuccess(data.failed === 0)
      setUploadHint(data.message)
      console.log("[文档上传] 成功:", data)
      await queryClient.invalidateQueries({ queryKey: ["documents"] })
    },
    onError: (error: Error) => {
      setUploadQueue((prev) => prev.map((item) => ({ ...item, status: "failed" as const, error: error.message })))
      setUploadSuccess(false)
      setUploadHint(error.message)
      console.error("[文档上传] 失败:", error.message)
    },
  })

  const data = useMemo(() => documentsQuery.data?.documents ?? [], [documentsQuery.data])

  const toggleSelect = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }, [])

  const toggleSelectAll = useCallback(() => {
    setSelectedIds((prev) => {
      if (prev.size === data.length && data.length > 0) {
        return new Set()
      }
      return new Set(data.map((d) => d.id))
    })
  }, [data])

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

  const deleteBatchMutation = useMutation({
    mutationFn: async (ids: string[]) => {
      const results: { id: string; ok: boolean; error?: string }[] = []
      for (let i = 0; i < ids.length; i++) {
        setBatchDeleteProgress({ current: i + 1, total: ids.length })
        const result = await api.deleteDocument(ids[i])
        if (result.error) {
          results.push({ id: ids[i], ok: false, error: result.error })
        } else {
          results.push({ id: ids[i], ok: true })
        }
      }
      return results
    },
    onSuccess: async (results) => {
      setBatchDeleteProgress(null)
      const successCount = results.filter((r) => r.ok).length
      const failedCount = results.length - successCount
      if (failedCount === 0) {
        setUploadSuccess(true)
        setUploadHint(`批量删除完成：${successCount} 篇文档已删除`)
      } else {
        setUploadSuccess(false)
        setUploadHint(`批量删除完成：${successCount} 成功，${failedCount} 失败`)
      }
      setSelectedIds(new Set())
      setShowBatchDeleteConfirm(false)
      await queryClient.invalidateQueries({ queryKey: ["documents"] })
    },
    onError: (error: Error) => {
      setBatchDeleteProgress(null)
      setUploadSuccess(false)
      setUploadHint(`批量删除失败：${error.message}`)
    },
  })

  const columns = useMemo(
    () => [
      {
        id: "select",
        header: () => (
          <input
            type="checkbox"
            className="size-4 cursor-pointer rounded border-slate-300 accent-slate-800"
            checked={data.length > 0 && selectedIds.size === data.length}
            onChange={toggleSelectAll}
          />
        ),
        cell: ({ row }: { row: { original: DocumentItem } }) => (
          <input
            type="checkbox"
            className="size-4 cursor-pointer rounded border-slate-300 accent-slate-800"
            checked={selectedIds.has(row.original.id)}
            onChange={() => toggleSelect(row.original.id)}
          />
        ),
      },
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
    [deleteMutation, selectedIds, toggleSelect, toggleSelectAll],
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

  const handlePickFolder = () => {
    folderRef.current?.click()
  }

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const fileList = event.target.files
    if (fileList && fileList.length > 0) {
      uploadMutation.mutate(Array.from(fileList))
    }
    // 清空 input，便于重复选择同一文件
    event.target.value = ""
  }

  const handleFolderChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const fileList = event.target.files
    if (fileList && fileList.length > 0) {
      // 过滤掉非文件项（如目录本身）和隐藏文件，只保留真实文件
      const files = Array.from(fileList).filter(
        (f) => f.size > 0 && !f.name.startsWith("."),
      )
      if (files.length > 0) {
        uploadMutation.mutate(files)
      }
    }
    // 清空 input，便于重复选择同一文件夹
    event.target.value = ""
  }

  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault()
      setIsDragging(false)
      const droppedFiles = event.dataTransfer.files
      if (droppedFiles && droppedFiles.length > 0) {
        uploadMutation.mutate(Array.from(droppedFiles))
      }
    },
    [uploadMutation],
  )

  const handleDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    setIsDragging(false)
  }, [])

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

          <div
            className={cn(
              "space-y-3 rounded-2xl border border-dashed p-4 transition-colors",
              isDragging ? "border-neutral-900 bg-neutral-50" : "border-slate-300 bg-white",
            )}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
          >
            <input
              className="hidden"
              multiple
              onChange={handleFileChange}
              ref={fileRef}
              type="file"
            />
            {/* 文件夹选择 input：webkitdirectory 支持递归选择文件夹下所有文件 */}
            <input
              className="hidden"
              multiple
              onChange={handleFolderChange}
              ref={folderRef}
              type="file"
              // @ts-expect-error webkitdirectory 是非标准属性，TS 类型定义未包含
              webkitdirectory=""
              directory=""
            />
            <div className="flex gap-2">
              <Button
                className="flex-1"
                onClick={handlePickFile}
                variant="outline"
                disabled={uploadMutation.isPending}
              >
                {uploadMutation.isPending ? (
                  <LoaderCircle className="size-4 animate-spin" />
                ) : (
                  <FileUp className="size-4" />
                )}
                选择文件
              </Button>
              <Button
                className="flex-1"
                onClick={handlePickFolder}
                variant="outline"
                disabled={uploadMutation.isPending}
                title="选择文件夹会自动添加该文件夹下所有支持格式的文件（含子目录）"
              >
                {uploadMutation.isPending ? (
                  <LoaderCircle className="size-4 animate-spin" />
                ) : (
                  <FolderUp className="size-4" />
                )}
                选择文件夹
              </Button>
            </div>
            <div className="text-xs leading-5 text-slate-500">
              支持 PDF / Word / Markdown / TXT / 图片等。可选择文件、文件夹或拖拽到此处批量上传。
            </div>

            {uploadQueue.length > 0 ? (
              <div className="space-y-1.5 border-t border-slate-100 pt-3">
                {uploadQueue.map((item, idx) => (
                  <div key={idx} className="flex items-center gap-2 text-xs">
                    {item.status === "uploading" ? (
                      <LoaderCircle className="size-3.5 shrink-0 animate-spin text-slate-400" />
                    ) : item.status === "success" ? (
                      <CheckCircle2 className="size-3.5 shrink-0 text-emerald-600" />
                    ) : (
                      <XCircle className="size-3.5 shrink-0 text-rose-600" />
                    )}
                    <span className="flex-1 truncate text-slate-700">{item.filename}</span>
                    {item.status === "duplicated" ? (
                      <span className="shrink-0 text-amber-600">重复</span>
                    ) : null}
                    {item.error && item.status !== "duplicated" ? (
                      <span className="shrink-0 text-rose-500">{item.error}</span>
                    ) : null}
                  </div>
                ))}
              </div>
            ) : null}
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
            <div className="flex items-center gap-2">
              <select
                className="rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-xs text-slate-600 outline-none focus:border-neutral-400"
                value={filterSpaceId}
                onChange={(e) => setFilterSpaceId(e.target.value)}
              >
                <option value="">全部分类</option>
                {(spacesQuery.data ?? []).map((space) => (
                  <option key={space.id} value={space.id}>
                    {space.name}
                  </option>
                ))}
              </select>
              <Badge>{rows.length} 篇</Badge>
              {selectedIds.size > 0 ? (
                <Button
                  onClick={() => setShowBatchDeleteConfirm(true)}
                  size="sm"
                  variant="outline"
                  className="border-rose-200 text-rose-600 hover:bg-rose-50"
                  disabled={deleteBatchMutation.isPending}
                >
                  <Trash2 className="size-3.5" />
                  批量删除 ({selectedIds.size})
                </Button>
              ) : null}
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
          <div className="grid shrink-0 grid-cols-[auto_2fr_1fr_1.5fr_1fr_1fr_auto] gap-4 border-b border-slate-200 bg-slate-50 px-4 py-3 text-xs uppercase tracking-[0.16em] text-slate-500">
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
                      className="grid grid-cols-[auto_2fr_1fr_1.5fr_1fr_1fr_auto] gap-4 border-b border-slate-100 px-4 py-4 text-sm text-slate-700"
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

      {/* 批量删除确认弹窗 */}
      {showBatchDeleteConfirm ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
          <div className="w-[360px] rounded-xl bg-white p-6 shadow-xl">
            <h3 className="text-sm font-semibold text-slate-800">确认批量删除</h3>
            <p className="mt-2 text-sm text-slate-600">
              确认删除选中的 {selectedIds.size} 篇文档？该操作不可恢复。
            </p>
            {batchDeleteProgress ? (
              <div className="mt-3">
                <div className="mb-1 flex items-center justify-between text-xs text-slate-500">
                  <span>正在删除...</span>
                  <span>{batchDeleteProgress.current} / {batchDeleteProgress.total}</span>
                </div>
                <div className="h-1.5 overflow-hidden rounded-full bg-slate-100">
                  <div
                    className="h-full bg-rose-500 transition-all"
                    style={{ width: `${(batchDeleteProgress.current / batchDeleteProgress.total) * 100}%` }}
                  />
                </div>
              </div>
            ) : null}
            <div className="mt-5 flex justify-end gap-2">
              <Button
                onClick={() => setShowBatchDeleteConfirm(false)}
                size="sm"
                variant="ghost"
                disabled={deleteBatchMutation.isPending}
              >
                取消
              </Button>
              <Button
                onClick={() => deleteBatchMutation.mutate(Array.from(selectedIds))}
                size="sm"
                className="border-rose-200 bg-rose-500 text-white hover:bg-rose-600"
                disabled={deleteBatchMutation.isPending}
              >
                {deleteBatchMutation.isPending ? <LoaderCircle className="size-3.5 animate-spin" /> : null}
                确认删除
              </Button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}
