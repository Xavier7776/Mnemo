import { useMutation } from "@tanstack/react-query"
import { FileSearch, LoaderCircle, Search, Sparkles } from "lucide-react"
import { useState } from "react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { api } from "@/lib/api"
import { cn, truncate } from "@/lib/utils"
import type { RetrievalSearchResponse } from "@/types/api"

const TOP_K_OPTIONS = [3, 5, 8, 10]

export function RetrievalSearch() {
  const [query, setQuery] = useState("")
  const [topK, setTopK] = useState(5)
  const [knowledgeSpaceIds, setKnowledgeSpaceIds] = useState("")
  const [result, setResult] = useState<RetrievalSearchResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const searchMutation = useMutation({
    mutationFn: async () => {
      const ids = knowledgeSpaceIds
        .split(/[,，\s]+/)
        .map((s) => s.trim())
        .filter(Boolean)
      const res = await api.searchRetrieval({
        query: query.trim(),
        top_k: topK,
        knowledge_space_ids: ids.length ? ids : undefined,
      })
      if (res.error || !res.data) {
        throw new Error(res.error || "检索失败")
      }
      return res.data
    },
    onSuccess: (data) => {
      setResult(data)
      setError(null)
    },
    onError: (err: Error) => {
      setError(err.message)
      setResult(null)
    },
  })

  const handleSearch = () => {
    if (!query.trim() || searchMutation.isPending) {
      return
    }
    searchMutation.mutate()
  }

  const handleKey = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault()
      handleSearch()
    }
  }

  return (
    <div className="grid gap-5 xl:grid-cols-[360px_minmax(0,1fr)]">
      <Card>
        <CardHeader>
          <CardTitle>Query</CardTitle>
          <CardDescription>POST /api/retrieval/search · 检索知识库</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <label className="text-xs font-medium uppercase tracking-[0.16em] text-slate-500">
              查询问题
            </label>
            <Textarea
              placeholder="输入检索问题，⌘/Ctrl + Enter 检索。例如：高级 RAG 的混合检索是如何合并结果的？"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={handleKey}
              className="min-h-[120px]"
            />
          </div>

          <div className="space-y-2">
            <label className="text-xs font-medium uppercase tracking-[0.16em] text-slate-500">
              Top K
            </label>
            <div className="flex flex-wrap gap-2">
              {TOP_K_OPTIONS.map((value) => (
                <button
                  key={value}
                  className={cn(
                    "rounded-full border px-3 py-1.5 text-sm transition-all",
                    topK === value
                      ? "border-black bg-white text-black font-medium"
                      : "border-[var(--blue-line)] bg-white text-neutral-600 hover:bg-neutral-50",
                  )}
                  onClick={() => setTopK(value)}
                  type="button"
                >
                  {value}
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-xs font-medium uppercase tracking-[0.16em] text-slate-500">
              知识空间 ID（可选，逗号分隔）
            </label>
            <Input
              placeholder="space-id-1, space-id-2"
              value={knowledgeSpaceIds}
              onChange={(event) => setKnowledgeSpaceIds(event.target.value)}
            />
          </div>

          <Button
            className="w-full"
            disabled={!query.trim() || searchMutation.isPending}
            onClick={handleSearch}
            size="lg"
          >
            {searchMutation.isPending ? (
              <LoaderCircle className="size-4 animate-spin" />
            ) : (
              <Search className="size-4" />
            )}
            检索
          </Button>

          {error ? (
            <div className="rounded-2xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
              {error}
            </div>
          ) : null}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <CardTitle>Results</CardTitle>
              <CardDescription>检索返回的上下文、来源、证据与推荐资源。</CardDescription>
            </div>
            {result ? (
              <div className="flex gap-2">
                <Badge>{result.retrieval_count} sources</Badge>
                <Badge>{result.evidence?.length ?? 0} evidence</Badge>
                <Badge>{result.recommended_resources?.length ?? 0} resources</Badge>
              </div>
            ) : null}
          </div>
        </CardHeader>
        <CardContent className="space-y-5">
          {!result && !searchMutation.isPending ? (
            <div className="flex h-[360px] flex-col items-center justify-center gap-3 text-center text-sm text-slate-400">
              <FileSearch className="size-10 opacity-40" />
              <div>输入问题并点击「检索」查看结果。</div>
            </div>
          ) : null}

          {searchMutation.isPending && !result ? (
            <div className="flex h-[360px] items-center justify-center text-sm text-slate-400">
              <LoaderCircle className="mr-2 size-4 animate-spin" />
              正在检索…
            </div>
          ) : null}

          {result ? (
            <>
              <section className="space-y-2">
                <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-[0.16em] text-slate-500">
                  <Sparkles className="size-3.5" />
                  Context
                </div>
                <div className="blue-panel max-h-[260px] overflow-y-auto rounded-2xl p-4 text-sm leading-6 text-slate-700 whitespace-pre-wrap break-words">
                  {result.context || "（无上下文）"}
                </div>
              </section>

              <section className="space-y-2">
                <div className="text-xs font-medium uppercase tracking-[0.16em] text-slate-500">
                  Sources
                </div>
                <div className="grid gap-2">
                  {(result.sources ?? []).map((source, index) => (
                    <div
                      key={source.chunk_id || source.document_id || index}
                      className="rounded-2xl border border-[var(--blue-line)] bg-white px-4 py-3"
                    >
                      <div className="flex items-center justify-between gap-3">
                        <div className="truncate text-sm font-medium text-slate-950">
                          {source.title || source.source || `Source ${index + 1}`}
                        </div>
                        {typeof source.score === "number" ? (
                          <Badge>score {source.score.toFixed(3)}</Badge>
                        ) : null}
                      </div>
                      {source.content ? (
                        <div className="mt-1 text-xs leading-5 text-slate-600">
                          {truncate(source.content, 240)}
                        </div>
                      ) : null}
                    </div>
                  ))}
                  {(result.sources ?? []).length === 0 ? (
                    <div className="text-sm text-slate-400">未检索到来源。</div>
                  ) : null}
                </div>
              </section>

              {result.query_plan?.sub_queries?.length ? (
                <section className="space-y-2">
                  <div className="text-xs font-medium uppercase tracking-[0.16em] text-slate-500">
                    Query Plan
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {result.query_plan.sub_queries.map((sub, index) => (
                      <Badge key={index} className="bg-white text-slate-700">
                        {sub}
                      </Badge>
                    ))}
                  </div>
                </section>
              ) : null}

              {result.recommended_resources?.length ? (
                <section className="space-y-2">
                  <div className="text-xs font-medium uppercase tracking-[0.16em] text-slate-500">
                    Recommended Resources
                  </div>
                  <div className="grid gap-2">
                    {result.recommended_resources.map((resource, index) => (
                      <a
                        key={index}
                        href={resource.url || "#"}
                        target="_blank"
                        rel="noreferrer"
                        className="block rounded-2xl border border-neutral-200 bg-neutral-50 px-4 py-3 transition-all hover:border-neutral-400 hover:bg-neutral-100"
                      >
                        <div className="truncate text-sm font-medium text-slate-900">
                          {resource.title || resource.url || "推荐资源"}
                        </div>
                        {resource.description ? (
                          <div className="mt-0.5 line-clamp-2 text-xs text-slate-600">
                            {resource.description}
                          </div>
                        ) : null}
                      </a>
                    ))}
                  </div>
                </section>
              ) : null}
            </>
          ) : null}
        </CardContent>
      </Card>
    </div>
  )
}
