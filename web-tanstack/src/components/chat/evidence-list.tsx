/**
 * EvidenceList — 阶段三证据列表展示
 *
 * 展示 complete 事件携带的 chunk 级证据：
 * - 每条证据的文档标题、相关性分数
 * - verified 标记（绿色 ✓ / 红色 ✗）
 * - retrieved_at_round 标签（第几轮检索获取）
 * - relevance_score 进度条
 */
import { memo, useState } from "react"
import { CheckCircle2, XCircle, ChevronDown, ChevronRight, FileText } from "lucide-react"

import type { RetrievalEvidence } from "@/types/api"
import { cn } from "@/lib/utils"

interface EvidenceListProps {
  evidence: RetrievalEvidence[]
  defaultExpanded?: boolean
}

const EvidenceCard = memo(function EvidenceCard({ item, index }: { item: RetrievalEvidence; index: number }) {
  const [expanded, setExpanded] = useState(false)
  const verified = item.verified === true
  const score = item.relevance_score ?? item.score ?? 0
  const title = item.document_title || `证据 ${index + 1}`
  const round = item.retrieved_at_round
  const text = item.text ?? ""

  return (
    <div
      className={cn(
        "rounded-lg border bg-white",
        verified ? "border-emerald-200" : "border-neutral-200",
      )}
    >
      <button
        type="button"
        className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-neutral-50"
        onClick={() => setExpanded(!expanded)}
      >
        {/* verified 标记 */}
        {verified ? (
          <CheckCircle2 className="size-3.5 shrink-0 text-emerald-500" />
        ) : (
          <XCircle className="size-3.5 shrink-0 text-neutral-300" />
        )}
        <FileText className="size-3 shrink-0 text-neutral-400" />
        <span className="flex-1 truncate text-xs font-medium text-neutral-800" title={title}>
          {title}
        </span>
        {/* 相关性分数进度条 */}
        <div className="flex items-center gap-1.5">
          <div className="hidden h-1.5 w-12 overflow-hidden rounded-full bg-neutral-100 sm:block">
            <div
              className={cn(
                "h-full transition-all",
                verified ? "bg-emerald-400" : "bg-neutral-300",
              )}
              style={{ width: `${Math.min(score * 100, 100)}%` }}
            />
          </div>
          <span className="text-[10px] tabular-nums text-neutral-400">
            {score.toFixed(3)}
          </span>
        </div>
        {/* 检索轮次标签 */}
        {typeof round === "number" ? (
          <span className="shrink-0 rounded bg-neutral-100 px-1.5 py-0.5 text-[9px] font-medium text-neutral-500">
            R{round}
          </span>
        ) : null}
        {expanded ? (
          <ChevronDown className="size-3 shrink-0 text-neutral-400" />
        ) : (
          <ChevronRight className="size-3 shrink-0 text-neutral-400" />
        )}
      </button>
      {expanded ? (
        <div className="border-t border-neutral-100 px-3 py-2">
          {item.section_path?.length ? (
            <div className="mb-1.5 text-[10px] text-neutral-400">
              {item.section_path.join(" › ")}
            </div>
          ) : null}
          <pre className="whitespace-pre-wrap break-words rounded bg-neutral-50 p-2 text-[11px] leading-5 text-neutral-700 max-h-40 overflow-y-auto">
{text.slice(0, 500)}{text.length > 500 ? "…" : ""}
          </pre>
        </div>
      ) : null}
    </div>
  )
})

export const EvidenceList = memo(function EvidenceList({
  evidence,
  defaultExpanded = false,
}: EvidenceListProps) {
  if (!evidence || evidence.length === 0) {
    return null
  }

  const verifiedCount = evidence.filter((e) => e.verified === true).length
  const [showAll, setShowAll] = useState(defaultExpanded)
  const displayed = showAll ? evidence : evidence.slice(0, 3)
  const hiddenCount = evidence.length - displayed.length

  return (
    <div className="mt-3 space-y-2 border-t border-[var(--blue-line)] pt-3">
      <div className="flex items-center justify-between">
        <div className="text-[11px] uppercase tracking-[0.18em] text-slate-500">
          Evidence · {evidence.length}
        </div>
        {verifiedCount > 0 ? (
          <div className="text-[10px] text-emerald-600">
            {verifiedCount} 条已验证
          </div>
        ) : null}
      </div>
      <div className="space-y-1.5">
        {displayed.map((item, idx) => (
          <EvidenceCard key={item.chunk_id ?? idx} item={item} index={idx} />
        ))}
      </div>
      {hiddenCount > 0 ? (
        <button
          type="button"
          className="w-full rounded-lg border border-dashed border-neutral-200 py-1.5 text-[11px] text-neutral-500 hover:bg-neutral-50"
          onClick={() => setShowAll(true)}
        >
          展开剩余 {hiddenCount} 条证据
        </button>
      ) : showAll && evidence.length > 3 ? (
        <button
          type="button"
          className="w-full rounded-lg border border-dashed border-neutral-200 py-1.5 text-[11px] text-neutral-500 hover:bg-neutral-50"
          onClick={() => setShowAll(false)}
        >
          收起证据
        </button>
      ) : null}
    </div>
  )
})
