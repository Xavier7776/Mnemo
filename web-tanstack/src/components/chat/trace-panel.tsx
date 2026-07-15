/**
 * TracePanel — 阶段三 PAR 循环轨迹展示
 *
 * 展示 Plan-Act-Observe-Reflect 循环的完整轨迹：
 * - 检索轮次统计（total_retrievals）
 * - 证据验证统计（verified_evidence / total_evidence）
 * - 历次观察列表（observations）
 */
import { memo, useState } from "react"
import { CheckCircle2, Search, AlertCircle, ChevronDown, ChevronRight } from "lucide-react"

import type { RagTrace, TraceObservation } from "@/types/api"

interface TracePanelProps {
  trace: RagTrace
}

const ObservationItem = memo(function ObservationItem({ obs }: { obs: TraceObservation }) {
  const verifiedCount = obs.verified_count ?? 0
  const evidenceCount = obs.evidence_count ?? 0
  const topScore = obs.top_score ?? 0
  const hasEvidence = evidenceCount > 0

  return (
    <div className="rounded-lg border border-neutral-200 bg-neutral-50/50 px-3 py-2">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 text-xs font-medium text-neutral-700">
          <Search className="size-3 text-neutral-400" />
          <span>第 {obs.round ?? "?"} 轮检索</span>
        </div>
        {hasEvidence ? (
          <span className="text-[10px] text-neutral-500">
            {verifiedCount}/{evidenceCount} 已验证
          </span>
        ) : (
          <span className="text-[10px] text-rose-500">无结果</span>
        )}
      </div>
      {obs.query ? (
        <div className="mt-1 truncate text-[11px] text-neutral-500" title={obs.query}>
          "{obs.query}"
        </div>
      ) : null}
      {hasEvidence ? (
        <div className="mt-1 flex items-center gap-3 text-[10px] text-neutral-400">
          <span>top_score · {topScore.toFixed(3)}</span>
          {verifiedCount > 0 ? (
            <span className="inline-flex items-center gap-0.5 text-emerald-600">
              <CheckCircle2 className="size-2.5" />
              {verifiedCount} 条验证通过
            </span>
          ) : null}
        </div>
      ) : null}
    </div>
  )
})

export const TracePanel = memo(function TracePanel({ trace }: TracePanelProps) {
  const [expanded, setExpanded] = useState(false)
  const observations = trace.observations ?? []
  const totalRetrievals = trace.total_retrievals ?? 0
  const totalEvidence = trace.total_evidence ?? 0
  const verifiedEvidence = trace.verified_evidence ?? 0
  const verifiedRate = totalEvidence > 0 ? (verifiedEvidence / totalEvidence) * 100 : 0

  if (observations.length === 0 && totalRetrievals === 0) {
    return null
  }

  return (
    <div className="mt-3 space-y-2 border-t border-[var(--blue-line)] pt-3">
      <button
        type="button"
        className="flex w-full items-center justify-between transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-slate-500">
          <Search className="size-3.5" />
          <span>检索轨迹 · PAR 循环</span>
        </div>
        <div className="flex items-center gap-2 text-[10px] text-neutral-400">
          <span>{totalRetrievals} 轮检索</span>
          <span>·</span>
          <span>{verifiedEvidence}/{totalEvidence} 验证</span>
          {verifiedRate > 0 ? (
            <span className="text-neutral-300">({verifiedRate.toFixed(0)}%)</span>
          ) : null}
          {expanded ? (
            <ChevronDown className="size-3.5 shrink-0" />
          ) : (
            <ChevronRight className="size-3.5 shrink-0" />
          )}
        </div>
      </button>
      {expanded ? (
        <>
          {/* 统计条 */}
          <div className="flex items-center gap-1.5">
            {totalEvidence > 0 ? (
              <div className="flex h-1.5 flex-1 overflow-hidden rounded-full bg-neutral-100">
                <div
                  className="bg-emerald-400 transition-all"
                  style={{ width: `${verifiedRate}%` }}
                />
              </div>
            ) : null}
          </div>

          {/* 观察列表 */}
          {observations.length > 0 ? (
            <div className="space-y-1.5">
              {observations.map((obs, idx) => (
                <ObservationItem key={idx} obs={obs} />
              ))}
            </div>
          ) : null}

          {/* 无验证证据告警 */}
          {totalEvidence > 0 && verifiedEvidence === 0 ? (
            <div className="flex items-center gap-1.5 rounded-lg bg-amber-50 px-2 py-1.5 text-[10px] text-amber-700">
              <AlertCircle className="size-3 shrink-0" />
              <span>所有证据均未通过验证，回答可能不准确</span>
            </div>
          ) : null}
        </>
      ) : null}
    </div>
  )
})
