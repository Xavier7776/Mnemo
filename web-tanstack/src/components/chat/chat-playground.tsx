import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Bot, Brain, ChevronLeft, Check, Copy, Cpu, Plus, SendHorizontal, Square, Trash2, User, Wrench, ChevronDown, ChevronRight } from "lucide-react"
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react"

import { Button } from "@/components/ui/button"
import { Markdown } from "@/components/ui/markdown"
import { Textarea } from "@/components/ui/textarea"
import { api } from "@/lib/api"
import { cn, formatTime } from "@/lib/utils"
import { useUiStore } from "@/stores/ui-store"
import type {
  ChatStreamEvent,
  ConversationDetail,
  ConversationMessage,
  MessageAddRequest,
  RetrievalEvidence,
  RagTrace,
  SourceInfo,
  StoredMessage,
  ToolCallEvent,
  ToolCallInfo,
} from "@/types/api"
import { TracePanel } from "@/components/chat/trace-panel"
import { EvidenceList } from "@/components/chat/evidence-list"

const SourceList = memo(function SourceList({ sources }: { sources: SourceInfo[] }) {
  if (sources.length === 0) {
    return null
  }
  return (
    <div className="mt-3 space-y-2 border-t border-[var(--blue-line)] pt-3">
      <div className="text-[11px] uppercase tracking-[0.18em] text-slate-500">
        Sources · {sources.length}
      </div>
      <div className="grid gap-2 sm:grid-cols-2">
        {sources.map((source, index) => (
          <div
            key={source.chunk_id || source.document_id || index}
            className="rounded-xl border border-[var(--blue-line)] bg-[var(--surface-blue)] px-3 py-2 text-xs text-slate-700"
          >
            <div className="truncate font-medium text-slate-900">
              {source.title || source.source || `Source ${index + 1}`}
            </div>
            {typeof source.score === "number" ? (
              <div className="mt-0.5 text-[10px] text-slate-500">
                score · {source.score.toFixed(3)}
              </div>
            ) : null}
          </div>
        ))}
      </div>
    </div>
  )
})

const ToolCallItem = memo(function ToolCallItem({ tool }: { tool: ToolCallInfo; round: number }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <div className="rounded-lg border border-neutral-200 bg-white">
      <button
        type="button"
        className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-neutral-50"
        onClick={() => setExpanded(!expanded)}
      >
        <span className="flex size-5 shrink-0 items-center justify-center rounded bg-neutral-100">
          <Wrench className="size-3 text-neutral-600" />
        </span>
        <span className="flex-1 truncate text-xs font-medium text-neutral-800">
          {tool.name}
        </span>
        <span
          className={cn(
            "shrink-0 rounded-full px-1.5 py-0.5 text-[10px] font-medium",
            tool.success
              ? "bg-emerald-50 text-emerald-700"
              : "bg-rose-50 text-rose-700",
          )}
        >
          {tool.success ? "成功" : "失败"}
        </span>
        {expanded ? (
          <ChevronDown className="size-3.5 shrink-0 text-neutral-400" />
        ) : (
          <ChevronRight className="size-3.5 shrink-0 text-neutral-400" />
        )}
      </button>
      {expanded ? (
        <div className="border-t border-neutral-100 px-3 py-2">
          <div className="mb-2">
            <div className="text-[10px] uppercase tracking-wider text-neutral-400 mb-1">
              参数
            </div>
            <pre className="rounded bg-neutral-50 p-2 text-[11px] text-neutral-700 overflow-x-auto">
{JSON.stringify(tool.params, null, 2)}
            </pre>
          </div>
          <div>
            <div className="text-[10px] uppercase tracking-wider text-neutral-400 mb-1">
              返回结果
            </div>
            <pre className="rounded bg-neutral-50 p-2 text-[11px] text-neutral-700 overflow-x-auto max-h-40 overflow-y-auto">
{typeof tool.result === "string"
  ? tool.result
  : JSON.stringify(tool.result, null, 2)}
            </pre>
          </div>
        </div>
      ) : null}
    </div>
  )
})

const ToolCallChain = memo(function ToolCallChain({ toolCalls }: { toolCalls: ToolCallEvent[] }) {
  const [expanded, setExpanded] = useState(false)
  if (toolCalls.length === 0) {
    return null
  }
  const totalTools = toolCalls.reduce((sum, tc) => sum + tc.tools.length, 0)
  return (
    <div className="mb-3 space-y-2">
      <button
        type="button"
        className="flex w-full items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-slate-500 transition-colors hover:text-slate-700"
        onClick={() => setExpanded(!expanded)}
      >
        <Wrench className="size-3.5" />
        <span>工具调用 · {toolCalls.length} 轮 · {totalTools} 次</span>
        {expanded ? (
          <ChevronDown className="size-3.5 shrink-0" />
        ) : (
          <ChevronRight className="size-3.5 shrink-0" />
        )}
      </button>
      {expanded ? (
        <div className="space-y-2">
          {toolCalls.map((tc, idx) => (
            <div key={idx} className="space-y-1.5">
              <div className="text-[10px] font-medium text-neutral-500 pl-1">
                第 {tc.round} 轮
              </div>
              <div className="space-y-1.5">
                {tc.tools.map((tool, tIdx) => (
                  <ToolCallItem key={tIdx} tool={tool} round={tc.round} />
                ))}
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  )
})

/** 思考过程面板 — 默认折叠，点击展开 */
const ThinkingPanel = memo(function ThinkingPanel({ thinking }: { thinking: string }) {
  const [expanded, setExpanded] = useState(false)
  if (!thinking.trim()) return null
  return (
    <div className="mb-3 space-y-1">
      <button
        type="button"
        className="flex w-full items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-slate-400 transition-colors hover:text-slate-600"
        onClick={() => setExpanded(!expanded)}
      >
        <Brain className="size-3.5" />
        <span>思考过程</span>
        {expanded ? (
          <ChevronDown className="size-3.5 shrink-0" />
        ) : (
          <ChevronRight className="size-3.5 shrink-0" />
        )}
      </button>
      {expanded ? (
        <div className="rounded-lg border border-slate-100 bg-slate-50 px-3 py-2">
          <pre className="whitespace-pre-wrap break-words text-[11px] leading-5 text-slate-600">
{thinking}
          </pre>
        </div>
      ) : null}
    </div>
  )
})

/** 过滤掉模型输出的 <function_calls>...</function_calls> 工具调用标签（保险层） */
const FUNCTION_CALLS_RE = /<function_calls>[\s\S]*?<\/function_calls>/g

function stripToolCalls(text: string): string {
  return text.replace(FUNCTION_CALLS_RE, "").trim()
}

const CopyButton = memo(function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      className="inline-flex items-center gap-1 rounded-md px-1.5 py-1 text-[11px] text-slate-400 opacity-0 transition-opacity hover:bg-slate-100 hover:text-slate-700 group-hover:opacity-100"
      title="复制"
      type="button"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text)
          setCopied(true)
          setTimeout(() => setCopied(false), 1500)
        } catch {
          // 忽略剪贴板权限错误
        }
      }}
    >
      {copied ? <Check className="size-3" /> : <Copy className="size-3" />}
      {copied ? "已复制" : "复制"}
    </button>
  )
})

const MessageBubble = memo(function MessageBubble({
  message,
  toolCalls,
  thinking,
}: {
  message: ConversationMessage
  toolCalls?: ToolCallEvent[]
  thinking?: string
}) {
  const isUser = message.role === "user"
  const cleanedContent = isUser ? message.content : stripToolCalls(message.content)
  return (
    <div className={cn("group flex gap-3", isUser ? "justify-end" : "justify-start")}>
      {!isUser ? (
        <div className="flex size-9 shrink-0 items-center justify-center rounded-2xl bg-neutral-100 text-neutral-900">
          <Bot className="size-4" />
        </div>
      ) : null}
      <div
        className={cn(
          "max-w-[80%] rounded-[1.4rem] px-4 py-3 text-sm leading-6",
          isUser
            ? "bg-black text-white shadow-[0_12px_28px_rgba(0,0,0,0.12)]"
            : "border border-[var(--blue-line)] bg-white text-neutral-800",
        )}
      >
        {!isUser && thinking ? (
          <ThinkingPanel thinking={thinking} />
        ) : null}
        {!isUser && toolCalls && toolCalls.length > 0 ? (
          <ToolCallChain toolCalls={toolCalls} />
        ) : null}
        {isUser ? (
          <div className="whitespace-pre-wrap break-words">{cleanedContent}</div>
        ) : (
          <Markdown content={cleanedContent} />
        )}
        <div className="mt-2 flex items-center justify-between gap-2 text-[11px] opacity-60">
          <div className="flex items-center gap-2">
            {isUser ? <User className="size-3" /> : <Cpu className="size-3" />}
            <span>{formatTime(message.timestamp, "刚刚")}</span>
            {message.sources?.length ? <span>· {message.sources.length} sources</span> : null}
          </div>
          {!isUser && cleanedContent ? <CopyButton text={cleanedContent} /> : null}
        </div>
        {message.sources?.length ? <SourceList sources={message.sources} /> : null}
        {message.evidence?.length ? <EvidenceList evidence={message.evidence} /> : null}
        {message.trace ? <TracePanel trace={message.trace} /> : null}
      </div>
    </div>
  )
})

export function ChatPlayground() {
  const queryClient = useQueryClient()
  const { activeConversationId, setActiveConversationId } = useUiStore()
  const sidebarCollapsed = useUiStore((s) => s.sidebarCollapsed)
  const setSidebarCollapsed = useUiStore((s) => s.setSidebarCollapsed)
  const [prompt, setPrompt] = useState("")
  const [draftAnswer, setDraftAnswer] = useState("")
  const [streaming, setStreaming] = useState(false)
  const [streamError, setStreamError] = useState<string | null>(null)
  const [draftToolCalls, setDraftToolCalls] = useState<ToolCallEvent[]>([])
  const [draftThinking, setDraftThinking] = useState("")
  const abortRef = useRef<AbortController | null>(null)
  const messageListRef = useRef<HTMLDivElement>(null)
  const draftTimestampRef = useRef<string>("")

  const conversationsQuery = useQuery({
    queryKey: ["conversations"],
    queryFn: async () => {
      const result = await api.listConversations()
      if (result.error) {
        throw new Error(result.error)
      }
      return result.data
    },
  })

  // 拉取激活对话的历史消息；react-query 缓存即单一数据源，无需本地副本
  const detailQuery = useQuery({
    queryKey: ["conversation", activeConversationId],
    enabled: !!activeConversationId,
    queryFn: async () => {
      const result = await api.getConversation(activeConversationId as string)
      if (result.error || !result.data) {
        throw new Error(result.error || "加载对话历史失败")
      }
      return result.data
    },
    staleTime: 0,
  })

  // 首次进入且未选中任何对话时，自动选第一条
  useEffect(() => {
    if (!activeConversationId && conversationsQuery.data?.conversations?.length) {
      setActiveConversationId(conversationsQuery.data.conversations[0].id)
    }
  }, [activeConversationId, conversationsQuery.data, setActiveConversationId])

  const stickToBottomRef = useRef(true)

  const handleScroll = useCallback(() => {
    if (!messageListRef.current) return
    const el = messageListRef.current
    stickToBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80
  }, [])

  // 新消息或切换对话时，自动滚动到底部（用户上滑时不强制贴底）
  useEffect(() => {
    if (!stickToBottomRef.current || !messageListRef.current) return
    const id = requestAnimationFrame(() => {
      if (messageListRef.current) {
        messageListRef.current.scrollTop = messageListRef.current.scrollHeight
      }
    })
    return () => cancelAnimationFrame(id)
  }, [draftAnswer, draftToolCalls, detailQuery.data, activeConversationId])

  // 组件卸载时中止流式请求，避免 setState on unmounted
  useEffect(() => {
    return () => {
      abortRef.current?.abort()
    }
  }, [])

  const createConversationMutation = useMutation({
    mutationFn: async () => {
      const result = await api.createConversation()
      if (result.error || !result.data) {
        throw new Error(result.error || "创建对话失败")
      }
      return result.data
    },
    onSuccess: async (data) => {
      // 新对话直接写入缓存，避免再次请求详情
      queryClient.setQueryData(["conversation", data.id], {
        id: data.id,
        title: data.title,
        assistant_id: data.assistant_id,
        messages: [],
        created_at: data.created_at,
        updated_at: data.updated_at,
      })
      setActiveConversationId(data.id)
      await queryClient.invalidateQueries({ queryKey: ["conversations"] })
    },
  })

  const deleteConversationMutation = useMutation({
    mutationFn: async (conversationId: string) => {
      const result = await api.deleteConversation(conversationId)
      if (result.error) {
        throw new Error(result.error)
      }
      return result.data
    },
    onSuccess: async (_data, deletedId) => {
      queryClient.removeQueries({ queryKey: ["conversation", deletedId] })
      if (activeConversationId === deletedId) {
        const remaining = conversationsQuery.data?.conversations?.filter((c) => c.id !== deletedId) ?? []
        setActiveConversationId(remaining[0]?.id ?? "")
      }
      await queryClient.invalidateQueries({ queryKey: ["conversations"] })
    },
  })

  // 直接从查询缓存派生消息列表，避免本地副本
  const messages = useMemo<ConversationMessage[]>(() => activeConversationId
    ? (detailQuery.data?.messages ?? []).map((msg: StoredMessage) => ({
        role: msg.role,
        content: msg.content,
        timestamp: msg.timestamp,
        sources: msg.sources,
        recommended_resources: msg.recommended_resources,
        // 阶段三：传递 trace/evidence/tool_calls
        trace: msg.trace,
        evidence: msg.evidence,
        tool_calls: msg.tool_calls,
        thinking: msg.thinking,
      }))
    : [], [activeConversationId, detailQuery.data])
  const isLoadingHistory = !!activeConversationId && detailQuery.isPending

  const sendMutation = useMutation({
    mutationFn: async (question: string) => {
      let conversationId = activeConversationId
      if (!conversationId) {
        const created = await createConversationMutation.mutateAsync()
        conversationId = created.id
      }
      const cid = conversationId as string

      const now = new Date().toISOString()
      const userMessage: StoredMessage = {
        role: "user",
        content: question,
        timestamp: now,
      }

      // 乐观更新：先把用户消息写进详情缓存
      queryClient.setQueryData<ConversationDetail>(["conversation", cid], (old) => {
        if (!old) {
          return {
            id: cid,
            title: "新对话",
            messages: [userMessage],
            created_at: now,
            updated_at: now,
          }
        }
        return {
          ...old,
          messages: [...old.messages, userMessage],
          updated_at: now,
        }
      })

      draftTimestampRef.current = new Date().toISOString()
      setDraftAnswer("")
      setStreamError(null)
      setDraftToolCalls([])
      setDraftThinking("")
      setStreaming(true)

      const controller = new AbortController()
      abortRef.current = controller

      let assistantText = ""
      let thinkingText = ""
      let collectedSources: SourceInfo[] = []
      let collectedError: string | null = null
      let collectedToolCalls: ToolCallEvent[] = []
      // 阶段三：trace 和 evidence 捕获
      let collectedTrace: RagTrace | undefined = undefined
      let collectedEvidence: RetrievalEvidence[] = []

      try {
        await api.sendChatStream(
          { query: question, conversation_id: cid, enable_rag: true },
          (event: ChatStreamEvent) => {
            if (event.error) {
              collectedError = event.error
              setStreamError(event.error)
              return
            }
            if (event.thinking) {
              thinkingText += event.thinking
              setDraftThinking(thinkingText)
            } else if (event.tool_call) {
              collectedToolCalls = [...collectedToolCalls, event.tool_call]
              setDraftToolCalls(collectedToolCalls)
            } else if (event.content) {
              assistantText += event.content
              setDraftAnswer(assistantText)
            } else if (event.done) {
              // 阶段三：complete 事件捕获 sources/trace/evidence
              if (event.sources) {
                collectedSources = event.sources
              }
              if (event.trace) {
                collectedTrace = event.trace
              }
              if (event.evidence) {
                collectedEvidence = event.evidence
              }
            }
          },
          controller.signal,
        )
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") {
          // 用户主动取消，不算错误
        } else {
          collectedError = err instanceof Error ? err.message : String(err)
          setStreamError(collectedError)
        }
      } finally {
        setStreaming(false)
        const finalAssistant: StoredMessage = {
          role: "assistant",
          content:
            assistantText ||
            (collectedError ? `请求失败：${collectedError}` : "（无内容）"),
          timestamp: new Date().toISOString(),
          sources: collectedSources.length ? collectedSources : undefined,
          // 阶段三：持久化 trace/evidence/tool_calls
          trace: collectedTrace,
          evidence: collectedEvidence.length ? collectedEvidence : undefined,
          tool_calls: collectedToolCalls.length ? collectedToolCalls : undefined,
          thinking: thinkingText || undefined,
        }
        // 乐观更新：追加助手回复到详情缓存
        queryClient.setQueryData<ConversationDetail>(["conversation", cid], (old) => {
          if (!old) return old
          return {
            ...old,
            messages: [...old.messages, finalAssistant],
            updated_at: finalAssistant.timestamp,
          }
        })
        setDraftAnswer("")
        setDraftToolCalls([])
        setDraftThinking("")
        abortRef.current = null

        // 持久化到服务端：必须串行，先 user 后 assistant，保证 MongoDB $push 顺序正确
        const userPayload: MessageAddRequest = {
          role: "user",
          content: question,
        }
        const userRes = await api.addMessage(cid, userPayload)
        if (userRes.error) {
          console.warn("持久化用户消息失败：", userRes.error)
        }

        const assistantPayload: MessageAddRequest = {
          role: "assistant",
          content: finalAssistant.content,
          sources: collectedSources.length ? collectedSources : undefined,
          // 阶段三：持久化到服务端
          trace: collectedTrace,
          evidence: collectedEvidence.length ? collectedEvidence : undefined,
          tool_calls: collectedToolCalls.length ? collectedToolCalls : undefined,
          thinking: thinkingText || undefined,
        }
        const assistantRes = await api.addMessage(cid, assistantPayload)
        if (assistantRes.error) {
          console.warn("持久化助手消息失败：", assistantRes.error)
        }

        void queryClient.invalidateQueries({ queryKey: ["conversations"] })
      }
    },
  })

  const handleSend = () => {
    const text = prompt.trim()
    if (!text || sendMutation.isPending || streaming) {
      return
    }
    setPrompt("")
    void sendMutation.mutateAsync(text)
  }

  const handleStop = () => {
    abortRef.current?.abort()
  }

  const handleKey = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault()
      handleSend()
    }
  }

  const errors = [
    conversationsQuery.error instanceof Error && !conversationsQuery.isPending
      ? `会话列表加载失败：${conversationsQuery.error.message}` : null,
    detailQuery.error instanceof Error && !detailQuery.isPending
      ? `历史加载失败：${detailQuery.error.message}` : null,
    sendMutation.error instanceof Error
      ? `消息发送失败：${sendMutation.error.message}` : null,
  ].filter(Boolean) as string[]

  const cleanedDraft = useMemo(() => stripToolCalls(draftAnswer), [draftAnswer])

  // 阶段三：draft 期间也展示 trace/evidence（从 ref 获取，避免重渲染）
  const draftTraceRef = useRef<RagTrace | undefined>(undefined)
  const draftEvidenceRef = useRef<RetrievalEvidence[]>([])

  const allMessages = useMemo<ConversationMessage[]>(
    () => (draftAnswer.length > 0 || draftToolCalls.length > 0
      ? [...messages, {
          role: "assistant" as const,
          content: cleanedDraft,
          timestamp: draftTimestampRef.current,
          trace: draftTraceRef.current,
          evidence: draftEvidenceRef.current.length > 0 ? draftEvidenceRef.current : undefined,
          tool_calls: draftToolCalls.length > 0 ? draftToolCalls : undefined,
          thinking: draftThinking || undefined,
        }]
      : messages),
    [messages, cleanedDraft, draftAnswer, draftToolCalls, draftThinking],
  )

  return (
    <div className="flex h-full">
      {/* 会话侧边栏 - 可折叠 */}
      <aside
        className={cn(
          "flex shrink-0 flex-col border-r border-[var(--blue-line)] bg-white/60 transition-[width] duration-200",
          sidebarCollapsed ? "w-0 overflow-hidden" : "w-64",
        )}
      >
        <div className="flex shrink-0 items-center justify-between border-b border-[var(--blue-line)] px-3 py-2">
          <span className="text-xs font-medium uppercase tracking-[0.16em] text-slate-500">会话</span>
          <button
            className="inline-flex size-6 items-center justify-center rounded text-slate-400 hover:bg-slate-100 hover:text-slate-900"
            onClick={() => setSidebarCollapsed(true)}
            title="收起会话列表"
            type="button"
          >
            <ChevronLeft className="size-4" />
          </button>
        </div>
        <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-hidden p-2">
          <Button
            className="w-full shrink-0"
            onClick={() => createConversationMutation.mutate()}
            variant="secondary"
            disabled={createConversationMutation.isPending}
          >
            <Plus className="size-4" />
            新建对话
          </Button>

          <div className="min-h-0 flex-1 space-y-1 overflow-y-auto pr-1">
            {(conversationsQuery.data?.conversations ?? []).map((conversation) => {
              const isActive = activeConversationId === conversation.id
              return (
                <div
                  key={conversation.id}
                  className={cn(
                    "group flex items-center gap-1 rounded-lg border border-l-2 px-3 py-2 text-left transition-all",
                    isActive
                      ? "border-neutral-200 border-l-black bg-white text-black font-medium"
                      : "border-transparent border-l-transparent text-neutral-700 hover:bg-neutral-50 hover:text-black",
                  )}
                >
                  <button
                    className="flex-1 min-w-0 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-neutral-200"
                    onClick={() => setActiveConversationId(conversation.id)}
                    type="button"
                  >
                    <div className="truncate text-sm font-medium">{conversation.title}</div>
                    <div className="mt-0.5 truncate text-xs opacity-60">
                      {conversation.message_count} 条 · {formatTime(conversation.updated_at)}
                    </div>
                  </button>
                  <button
                    className="inline-flex size-6 shrink-0 items-center justify-center rounded text-slate-400 opacity-0 transition-opacity hover:bg-rose-50 hover:text-rose-600 group-hover:opacity-100"
                    title="删除对话"
                    type="button"
                    onClick={(event) => {
                      event.stopPropagation()
                      if (window.confirm(`确认删除对话「${conversation.title}」？该操作不可恢复。`)) {
                        deleteConversationMutation.mutate(conversation.id)
                      }
                    }}
                  >
                    <Trash2 className="size-3.5" />
                  </button>
                </div>
              )
            })}
            {conversationsQuery.isPending ? (
              <div className="px-3 py-4 text-center text-xs text-slate-400">加载中…</div>
            ) : null}
            {!conversationsQuery.isPending && (conversationsQuery.data?.conversations?.length ?? 0) === 0 ? (
              <div className="px-3 py-4 text-center text-xs text-slate-400">暂无对话</div>
            ) : null}
          </div>
        </div>
      </aside>

      {/* 对话主区 - 全宽自适应，无横幅 */}
      <div className="relative flex min-h-0 flex-1 flex-col overflow-hidden">
        {/* 消息列表区 - 自适应高度，底部留出悬浮输入框空间 */}
        <div
          className="min-h-0 flex-1 space-y-4 overflow-y-auto px-4 pb-4 pt-2"
          ref={messageListRef}
          onScroll={handleScroll}
        >
          {errors.length > 0 ? (
            <div className="rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700">
              {errors.map((error) => (
                <div key={error}>{error}</div>
              ))}
            </div>
          ) : null}

          {isLoadingHistory ? (
            <div className="flex h-full items-center justify-center text-center text-sm text-slate-500">
              正在加载历史消息…
            </div>
          ) : allMessages.length === 0 ? (
            <div className="flex h-full items-center justify-center text-center text-sm text-slate-500">
              发一条消息试试，回答会通过 SSE 流式返回，结束后展示引用来源。
            </div>
          ) : (
            <>
              {allMessages.map((message, index) => {
                const isLast = index === allMessages.length - 1
                const showToolCalls = isLast && streaming && draftToolCalls.length > 0
                return (
                  <MessageBubble
                    key={`${message.role}-${index}-${message.timestamp ?? ""}`}
                    message={message}
                    toolCalls={showToolCalls ? draftToolCalls : message.tool_calls}
                    thinking={showToolCalls ? draftThinking : message.thinking}
                  />
                )
              })}
              {/* TTFB 占位气泡：流式已开始但首个 token 还没到 */}
              {streaming && draftAnswer.length === 0 && draftToolCalls.length === 0 ? (
                <div className="flex gap-3 justify-start">
                  <div className="flex size-9 shrink-0 items-center justify-center rounded-2xl bg-neutral-100 text-neutral-900">
                    <Bot className="size-4" />
                  </div>
                  <div className="max-w-[80%] rounded-[1.4rem] border border-[var(--blue-line)] bg-white px-4 py-3 text-sm leading-6">
                    <div className="flex items-center gap-2 text-neutral-500">
                      <span className="inline-flex gap-1">
                        <span className="size-1.5 animate-bounce rounded-full bg-neutral-400 [animation-delay:-0.3s]" />
                        <span className="size-1.5 animate-bounce rounded-full bg-neutral-400 [animation-delay:-0.15s]" />
                        <span className="size-1.5 animate-bounce rounded-full bg-neutral-400 [animation-delay:0s]" />
                      </span>
                      <span className="text-xs">正在思考…</span>
                    </div>
                  </div>
                </div>
              ) : null}
            </>
          )}
          {streamError && draftAnswer.length === 0 ? (
            <div className="rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700">
              {streamError}
            </div>
          ) : null}
        </div>

        {/* 输入区 - 正常文档流，圆角胶囊 */}
        <div className="shrink-0 px-4 pb-4 pt-2">
          <div className="mx-auto max-w-3xl rounded-[1.5rem] bg-white shadow-[0_2px_24px_rgba(0,0,0,0.06)] ring-1 ring-neutral-200/60 transition-shadow focus-within:ring-neutral-300/80 focus-within:shadow-[0_4px_32px_rgba(0,0,0,0.08)]">
            <div className="flex items-end gap-2 p-2.5">
              <Textarea
                placeholder="输入问题，Enter 发送，Shift+Enter 换行"
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                onKeyDown={handleKey}
                className="min-h-[44px] flex-1 resize-none border-0 bg-transparent px-2 py-2 text-sm shadow-none focus-visible:ring-0"
              />
              {streaming ? (
                <Button
                  className="size-9 shrink-0 rounded-full"
                  onClick={handleStop}
                  variant="outline"
                  size="icon"
                >
                  <Square className="size-4" />
                </Button>
              ) : (
                <Button
                  className="size-9 shrink-0 rounded-full"
                  disabled={!prompt.trim() || sendMutation.isPending}
                  onClick={handleSend}
                  size="icon"
                >
                  <SendHorizontal className="size-4" />
                </Button>
              )}
            </div>
          </div>
          {streaming ? (
            <div className="mt-1.5 flex items-center justify-center gap-2 text-xs text-neutral-400">
              <span className="size-1.5 animate-pulse rounded-full bg-neutral-400" />
              生成中…
            </div>
          ) : null}
        </div>
      </div>
    </div>
  )
}
