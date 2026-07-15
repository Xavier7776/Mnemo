/**
 * 前端调用的后端接口（路径已对齐 routers/ 实际注册）：
 *  POST /api/chat                          → 发送消息，流式返回（SSE），不持久化消息
 *  POST /api/documents/upload              → 上传文件（multipart/form-data）
 *  GET  /api/documents                      → 文档列表
 *  GET  /api/chat/conversations             → 对话列表
 *  POST /api/chat/conversations             → 创建新对话
 *  GET  /api/chat/conversations/{id}        → 对话详情（含历史消息，用于查看历史）
 *  POST /api/chat/conversations/{id}/messages → 持久化一条消息（用户/助手）
 *  POST /api/retrieval                      → 检索知识库
 *  GET  /api/health                          → 健康检查
 */

export interface HealthServiceStatus {
  status: string
  connected: boolean
  error?: string
}

export interface HealthSystemInfo {
  cpu_percent: number
  memory_percent: number
  memory_available_mb: number
  memory_total_mb: number
}

export interface HealthResponse {
  status: string
  version: string
  services: Record<string, HealthServiceStatus>
  system?: HealthSystemInfo | null
}

export interface ConversationSummary {
  id: string
  user_id?: string | null
  title: string
  message_count: number
  assistant_id?: string | null
  created_at?: string | null
  updated_at?: string | null
}

export interface ConversationListResponse {
  conversations: ConversationSummary[]
  total: number
  skip: number
  limit: number
}

export interface ConversationCreateResponse {
  id: string
  title: string
  assistant_id?: string | null
  created_at?: string | null
  updated_at?: string | null
}

/** 服务端历史消息（来自 GET /api/chat/conversations/{id}） */
export interface StoredMessage {
  message_id?: string
  role: "user" | "assistant"
  content: string
  timestamp?: string | null
  sources?: SourceInfo[]
  evidence?: RetrievalEvidence[]
  citation_warnings?: string[]
  recommended_resources?: RecommendedResource[]
  /** 阶段三：PAR 循环轨迹 */
  trace?: RagTrace
  /** 阶段三：工具调用历史 */
  tool_calls?: ToolCallEvent[]
  /** LLM 思考链（reasoning_content） */
  thinking?: string
}

export interface ConversationDetail {
  id: string
  user_id?: string | null
  title: string
  assistant_id?: string | null
  messages: StoredMessage[]
  created_at?: string | null
  updated_at?: string | null
}

/** POST /api/chat/conversations/{id}/messages 请求体（对齐后端 MessageAdd） */
export interface MessageAddRequest {
  role: "user" | "assistant"
  content: string
  sources?: SourceInfo[]
  evidence?: RetrievalEvidence[]
  citation_warnings?: string[]
  recommended_resources?: RecommendedResource[]
  /** 阶段三：PAR 循环轨迹 */
  trace?: RagTrace
  /** 阶段三：工具调用历史 */
  tool_calls?: ToolCallEvent[]
  /** LLM 思考链（reasoning_content） */
  thinking?: string
}

export interface MessageAddResponse {
  success: boolean
  message?: string
  timestamp?: string
}

/** POST /api/chat 请求体（对齐后端 ChatRequest） */
export interface ChatSendRequest {
  query: string
  conversation_id?: string
  assistant_id?: string
  knowledge_space_ids?: string[]
  enable_rag?: boolean
  mode?: "normal" | "network"
}

export interface SourceInfo {
  title?: string
  content?: string
  document_id?: string
  chunk_id?: string
  score?: number
  source?: string
}

export interface RecommendedResource {
  title?: string
  url?: string
  description?: string
}

/** 阶段三：单次检索观察（Observe 阶段产物） */
export interface TraceObservation {
  round?: number
  tool?: string
  query?: string
  evidence_ids?: string[]
  evidence_count?: number
  top_score?: number
  verified_count?: number
  summary?: string
}

/** 阶段三：PAR 循环 trace（Plan-Act-Observe-Reflect 完整轨迹） */
export interface RagTrace {
  observations?: TraceObservation[]
  total_retrievals?: number
  total_evidence?: number
  verified_evidence?: number
  [key: string]: unknown
}

export interface ToolCallInfo {
  name: string
  params: Record<string, unknown>
  success: boolean
  result?: unknown
}

export interface ToolCallEvent {
  round: number
  tools: ToolCallInfo[]
}

export interface ChatStreamEvent {
  content?: string
  done?: boolean
  error?: string
  sources?: SourceInfo[]
  recommended_resources?: RecommendedResource[]
  tool_call?: ToolCallEvent
  /** 阶段三：complete 事件携带的 PAR 循环轨迹 */
  trace?: RagTrace
  /** 阶段三：complete 事件携带的 chunk 级证据列表 */
  evidence?: RetrievalEvidence[]
  /** 阶段三：complete 事件携带的查询分解计划 */
  query_plan?: RetrievalQueryPlan
  /** 阶段三：引用校验告警 */
  citation_warnings?: string[]
  /** LLM 思考链 chunk（推理模型的 reasoning_content） */
  thinking?: string
}

/** 前端本地维护的对话消息（与服务端 StoredMessage 对齐，用于流式期间即时渲染） */
export interface ConversationMessage {
  role: "user" | "assistant"
  content: string
  timestamp?: string | null
  sources?: SourceInfo[]
  recommended_resources?: RecommendedResource[]
  /** 阶段三：PAR 循环轨迹 */
  trace?: RagTrace
  /** 阶段三：chunk 级证据列表 */
  evidence?: RetrievalEvidence[]
  /** 阶段三：工具调用历史 */
  tool_calls?: ToolCallEvent[]
  /** LLM 思考链（reasoning_content） */
  thinking?: string
}

export interface DocumentItem {
  id: string
  title: string
  file_type: string
  file_size: number
  created_at: string
  status: string
  progress_percentage?: number | null
  current_stage?: string | null
  stage_details?: string | null
  knowledge_space_id?: string | null
}

export interface DocumentListResponse {
  documents: DocumentItem[]
  total: number
}

export interface DocumentUploadResponse {
  message?: string
  document_id: string
  filename?: string
  file_size?: number
  status?: string
}

export interface DocumentUploadBatchItem {
  filename: string
  status: "success" | "failed" | "duplicated"
  document_id?: string
  file_size?: number
  error?: string
}

export interface DocumentUploadBatchResponse {
  message: string
  total: number
  success: number
  failed: number
  duplicated: number
  results: DocumentUploadBatchItem[]
}

export interface RetrievalSearchRequest {
  query: string
  top_k?: number
  knowledge_space_ids?: string[]
  conversation_id?: string
}

export interface RetrievalEvidence {
  chunk_id?: string
  text?: string
  score?: number
  document_id?: string
  /** 阶段三：文档标题 */
  document_title?: string
  /** 阶段三：检索类型（agentic_rag 等） */
  retrieval_type?: string
  /** 阶段三：是否经过 EvidenceVerifier 验证为相关 */
  verified?: boolean
  /** 阶段三：验证后的相关性分数（0-1） */
  relevance_score?: number
  /** 阶段三：第几轮检索获取 */
  retrieved_at_round?: number
  /** 阶段三：文档章节路径 */
  section_path?: string[]
}

export interface RetrievalQueryPlan {
  sub_queries?: string[]
  intent?: string
  [key: string]: unknown
}

export interface RetrievalTrace {
  [key: string]: unknown
}

export interface RetrievalSearchResponse {
  context: string
  sources: SourceInfo[]
  evidence?: RetrievalEvidence[]
  query_plan?: RetrievalQueryPlan
  trace?: RetrievalTrace
  retrieval_count: number
  recommended_resources?: RecommendedResource[]
}

/** 知识空间（GET /api/knowledge-spaces 返回项） */
export interface KnowledgeSpace {
  id?: string
  name: string
  description?: string
  collection_name?: string
  document_count?: number
  created_at?: string | null
}

export interface KnowledgeSpaceListResponse {
  spaces: KnowledgeSpace[]
  total: number
}

// —— MCP 管理 ——

export interface McpServerConfig {
  transport: "stdio" | "sse"
  command?: string | null
  args?: string[]
  env?: Record<string, string>
  url?: string | null
  enabled: boolean
  timeout: number
}

export interface McpServerStatus {
  name: string
  transport: string
  enabled: boolean
  connected: boolean
  tool_count: number
  tools: string[]
  timeout: number
}

export interface McpStatus {
  initialized: boolean
  enabled: boolean
  total_servers: number
  connected_servers: number
  total_tools: number
  compact_mode: boolean
  servers: McpServerStatus[]
}

export interface McpOperationResult {
  success: boolean
  message: string
  tool_count?: number | null
}

export interface McpToolDetail {
  name: string
  description: string
  parameters: Record<string, unknown>
}

export interface McpToolsListResponse {
  success: boolean
  servers: Record<string, McpToolDetail[]>
}
