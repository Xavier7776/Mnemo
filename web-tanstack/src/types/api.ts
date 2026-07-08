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
}

/** 前端本地维护的对话消息（与服务端 StoredMessage 对齐，用于流式期间即时渲染） */
export interface ConversationMessage {
  role: "user" | "assistant"
  content: string
  timestamp?: string | null
  sources?: SourceInfo[]
  recommended_resources?: RecommendedResource[]
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
