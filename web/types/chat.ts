// Types for chat messages, sources, and RAG evaluation

export interface SourceInfo {
  source: string;
  score: number;
  content: string;
  chunk_id?: string;
}

export interface EvidenceItem {
  chunk_id: string;
  score: number;
  content: string;
}

export interface RAGEvaluationMetrics {
  faithfulness?: number;
  answer_relevance?: number;
  context_precision?: number;
  context_recall?: number;
  overall?: number;
  details?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  sources?: SourceInfo[];
  isDeepResearch?: boolean;
  deepResearchAgents?: any[];
  deepResearchStatus?: string;
  evaluation?: RAGEvaluationMetrics;
  citations?: string[];
  thinking?: string;
  isStreaming?: boolean;
  error?: string;
}
