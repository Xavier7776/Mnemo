"""查询分析服务 - 判断是否需要检索上下文"""
from typing import Dict, Any, Optional
from utils.logger import logger
from utils.llm_client import get_openai_client
import os
import json
import re


class QueryAnalyzer:
    """查询分析器 - 判断用户问题是否需要检索知识库"""
    
    def __init__(self):
        self._client = None
        self._analysis_model = None

    @property
    def client(self):
        return get_openai_client()

    @property
    def analysis_model(self):
        if self._analysis_model is None:
            self._analysis_model = os.getenv("ANALYSIS_MODEL", os.getenv("LLM_MODEL", "mimo-v2.5"))
        return self._analysis_model

    def _build_analysis_prompt(self, query: str) -> str:
        """构建分析提示词"""
        return f"""你是一个查询分析助手。请分析以下用户问题，判断是否需要从知识库中检索相关信息来回答。

判断标准：
1. **需要检索**：问题涉及具体的传感器知识、技术细节、文档内容、课程内容等需要从知识库获取的信息
2. **不需要检索**：问题是一般性对话、问候、系统操作询问、数学计算、编程问题等不需要知识库信息的问题

用户问题：{query}

请只回答 JSON 格式：
{{
  "need_retrieval": true/false,
  "reason": "简要说明判断理由"
}}"""
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """
        分析查询，判断是否需要检索上下文
        """
        try:
            prompt = self._build_analysis_prompt(query)
            
            response = self.client.chat.completions.create(
                model=self.analysis_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=100,
                timeout=10.0
            )
            response_text = response.choices[0].message.content or ""
            
            try:
                json_match = re.search(r'\{[^{}]*"need_retrieval"[^{}]*\}', response_text, re.DOTALL)
                if json_match:
                    analysis_result = json.loads(json_match.group())
                else:
                    analysis_result = json.loads(response_text)
                
                need_retrieval = analysis_result.get("need_retrieval", True)
                reason = analysis_result.get("reason", "未提供理由")
                
                logger.info(f"查询分析结果 - 查询: {query[:50]}..., 需要检索: {need_retrieval}, 理由: {reason}")
                
                return {
                    "need_retrieval": bool(need_retrieval),
                    "reason": reason,
                    "confidence": "high"
                }
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                logger.warning(f"解析分析结果失败: {response_text[:100]}, 错误: {str(e)}")
                return self._fallback_analysis(query)
                
        except Exception as e:
            logger.warning(f"查询分析请求失败: {str(e)}，使用后备方案")
            return self._fallback_analysis(query)
    
    def _fallback_analysis(self, query: str) -> Dict[str, Any]:
        """后备分析方案：基于关键词匹配"""
        query_lower = query.lower()
        
        no_retrieval_keywords = [
            "你好", "hello", "hi", "谢谢", "thanks", "再见", "bye",
            "帮助", "help", "怎么用", "如何使用", "功能", "操作",
            "计算", "算", "等于", "等于多少",
            "代码", "编程", "写程序", "写代码",
            "时间", "现在几点", "日期",
            "天气", "今天天气"
        ]
        
        need_retrieval_keywords = [
            "传感器", "sensor", "原理", "工作原理", "应用", "分类",
            "电阻", "电容", "电感", "热电", "光电", "磁电",
            "温度", "压力", "位移", "速度", "加速度",
            "文档", "资料", "教材", "课本", "课程",
            "什么是", "如何选择", "如何安装", "如何调试",
            "特点", "特性", "参数", "规格", "选型"
        ]
        
        has_retrieval_keyword = any(keyword in query_lower for keyword in need_retrieval_keywords)
        has_no_retrieval_keyword = any(keyword in query_lower for keyword in no_retrieval_keywords)
        
        if has_retrieval_keyword:
            need_retrieval = True
            reason = "问题包含传感器相关知识关键词"
        elif has_no_retrieval_keyword:
            need_retrieval = False
            reason = "问题属于一般性对话或系统操作"
        else:
            need_retrieval = True
            reason = "未匹配到明确关键词，默认需要检索"
        
        logger.info(f"后备分析结果 - 查询: {query[:50]}..., 需要检索: {need_retrieval}, 理由: {reason}")
        
        return {
            "need_retrieval": need_retrieval,
            "reason": reason,
            "confidence": "medium"
        }


# 全局查询分析器实例
query_analyzer = QueryAnalyzer()
