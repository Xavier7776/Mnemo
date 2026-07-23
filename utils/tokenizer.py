"""统一分词器（BM25 索引端 + 查询端共用）

优化点：
1. 英文驼峰切分：WrittenContentCompressor → written, content, compressor
2. 保留代码 token：@CreatedBy、MCP、RRF 等不被过滤
3. 项目专有名词词典：保证 Letta、MemGPT、RRF 等不被错误切分
4. 中英分离分词：中文走 jieba，英文走驼峰切分，避免大小写信息丢失
5. 查询端停用词过滤：去掉"的/是/多少/什么"等高频虚词，避免淹没英文关键词
6. 查询端 token 类型标注：英文 token 在 BM25 查询时获得更高权重

设计原则：索引端和查询端使用同一份分词逻辑，但查询端额外做停用词过滤和权重标注。
"""
import re
from typing import List, Tuple

# 真正不该被驼峰切分的专有名词（整体保留）
# 谨慎添加：只有"切了反而错"的词才加进来
# 例如 Letta 切成 let + ta 就错了；WrittenContentCompressor 切成 written + content + compressor 是对的
PROPER_NOUNS = [
    # 项目名 / 产品名（切了就错）
    "Letta", "MemGPT", "Mnemo", "Qdrant", "Neo4j", "MongoDB",
    "Kafka", "RabbitMQ", "RedisInsight", "PaddleOCR",
    "GPT-Researcher", "GRDIformer", "SpringBoot", "SpringData",
    "LangChain", "FastAPI", "PyTorch", "SentenceTransformer",
    "Claude", "GitHub",
    # 缩写词（不应被切分）
    "MCP", "RRF", "HyDE", "MMR", "BM25", "RAG", "JVM", "JDK", "JRE",
    "G1GC", "CMS", "TF-IDF", "nDCG", "MRR", "API", "JSON", "XML",
    "LSTM", "GRU", "BGE", "LLM",
]

# 中文停用词（虚词、助词、疑问代词等，对 BM25 召回无贡献且高频）
# 这些词在所有文档里都出现，IDF 极低，但会稀释 query 中真正有意义的 token 权重
STOPWORDS = {
    # 助词
    "的", "了", "着", "过", "们", "地", "得",
    # 判断词
    "是", "不是", "有", "没有", "在", "不", "也", "都", "就", "还",
    # 疑问代词/疑问词
    "什么", "怎么", "怎样", "怎么样", "哪些", "哪个", "哪", "多少",
    "为什么", "如何", "何", "何时", "何处",
    # 量词/数词相关
    "个", "些", "种", "条", "次",
    # 介词/连词
    "和", "与", "及", "或", "或者", "以及", "但", "但是", "而", "而且",
    "把", "被", "让", "使", "给", "为", "为了", "对", "关于",
    # 代词
    "这", "那", "这个", "那个", "这些", "那些", "其", "其他", "其它",
    # 时间/方位词（高频但通常无检索意义）
    "时", "中", "上", "下", "里", "外", "前", "后",
    # 单字噪声
    "用", "由", "从", "向", "到", "于", "以", "可", "可以",

    # === 英文停用词（高频虚词，IDF 极低，稀释关键词权重） ===
    # 冠词
    "a", "an", "the",
    # be 动词变形
    "is", "am", "are", "was", "were", "be", "been", "being",
    # 助动词
    "do", "does", "did", "doing", "have", "has", "had", "having",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    # 介词
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as", "into",
    "onto", "upon", "about", "above", "below", "between", "through", "during",
    "before", "after", "over", "under", "again", "further", "than",
    # 连词
    "and", "or", "but", "if", "then", "else", "when", "where", "why", "how",
    "all", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "so", "too", "very",
    # 代词
    "i", "me", "my", "myself", "we", "us", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    # 常见弱意义词
    "there", "here", "out", "up", "down", "off", "away",
}

# 驼峰切分正则（保留原始大小写时使用）
_CAMEL_SPLIT_RE = re.compile(
    r"""
    (?<=[a-z])(?=[A-Z])
    | (?<=[A-Za-z])(?=[0-9])
    | (?<=[0-9])(?=[A-Za-z])
    | [\-_.]
    """,
    re.VERBOSE,
)

# 中英文分离正则
_CN_RE = re.compile(r"[\u4e00-\u9fff]+")
_EN_RE = re.compile(r"[A-Za-z0-9@#+/\-_.]+")

# 噪声过滤
_NOISE_RE = re.compile(r"^[\s\W]+$", re.UNICODE)

# 专有名词集合（用于判断是否跳过驼峰切分）
_PROPER_SET = {w.lower() for w in PROPER_NOUNS}

_jieba_initialized = False


def _init_jieba():
    """初始化 jieba 并加载专有名词词典（只执行一次）"""
    global _jieba_initialized
    if _jieba_initialized:
        return
    try:
        import jieba  # type: ignore
        for word in PROPER_NOUNS:
            jieba.add_word(word, freq=10000)
        _jieba_initialized = True
    except Exception:
        pass


def tokenize(text: str) -> List[str]:
    """统一分词函数（索引端用，保留所有 token 包括停用词）

    索引端必须保留停用词，否则文档端缺失这些词会导致查询时无法匹配。
    停用词过滤只在查询端做。

    Args:
        text: 待分词文本

    Returns:
        分词后的 token 列表（小写）
    """
    if not text:
        return []

    _init_jieba()
    tokens: List[str] = []

    # 1. 提取所有英文片段（含数字、代码符号），保留原始大小写
    en_fragments = _EN_RE.findall(text)
    # 2. 提取中文片段
    cn_fragments = _CN_RE.findall(text)

    # 3. 中文分词（jieba）
    try:
        import jieba  # type: ignore
        for frag in cn_fragments:
            for t in jieba.cut(frag):
                t = t.strip()
                if t and not _NOISE_RE.match(t):
                    tokens.append(t.lower())
    except Exception:
        for frag in cn_fragments:
            tokens.extend(list(frag))

    # 4. 英文分词（专有名词保护 + 驼峰切分）
    for frag in en_fragments:
        frag_orig = frag
        frag_lower = frag.lower()

        if frag_lower in _PROPER_SET:
            tokens.append(frag_lower)
            continue

        sub_parts = _CAMEL_SPLIT_RE.split(frag_orig)
        for sub in sub_parts:
            sub = sub.strip()
            if not sub or _NOISE_RE.match(sub):
                continue
            sub_lower = sub.lower()
            if sub_lower in _PROPER_SET:
                tokens.append(sub_lower)
                continue
            if len(sub) == 1:
                if sub.isalnum():
                    tokens.append(sub_lower)
                continue
            tokens.append(sub_lower)

    return tokens


def tokenize_for_index(text: str) -> List[str]:
    """索引端分词（保留所有 token，包括停用词）"""
    return tokenize(text)


def tokenize_for_query(text: str) -> List[str]:
    """查询端分词（过滤停用词，避免高频虚词淹没关键词）

    返回的 token 用于构造 BM25 查询，停用词会显著稀释英文关键词的权重。
    """
    raw_tokens = tokenize(text)
    return [t for t in raw_tokens if t not in STOPWORDS]


def tokenize_for_query_weighted(text: str) -> List[Tuple[str, float]]:
    """查询端分词 + 权重标注（英文 token 权重更高）

    BM25 查询时，英文 token（代码标识符、专有名词）通常比中文通用词更具区分度。
    返回 [(token, weight), ...]，weight 默认 1.0，英文 token 2.0，专有名词 3.0。

    用法：构造 RediSearch 查询时，高权重 token 用 () 包裹，低权重用 | 连接。
    """
    raw_tokens = tokenize(text)
    weighted: List[Tuple[str, float]] = []

    for t in raw_tokens:
        if t in STOPWORDS:
            continue
        # 判断 token 类型
        is_english = bool(re.match(r"^[a-z0-9]+$", t))
        is_proper = t in _PROPER_SET
        if is_proper:
            weighted.append((t, 3.0))
        elif is_english:
            weighted.append((t, 2.0))
        else:
            weighted.append((t, 1.0))

    return weighted


# 自测
if __name__ == "__main__":
    tests = [
        "WrittenContentCompressor 的相似度阈值是多少？",
        "GPT-Researcher 的 MCP 策略默认值是 disabled、fast 还是 deep？",
        "Spring 怎么自动记录每条数据的创建人和修改时间？",
        "Letta 的 ArchivalMemory 和 Mnemo 的 RAG 检索",
        "GRDIformer 在 BRA 数据集 PL=96 时",
        "Function Calling 中工具定义用什么格式描述？",
        "消息队列 RabbitMQ 和 Kafka 的区别",
        "@CreatedBy 和 @LastModifiedBy 注解",
        "SpringBoot 的 AuditingEntityListener",
    ]
    print("=== 索引端分词（保留停用词）===")
    for q in tests:
        toks = tokenize_for_index(q)
        print(f"  Q: {q}")
        print(f"  T: {toks}")
    print("\n=== 查询端分词（过滤停用词）===")
    for q in tests:
        toks = tokenize_for_query(q)
        print(f"  Q: {q}")
        print(f"  T: {toks}")
    print("\n=== 查询端分词 + 权重 ===")
    for q in tests:
        toks = tokenize_for_query_weighted(q)
        print(f"  Q: {q}")
        print(f"  T: {toks}")

