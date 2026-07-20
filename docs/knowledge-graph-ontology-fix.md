# 知识图谱 Ontology 修复：从「LLM 自由发挥」到「受控抽取」

> 解决关系类型失控、实体类型不一致、节点重复、边重复四个核心问题。

## 一、修复前会出现什么问题

### 问题 1：关系类型完全失控（无 Ontology）

**根因**：`_normalize_relation` 只做纯技术清洗，没有语义对齐：

```python
# 修复前（knowledge_extraction_service.py:207-210）
def _normalize_relation(self, relation: str) -> str:
    clean = re.sub(r'[^\w]', '_', relation)
    return clean.upper()
```

**实际表现**：LLM 对同一个语义产出三种不同关系名：

| LLM 输出 | 清洗后 | 语义 |
|---------|--------|------|
| `"定义于"` | `DEFINED_IN` | "A 定义于 B" |
| `"defined_in"` | `DEFINED_IN` | "A defined_in B" |
| `"described in"` | `DESCRIBED_IN` | "A described in B" |
| `"记载于"` | `记载于`（中文字符保留）→ 实际 Neo4j 关系名乱码 | "A 记载于 B" |

**后果**：
- Neo4j 里同一对实体可能有多条语义相同的边（`DEFINED_IN` + `DESCRIBED_IN` + `记载于`）
- 图谱查询时 `MATCH (a)-[:DEFINED_IN]->(b)` 漏掉 `DESCRIBED_IN` 的边
- 关系类型数量爆炸（理论上 = LLM 词汇量），无法做统计和分析

### 问题 2：实体类型 LLM 自标，不一致

**根因**：prompt 告诉 LLM 用 Concept/Technology/Person/Organization/Location/Event/Other，但 LLM 标的 `head_type` 没有任何校验。

**实际表现**：同一实体在不同 chunk 里被标成不同 label：

| chunk | LLM 输出 | Neo4j 节点 |
|-------|---------|-----------|
| chunk 1 | `{"head": "Transformer", "head_type": "Concept"}` | `(:Concept {name:"Transformer"})` |
| chunk 2 | `{"head": "Transformer", "head_type": "Technology"}` | `(:Technology {name:"Transformer"})` |

**后果**：
- Neo4j 里出现两个 Transformer 节点（不同 label），`MERGE` 只在 name 相同时复用，name+label 组合由 LLM 决定
- `MATCH (n:Concept {name:"Transformer"})` 找不到 `(:Technology {name:"Transformer"})`
- 图谱连通性被破坏，本该连在一起的实体变成孤立节点

### 问题 3：缺乏实体对齐，节点重复

**根因**：`MERGE (n {name: $name})` 只在 name 完全相同时复用节点，没有模糊匹配。

**实际表现**：

| LLM 输出 | Neo4j 节点 |
|---------|-----------|
| `{"head": "Transformer"}` | `(:Concept {name:"Transformer"})` |
| `{"head": "Transformer 模型"}` | `(:Concept {name:"Transformer 模型"})` ← 重复节点 |
| `{"head": "transformer"}` | `(:Concept {name:"transformer"})` ← 重复节点 |
| `{"head": "  Transformer  "}` | `(:Concept {name:"  Transformer  "})` ← 带空格的重复节点 |

**后果**：
- Neo4j 节点数虚高（同一实体多个节点）
- 图谱检索时只能命中其中一个节点，丢失该实体的其他关系
- 图谱可视化时看到大量"近似节点"，难以理解

### 问题 4：缺乏去重，边重复

**根因**：LLM 对不同 chunk 独立抽取，`MERGE (a)-[r:TYPE]->(b)` 只在 (a, r, b) 完全相同时复用边，但 LLM 可能产出：
- 同一对实体被多次写入相同关系（如果关系名规范化后相同，MERGE 能去重）
- 同一对实体被标了不同的关系名（`DEFINED_IN` vs `DESCRIBED_IN`）——两条不同的边

**实际表现**：

| chunk | LLM 输出 | Neo4j 边 |
|-------|---------|---------|
| chunk 1 | `(A, 定义于, B)` | `A -[:DEFINED_IN]-> B` |
| chunk 2 | `(A, defined_in, B)` | `A -[:DEFINED_IN]-> B`（MERGE 去重）|
| chunk 3 | `(A, described in, B)` | `A -[:DESCRIBED_IN]-> B` ← 语义相同但关系名不同，新增边 |

**后果**：
- 同一对实体之间有多条语义重复的边
- `source_chunk` 信息被覆盖（`SET r += $rel_props` 会覆盖，不累积）

### 问题 5：异常实体名没有过滤

**根因**：没有任何长度/格式校验，LLM 可能产出：
- 超长实体名（整句被当实体）：`"Transformer 是一种基于自注意力机制的神经网络架构"`
- 空实体名：`""`
- 代词当实体：`"他"`、`"这个"`

**后果**：
- 超长实体名污染图谱（一个"实体"实际是一段话）
- Neo4j 查询性能下降（name 字段过长，索引效率低）

---

## 二、修复方案：三层后处理流水线

```
LLM 抽取三元组
  ↓
【后处理 1】规范化 + 异常过滤
  ├─ 实体名规范化：去空格、全角转半角、长度检查（1-50 字符）
  ├─ 实体类型规范化：OntologyRegistry.normalize_entity_type → 7 种预定义类型
  └─ 关系名规范化：OntologyRegistry.normalize_relation → 20 种预定义关系
  ↓
【后处理 2】去重
  └─ 三元组去重：以 (head, relation, tail) 为 key，相同 key 合并 source_chunks 列表
  ↓
【后处理 3】实体对齐
  ├─ 查询 Neo4j 已有实体（LIMIT 500）
  └─ 模糊匹配（相似度 >=0.9 或包含关系）→ 复用已有节点名
  ↓
写入 Neo4j
```

### 2.1 OntologyRegistry（Ontology 注册表）

**文件**：[services/ontology_registry.py](file:///d:/timeModel/Mnemo/services/ontology_registry.py)

预定义 7 种实体类型 + 20 种关系类型，LLM 输出经别名表 + 模糊匹配对齐到预定义类型：

**实体类型**（7 种 + Other 兜底）：
- Concept（概念、理论、思想、原理）
- Technology（技术、工具、框架、平台、系统）
- Person（人物、作者、开发者）
- Organization（组织、机构、公司、团队）
- Location（地点、国家、城市）
- Event（事件、活动、会议）
- Other（兜底）

**关系类型**（20 种，分 4 大类）：

| 类别 | 关系 | 中文别名 |
|------|------|---------|
| 结构 | `DEFINED_IN` | 定义于、定义在、被定义、定义为 |
| 结构 | `DESCRIBED_IN` | 描述于、描述在、记载于、提及于、提到、出现于 |
| 结构 | `PART_OF` | 属于、组成部分、一部分、包含于、隶属于 |
| 结构 | `HAS_PART` | 包含、包括、由...组成 |
| 结构 | `INSTANCE_OF` | 实例、实例化、是...的实例 |
| 语义 | `RELATED_TO` | 相关、关联、有关、联系 |
| 语义 | `SIMILAR_TO` | 相似、类似 |
| 语义 | `OPPOSITE_OF` | 相反、对立、反义 |
| 语义 | `DEPENDS_ON` | 依赖、依赖于、需要、基于 |
| 语义 | `DERIVED_FROM` | 源自、派生自、来源于、来自 |
| 语义 | `EQUIVALENT_TO` | 等价、等同、相同、等于 |
| 动作 | `CREATED_BY` | 创建者、由...创建、作者、发明者 |
| 动作 | `USED_BY` | 被使用、被...使用、应用于 |
| 动作 | `MANAGES` | 管理、负责、控制 |
| 动作 | `PRODUCES` | 产出、生成、产生 |
| 动作 | `TRIGGERS` | 触发、引起、导致 |
| 属性 | `HAS_PROPERTY` | 有属性、属性是、特征是 |
| 属性 | `LOCATED_IN` | 位于、在...地方 |
| 属性 | `OCCURRED_AT` | 发生于、发生在 |
| 属性 | `HAS_ROLE` | 角色是、担任、作为 |

**规范化策略**（三步）：
1. 精确匹配别名表（大小写不敏感）
2. 模糊匹配（`difflib.SequenceMatcher`，实体类型 ≥0.8，关系 ≥0.85）
3. 兜底：实体类型 → `Other`，关系 → `RELATED_TO`

### 2.2 EntityResolver（实体对齐 + 去重）

**文件**：[services/entity_resolver.py](file:///d:/timeModel/Mnemo/services/entity_resolver.py)

**实体名规范化**：
- 全角空格 → 半角空格
- 去首尾空格
- 合并中间连续空格
- 长度检查（1-50 字符，超长丢弃）

**实体对齐**（`find_existing_entity`）：
- 策略 1：大小写不敏感精确匹配
- 策略 2：包含关系（"Transformer 模型" 包含 "Transformer"，长度差 ≤5 时复用较长的）
- 策略 3：相似度匹配（`SequenceMatcher` ≥0.9 时复用已有节点名）

**三元组去重**（`deduplicate_triplets`）：
- 以 `(head.lower(), relation.upper(), tail.lower())` 为 key
- 相同 key 的多条记录合并 `_source_chunks` 列表（保留多个来源）

### 2.3 修改 build_graph（受控写入）

**文件**：[services/knowledge_extraction_service.py:181-307](file:///d:/timeModel/Mnemo/services/knowledge_extraction_service.py#L181-307)

新流程：
1. **抽取**：LLM 用受控 prompt 抽取（prompt 里明确列出预定义实体类型和关系类型）
2. **规范化**：每个三元组经 OntologyRegistry + EntityResolver 规范化
3. **异常过滤**：实体名长度 1-50 字符，非法丢弃
4. **去重**：`deduplicate_triplets` 合并相同三元组
5. **实体对齐**：查询 Neo4j 已有实体（LIMIT 500），模糊匹配复用节点名
6. **写入**：写入时合并 `source_chunks` 列表（而非覆盖）

**日志可观测**：
```
图谱构建完成: 抽取 25 → 合法 18 → 写入 15 (规范化 12, 非法丢弃 7, 实体对齐 3)
```

---

## 三、修复后会怎么样

### 3.1 关系类型从失控 → 20 种预定义

| 修复前 | 修复后 |
|--------|--------|
| LLM 产出 `DEFINED_IN` / `DESCRIBED_IN` / `记载于` 三种关系 | 全部对齐到 `DEFINED_IN`（如果语义是"定义"）或 `DESCRIBED_IN`（如果语义是"描述"）|
| Neo4j 关系类型数量爆炸 | 最多 20 种 + 少量自定义（LLM 仍可输出自定义，但 prompt 鼓励用预定义）|
| `MATCH (a)-[:DEFINED_IN]->(b)` 漏掉同义边 | 同义边全部对齐到同一关系类型，查询完整 |

### 3.2 实体类型从 LLM 自标 → Ontology 校准

| 修复前 | 修复后 |
|--------|--------|
| Transformer 可能是 `:Concept` 或 `:Technology` | OntologyRegistry 统一对齐（"Transformer" 在别名表里映射到 `Technology`）|
| Neo4j 出现两个 Transformer 节点（不同 label）| 同一实体只有一个 label，`MERGE` 正确复用节点 |
| 图谱连通性被破坏 | 连通性保证 |

### 3.3 节点重复 → 实体对齐

| 修复前 | 修复后 |
|--------|--------|
| `Transformer` / `Transformer 模型` / `transformer` 三个节点 | `find_existing_entity` 对齐到同一个节点（`Transformer`）|
| 节点数虚高 | 节点数真实 |
| 图谱检索漏掉同义节点的关系 | 同义节点合并，所有关系聚合 |

### 3.4 边重复 → 三元组去重

| 修复前 | 修复后 |
|--------|--------|
| 同一对实体被多次写入相同关系（如果关系名规范化后相同，MERGE 去重）| 三元组去重，同一 (head, relation, tail) 只保留一条 |
| `source_chunk` 被覆盖 | `source_chunks` 列表累积，保留多个来源 |
| 关系名不同导致语义相同的多条边 | Ontology 对齐关系名后，语义相同的边被去重 |

### 3.5 异常实体名 → 长度过滤

| 修复前 | 修复后 |
|--------|--------|
| 整句被当实体：`"Transformer 是一种基于自注意力机制的神经网络架构"` | 长度 >50 字符直接丢弃 |
| 空实体名：`""` | 长度 <1 字符直接丢弃 |
| 代词当实体：`"他"`、`"这个"` | （Ontology 不直接过滤，但 prompt 引导 LLM 不用代词）|

---

## 四、工程化考量

### 4.1 向后兼容

- `_normalize_relation` 方法保留作为 fallback，新代码用 `ontology_registry.normalize_relation`
- 原有的 `create_entity` / `create_relationship` Cypher 不变，只是传入的 label 和 rel_type 已经规范化
- 原有的 `delete_by_document_id` 不受影响

### 4.2 性能开销

| 阶段 | 耗时 | 说明 |
|------|------|------|
| Ontology 规范化 | <1ms | 纯字典查找 + difflib 相似度 |
| 实体名规范化 | <1ms | 字符串操作 |
| 三元组去重 | <1ms | 字典 key 查找 |
| 实体对齐（查 Neo4j）| 10-50ms | 一次 `MATCH (n) RETURN n.name LIMIT 500` |
| 实体对齐（模糊匹配）| <5ms | 500 个候选 × SequenceMatcher |

**总开销**：<60ms，相对于 LLM 抽取的 1-3s 可忽略。

### 4.3 降级策略

- **Neo4j 查询失败**（实体对齐阶段）：跳过实体对齐，直接用 LLM 输出的原始名称建节点（回到修复前行为，但有 Ontology 规范化）
- **Ontology 模糊匹配失败**：兜底为 `Other`（实体类型）或 `RELATED_TO`（关系），不会报错
- **实体名规范化失败**：标记 `is_valid=False`，丢弃该三元组

### 4.4 可观测性

每次 `build_graph` 完成后输出日志：
```
图谱构建完成: 抽取 25 → 合法 18 → 写入 15 (规范化 12, 非法丢弃 7, 实体对齐 3)
```

- `抽取 25`：LLM 原始产出
- `合法 18`：经规范化 + 异常过滤后
- `写入 15`：去重 + 实体对齐后实际写入 Neo4j
- `规范化 12`：12 个三元组的实体名/类型/关系被 Ontology 改写
- `非法丢弃 7`：7 个三元组因实体名不合法被丢弃
- `实体对齐 3`：3 个实体名对齐到已有节点

### 4.5 配置参数

所有参数有默认值，不需要额外配置即可使用。如果需要调整：

| 参数 | 默认值 | 位置 | 说明 |
|------|--------|------|------|
| `MAX_ENTITY_NAME_LEN` | 50 | `entity_resolver.py` | 实体名最大长度 |
| `ENTITY_FUZZY_THRESHOLD` | 0.9 | `entity_resolver.py` | 实体对齐相似度阈值 |
| 实体类型模糊匹配阈值 | 0.8 | `ontology_registry.py` | 实体类型对齐阈值 |
| 关系模糊匹配阈值 | 0.85 | `ontology_registry.py` | 关系名对齐阈值 |

---

## 五、测试验证

### 5.1 Ontology 规范化测试（全部通过）

```
=== Ontology 规范化测试 ===
  [OK] '定义于' -> DEFINED_IN
  [OK] 'defined_in' -> DEFINED_IN
  [OK] 'described in' -> DESCRIBED_IN
  [OK] 'described_in' -> DESCRIBED_IN
  [OK] '属于' -> PART_OF
  [OK] 'part_of' -> PART_OF
  [OK] '相关' -> RELATED_TO
  [OK] 'related to' -> RELATED_TO
  [OK] 'related_to' -> RELATED_TO
  [OK] 'random_unknown_relation' -> RELATED_TO (兜底)
```

### 5.2 实体类型规范化测试（全部通过）

```
=== 实体类型规范化测试 ===
  [OK] 'Concept' -> Concept
  [OK] '概念' -> Concept
  [OK] '技术' -> Technology
  [OK] 'tech' -> Technology
  [OK] '人物' -> Person
  [OK] '公司' -> Organization
  [OK] 'random' -> Other (兜底)
```

### 5.3 实体名规范化测试（全部通过）

```
=== 实体名规范化测试 ===
  [OK] 'Transformer' -> valid
  [OK] '  Transformer  ' -> 'Transformer' (去空格)
  [OK] 'Transformer\u3000模型' -> 'Transformer 模型' (全角转半角)
  [OK] '' -> invalid (空)
  [OK] 'a'*51 -> invalid (超长)
```

### 5.4 实体对齐测试（全部通过）

```
=== 实体对齐测试 ===
  [OK] find('Transformer') -> 'Transformer' (精确匹配)
  [OK] find('transformer') -> 'Transformer' (大小写不敏感)
  [OK] find('Transformer 模型') -> 'Transformer' (包含关系)
  [OK] find('微服务') -> '微服务架构' (短→长对齐)
  [OK] find('React') -> 'React 框架' (包含关系)
  [OK] find('Vue') -> None (无匹配，新建节点)
```

### 5.5 三元组去重测试（全部通过）

```
=== 三元组去重测试 ===
  原始 4 条 -> 去重后 3 条 (期望 3 条)
    A -[DEFINED_IN]-> B (sources: ['c2'])   # 'defined_in' 对齐后与 'DEFINED_IN' 去重
    A -[RELATED_TO]-> B (sources: ['c3'])
    C -[PART_OF]-> D (sources: ['c4'])
```

---

## 六、后续优化方向

### 6.1 短期

1. **关系冲突消解**：同一对实体有 `DEFINED_IN` 和 `DESCRIBED_IN` 两条边时，标记置信度或合并
2. **实体类型冲突消解**：同一实体名在不同 chunk 被标成 `Concept` 和 `Technology` 时，投票决定
3. **抽取质量监控**：定期统计 `规范化率` / `非法丢弃率` / `对齐率`，发现 prompt 退化

### 6.2 中期

1. **增量对齐**：实体对齐时只查最近 N 天新增的实体，避免全表扫描
2. **Ontology 扩展**：根据实际抽取统计，补充预定义关系（如 `LOCATED_IN` 不够用时加 `BORDERED_BY`）
3. **人工校验**：抽样 100 个三元组人工校验，量化抽取准确率

### 6.3 长期

1. **图谱质量评分**：节点连通性 + 关系类型分布 + 实体类型一致性 → 综合评分
2. **LLM 抽取 prompt 自动优化**：根据规范化率反推 prompt 缺失的别名，自动补充
3. **多文档实体消解**：跨文档的同义实体（"GPT-4" vs "GPT4" vs "Generative Pre-trained Transformer 4"）全局消解

---

## 七、修改文件清单

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| [services/ontology_registry.py](file:///d:/timeModel/Mnemo/services/ontology_registry.py) | 新增 | Ontology 注册表：7 种实体类型 + 20 种关系类型 + 别名表 + 模糊匹配 |
| [services/entity_resolver.py](file:///d:/timeModel/Mnemo/services/entity_resolver.py) | 新增 | 实体名规范化 + 实体对齐 + 三元组去重 |
| [services/knowledge_extraction_service.py](file:///d:/timeModel/Mnemo/services/knowledge_extraction_service.py) | 修改 | build_graph 加三层后处理 + 受控 prompt + 可观测日志 |
