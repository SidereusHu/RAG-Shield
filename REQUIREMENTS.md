# RAG-Shield 需求分析文档

## 项目概述

RAG-Shield 是一个针对检索增强生成（RAG）系统的安全防护框架，旨在解决 RAG 系统面临的知识库投毒、数据泄露、检索完整性等安全威胁。

**背景**：
- USENIX Security 2025: PoisonedRAG 攻击可达 90% 成功率
- ACM Web 2025: RAGForensics 首次实现投毒溯源
- 当前 RAG 系统缺乏系统性安全防护方案

**与 PromptGuard 的关系**：
- PromptGuard 关注单个 Agent 的输入输出安全
- RAG-Shield 关注 RAG 系统的检索过程和知识库安全
- 两者互补，共同构建完整的 Agent 安全体系

---

## 核心威胁分析

### 1. 知识库投毒攻击（Knowledge Poisoning）

**攻击方式**：
- 注入恶意文档到知识库
- 篡改现有文档内容
- 构造对抗性文本干扰检索

**威胁场景**：
```
攻击者 → 注入恶意文档 → 知识库
                          ↓
用户查询 → 检索 → 恶意文档被检索 → LLM生成错误答案
```

**真实影响**：
- 误导用户决策
- 泄露敏感信息
- 破坏系统可信度

### 2. 检索隐私泄露（Retrieval Privacy）

**威胁**：
- 服务器可以知道用户查询了什么
- 查询历史可能泄露用户意图
- 检索结果可能包含敏感信息

### 3. 检索完整性破坏（Retrieval Integrity）

**威胁**：
- 检索结果被中间人篡改
- 知识库版本不一致
- 无法验证检索结果真实性

### 4. 数据泄露风险（Data Leakage）

**威胁**：
- 通过精心构造的查询提取训练数据
- RAG 系统可能泄露知识库内容
- 跨文档推理导致的隐私泄露

---

## 功能需求

### Phase 1: 核心架构与投毒检测

#### 1.1 RAG 系统抽象
```python
class RAGSystem:
    """RAG 系统基类"""
    - 知识库管理
    - 文档索引
    - 相似度检索
    - LLM 生成接口
```

#### 1.2 投毒检测器
```python
class PoisonDetector:
    """投毒文档检测"""
    - 困惑度异常检测（Perplexity-based）
    - 相似度聚类检测（Similarity-based）
    - 语义一致性检测（Semantic-based）
```

#### 1.3 检测指标
- **准确率**：正确识别投毒文档的比例
- **召回率**：检测出的投毒文档占所有投毒文档的比例
- **误报率**：正常文档被误判为投毒的比例

### Phase 2: 密码学完整性保护

#### 2.1 Merkle Tree 知识库验证
```python
class MerkleTreeVerifier:
    """使用 Merkle Tree 验证知识库完整性"""
    - build_tree(documents) → root_hash
    - generate_proof(doc_id) → proof
    - verify_document(doc, proof, root) → bool
    - detect_tampering() → tampered_docs
```

**技术亮点**：
- 任何文档被篡改，根哈希会改变
- O(log n) 验证效率
- 支持增量更新

#### 2.2 向量承诺方案
```python
class VectorCommitment:
    """对 embedding 向量的密码学承诺"""
    - commit(embedding) → commitment
    - verify(embedding, commitment) → bool
    - batch_verify(embeddings, commitments) → bool
```

**应用场景**：
- 检索服务返回的向量未被篡改
- 验证文档 embedding 的真实性

#### 2.3 可验证审计日志
```python
class AuditLog:
    """防篡改的审计日志"""
    - log_retrieval(query, results, timestamp)
    - verify_log_chain() → bool
    - detect_log_tampering() → tampering_events
```

### Phase 3: 隐私保护检索

#### 3.1 差分隐私检索
```python
class DifferentialPrivacyRetrieval:
    """差分隐私保护的检索"""
    - private_retrieve(query, epsilon) → results
    - add_noise_to_scores(scores, epsilon)
    - privacy_budget_tracking()
```

#### 3.2 Private Information Retrieval (PIR)
```python
class PIRRetrieval:
    """服务器无法得知用户查询内容"""
    - setup(database) → server_state
    - query(query_embedding) → encrypted_query
    - answer(encrypted_query) → encrypted_result
    - decode(encrypted_result) → documents
```

**技术难点**：
- 计算开销大
- 需要同态加密或多服务器协议
- 实用性与安全性平衡

### Phase 4: 攻击溯源与防御

#### 4.1 投毒溯源
```python
class PoisonForensics:
    """定位导致攻击的投毒文档"""
    - trace_attack(query, malicious_output) → poisoned_docs
    - analyze_influence(doc, query) → influence_score
    - generate_forensics_report() → report
```

#### 4.2 主动防御机制
```python
class ActiveDefense:
    """主动防御策略"""
    - expand_retrieval_scope(k → k+n)  # 检索更多文档稀释投毒
    - ensemble_filtering()              # 多模型投票过滤
    - dynamic_threshold_adjustment()    # 动态阈值调整
```

### Phase 5: 红队测试工具

#### 5.1 投毒攻击模拟
```python
class PoisoningAttack:
    """模拟投毒攻击"""
    - craft_poisoned_document(target_query, target_answer)
    - inject_to_knowledge_base(poisoned_docs)
    - evaluate_attack_success_rate() → metrics
```

#### 5.2 攻击类型
- **直接投毒**：注入明确的恶意答案
- **对抗性投毒**：构造高相似度的对抗文本
- **链式投毒**：多个文档协同误导
- **隐蔽投毒**：长期潜伏，特定条件触发

---

## 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                    RAG-Shield Framework                  │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │   Detection │  │  Integrity   │  │    Privacy     │ │
│  │   Module    │  │   Module     │  │    Module      │ │
│  ├─────────────┤  ├──────────────┤  ├────────────────┤ │
│  │ Perplexity  │  │ Merkle Tree  │  │ DP Retrieval   │ │
│  │ Similarity  │  │ Vector Commit│  │ PIR (optional) │ │
│  │ ML Detector │  │ Audit Log    │  │ Query Privacy  │ │
│  └─────────────┘  └──────────────┘  └────────────────┘ │
│                                                           │
│  ┌─────────────┐  ┌──────────────┐                      │
│  │  Forensics  │  │   Red Team   │                      │
│  │   Module    │  │    Module    │                      │
│  ├─────────────┤  ├──────────────┤                      │
│  │ Trace Attack│  │ Poisoning    │                      │
│  │ Influence   │  │ Adversarial  │                      │
│  │ Analysis    │  │ Evaluation   │                      │
│  └─────────────┘  └──────────────┘                      │
│                                                           │
├─────────────────────────────────────────────────────────┤
│                    RAG System Core                       │
│  Knowledge Base │ Retriever │ Embedder │ LLM Generator  │
└─────────────────────────────────────────────────────────┘
```

---

## 技术栈

### 核心依赖
- **LangChain / LlamaIndex**: RAG 框架基础
- **FAISS / ChromaDB**: 向量检索
- **Sentence-Transformers**: 文本 embedding
- **OpenAI / Ollama**: LLM 后端

### 密码学库
- **hashlib**: Merkle Tree 实现
- **cryptography**: 哈希、承诺方案
- **PySEAL (optional)**: 同态加密（PIR）

### ML/检测
- **scikit-learn**: 异常检测
- **numpy/scipy**: 统计分析
- **transformers**: 困惑度计算

### 继承自 PromptGuard
- `agentsec.core`: LLM 抽象接口
- `agentsec.guardrails`: 输入输出过滤（可选）
- `agentsec.crypto.signer`: HMAC 签名工具

---

## 项目目录结构

```
RAG-Shield/
├── README.md
├── REQUIREMENTS.md            # 本文档
├── setup.py
├── requirements.txt
├── pyproject.toml
│
├── src/
│   └── ragshield/
│       ├── __init__.py
│       ├── core/              # Phase 1: 核心架构
│       │   ├── rag_system.py
│       │   ├── knowledge_base.py
│       │   ├── retriever.py
│       │   └── embedder.py
│       ├── detection/         # Phase 1: 投毒检测
│       │   ├── perplexity.py
│       │   ├── similarity.py
│       │   ├── semantic.py
│       │   └── ml_detector.py
│       ├── integrity/         # Phase 2: 完整性保护
│       │   ├── merkle_tree.py
│       │   ├── vector_commit.py
│       │   └── audit_log.py
│       ├── privacy/           # Phase 3: 隐私保护
│       │   ├── dp_retrieval.py
│       │   └── pir.py (optional)
│       ├── forensics/         # Phase 4: 攻击溯源
│       │   ├── tracer.py
│       │   └── influence.py
│       └── redteam/           # Phase 5: 红队测试
│           ├── poisoning.py
│           └── adversarial.py
│
├── tests/
│   ├── test_core.py
│   ├── test_detection.py
│   ├── test_integrity.py
│   ├── test_privacy.py
│   ├── test_forensics.py
│   └── test_redteam.py
│
├── examples/
│   ├── basic_rag.py
│   ├── detection_demo.py
│   ├── integrity_demo.py
│   ├── attack_demo.py
│   └── full_defense_demo.py
│
├── docs/
│   ├── zh/
│   │   ├── index.md
│   │   ├── quickstart.md
│   │   ├── detection.md
│   │   ├── integrity.md
│   │   ├── privacy.md
│   │   └── redteam.md
│   └── en/
│       └── (same structure)
│
├── blog/
│   ├── blog1_rag_threat_landscape.md    # Phase 1 完成后
│   ├── blog2_poison_detection.md        # Phase 1 完成后
│   ├── blog3_merkle_integrity.md        # Phase 2 完成后
│   ├── blog4_privacy_retrieval.md       # Phase 3 完成后
│   └── blog5_forensics.md               # Phase 4 完成后
│
└── benchmarks/
    ├── poison_benchmark.py
    └── defense_benchmark.py
```

---

## 开发计划

### Phase 1: 核心架构与投毒检测（2-3周）

**目标**：
- 实现基础 RAG 系统
- 实现 3 种投毒检测方法
- 复现 PoisonedRAG 攻击
- 评估检测效果

**交付物**：
- 可运行的 RAG 系统
- 投毒检测器（准确率 > 80%）
- 测试覆盖率 > 85%
- Blog 1: 《RAG 系统安全威胁全景》
- Blog 2: 《投毒攻击检测技术实战》

### Phase 2: 密码学完整性保护（2周）

**目标**：
- 实现 Merkle Tree 知识库验证
- 实现向量承诺方案
- 实现可验证审计日志
- 与 PromptGuard 集成

**交付物**：
- Merkle Tree 验证器（O(log n) 效率）
- 向量承诺方案实现
- 完整性监控系统
- Blog 3: 《Merkle Tree 在 RAG 完整性验证中的应用》

### Phase 3: 隐私保护检索（2周）

**目标**：
- 实现差分隐私检索
- 探索 PIR 可行性（optional）
- 隐私预算追踪
- 效用-隐私权衡分析

**交付物**：
- DP 检索实现（ε-DP 保证）
- 隐私预算管理器
- 效用评估报告
- Blog 4: 《差分隐私在 RAG 检索中的应用》

### Phase 4: 攻击溯源与防御（1-2周）

**目标**：
- 实现投毒溯源算法
- 实现主动防御策略
- 端到端防御系统

**交付物**：
- 溯源系统（定位投毒文档）
- 主动防御策略
- 完整防护 pipeline
- Blog 5: 《RAG 投毒攻击溯源技术》

### Phase 5: 红队工具与评估（1周）

**目标**：
- 实现多种投毒攻击
- 建立评估基准
- 撰写技术白皮书

**交付物**：
- 红队攻击工具集
- 标准评估基准
- 技术白皮书
- Blog 6: 《RAG-Shield 技术白皮书》

---

## 关键创新点

### 1. 密码学完整性验证（博士级）
- **Merkle Tree 知识库**：首次将 Merkle Tree 应用于 RAG 知识库
- **向量承诺方案**：验证 embedding 向量完整性
- **形式化安全性证明**：提供数学证明

### 2. 多层次防御体系
- **检测层**：ML + 统计 + 语义
- **验证层**：密码学完整性
- **溯源层**：攻击来源定位
- **防御层**：主动对抗策略

### 3. 实用性与安全性平衡
- 低延迟检测（< 100ms）
- 高准确率（> 90%）
- 可扩展到百万级知识库

---

## 评估指标

### 防御效果
- **攻击成功率**：< 10%（vs. 无防御 90%）
- **误报率**：< 5%
- **检测延迟**：< 100ms

### 性能指标
- **检索延迟**：增加 < 20%
- **存储开销**：增加 < 30%
- **计算开销**：可接受范围

### 学术价值
- 对标 USENIX/ACM 顶会标准
- 形式化安全性证明
- 开源可复现

---

## 风险与挑战

### 技术风险
1. **PIR 性能**：同态加密开销可能太大（可作为 optional 特性）
2. **检测准确率**：对抗性投毒难以检测
3. **可扩展性**：大规模知识库的 Merkle Tree 更新

### 解决方案
1. PIR 作为可选高级特性，主推 DP
2. 结合多种检测方法，集成学习
3. 增量更新算法 + 缓存优化

---

## 预期成果

### 技术成果
- 完整的 RAG 安全防护框架
- 开源项目（GitHub stars > 100）
- 5-6 篇技术博客

### 职业价值
- 完全匹配岗位"智能体安全架构"要求
- 体现密码学博士水平（Merkle, Commitment, DP）
- 展示工程能力（完整项目 + 测试 + 文档）
- 证明研究能力（对标顶会论文）

### 面试亮点
1. **岗位匹配**：覆盖"RAG 安全"、"数据泄露"、"异常检测"
2. **技术深度**：密码学 + ML + 工程
3. **实战经验**：复现顶会攻击 + 创新防御
4. **完整项目**：从需求 → 设计 → 实现 → 评估

---

## 下一步行动

1. **Review 本需求文档**：确认技术路线
2. **环境搭建**：安装依赖、初始化项目
3. **Phase 1 启动**：实现核心 RAG 系统
4. **每个 Phase 完成后写 Blog**

准备好开始了吗？🚀
