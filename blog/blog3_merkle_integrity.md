# 密码学视角下的RAG完整性保护：Merkle Tree实战

> 本文是 RAG-Shield 系列博客的第三篇，从密码学角度深入探讨如何使用 Merkle Tree 保护知识库完整性。

## 引言

前两篇文章介绍了RAG系统的威胁全景和投毒检测技术。但检测只是防御的一部分——我们还需要确保知识库本身没有被篡改。

本文将探讨一个核心问题：**如何用密码学方法证明知识库的完整性？**

我们将实现：
- **Merkle Tree**：O(log n) 复杂度验证任意文档
- **Vector Commitment**：保护嵌入向量不被篡改
- **Audit Log**：不可抵赖的操作日志

## 为什么需要密码学完整性保护？

传统的校验和（如 MD5 哈希整个数据库）存在明显问题：

```
┌──────────────────────────────────────────────────────────────┐
│ 传统方案的问题                                                │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  知识库 (1TB)                                                │
│  ┌─────┬─────┬─────┬─────┬─────┐                           │
│  │Doc1 │Doc2 │Doc3 │ ... │DocN │                           │
│  └─────┴─────┴─────┴─────┴─────┘                           │
│         │                                                    │
│         ▼                                                    │
│    Hash(全部) = SHA256(Doc1||Doc2||...||DocN)               │
│         │                                                    │
│         ▼                                                    │
│  验证时需要重新计算全部 → O(n) 时间，不可扩展！              │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**问题：**
1. 验证单个文档需要读取全部数据
2. 无法提供「某个文档属于知识库」的简洁证明
3. 增量更新需要重算整个哈希

## Merkle Tree：对数复杂度验证

### 核心思想

Merkle Tree 是一种二叉哈希树：
- **叶节点**：存储文档的哈希
- **内部节点**：存储子节点哈希的哈希
- **根节点**：代表整个数据集的唯一摘要

```
                        ┌────────────────────┐
                        │     Root Hash      │
                        │   H(H12 || H34)    │
                        └────────────────────┘
                               /    \
                              /      \
               ┌────────────┐          ┌────────────┐
               │    H12     │          │    H34     │
               │ H(H1||H2)  │          │ H(H3||H4)  │
               └────────────┘          └────────────┘
                  /     \                  /     \
                 /       \                /       \
          ┌─────┐   ┌─────┐        ┌─────┐   ┌─────┐
          │ H1  │   │ H2  │        │ H3  │   │ H4  │
          │H(D1)│   │H(D2)│        │H(D3)│   │H(D4)│
          └─────┘   └─────┘        └─────┘   └─────┘
             ▲         ▲              ▲         ▲
             │         │              │         │
          ┌─────┐   ┌─────┐        ┌─────┐   ┌─────┐
          │Doc1 │   │Doc2 │        │Doc3 │   │Doc4 │
          └─────┘   └─────┘        └─────┘   └─────┘
```

### 安全属性

1. **碰撞抗性**：找到两个不同数据集有相同根哈希是计算不可行的
2. **绑定性**：一旦承诺，无法在不改变根哈希的情况下修改数据
3. **域分离**：叶节点和内部节点使用不同前缀，防止第二原像攻击

### 实现代码

```python
class MerkleTree:
    """Merkle Tree for knowledge base integrity verification."""

    def _hash_leaf(self, data: str) -> str:
        """哈希叶节点 - 使用域分离"""
        # "00:" 前缀标识叶节点
        prefixed = f"00:{data}"
        return hashlib.sha256(prefixed.encode()).hexdigest()

    def _hash_pair(self, left: str, right: str) -> str:
        """哈希内部节点 - 使用域分离"""
        # "01:" 前缀标识内部节点
        combined = f"01:{left}:{right}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def build(self, documents: List[str]) -> str:
        """构建 Merkle Tree"""
        # 创建叶节点
        leaves = [MerkleNode(hash=self._hash_leaf(doc), data=doc)
                  for doc in documents]

        # 自底向上构建
        current_level = leaves
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                parent_hash = self._hash_pair(left.hash, right.hash)
                parent = MerkleNode(hash=parent_hash, left=left, right=right)
                next_level.append(parent)
            current_level = next_level

        self.root = current_level[0]
        return self.root.hash
```

### Merkle Proof：O(log n) 验证

要证明某个文档属于知识库，只需提供从叶到根的路径上的兄弟哈希：

```
验证 Doc2 的 Merkle Proof:

                      Root (已知)
                        │
              ┌─────────┴─────────┐
              │                   │
            H12 ◄─ 计算           H34 ◄─ 提供
              │
        ┌─────┴─────┐
        │           │
       H1 ◄─ 提供   H2 ◄─ 计算
                    │
                  Doc2 (待验证)

Proof = [H1, H34]  // 只需 2 个哈希（log₂4 = 2）

验证步骤：
1. H2' = Hash("00:" || Doc2)
2. H12' = Hash("01:" || H1 || H2')
3. Root' = Hash("01:" || H12' || H34)
4. 检查 Root' == Root ✓
```

### 代码：生成和验证证明

```python
def generate_proof(self, index: int) -> MerkleProof:
    """生成 Merkle 证明"""
    siblings = []
    current_index = index

    for level in tree_levels[:-1]:
        sibling_idx = current_index ^ 1  # XOR 得到兄弟索引
        if current_index % 2 == 0:
            siblings.append((level[sibling_idx].hash, "right"))
        else:
            siblings.append((level[sibling_idx].hash, "left"))
        current_index //= 2

    return MerkleProof(
        leaf_hash=self.leaves[index].hash,
        siblings=siblings,
        root_hash=self.root.hash
    )

def verify_proof(self, document: str, proof: MerkleProof) -> bool:
    """验证 Merkle 证明"""
    current_hash = self._hash_leaf(document)

    if current_hash != proof.leaf_hash:
        return False

    for sibling_hash, direction in proof.siblings:
        if direction == "left":
            current_hash = self._hash_pair(sibling_hash, current_hash)
        else:
            current_hash = self._hash_pair(current_hash, sibling_hash)

    return current_hash == proof.root_hash
```

### 性能分析

| 操作 | 传统哈希 | Merkle Tree |
|------|---------|-------------|
| 构建 | O(n) | O(n) |
| 验证单文档 | O(n) | **O(log n)** |
| 证明大小 | O(n) | **O(log n)** |
| 更新单文档 | O(n) | **O(log n)** |

对于 100 万文档的知识库：
- 传统方案：验证需读取全部 100 万文档
- Merkle Tree：仅需 20 个哈希值（log₂(10⁶) ≈ 20）

## Vector Commitment：嵌入向量保护

### 威胁模型

RAG 攻击不仅可以篡改文档内容，还可以**直接操纵嵌入向量**：

```
正常流程：
  Document → Embedding Model → [0.12, -0.34, 0.56, ...]
                                      │
                                      ▼
                               Vector Database
                                      │
                                      ▼
                               Similarity Search

攻击方式：
  攻击者直接注入精心构造的向量 [x₁, x₂, ..., xₙ]
  使其与目标查询高度相似，劫持检索结果
```

### 解决方案：Vector Commitment

对每个文档的嵌入向量创建密码学承诺：

```python
class HashBasedCommitment:
    """基于 Merkle 结构的向量承诺"""

    def commit(self, vector: np.ndarray) -> Commitment:
        """对向量创建承诺"""
        # 将向量转为叶哈希
        leaf_hashes = []
        for i, val in enumerate(vector):
            # 包含位置以实现位置绑定
            leaf_data = f"leaf:{i}:{val:.8e}"
            leaf_hashes.append(self._hash(leaf_data))

        # 构建 Merkle 根
        root_hash = self._build_merkle_root(leaf_hashes)

        return Commitment(
            value=root_hash,
            metadata={"dimension": len(vector)}
        )

    def verify(self, vector: np.ndarray, commitment: Commitment) -> bool:
        """验证向量与承诺是否匹配"""
        if len(vector) != commitment.metadata["dimension"]:
            return False
        recomputed = self.commit(vector)
        return recomputed.value == commitment.value
```

### 实战：检测嵌入篡改

```python
# 创建承诺存储
store = CommitmentStore()

# 添加文档时存储承诺
embedding = embedding_model.encode("Paris is the capital of France")
store.store_commitment("doc_001", embedding)

# 检索时验证
if not store.verify_embedding("doc_001", retrieved_embedding):
    raise IntegrityError("Embedding has been tampered!")
```

## Audit Log：不可抵赖的操作日志

### 设计目标

1. **只追加**：条目一旦写入不可删除或修改
2. **链式完整性**：每条记录链接前一条的哈希
3. **可验证**：任何篡改都可被检测

### 数据结构

```
┌────────────────────────────────────────────────────────────────┐
│                       Audit Log (Hash Chain)                   │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    │
│  │ Entry 0 │───▶│ Entry 1 │───▶│ Entry 2 │───▶│ Entry 3 │    │
│  ├─────────┤    ├─────────┤    ├─────────┤    ├─────────┤    │
│  │prev: 00 │    │prev: H0 │    │prev: H1 │    │prev: H2 │    │
│  │op: ADD  │    │op: QUERY│    │op: CHECK│    │op: ADD  │    │
│  │data:... │    │data:... │    │data:... │    │data:... │    │
│  │hash: H0 │    │hash: H1 │    │hash: H2 │    │hash: H3 │    │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘    │
│                                                                │
│  H0 = SHA256(seq || ts || op || data || "00...00")            │
│  H1 = SHA256(seq || ts || op || data || H0)                   │
│  ...                                                           │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 实现

```python
@dataclass
class AuditEntry:
    sequence_number: int
    timestamp: float
    operation_type: OperationType
    operation_data: Dict[str, Any]
    previous_hash: str
    entry_hash: str = ""

    def _compute_hash(self) -> str:
        data = {
            "sequence_number": self.sequence_number,
            "timestamp": self.timestamp,
            "operation_type": self.operation_type.value,
            "operation_data": self.operation_data,
            "previous_hash": self.previous_hash,
        }
        serialized = json.dumps(data, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()


class AuditLog:
    GENESIS_HASH = "0" * 64  # 创世块

    def log_operation(self, op_type: OperationType, data: dict) -> AuditEntry:
        entry = AuditEntry(
            sequence_number=len(self.entries),
            timestamp=time.time(),
            operation_type=op_type,
            operation_data=data,
            previous_hash=self._last_hash,
        )
        entry.entry_hash = entry._compute_hash()
        self.entries.append(entry)
        self._last_hash = entry.entry_hash
        return entry

    def verify_chain(self) -> bool:
        """验证整个链的完整性"""
        for i, entry in enumerate(self.entries):
            # 验证自身哈希
            if entry.entry_hash != entry._compute_hash():
                return False
            # 验证链接
            expected_prev = self.GENESIS_HASH if i == 0 else self.entries[i-1].entry_hash
            if entry.previous_hash != expected_prev:
                return False
        return True
```

### 篡改检测演示

```python
log = AuditLog()
log.log_document_add("doc_001", "hash_abc")
log.log_query("What is AI?", "query_hash")

print(log.verify_chain())  # True

# 篡改尝试
log.entries[0].operation_data["doc_id"] = "TAMPERED"
print(log.verify_chain())  # False
print(log.find_tampering())  # [0]
```

## IntegrityGuard：统一保护层

将所有组件整合为高级 API：

```python
class IntegrityGuard:
    """RAG 系统完整性守护"""

    def protect_knowledge_base(self, kb: KnowledgeBase) -> str:
        """保护整个知识库"""
        # 构建 Merkle Tree
        contents = [doc.content for doc in kb.documents]
        merkle_root = self.merkle_tree.build(contents)

        # 存储向量承诺
        for doc in kb.documents:
            self.commitment_store.store_commitment(
                doc.doc_id,
                np.array(doc.embedding)
            )

        # 记录审计日志
        self.audit_log.log_operation(
            OperationType.SYSTEM_EVENT,
            {"event": "knowledge_base_protected", "root": merkle_root}
        )

        return merkle_root

    def verify_document(self, doc_id: str, doc: Document) -> IntegrityResult:
        """验证单个文档的完整性"""
        issues = []

        # 验证内容
        proof = self.merkle_tree.generate_proof(self._protected_docs[doc_id])
        if not self.merkle_tree.verify_proof(doc.content, proof):
            issues.append("content_mismatch")

        # 验证嵌入
        if not self.commitment_store.verify_embedding(doc_id, doc.embedding):
            issues.append("embedding_mismatch")

        return IntegrityResult(
            is_valid=len(issues) == 0,
            status=IntegrityStatus.CONTENT_TAMPERED if issues else IntegrityStatus.VALID
        )
```

## 实验结果

使用 RAG-Shield 测试完整性保护效果：

```bash
$ python examples/integrity_demo.py

======================================================================
RAG-Shield: Cryptographic Integrity Protection Demo
======================================================================

1. MERKLE TREE - Knowledge Base Integrity
   Built tree with 5 documents
   Root hash: 7ccd46366605db5598dadc487374c0ae...
   Proof size: 3 sibling hashes
   Tampered documents detected at indices: [1]

2. VECTOR COMMITMENT - Embedding Integrity
   Original embedding verification: VALID
   Subtly tampered (delta=0.001): INVALID

3. AUDIT LOG - Tamper-Evident Logging
   Chain valid after tampering: False
   Tampered entries: [1]

4. INTEGRITY GUARD - Unified Protection
   Tampered doc status: content_tampered
   Detection: Document content has been tampered with
```

## 总结

本文介绍了 RAG-Shield 的密码学完整性保护模块：

| 组件 | 功能 | 复杂度 |
|------|------|--------|
| Merkle Tree | 文档完整性 | O(log n) 验证 |
| Vector Commitment | 嵌入保护 | O(d) 验证 |
| Audit Log | 操作追溯 | O(1) 追加 |
| IntegrityGuard | 统一接口 | 组合所有保护 |

**关键洞察**：
1. 检测和完整性保护是互补的——检测发现攻击，完整性证明攻击未发生
2. 密码学方法提供可验证的安全保证，而非启发式判断
3. O(log n) 复杂度使保护可扩展到大规模知识库

下一篇文章将介绍如何构建端到端的 RAG 安全防御体系。

---

## 参考资料

1. Merkle, R. C. (1987). A Digital Signature Based on a Conventional Encryption Function
2. Catalano, D., & Fiore, D. (2013). Vector Commitments and Their Applications
3. RAG-Shield 项目：https://github.com/[your-repo]/RAG-Shield

## 代码

完整代码见：`src/ragshield/integrity/`

```
ragshield/integrity/
├── __init__.py
├── merkle_tree.py    # Merkle Tree 实现
├── vector_commit.py  # Vector Commitment
├── audit_log.py      # 审计日志
└── guard.py          # 统一接口
```
