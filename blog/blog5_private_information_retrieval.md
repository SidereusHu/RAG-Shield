# Blog 5: Private Information Retrieval - Cryptographic Query Privacy for RAG Systems

*February 15, 2025*

In our previous blog, we explored differential privacy (DP) for protecting retrieval patterns. While DP adds controlled noise to provide plausible deniability, it doesn't provide **cryptographic guarantees**. Today, we introduce **Private Information Retrieval (PIR)** - a cryptographic primitive that ensures the server learns absolutely nothing about which documents you're retrieving.

## The Query Privacy Problem

Consider a scenario where you're using a RAG system hosted by a third party:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Standard RAG Retrieval                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   User                          Server                           │
│    │                              │                              │
│    │  "Query: cancer treatment"   │                              │
│    │─────────────────────────────>│                              │
│    │                              │  Server sees:                │
│    │                              │  - Your query                │
│    │                              │  - Which docs match          │
│    │                              │  - Your access patterns      │
│    │                              │                              │
│    │  Returns: Documents 3, 7, 12 │                              │
│    │<─────────────────────────────│                              │
│                                                                  │
│   Problem: Server learns sensitive information about user!       │
└─────────────────────────────────────────────────────────────────┘
```

This reveals potentially sensitive information:
- What topics interest you
- What documents you're accessing
- Your research patterns over time

## What is Private Information Retrieval?

PIR allows a client to retrieve an item from a database without the server learning which item was retrieved.

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIR-Protected Retrieval                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   User                          Server                           │
│    │                              │                              │
│    │  Encrypted/Shared Query      │                              │
│    │─────────────────────────────>│                              │
│    │                              │  Server sees:                │
│    │                              │  - Encrypted query           │
│    │                              │  - NOTHING about intent!     │
│    │                              │                              │
│    │  Encrypted/Shared Response   │                              │
│    │<─────────────────────────────│                              │
│    │                              │                              │
│    │  User decrypts locally       │                              │
│    │  Gets: Document 7            │                              │
│                                                                  │
│   Result: User gets document, server learns nothing!             │
└─────────────────────────────────────────────────────────────────┘
```

## PIR Protocol Taxonomy

There are two main families of PIR protocols, each with different security guarantees:

```
┌─────────────────────────────────────────────────────────────────┐
│                      PIR Protocol Types                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────────────────────┐   ┌─────────────────────────────┐ │
│   │    Single-Server PIR    │   │     Multi-Server PIR        │ │
│   ├─────────────────────────┤   ├─────────────────────────────┤ │
│   │                         │   │                             │ │
│   │  Security:              │   │  Security:                  │ │
│   │  Computational          │   │  Information-Theoretic      │ │
│   │  (based on crypto       │   │  (unconditional, even       │ │
│   │   assumptions)          │   │   against infinite power)   │ │
│   │                         │   │                             │ │
│   │  Servers: 1             │   │  Servers: 2+                │ │
│   │                         │   │  (non-colluding)            │ │
│   │  Technique:             │   │                             │ │
│   │  Homomorphic            │   │  Technique:                 │ │
│   │  Encryption             │   │  Secret Sharing             │ │
│   │                         │   │                             │ │
│   │  Pro: No trust          │   │  Pro: Very fast,            │ │
│   │       assumptions       │   │       perfect secrecy       │ │
│   │                         │   │                             │ │
│   │  Con: Slow (crypto      │   │  Con: Requires multiple     │ │
│   │       operations)       │   │       independent servers   │ │
│   │                         │   │                             │ │
│   └─────────────────────────┘   └─────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Single-Server PIR: Homomorphic Encryption Approach

Single-server PIR uses **additively homomorphic encryption** (like Paillier) to hide the query:

```
┌─────────────────────────────────────────────────────────────────┐
│               Single-Server PIR Protocol                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Database: [D₀, D₁, D₂, D₃, D₄]    Target: D₂                   │
│                                                                  │
│  Step 1: Client creates selection vector                        │
│  ┌───┬───┬───┬───┬───┐                                          │
│  │ 0 │ 0 │ 1 │ 0 │ 0 │  ← "1" at position 2                     │
│  └───┴───┴───┴───┴───┘                                          │
│                                                                  │
│  Step 2: Client encrypts each bit                               │
│  ┌──────┬──────┬──────┬──────┬──────┐                           │
│  │Enc(0)│Enc(0)│Enc(1)│Enc(0)│Enc(0)│                           │
│  └──────┴──────┴──────┴──────┴──────┘                           │
│     ↓      ↓      ↓      ↓      ↓                               │
│  Step 3: Server computes (using homomorphic properties)         │
│                                                                  │
│     ∑ Enc(sᵢ) × Dᵢ = Enc(∑ sᵢ × Dᵢ) = Enc(D₂)                   │
│                                                                  │
│  Step 4: Client decrypts to get D₂                              │
│                                                                  │
│  Key insight: Server only sees encrypted query,                 │
│  cannot distinguish which index was selected!                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Homomorphic Encryption Properties

```
┌─────────────────────────────────────────────────────────────────┐
│            Additive Homomorphic Encryption                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Core Property:                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Enc(a) ⊕ Enc(b) = Enc(a + b)                            │    │
│  │                                                          │    │
│  │  Operations on ciphertexts → Operations on plaintexts   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Example (Paillier-like):                                        │
│                                                                  │
│    Enc(5) × Enc(3) mod n² = Enc(5 + 3) = Enc(8)                 │
│                                                                  │
│    Enc(7)³ mod n² = Enc(7 × 3) = Enc(21)                        │
│                                                                  │
│  This enables server-side computation without decryption!       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Multi-Server PIR: Secret Sharing Approach

Multi-server PIR achieves **information-theoretic security** through secret sharing:

```
┌─────────────────────────────────────────────────────────────────┐
│               Multi-Server PIR with XOR Sharing                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Database: [D₀, D₁, D₂, D₃]    Target: D₁                       │
│                                                                  │
│  Selection vector: [0, 1, 0, 0]                                 │
│                                                                  │
│  Step 1: Split into random shares                               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Original:  [0, 1, 0, 0]                                  │  │
│  │       ↓                                                    │  │
│  │  Share 1:   [1, 0, 1, 1]  (random)                        │  │
│  │  Share 2:   [1, 1, 1, 1]  = Original XOR Share 1          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Step 2: Send shares to different servers                       │
│                                                                  │
│    Server 1                    Server 2                         │
│  ┌─────────────┐            ┌─────────────┐                     │
│  │ Receives:   │            │ Receives:   │                     │
│  │ [1,0,1,1]   │            │ [1,1,1,1]   │                     │
│  │             │            │             │                     │
│  │ Computes:   │            │ Computes:   │                     │
│  │ D₀⊕D₂⊕D₃   │            │ D₀⊕D₁⊕D₂⊕D₃│                     │
│  │ = R₁        │            │ = R₂        │                     │
│  └─────────────┘            └─────────────┘                     │
│         │                          │                            │
│         └──────────┬───────────────┘                            │
│                    ↓                                            │
│  Step 3: Client combines: R₁ ⊕ R₂ = D₁                          │
│                                                                  │
│  Security: Each server sees random-looking query,               │
│  learns nothing unless they collude!                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Why XOR Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    XOR Cancellation Magic                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Given: Selection vector s = [0, 1, 0, 0]                       │
│         Shares: s₁ ⊕ s₂ = s                                     │
│                                                                  │
│  Server 1 computes: R₁ = ⊕(Dᵢ where s₁[i]=1)                    │
│  Server 2 computes: R₂ = ⊕(Dᵢ where s₂[i]=1)                    │
│                                                                  │
│  Client combines:                                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  R₁ ⊕ R₂ = ⊕(all Dᵢ where s₁[i] ≠ s₂[i])                │    │
│  │          = ⊕(all Dᵢ where s[i] = 1)                      │    │
│  │          = D₁                                            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Items at positions where s₁[i] = s₂[i] cancel out!            │
│  Only D₁ (where original s[1]=1) remains.                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Threshold PIR: Fault Tolerance

For robustness, we can use **Shamir Secret Sharing** to tolerate server failures:

```
┌─────────────────────────────────────────────────────────────────┐
│                (t,n)-Threshold Secret Sharing                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Secret: S                                                       │
│                                                                  │
│  Step 1: Construct random polynomial of degree t-1              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  f(x) = S + a₁x + a₂x² + ... + aₜ₋₁xᵗ⁻¹                  │    │
│  │                                                          │    │
│  │  Note: f(0) = S (the secret)                             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Step 2: Generate n shares as points on the polynomial          │
│                                                                  │
│                     f(x)                                         │
│                      │    ╱                                      │
│                      │  ╱                                        │
│              (3,f(3))●╱                                          │
│                    ╱│                                            │
│            (2,f(2))● │                                           │
│                  ╱   │                                           │
│          (1,f(1))●   │                                           │
│                ╱     │                                           │
│            S ●───────┼─────→ x                                   │
│              0   1   2   3                                       │
│                                                                  │
│  Property: Any t points can reconstruct f(x), thus S            │
│            Fewer than t points reveal nothing about S           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Lagrange Interpolation

```
┌─────────────────────────────────────────────────────────────────┐
│              Reconstructing the Secret                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Given t points: (x₁, y₁), (x₂, y₂), ..., (xₜ, yₜ)              │
│                                                                  │
│  Lagrange basis polynomials:                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │           ∏ (x - xⱼ)                                     │    │
│  │  Lᵢ(x) = ───────────────   for j ≠ i                    │    │
│  │           ∏ (xᵢ - xⱼ)                                    │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  Reconstruct polynomial:                                         │
│  f(x) = ∑ yᵢ × Lᵢ(x)                                            │
│                                                                  │
│  Get secret:                                                     │
│  S = f(0) = ∑ yᵢ × Lᵢ(0)                                        │
│                                                                  │
│  This works for any t shares from the original n!               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## PIR in RAG Systems

Integrating PIR with RAG requires careful design:

```
┌─────────────────────────────────────────────────────────────────┐
│                PIR-Enabled RAG Architecture                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   User Query: "treatment options"                                │
│         │                                                        │
│         ▼                                                        │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │              LOCAL: Top-K Selection                       │  │
│   │  ┌────────────────────────────────────────────────────┐  │  │
│   │  │  User has cached embeddings locally                 │  │  │
│   │  │  Computes similarity scores                         │  │  │
│   │  │  Selects top-k indices: [7, 23, 45]                 │  │  │
│   │  └────────────────────────────────────────────────────┘  │  │
│   └──────────────────────────────────────────────────────────┘  │
│         │                                                        │
│         ▼                                                        │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │              PIR: Private Document Fetch                  │  │
│   │  ┌────────────────────────────────────────────────────┐  │  │
│   │  │  For each index i in [7, 23, 45]:                   │  │  │
│   │  │    PIR.retrieve(i) → Document                       │  │  │
│   │  │                                                     │  │  │
│   │  │  Server processes query but learns nothing          │  │  │
│   │  │  about which documents were retrieved!              │  │  │
│   │  └────────────────────────────────────────────────────┘  │  │
│   └──────────────────────────────────────────────────────────┘  │
│         │                                                        │
│         ▼                                                        │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │              LOCAL: LLM Generation                        │  │
│   │  ┌────────────────────────────────────────────────────┐  │  │
│   │  │  Retrieved documents used as context                │  │  │
│   │  │  LLM generates response                             │  │  │
│   │  └────────────────────────────────────────────────────┘  │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Hybrid DP + PIR: Defense in Depth

For maximum privacy, combine differential privacy with PIR:

```
┌─────────────────────────────────────────────────────────────────┐
│                  Hybrid DP + PIR Protection                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Layer 1: Differential Privacy                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  • Add noise to similarity scores                        │   │
│   │  • Protects: Which documents are "most relevant"         │   │
│   │  • Provides: Plausible deniability in rankings           │   │
│   └─────────────────────────────────────────────────────────┘   │
│                          │                                       │
│                          ▼                                       │
│   Layer 2: Private Information Retrieval                         │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  • Fetch documents privately                             │   │
│   │  • Protects: Which documents are actually retrieved      │   │
│   │  • Provides: Cryptographic query privacy                 │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Combined Protection:                                           │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  ✓ Query pattern hidden (PIR)                            │   │
│   │  ✓ Relevance scores obfuscated (DP)                      │   │
│   │  ✓ Access patterns unlinkable                            │   │
│   │  ✓ Even timing analysis is difficult                     │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Performance Considerations

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIR Performance Trade-offs                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│                    Communication    Computation    Trust         │
│   ┌─────────────┬────────────────┬──────────────┬─────────────┐ │
│   │ Single-     │ O(n) encrypted │ Heavy        │ None        │ │
│   │ Server PIR  │ elements       │ (HE ops)     │ required    │ │
│   ├─────────────┼────────────────┼──────────────┼─────────────┤ │
│   │ Multi-      │ O(n) per       │ Light        │ Non-        │ │
│   │ Server PIR  │ server         │ (XOR ops)    │ collusion   │ │
│   ├─────────────┼────────────────┼──────────────┼─────────────┤ │
│   │ Threshold   │ O(n) per       │ Light +      │ t-of-n      │ │
│   │ PIR         │ server         │ interpolation│ honest      │ │
│   └─────────────┴────────────────┴──────────────┴─────────────┘ │
│                                                                  │
│   Optimization strategies:                                       │
│   • Matrix PIR: Reduce to O(√n) communication                   │
│   • Batch PIR: Amortize costs across multiple queries           │
│   • Caching: Pre-compute common structures                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Security Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│               Privacy Mechanism Comparison                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Mechanism          What Server Learns       Guarantee          │
│   ─────────────────────────────────────────────────────────────  │
│                                                                  │
│   No Protection      Everything               None               │
│   └─ Query, results, patterns                                    │
│                                                                  │
│   Differential       Noisy statistics         Statistical        │
│   Privacy            └─ Approximate patterns                     │
│                                                                  │
│   Single-Server      Nothing                  Computational      │
│   PIR                └─ Under crypto assumptions                 │
│                                                                  │
│   Multi-Server       Nothing                  Information-       │
│   PIR                └─ Even with infinite    Theoretic          │
│                         compute power                            │
│                                                                  │
│   DP + PIR           Nothing + Noise          Both guarantees    │
│   (Hybrid)           └─ Defense in depth                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## RAG-Shield's PIR Implementation

RAG-Shield provides a unified interface for PIR-based retrieval:

```
┌─────────────────────────────────────────────────────────────────┐
│                   RAG-Shield PIR Module                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ragshield.pir                                                  │
│   ├── base.py              # Core abstractions                   │
│   │   ├── PIRClient        # Client-side operations              │
│   │   ├── PIRServer        # Server-side operations              │
│   │   └── PIRProtocol      # Protocol interface                  │
│   │                                                              │
│   ├── single_server.py     # Homomorphic encryption PIR          │
│   │   ├── SimplifiedPaillier                                     │
│   │   └── SingleServerPIR                                        │
│   │                                                              │
│   ├── multi_server.py      # Secret sharing PIR                  │
│   │   ├── XORSecretSharing                                       │
│   │   ├── ShamirSecretSharing                                    │
│   │   ├── MultiServerPIR                                         │
│   │   └── ThresholdPIR                                           │
│   │                                                              │
│   └── pir_retriever.py     # RAG integration                     │
│       ├── PIRRetriever     # Main interface                      │
│       ├── HybridPIRRetriever  # DP + PIR                         │
│       └── BatchPIRRetriever   # Batch operations                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## When to Use PIR

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIR Usage Guidelines                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   USE PIR when:                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  ✓ Query patterns are sensitive (medical, legal, etc.)  │   │
│   │  ✓ Server is untrusted but provides necessary service   │   │
│   │  ✓ Regulatory compliance requires query privacy         │   │
│   │  ✓ You can afford the computational overhead            │   │
│   │  ✓ (Multi-server) You have access to non-colluding      │   │
│   │     infrastructure                                       │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   SKIP PIR when:                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  ✗ Server is fully trusted                               │   │
│   │  ✗ Query patterns aren't sensitive                       │   │
│   │  ✗ Latency requirements are very strict                  │   │
│   │  ✗ Database is extremely large (billions of docs)        │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Consider DP instead when:                                      │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  ~ Statistical privacy is sufficient                     │   │
│   │  ~ Performance is critical                               │   │
│   │  ~ Approximate privacy guarantees are acceptable         │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Summary

Private Information Retrieval provides the strongest form of query privacy for RAG systems:

| Aspect | DP | Single-Server PIR | Multi-Server PIR |
|--------|-----|-------------------|------------------|
| Privacy | Statistical | Computational | Information-theoretic |
| Performance | Fast | Slow | Medium |
| Trust | None | None | Non-collusion |
| Implementation | Simple | Complex | Medium |

By combining DP (for relevance score obfuscation) with PIR (for access pattern hiding), RAG-Shield provides comprehensive privacy protection that addresses both what documents are relevant and which documents are actually retrieved.

## Next Steps

In the upcoming phases, we'll explore:
- **Attack Forensics**: Tracing the source of poisoning attacks
- **Defense Mechanisms**: Active protection against adversarial documents
- **Benchmarks**: Evaluating security-performance trade-offs

---

*RAG-Shield: Making RAG systems secure, private, and trustworthy.*
