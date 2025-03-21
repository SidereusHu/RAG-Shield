# Blog 6: Attack Forensics - Tracing Poisoning Attacks in RAG Systems

*February 20, 2025*

When a poisoning attack is detected in your RAG system, the first question is: **where did it come from?** Understanding the attack's origin, pattern, and scope is crucial for effective response and prevention. This blog explores forensic techniques for investigating RAG poisoning attacks.

## The Forensics Challenge

After detecting suspicious documents, security teams face several questions:

```
+-------------------------------------------------------------------+
|                    Post-Detection Questions                        |
+-------------------------------------------------------------------+
|                                                                    |
|   1. ORIGIN                                                        |
|      Where did this document come from?                            |
|      Who submitted it? When? Through which channel?                |
|                                                                    |
|   2. SCOPE                                                         |
|      Is this an isolated incident or part of a campaign?           |
|      Are there other related malicious documents?                  |
|                                                                    |
|   3. PATTERN                                                       |
|      What attack technique was used?                               |
|      Does it match known attack signatures?                        |
|                                                                    |
|   4. ATTRIBUTION                                                   |
|      Can we identify the attacker?                                 |
|      Have they attacked before?                                    |
|                                                                    |
+-------------------------------------------------------------------+
```

RAG-Shield's forensics module provides tools to answer these questions systematically.

## Component 1: Provenance Tracking

Provenance tracking maintains a **chain of custody** for every document, recording all events in its lifecycle.

### The Provenance Chain

```
+-------------------------------------------------------------------+
|                    Document Provenance Chain                       |
+-------------------------------------------------------------------+
|                                                                    |
|   Genesis Block              Event Blocks                          |
|   +-----------+    +------------+    +------------+    +--------+  |
|   | CREATED   |--->| INGESTED   |--->| VERIFIED   |--->| FLAGGED|  |
|   +-----------+    +------------+    +------------+    +--------+  |
|   | timestamp |    | timestamp  |    | timestamp  |    |timestamp| |
|   | actor     |    | actor      |    | actor      |    | actor   | |
|   | source    |    | system_id  |    | check_type |    | reason  | |
|   | hash: H0  |    | prev: H0   |    | prev: H1   |    | prev: H2| |
|   |           |    | hash: H1   |    | hash: H2   |    | hash: H3| |
|   +-----------+    +------------+    +------------+    +--------+  |
|                                                                    |
|   Chain Property: Each block links to previous via hash            |
|   Tamper Detection: Breaking any link invalidates chain            |
|                                                                    |
+-------------------------------------------------------------------+
```

### Event Types in Provenance

```
+-------------------------------------------------------------------+
|                    Provenance Event Types                          |
+-------------------------------------------------------------------+
|                                                                    |
|   Lifecycle Events:                                                |
|   +-------------+  +-------------+  +-------------+                |
|   |   CREATED   |  |  INGESTED   |  |  MODIFIED   |                |
|   | First entry |  | Added to KB |  |Content change|               |
|   +-------------+  +-------------+  +-------------+                |
|                                                                    |
|   Security Events:                                                 |
|   +-------------+  +-------------+  +-------------+                |
|   |   FLAGGED   |  | QUARANTINED |  |  VERIFIED   |                |
|   |  Suspicious |  |  Isolated   |  | Checked OK  |                |
|   +-------------+  +-------------+  +-------------+                |
|                                                                    |
|   Resolution Events:                                               |
|   +-------------+  +-------------+  +-------------+                |
|   |  RELEASED   |  |  REJECTED   |  |   DELETED   |                |
|   |  Approved   |  |  Confirmed  |  |   Removed   |                |
|   +-------------+  +-------------+  +-------------+                |
|                                                                    |
+-------------------------------------------------------------------+
```

### Chain Integrity Verification

```
+-------------------------------------------------------------------+
|                    Integrity Verification                          |
+-------------------------------------------------------------------+
|                                                                    |
|   Valid Chain:                                                     |
|   +------+      +------+      +------+      +------+               |
|   | E1   |----->| E2   |----->| E3   |----->| E4   |               |
|   | H:abc|      | P:abc|      | P:def|      | P:ghi|               |
|   |      |      | H:def|      | H:ghi|      | H:jkl|               |
|   +------+      +------+      +------+      +------+               |
|       ^              ^             ^             ^                  |
|       |              |             |             |                  |
|   hash(E1)=abc   hash(E2)=def  hash(E3)=ghi  hash(E4)=jkl          |
|                                                                    |
|   Tampered Chain (detected):                                       |
|   +------+      +------+      +------+      +------+               |
|   | E1   |----->| E2*  |--X-->| E3   |----->| E4   |               |
|   | H:abc|      | P:abc|      | P:def|      | P:ghi|               |
|   |      |      | H:xyz|      | H:ghi|      | H:jkl|               |
|   +------+      +------+      +------+      +------+               |
|                      ^                                             |
|               Modified! hash(E2*)=xyz != def                       |
|               Chain broken at E2-E3 link                           |
|                                                                    |
+-------------------------------------------------------------------+
```

## Component 2: Attack Pattern Analysis

Pattern analysis identifies the **attack technique** used by examining document characteristics.

### Known Attack Patterns

```
+-------------------------------------------------------------------+
|                    Attack Pattern Library                          |
+-------------------------------------------------------------------+
|                                                                    |
|   KEYWORD_STUFFING                                                 |
|   +---------------------------------------------------------------+|
|   | "python python python python machine learning python..."      ||
|   | Indicator: Word repeated 5+ times                             ||
|   | Purpose: Maximize retrieval score                             ||
|   +---------------------------------------------------------------+|
|                                                                    |
|   QUERY_INJECTION                                                  |
|   +---------------------------------------------------------------+|
|   | "Question: What is X?                                         ||
|   |  Answer: [malicious content]"                                 ||
|   | Indicator: Q&A format targeting specific queries              ||
|   +---------------------------------------------------------------+|
|                                                                    |
|   AUTHORITY_MIMICKING                                              |
|   +---------------------------------------------------------------+|
|   | "According to official sources..."                            ||
|   | "Expert consensus indicates..."                               ||
|   | "Verified information confirms..."                            ||
|   | Indicator: Trust-building phrases                             ||
|   +---------------------------------------------------------------+|
|                                                                    |
|   TEMPLATE_BASED                                                   |
|   +---------------------------------------------------------------+|
|   | "The correct answer is: [X]"                                  ||
|   | "Q: ... A: ..."                                               ||
|   | Indicator: Structured answer injection                        ||
|   +---------------------------------------------------------------+|
|                                                                    |
+-------------------------------------------------------------------+
```

### Pattern Detection Process

```
+-------------------------------------------------------------------+
|                    Pattern Detection Flow                          |
+-------------------------------------------------------------------+
|                                                                    |
|   Suspicious Document                                              |
|          |                                                         |
|          v                                                         |
|   +-------------+                                                  |
|   | Tokenize &  |                                                  |
|   | Normalize   |                                                  |
|   +-------------+                                                  |
|          |                                                         |
|          +------------------+------------------+                   |
|          |                  |                  |                   |
|          v                  v                  v                   |
|   +-----------+      +-----------+      +-----------+              |
|   | Keyword   |      |  Query    |      | Authority |              |
|   | Analysis  |      | Pattern   |      |  Phrase   |              |
|   +-----------+      +-----------+      +-----------+              |
|          |                  |                  |                   |
|          v                  v                  v                   |
|   +---------------------------------------------------+           |
|   |              Pattern Match Aggregator              |           |
|   +---------------------------------------------------+           |
|          |                                                         |
|          v                                                         |
|   +---------------------------------------------------+           |
|   |   Detected Patterns:                               |           |
|   |   - AUTHORITY_MIMICKING (confidence: 0.85)         |           |
|   |   - QUERY_INJECTION (confidence: 0.72)             |           |
|   |   Estimated Attack Type: DIRECT                    |           |
|   +---------------------------------------------------+           |
|                                                                    |
+-------------------------------------------------------------------+
```

### Attack Fingerprinting

Each analyzed document generates a **fingerprint** for comparison and clustering:

```
+-------------------------------------------------------------------+
|                    Attack Fingerprint                              |
+-------------------------------------------------------------------+
|                                                                    |
|   +-------------------------------------------------------------+ |
|   |                    Fingerprint Components                    | |
|   +-------------------------------------------------------------+ |
|   |                                                              | |
|   |   Content Hash          Structural Hash                      | |
|   |   +--------------+      +--------------+                     | |
|   |   | SHA256 of    |      | SHA256 of    |                     | |
|   |   | exact content|      | structure    |                     | |
|   |   | "abc123..."  |      | "WWW.NNN..." |                     | |
|   |   +--------------+      +--------------+                     | |
|   |                                                              | |
|   |   Vocabulary Signature   Style Metrics                       | |
|   |   +--------------+      +--------------+                     | |
|   |   | Distinctive  |      | avg_sentence |                     | |
|   |   | words set    |      | avg_word_len |                     | |
|   |   | {official,   |      | punct_ratio  |                     | |
|   |   |  verified,   |      | upper_ratio  |                     | |
|   |   |  expert...}  |      |              |                     | |
|   |   +--------------+      +--------------+                     | |
|   |                                                              | |
|   |   Detected Patterns                                          | |
|   |   [AUTHORITY_MIMICKING, QUERY_INJECTION]                     | |
|   |                                                              | |
|   +-------------------------------------------------------------+ |
|                                                                    |
|   Similarity Calculation:                                          |
|   sim(A,B) = 0.3*pattern_overlap + 0.4*vocab_jaccard + 0.3*style  |
|                                                                    |
+-------------------------------------------------------------------+
```

## Component 3: Timeline Reconstruction

When multiple suspicious documents are found, timeline reconstruction reveals the **attack progression**.

### Attack Timeline Visualization

```
+-------------------------------------------------------------------+
|                    Attack Timeline                                 |
+-------------------------------------------------------------------+
|                                                                    |
|   Time ─────────────────────────────────────────────────────────>  |
|                                                                    |
|   Phase 1: Reconnaissance        Phase 2: Main Attack              |
|   ├──────────────────────────────┼────────────────────────────────>|
|   |                              |                                 |
|   10:00    10:15    10:30       14:00   14:05   14:10   14:15     |
|     |        |        |           |       |       |       |       |
|     v        v        v           v       v       v       v       |
|   +---+    +---+    +---+       +---+   +---+   +---+   +---+     |
|   |D1 |    |D2 |    |D3 |       |D4 |   |D5 |   |D6 |   |D7 |     |
|   +---+    +---+    +---+       +---+   +---+   +---+   +---+     |
|     |        |        |           |       |       |       |       |
|   probe    probe    probe      attack  attack  attack  attack    |
|   doc      doc      doc         doc     doc     doc     doc      |
|                                                                    |
|   Wave Detection:                                                  |
|   - Wave 1 (10:00-10:30): 3 documents, reconnaissance             |
|   - Wave 2 (14:00-14:15): 4 documents, main attack                |
|                                                                    |
|   Gap Analysis: 3.5 hour gap suggests staged attack               |
|                                                                    |
+-------------------------------------------------------------------+
```

### Attack Phase Identification

```
+-------------------------------------------------------------------+
|                    Attack Phase Analysis                           |
+-------------------------------------------------------------------+
|                                                                    |
|   +-----------------------+                                        |
|   |    Phase 1: Setup     |                                        |
|   |-----------------------|                                        |
|   | Duration: 30 min      |                                        |
|   | Documents: 3          |                                        |
|   | Pattern: Low severity |                                        |
|   | Purpose: Test defenses|                                        |
|   +-----------------------+                                        |
|              |                                                     |
|              v                                                     |
|   +-----------------------+                                        |
|   |  Phase 2: Escalation  |                                        |
|   |-----------------------|                                        |
|   | Duration: 15 min      |                                        |
|   | Documents: 4          |                                        |
|   | Pattern: High severity|                                        |
|   | Purpose: Main payload |                                        |
|   +-----------------------+                                        |
|              |                                                     |
|              v                                                     |
|   +-----------------------+                                        |
|   |  Phase 3: Persistence |                                        |
|   |-----------------------|                                        |
|   | Duration: Ongoing     |                                        |
|   | Documents: 2          |                                        |
|   | Pattern: Stealth      |                                        |
|   | Purpose: Maintain     |                                        |
|   |          foothold     |                                        |
|   +-----------------------+                                        |
|                                                                    |
+-------------------------------------------------------------------+
```

### Correlated Document Discovery

```
+-------------------------------------------------------------------+
|                    Document Correlation                            |
+-------------------------------------------------------------------+
|                                                                    |
|   Reference Document: doc_suspicious_001                           |
|   Ingestion Time: 14:05:32                                         |
|   Time Window: +/- 30 minutes                                      |
|                                                                    |
|   Correlated Documents Found:                                      |
|                                                                    |
|   +-----------+  +-----------+  +-----------+  +-----------+       |
|   | doc_004   |  | doc_005   |  | doc_006   |  | doc_007   |       |
|   | 14:00:12  |  | 14:05:32  |  | 14:10:45  |  | 14:15:01  |       |
|   | sim: 0.82 |  | REFERENCE |  | sim: 0.78 |  | sim: 0.85 |       |
|   +-----------+  +-----------+  +-----------+  +-----------+       |
|        |              |              |              |              |
|        +------+-------+-------+------+              |              |
|               |               |                     |              |
|               v               v                     v              |
|        Same Source?     Same Pattern?         Same Campaign?       |
|             YES              YES                   YES             |
|                                                                    |
|   Conclusion: 4 documents from coordinated attack campaign         |
|                                                                    |
+-------------------------------------------------------------------+
```

## Component 4: Attack Attribution

Attribution links attacks to sources and identifies campaigns.

### Attribution Confidence Levels

```
+-------------------------------------------------------------------+
|                    Attribution Confidence                          |
+-------------------------------------------------------------------+
|                                                                    |
|   DEFINITE (>95%)                                                  |
|   +---------------------------------------------------------------+|
|   | Direct evidence: Same API key, IP, user account               ||
|   | Action: Block source immediately                              ||
|   +---------------------------------------------------------------+|
|                                                                    |
|   HIGH (80-95%)                                                    |
|   +---------------------------------------------------------------+|
|   | Strong correlation: Fingerprint match, behavioral pattern     ||
|   | Action: Flag for review, prepare blocking                     ||
|   +---------------------------------------------------------------+|
|                                                                    |
|   MEDIUM (50-80%)                                                  |
|   +---------------------------------------------------------------+|
|   | Moderate evidence: Similar style, overlapping tactics         ||
|   | Action: Monitor closely, gather more evidence                 ||
|   +---------------------------------------------------------------+|
|                                                                    |
|   LOW (20-50%)                                                     |
|   +---------------------------------------------------------------+|
|   | Weak signals: Some pattern similarity                         ||
|   | Action: Log for future correlation                            ||
|   +---------------------------------------------------------------+|
|                                                                    |
|   UNCERTAIN (<20%)                                                 |
|   +---------------------------------------------------------------+|
|   | Insufficient evidence                                         ||
|   | Action: Continue monitoring                                   ||
|   +---------------------------------------------------------------+|
|                                                                    |
+-------------------------------------------------------------------+
```

### Campaign Identification

```
+-------------------------------------------------------------------+
|                    Attack Campaign Structure                       |
+-------------------------------------------------------------------+
|                                                                    |
|   Campaign: "Capital City Misinformation"                          |
|   ID: camp_abc123                                                  |
|                                                                    |
|   +-------------------------------------------------------------+ |
|   |                        Sources                               | |
|   | +------------------+  +------------------+                   | |
|   | | Source 1         |  | Source 2         |                   | |
|   | | Type: IP         |  | Type: API Key    |                   | |
|   | | ID: 192.168.x.x  |  | ID: key_xyz      |                   | |
|   | | Confidence: HIGH |  | Confidence: MED  |                   | |
|   | +------------------+  +------------------+                   | |
|   +-------------------------------------------------------------+ |
|                                                                    |
|   +-------------------------------------------------------------+ |
|   |                       Documents                              | |
|   | doc_001, doc_002, doc_003, doc_004, doc_005                  | |
|   | Total: 5 documents                                           | |
|   +-------------------------------------------------------------+ |
|                                                                    |
|   +-------------------------------------------------------------+ |
|   |                        Tactics                               | |
|   | - Authority mimicking                                        | |
|   | - Query injection                                            | |
|   | - Template-based answers                                     | |
|   +-------------------------------------------------------------+ |
|                                                                    |
|   +-------------------------------------------------------------+ |
|   |                     Common Fingerprint                       | |
|   | Vocabulary: {official, verified, expert, consensus}          | |
|   | Style: Formal, assertion-heavy                               | |
|   | Structure: Q&A format                                        | |
|   +-------------------------------------------------------------+ |
|                                                                    |
+-------------------------------------------------------------------+
```

### Attribution Workflow

```
+-------------------------------------------------------------------+
|                    Attribution Process                             |
+-------------------------------------------------------------------+
|                                                                    |
|   New Suspicious Document                                          |
|          |                                                         |
|          v                                                         |
|   +----------------+                                               |
|   | Generate       |                                               |
|   | Fingerprint    |                                               |
|   +----------------+                                               |
|          |                                                         |
|          v                                                         |
|   +----------------+     +-------------------+                     |
|   | Search Known   |---->| Match Found?      |                     |
|   | Fingerprints   |     | Similarity > 0.7  |                     |
|   +----------------+     +-------------------+                     |
|                                |         |                         |
|                               YES        NO                        |
|                                |         |                         |
|                                v         v                         |
|                    +-------------+  +-------------+                |
|                    | Link to     |  | Check       |                |
|                    | Known Source|  | Provenance  |                |
|                    +-------------+  +-------------+                |
|                          |                |                        |
|                          v                v                        |
|                    +-------------+  +-------------+                |
|                    | Add to      |  | Create New  |                |
|                    | Campaign    |  | Source Entry|                |
|                    +-------------+  +-------------+                |
|                          |                |                        |
|                          +-------+--------+                        |
|                                  |                                 |
|                                  v                                 |
|                    +---------------------------+                   |
|                    |    Attribution Report      |                   |
|                    | - Confidence level         |                   |
|                    | - Related sources          |                   |
|                    | - Campaign membership      |                   |
|                    | - Recommendations          |                   |
|                    +---------------------------+                   |
|                                                                    |
+-------------------------------------------------------------------+
```

## Putting It All Together

The forensics workflow integrates all components:

```
+-------------------------------------------------------------------+
|                    Complete Forensics Flow                         |
+-------------------------------------------------------------------+
|                                                                    |
|   DETECTION TRIGGER                                                |
|   Document flagged as suspicious                                   |
|          |                                                         |
|          v                                                         |
|   +------------------+                                             |
|   |   PROVENANCE     |  Where did it come from?                    |
|   |   - Get chain    |  Who submitted it?                          |
|   |   - Verify       |  When?                                      |
|   +------------------+                                             |
|          |                                                         |
|          v                                                         |
|   +------------------+                                             |
|   |    ANALYSIS      |  What attack was used?                      |
|   |   - Patterns     |  What's the fingerprint?                    |
|   |   - Fingerprint  |                                             |
|   +------------------+                                             |
|          |                                                         |
|          v                                                         |
|   +------------------+                                             |
|   |    TIMELINE      |  Is this part of a campaign?                |
|   |   - Correlate    |  When did it start?                         |
|   |   - Phases       |  How many documents?                        |
|   +------------------+                                             |
|          |                                                         |
|          v                                                         |
|   +------------------+                                             |
|   |   ATTRIBUTION    |  Who is responsible?                        |
|   |   - Sources      |  Have they attacked before?                 |
|   |   - Campaigns    |  What's the confidence?                     |
|   +------------------+                                             |
|          |                                                         |
|          v                                                         |
|   +------------------+                                             |
|   | FORENSIC REPORT  |                                             |
|   | - Full timeline  |                                             |
|   | - Attack scope   |                                             |
|   | - Attribution    |                                             |
|   | - Recommendations|                                             |
|   +------------------+                                             |
|                                                                    |
+-------------------------------------------------------------------+
```

## Key Takeaways

| Component | Purpose | Output |
|-----------|---------|--------|
| Provenance | Track document lifecycle | Chain of custody |
| Analysis | Identify attack technique | Pattern + fingerprint |
| Timeline | Understand attack progression | Phases + waves |
| Attribution | Identify attacker | Source + campaign |

Forensics transforms a detected threat into **actionable intelligence**:
- Block identified sources
- Find related malicious documents
- Understand attacker tactics
- Prevent future attacks

## Next Steps

In the next blog, we'll explore how to use forensic intelligence to build **active defenses** - from quarantine systems to real-time monitoring and unified protection shields.

---

*RAG-Shield: From detection to investigation to protection.*
