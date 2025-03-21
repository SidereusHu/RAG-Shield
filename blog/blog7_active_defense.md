# Blog 7: Active Defense - Building a Security Shield for RAG Systems

*February 20, 2025*

Detection and forensics tell us when attacks happen and who's responsible. But the best security is **prevention**. This blog explores RAG-Shield's active defense mechanisms - from input sanitization to real-time monitoring to a unified protection layer.

## Defense-in-Depth Architecture

Modern security requires multiple layers of protection:

```
+-------------------------------------------------------------------+
|                    Defense-in-Depth Model                          |
+-------------------------------------------------------------------+
|                                                                    |
|   Incoming Document                                                |
|          |                                                         |
|          v                                                         |
|   +==================+                                             |
|   ||  LAYER 1       ||  Rate Limiting                              |
|   ||  Monitoring    ||  Source Blocking                            |
|   ||                ||  Anomaly Detection                          |
|   +==================+                                             |
|          |                                                         |
|          v                                                         |
|   +==================+                                             |
|   ||  LAYER 2       ||  Content Filtering                          |
|   ||  Sanitization  ||  Embedding Validation                       |
|   ||                ||  Metadata Cleaning                          |
|   +==================+                                             |
|          |                                                         |
|          v                                                         |
|   +==================+                                             |
|   ||  LAYER 3       ||  Pattern Detection                          |
|   ||  Detection     ||  Similarity Analysis                        |
|   ||                ||  Semantic Checking                          |
|   +==================+                                             |
|          |                                                         |
|          v                                                         |
|   +==================+                                             |
|   ||  LAYER 4       ||  Suspicious? -> Quarantine                  |
|   ||  Quarantine    ||  Safe? -> Knowledge Base                    |
|   ||                ||  Malicious? -> Block                        |
|   +==================+                                             |
|          |                                                         |
|          v                                                         |
|   +-------------------+                                            |
|   |   Knowledge Base  |  Protected Content                         |
|   +-------------------+                                            |
|                                                                    |
+-------------------------------------------------------------------+
```

## Component 1: Security Monitor

The security monitor provides **real-time protection** at the ingestion boundary.

### Rate Limiting

```
+-------------------------------------------------------------------+
|                    Rate Limiting System                            |
+-------------------------------------------------------------------+
|                                                                    |
|   Sliding Window Algorithm:                                        |
|                                                                    |
|   Time ──────────────────────────────────────────────────────────> |
|                                                                    |
|   Window (60 seconds)                                              |
|   |<─────────────────────────────────────────>|                    |
|   |  R  R  R  R  R  R  R  R  R  R             |  R = Request       |
|   |  1  2  3  4  5  6  7  8  9  10            |                    |
|                                                                    |
|   Limit: 10 requests per 60 seconds                                |
|                                                                    |
|   Request 11 arrives:                                              |
|   |<─────────────────────────────────────────>|                    |
|   |     R  R  R  R  R  R  R  R  R  R          | R                  |
|   |     2  3  4  5  6  7  8  9  10 11         | BLOCKED            |
|                                                                    |
|   After R1 expires:                                                |
|   |<─────────────────────────────────────────>|                    |
|      |  R  R  R  R  R  R  R  R  R             |  R                 |
|      |  2  3  4  5  6  7  8  9  10            |  11 ALLOWED        |
|                                                                    |
+-------------------------------------------------------------------+
```

### Multi-Level Rate Limits

```
+-------------------------------------------------------------------+
|                    Rate Limit Hierarchy                            |
+-------------------------------------------------------------------+
|                                                                    |
|   +-----------------------------------------------------------+   |
|   |                    Global Limit                            |   |
|   |              100 documents / minute                        |   |
|   |                 (all sources)                              |   |
|   +-----------------------------------------------------------+   |
|                              |                                     |
|         +--------------------+--------------------+                |
|         |                    |                    |                |
|         v                    v                    v                |
|   +-------------+      +-------------+      +-------------+        |
|   | Source A    |      | Source B    |      | Source C    |        |
|   | 20 docs/min |      | 20 docs/min |      | 20 docs/min |        |
|   +-------------+      +-------------+      +-------------+        |
|                                                                    |
|   Benefits:                                                        |
|   - Prevents single source from monopolizing                       |
|   - Limits damage from compromised credentials                     |
|   - Allows fair access for legitimate sources                      |
|                                                                    |
+-------------------------------------------------------------------+
```

### Anomaly Detection

```
+-------------------------------------------------------------------+
|                    Anomaly Detection                               |
+-------------------------------------------------------------------+
|                                                                    |
|   Baseline Building (rolling window of 100 samples):               |
|                                                                    |
|   Ingestion Rate                                                   |
|   ^                                                                |
|   |         Baseline Range                                         |
|   |    .....|=============|.....                                   |
|   |   .     | normal zone |     .                                  |
|   |  .      |             |      .                                 |
|   | .       |   mean=50   |       .                                |
|   |.        |   std=10    |        .                               |
|   +-----------------------------------------> Time                 |
|                                                                    |
|   Anomaly Triggers (Z-score > 3):                                  |
|                                                                    |
|   Rate                                                             |
|   ^                                                                |
|   |                              * <- 150 docs/min                 |
|   |                              |    Z-score = 10                 |
|   |                              |    ANOMALY!                     |
|   |    .....|=============|.....|                                  |
|   |   .     |             |     .                                  |
|   +-----------------------------------------> Time                 |
|                                                                    |
|   Alert: "Unusual bulk ingestion detected (z=10.0)"                |
|                                                                    |
+-------------------------------------------------------------------+
```

### Source Blocking

```
+-------------------------------------------------------------------+
|                    Source Blocking                                 |
+-------------------------------------------------------------------+
|                                                                    |
|   Block Triggers:                                                  |
|   +-----------------------------------------------------------+   |
|   | Trigger              | Block Duration | Auto-Unblock      |   |
|   |----------------------|----------------|-------------------|   |
|   | Rate limit abuse     | 1 hour         | Yes               |   |
|   | High-threat document | 24 hours       | Yes               |   |
|   | Multiple detections  | 7 days         | Manual only       |   |
|   | Admin blacklist      | Permanent      | Manual only       |   |
|   +-----------------------------------------------------------+   |
|                                                                    |
|   Block Status Check:                                              |
|                                                                    |
|   Request from source_x                                            |
|          |                                                         |
|          v                                                         |
|   +---------------+                                                |
|   | Is Blocked?   |                                                |
|   +---------------+                                                |
|          |                                                         |
|     +----+----+                                                    |
|     |         |                                                    |
|    YES        NO                                                   |
|     |         |                                                    |
|     v         v                                                    |
|   +-------+  +--------+                                            |
|   | Check |  |Proceed |                                            |
|   | Expiry|  |        |                                            |
|   +-------+  +--------+                                            |
|     |                                                              |
|  +--+--+                                                           |
|  |     |                                                           |
| Exp'd  Active                                                      |
|  |     |                                                           |
|  v     v                                                           |
| Unblock REJECT                                                     |
|                                                                    |
+-------------------------------------------------------------------+
```

### Alert System

```
+-------------------------------------------------------------------+
|                    Alert Architecture                              |
+-------------------------------------------------------------------+
|                                                                    |
|   Alert Severity Levels:                                           |
|                                                                    |
|   +-----------------------------------------------------------+   |
|   | CRITICAL |  System under active attack                    |   |
|   |          |  Immediate action required                     |   |
|   +-----------------------------------------------------------+   |
|   | HIGH     |  Significant threat detected                   |   |
|   |          |  Prompt investigation needed                   |   |
|   +-----------------------------------------------------------+   |
|   | WARNING  |  Suspicious activity observed                  |   |
|   |          |  Monitor closely                               |   |
|   +-----------------------------------------------------------+   |
|   | INFO     |  Notable event for logging                     |   |
|   |          |  No action required                            |   |
|   +-----------------------------------------------------------+   |
|                                                                    |
|   Alert Types:                                                     |
|   - RATE_LIMIT_EXCEEDED                                            |
|   - SUSPICIOUS_DOCUMENT                                            |
|   - ANOMALY_DETECTED                                               |
|   - BULK_INGESTION                                                 |
|   - SOURCE_BLOCKED                                                 |
|   - THRESHOLD_EXCEEDED                                             |
|                                                                    |
+-------------------------------------------------------------------+
```

## Component 2: Document Sanitizer

Sanitization cleans documents before they enter the knowledge base.

### Three-Layer Sanitization

```
+-------------------------------------------------------------------+
|                    Sanitization Pipeline                           |
+-------------------------------------------------------------------+
|                                                                    |
|   Raw Document                                                     |
|          |                                                         |
|          v                                                         |
|   +==========================================+                     |
|   ||         CONTENT SANITIZER              ||                     |
|   ||----------------------------------------||                     |
|   ||  Rules:                                ||                     |
|   ||  - Block <script> tags                 ||                     |
|   ||  - Block javascript: URLs              ||                     |
|   ||  - Clean control characters            ||                     |
|   ||  - Trim excessive whitespace           ||                     |
|   ||  - Check length limits                 ||                     |
|   ||                                        ||                     |
|   ||  Result: ALLOW / BLOCK / CLEAN         ||                     |
|   +==========================================+                     |
|          |                                                         |
|          v                                                         |
|   +==========================================+                     |
|   ||        EMBEDDING SANITIZER             ||                     |
|   ||----------------------------------------||                     |
|   ||  Checks:                               ||                     |
|   ||  - Correct dimension?                  ||                     |
|   ||  - No NaN/Inf values?                  ||                     |
|   ||  - Norm in valid range?                ||                     |
|   ||  - Not too sparse?                     ||                     |
|   ||                                        ||                     |
|   ||  Fixes: Normalize, clip values         ||                     |
|   +==========================================+                     |
|          |                                                         |
|          v                                                         |
|   +==========================================+                     |
|   ||        METADATA SANITIZER              ||                     |
|   ||----------------------------------------||                     |
|   ||  Actions:                              ||                     |
|   ||  - Strip reserved fields (poisoned,    ||                     |
|   ||    attack_type, target_query)          ||                     |
|   ||  - Sanitize keys (alphanumeric only)   ||                     |
|   ||  - Truncate long values                ||                     |
|   ||  - Limit field count                   ||                     |
|   +==========================================+                     |
|          |                                                         |
|          v                                                         |
|   Clean Document                                                   |
|                                                                    |
+-------------------------------------------------------------------+
```

### Content Sanitization Rules

```
+-------------------------------------------------------------------+
|                    Content Rules Engine                            |
+-------------------------------------------------------------------+
|                                                                    |
|   Rule Structure:                                                  |
|   +-----------------------------------------------------------+   |
|   | Name     | Pattern           | Action  | Priority          |   |
|   |----------|-------------------|---------|-------------------|   |
|   | XSS      | <script.*>        | BLOCK   | 100 (highest)     |   |
|   | JS URLs  | javascript:       | BLOCK   | 100               |   |
|   | Events   | on\w+=            | BLOCK   | 100               |   |
|   | Control  | [\x00-\x08]       | CLEAN   | 50                |   |
|   | Space    | \s{10,}           | CLEAN   | 50                |   |
|   | Comments | <!--.*-->         | CLEAN   | 50                |   |
|   +-----------------------------------------------------------+   |
|                                                                    |
|   Rule Execution (by priority):                                    |
|                                                                    |
|   Content: "<script>alert('x')</script> Hello"                     |
|          |                                                         |
|          v                                                         |
|   Rule: XSS (priority 100)                                         |
|   Match: YES                                                       |
|   Action: BLOCK                                                    |
|          |                                                         |
|          v                                                         |
|   Result: Document BLOCKED                                         |
|   Reason: "Blocked by rule: block_script_tags"                     |
|                                                                    |
+-------------------------------------------------------------------+
```

### Embedding Validation

```
+-------------------------------------------------------------------+
|                    Embedding Validation                            |
+-------------------------------------------------------------------+
|                                                                    |
|   Valid Embedding Properties:                                      |
|                                                                    |
|   +-----------------------------------------------------------+   |
|   | Property       | Valid Range    | Failure Action           |   |
|   |----------------|----------------|--------------------------|   |
|   | Dimension      | Exactly 384    | Reject                   |   |
|   | Values         | No NaN/Inf     | Reject                   |   |
|   | L2 Norm        | 0.1 - 10.0     | Normalize                |   |
|   | Sparsity       | < 95% zeros    | Flag for review          |   |
|   +-----------------------------------------------------------+   |
|                                                                    |
|   Validation Example:                                              |
|                                                                    |
|   Input: [0.1, 0.2, NaN, 0.4, ...]                                 |
|          |                                                         |
|          v                                                         |
|   +-------------+                                                  |
|   | Check NaN   | --> FAIL: Contains NaN                           |
|   +-------------+                                                  |
|          |                                                         |
|          v                                                         |
|   Result: is_valid=False                                           |
|   Issues: ["Contains NaN values"]                                  |
|                                                                    |
+-------------------------------------------------------------------+
```

## Component 3: Quarantine System

Suspicious documents are isolated for review before admission.

### Quarantine Workflow

```
+-------------------------------------------------------------------+
|                    Quarantine Workflow                             |
+-------------------------------------------------------------------+
|                                                                    |
|   Suspicious Document                                              |
|          |                                                         |
|          v                                                         |
|   +-------------------+                                            |
|   |    QUARANTINE     |                                            |
|   |-------------------|                                            |
|   | Status: PENDING   |                                            |
|   | Expiry: 7 days    |                                            |
|   | Reason: Detection |                                            |
|   +-------------------+                                            |
|          |                                                         |
|          v                                                         |
|   +-------------------+                                            |
|   |   UNDER REVIEW    |                                            |
|   |-------------------|                                            |
|   | Reviewer: analyst |                                            |
|   | Started: 10:00    |                                            |
|   +-------------------+                                            |
|          |                                                         |
|     +----+----+                                                    |
|     |         |                                                    |
|     v         v                                                    |
|   APPROVE   REJECT                                                 |
|     |         |                                                    |
|     v         v                                                    |
|   +-------+ +--------+                                             |
|   |RELEASE| | DELETE |                                             |
|   +-------+ +--------+                                             |
|     |                                                              |
|     v                                                              |
|   Knowledge Base                                                   |
|                                                                    |
+-------------------------------------------------------------------+
```

### Quarantine States

```
+-------------------------------------------------------------------+
|                    Quarantine State Machine                        |
+-------------------------------------------------------------------+
|                                                                    |
|                                                                    |
|                    +----------------+                              |
|                    | PENDING_REVIEW |<-----------+                 |
|                    +----------------+            |                 |
|                           |                      |                 |
|               +-----------+-----------+          |                 |
|               |                       |          |                 |
|               v                       |          |                 |
|        +--------------+               |    +----------+            |
|        | UNDER_REVIEW |               +----|  EXTEND  |            |
|        +--------------+                    +----------+            |
|               |                                                    |
|      +--------+--------+                                           |
|      |                 |                                           |
|      v                 v                                           |
|   +--------+      +----------+                                     |
|   |APPROVED|      | REJECTED |                                     |
|   +--------+      +----------+                                     |
|      |                 |                                           |
|      v                 v                                           |
|   +--------+      +----------+                                     |
|   |RELEASED|      |  DELETED |                                     |
|   +--------+      +----------+                                     |
|                                                                    |
|   Timeout Path:                                                    |
|   PENDING_REVIEW --[expiry]--> EXPIRED --[auto_action]--> DELETED  |
|                                                                    |
+-------------------------------------------------------------------+
```

### Quarantine Entry Details

```
+-------------------------------------------------------------------+
|                    Quarantine Entry                                |
+-------------------------------------------------------------------+
|                                                                    |
|   +-----------------------------------------------------------+   |
|   |                    Entry: doc_sus_001                      |   |
|   +-----------------------------------------------------------+   |
|   |                                                            |   |
|   |   Document                                                 |   |
|   |   +------------------------------------------------------+ |   |
|   |   | ID: doc_sus_001                                      | |   |
|   |   | Content: "According to official sources, the         | |   |
|   |   |           capital of France is Berlin..."            | |   |
|   |   +------------------------------------------------------+ |   |
|   |                                                            |   |
|   |   Detection Result                                         |   |
|   |   +------------------------------------------------------+ |   |
|   |   | Is Poisoned: True                                    | |   |
|   |   | Confidence: 0.85                                     | |   |
|   |   | Threat Level: HIGH                                   | |   |
|   |   | Reason: Authority mimicking pattern detected         | |   |
|   |   +------------------------------------------------------+ |   |
|   |                                                            |   |
|   |   Quarantine Info                                          |   |
|   |   +------------------------------------------------------+ |   |
|   |   | Status: UNDER_REVIEW                                 | |   |
|   |   | Quarantined: 2025-02-20 10:00:00                     | |   |
|   |   | Expires: 2025-02-27 10:00:00                         | |   |
|   |   | Reviewer: security_analyst                          | |   |
|   |   +------------------------------------------------------+ |   |
|   |                                                            |   |
|   |   Review Notes                                             |   |
|   |   +------------------------------------------------------+ |   |
|   |   | - 10:05: Review started by security_analyst          | |   |
|   |   | - 10:10: Confirmed false information                 | |   |
|   |   | - 10:12: Checking for related documents...           | |   |
|   |   +------------------------------------------------------+ |   |
|   |                                                            |   |
|   +-----------------------------------------------------------+   |
|                                                                    |
+-------------------------------------------------------------------+
```

## Component 4: Unified Shield

RAGShield combines all defense mechanisms into one protection layer.

### Shield Architecture

```
+-------------------------------------------------------------------+
|                    RAGShield Architecture                          |
+-------------------------------------------------------------------+
|                                                                    |
|   +-----------------------------------------------------------+   |
|   |                      RAGShield                             |   |
|   +-----------------------------------------------------------+   |
|   |                                                            |   |
|   |   Configuration                                            |   |
|   |   +------------------------------------------------------+ |   |
|   |   | Level: STRICT                                        | |   |
|   |   | Auto-quarantine: Yes                                 | |   |
|   |   | Auto-block sources: Yes                              | |   |
|   |   | Detection threshold: 0.5                             | |   |
|   |   +------------------------------------------------------+ |   |
|   |                                                            |   |
|   |   Components                                               |   |
|   |   +------------+  +------------+  +------------+           |   |
|   |   | Sanitizer  |  |  Monitor   |  | Quarantine |           |   |
|   |   +------------+  +------------+  +------------+           |   |
|   |                                                            |   |
|   |   +------------+  +------------+  +------------+           |   |
|   |   | Detector   |  | Provenance |  |  Analyzer  |           |   |
|   |   +------------+  +------------+  +------------+           |   |
|   |                                                            |   |
|   +-----------------------------------------------------------+   |
|          |                                                         |
|          v                                                         |
|   +-----------------------------------------------------------+   |
|   |                    Knowledge Base                          |   |
|   +-----------------------------------------------------------+   |
|                                                                    |
+-------------------------------------------------------------------+
```

### Defense Levels

```
+-------------------------------------------------------------------+
|                    Defense Level Presets                           |
+-------------------------------------------------------------------+
|                                                                    |
|   MINIMAL                                                          |
|   +-----------------------------------------------------------+   |
|   | Sanitization only                                          |   |
|   | No detection, no monitoring                                |   |
|   | Use case: Trusted internal sources                         |   |
|   +-----------------------------------------------------------+   |
|                                                                    |
|   STANDARD                                                         |
|   +-----------------------------------------------------------+   |
|   | Sanitization + Detection                                   |   |
|   | Monitoring enabled                                         |   |
|   | Auto-quarantine suspicious docs                            |   |
|   | Threshold: 0.7                                             |   |
|   | Use case: Normal operations                                |   |
|   +-----------------------------------------------------------+   |
|                                                                    |
|   STRICT                                                           |
|   +-----------------------------------------------------------+   |
|   | Full protection                                            |   |
|   | Auto-block malicious sources                               |   |
|   | Lower threshold: 0.5                                       |   |
|   | All forensics enabled                                      |   |
|   | Use case: Public-facing systems                            |   |
|   +-----------------------------------------------------------+   |
|                                                                    |
|   PARANOID                                                         |
|   +-----------------------------------------------------------+   |
|   | Maximum security                                           |   |
|   | Very low threshold: 0.3                                    |   |
|   | Aggressive blocking                                        |   |
|   | Use case: High-security environments                       |   |
|   +-----------------------------------------------------------+   |
|                                                                    |
+-------------------------------------------------------------------+
```

### Ingestion Flow Through Shield

```
+-------------------------------------------------------------------+
|                    Protected Ingestion Flow                        |
+-------------------------------------------------------------------+
|                                                                    |
|   Document + Source                                                |
|          |                                                         |
|          v                                                         |
|   +------------------+                                             |
|   | 1. Rate Check    |-----> BLOCKED (rate limit)                  |
|   +------------------+                                             |
|          |                                                         |
|          v                                                         |
|   +------------------+                                             |
|   | 2. Sanitization  |-----> BLOCKED (malicious content)           |
|   +------------------+                                             |
|          |                                                         |
|          v                                                         |
|   +------------------+                                             |
|   | 3. Detection     |-----> QUARANTINED (suspicious)              |
|   +------------------+       |                                     |
|          |                   +---> Source blocked (if HIGH threat) |
|          v                                                         |
|   +------------------+                                             |
|   | 4. Forensics     |  (fingerprint stored)                       |
|   +------------------+                                             |
|          |                                                         |
|          v                                                         |
|   +------------------+                                             |
|   | 5. Provenance    |  (chain created)                            |
|   +------------------+                                             |
|          |                                                         |
|          v                                                         |
|   +------------------+                                             |
|   | 6. Ingest to KB  |  SUCCESS                                    |
|   +------------------+                                             |
|                                                                    |
+-------------------------------------------------------------------+
```

### Ingestion Result

```
+-------------------------------------------------------------------+
|                    Ingestion Result                                |
+-------------------------------------------------------------------+
|                                                                    |
|   Successful Ingestion:                                            |
|   +-----------------------------------------------------------+   |
|   | success: True                                              |   |
|   | action_taken: "ingested"                                   |   |
|   | document: <sanitized document>                             |   |
|   | detection_result: None (clean)                             |   |
|   | warnings: []                                               |   |
|   +-----------------------------------------------------------+   |
|                                                                    |
|   Quarantined:                                                     |
|   +-----------------------------------------------------------+   |
|   | success: False                                             |   |
|   | action_taken: "quarantined"                                |   |
|   | document: <suspicious document>                            |   |
|   | detection_result:                                          |   |
|   |   is_poisoned: True                                        |   |
|   |   confidence: 0.85                                         |   |
|   |   threat_level: HIGH                                       |   |
|   | warnings: ["Detected authority mimicking pattern"]         |   |
|   +-----------------------------------------------------------+   |
|                                                                    |
|   Blocked:                                                         |
|   +-----------------------------------------------------------+   |
|   | success: False                                             |   |
|   | action_taken: "blocked_sanitization"                       |   |
|   | document: <original document>                              |   |
|   | sanitization_report:                                       |   |
|   |   is_blocked: True                                         |   |
|   |   matched_rules: ["block_script_tags"]                     |   |
|   | warnings: ["Blocked by rule: block_script_tags"]           |   |
|   +-----------------------------------------------------------+   |
|                                                                    |
+-------------------------------------------------------------------+
```

## Shield Statistics Dashboard

```
+-------------------------------------------------------------------+
|                    Shield Status Dashboard                         |
+-------------------------------------------------------------------+
|                                                                    |
|   +-----------------------------------------------------------+   |
|   |                    Overall Status                          |   |
|   +-----------------------------------------------------------+   |
|   |                                                            |   |
|   |   Defense Level: [====STRICT====]                          |   |
|   |                                                            |   |
|   |   +------------------+  +------------------+               |   |
|   |   | Processed        |  | Ingested         |               |   |
|   |   |      1,247       |  |      1,180       |               |   |
|   |   +------------------+  +------------------+               |   |
|   |                                                            |   |
|   |   +------------------+  +------------------+               |   |
|   |   | Blocked          |  | Quarantined      |               |   |
|   |   |        42        |  |        25        |               |   |
|   |   +------------------+  +------------------+               |   |
|   |                                                            |   |
|   +-----------------------------------------------------------+   |
|                                                                    |
|   +-----------------------------------------------------------+   |
|   |                    Monitoring                              |   |
|   +-----------------------------------------------------------+   |
|   |                                                            |   |
|   |   Detection Rate: [====|====] 5.4%                         |   |
|   |   Active Alerts:  12                                       |   |
|   |   Blocked Sources: 3                                       |   |
|   |                                                            |   |
|   +-----------------------------------------------------------+   |
|                                                                    |
|   +-----------------------------------------------------------+   |
|   |                    Quarantine Queue                        |   |
|   +-----------------------------------------------------------+   |
|   |                                                            |   |
|   |   Pending Review: 8                                        |   |
|   |   Under Review: 2                                          |   |
|   |   Expiring Soon: 3                                         |   |
|   |                                                            |   |
|   +-----------------------------------------------------------+   |
|                                                                    |
+-------------------------------------------------------------------+
```

## Complete Defense Flow

```
+-------------------------------------------------------------------+
|                    End-to-End Protection                           |
+-------------------------------------------------------------------+
|                                                                    |
|   External Sources                                                 |
|   +-------+  +-------+  +-------+                                  |
|   | API   |  | Upload|  | Crawl |                                  |
|   +-------+  +-------+  +-------+                                  |
|       |          |          |                                      |
|       +----------+----------+                                      |
|                  |                                                 |
|                  v                                                 |
|   +==========================================+                     |
|   ||           SECURITY MONITOR             ||                     |
|   ||  Rate limits | Anomaly | Blocking      ||                     |
|   +==========================================+                     |
|                  |                                                 |
|                  v                                                 |
|   +==========================================+                     |
|   ||            SANITIZER                   ||                     |
|   ||  Content | Embedding | Metadata        ||                     |
|   +==========================================+                     |
|                  |                                                 |
|                  v                                                 |
|   +==========================================+                     |
|   ||            DETECTOR                    ||                     |
|   ||  Perplexity | Similarity | Semantic    ||                     |
|   +==========================================+                     |
|                  |                                                 |
|            +-----+-----+                                           |
|            |           |                                           |
|         Clean      Suspicious                                      |
|            |           |                                           |
|            |           v                                           |
|            |    +==================+                               |
|            |    ||   QUARANTINE   ||                               |
|            |    ||  Review Queue  ||                               |
|            |    +==================+                               |
|            |           |                                           |
|            |      +----+----+                                      |
|            |      |         |                                      |
|            |   Approve   Reject                                    |
|            |      |         |                                      |
|            v      v         v                                      |
|   +------------------+  +--------+                                 |
|   | Knowledge Base   |  | Delete |                                 |
|   +------------------+  +--------+                                 |
|            |                                                       |
|            v                                                       |
|   +------------------+                                             |
|   | RAG Application  |                                             |
|   +------------------+                                             |
|                                                                    |
+-------------------------------------------------------------------+
```

## Summary

| Component | Function | Key Features |
|-----------|----------|--------------|
| Monitor | Boundary protection | Rate limiting, anomaly detection, alerts |
| Sanitizer | Input cleaning | Content filtering, embedding validation |
| Quarantine | Isolation | Review workflow, expiry management |
| Shield | Unified layer | Defense levels, comprehensive protection |

Active defense transforms RAG-Shield from a **detection tool** into a **protection system**:
- Block attacks at the boundary
- Clean suspicious content
- Isolate for review
- Provide visibility through dashboards

---

*RAG-Shield: Comprehensive protection for your RAG knowledge base.*
