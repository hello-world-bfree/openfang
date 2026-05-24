# DuckLake Shared Memory Substrate — Implementation Plan

## Executive Summary

This plan replaces OpenFang's current SQLite-backed KV shared memory with a two-layer architecture: **DuckLake** (Postgres catalog + Parquet) for structured, ACID-consistent, time-traveling memory, and **Lance** tables for high-performance vector retrieval. The system is fully self-describing — an agent with nothing but a catalog DSN can bootstrap complete knowledge of the memory topology.

---

## Architecture Overview

```
s3://bucket/
├── ducklake/                        ← DuckLake Parquet (structured memory)
│   ├── memory.facts/
│   ├── memory.events/
│   ├── memory.claims/
│   ├── memory.vector_stores/        ← registry of all Lance tables
│   ├── memory.search_log/
│   └── memory._manifest/           ← self-describing system catalog
├── lance/
│   ├── _meta_routing.lance/        ← coarse routing index (system-managed)
│   ├── agent_observations.lance/   ← emergent, agent-created
│   ├── codebase_chunks.lance/      ← emergent
│   └── ...
└── catalog.db                       ← Postgres catalog (ACID coordination)
```

**DuckLake owns**: structured metadata, lifecycle management, versioning, cross-agent coordination, the registry of what Lance tables exist.

**Lance owns**: embeddings, chunks, vector indexes, similarity search.

**The bridge**: A `memory.vector_stores` table in DuckLake that tracks all Lance tables, plus a `_meta_routing.lance` table that enables single-query store discovery.

---

## Phase 1: Foundation (Weeks 1–3)

### Goals
- Establish the DuckLake schema and self-describing manifest
- Implement the Rust actor model for DuckDB connection management
- Prove multi-writer ACID on structured tables
- Implement batched write buffer

### 1.1 Schema Definition

#### `memory.facts` — Stable Knowledge

```sql
CREATE TABLE memory.facts (
    fact_id         UUID DEFAULT gen_random_uuid(),
    agent_id        UUID NOT NULL,
    namespace       VARCHAR NOT NULL,
    key             VARCHAR NOT NULL,
    value           JSON NOT NULL,
    confidence      DOUBLE DEFAULT 1.0,
    supersedes      UUID,
    established_at  TIMESTAMPTZ NOT NULL,  -- when the fact was known
    persisted_at    TIMESTAMPTZ DEFAULT now(),  -- when it was written
    expires_at      TIMESTAMPTZ,
    PRIMARY KEY (fact_id)
);
```

**Critical distinction**: `established_at` vs `persisted_at`. Confidence decay operates on `established_at`; write batching introduces latency reflected in `persisted_at`. Conflating them introduces systematic decay calculation errors proportional to batch latency.

**Confidence decay is a view, not a mutation**:

```sql
CREATE VIEW memory.facts_decayed AS
SELECT *,
    confidence * exp(
        -0.05 * extract(epoch FROM now() - established_at) / 3600
    ) AS effective_confidence
FROM memory.facts
WHERE expires_at IS NULL OR expires_at > now();
```

**Embedding strategy for facts**: Do not embed the full row. Embed `(namespace || ' ' || key || ' ' || value)` as the retrieval unit. Embed separately without namespace for cross-namespace "what do we know about X?" queries. Never encode confidence into the embedding — surface it as a metadata filter in hybrid queries.

#### `memory.events` — Episodic Observations

```sql
CREATE TABLE memory.events (
    event_id      UUID NOT NULL,       -- UUIDv7: time-ordered, collision-resistant
    agent_id      UUID NOT NULL,
    event_type    VARCHAR NOT NULL,    -- 'tool_result', 'user_interaction', 'inference',
                                      -- 'error', 'ingest_request', 'system.bootstrap'
    session_id    UUID,
    payload       JSON NOT NULL,
    tags          VARCHAR[],
    occurred_at   TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (event_id)
);
```

**UUIDv7 rationale**: Embeds millisecond timestamp in high bits → naturally sortable by creation time. DuckDB/Parquet row group min-max statistics exploit this for pruning. Acts as both primary key and temporal cursor for the ingest queue pattern.

**Chunking for embedding**: Events alone are often underspecified. Use sliding-window context chunks — embed the event with its N preceding events (window of 3–5) as context. Store the embedding against the anchor event's UUIDv7. Also maintain session-level summary embeddings for coarser retrieval.

#### `memory.claims` — Multi-Agent Consensus

```sql
CREATE TABLE memory.claims (
    claim_id        UUID DEFAULT gen_random_uuid(),
    source_agent    UUID NOT NULL,
    proposition     VARCHAR NOT NULL,
    evidence        JSON,
    confidence      DOUBLE NOT NULL,
    corroborated    INTEGER DEFAULT 0,
    contested       INTEGER DEFAULT 0,
    status          VARCHAR DEFAULT 'pending',  -- FSM defined below
    claim_type      VARCHAR DEFAULT 'assertion',
    disputed_at     TIMESTAMPTZ,
    resolved_at     TIMESTAMPTZ,
    ttl_hours       INTEGER DEFAULT 72,
    created_at      TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (claim_id)
);
```

**Claims state machine**:

```
pending → corroborated    (corroborated >= threshold)
pending → disputed        (first contestation, immediate)
disputed → confirmed_contested  (corroborations win despite contests)
disputed → refuted        (contestations reach majority)
disputed → unresolved     (TTL expires with no resolution → escalate)
any → retracted           (source agent withdraws, or agent dies)
```

**Corroboration thresholds** (configurable per `claim_type`):

| Claim Type | Corroboration Threshold | Contestation Behavior |
|---|---|---|
| `assertion` | 2 independent corroborators | Single contest → `disputed` |
| `high_confidence` | 3+ corroborators | 2 contests to reopen |
| `contested` | Majority of active agents | Falls back to escalation |

**On consensus**: A claim that achieves `corroborated` status gets written to `memory.facts` with initial confidence 0.90, decaying over time unless refreshed. Claims are the process; facts are the output.

**Embedding for claims**: Embed claim text with stance prefix for geometric separation: `"CONTESTED: [text]"` vs `"CORROBORATED: [text]"`. This lets the routing table discriminate them.

#### `memory.vector_stores` — Lance Table Registry

```sql
CREATE TABLE memory.vector_stores (
    store_id        UUID DEFAULT gen_random_uuid(),
    name            VARCHAR NOT NULL UNIQUE,
    role            VARCHAR DEFAULT 'data',    -- 'data', 'routing', 'archive', 'scratch'
    category        VARCHAR NOT NULL,          -- 'episodic', 'semantic', 'domain', 'system'
    lance_uri       VARCHAR NOT NULL,
    schema_hint     JSON,
    embedding_model VARCHAR NOT NULL,
    embedding_dim   INTEGER NOT NULL,
    index_type      VARCHAR DEFAULT 'IVF_PQ',
    status          VARCHAR DEFAULT 'active',  -- 'active', 'archived', 'compacting', 'tombstoned'
    writer_agent    UUID,                      -- enforced single writer
    writer_lease    JSON,                      -- {agent_id, expires_at, heartbeat_at}
    created_by      UUID NOT NULL,
    row_count       BIGINT DEFAULT 0,
    last_indexed    TIMESTAMPTZ,
    last_sampled    TIMESTAMPTZ,
    sample_size     INTEGER DEFAULT 0,
    on_owner_death  VARCHAR DEFAULT 'seal',    -- 'transfer', 'seal', 'orphan'
    archived_at     TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT now(),
    meta            JSON,
    PRIMARY KEY (store_id)
);
```

#### `memory._manifest` — Self-Describing System Catalog

```sql
CREATE TABLE memory._manifest (
    component       VARCHAR PRIMARY KEY,
    kind            VARCHAR NOT NULL,     -- 'ducklake_table', 'lance_registry', 'view'
    description     VARCHAR,
    query_pattern   VARCHAR,
    schema_version  INTEGER DEFAULT 1,
    created_at      TIMESTAMPTZ DEFAULT now()
);

INSERT INTO memory._manifest VALUES
    ('facts',         'ducklake_table',  'Stable knowledge with confidence decay. Immutable snapshots, decay at read-time.',
     'SELECT * FROM memory.facts_decayed WHERE namespace = $1 AND effective_confidence > 0.1', 1),
    ('events',        'ducklake_table',  'Episodic observations. Append-only, UUIDv7-ordered. Also serves as ingest queue.',
     'SELECT * FROM memory.events WHERE event_type = $1 ORDER BY event_id DESC LIMIT $2', 1),
    ('claims',        'ducklake_table',  'Multi-agent consensus. Query pending claims to corroborate or contest.',
     'SELECT * FROM memory.claims WHERE status = ''pending'' AND source_agent != $1', 1),
    ('vector_stores', 'lance_registry',  'Registry of all Lance tables. Includes routing, data, archive, and scratch stores.',
     'SELECT * FROM memory.vector_stores WHERE status = ''active'' ORDER BY role', 1),
    ('search_log',    'ducklake_table',  'Observability. Every vector search recorded with routing decisions and latency.',
     'SELECT * FROM memory.search_log WHERE agent_id = $1 ORDER BY searched_at DESC', 1),
    ('facts_decayed', 'view',            'Read-time confidence decay over memory.facts.',
     'SELECT * FROM memory.facts_decayed WHERE namespace = $1', 1);
```

#### `memory.search_log` — Observability

```sql
CREATE TABLE memory.search_log (
    search_id       UUID DEFAULT gen_random_uuid(),
    agent_id        UUID NOT NULL,
    query_text      VARCHAR,
    routing_hits    JSON,              -- [{store, distance}]
    stores_searched VARCHAR[],
    results_count   INTEGER,
    best_distance   FLOAT,
    latency_ms      INTEGER,
    searched_at     TIMESTAMPTZ DEFAULT now()
);
```

### 1.2 Rust Actor Model

**Core constraint**: `duckdb::Connection` is neither `Send` nor `Sync`. The actor model resolves this by confining the connection to a dedicated OS thread and exposing a cloneable, `Send + Sync` handle.

```rust
// Memory actor message types
enum MemoryCmd {
    Write { record: MemoryRecord, reply: oneshot::Sender<Result<(), MemoryError>> },
    Query { filter: MemoryFilter, reply: oneshot::Sender<Result<Vec<Row>, MemoryError>> },
    Flush { reply: oneshot::Sender<Result<FlushStats, MemoryError>> },
    Shutdown,
}

// The actor owns the connection — lives on one std::thread
struct DuckLakeActor {
    conn: duckdb::Connection,
    buffer: WriteBuffer,
    rx: mpsc::Receiver<MemoryCmd>,
}

// The handle is cheap to clone, Send+Sync, shared across agents
#[derive(Clone)]
pub struct DuckLakeHandle {
    tx: mpsc::Sender<MemoryCmd>,
}
```

**Key rules**:
- The connection never crosses a thread boundary. Spawn via `std::thread::spawn`, not `tokio::spawn`.
- Use `std::sync::Mutex` (not `tokio::sync::Mutex`) for the write guard inside the actor — no `.await` while holding it.
- Use `tokio::task::spawn_blocking` only for ad-hoc one-shot queries, not for long-lived write flows.
- One actor per DuckDB database file. Share handles via `Arc<DuckLakeHandle>`.

### 1.3 Write Buffer

**The buffer owns its staged data; flush borrows the connection.**

```rust
pub struct WriteBuffer {
    staged_facts: Vec<Fact>,
    staged_events: Vec<Event>,
    threshold: usize,       // flush at N records
    max_age: Duration,      // flush after M seconds regardless
    last_flush: Instant,
}
```

**Flush-on-drop**: Expose an explicit `async fn shutdown(&self) -> Result<()>` for graceful teardown. The `Drop` impl on `DuckLakeHandle` sends a best-effort `Shutdown` signal when the last handle is dropped. The actor's run loop handles `Shutdown` by flushing before exiting.

**DuckDB Appender for bulk writes**: Use `conn.appender("table")` instead of INSERT loops — significantly faster for batch operations.

### 1.4 Error Handling

Single domain error enum at the adapter boundary using `thiserror`:

```rust
#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("DuckDB error: {0}")]
    DuckDb(#[from] duckdb::Error),
    #[error("Lance error: {0}")]
    Lance(#[from] lance::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Registry inconsistency: {msg}")]
    RegistryInconsistency { msg: String },
    #[error("Actor channel closed")]
    ActorDead,
    #[error("Lease expired for store {store_id}")]
    LeaseExpired { store_id: String, agent_id: String },
    #[error("Embedding model mismatch: expected {expected}, got {got}")]
    ModelMismatch { expected: String, got: String },
}
```

Place this in a shared `memory-types` crate in the workspace. Each downstream crate uses `#[from] MemoryError` in its own error enum.

### 1.5 Phase 1 Validation

- [ ] Two agents writing concurrent facts via separate `DuckLakeHandle` instances — both visible after flush
- [ ] Time-travel query returns correct state at arbitrary past timestamp
- [ ] Write buffer batches correctly: no Parquet file per individual write
- [ ] `_manifest` is queryable and accurately describes all tables
- [ ] Confidence decay view returns different results for same data at different query times

---

## Phase 2: Vector Layer (Weeks 4–6)

### Goals
- Integrate Lance tables via DuckDB lance extension
- Implement the vector store registry and lifecycle
- Build the routing table with centroid-based sampling
- Implement two-phase search with observability

### 2.1 Single Writer Enforcement

Each Lance table has exactly one writer agent. Other agents submit ingest requests through the events table:

```sql
-- Other agents post ingest requests (multi-writer safe via DuckLake)
INSERT INTO memory.events (event_id, agent_id, event_type, payload)
VALUES (uuid_v7(), $agent, 'ingest_request', json_object(
    'target_store', $store_name,
    'content', $content,
    'embedding', $embedding_vec
));

-- The store's writer drains these using UUIDv7 as cursor
SELECT * FROM memory.events
WHERE event_type = 'ingest_request'
  AND payload->>'target_store' = $my_store
  AND event_id > $last_processed_id
ORDER BY event_id;
```

**Ingest event status FSM**: `pending → claimed → ingested | failed`. Add `claimed_by` and `claimed_at` to detect stuck claims and reassign without full-table scans.

**Lease management** uses `AtomicBool` fast path → `Mutex<LeaseState>` for mutation → `watch` channel for broadcasting changes. RAII `LeaseGuard` guarantees release on any exit path. Heartbeat interval < lease TTL (recommend 2–3× write-batch interval for TTL).

### 2.2 Embedding Model Enforcement

Hard constraint at the routing layer in the POC:

```rust
pub fn create_vector_store(&self, embedding_model: &str, /* ... */) -> Result<VectorStore> {
    let routing_model = self.get_routing_model()?;
    if embedding_model != routing_model {
        return Err(MemoryError::ModelMismatch {
            expected: routing_model, got: embedding_model.into()
        });
    }
    // ...
}
```

**Future**: When multiple models are needed, introduce `routing_group` — one routing table per embedding model family. Don't build this in the POC.

### 2.3 Store Creation Protocol

Optimistic lock via registry:

```sql
INSERT INTO memory.vector_stores (name, category, lance_uri, created_by, writer_agent, ...)
VALUES ($name, $category, $uri, $agent, $agent, ...)
ON CONFLICT (name) DO NOTHING
RETURNING store_id;
```

If `RETURNING` yields nothing, another agent won the race. The loser reads the winner's `writer_agent` and routes through the events table.

### 2.4 Routing Table — Centroid-Based Sampling

**Do not use `USING SAMPLE` (reservoir sampling)**. Embedding spaces are clustered; uniform random sampling over-represents dense clusters and misses sparse but important ones.

Instead, extract cluster representatives:

```sql
WITH partitioned AS (
    SELECT embedding, chunk_id, content,
           ntile(100) OVER (ORDER BY chunk_id) AS partition_id
    FROM '{lance_uri}'
),
centroids AS (
    SELECT
        partition_id,
        list_avg(list(embedding)) AS centroid
    FROM partitioned
    GROUP BY partition_id
)
-- Find the nearest actual chunk to each centroid
-- Insert these as representatives into _meta_routing.lance
```

**Sample size heuristic**: `sqrt(n)` capped at 1000, floor at 50. For 10k chunks → ~100 samples, for 100k → ~316, caps at 1000.

**Resample triggers**: When store row count exceeds `(last_sample_size)^2 * 4`, or on explicit request. Track `last_sampled` and `sample_size` in the registry.

**Circuit breaker**: If routing confidence (best distance) is below threshold, fall back to broadcast search across all active stores. Expensive but correct during transition periods.

### 2.5 Two-Phase Search

```
Phase 1 (Route):  Search _meta_routing.lance → identify top N relevant stores
Phase 2 (Retrieve): Targeted lance_scan on each relevant store → merge top-k
```

**Hybrid search integration** (pre-filter in DuckLake, vector search in Lance):

1. **Pre-filter in DuckLake**: Narrow candidate space by confidence, recency, agent trust, namespace. Generate a set of candidate IDs.
2. **Vector search in Lance**: Pass candidate IDs as metadata filter to Lance ANN scan. Pre-filtering consistently outperforms post-filtering because post-filtering can decimate top-K.
3. **RRF fusion**: If combining vector + keyword results, use Reciprocal Rank Fusion: `RRF(d) = Σ 1/(k + rank_i(d))` with k=60.
4. **Optional cross-encoder reranking**: For low-specificity exploratory queries only. Retrieve top-50 from Lance+RRF, rerank to top-5. Skip for targeted lookups.

**Log every search** to `memory.search_log` — routing decisions, stores hit, latency, result count.

### 2.6 Phase 2 Validation

- [ ] Agent creates a Lance store, registers it, writes chunks, indexes — all visible to other agents
- [ ] Routing table correctly identifies relevant stores for a query (> 80% precision on synthetic test set)
- [ ] Two-phase search returns results within 200ms for < 10 active stores
- [ ] Ingest request queue drains correctly under concurrent submitters
- [ ] Store with `writer_agent` gone is detected via lease expiry and sealed
- [ ] Search log captures all search operations with accurate latency

---

## Phase 3: Consensus & Lifecycle (Weeks 7–9)

### Goals
- Implement the claims corroboration protocol
- Build agent lifecycle management (death, lease transfer, GC)
- Add the cold-start bootstrap protocol
- Implement store archival and reaping

### 3.1 Claims Protocol

**Corroboration flow**:
1. Agent A writes a claim with `status = 'pending'`
2. Other agents poll: `SELECT * FROM memory.claims WHERE status = 'pending' AND source_agent != $me`
3. Corroborating agent writes an event: `{event_type: 'corroboration', payload: {claim_id, stance: 'agree'}}`
4. A watcher process (or the claiming agent) updates `corroborated` count
5. When threshold is met: `status → 'corroborated'`, write resulting fact to `memory.facts`

**Contestation**: First contest immediately flips status to `disputed` and sets `disputed_at`. A disputed fact is capped at effective confidence 0.4 regardless of corroboration count. Agents querying disputed facts receive a flag.

**Deadlock prevention**: Open disputes have a TTL (default 72 hours). Unresolved disputes escalate to `unresolved` status and emit a `claim.escalation` event.

**Agent death**: Pending claims from dead agents are automatically retracted unless independently corroborated.

### 3.2 Agent Lifecycle

**On agent death detection** (lease heartbeat failure):

1. **Owned Lance stores**: Apply `on_owner_death` policy from registry:
   - `seal` (default): Mark store read-only, emit `store.sealed` event, await orchestrator reassignment
   - `transfer`: Elect new writer from agents that have recently submitted ingest requests to this store
   - `orphan`: Mark `status = 'orphaned'`, searchable but not writable
2. **Pending claims**: Retract all `pending` claims from dead agent (`status → 'retracted'`)
3. **Pending ingest events**: Mark as `orphaned`, surface for reassignment
4. **Write buffer**: Agents should periodically checkpoint buffer state to a `memory.agent_state` DuckLake table. On death, recovery process can replay from last checkpoint.

### 3.3 Cold-Start Bootstrap

```rust
pub fn bootstrap(catalog_dsn: &str, data_path: &str) -> Result<Self> {
    let mem = Self::attach(catalog_dsn, data_path)?;

    if !mem.schema_exists("memory")? {
        // First agent — initialize everything
        mem.initialize_schema()?;        // Create all tables
        mem.seed_manifest()?;            // Populate _manifest
        mem.create_routing_store()?;     // Create _meta_routing.lance (empty)
        mem.emit_event("system.bootstrap", json!({
            "schema_version": 1,
            "bootstrap_agent": agent_id,
        }))?;
    }

    // Race condition: two agents both see empty system
    // Handled by INSERT ... ON CONFLICT DO NOTHING on _manifest
    // Loser detects winner's entries and transitions to normal join

    mem.topology = mem.discover_topology()?;
    Ok(mem)
}
```

**Grace period**: While routing table is empty (cold start), fall back to brute-force search across all stores. Log this as a routing miss in `search_log` to track convergence.

### 3.4 Store Reaping

**Policy**: `scratch` stores older than configurable max age (default 24h) are tombstoned. Tombstoned stores older than 7 days are purged (Lance files deleted from bucket).

```sql
-- Tombstone stale scratch stores
UPDATE memory.vector_stores
SET status = 'tombstoned', archived_at = now()
WHERE status = 'scratch'
  AND created_at < now() - INTERVAL '24 hours';

-- Purge old tombstones (separate job, runs daily)
-- Delete Lance files from bucket, then:
DELETE FROM memory.vector_stores
WHERE status = 'tombstoned'
  AND archived_at < now() - INTERVAL '7 days';
```

DuckLake time travel means tombstoned stores remain visible in historical snapshots until Parquet compaction.

### 3.5 Phase 3 Validation

- [ ] Claim achieves consensus → written to facts with correct confidence
- [ ] Contested claim is capped at 0.4 effective confidence
- [ ] Dead agent's stores are sealed and claims retracted
- [ ] Cold start with two simultaneous agents resolves cleanly (no duplicate schemas)
- [ ] Reaper tombstones scratch stores without affecting active stores
- [ ] Full bootstrap from empty system to functional search in < 5 seconds

---

## Phase 4: Hardening (Weeks 10–12)

### Goals
- Implement DuckLake Parquet compaction
- Build embedding drift detection
- Add retrieval quality measurement (RAGAS-style)
- Prepare for embedding model upgrades

### 4.1 Compaction

Batched writes produce many small Parquet files over time. Implement periodic compaction:

```sql
-- DuckLake compaction (merge small files into larger ones)
-- Run as a scheduled job, not on the hot path
CALL ducklake_compact('openfang_mem', 'memory', 'facts');
CALL ducklake_compact('openfang_mem', 'memory', 'events');
```

**Archive processed events**: Move `ingested` ingest-request events to a cold partition on a schedule. The events table should stay small and hot.

### 4.2 Embedding Drift Detection

Periodically compute centroid drift per Lance store:

```sql
-- Compare recent centroid to historical centroid
WITH recent AS (
    SELECT list_avg(list(embedding)) AS centroid
    FROM '{lance_uri}'
    WHERE created_at > now() - INTERVAL '7 days'
),
historical AS (
    SELECT list_avg(list(embedding)) AS centroid
    FROM '{lance_uri}'
    WHERE created_at <= now() - INTERVAL '7 days'
)
SELECT cosine_distance(recent.centroid, historical.centroid) AS drift
FROM recent, historical;
```

Store drift metrics in `vector_stores.meta`. Alert when drift exceeds threshold (e.g., cosine distance > 0.15).

**Reference corpus**: Maintain ~500 canonical facts/events with known query-retrieval pairs. Run this suite weekly as a regression test for retrieval quality.

### 4.3 Retrieval Quality Metrics

**Retrieval-intrinsic** (sampled, not on every query):
- Context Precision: Are top-K results relevant?
- Context Recall: Did retrieval miss known relevant facts? (LLM-as-judge)
- Faithfulness: Does generated content match retrieved content? (Critical for claims)

**Multi-agent agreement**:
- Claim convergence rate: Do agents retrieve the same top-K for identical queries?
- Corroboration lag: Time from evidence arrival to claim consensus
- Epistemic staleness: Fraction of retrieved facts below confidence threshold

**Operational**:
- Routing accuracy: % of searches where routing table sent agent to a store returning 0 results
- Ingest queue depth per store (leading indicator of retrieval staleness)
- Index freshness: Time since last Lance index rebuild

### 4.4 Model Upgrade Strategy (Preparation)

**Recommended approach**: Dual-index transition.

1. Add `embedding_model_version` column to `vector_stores` now
2. When upgrading: new writes go to both old-model and new-model Lance tables
3. Query both, fuse with RRF (treat as two separate stores)
4. Backfill old data into new-model tables asynchronously
5. When quality metrics confirm parity, deprecate old tables
6. DuckLake registry tracks transition state per table

**Do not build this in the POC.** The column and the architectural awareness are sufficient preparation.

---

## Trait Design (Cross-Phase)

Split by access pattern for testing and composability:

```rust
#[async_trait]
pub trait MemoryReader: Send + Sync {
    async fn query_structured(&self, filter: &MemoryFilter) -> Result<Vec<MemoryRecord>>;
    async fn vector_search(&self, embedding: &[f32], top_k: usize) -> Result<Vec<ScoredRecord>>;
    async fn facts_as_of(&self, ts: &str, ns: Option<&str>) -> Result<Vec<Fact>>;
}

#[async_trait]
pub trait MemoryWriter: Send + Sync {
    async fn write(&self, record: MemoryRecord) -> Result<()>;
    async fn write_batch(&self, records: Vec<MemoryRecord>) -> Result<BatchStats>;
    async fn flush(&self) -> Result<FlushStats>;
}

#[async_trait]
pub trait MemoryStore: MemoryReader + MemoryWriter {
    async fn health_check(&self) -> Result<StoreHealth>;
    async fn bootstrap(&self) -> Result<MemoryTopology>;
    fn store_id(&self) -> StoreId;
}
```

Agents depend on `Arc<dyn MemoryStore>`, never on the concrete DuckLake type.

---

## Testing Strategy (Cross-Phase)

### Layer 1 — Unit Tests (No DuckDB)

In-memory `MemoryStore` implementation for testing agent logic:

```rust
struct InMemoryStore { records: Arc<Mutex<Vec<MemoryRecord>>> }
```

### Layer 2 — Integration Tests (Embedded DuckDB, No Extensions)

Test write buffer mechanics, flush semantics, error propagation against vanilla DuckDB. No Postgres, no DuckLake extension required.

### Layer 3 — Extension Integration Tests (Feature-Gated)

```toml
[features]
ducklake-integration = []
lance-integration = []
```

Run in CI only on machines with extensions installed. Test actual DuckLake attach, Lance table creation, vector search.

### Layer 4 — Property-Based Tests (`proptest`)

- Buffer flush is idempotent
- Lease manager never grants two simultaneous leases
- UUIDv7 ordering matches `occurred_at` ordering (within clock precision)
- Confidence decay is monotonically decreasing over time

### Layer 5 — Fault Injection

`FaultInjectingStore` wrapper that returns predetermined errors on configured calls. Tests retry logic, circuit breakers, dead-actor detection.

---

## Workspace Crate Structure

```
openfang/
├── crates/
│   ├── memory-types/       ← MemoryError, MemoryRecord, Fact, Event, Claim, traits
│   ├── memory-ducklake/    ← DuckLakeActor, DuckLakeHandle, WriteBuffer, LeaseManager
│   ├── memory-lance/       ← VectorStore lifecycle, routing table, two-phase search
│   ├── memory-consensus/   ← Claims protocol, corroboration, contestation FSM
│   └── ... (existing crates)
```

`memory-types` is the leaf crate — no dependencies on DuckDB or Lance. Everything else depends on it. This enables the in-memory test implementations without pulling in FFI dependencies.

---

## Risk Mitigations Summary

| Risk | Mitigation | Phase |
|---|---|---|
| Lance write conflicts | Single writer per store, enforced via registry | 2 |
| DuckLake/Lance consistency gap | Writer does both operations; events table as coordination queue | 2 |
| Parquet small-file proliferation | Write buffer with threshold + time-based flush | 1 |
| Routing table misses sparse clusters | Centroid extraction instead of reservoir sampling | 2 |
| Embedding model mismatch | Hard constraint at routing layer; single model enforced | 2 |
| Lance index staleness | Track `last_indexed` in registry; reindex on threshold | 2 |
| Agent death orphans stores | Lease TTL + `on_owner_death` policy | 3 |
| Table proliferation | Reaper with tombstone → purge lifecycle | 3 |
| Cold start race condition | `INSERT ... ON CONFLICT DO NOTHING` on manifest | 3 |
| Embedding drift | Centroid drift monitoring + reference corpus regression | 4 |
| Debugging retrieval failures | `search_log` table from day one | 2 |
| `established_at` vs `persisted_at` confusion | Separate columns, decay operates on `established_at` only | 1 |

---

## Success Criteria

**POC Complete** when:
1. Two agents write concurrent structured memory with ACID guarantees
2. An agent can bootstrap from empty system to functional search in one `ATTACH`
3. Time-travel query reconstructs historical agent worldview
4. Routing table correctly identifies relevant stores (> 80% precision)
5. End-to-end search latency < 200ms for typical queries (< 10 stores)
6. Dead agent's resources are automatically reclaimed
7. `memory._manifest` accurately describes the full system topology
8. All searches are logged with sufficient detail to diagnose routing failures
