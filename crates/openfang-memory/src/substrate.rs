//! MemorySubstrate: unified implementation of the `Memory` trait.
//!
//! Composes the structured store, semantic store, knowledge store,
//! session store, and consolidation engine behind a single async API.

use crate::consolidation::ConsolidationEngine;
use crate::knowledge::KnowledgeStore;
use crate::migration::run_migrations;
use crate::semantic::SemanticStore;
use crate::session::{Session, SessionStore};
use crate::structured::StructuredStore;
use crate::usage::UsageStore;

use async_trait::async_trait;
use openfang_types::agent::{AgentEntry, AgentId, SessionId};
use openfang_types::config::MemoryConfig;
use openfang_types::error::{OpenFangError, OpenFangResult};
use openfang_types::memory::{
    ConsolidationReport, Entity, ExportFormat, GraphMatch, GraphPattern, ImportReport, Memory,
    MemoryFilter, MemoryFragment, MemoryId, MemorySource, Relation,
};
use rusqlite::Connection;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

/// The unified memory substrate. Implements the `Memory` trait by delegating
/// to specialized stores backed by a shared SQLite connection.
pub struct MemorySubstrate {
    conn: Arc<Mutex<Connection>>,
    structured: StructuredStore,
    semantic: SemanticStore,
    knowledge: KnowledgeStore,
    sessions: SessionStore,
    consolidation: ConsolidationEngine,
    usage: UsageStore,
}

impl MemorySubstrate {
    /// Open or create a memory substrate at the given database path.
    ///
    /// When `memory_config.backend == "http"` and `http_url`/`http_token_env` are set,
    /// the semantic store routes `remember`/`recall` to the memory-api gateway.
    /// All other stores (KV, knowledge graph, sessions) remain local SQLite.
    pub fn open(
        db_path: &Path,
        decay_rate: f32,
        memory_config: &MemoryConfig,
    ) -> OpenFangResult<Self> {
        let conn = Connection::open(db_path).map_err(|e| OpenFangError::Memory(e.to_string()))?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        run_migrations(&conn).map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let shared = Arc::new(Mutex::new(conn));

        let semantic = Self::create_semantic_store(Arc::clone(&shared), memory_config);

        Ok(Self {
            conn: Arc::clone(&shared),
            structured: StructuredStore::new(Arc::clone(&shared)),
            semantic,
            knowledge: KnowledgeStore::new(Arc::clone(&shared)),
            sessions: SessionStore::new(Arc::clone(&shared)),
            usage: UsageStore::new(Arc::clone(&shared)),
            consolidation: ConsolidationEngine::new(shared, decay_rate),
        })
    }

    /// Create the semantic store, optionally with HTTP backend.
    fn create_semantic_store(
        conn: Arc<Mutex<Connection>>,
        memory_config: &MemoryConfig,
    ) -> SemanticStore {
        #[cfg(feature = "http-memory")]
        if memory_config.backend == "http" {
            if let (Some(url), Some(token_env)) =
                (&memory_config.http_url, &memory_config.http_token_env)
            {
                match crate::http_client::MemoryApiClient::new(url, token_env) {
                    Ok(client) => {
                        // Best-effort health check on startup
                        match client.health_check() {
                            Ok(()) => info!(url = %url, "HTTP memory backend connected"),
                            Err(e) => {
                                warn!(url = %url, error = %e, "HTTP memory backend health check failed, will retry on use")
                            }
                        }
                        return SemanticStore::new_with_http(conn, client);
                    }
                    Err(e) => {
                        warn!(error = %e, "Failed to create HTTP memory client, falling back to SQLite");
                    }
                }
            } else {
                warn!("backend=http but http_url/http_token_env not set, falling back to SQLite");
            }
        }

        #[cfg(not(feature = "http-memory"))]
        let _ = memory_config;

        SemanticStore::new(conn)
    }

    /// Create an in-memory substrate (for testing). Always uses SQLite backend.
    pub fn open_in_memory(decay_rate: f32) -> OpenFangResult<Self> {
        let conn =
            Connection::open_in_memory().map_err(|e| OpenFangError::Memory(e.to_string()))?;
        run_migrations(&conn).map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let shared = Arc::new(Mutex::new(conn));

        Ok(Self {
            conn: Arc::clone(&shared),
            structured: StructuredStore::new(Arc::clone(&shared)),
            semantic: SemanticStore::new(Arc::clone(&shared)),
            knowledge: KnowledgeStore::new(Arc::clone(&shared)),
            sessions: SessionStore::new(Arc::clone(&shared)),
            usage: UsageStore::new(Arc::clone(&shared)),
            consolidation: ConsolidationEngine::new(shared, decay_rate),
        })
    }

    /// Get a reference to the usage store.
    pub fn usage(&self) -> &UsageStore {
        &self.usage
    }

    /// Get the shared database connection (for constructing stores from outside).
    pub fn usage_conn(&self) -> Arc<Mutex<Connection>> {
        Arc::clone(&self.conn)
    }

    /// Save an agent entry to persistent storage.
    pub fn save_agent(&self, entry: &AgentEntry) -> OpenFangResult<()> {
        self.structured.save_agent(entry)
    }

    /// Load an agent entry from persistent storage.
    pub fn load_agent(&self, agent_id: AgentId) -> OpenFangResult<Option<AgentEntry>> {
        self.structured.load_agent(agent_id)
    }

    /// Remove an agent from persistent storage and cascade-delete sessions.
    pub fn remove_agent(&self, agent_id: AgentId) -> OpenFangResult<()> {
        // Delete associated sessions first
        let _ = self.sessions.delete_agent_sessions(agent_id);
        self.structured.remove_agent(agent_id)
    }

    /// Load all agent entries from persistent storage.
    pub fn load_all_agents(&self) -> OpenFangResult<Vec<AgentEntry>> {
        self.structured.load_all_agents()
    }

    /// List all saved agents.
    pub fn list_agents(&self) -> OpenFangResult<Vec<(String, String, String)>> {
        self.structured.list_agents()
    }

    /// Synchronous get from the structured store (for kernel handle use).
    pub fn structured_get(
        &self,
        agent_id: AgentId,
        key: &str,
    ) -> OpenFangResult<Option<serde_json::Value>> {
        self.structured.get(agent_id, key)
    }

    /// List all KV pairs for an agent.
    pub fn list_kv(&self, agent_id: AgentId) -> OpenFangResult<Vec<(String, serde_json::Value)>> {
        self.structured.list_kv(agent_id)
    }

    /// Delete a KV entry for an agent.
    pub fn structured_delete(&self, agent_id: AgentId, key: &str) -> OpenFangResult<()> {
        self.structured.delete(agent_id, key)
    }

    /// Synchronous set in the structured store (for kernel handle use).
    pub fn structured_set(
        &self,
        agent_id: AgentId,
        key: &str,
        value: serde_json::Value,
    ) -> OpenFangResult<()> {
        self.structured.set(agent_id, key, value)
    }

    /// Get a session by ID.
    pub fn get_session(&self, session_id: SessionId) -> OpenFangResult<Option<Session>> {
        self.sessions.get_session(session_id)
    }

    /// Save a session.
    pub fn save_session(&self, session: &Session) -> OpenFangResult<()> {
        self.sessions.save_session(session)
    }

    /// Save a session asynchronously — runs the SQLite write in a blocking
    /// thread so the tokio runtime stays responsive.
    pub async fn save_session_async(&self, session: &Session) -> OpenFangResult<()> {
        let sessions = self.sessions.clone();
        let session = session.clone();
        tokio::task::spawn_blocking(move || sessions.save_session(&session))
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    /// Create a new empty session for an agent.
    pub fn create_session(&self, agent_id: AgentId) -> OpenFangResult<Session> {
        self.sessions.create_session(agent_id)
    }

    /// List all sessions with metadata.
    pub fn list_sessions(&self) -> OpenFangResult<Vec<serde_json::Value>> {
        self.sessions.list_sessions()
    }

    /// Delete a session by ID.
    pub fn delete_session(&self, session_id: SessionId) -> OpenFangResult<()> {
        self.sessions.delete_session(session_id)
    }

    /// Delete all sessions belonging to an agent.
    pub fn delete_agent_sessions(&self, agent_id: AgentId) -> OpenFangResult<()> {
        self.sessions.delete_agent_sessions(agent_id)
    }

    /// Delete the canonical (cross-channel) session for an agent.
    pub fn delete_canonical_session(&self, agent_id: AgentId) -> OpenFangResult<()> {
        self.sessions.delete_canonical_session(agent_id)
    }

    /// Set or clear a session label.
    pub fn set_session_label(
        &self,
        session_id: SessionId,
        label: Option<&str>,
    ) -> OpenFangResult<()> {
        self.sessions.set_session_label(session_id, label)
    }

    /// Find a session by label for a given agent.
    pub fn find_session_by_label(
        &self,
        agent_id: AgentId,
        label: &str,
    ) -> OpenFangResult<Option<Session>> {
        self.sessions.find_session_by_label(agent_id, label)
    }

    /// List all sessions for a specific agent.
    pub fn list_agent_sessions(&self, agent_id: AgentId) -> OpenFangResult<Vec<serde_json::Value>> {
        self.sessions.list_agent_sessions(agent_id)
    }

    /// Create a new session with an optional label.
    pub fn create_session_with_label(
        &self,
        agent_id: AgentId,
        label: Option<&str>,
    ) -> OpenFangResult<Session> {
        self.sessions.create_session_with_label(agent_id, label)
    }

    /// Load canonical session context for cross-channel memory.
    ///
    /// Returns the compacted summary (if any) and recent messages from the
    /// agent's persistent canonical session.
    pub fn canonical_context(
        &self,
        agent_id: AgentId,
        window_size: Option<usize>,
    ) -> OpenFangResult<(Option<String>, Vec<openfang_types::message::Message>)> {
        self.sessions.canonical_context(agent_id, window_size)
    }

    /// Store an LLM-generated summary, replacing older messages with the kept subset.
    ///
    /// Used by the compactor to replace text-truncation compaction with an
    /// LLM-generated summary of older conversation history.
    pub fn store_llm_summary(
        &self,
        agent_id: AgentId,
        summary: &str,
        kept_messages: Vec<openfang_types::message::Message>,
    ) -> OpenFangResult<()> {
        self.sessions
            .store_llm_summary(agent_id, summary, kept_messages)
    }

    /// Write a human-readable JSONL mirror of a session to disk.
    ///
    /// Best-effort — errors are returned but should be logged,
    /// never affecting the primary SQLite store.
    pub fn write_jsonl_mirror(
        &self,
        session: &Session,
        sessions_dir: &Path,
    ) -> Result<(), std::io::Error> {
        self.sessions.write_jsonl_mirror(session, sessions_dir)
    }

    /// Append messages to the agent's canonical session for cross-channel persistence.
    pub fn append_canonical(
        &self,
        agent_id: AgentId,
        messages: &[openfang_types::message::Message],
        compaction_threshold: Option<usize>,
    ) -> OpenFangResult<()> {
        self.sessions
            .append_canonical(agent_id, messages, compaction_threshold)?;
        Ok(())
    }

    // -----------------------------------------------------------------
    // Paired devices persistence
    // -----------------------------------------------------------------

    /// Load all paired devices from the database.
    pub fn load_paired_devices(&self) -> OpenFangResult<Vec<serde_json::Value>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT device_id, display_name, platform, paired_at, last_seen, push_token FROM paired_devices"
        ).map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let rows = stmt
            .query_map([], |row| {
                Ok(serde_json::json!({
                    "device_id": row.get::<_, String>(0)?,
                    "display_name": row.get::<_, String>(1)?,
                    "platform": row.get::<_, String>(2)?,
                    "paired_at": row.get::<_, String>(3)?,
                    "last_seen": row.get::<_, String>(4)?,
                    "push_token": row.get::<_, Option<String>>(5)?,
                }))
            })
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let mut devices = Vec::new();
        for row in rows {
            devices.push(row.map_err(|e| OpenFangError::Memory(e.to_string()))?);
        }
        Ok(devices)
    }

    /// Save a paired device to the database (insert or replace).
    pub fn save_paired_device(
        &self,
        device_id: &str,
        display_name: &str,
        platform: &str,
        paired_at: &str,
        last_seen: &str,
        push_token: Option<&str>,
    ) -> OpenFangResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        conn.execute(
            "INSERT OR REPLACE INTO paired_devices (device_id, display_name, platform, paired_at, last_seen, push_token) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![device_id, display_name, platform, paired_at, last_seen, push_token],
        ).map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(())
    }

    /// Remove a paired device from the database.
    pub fn remove_paired_device(&self, device_id: &str) -> OpenFangResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        conn.execute(
            "DELETE FROM paired_devices WHERE device_id = ?1",
            rusqlite::params![device_id],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(())
    }

    // -----------------------------------------------------------------
    // Embedding-aware memory operations
    // -----------------------------------------------------------------

    /// Store a memory with an embedding vector.
    pub fn remember_with_embedding(
        &self,
        agent_id: AgentId,
        content: &str,
        source: MemorySource,
        scope: &str,
        metadata: HashMap<String, serde_json::Value>,
        embedding: Option<&[f32]>,
    ) -> OpenFangResult<MemoryId> {
        self.semantic
            .remember_with_embedding(agent_id, content, source, scope, metadata, embedding)
    }

    /// Recall memories using vector similarity when a query embedding is provided.
    pub fn recall_with_embedding(
        &self,
        query: &str,
        limit: usize,
        filter: Option<MemoryFilter>,
        query_embedding: Option<&[f32]>,
    ) -> OpenFangResult<Vec<MemoryFragment>> {
        self.semantic
            .recall_with_embedding(query, limit, filter, query_embedding)
    }

    /// Update the embedding for an existing memory.
    pub fn update_embedding(&self, id: MemoryId, embedding: &[f32]) -> OpenFangResult<()> {
        self.semantic.update_embedding(id, embedding)
    }

    /// Async wrapper for `recall_with_embedding` — runs in a blocking thread.
    pub async fn recall_with_embedding_async(
        &self,
        query: &str,
        limit: usize,
        filter: Option<MemoryFilter>,
        query_embedding: Option<&[f32]>,
    ) -> OpenFangResult<Vec<MemoryFragment>> {
        let store = self.semantic.clone();
        let query = query.to_string();
        let embedding_owned = query_embedding.map(|e| e.to_vec());
        tokio::task::spawn_blocking(move || {
            store.recall_with_embedding(&query, limit, filter, embedding_owned.as_deref())
        })
        .await
        .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    /// Async wrapper for `remember_with_embedding` — runs in a blocking thread.
    pub async fn remember_with_embedding_async(
        &self,
        agent_id: AgentId,
        content: &str,
        source: MemorySource,
        scope: &str,
        metadata: HashMap<String, serde_json::Value>,
        embedding: Option<&[f32]>,
    ) -> OpenFangResult<MemoryId> {
        let store = self.semantic.clone();
        let content = content.to_string();
        let scope = scope.to_string();
        let embedding_owned = embedding.map(|e| e.to_vec());
        tokio::task::spawn_blocking(move || {
            store.remember_with_embedding(
                agent_id,
                &content,
                source,
                &scope,
                metadata,
                embedding_owned.as_deref(),
            )
        })
        .await
        .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    // -----------------------------------------------------------------
    // Task queue operations
    // -----------------------------------------------------------------

    /// Post a new task to the shared queue. Returns the task ID.
    pub async fn task_post(
        &self,
        title: &str,
        description: &str,
        assigned_to: Option<&str>,
        created_by: Option<&str>,
    ) -> OpenFangResult<String> {
        let conn = Arc::clone(&self.conn);
        let title = title.to_string();
        let description = description.to_string();
        let assigned_to = assigned_to.unwrap_or("").to_string();
        let created_by = created_by.unwrap_or("").to_string();

        tokio::task::spawn_blocking(move || {
            let id = uuid::Uuid::new_v4().to_string();
            let now = chrono::Utc::now().to_rfc3339();
            let db = conn.lock().map_err(|e| OpenFangError::Internal(e.to_string()))?;
            db.execute(
                "INSERT INTO task_queue (id, agent_id, task_type, payload, status, priority, created_at, title, description, assigned_to, created_by)
                 VALUES (?1, ?2, ?3, ?4, 'pending', 0, ?5, ?6, ?7, ?8, ?9)",
                rusqlite::params![id, &created_by, &title, b"", now, title, description, assigned_to, created_by],
            )
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
            Ok(id)
        })
        .await
        .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    /// Claim the next pending task (optionally for a specific assignee). Returns task JSON or None.
    pub async fn task_claim(&self, agent_id: &str) -> OpenFangResult<Option<serde_json::Value>> {
        let conn = Arc::clone(&self.conn);
        let agent_id = agent_id.to_string();

        tokio::task::spawn_blocking(move || {
            let mut db = conn.lock().map_err(|e| OpenFangError::Internal(e.to_string()))?;
            let tx = db
                .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
                .map_err(|e| OpenFangError::Memory(e.to_string()))?;

            let result = tx.query_row(
                "UPDATE task_queue SET status = 'in_progress', assigned_to = ?1
                 WHERE id = (
                     SELECT id FROM task_queue
                     WHERE status = 'pending' AND (assigned_to = ?1 OR assigned_to = '')
                     ORDER BY priority DESC, created_at ASC
                     LIMIT 1
                 )
                 RETURNING id, title, description, assigned_to, created_by, created_at",
                rusqlite::params![agent_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                    ))
                },
            );

            let claimed = match result {
                Ok((id, title, description, assigned, created_by, created_at)) => {
                    Some(serde_json::json!({
                        "id": id,
                        "title": title,
                        "description": description,
                        "status": "in_progress",
                        "assigned_to": if assigned.is_empty() { &agent_id } else { &assigned },
                        "created_by": created_by,
                        "created_at": created_at,
                    }))
                }
                Err(rusqlite::Error::QueryReturnedNoRows) => None,
                Err(e) => return Err(OpenFangError::Memory(e.to_string())),
            };

            tx.commit().map_err(|e| OpenFangError::Memory(e.to_string()))?;
            Ok(claimed)
        })
        .await
        .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    /// Mark a task as completed with a result string.
    pub async fn task_complete(&self, task_id: &str, result: &str) -> OpenFangResult<()> {
        let conn = Arc::clone(&self.conn);
        let task_id = task_id.to_string();
        let result = result.to_string();

        tokio::task::spawn_blocking(move || {
            let now = chrono::Utc::now().to_rfc3339();
            let db = conn.lock().map_err(|e| OpenFangError::Internal(e.to_string()))?;
            let rows = db.execute(
                "UPDATE task_queue SET status = 'completed', result = ?2, completed_at = ?3
                 WHERE id = ?1 AND status != 'completed'",
                rusqlite::params![task_id, result, now],
            ).map_err(|e| OpenFangError::Memory(e.to_string()))?;
            if rows == 0 {
                match db.query_row(
                    "SELECT 1 FROM task_queue WHERE id = ?1",
                    rusqlite::params![task_id],
                    |_| Ok(()),
                ) {
                    Ok(()) => {}
                    Err(rusqlite::Error::QueryReturnedNoRows) => {
                        return Err(OpenFangError::Internal(format!("Task not found: {task_id}")));
                    }
                    Err(e) => return Err(OpenFangError::Memory(e.to_string())),
                }
            }
            Ok(())
        })
        .await
        .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    // -----------------------------------------------------------------
    // v1.5 carve-outs: failure fingerprint, lineage, supersession
    // -----------------------------------------------------------------

    /// UPSERT a failure fingerprint keyed by `(input_hash, failure_mode)`.
    ///
    /// First write inserts; later writes with the same fingerprint bump
    /// `repeated_count` and refresh `mutation_applied`/`occurred_at`. Per the
    /// v1.5 plan §2.2 UPSERT contract: the `occurred_at` only advances if the
    /// incoming write is newer, preventing out-of-order writes from rewinding
    /// the recency signal.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_failure_fingerprint(
        &self,
        agent_id: AgentId,
        session_id: Option<&str>,
        input_hash: &str,
        failure_mode: &str,
        error_text: &str,
        mutation_applied: Option<&str>,
        recovery_succeeded: bool,
    ) -> OpenFangResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let now = chrono::Utc::now().to_rfc3339();
        let failure_uuid = uuid::Uuid::new_v4().to_string();
        conn.execute(
            "INSERT INTO memory_failures
                (failure_uuid, agent_id, session_id, occurred_at, input_hash, failure_mode,
                 error_text, mutation_applied, recovery_succeeded, repeated_count, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 1, ?4)
             ON CONFLICT(input_hash, failure_mode) DO UPDATE SET
                repeated_count   = memory_failures.repeated_count + 1,
                mutation_applied = excluded.mutation_applied,
                occurred_at      = CASE
                    WHEN excluded.occurred_at > memory_failures.occurred_at
                    THEN excluded.occurred_at
                    ELSE memory_failures.occurred_at
                END,
                recovery_succeeded = CASE
                    WHEN excluded.recovery_succeeded = 1 THEN 1
                    ELSE memory_failures.recovery_succeeded
                END",
            rusqlite::params![
                failure_uuid,
                agent_id.0.to_string(),
                session_id,
                now,
                input_hash,
                failure_mode,
                error_text,
                mutation_applied,
                if recovery_succeeded { 1i64 } else { 0i64 },
            ],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(())
    }

    /// List recent failure fingerprints (newest first).
    pub fn list_failure_fingerprints(
        &self,
        limit: usize,
    ) -> OpenFangResult<Vec<serde_json::Value>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT failure_uuid, agent_id, session_id, occurred_at, input_hash,
                        failure_mode, error_text, mutation_applied, recovery_succeeded, repeated_count
                 FROM memory_failures
                 ORDER BY occurred_at DESC
                 LIMIT ?1",
            )
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let rows = stmt
            .query_map([limit as i64], |row| {
                Ok(serde_json::json!({
                    "failure_uuid":       row.get::<_, String>(0)?,
                    "agent_id":           row.get::<_, String>(1)?,
                    "session_id":         row.get::<_, Option<String>>(2)?,
                    "occurred_at":        row.get::<_, String>(3)?,
                    "input_hash":         row.get::<_, String>(4)?,
                    "failure_mode":       row.get::<_, String>(5)?,
                    "error_text":         row.get::<_, String>(6)?,
                    "mutation_applied":   row.get::<_, Option<String>>(7)?,
                    "recovery_succeeded": row.get::<_, i64>(8)? != 0,
                    "repeated_count":     row.get::<_, i64>(9)?,
                }))
            })
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(|e| OpenFangError::Memory(e.to_string()))?);
        }
        Ok(out)
    }

    /// Check whether a known failure fingerprint exists. Lets the next swarm
    /// short-circuit known dead-ends without re-running the retry loop.
    pub fn has_failure_fingerprint(
        &self,
        input_hash: &str,
        failure_mode: &str,
    ) -> OpenFangResult<bool> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memory_failures WHERE input_hash = ?1 AND failure_mode = ?2",
                rusqlite::params![input_hash, failure_mode],
                |r| r.get(0),
            )
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(count > 0)
    }

    /// Insert a lineage edge `(child_id ← parent_id)` with cycle + depth guards.
    ///
    /// Cycle guard: rejects if `parent_id` already has `child_id` as a
    /// transitive ancestor in the lineage table. Returns
    /// `OpenFangError::Memory("LineageCycle: ...")`.
    ///
    /// Depth guard: enforces `derivation_depth = parent.derivation_depth + 1`
    /// when the parent row is present in `memories`. If the parent isn't a
    /// `memories` row (e.g. self-bootstrap edges), depth is taken as supplied.
    pub fn insert_lineage(
        &self,
        child_id: &str,
        parent_id: &str,
        derivation_type: &str,
        derivation_method: Option<&str>,
    ) -> OpenFangResult<()> {
        const ALLOWED_TYPES: &[&str] =
            &["SUMMARIZE", "PROMOTE", "MERGE", "DISTILL", "CONTRADICT"];
        if !ALLOWED_TYPES.contains(&derivation_type) {
            return Err(OpenFangError::Memory(format!(
                "invalid derivation_type: {derivation_type}"
            )));
        }

        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        if child_id != parent_id {
            // Cycle guard via recursive CTE: would adding (child←parent) close a cycle?
            // A cycle exists if `child_id` is already an ancestor of `parent_id`.
            let cycle: i64 = conn
                .query_row(
                    "WITH RECURSIVE ancestors(node) AS (
                         SELECT parent_id FROM memory_lineage WHERE child_id = ?1
                         UNION
                         SELECT ml.parent_id FROM memory_lineage ml
                         JOIN ancestors a ON ml.child_id = a.node
                     )
                     SELECT COUNT(*) FROM ancestors WHERE node = ?2",
                    rusqlite::params![parent_id, child_id],
                    |r| r.get(0),
                )
                .map_err(|e| OpenFangError::Memory(e.to_string()))?;
            if cycle > 0 {
                return Err(OpenFangError::Memory(format!(
                    "LineageCycle: inserting ({child_id} ← {parent_id}) would close a cycle"
                )));
            }
        }

        // Depth guard: derive depth from parent's derivation_depth if parent
        // is a `memories` row, else 0 (self-bootstrap) or supplied.
        let parent_depth: Option<i64> = conn
            .query_row(
                "SELECT derivation_depth FROM memories WHERE id = ?1",
                rusqlite::params![parent_id],
                |r| r.get(0),
            )
            .ok();
        let child_depth = if child_id == parent_id {
            0
        } else {
            parent_depth.unwrap_or(0) + 1
        };
        if !(0..=8).contains(&child_depth) {
            return Err(OpenFangError::Memory(format!(
                "derivation_depth {child_depth} out of bounds 0..=8"
            )));
        }

        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT OR IGNORE INTO memory_lineage
                (child_id, parent_id, derivation_type, derivation_depth, derivation_method, derived_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![
                child_id,
                parent_id,
                derivation_type,
                child_depth,
                derivation_method,
                now,
            ],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(())
    }

    /// Atomic supersession: mark `old_id` as superseded by `new_id` and insert
    /// a `CONTRADICT` lineage edge in one SQLite transaction. Never UPDATEs the
    /// old row's content.
    pub fn supersede_memory(&self, old_id: &str, new_id: &str) -> OpenFangResult<()> {
        let mut conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let tx = conn
            .transaction()
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        tx.execute(
            "UPDATE memories
             SET status = 'superseded', superseded_by_id = ?1
             WHERE id = ?2 AND status = 'active'",
            rusqlite::params![new_id, old_id],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        let parent_depth: i64 = tx
            .query_row(
                "SELECT derivation_depth FROM memories WHERE id = ?1",
                rusqlite::params![old_id],
                |r| r.get(0),
            )
            .unwrap_or(0);
        let child_depth = (parent_depth + 1).clamp(0, 8);

        let now = chrono::Utc::now().to_rfc3339();
        tx.execute(
            "INSERT OR IGNORE INTO memory_lineage
                (child_id, parent_id, derivation_type, derivation_depth, derivation_method, derived_at)
             VALUES (?1, ?2, 'CONTRADICT', ?3, 'curator_distill', ?4)",
            rusqlite::params![new_id, old_id, child_depth, now],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        // Propagate depth to the new row so future lineage inserts compute correctly.
        tx.execute(
            "UPDATE memories SET derivation_depth = ?1 WHERE id = ?2",
            rusqlite::params![child_depth, new_id],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        tx.commit()
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(())
    }

    /// Set `is_instruction_bearing` for a memory row. Used by the Curator
    /// post-write to retro-flag a freshly-inserted learning whose content
    /// matches the classifier.
    pub fn set_instruction_bearing(
        &self,
        memory_id: &str,
        flagged: bool,
    ) -> OpenFangResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        conn.execute(
            "UPDATE memories SET is_instruction_bearing = ?1 WHERE id = ?2",
            rusqlite::params![if flagged { 1i64 } else { 0i64 }, memory_id],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(())
    }

    /// Return the subset of `ids` that are currently flagged
    /// `is_instruction_bearing = 1`. Empty result means none flagged.
    pub fn flagged_memory_ids(&self, ids: &[String]) -> OpenFangResult<Vec<String>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let placeholders = vec!["?"; ids.len()].join(",");
        let sql = format!(
            "SELECT id FROM memories WHERE is_instruction_bearing = 1 AND id IN ({placeholders})"
        );
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let params: Vec<&dyn rusqlite::types::ToSql> =
            ids.iter().map(|s| s as &dyn rusqlite::types::ToSql).collect();
        let rows = stmt
            .query_map(params.as_slice(), |r| r.get::<_, String>(0))
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(|e| OpenFangError::Memory(e.to_string()))?);
        }
        Ok(out)
    }

    /// Set `event_time` on a memory row. Curator sets this to the chore-end
    /// timestamp so decay measures from the moment of learning, not from the
    /// SQLite insert time (which may lag by buffer-flush latency).
    pub fn set_event_time(&self, memory_id: &str, event_time: &str) -> OpenFangResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        conn.execute(
            "UPDATE memories SET event_time = ?1 WHERE id = ?2",
            rusqlite::params![event_time, memory_id],
        )
        .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        Ok(())
    }

    /// Snapshot of memory-substrate state for the dashboard / API.
    pub fn memory_summary(&self) -> OpenFangResult<serde_json::Value> {
        let mut counts_by_status = serde_json::Map::new();
        let mut counts_by_depth = serde_json::Map::new();
        let instruction_bearing;
        {
            let conn = self
                .conn
                .lock()
                .map_err(|e| OpenFangError::Memory(e.to_string()))?;
            {
                let mut stmt = conn
                    .prepare(
                        "SELECT status, COUNT(*) FROM memories WHERE deleted = 0 GROUP BY status",
                    )
                    .map_err(|e| OpenFangError::Memory(e.to_string()))?;
                let rows = stmt
                    .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))
                    .map_err(|e| OpenFangError::Memory(e.to_string()))?;
                for r in rows {
                    let (s, c) = r.map_err(|e| OpenFangError::Memory(e.to_string()))?;
                    counts_by_status.insert(s, serde_json::Value::from(c));
                }
            }
            instruction_bearing = conn
                .query_row(
                    "SELECT COUNT(*) FROM memories WHERE deleted = 0 AND is_instruction_bearing = 1",
                    [],
                    |r| r.get::<_, i64>(0),
                )
                .unwrap_or(0i64);
            {
                let mut stmt = conn
                    .prepare(
                        "SELECT derivation_depth, COUNT(*) FROM memories WHERE deleted = 0 GROUP BY derivation_depth",
                    )
                    .map_err(|e| OpenFangError::Memory(e.to_string()))?;
                let rows = stmt
                    .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?)))
                    .map_err(|e| OpenFangError::Memory(e.to_string()))?;
                for r in rows {
                    let (d, c) = r.map_err(|e| OpenFangError::Memory(e.to_string()))?;
                    counts_by_depth.insert(d.to_string(), serde_json::Value::from(c));
                }
            }
        }
        let recent_failures = self.list_failure_fingerprints(10)?;

        Ok(serde_json::json!({
            "counts_by_status": counts_by_status,
            "instruction_bearing": instruction_bearing,
            "counts_by_depth": counts_by_depth,
            "recent_failures": recent_failures,
        }))
    }

    /// List active (`status='active'`) memory rows with optional filters.
    /// Powers `openfang memory list` CLI + dashboard tile.
    pub fn list_memories_for_operator(
        &self,
        flagged_only: bool,
        max_depth: Option<i64>,
    ) -> OpenFangResult<Vec<serde_json::Value>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let mut sql = String::from(
            "SELECT id, agent_id, content, scope, confidence, event_time, created_at,
                    status, derivation_depth, is_instruction_bearing
             FROM memories
             WHERE deleted = 0 AND status = 'active'",
        );
        if flagged_only {
            sql.push_str(" AND is_instruction_bearing = 1");
        }
        if let Some(d) = max_depth {
            sql.push_str(&format!(" AND derivation_depth <= {d}"));
        }
        sql.push_str(" ORDER BY COALESCE(event_time, created_at) DESC LIMIT 200");

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let rows = stmt
            .query_map([], |row| {
                Ok(serde_json::json!({
                    "id":                     row.get::<_, String>(0)?,
                    "agent_id":               row.get::<_, String>(1)?,
                    "content":                row.get::<_, String>(2)?,
                    "scope":                  row.get::<_, String>(3)?,
                    "confidence":             row.get::<_, f64>(4)?,
                    "event_time":             row.get::<_, Option<String>>(5)?,
                    "created_at":             row.get::<_, String>(6)?,
                    "status":                 row.get::<_, String>(7)?,
                    "derivation_depth":       row.get::<_, i64>(8)?,
                    "is_instruction_bearing": row.get::<_, i64>(9)? != 0,
                }))
            })
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(|e| OpenFangError::Memory(e.to_string()))?);
        }
        Ok(out)
    }

    /// Fetch a single memory + its lineage chain for `openfang memory show`.
    pub fn show_memory(&self, memory_id: &str) -> OpenFangResult<Option<serde_json::Value>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;

        let row: Option<serde_json::Value> = conn
            .query_row(
                "SELECT id, agent_id, content, source, scope, confidence, metadata,
                        created_at, event_time, status, superseded_by_id, derivation_depth, is_instruction_bearing
                 FROM memories WHERE id = ?1",
                rusqlite::params![memory_id],
                |r| {
                    Ok(serde_json::json!({
                        "id":                     r.get::<_, String>(0)?,
                        "agent_id":               r.get::<_, String>(1)?,
                        "content":                r.get::<_, String>(2)?,
                        "source":                 r.get::<_, String>(3)?,
                        "scope":                  r.get::<_, String>(4)?,
                        "confidence":             r.get::<_, f64>(5)?,
                        "metadata":               r.get::<_, String>(6)?,
                        "created_at":             r.get::<_, String>(7)?,
                        "event_time":             r.get::<_, Option<String>>(8)?,
                        "status":                 r.get::<_, String>(9)?,
                        "superseded_by_id":       r.get::<_, Option<String>>(10)?,
                        "derivation_depth":       r.get::<_, i64>(11)?,
                        "is_instruction_bearing": r.get::<_, i64>(12)? != 0,
                    }))
                },
            )
            .ok();

        let Some(mut row) = row else { return Ok(None) };

        let mut stmt = conn
            .prepare(
                "SELECT child_id, parent_id, derivation_type, derivation_depth, derivation_method, derived_at
                 FROM memory_lineage
                 WHERE child_id = ?1 OR parent_id = ?1
                 ORDER BY derived_at",
            )
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let edges_iter = stmt
            .query_map(rusqlite::params![memory_id], |r| {
                Ok(serde_json::json!({
                    "child_id":          r.get::<_, String>(0)?,
                    "parent_id":         r.get::<_, String>(1)?,
                    "derivation_type":   r.get::<_, String>(2)?,
                    "derivation_depth":  r.get::<_, i64>(3)?,
                    "derivation_method": r.get::<_, Option<String>>(4)?,
                    "derived_at":        r.get::<_, String>(5)?,
                }))
            })
            .map_err(|e| OpenFangError::Memory(e.to_string()))?;
        let mut edges = Vec::new();
        for e in edges_iter {
            edges.push(e.map_err(|err| OpenFangError::Memory(err.to_string()))?);
        }
        row["lineage"] = serde_json::Value::Array(edges);
        Ok(Some(row))
    }

    /// List tasks, optionally filtered by status.
    pub async fn task_list(&self, status: Option<&str>) -> OpenFangResult<Vec<serde_json::Value>> {
        let conn = Arc::clone(&self.conn);
        let status = status.map(|s| s.to_string());

        tokio::task::spawn_blocking(move || {
            let db = conn.lock().map_err(|e| OpenFangError::Internal(e.to_string()))?;
            let (sql, params): (&str, Vec<Box<dyn rusqlite::types::ToSql>>) = match &status {
                Some(s) => (
                    "SELECT id, title, description, status, assigned_to, created_by, created_at, completed_at, result FROM task_queue WHERE status = ?1 ORDER BY created_at DESC",
                    vec![Box::new(s.clone())],
                ),
                None => (
                    "SELECT id, title, description, status, assigned_to, created_by, created_at, completed_at, result FROM task_queue ORDER BY created_at DESC",
                    vec![],
                ),
            };

            let mut stmt = db.prepare(sql).map_err(|e| OpenFangError::Memory(e.to_string()))?;
            let params_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();
            let rows = stmt.query_map(params_refs.as_slice(), |row| {
                Ok(serde_json::json!({
                    "id": row.get::<_, String>(0)?,
                    "title": row.get::<_, String>(1).unwrap_or_default(),
                    "description": row.get::<_, String>(2).unwrap_or_default(),
                    "status": row.get::<_, String>(3)?,
                    "assigned_to": row.get::<_, String>(4).unwrap_or_default(),
                    "created_by": row.get::<_, String>(5).unwrap_or_default(),
                    "created_at": row.get::<_, String>(6).unwrap_or_default(),
                    "completed_at": row.get::<_, Option<String>>(7).unwrap_or(None),
                    "result": row.get::<_, Option<String>>(8).unwrap_or(None),
                }))
            }).map_err(|e| OpenFangError::Memory(e.to_string()))?;

            let mut tasks = Vec::new();
            for row in rows {
                tasks.push(row.map_err(|e| OpenFangError::Memory(e.to_string()))?);
            }
            Ok(tasks)
        })
        .await
        .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }
}

#[async_trait]
impl Memory for MemorySubstrate {
    async fn get(&self, agent_id: AgentId, key: &str) -> OpenFangResult<Option<serde_json::Value>> {
        let store = self.structured.clone();
        let key = key.to_string();
        tokio::task::spawn_blocking(move || store.get(agent_id, &key))
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    async fn set(
        &self,
        agent_id: AgentId,
        key: &str,
        value: serde_json::Value,
    ) -> OpenFangResult<()> {
        let store = self.structured.clone();
        let key = key.to_string();
        tokio::task::spawn_blocking(move || store.set(agent_id, &key, value))
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    async fn delete(&self, agent_id: AgentId, key: &str) -> OpenFangResult<()> {
        let store = self.structured.clone();
        let key = key.to_string();
        tokio::task::spawn_blocking(move || store.delete(agent_id, &key))
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    async fn remember(
        &self,
        agent_id: AgentId,
        content: &str,
        source: MemorySource,
        scope: &str,
        metadata: HashMap<String, serde_json::Value>,
    ) -> OpenFangResult<MemoryId> {
        let store = self.semantic.clone();
        let content = content.to_string();
        let scope = scope.to_string();
        tokio::task::spawn_blocking(move || {
            store.remember(agent_id, &content, source, &scope, metadata)
        })
        .await
        .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    async fn recall(
        &self,
        query: &str,
        limit: usize,
        filter: Option<MemoryFilter>,
    ) -> OpenFangResult<Vec<MemoryFragment>> {
        let store = self.semantic.clone();
        let query = query.to_string();
        tokio::task::spawn_blocking(move || store.recall(&query, limit, filter))
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    async fn forget(&self, id: MemoryId) -> OpenFangResult<()> {
        let store = self.semantic.clone();
        tokio::task::spawn_blocking(move || store.forget(id))
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    async fn add_entity(&self, entity: Entity) -> OpenFangResult<String> {
        let store = self.knowledge.clone();
        tokio::task::spawn_blocking(move || store.add_entity(entity))
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    async fn add_relation(&self, relation: Relation) -> OpenFangResult<String> {
        let store = self.knowledge.clone();
        tokio::task::spawn_blocking(move || store.add_relation(relation))
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    async fn query_graph(&self, pattern: GraphPattern) -> OpenFangResult<Vec<GraphMatch>> {
        let store = self.knowledge.clone();
        tokio::task::spawn_blocking(move || store.query_graph(pattern))
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    async fn consolidate(&self) -> OpenFangResult<ConsolidationReport> {
        let engine = self.consolidation.clone();
        tokio::task::spawn_blocking(move || engine.consolidate())
            .await
            .map_err(|e| OpenFangError::Internal(e.to_string()))?
    }

    async fn export(&self, format: ExportFormat) -> OpenFangResult<Vec<u8>> {
        let _ = format;
        Ok(Vec::new())
    }

    async fn import(&self, _data: &[u8], _format: ExportFormat) -> OpenFangResult<ImportReport> {
        Ok(ImportReport {
            entities_imported: 0,
            relations_imported: 0,
            memories_imported: 0,
            errors: vec!["Import not yet implemented in Phase 1".to_string()],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_substrate_kv() {
        let substrate = MemorySubstrate::open_in_memory(0.1).unwrap();
        let agent_id = AgentId::new();
        substrate
            .set(agent_id, "key", serde_json::json!("value"))
            .await
            .unwrap();
        let val = substrate.get(agent_id, "key").await.unwrap();
        assert_eq!(val, Some(serde_json::json!("value")));
    }

    #[tokio::test]
    async fn test_substrate_remember_recall() {
        let substrate = MemorySubstrate::open_in_memory(0.1).unwrap();
        let agent_id = AgentId::new();
        substrate
            .remember(
                agent_id,
                "Rust is a great language",
                MemorySource::Conversation,
                "episodic",
                HashMap::new(),
            )
            .await
            .unwrap();
        let results = substrate.recall("Rust", 10, None).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_task_post_and_list() {
        let substrate = MemorySubstrate::open_in_memory(0.1).unwrap();
        let id = substrate
            .task_post(
                "Review code",
                "Check the auth module for issues",
                Some("auditor"),
                Some("orchestrator"),
            )
            .await
            .unwrap();
        assert!(!id.is_empty());

        let tasks = substrate.task_list(Some("pending")).await.unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0]["title"], "Review code");
        assert_eq!(tasks[0]["assigned_to"], "auditor");
        assert_eq!(tasks[0]["status"], "pending");
    }

    #[tokio::test]
    async fn test_task_claim_and_complete() {
        let substrate = MemorySubstrate::open_in_memory(0.1).unwrap();
        let task_id = substrate
            .task_post(
                "Audit endpoint",
                "Security audit the /api/login endpoint",
                Some("auditor"),
                None,
            )
            .await
            .unwrap();

        // Claim the task
        let claimed = substrate.task_claim("auditor").await.unwrap();
        assert!(claimed.is_some());
        let claimed = claimed.unwrap();
        assert_eq!(claimed["id"], task_id);
        assert_eq!(claimed["status"], "in_progress");

        // Complete the task
        substrate
            .task_complete(&task_id, "No vulnerabilities found")
            .await
            .unwrap();

        // Verify it shows as completed
        let tasks = substrate.task_list(Some("completed")).await.unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0]["result"], "No vulnerabilities found");
    }

    #[tokio::test]
    async fn test_task_claim_empty() {
        let substrate = MemorySubstrate::open_in_memory(0.1).unwrap();
        let claimed = substrate.task_claim("nobody").await.unwrap();
        assert!(claimed.is_none());
    }

    #[tokio::test]
    async fn test_task_claim_is_atomic_single_winner() {
        let substrate = Arc::new(MemorySubstrate::open_in_memory(0.1).unwrap());
        let task_id = substrate
            .task_post("Solo task", "Only one claimer may win", None, None)
            .await
            .unwrap();

        let a = {
            let s = Arc::clone(&substrate);
            tokio::spawn(async move { s.task_claim("agent-a").await.unwrap() })
        };
        let b = {
            let s = Arc::clone(&substrate);
            tokio::spawn(async move { s.task_claim("agent-b").await.unwrap() })
        };
        let (ra, rb) = (a.await.unwrap(), b.await.unwrap());

        let winners = [&ra, &rb].iter().filter(|r| r.is_some()).count();
        assert_eq!(winners, 1, "exactly one claimer must win the single task");
        let winner = ra.or(rb).unwrap();
        assert_eq!(winner["id"], task_id);
        assert_eq!(winner["status"], "in_progress");

        assert!(substrate.task_claim("agent-c").await.unwrap().is_none());
        assert_eq!(substrate.task_list(Some("in_progress")).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_failure_fingerprint_upsert_bumps_count() {
        let substrate = MemorySubstrate::open_in_memory(0.1).unwrap();
        let agent = AgentId::new();
        substrate
            .insert_failure_fingerprint(
                agent,
                Some("s1"),
                "hash-abc",
                "tool_error",
                "boom",
                Some("retry-with-backoff"),
                false,
            )
            .unwrap();
        substrate
            .insert_failure_fingerprint(
                agent,
                Some("s1"),
                "hash-abc",
                "tool_error",
                "boom again",
                Some("retry-with-backoff"),
                false,
            )
            .unwrap();
        let listed = substrate.list_failure_fingerprints(10).unwrap();
        assert_eq!(listed.len(), 1, "UPSERT must collapse to one row");
        assert_eq!(listed[0]["repeated_count"].as_i64(), Some(2));
        assert!(substrate
            .has_failure_fingerprint("hash-abc", "tool_error")
            .unwrap());
        assert!(!substrate
            .has_failure_fingerprint("hash-xyz", "tool_error")
            .unwrap());
    }

    #[tokio::test]
    async fn test_lineage_cycle_rejected() {
        let substrate = MemorySubstrate::open_in_memory(0.1).unwrap();
        // Seed two memories so depth lookup finds parent rows.
        {
            let conn = substrate.conn.lock().unwrap();
            for id in ["mA", "mB"] {
                conn.execute(
                    "INSERT INTO memories (id, agent_id, content, source, scope, confidence, metadata, created_at, accessed_at, access_count, deleted, event_time, status, derivation_depth, is_instruction_bearing)
                     VALUES (?1, 'agent', 'x', '\"conversation\"', 'episodic', 0.9, '{}', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', 0, 0, '2026-01-01T00:00:00Z', 'active', 0, 0)",
                    [id],
                ).unwrap();
            }
        }
        // mB ← mA is fine.
        substrate
            .insert_lineage("mB", "mA", "CONTRADICT", Some("test"))
            .unwrap();
        // mA ← mB would close a cycle.
        let err = substrate
            .insert_lineage("mA", "mB", "CONTRADICT", Some("test"))
            .unwrap_err();
        assert!(
            matches!(&err, OpenFangError::Memory(m) if m.contains("LineageCycle")),
            "expected LineageCycle, got {err:?}"
        );
    }

    #[tokio::test]
    async fn test_supersede_marks_old_and_writes_lineage() {
        let substrate = MemorySubstrate::open_in_memory(0.1).unwrap();
        {
            let conn = substrate.conn.lock().unwrap();
            for id in ["old1", "new1"] {
                conn.execute(
                    "INSERT INTO memories (id, agent_id, content, source, scope, confidence, metadata, created_at, accessed_at, access_count, deleted, event_time, status, derivation_depth, is_instruction_bearing)
                     VALUES (?1, 'agent', 'fact', '\"conversation\"', 'episodic', 0.9, '{}', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', 0, 0, '2026-01-01T00:00:00Z', 'active', 0, 0)",
                    [id],
                ).unwrap();
            }
        }
        substrate.supersede_memory("old1", "new1").unwrap();
        let conn = substrate.conn.lock().unwrap();
        let (status, by): (String, Option<String>) = conn
            .query_row(
                "SELECT status, superseded_by_id FROM memories WHERE id = 'old1'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(status, "superseded");
        assert_eq!(by.as_deref(), Some("new1"));
        let kind: String = conn
            .query_row(
                "SELECT derivation_type FROM memory_lineage WHERE child_id = 'new1' AND parent_id = 'old1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(kind, "CONTRADICT");
    }

    #[tokio::test]
    async fn test_task_complete_is_idempotent() {
        let substrate = MemorySubstrate::open_in_memory(0.1).unwrap();
        let task_id = substrate
            .task_post("Finalize me", "desc", Some("auditor"), None)
            .await
            .unwrap();
        substrate.task_claim("auditor").await.unwrap();

        substrate.task_complete(&task_id, "done").await.unwrap();
        substrate.task_complete(&task_id, "done again").await.unwrap();

        let tasks = substrate.task_list(Some("completed")).await.unwrap();
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0]["result"], "done");

        assert!(substrate.task_complete("no-such-task", "x").await.is_err());
    }
}
