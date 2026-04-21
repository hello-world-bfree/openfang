//! UDS server: accept `openfang-mcp-bridge` connections + dispatch tool calls.
//!
//! Per-run `tokio::net::UnixListener` at
//! `$STATE_DIR/repo-digger/kernel-<run_id>.sock`. Each connection carries
//! length-prefixed JSON requests from a bridge subprocess (spawned by a
//! `claude` CLI under one of the investigation's agents). The server:
//!
//! 1. Reads `u32be` length + `{tool_name, arguments, agent_id, run_id, cookie}` JSON
//! 2. Looks up `(agent_id, run_id)` in the shared [`AgentCookieRegistry`]
//! 3. Rejects any request whose cookie doesn't match (prompt-injection
//!    confused-deputy path closed — a sub-agent can't forge the coordinator's
//!    identity because its cookie was generated fresh and ties it to its
//!    declared `agent_id`)
//! 4. Dispatches the tool via [`openfang_runtime::tool_runner::execute_tool`]
//!    with the caller's `allowed_tools` + `workspace_root`
//! 5. Returns `{content: [{type: "text", text: "..."}], isError: bool}` JSON
//!    framed the same way
//!
//! Multiple sub-agents in the same investigation SHARE a socket (same
//! `run_id`) but get PER-AGENT cookies. The daemon distinguishes their
//! contexts by the `agent_id` field on every RPC — not by connection
//! identity, so connection multiplexing is transparent.

use crate::OpenFangKernel;
use dashmap::DashMap;
use openfang_runtime::kernel_handle::KernelHandle;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use subtle::ConstantTimeEq;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{debug, info, warn};

/// Max in-flight request/response body size. Prevents a malicious bridge
/// from claiming a multi-gigabyte payload and forcing the daemon to allocate.
const MAX_FRAME_BYTES: u32 = 4 * 1024 * 1024;

/// Per-agent record kept by the kernel so the UDS server can authenticate
/// and scope tool calls. Populated at `activate_hand` time for any Hand that
/// uses the MCP bridge (claude-code + workspace_override_setting).
#[derive(Debug, Clone)]
pub struct AgentBridgeEntry {
    /// Cookie the bridge must present on every dispatch request. Verified
    /// in constant time to avoid timing side channels.
    pub cookie: String,
    /// Investigation run ID — distinct sub-agents of the same investigation
    /// share this.
    pub run_id: String,
    /// Canonical workspace root (respecting `workspace_override_setting`).
    pub workspace_root: PathBuf,
    /// Tool subset this agent may invoke. Matches the coordinator's
    /// `allowed_tools`; sub-agents spawned via `code_agent_spawn` get a
    /// narrower subset stamped at spawn time.
    pub allowed_tools: Vec<String>,
}

/// Shared per-daemon registry of active bridge-auth entries.
///
/// Keyed by `agent_id`. One-writer-many-readers, concurrent-safe via
/// [`DashMap`]. Populated by `activate_hand`, removed on agent kill.
pub type AgentCookieRegistry = Arc<DashMap<String, AgentBridgeEntry>>;

/// Request envelope received from the bridge.
#[derive(Debug, Deserialize)]
struct DispatchRequest {
    tool_name: String,
    arguments: serde_json::Value,
    agent_id: String,
    run_id: String,
    cookie: String,
}

/// Response envelope returned to the bridge. Mirrors MCP content block shape.
#[derive(Debug, Serialize)]
struct DispatchResponse {
    content: Vec<DispatchContent>,
    is_error: bool,
}

#[derive(Debug, Serialize)]
struct DispatchContent {
    #[serde(rename = "type")]
    kind: &'static str,
    text: String,
}

/// Start the UDS server for a given run. Returns immediately; the listener
/// accepts connections in a spawned task until the socket is unlinked or
/// the daemon shuts down.
///
/// Caller (kernel) is responsible for:
/// - Populating `registry` before bridge subprocesses connect
/// - Cleaning up the socket file on daemon shutdown (handled by
///   `cleanup_bridge_config` during agent kill, or by the orphan reaper)
pub fn start_run_listener(
    kernel: Arc<OpenFangKernel>,
    registry: AgentCookieRegistry,
    state_dir: &Path,
    run_id: &str,
) -> std::io::Result<()> {
    let socket_path = crate::mcp_bridge::socket_path_for_run(state_dir, run_id);
    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // Unlink any stale socket first — if it exists, bind() fails.
    let _ = std::fs::remove_file(&socket_path);

    #[cfg(unix)]
    let listener = {
        let l = tokio::net::UnixListener::bind(&socket_path)?;
        // 0600 on the socket so only this uid can connect. UnixListener
        // doesn't expose the fd for fchmod; setting via path after bind.
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o600);
        std::fs::set_permissions(&socket_path, perms)?;
        l
    };
    #[cfg(not(unix))]
    let listener: () = {
        // Windows named-pipe transport is a follow-up. Return Ok so the
        // daemon still boots; bridge calls will fail until implemented.
        let _ = (socket_path, kernel, registry, run_id);
        return Ok(());
    };

    #[cfg(unix)]
    {
        let run_id = run_id.to_string();
        tokio::spawn(async move {
            info!(
                socket = %socket_path.display(),
                run_id = %run_id,
                "MCP bridge UDS listener started"
            );
            loop {
                match listener.accept().await {
                    Ok((stream, _addr)) => {
                        let kernel = kernel.clone();
                        let registry = registry.clone();
                        tokio::spawn(async move {
                            if let Err(e) = handle_connection(stream, kernel, registry).await {
                                warn!(error = %e, "MCP bridge connection ended with error");
                            }
                        });
                    }
                    Err(e) => {
                        // accept() returning an error typically means the
                        // socket file was unlinked (clean shutdown) or the
                        // listener itself is closed. Exit the loop.
                        debug!(error = %e, "MCP bridge listener stopped accepting");
                        break;
                    }
                }
            }
            info!(run_id = %run_id, "MCP bridge UDS listener exited");
        });
    }

    Ok(())
}

/// Test hook: run [`dispatch`] against an arbitrary [`KernelHandle`] without
/// needing a full `OpenFangKernel`. Only available in cfg(test) — production
/// callers use the listener spawned by [`start_run_listener`].
#[cfg(test)]
pub async fn dispatch_for_test(
    req_buf: &[u8],
    kernel: &Arc<dyn KernelHandle>,
    registry: &AgentCookieRegistry,
) -> serde_json::Value {
    let resp = dispatch_with_handle(req_buf, kernel, registry).await;
    serde_json::to_value(&resp).expect("DispatchResponse serializes")
}

#[cfg(unix)]
async fn handle_connection(
    mut stream: tokio::net::UnixStream,
    kernel: Arc<OpenFangKernel>,
    registry: AgentCookieRegistry,
) -> std::io::Result<()> {
    // Each connection handles multiple request/response cycles serially.
    // Claude Code's MCP client pipelines tool calls on one stdio session,
    // which maps to one UDS connection per bridge subprocess.
    loop {
        let mut len_buf = [0u8; 4];
        match stream.read_exact(&mut len_buf).await {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e),
        }
        let req_len = u32::from_be_bytes(len_buf);
        if req_len > MAX_FRAME_BYTES {
            warn!(
                len = req_len,
                max = MAX_FRAME_BYTES,
                "MCP bridge request exceeds max frame size; closing"
            );
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "request frame too large",
            ));
        }
        let mut req_buf = vec![0u8; req_len as usize];
        stream.read_exact(&mut req_buf).await?;

        let response = dispatch(&req_buf, &kernel, &registry).await;
        let response_bytes = serde_json::to_vec(&response).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, format!("serialize: {e}"))
        })?;
        let resp_len = u32::try_from(response_bytes.len()).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "response exceeds u32 length",
            )
        })?;
        stream.write_all(&resp_len.to_be_bytes()).await?;
        stream.write_all(&response_bytes).await?;
        stream.flush().await?;
    }
}

async fn dispatch(
    req_buf: &[u8],
    kernel: &Arc<OpenFangKernel>,
    registry: &AgentCookieRegistry,
) -> DispatchResponse {
    let handle: Arc<dyn KernelHandle> = kernel.clone();
    dispatch_with_handle(req_buf, &handle, registry).await
}

/// Dispatch against an abstract [`KernelHandle`]. Shared by the production
/// server path (which constructs the handle from `Arc<OpenFangKernel>`) and
/// the `dispatch_for_test` hook (which drives it with a fake handle).
async fn dispatch_with_handle(
    req_buf: &[u8],
    handle: &Arc<dyn KernelHandle>,
    registry: &AgentCookieRegistry,
) -> DispatchResponse {
    let req: DispatchRequest = match serde_json::from_slice(req_buf) {
        Ok(r) => r,
        Err(e) => {
            return DispatchResponse {
                content: vec![DispatchContent {
                    kind: "text",
                    text: format!("malformed request: {e}"),
                }],
                is_error: true,
            };
        }
    };

    // 1. Look up the agent entry.
    let entry = match registry.get(&req.agent_id) {
        Some(e) => e.clone(),
        None => {
            warn!(
                agent_id = %req.agent_id,
                "MCP bridge rejected: agent_id not in registry"
            );
            return DispatchResponse {
                content: vec![DispatchContent {
                    kind: "text",
                    text: format!("unknown agent_id: {}", req.agent_id),
                }],
                is_error: true,
            };
        }
    };

    // 2. Verify cookie (constant-time) and run_id.
    let cookie_ok: bool = req
        .cookie
        .as_bytes()
        .ct_eq(entry.cookie.as_bytes())
        .unwrap_u8()
        == 1;
    if !cookie_ok || req.run_id != entry.run_id {
        warn!(
            agent_id = %req.agent_id,
            run_id_match = req.run_id == entry.run_id,
            cookie_match = cookie_ok,
            "MCP bridge rejected: cookie or run_id mismatch"
        );
        return DispatchResponse {
            content: vec![DispatchContent {
                kind: "text",
                text: "auth failed".into(),
            }],
            is_error: true,
        };
    }

    // 3. Dispatch. execute_tool takes a pile of optional contexts — we
    //    pass the minimal set the bridge's restricted tool subset needs:
    //    kernel handle (for memory/knowledge/code_agent_spawn tools),
    //    allowed_tools (for the cap-gate), caller agent_id, workspace_root.
    //    Web/browser/media/etc contexts are None — calling those tools via
    //    the bridge will surface a "no context" error, which is correct:
    //    the bridge deliberately exposes only a sandboxed subset.
    let tool_use_id = format!("bridge-{}", uuid::Uuid::new_v4());
    let allowed_tools_ref: Vec<String> = entry.allowed_tools.clone();

    let result = openfang_runtime::tool_runner::execute_tool(
        &tool_use_id,
        &req.tool_name,
        &req.arguments,
        Some(handle),
        Some(&allowed_tools_ref),
        Some(&req.agent_id),
        None, // skill_registry
        None, // mcp_connections — bridge-side MCP is flat; server-side MCP is unused here
        None, // web_ctx
        None, // browser_ctx
        None, // allowed_env_vars
        Some(&entry.workspace_root),
        None, // media_engine
        None, // exec_policy — bridge tool subset excludes shell_exec
        None, // tts_engine
        None, // docker_config
        None, // process_manager
    )
    .await;

    DispatchResponse {
        content: vec![DispatchContent {
            kind: "text",
            text: result.content,
        }],
        is_error: result.is_error,
    }
}

/// Insert or update a bridge-auth entry for an agent. Called by
/// `activate_hand` at repo-digger activation time.
pub fn register_agent(
    registry: &AgentCookieRegistry,
    agent_id: String,
    entry: AgentBridgeEntry,
) {
    registry.insert(agent_id, entry);
}

/// Remove a bridge-auth entry on agent kill / investigation completion.
pub fn unregister_agent(registry: &AgentCookieRegistry, agent_id: &str) {
    registry.remove(agent_id);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(cookie: &str, run_id: &str, tools: &[&str]) -> AgentBridgeEntry {
        AgentBridgeEntry {
            cookie: cookie.to_string(),
            run_id: run_id.to_string(),
            workspace_root: std::env::temp_dir(),
            allowed_tools: tools.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn register_and_unregister_roundtrip() {
        let reg: AgentCookieRegistry = Arc::new(DashMap::new());
        register_agent(&reg, "agent-1".into(), make_entry("c", "r", &["file_read"]));
        assert!(reg.contains_key("agent-1"));
        unregister_agent(&reg, "agent-1");
        assert!(!reg.contains_key("agent-1"));
    }

    #[test]
    fn constant_time_cookie_check_accepts_match() {
        // Sanity check for subtle::ConstantTimeEq — it returns 1 on equal.
        let eq = b"abc".ct_eq(b"abc").unwrap_u8();
        assert_eq!(eq, 1);
        let neq = b"abc".ct_eq(b"abx").unwrap_u8();
        assert_eq!(neq, 0);
        let wrong_len = b"abc".as_ref().ct_eq(b"abcd".as_ref()).unwrap_u8();
        assert_eq!(wrong_len, 0);
    }

    // ── Integration tests using a minimal fake KernelHandle ──────────────
    //
    // Exercise the full dispatch pipeline — JSON parsing, cookie auth,
    // unknown-agent rejection, successful tool dispatch — without needing
    // a real OpenFangKernel. The fake only implements the KernelHandle
    // methods execute_tool consults for `memory_store`/`memory_recall`.

    use std::collections::HashMap;
    use std::sync::Mutex;

    /// Minimal in-memory KernelHandle. Supports memory_store + memory_recall
    /// so the integration tests can validate a real round-trip through
    /// execute_tool without booting a full kernel.
    struct TestKernel {
        store: Mutex<HashMap<String, serde_json::Value>>,
    }

    impl TestKernel {
        fn new() -> Self {
            Self {
                store: Mutex::new(HashMap::new()),
            }
        }
    }

    #[async_trait::async_trait]
    impl KernelHandle for TestKernel {
        async fn spawn_agent(
            &self,
            _manifest_toml: &str,
            _parent_id: Option<&str>,
        ) -> Result<(String, String), String> {
            Err("not implemented in TestKernel".into())
        }
        async fn send_to_agent(
            &self,
            _agent_id: &str,
            _message: &str,
        ) -> Result<String, String> {
            Err("not implemented".into())
        }
        fn list_agents(&self) -> Vec<openfang_runtime::kernel_handle::AgentInfo> {
            vec![]
        }
        fn kill_agent(&self, _agent_id: &str) -> Result<(), String> {
            Ok(())
        }
        fn memory_store(
            &self,
            key: &str,
            value: serde_json::Value,
        ) -> Result<(), String> {
            self.store
                .lock()
                .unwrap()
                .insert(key.to_string(), value);
            Ok(())
        }
        fn memory_recall(&self, key: &str) -> Result<Option<serde_json::Value>, String> {
            Ok(self.store.lock().unwrap().get(key).cloned())
        }
        fn find_agents(&self, _query: &str) -> Vec<openfang_runtime::kernel_handle::AgentInfo> {
            vec![]
        }
        async fn task_post(
            &self,
            _title: &str,
            _description: &str,
            _assigned_to: Option<&str>,
            _created_by: Option<&str>,
        ) -> Result<String, String> {
            Err("not implemented".into())
        }
        async fn task_claim(
            &self,
            _agent_id: &str,
        ) -> Result<Option<serde_json::Value>, String> {
            Ok(None)
        }
        async fn task_complete(
            &self,
            _task_id: &str,
            _result: &str,
        ) -> Result<(), String> {
            Ok(())
        }
        async fn task_list(
            &self,
            _status: Option<&str>,
        ) -> Result<Vec<serde_json::Value>, String> {
            Ok(vec![])
        }
        async fn publish_event(
            &self,
            _event_type: &str,
            _payload: serde_json::Value,
        ) -> Result<(), String> {
            Ok(())
        }
        async fn knowledge_add_entity(
            &self,
            _entity: openfang_types::memory::Entity,
        ) -> Result<String, String> {
            Err("not implemented".into())
        }
        async fn knowledge_add_relation(
            &self,
            _relation: openfang_types::memory::Relation,
        ) -> Result<String, String> {
            Err("not implemented".into())
        }
        async fn knowledge_query(
            &self,
            _pattern: openfang_types::memory::GraphPattern,
        ) -> Result<Vec<openfang_types::memory::GraphMatch>, String> {
            Ok(vec![])
        }
    }

    fn make_handle() -> Arc<dyn KernelHandle> {
        Arc::new(TestKernel::new())
    }

    #[tokio::test]
    async fn dispatch_rejects_unknown_agent_id() {
        let reg: AgentCookieRegistry = Arc::new(DashMap::new());
        let handle = make_handle();
        let req = serde_json::json!({
            "tool_name": "memory_recall",
            "arguments": {"key": "x"},
            "agent_id": "ghost-agent",
            "run_id": "any",
            "cookie": "any",
        });
        let resp = dispatch_for_test(
            serde_json::to_vec(&req).unwrap().as_slice(),
            &handle,
            &reg,
        )
        .await;
        assert_eq!(resp["is_error"], true);
        assert!(resp["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("unknown agent_id"));
    }

    #[tokio::test]
    async fn dispatch_rejects_wrong_cookie() {
        let reg: AgentCookieRegistry = Arc::new(DashMap::new());
        register_agent(
            &reg,
            "agent-1".into(),
            make_entry("real-cookie", "run-1", &["memory_recall"]),
        );
        let handle = make_handle();
        let req = serde_json::json!({
            "tool_name": "memory_recall",
            "arguments": {"key": "x"},
            "agent_id": "agent-1",
            "run_id": "run-1",
            "cookie": "wrong-cookie",
        });
        let resp = dispatch_for_test(
            serde_json::to_vec(&req).unwrap().as_slice(),
            &handle,
            &reg,
        )
        .await;
        assert_eq!(resp["is_error"], true);
        assert!(resp["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("auth failed"));
    }

    #[tokio::test]
    async fn dispatch_rejects_wrong_run_id() {
        let reg: AgentCookieRegistry = Arc::new(DashMap::new());
        register_agent(
            &reg,
            "agent-1".into(),
            make_entry("c", "run-legit", &["memory_recall"]),
        );
        let handle = make_handle();
        let req = serde_json::json!({
            "tool_name": "memory_recall",
            "arguments": {"key": "x"},
            "agent_id": "agent-1",
            "run_id": "run-imposter",
            "cookie": "c",
        });
        let resp = dispatch_for_test(
            serde_json::to_vec(&req).unwrap().as_slice(),
            &handle,
            &reg,
        )
        .await;
        assert_eq!(resp["is_error"], true);
        assert!(resp["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("auth failed"));
    }

    #[tokio::test]
    async fn dispatch_rejects_tool_not_in_allowed_list() {
        let reg: AgentCookieRegistry = Arc::new(DashMap::new());
        // Agent allowed only memory_recall; attempts file_read.
        register_agent(
            &reg,
            "agent-1".into(),
            make_entry("c", "run-1", &["memory_recall"]),
        );
        let handle = make_handle();
        let req = serde_json::json!({
            "tool_name": "file_read",
            "arguments": {"path": "/etc/passwd"},
            "agent_id": "agent-1",
            "run_id": "run-1",
            "cookie": "c",
        });
        let resp = dispatch_for_test(
            serde_json::to_vec(&req).unwrap().as_slice(),
            &handle,
            &reg,
        )
        .await;
        assert_eq!(resp["is_error"], true);
        let text = resp["content"][0]["text"].as_str().unwrap();
        assert!(
            text.contains("Permission denied") || text.contains("capability"),
            "expected capability rejection; got: {text}"
        );
    }

    #[tokio::test]
    async fn dispatch_succeeds_for_memory_store_and_recall() {
        let reg: AgentCookieRegistry = Arc::new(DashMap::new());
        register_agent(
            &reg,
            "agent-1".into(),
            make_entry("c", "run-1", &["memory_store", "memory_recall"]),
        );
        let handle = make_handle();

        // Store.
        let store_req = serde_json::json!({
            "tool_name": "memory_store",
            "arguments": {"key": "greeting", "value": "hello"},
            "agent_id": "agent-1",
            "run_id": "run-1",
            "cookie": "c",
        });
        let resp = dispatch_for_test(
            serde_json::to_vec(&store_req).unwrap().as_slice(),
            &handle,
            &reg,
        )
        .await;
        assert_eq!(resp["is_error"], false, "store must succeed: {resp:?}");

        // Recall.
        let recall_req = serde_json::json!({
            "tool_name": "memory_recall",
            "arguments": {"key": "greeting"},
            "agent_id": "agent-1",
            "run_id": "run-1",
            "cookie": "c",
        });
        let resp = dispatch_for_test(
            serde_json::to_vec(&recall_req).unwrap().as_slice(),
            &handle,
            &reg,
        )
        .await;
        assert_eq!(resp["is_error"], false, "recall must succeed: {resp:?}");
        assert!(resp["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("hello"));
    }

    /// Wire-level UDS round-trip. Spawns the actual listener on a temp
    /// socket, connects as a bridge client would, frames + sends a request,
    /// reads + unframes the response. Validates the end-to-end binary
    /// protocol not just the dispatch logic.
    #[cfg(unix)]
    #[tokio::test]
    async fn uds_wire_protocol_round_trip() {
        use std::time::Duration;
        use tempfile::TempDir;
        use tokio::net::UnixStream;

        let dir = TempDir::new().unwrap();
        let state_dir = dir.path().to_path_buf();
        std::fs::create_dir_all(state_dir.join("repo-digger")).unwrap();

        let reg: AgentCookieRegistry = Arc::new(DashMap::new());
        register_agent(
            &reg,
            "wire-agent".into(),
            AgentBridgeEntry {
                cookie: "wire-cookie".into(),
                run_id: "wire-run".into(),
                workspace_root: std::env::temp_dir(),
                allowed_tools: vec!["memory_store".into(), "memory_recall".into()],
            },
        );

        // The production start_run_listener requires Arc<OpenFangKernel>;
        // we test the wire path by spawning our own listener that wraps
        // dispatch_with_handle with our TestKernel. This mirrors the real
        // handle_connection loop byte-for-byte.
        let sock_path = crate::mcp_bridge::socket_path_for_run(&state_dir, "wire-run");
        let _ = std::fs::remove_file(&sock_path);
        let listener = tokio::net::UnixListener::bind(&sock_path).unwrap();
        let handle: Arc<dyn KernelHandle> = make_handle();
        let reg_clone = reg.clone();
        let handle_clone = handle.clone();

        tokio::spawn(async move {
            if let Ok((mut stream, _)) = listener.accept().await {
                // Single request/response — matches what the wire client does.
                let mut len_buf = [0u8; 4];
                tokio::io::AsyncReadExt::read_exact(&mut stream, &mut len_buf)
                    .await
                    .unwrap();
                let n = u32::from_be_bytes(len_buf) as usize;
                let mut req_buf = vec![0u8; n];
                tokio::io::AsyncReadExt::read_exact(&mut stream, &mut req_buf)
                    .await
                    .unwrap();
                let resp = dispatch_with_handle(&req_buf, &handle_clone, &reg_clone).await;
                let resp_bytes = serde_json::to_vec(&resp).unwrap();
                let resp_len = resp_bytes.len() as u32;
                tokio::io::AsyncWriteExt::write_all(&mut stream, &resp_len.to_be_bytes())
                    .await
                    .unwrap();
                tokio::io::AsyncWriteExt::write_all(&mut stream, &resp_bytes)
                    .await
                    .unwrap();
                tokio::io::AsyncWriteExt::flush(&mut stream).await.unwrap();
            }
        });

        // Give the listener a tick to bind.
        tokio::time::sleep(Duration::from_millis(50)).await;

        // Client side — same framing.
        let mut client = UnixStream::connect(&sock_path).await.unwrap();
        let req = serde_json::json!({
            "tool_name": "memory_store",
            "arguments": {"key": "wire-k", "value": "wire-v"},
            "agent_id": "wire-agent",
            "run_id": "wire-run",
            "cookie": "wire-cookie",
        });
        let req_bytes = serde_json::to_vec(&req).unwrap();
        let req_len = req_bytes.len() as u32;
        tokio::io::AsyncWriteExt::write_all(&mut client, &req_len.to_be_bytes())
            .await
            .unwrap();
        tokio::io::AsyncWriteExt::write_all(&mut client, &req_bytes).await.unwrap();
        tokio::io::AsyncWriteExt::flush(&mut client).await.unwrap();

        let mut resp_len_buf = [0u8; 4];
        tokio::io::AsyncReadExt::read_exact(&mut client, &mut resp_len_buf)
            .await
            .unwrap();
        let resp_n = u32::from_be_bytes(resp_len_buf) as usize;
        let mut resp_buf = vec![0u8; resp_n];
        tokio::io::AsyncReadExt::read_exact(&mut client, &mut resp_buf)
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&resp_buf).unwrap();
        assert_eq!(parsed["is_error"], false, "wire round-trip must succeed: {parsed}");

        let _ = std::fs::remove_file(&sock_path);
    }

    #[tokio::test]
    async fn dispatch_rejects_malformed_json() {
        let reg: AgentCookieRegistry = Arc::new(DashMap::new());
        let handle = make_handle();
        let resp = dispatch_for_test(b"{not valid json", &handle, &reg).await;
        assert_eq!(resp["is_error"], true);
        assert!(resp["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("malformed"));
    }
}
