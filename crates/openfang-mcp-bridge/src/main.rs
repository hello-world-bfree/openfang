//! openfang-mcp-bridge — stdio MCP server that forwards `tools/call` to the
//! OpenFang daemon over a Unix domain socket (Windows: named pipe).
//!
//! Claude Code spawns this binary via `claude --mcp-config <path>` during a
//! `repo-digger` investigation. The MCP config JSON carries the tool list, a
//! per-agent cookie, and the UDS path the bridge must connect to.
//!
//! Protocol:
//! - stdin:  line-delimited JSON-RPC 2.0 requests from Claude Code
//! - stdout: line-delimited JSON-RPC 2.0 responses (notifications drop the
//!   response frame via `handle_mcp_request` returning `None`)
//! - UDS:    length-prefixed JSON messages `{tool_name, arguments, agent_id,
//!   run_id, cookie}` → `{content: [...], isError: bool}`
//!
//! The UDS transport is abstracted so a Windows named-pipe impl can be added
//! without touching the protocol layer.

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use openfang_types::mcp::{handle_mcp_request, ToolCallResult, ToolDispatcher};
use openfang_types::tool::ToolDefinition;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::{info, warn};

/// Content of the MCP config JSON written by the openfang daemon.
///
/// Standard MCP config fields (`mcpServers`) are used by Claude Code itself.
/// OpenFang-specific fields live under the `openfang` key — the bridge reads
/// these at startup. Fields outside `openfang` are opaque to us.
#[derive(Debug, Deserialize)]
struct BridgeConfigFile {
    openfang: OpenFangBridgeConfig,
}

#[derive(Debug, Deserialize)]
struct OpenFangBridgeConfig {
    /// Absolute path to the daemon UDS socket.
    socket_path: PathBuf,
    /// Opaque auth cookie (256-bit random, hex-encoded).
    cookie: String,
    /// Agent ID presented on every dispatch request. Daemon uses this to pick
    /// the caller's workspace + allowed_tools for cap-checking.
    agent_id: String,
    /// Investigation run ID.
    run_id: String,
    /// Tool definitions the bridge should advertise via `tools/list`. Mirrors
    /// the caller agent's `allowed_tools`, written by the kernel at spawn time.
    tools: Vec<ToolDefinition>,
    /// PID of the daemon that wrote this config. Used by the daemon's orphan
    /// reaper; bridge doesn't read it. Tolerated here so JSON round-trips.
    #[serde(default, rename = "daemon_pid")]
    _daemon_pid: Option<u32>,
}

/// RPC envelope sent to the daemon over the UDS socket.
#[derive(Debug, Serialize)]
struct DispatchRequest<'a> {
    tool_name: &'a str,
    arguments: serde_json::Value,
    agent_id: &'a str,
    run_id: &'a str,
    cookie: &'a str,
}

#[derive(Debug, Deserialize)]
struct DispatchResponse {
    content: Vec<DispatchContent>,
    #[serde(default)]
    is_error: bool,
}

#[derive(Debug, Deserialize)]
struct DispatchContent {
    #[serde(rename = "type")]
    kind: String,
    text: String,
}

/// `ToolDispatcher` impl that forwards to the daemon over UDS.
struct UdsDispatcher {
    config: OpenFangBridgeConfig,
}

#[async_trait]
impl ToolDispatcher for UdsDispatcher {
    async fn dispatch(&self, tool_name: &str, arguments: serde_json::Value) -> ToolCallResult {
        match self.dispatch_inner(tool_name, arguments).await {
            Ok(r) => r,
            Err(e) => {
                // MCP convention: recoverable errors ride on the success response
                // with isError=true so the LLM can self-correct.
                ToolCallResult::err(format!("bridge dispatch error: {e}"))
            }
        }
    }
}

impl UdsDispatcher {
    async fn dispatch_inner(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolCallResult> {
        let req = DispatchRequest {
            tool_name,
            arguments,
            agent_id: &self.config.agent_id,
            run_id: &self.config.run_id,
            cookie: &self.config.cookie,
        };
        let bytes = serde_json::to_vec(&req)?;
        let resp_bytes = transport::call(&self.config.socket_path, &bytes).await?;
        let resp: DispatchResponse = serde_json::from_slice(&resp_bytes)
            .with_context(|| format!("daemon returned invalid JSON ({} bytes)", resp_bytes.len()))?;

        let content = resp
            .content
            .into_iter()
            .map(|c| openfang_types::mcp::ContentBlock {
                kind: c.kind,
                text: c.text,
            })
            .collect();
        Ok(ToolCallResult {
            content,
            is_error: resp.is_error,
        })
    }
}

/// Transport abstraction: length-prefixed JSON over a connected socket.
///
/// Unix: `tokio::net::UnixStream`. Windows: `tokio::net::windows::named_pipe`
/// (not yet implemented — Windows support is a follow-up).
mod transport {
    use anyhow::{anyhow, Context, Result};
    use std::path::Path;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[cfg(unix)]
    pub async fn call(socket_path: &Path, payload: &[u8]) -> Result<Vec<u8>> {
        use tokio::net::UnixStream;
        let mut stream = UnixStream::connect(socket_path)
            .await
            .with_context(|| format!("connect UDS {}", socket_path.display()))?;
        // Frame: u32-be length || bytes
        let len = u32::try_from(payload.len()).map_err(|_| anyhow!("payload too large"))?;
        stream.write_all(&len.to_be_bytes()).await?;
        stream.write_all(payload).await?;
        stream.flush().await?;

        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf).await?;
        let resp_len = u32::from_be_bytes(len_buf) as usize;
        if resp_len > 16 * 1024 * 1024 {
            return Err(anyhow!("response too large: {resp_len} bytes"));
        }
        let mut resp_buf = vec![0u8; resp_len];
        stream.read_exact(&mut resp_buf).await?;
        Ok(resp_buf)
    }

    #[cfg(not(unix))]
    pub async fn call(_socket_path: &Path, _payload: &[u8]) -> Result<Vec<u8>> {
        Err(anyhow!(
            "openfang-mcp-bridge Windows named-pipe transport not yet implemented"
        ))
    }
}

fn load_config(path: &Path) -> Result<OpenFangBridgeConfig> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("read bridge config at {}", path.display()))?;
    let parsed: BridgeConfigFile = serde_json::from_str(&raw)
        .with_context(|| format!("parse bridge config at {}", path.display()))?;
    Ok(parsed.openfang)
}

use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    // Minimal tracing — the daemon owns the main observability stack; the bridge
    // writes diagnostic events to stderr only.
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("OPENFANG_MCP_BRIDGE_LOG")
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .try_init();

    let config_path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow!("usage: openfang-mcp-bridge <path-to-config.json>"))?;
    let config = load_config(Path::new(&config_path))
        .with_context(|| format!("load config from {config_path}"))?;

    info!(
        agent_id = %config.agent_id,
        run_id = %config.run_id,
        tools = config.tools.len(),
        "openfang-mcp-bridge starting",
    );

    let tools = config.tools.clone();
    let dispatcher = UdsDispatcher { config };

    // Stdio transport loop.
    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin).lines();
    let mut stdout = tokio::io::stdout();

    while let Some(line) = reader.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let request: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                warn!(error = %e, "malformed JSON-RPC request, skipping");
                continue;
            }
        };

        let response = handle_mcp_request(&request, &tools, Some(&dispatcher)).await;
        if let Some(resp) = response {
            let serialized = serde_json::to_vec(&resp)?;
            stdout.write_all(&serialized).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        }
        // else: notification — no response frame
    }

    info!("stdin closed, exiting");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn load_config_parses_minimal() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("mcp.json");
        let content = r#"{
            "mcpServers": {
                "openfang": {"command": "openfang-mcp-bridge", "args": ["/tmp/mcp.json"]}
            },
            "openfang": {
                "socket_path": "/tmp/openfang.sock",
                "cookie": "deadbeef",
                "agent_id": "agent-123",
                "run_id": "run-456",
                "tools": [
                    {"name": "file_read", "description": "read file",
                     "input_schema": {"type":"object","properties":{"path":{"type":"string"}}}}
                ]
            }
        }"#;
        std::fs::write(&path, content).unwrap();
        let cfg = load_config(&path).unwrap();
        assert_eq!(cfg.cookie, "deadbeef");
        assert_eq!(cfg.agent_id, "agent-123");
        assert_eq!(cfg.run_id, "run-456");
        assert_eq!(cfg.tools.len(), 1);
    }

    #[test]
    fn load_config_rejects_missing_openfang_block() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("mcp.json");
        std::fs::write(&path, r#"{"mcpServers": {}}"#).unwrap();
        assert!(load_config(&path).is_err());
    }
}
