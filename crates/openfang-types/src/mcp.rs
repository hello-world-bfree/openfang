//! MCP Server protocol handler — expose OpenFang tools via the Model Context Protocol.
//!
//! Stateless JSON-RPC 2.0 handler implementing the server side of MCP 2024-11-05.
//! Callers (bridge subprocess, external CLI tools, etc.) wire this into a transport
//! (stdio, HTTP, Unix socket) and optionally supply a `ToolDispatcher` to execute
//! `tools/call` requests. Without a dispatcher, `tools/call` returns a stub text
//! response confirming the tool is registered.
//!
//! Lives in `openfang-types` (not `openfang-runtime`) so the MCP bridge binary can
//! reuse the protocol layer without pulling in `wasmtime`, `reqwest`, `rusqlite`,
//! and the LLM drivers.

use crate::tool::{normalize_schema_for_provider, ToolDefinition};
use async_trait::async_trait;
use serde_json::json;

/// MCP protocol version supported by this server.
pub const PROTOCOL_VERSION: &str = "2024-11-05";

/// Content block inside an MCP tool call result.
#[derive(Debug, Clone)]
pub struct ContentBlock {
    pub kind: String,
    pub text: String,
}

impl ContentBlock {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            kind: "text".to_string(),
            text: text.into(),
        }
    }

    fn to_json(&self) -> serde_json::Value {
        json!({ "type": self.kind, "text": self.text })
    }
}

/// Result of executing a tool via `ToolDispatcher`.
///
/// MCP convention (2024-11-05 §4.3): a *recoverable* tool error (bad args, transient
/// failure, cap reached) returns a SUCCESS response with `isError: true` in the
/// content array so the LLM can self-correct. A *protocol* error (unknown tool,
/// malformed request) returns a JSON-RPC error — handled by `handle_mcp_request`
/// itself, not the dispatcher.
#[derive(Debug, Clone)]
pub struct ToolCallResult {
    pub content: Vec<ContentBlock>,
    pub is_error: bool,
}

impl ToolCallResult {
    pub fn ok(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::text(text)],
            is_error: false,
        }
    }

    pub fn err(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::text(text)],
            is_error: true,
        }
    }

    fn to_json(&self) -> serde_json::Value {
        let content: Vec<serde_json::Value> = self.content.iter().map(ContentBlock::to_json).collect();
        json!({ "content": content, "isError": self.is_error })
    }
}

/// Caller-supplied hook that executes `tools/call` requests.
///
/// The bridge implementation forwards over a UDS RPC back to the openfang daemon.
/// Tests and the legacy stub path pass `None` to `handle_mcp_request` and get
/// a textual "execution must be wired by the host" response.
#[async_trait]
pub trait ToolDispatcher: Send + Sync {
    async fn dispatch(&self, tool_name: &str, arguments: serde_json::Value) -> ToolCallResult;
}

/// Handle an incoming MCP JSON-RPC request and return a response.
///
/// Returns `None` for notifications (`notifications/initialized`) — the transport
/// loop MUST NOT write a response frame in that case. For method requests, returns
/// `Some(Value)` containing the JSON-RPC response.
///
/// `dispatcher` is optional: when `Some`, `tools/call` requests are forwarded to
/// it and wrapped per MCP convention. When `None`, `tools/call` returns a stub
/// text response ("execution must be wired") — backward-compatible with the
/// original stateless handler.
pub async fn handle_mcp_request(
    request: &serde_json::Value,
    tools: &[ToolDefinition],
    dispatcher: Option<&dyn ToolDispatcher>,
) -> Option<serde_json::Value> {
    let method = request["method"].as_str().unwrap_or("");
    let id = request.get("id").cloned();

    match method {
        "initialize" => Some(make_response(
            id,
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": { "tools": {} },
                "serverInfo": {
                    "name": "openfang",
                    "version": env!("CARGO_PKG_VERSION"),
                }
            }),
        )),
        // Notifications carry no `id` and receive no response frame.
        m if m.starts_with("notifications/") => None,
        "tools/list" => {
            let tool_list: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    // Normalize for broadest provider compatibility. Claude Code's
                    // MCP client accepts anthropic-style schemas as-is, but other
                    // MCP clients may reject $schema / $defs / $ref, so we run the
                    // openai path which strips those and flattens anyOf.
                    let schema = normalize_schema_for_provider(&t.input_schema, "openai");
                    json!({
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": schema,
                    })
                })
                .collect();
            Some(make_response(id, json!({ "tools": tool_list })))
        }
        "tools/call" => {
            let tool_name = request["params"]["name"].as_str().unwrap_or("");
            let arguments = request["params"]
                .get("arguments")
                .cloned()
                .unwrap_or(json!({}));

            // Protocol-level: unknown tool is a JSON-RPC error.
            if !tools.iter().any(|t| t.name == tool_name) {
                return Some(make_error(id, -32602, &format!("Unknown tool: {tool_name}")));
            }

            let result = if let Some(d) = dispatcher {
                d.dispatch(tool_name, arguments).await
            } else {
                // Backward-compatible stub: execution not wired.
                ToolCallResult::ok(format!(
                    "Tool '{tool_name}' is available. Execution must be wired by the host."
                ))
            };

            Some(make_response(id, result.to_json()))
        }
        _ => Some(make_error(id, -32601, &format!("Method not found: {method}"))),
    }
}

/// Build a JSON-RPC 2.0 success response.
fn make_response(id: Option<serde_json::Value>, result: serde_json::Value) -> serde_json::Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

/// Build a JSON-RPC 2.0 error response.
fn make_error(id: Option<serde_json::Value>, code: i64, message: &str) -> serde_json::Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_tools() -> Vec<ToolDefinition> {
        vec![
            ToolDefinition {
                name: "file_read".to_string(),
                description: "Read a file".to_string(),
                input_schema: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
            },
            ToolDefinition {
                name: "web_fetch".to_string(),
                description: "Fetch a URL".to_string(),
                input_schema: json!({"type": "object"}),
            },
        ]
    }

    #[tokio::test]
    async fn tools_list_returns_normalized_schemas() {
        let tools = test_tools();
        let request = json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"});

        let response = handle_mcp_request(&request, &tools, None).await.unwrap();
        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 1);

        let tool_list = response["result"]["tools"].as_array().unwrap();
        assert_eq!(tool_list.len(), 2);
        assert_eq!(tool_list[0]["name"], "file_read");
        // Schema normalization preserved `type`+`properties`.
        assert_eq!(tool_list[0]["inputSchema"]["type"], "object");
    }

    #[tokio::test]
    async fn initialize_returns_protocol_version() {
        let tools = test_tools();
        let request = json!({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": {"name": "test"}}
        });
        let response = handle_mcp_request(&request, &tools, None).await.unwrap();
        assert_eq!(response["result"]["protocolVersion"], PROTOCOL_VERSION);
        assert_eq!(response["result"]["serverInfo"]["name"], "openfang");
    }

    #[tokio::test]
    async fn unknown_method_returns_error() {
        let tools = test_tools();
        let request = json!({"jsonrpc": "2.0", "id": 5, "method": "nonexistent/method"});
        let response = handle_mcp_request(&request, &tools, None).await.unwrap();
        assert_eq!(response["error"]["code"], -32601);
    }

    #[tokio::test]
    async fn notifications_return_none() {
        let tools = test_tools();
        let request = json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
        let response = handle_mcp_request(&request, &tools, None).await;
        assert!(response.is_none(), "notifications MUST NOT produce a response frame");
    }

    #[tokio::test]
    async fn tools_call_without_dispatcher_returns_stub() {
        let tools = test_tools();
        let request = json!({
            "jsonrpc": "2.0", "id": 7, "method": "tools/call",
            "params": {"name": "file_read", "arguments": {"path": "/tmp/x"}}
        });
        let response = handle_mcp_request(&request, &tools, None).await.unwrap();
        assert_eq!(response["result"]["isError"], false);
        assert!(response["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("Execution must be wired"));
    }

    #[tokio::test]
    async fn tools_call_unknown_tool_returns_protocol_error() {
        let tools = test_tools();
        let request = json!({
            "jsonrpc": "2.0", "id": 8, "method": "tools/call",
            "params": {"name": "does_not_exist", "arguments": {}}
        });
        let response = handle_mcp_request(&request, &tools, None).await.unwrap();
        assert_eq!(response["error"]["code"], -32602);
    }

    struct FakeDispatcher;

    #[async_trait]
    impl ToolDispatcher for FakeDispatcher {
        async fn dispatch(&self, tool_name: &str, arguments: serde_json::Value) -> ToolCallResult {
            if tool_name == "file_read" {
                ToolCallResult::ok(format!("dispatched file_read with {arguments}"))
            } else {
                ToolCallResult::err(format!("dispatcher rejected {tool_name}"))
            }
        }
    }

    #[tokio::test]
    async fn tools_call_with_dispatcher_forwards() {
        let tools = test_tools();
        let request = json!({
            "jsonrpc": "2.0", "id": 9, "method": "tools/call",
            "params": {"name": "file_read", "arguments": {"path": "/tmp/x"}}
        });
        let dispatcher = FakeDispatcher;
        let response = handle_mcp_request(&request, &tools, Some(&dispatcher))
            .await
            .unwrap();
        assert_eq!(response["result"]["isError"], false);
        let text = response["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.starts_with("dispatched file_read"));
    }

    #[tokio::test]
    async fn tools_call_with_dispatcher_propagates_is_error() {
        let tools = test_tools();
        let request = json!({
            "jsonrpc": "2.0", "id": 10, "method": "tools/call",
            "params": {"name": "web_fetch", "arguments": {"url": "http://x"}}
        });
        let dispatcher = FakeDispatcher;
        let response = handle_mcp_request(&request, &tools, Some(&dispatcher))
            .await
            .unwrap();
        assert_eq!(response["result"]["isError"], true);
    }
}
