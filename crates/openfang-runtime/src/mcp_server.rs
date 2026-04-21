//! MCP Server — re-exports from `openfang_types::mcp`.
//!
//! The protocol handler was moved into `openfang-types` so the MCP bridge binary
//! (`openfang-mcp-bridge`) can reuse it without pulling in `wasmtime`, `reqwest`,
//! and other heavy runtime dependencies. This shim preserves the original
//! `handle_mcp_request(&Value, &[ToolDefinition]) -> Value` signature for
//! existing callers; new callers should use `openfang_types::mcp` directly.

pub use openfang_types::mcp::{
    handle_mcp_request as handle_mcp_request_v2, ContentBlock, ToolCallResult, ToolDispatcher,
    PROTOCOL_VERSION,
};

use openfang_types::tool::ToolDefinition;

/// Legacy signature: returns a concrete `Value` (notifications get a `null`).
///
/// The v2 handler returns `Option<Value>` where `None` signals "no response frame
/// for this notification." This shim unwraps `None` to `Value::Null` so existing
/// HTTP-endpoint callers (which always serialize something) don't break.
pub async fn handle_mcp_request(
    request: &serde_json::Value,
    tools: &[ToolDefinition],
) -> serde_json::Value {
    handle_mcp_request_v2(request, tools, None)
        .await
        .unwrap_or(serde_json::Value::Null)
}
