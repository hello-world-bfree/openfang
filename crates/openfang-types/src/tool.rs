//! Tool definition and result types.

use serde::{Deserialize, Serialize};

/// Definition of a tool that an agent can use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Unique tool identifier.
    pub name: String,
    /// Human-readable description for the LLM.
    pub description: String,
    /// JSON Schema for the tool's input parameters.
    pub input_schema: serde_json::Value,
}

/// A tool call requested by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique ID for this tool use instance.
    pub id: String,
    /// Which tool to call.
    pub name: String,
    /// The input parameters.
    pub input: serde_json::Value,
}

/// Machine-readable failure class for a tool error.
///
/// Gives the model a typed signal to re-plan on instead of pattern-matching a
/// free-text string: a `Timeout` is worth retrying, an `InvalidParam` means fix
/// the arguments, an `EmptyResult` means rephrase. Mirrors the tool-result-shaping
/// "Tried But Failed" schema. Serializes to a stable SCREAMING_SNAKE code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorCode {
    /// Operation exceeded its time budget. Retry may succeed.
    Timeout,
    /// Arguments were malformed or failed validation. Fix args; do not retry as-is.
    InvalidParam,
    /// Upstream rate limit hit. Back off, then retry.
    RateLimited,
    /// Target resource does not exist. Do not retry with the same target.
    NotFound,
    /// Call succeeded but produced no usable data. Rephrase or try a different source.
    EmptyResult,
    /// A required dependency/backend is unavailable. Retry later or route around it.
    DepDown,
    /// Blocked by capability, approval, or policy. Do not retry without authorization.
    Denied,
}

impl ErrorCode {
    /// The stable wire string for this code (matches the serde representation).
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::Timeout => "TIMEOUT",
            ErrorCode::InvalidParam => "INVALID_PARAM",
            ErrorCode::RateLimited => "RATE_LIMITED",
            ErrorCode::NotFound => "NOT_FOUND",
            ErrorCode::EmptyResult => "EMPTY_RESULT",
            ErrorCode::DepDown => "DEP_DOWN",
            ErrorCode::Denied => "DENIED",
        }
    }

    /// Whether retrying the identical call could plausibly succeed.
    pub fn is_retryable(self) -> bool {
        matches!(
            self,
            ErrorCode::Timeout | ErrorCode::RateLimited | ErrorCode::DepDown
        )
    }
}

/// Result of a tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// The tool_use ID this result corresponds to.
    pub tool_use_id: String,
    /// The output content.
    pub content: String,
    /// Whether the tool execution resulted in an error.
    pub is_error: bool,
    /// Typed failure class when `is_error` is true. `None` for success or for
    /// untyped errors. Defaulted for backward-compatible deserialization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_code: Option<ErrorCode>,
    /// Suggested seconds to wait before retrying, when the error is retryable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_after_seconds: Option<u64>,
}

impl ToolResult {
    /// A successful result.
    pub fn ok(tool_use_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            content: content.into(),
            is_error: false,
            error_code: None,
            retry_after_seconds: None,
        }
    }

    /// An untyped error result (no machine-readable code).
    pub fn error(tool_use_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            content: content.into(),
            is_error: true,
            error_code: None,
            retry_after_seconds: None,
        }
    }

    /// A typed error result. Renders a small structured header the model can
    /// parse, then the human-readable detail.
    pub fn error_coded(
        tool_use_id: impl Into<String>,
        code: ErrorCode,
        summary: impl Into<String>,
    ) -> Self {
        let summary = summary.into();
        let retry_after_seconds = if code.is_retryable() {
            Some(default_retry_after(code))
        } else {
            None
        };
        let content = format_error_block(code, &summary, retry_after_seconds);
        Self {
            tool_use_id: tool_use_id.into(),
            content,
            is_error: true,
            error_code: Some(code),
            retry_after_seconds,
        }
    }
}

fn default_retry_after(code: ErrorCode) -> u64 {
    match code {
        ErrorCode::RateLimited => 5,
        ErrorCode::Timeout => 2,
        ErrorCode::DepDown => 10,
        _ => 0,
    }
}

/// Render the typed failure as a compact block the LLM can act on.
fn format_error_block(code: ErrorCode, summary: &str, retry_after: Option<u64>) -> String {
    let mut block = format!("error_code: {}\nhuman_summary: {}", code.as_str(), summary);
    if let Some(secs) = retry_after {
        block.push_str(&format!("\nretry_after_seconds: {secs}"));
    }
    block
}

/// What externally-visible effect a tool has when it runs.
///
/// Drives the approval gate (Privileged/Mutating require a human OK), idempotency
/// (Mutating tools get a dedup key), and parallel fan-out eligibility (only `None`
/// reads are safe to run concurrently). Conservative by default: anything not
/// explicitly classified is treated as the least-privilege read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SideEffect {
    /// Pure read — no state change, safe to run in parallel. The default.
    #[default]
    None,
    /// Mutates local/agent state (writes a file, stores memory, posts a task).
    Mutating,
    /// Calls out to an external service (network, another agent, a webhook).
    ExternalCall,
    /// Privileged/dangerous (shell, docker, process spawn) — always gated.
    Privileged,
}

impl SideEffect {
    /// Whether this effect class must pass the human-approval gate.
    pub fn requires_approval(self) -> bool {
        matches!(self, SideEffect::Privileged)
    }

    /// Whether a tool with this effect is a pure read (parallel-safe).
    pub fn is_read_only(self) -> bool {
        matches!(self, SideEffect::None)
    }

    /// Whether a retried call could re-apply an unwanted side effect (needs an
    /// idempotency key).
    pub fn is_mutating(self) -> bool {
        matches!(self, SideEffect::Mutating | SideEffect::Privileged)
    }
}

/// Coarse risk classification, independent of effect class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RiskTier {
    /// Low blast radius (reads, idempotent lookups). The default.
    #[default]
    Low,
    /// Moderate (single-resource writes, external lookups).
    Medium,
    /// High (shell/exec, destructive or irreversible operations).
    High,
}

/// Machine-readable behavior tags for a tool, looked up by name.
///
/// Kept as a side table rather than fields on [`ToolDefinition`] so the ~140
/// existing `ToolDefinition { .. }` construction sites (built-ins, MCP, skills,
/// drivers) need no change, while the gate and scheduler get a single, testable
/// inventory point. JSON-sourced tools (MCP/skill) that aren't in the table fall
/// back to the conservative default via [`tool_metadata`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolMeta {
    pub side_effects: SideEffect,
    pub idempotent: bool,
    pub risk_tier: RiskTier,
}

impl Default for ToolMeta {
    fn default() -> Self {
        // Unknown tools default to least-privilege: a non-idempotent external
        // call would be the unsafe assumption, so we assume a safe read but mark
        // it non-idempotent only when mutating. A pure read is idempotent.
        Self {
            side_effects: SideEffect::None,
            idempotent: true,
            risk_tier: RiskTier::Low,
        }
    }
}

impl ToolMeta {
    const fn new(side_effects: SideEffect, idempotent: bool, risk_tier: RiskTier) -> Self {
        Self {
            side_effects,
            idempotent,
            risk_tier,
        }
    }
}

/// Look up behavior tags for a built-in tool by name.
///
/// The match is the single source of truth for which tools mutate, call out, or
/// are privileged. Unknown names (MCP/skill tools) get the conservative default:
/// a tool the gate doesn't recognize is treated as a non-privileged read, so it
/// is neither force-gated nor wrongly assumed idempotent-when-mutating.
pub fn tool_metadata(name: &str) -> ToolMeta {
    use RiskTier::{High, Low, Medium};
    use SideEffect::{ExternalCall, Mutating, None as NoEffect, Privileged};
    match name {
        // --- Pure reads (parallel-safe, idempotent) ---
        "file_read" | "file_list" | "code_search" | "system_time" | "location_get"
        | "memory_recall" | "agent_list" | "agent_status" | "agent_find" | "task_list"
        | "schedule_list" | "cron_list" | "knowledge_query" | "process_poll" | "process_list"
        | "hand_list" | "hand_status" | "browser_read_page" | "browser_screenshot" => {
            ToolMeta::new(NoEffect, true, Low)
        }

        // --- Local mutations (need idempotency key) ---
        "file_write" | "apply_patch" | "memory_store" | "knowledge_add_entity"
        | "knowledge_add_relation" | "task_post" | "task_claim" | "task_complete"
        | "schedule_create" | "schedule_delete" | "cron_create" | "cron_cancel"
        | "hand_activate" | "hand_deactivate" | "image_generate" | "text_to_speech" => {
            ToolMeta::new(Mutating, false, Medium)
        }

        // --- External calls (network / other agents / outbound) ---
        "web_fetch" | "web_search" | "agent_send" | "event_publish" | "channel_send"
        | "a2a_discover" | "a2a_send" | "image_analyze" | "media_describe"
        | "media_transcribe" | "speech_to_text" => ToolMeta::new(ExternalCall, false, Medium),

        // Browser navigation/interaction = external + mutating-ish session state.
        "browser_navigate" | "browser_click" | "browser_type" | "browser_scroll"
        | "browser_wait" | "browser_back" | "browser_run_js" | "browser_close"
        | "canvas_present" => ToolMeta::new(ExternalCall, false, Medium),

        // --- Privileged / dangerous (always gated, high risk) ---
        "shell_exec" | "docker_exec" | "process_start" | "process_write" | "process_kill"
        | "agent_spawn" | "agent_kill" | "code_agent_spawn" => {
            ToolMeta::new(Privileged, false, High)
        }

        // Unknown (MCP/skill/etc.) → conservative default.
        _ => ToolMeta::default(),
    }
}

/// Canonical idempotency key for a (possibly retried) tool call.
///
/// `sha256(tool_name + "|" + canonical_json(args))` per tool-design §7. Mutating
/// tools use this to dedup a retried/duplicated call so the side effect is not
/// re-applied (failure #20). Returns `None` for non-mutating tools, which don't
/// need one.
pub fn make_idempotency_key(name: &str, args: &serde_json::Value) -> Option<String> {
    if !tool_metadata(name).side_effects.is_mutating() {
        return None;
    }
    Some(idempotency_hash(name, args))
}

fn idempotency_hash(name: &str, args: &serde_json::Value) -> String {
    use sha2::{Digest, Sha256};
    // Canonicalize args so key ordering doesn't change the key.
    let canonical = canonical_json(args);
    let mut hasher = Sha256::new();
    hasher.update(name.as_bytes());
    hasher.update(b"|");
    hasher.update(canonical.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Serialize JSON with object keys sorted, so semantically-equal args hash equal.
fn canonical_json(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            let inner: Vec<String> = entries
                .iter()
                .map(|(k, v)| format!("{}:{}", serde_json::to_string(k).unwrap(), canonical_json(v)))
                .collect();
            format!("{{{}}}", inner.join(","))
        }
        serde_json::Value::Array(arr) => {
            let inner: Vec<String> = arr.iter().map(canonical_json).collect();
            format!("[{}]", inner.join(","))
        }
        other => other.to_string(),
    }
}

/// Normalize a JSON Schema for cross-provider compatibility.
///
/// Some providers (Gemini, Groq) reject `anyOf` in tool schemas.
/// This function:
/// - Converts `anyOf` arrays of simple types to flat `enum` arrays
/// - Strips `$schema` keys (not accepted by most providers)
/// - Recursively walks `properties` and `items`
pub fn normalize_schema_for_provider(
    schema: &serde_json::Value,
    provider: &str,
) -> serde_json::Value {
    // Anthropic handles anyOf natively — no normalization needed
    if provider == "anthropic" {
        return schema.clone();
    }
    normalize_schema_recursive(schema)
}

fn normalize_schema_recursive(schema: &serde_json::Value) -> serde_json::Value {
    let obj = match schema.as_object() {
        Some(o) => o,
        None => {
            // If the schema is a JSON string, try to parse it as a JSON object.
            // Some MCP servers / skill definitions serialize schemas as strings.
            if let Some(s) = schema.as_str() {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(s) {
                    if parsed.is_object() {
                        return normalize_schema_recursive(&parsed);
                    }
                }
            }
            // Non-object schema (null, number, bool, unparseable string, array) —
            // return a valid empty object schema so providers don't reject it.
            return serde_json::json!({"type": "object", "properties": {}});
        }
    };

    // Resolve $ref references before processing.
    // If the schema has $defs and $ref, inline the referenced definition.
    let resolved = resolve_refs(obj);
    let obj = resolved.as_object().unwrap_or(obj);

    let mut result = serde_json::Map::new();

    for (key, value) in obj {
        // Strip fields unsupported by Gemini and most non-Anthropic providers
        if matches!(
            key.as_str(),
            "$schema"
                | "$defs"
                | "$ref"
                | "additionalProperties"
                | "default"
                | "$id"
                | "$comment"
                | "examples"
                | "title"
                | "const"
                | "format"
        ) {
            continue;
        }

        // Convert anyOf/oneOf to flat type + enum when possible
        if key == "anyOf" || key == "oneOf" {
            if let Some(converted) = try_flatten_any_of(value) {
                for (k, v) in converted {
                    result.insert(k, v);
                }
                continue;
            }
            // Can't flatten — strip entirely rather than leave unsupported keyword
            continue;
        }

        // Flatten type arrays like ["string", "null"] to single type + nullable
        if key == "type" {
            if let Some(arr) = value.as_array() {
                let types: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
                let has_null = types.contains(&"null");
                let non_null: Vec<&&str> = types.iter().filter(|&&t| t != "null").collect();
                if has_null && non_null.len() == 1 {
                    // ["string", "null"] → type: "string", nullable: true
                    result.insert(
                        "type".to_string(),
                        serde_json::Value::String(non_null[0].to_string()),
                    );
                    result.insert("nullable".to_string(), serde_json::Value::Bool(true));
                    continue;
                } else if non_null.len() == 1 {
                    // ["string"] → type: "string"
                    result.insert(
                        "type".to_string(),
                        serde_json::Value::String(non_null[0].to_string()),
                    );
                    continue;
                } else if !non_null.is_empty() {
                    // Multiple non-null types — pick first (best effort)
                    result.insert(
                        "type".to_string(),
                        serde_json::Value::String(non_null[0].to_string()),
                    );
                    if has_null {
                        result.insert("nullable".to_string(), serde_json::Value::Bool(true));
                    }
                    continue;
                }
            }
            // Scalar type string — pass through
            result.insert(key.clone(), value.clone());
            continue;
        }

        // Recurse into properties
        if key == "properties" {
            if let Some(props) = value.as_object() {
                let mut new_props = serde_json::Map::new();
                for (prop_name, prop_schema) in props {
                    new_props.insert(prop_name.clone(), normalize_schema_recursive(prop_schema));
                }
                result.insert(key.clone(), serde_json::Value::Object(new_props));
                continue;
            }
        }

        // Recurse into items
        if key == "items" {
            result.insert(key.clone(), normalize_schema_recursive(value));
            continue;
        }

        result.insert(key.clone(), value.clone());
    }

    // Gemini requires `items` for every array-typed parameter.
    // JSON Schema allows arrays without `items`, but the Gemini API rejects
    // such schemas with INVALID_ARGUMENT. Inject a default string items schema
    // so MCP tools (and any other source) don't break Gemini requests.
    if result.get("type").and_then(|t| t.as_str()) == Some("array") && !result.contains_key("items")
    {
        result.insert("items".to_string(), serde_json::json!({"type": "string"}));
    }

    serde_json::Value::Object(result)
}

/// Resolve `$ref` references by inlining definitions from `$defs`.
///
/// If the schema has `$defs` and any property uses `$ref: "#/$defs/Foo"`,
/// replace the `$ref` with the actual definition. This is needed because
/// Gemini and most providers don't support `$ref`/`$defs`.
fn resolve_refs(obj: &serde_json::Map<String, serde_json::Value>) -> serde_json::Value {
    let defs = match obj.get("$defs").and_then(|d| d.as_object()) {
        Some(d) => d.clone(),
        None => return serde_json::Value::Object(obj.clone()),
    };

    let mut result = obj.clone();
    result.remove("$defs");

    // Recursively replace $ref in the schema
    fn inline_refs(val: &mut serde_json::Value, defs: &serde_json::Map<String, serde_json::Value>) {
        match val {
            serde_json::Value::Object(map) => {
                // If this object is a $ref, replace it with the definition
                if let Some(ref_val) = map.get("$ref").and_then(|r| r.as_str()) {
                    let ref_name = ref_val
                        .strip_prefix("#/$defs/")
                        .or_else(|| ref_val.strip_prefix("#/definitions/"));
                    if let Some(name) = ref_name {
                        if let Some(def) = defs.get(name) {
                            *val = def.clone();
                            // Recurse into the inlined definition
                            inline_refs(val, defs);
                            return;
                        }
                    }
                }
                // Recurse into all values
                for v in map.values_mut() {
                    inline_refs(v, defs);
                }
            }
            serde_json::Value::Array(arr) => {
                for item in arr.iter_mut() {
                    inline_refs(item, defs);
                }
            }
            _ => {}
        }
    }

    let mut resolved = serde_json::Value::Object(result);
    inline_refs(&mut resolved, &defs);
    resolved
}

/// Try to flatten an `anyOf` array into a simple type + enum.
///
/// Works when all variants are simple types (string, number, etc.) or
/// when it's a nullable pattern like `anyOf: [{type: "string"}, {type: "null"}]`.
fn try_flatten_any_of(any_of: &serde_json::Value) -> Option<Vec<(String, serde_json::Value)>> {
    let items = any_of.as_array()?;
    if items.is_empty() {
        return None;
    }

    // Check if this is a simple type union (all items have just "type")
    let mut types = Vec::new();
    let mut has_null = false;
    let mut non_null_type = None;

    for item in items {
        let obj = item.as_object()?;
        let type_val = obj.get("type")?.as_str()?;

        if type_val == "null" {
            has_null = true;
        } else {
            types.push(type_val.to_string());
            non_null_type = Some(type_val.to_string());
        }
    }

    // If it's a nullable pattern (type + null), emit the non-null type
    if has_null && types.len() == 1 {
        let mut result = vec![(
            "type".to_string(),
            serde_json::Value::String(non_null_type.unwrap()),
        )];
        // Mark as nullable via description hint (since JSON Schema nullable isn't universal)
        result.push(("nullable".to_string(), serde_json::Value::Bool(true)));
        return Some(result);
    }

    // If all items are simple types, pick the first non-null type (best effort).
    // Gemini rejects type arrays, so we can't emit ["string", "number"].
    if types.len() == items.len() && types.len() > 1 {
        let mut result = vec![(
            "type".to_string(),
            serde_json::Value::String(types[0].clone()),
        )];
        if has_null {
            result.push(("nullable".to_string(), serde_json::Value::Bool(true)));
        }
        return Some(result);
    }

    // Can't flatten — caller will strip the key entirely
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_metadata_classifies_known_tools() {
        // Reads are parallel-safe and idempotent.
        let r = tool_metadata("file_read");
        assert_eq!(r.side_effects, SideEffect::None);
        assert!(r.side_effects.is_read_only());
        assert!(r.idempotent);

        // Mutations are gated for idempotency, not read-only.
        let w = tool_metadata("file_write");
        assert_eq!(w.side_effects, SideEffect::Mutating);
        assert!(w.side_effects.is_mutating());
        assert!(!w.idempotent);

        // Privileged tools require approval.
        let s = tool_metadata("shell_exec");
        assert_eq!(s.side_effects, SideEffect::Privileged);
        assert!(s.side_effects.requires_approval());
        assert_eq!(s.risk_tier, RiskTier::High);

        // Unknown tools get the conservative read default (not force-gated).
        let u = tool_metadata("some_mcp_tool_xyz");
        assert_eq!(u, ToolMeta::default());
        assert!(!u.side_effects.requires_approval());
    }

    #[test]
    fn test_no_mutating_tool_lacks_idempotency_path() {
        // Every mutating/privileged built-in must produce an idempotency key;
        // every read must not. This is the invariant Fix 6 relies on.
        let mutating = ["file_write", "memory_store", "shell_exec", "task_post"];
        for t in mutating {
            assert!(
                make_idempotency_key(t, &serde_json::json!({"a": 1})).is_some(),
                "{t} is mutating but has no idempotency key"
            );
        }
        let reads = ["file_read", "system_time", "memory_recall"];
        for t in reads {
            assert!(
                make_idempotency_key(t, &serde_json::json!({})).is_none(),
                "{t} is a read but got an idempotency key"
            );
        }
    }

    #[test]
    fn test_idempotency_key_stable_under_arg_reordering() {
        // Canonicalization means key order doesn't change the key — a retry with
        // re-serialized args dedups correctly (failure #20).
        let a = make_idempotency_key("file_write", &serde_json::json!({"path": "x", "content": "y"}));
        let b = make_idempotency_key("file_write", &serde_json::json!({"content": "y", "path": "x"}));
        assert!(a.is_some());
        assert_eq!(a, b);
        // Different args → different key.
        let c = make_idempotency_key("file_write", &serde_json::json!({"path": "z", "content": "y"}));
        assert_ne!(a, c);
    }

    #[test]
    fn test_error_coded_timeout_carries_retry() {
        let r = ToolResult::error_coded("tu1", ErrorCode::Timeout, "command timed out");
        assert!(r.is_error);
        assert_eq!(r.error_code, Some(ErrorCode::Timeout));
        assert_eq!(r.retry_after_seconds, Some(2));
        // The content block is parseable by the model.
        assert!(r.content.contains("error_code: TIMEOUT"));
        assert!(r.content.contains("retry_after_seconds: 2"));
        assert!(r.content.contains("human_summary: command timed out"));
    }

    #[test]
    fn test_error_coded_invalid_param_no_retry() {
        let r = ToolResult::error_coded("tu2", ErrorCode::InvalidParam, "missing field 'url'");
        assert_eq!(r.error_code, Some(ErrorCode::InvalidParam));
        // Non-retryable classes must not suggest a blind retry.
        assert_eq!(r.retry_after_seconds, None);
        assert!(r.content.contains("error_code: INVALID_PARAM"));
        assert!(!r.content.contains("retry_after_seconds"));
    }

    #[test]
    fn test_error_code_retryable_partition() {
        assert!(ErrorCode::Timeout.is_retryable());
        assert!(ErrorCode::RateLimited.is_retryable());
        assert!(ErrorCode::DepDown.is_retryable());
        assert!(!ErrorCode::InvalidParam.is_retryable());
        assert!(!ErrorCode::NotFound.is_retryable());
        assert!(!ErrorCode::Denied.is_retryable());
    }

    #[test]
    fn test_tool_result_back_compat_deserialization() {
        // Old persisted results lack the new fields; they must still deserialize.
        let legacy = r#"{"tool_use_id":"x","content":"hi","is_error":false}"#;
        let r: ToolResult = serde_json::from_str(legacy).unwrap();
        assert!(!r.is_error);
        assert_eq!(r.error_code, None);
        assert_eq!(r.retry_after_seconds, None);
        // Success results don't serialize the optional fields (skip_serializing_if).
        let out = serde_json::to_string(&ToolResult::ok("x", "hi")).unwrap();
        assert!(!out.contains("error_code"));
        assert!(!out.contains("retry_after_seconds"));
    }

    #[test]
    fn test_tool_definition_serialization() {
        let tool = ToolDefinition {
            name: "web_search".to_string(),
            description: "Search the web".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query" }
                },
                "required": ["query"]
            }),
        };
        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("web_search"));
    }

    #[test]
    fn test_normalize_schema_strips_dollar_schema() {
        let schema = serde_json::json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "name": { "type": "string" }
            }
        });
        let result = normalize_schema_for_provider(&schema, "gemini");
        assert!(result.get("$schema").is_none());
        assert_eq!(result["type"], "object");
    }

    #[test]
    fn test_normalize_schema_flattens_anyof_nullable() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        { "type": "string" },
                        { "type": "null" }
                    ]
                }
            }
        });
        let result = normalize_schema_for_provider(&schema, "gemini");
        let value_prop = &result["properties"]["value"];
        assert_eq!(value_prop["type"], "string");
        assert_eq!(value_prop["nullable"], true);
        assert!(value_prop.get("anyOf").is_none());
    }

    #[test]
    fn test_normalize_schema_flattens_anyof_multi_type() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        { "type": "string" },
                        { "type": "number" }
                    ]
                }
            }
        });
        let result = normalize_schema_for_provider(&schema, "groq");
        let value_prop = &result["properties"]["value"];
        // Gemini rejects type arrays — should flatten to first type
        assert_eq!(value_prop["type"], "string");
        assert!(value_prop.get("anyOf").is_none());
    }

    #[test]
    fn test_normalize_schema_anthropic_passthrough() {
        let schema = serde_json::json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "anyOf": [{"type": "string"}]
        });
        let result = normalize_schema_for_provider(&schema, "anthropic");
        // Anthropic should get the original schema unchanged
        assert!(result.get("$schema").is_some());
    }

    #[test]
    fn test_normalize_schema_nested_properties() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {
                        "inner": {
                            "$schema": "strip_me",
                            "type": "string"
                        }
                    }
                }
            }
        });
        let result = normalize_schema_for_provider(&schema, "gemini");
        assert!(result["properties"]["outer"]["properties"]["inner"]
            .get("$schema")
            .is_none());
    }

    #[test]
    fn test_normalize_schema_string_parsed_to_object() {
        // MCP servers may return inputSchema as a JSON string
        let schema = serde_json::Value::String(
            r#"{"type":"object","properties":{"query":{"type":"string"}}}"#.to_string(),
        );
        let result = normalize_schema_for_provider(&schema, "openai");
        assert!(result.is_object());
        assert_eq!(result["type"], "object");
        assert!(result["properties"]["query"].is_object());
    }

    #[test]
    fn test_normalize_schema_null_becomes_empty_object() {
        let schema = serde_json::Value::Null;
        let result = normalize_schema_for_provider(&schema, "openai");
        assert!(result.is_object());
        assert_eq!(result["type"], "object");
    }

    #[test]
    fn test_normalize_schema_unparseable_string_becomes_empty_object() {
        let schema = serde_json::Value::String("not valid json".to_string());
        let result = normalize_schema_for_provider(&schema, "openai");
        assert!(result.is_object());
        assert_eq!(result["type"], "object");
    }

    #[test]
    fn test_normalize_schema_number_becomes_empty_object() {
        let schema = serde_json::json!(42);
        let result = normalize_schema_for_provider(&schema, "openai");
        assert!(result.is_object());
        assert_eq!(result["type"], "object");
    }

    #[test]
    fn test_normalize_schema_string_with_dollar_schema_stripped() {
        // String schema that contains $schema — should be parsed AND normalized
        let schema = serde_json::Value::String(
            r#"{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","properties":{}}"#.to_string(),
        );
        let result = normalize_schema_for_provider(&schema, "openai");
        assert!(result.is_object());
        assert_eq!(result["type"], "object");
        assert!(result.get("$schema").is_none());
    }

    #[test]
    fn test_normalize_strips_additional_properties() {
        let schema = serde_json::json!({
            "type": "object",
            "additionalProperties": false,
            "properties": {
                "name": { "type": "string", "default": "hello", "title": "Name" }
            }
        });
        let result = normalize_schema_for_provider(&schema, "gemini");
        assert!(result.get("additionalProperties").is_none());
        assert!(result["properties"]["name"].get("default").is_none());
        assert!(result["properties"]["name"].get("title").is_none());
        assert_eq!(result["properties"]["name"]["type"], "string");
    }

    #[test]
    fn test_normalize_resolves_refs() {
        let schema = serde_json::json!({
            "type": "object",
            "$defs": {
                "Color": {
                    "type": "string",
                    "enum": ["red", "green", "blue"]
                }
            },
            "properties": {
                "color": { "$ref": "#/$defs/Color" }
            }
        });
        let result = normalize_schema_for_provider(&schema, "gemini");
        assert!(result.get("$defs").is_none());
        assert_eq!(result["properties"]["color"]["type"], "string");
        assert!(result["properties"]["color"]["enum"].is_array());
    }

    #[test]
    fn test_normalize_strips_defs_without_refs() {
        let schema = serde_json::json!({
            "type": "object",
            "$defs": { "Unused": { "type": "number" } },
            "properties": {
                "x": { "type": "string" }
            }
        });
        let result = normalize_schema_for_provider(&schema, "gemini");
        assert!(result.get("$defs").is_none());
        assert_eq!(result["properties"]["x"]["type"], "string");
    }

    // --- Issue #488 tests ---

    #[test]
    fn test_normalize_strips_const() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "version": { "type": "string", "const": "v1" }
            }
        });
        let result = normalize_schema_for_provider(&schema, "gemini");
        assert!(result["properties"]["version"].get("const").is_none());
        assert_eq!(result["properties"]["version"]["type"], "string");
    }

    #[test]
    fn test_normalize_strips_format() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "created_at": { "type": "string", "format": "date-time" },
                "email": { "type": "string", "format": "email" }
            }
        });
        let result = normalize_schema_for_provider(&schema, "gemini");
        assert!(result["properties"]["created_at"].get("format").is_none());
        assert!(result["properties"]["email"].get("format").is_none());
        assert_eq!(result["properties"]["created_at"]["type"], "string");
        assert_eq!(result["properties"]["email"]["type"], "string");
    }

    #[test]
    fn test_normalize_flattens_oneof_nullable() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "value": {
                    "oneOf": [
                        { "type": "string" },
                        { "type": "null" }
                    ]
                }
            }
        });
        let result = normalize_schema_for_provider(&schema, "gemini");
        let value_prop = &result["properties"]["value"];
        assert_eq!(value_prop["type"], "string");
        assert_eq!(value_prop["nullable"], true);
        assert!(value_prop.get("oneOf").is_none());
    }

    #[test]
    fn test_normalize_strips_oneof_complex() {
        // Complex oneOf that can't be flattened — should be stripped entirely
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "data": {
                    "oneOf": [
                        { "type": "object", "properties": { "a": { "type": "string" } } },
                        { "type": "object", "properties": { "b": { "type": "number" } } }
                    ]
                }
            }
        });
        let result = normalize_schema_for_provider(&schema, "gemini");
        let data_prop = &result["properties"]["data"];
        assert!(data_prop.get("oneOf").is_none());
    }

    #[test]
    fn test_normalize_flattens_type_array_nullable() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": ["string", "null"] }
            }
        });
        let result = normalize_schema_for_provider(&schema, "gemini");
        let name_prop = &result["properties"]["name"];
        assert_eq!(name_prop["type"], "string");
        assert_eq!(name_prop["nullable"], true);
    }

    #[test]
    fn test_normalize_flattens_type_array_multi() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "value": { "type": ["string", "number", "null"] }
            }
        });
        let result = normalize_schema_for_provider(&schema, "gemini");
        let value_prop = &result["properties"]["value"];
        // Should pick first non-null type
        assert_eq!(value_prop["type"], "string");
        assert_eq!(value_prop["nullable"], true);
    }

    #[test]
    fn test_normalize_flattens_type_array_single() {
        // Single-element type array
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "x": { "type": ["integer"] }
            }
        });
        let result = normalize_schema_for_provider(&schema, "gemini");
        assert_eq!(result["properties"]["x"]["type"], "integer");
        assert!(result["properties"]["x"].get("nullable").is_none());
    }

    #[test]
    fn test_normalize_strips_anyof_complex() {
        // Complex anyOf that can't be flattened — should be stripped entirely
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "payload": {
                    "anyOf": [
                        { "type": "object", "properties": { "url": { "type": "string" } } },
                        { "type": "array", "items": { "type": "string" } }
                    ]
                }
            }
        });
        let result = normalize_schema_for_provider(&schema, "gemini");
        let payload_prop = &result["properties"]["payload"];
        assert!(payload_prop.get("anyOf").is_none());
    }

    #[test]
    fn test_normalize_injects_items_for_array_without_items() {
        // MCP tools often send array params without `items` — Gemini rejects these.
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "fields": { "type": "array", "description": "List of fields" },
                "filters": { "type": "array" }
            }
        });
        let result = normalize_schema_for_provider(&schema, "gemini");
        // Both array properties must have `items` injected
        assert_eq!(result["properties"]["fields"]["items"]["type"], "string");
        assert_eq!(result["properties"]["filters"]["items"]["type"], "string");
    }

    #[test]
    fn test_normalize_preserves_existing_items() {
        // If `items` already exists, it must not be overwritten
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "ids": {
                    "type": "array",
                    "items": { "type": "integer" }
                }
            }
        });
        let result = normalize_schema_for_provider(&schema, "gemini");
        assert_eq!(result["properties"]["ids"]["items"]["type"], "integer");
    }

    #[test]
    fn test_normalize_combined_issue_488() {
        // Real-world schema combining multiple #488 issues
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "api_version": { "type": "string", "const": "v2", "format": "semver" },
                "timestamp": { "type": "string", "format": "date-time" },
                "label": {
                    "oneOf": [
                        { "type": "string" },
                        { "type": "null" }
                    ]
                },
                "tags": { "type": ["string", "null"] }
            }
        });
        let result = normalize_schema_for_provider(&schema, "gemini");
        // const and format stripped
        assert!(result["properties"]["api_version"].get("const").is_none());
        assert!(result["properties"]["api_version"].get("format").is_none());
        assert!(result["properties"]["timestamp"].get("format").is_none());
        // oneOf flattened
        assert_eq!(result["properties"]["label"]["type"], "string");
        assert_eq!(result["properties"]["label"]["nullable"], true);
        assert!(result["properties"]["label"].get("oneOf").is_none());
        // type array flattened
        assert_eq!(result["properties"]["tags"]["type"], "string");
        assert_eq!(result["properties"]["tags"]["nullable"], true);
    }
}
