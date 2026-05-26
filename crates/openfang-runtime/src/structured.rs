//! Schema-constrained structured output from an LLM.
//!
//! Forces the model to emit JSON matching a caller-supplied schema by presenting
//! a single tool whose `input_schema` is that schema, then deserializing the
//! tool call's `input` into a typed value. Falls back to extracting a JSON
//! object from the response text when a provider returns the payload as prose
//! instead of a native tool call (e.g. the Groq/Llama text-tool-call path).
//!
//! This reuses OpenFang's existing tool-calling machinery, which is already
//! wired across every driver, rather than adding a `response_format` field that
//! each driver would have to implement separately.

use crate::llm_driver::{CompletionRequest, LlmDriver};
use openfang_types::message::{ContentBlock, Message, MessageContent, Role};
use openfang_types::tool::ToolDefinition;
use serde::de::DeserializeOwned;
use std::sync::Arc;
use tracing::warn;

/// Options for a structured-output call. Sensible defaults via [`Default`].
#[derive(Debug, Clone)]
pub struct StructuredOptions {
    /// Sampling temperature. Low by default for deterministic extraction.
    pub temperature: f32,
    /// Maximum tokens to generate.
    pub max_tokens: u32,
    /// Number of attempts before giving up. Each retry is a fresh call.
    pub max_retries: u32,
}

impl Default for StructuredOptions {
    fn default() -> Self {
        Self {
            temperature: 0.1,
            max_tokens: 2048,
            max_retries: 2,
        }
    }
}

/// Why a structured-output call failed.
#[derive(Debug, thiserror::Error)]
pub enum StructuredError {
    /// The driver call failed on every attempt.
    #[error("LLM call failed: {0}")]
    Driver(String),
    /// A response was produced but no JSON could be located or parsed.
    #[error("no parseable structured output after {attempts} attempt(s): {last}")]
    NoOutput {
        /// How many attempts were made.
        attempts: u32,
        /// The last parse/deserialize error encountered.
        last: String,
    },
}

/// Request `T` from `model`, constrained to JSON matching `schema`.
///
/// `schema` is a JSON Schema object describing `T`. `system` is an optional
/// system prompt; `user_prompt` is the instruction/content to act on. Returns a
/// deserialized `T`, or [`StructuredError`] if no attempt yields valid output.
pub async fn complete_structured<T: DeserializeOwned>(
    driver: Arc<dyn LlmDriver>,
    model: &str,
    system: Option<String>,
    user_prompt: &str,
    schema: serde_json::Value,
    opts: &StructuredOptions,
) -> Result<T, StructuredError> {
    const TOOL_NAME: &str = "emit_structured_output";

    let tool = ToolDefinition {
        name: TOOL_NAME.to_string(),
        description:
            "Return the requested result by calling this tool exactly once with the structured \
             arguments. Do not reply with prose."
                .to_string(),
        input_schema: schema,
    };

    let request = CompletionRequest {
        model: model.to_string(),
        messages: vec![Message {
            role: Role::User,
            content: MessageContent::Blocks(vec![ContentBlock::Text {
                text: user_prompt.to_string(),
                provider_metadata: None,
            }]),
        }],
        tools: vec![tool],
        max_tokens: opts.max_tokens,
        temperature: opts.temperature,
        system,
        thinking: None,
        cache_system_prompt: false,
        min_cache_tokens: 0,
        mcp_config_path: None,
    };

    let mut last_err = String::new();
    let attempts = opts.max_retries.max(1);
    for attempt in 0..attempts {
        let response = match driver.complete(request.clone()).await {
            Ok(r) => r,
            Err(e) => {
                last_err = e.to_string();
                warn!(attempt, error = %last_err, "structured: driver call failed");
                continue;
            }
        };

        let raw = response
            .tool_calls
            .iter()
            .find(|c| c.name == TOOL_NAME)
            .map(|c| c.input.clone())
            .or_else(|| extract_json_object(&response.text()));

        let Some(value) = raw else {
            last_err = "model returned neither a tool call nor a JSON object".to_string();
            warn!(attempt, "structured: no JSON in response");
            continue;
        };

        match serde_json::from_value::<T>(value) {
            Ok(parsed) => return Ok(parsed),
            Err(e) => {
                last_err = e.to_string();
                warn!(attempt, error = %last_err, "structured: deserialize failed");
            }
        }
    }

    if last_err.contains("driver call failed") {
        Err(StructuredError::Driver(last_err))
    } else {
        Err(StructuredError::NoOutput {
            attempts,
            last: last_err,
        })
    }
}

/// Extract the first balanced top-level JSON object from `text`, tolerating
/// surrounding prose and markdown code fences. Returns `None` if no object
/// parses cleanly.
fn extract_json_object(text: &str) -> Option<serde_json::Value> {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(text.trim()) {
        if v.is_object() {
            return Some(v);
        }
    }

    let bytes = text.as_bytes();
    let start = text.find('{')?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, &b) in bytes[start..].iter().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    let candidate = &text[start..=start + offset];
                    return serde_json::from_str::<serde_json::Value>(candidate)
                        .ok()
                        .filter(serde_json::Value::is_object);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_bare_object() {
        let v = extract_json_object(r#"{"a":1,"b":"x"}"#).unwrap();
        assert_eq!(v["a"], 1);
        assert_eq!(v["b"], "x");
    }

    #[test]
    fn extracts_object_from_fenced_prose() {
        let text = "Here is the result:\n```json\n{\"tag\":\"gotcha\",\"n\":2}\n```\nDone.";
        let v = extract_json_object(text).unwrap();
        assert_eq!(v["tag"], "gotcha");
        assert_eq!(v["n"], 2);
    }

    #[test]
    fn ignores_braces_inside_strings() {
        let v = extract_json_object(r#"prefix {"msg":"a } b","ok":true} suffix"#).unwrap();
        assert_eq!(v["msg"], "a } b");
        assert_eq!(v["ok"], true);
    }

    #[test]
    fn returns_none_when_no_object() {
        assert!(extract_json_object("no json here").is_none());
        assert!(extract_json_object("[1,2,3]").is_none());
    }
}
