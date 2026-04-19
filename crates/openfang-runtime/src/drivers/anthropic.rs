//! Anthropic Claude API driver.
//!
//! Full implementation of the Anthropic Messages API with tool use support,
//! system prompt extraction, and retry on 429/529 errors.

use crate::llm_driver::{CompletionRequest, CompletionResponse, LlmDriver, LlmError, StreamEvent};
use async_trait::async_trait;
use futures::StreamExt;
use openfang_types::message::{
    ContentBlock, Message, MessageContent, Role, StopReason, TokenUsage,
};
use openfang_types::tool::ToolCall;
use serde::{Deserialize, Serialize};
use tracing::{debug, warn};
use zeroize::Zeroizing;

/// Anthropic Claude API driver.
pub struct AnthropicDriver {
    api_key: Zeroizing<String>,
    base_url: String,
    client: reqwest::Client,
}

impl AnthropicDriver {
    /// Create a new Anthropic driver.
    pub fn new(api_key: String, base_url: String) -> Self {
        Self {
            api_key: Zeroizing::new(api_key),
            base_url,
            client: reqwest::Client::builder()
                .user_agent(crate::USER_AGENT)
                .build()
                .unwrap_or_default(),
        }
    }
}

/// Anthropic Messages API request body.
///
/// The `system` field is emitted as an array of content blocks (not a bare
/// string) so that prompt caching can attach `cache_control: {type: "ephemeral"}`
/// to the stable prefix. Anthropic accepts either shape; the block form is
/// always used so the caching path has one code site.
#[derive(Debug, Serialize)]
struct ApiRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<Vec<ApiSystemBlock>>,
    messages: Vec<ApiMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ApiTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
}

/// One block in the Anthropic `system` array. The optional `cache_control`
/// marker enables provider prompt caching for this block.
#[derive(Debug, Serialize)]
struct ApiSystemBlock {
    #[serde(rename = "type")]
    block_type: &'static str,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<ApiCacheControl>,
}

/// Anthropic cache_control marker. Only `ephemeral` (5-min TTL) is emitted;
/// the `ttl: "1h"` beta is scoped out of v1.
#[derive(Debug, Serialize)]
struct ApiCacheControl {
    #[serde(rename = "type")]
    cache_type: &'static str,
}

#[derive(Debug, Serialize)]
struct ApiMessage {
    role: String,
    content: ApiContent,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum ApiContent {
    Text(String),
    Blocks(Vec<ApiContentBlock>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum ApiContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: ApiImageSource },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
    },
}

#[derive(Debug, Serialize)]
struct ApiImageSource {
    #[serde(rename = "type")]
    source_type: String,
    media_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
struct ApiTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

/// Anthropic Messages API response body.
#[derive(Debug, Deserialize)]
struct ApiResponse {
    content: Vec<ResponseContentBlock>,
    stop_reason: String,
    usage: ApiUsage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ResponseContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
}

#[derive(Debug, Default, Deserialize)]
struct ApiUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    /// Tokens written to prompt cache on this turn (billed at 1.25x input rate).
    /// Absent on older API responses / non-caching requests.
    #[serde(default)]
    cache_creation_input_tokens: u64,
    /// Tokens read from prompt cache on this turn (billed at 0.1x input rate).
    #[serde(default)]
    cache_read_input_tokens: u64,
}

/// Anthropic API error response.
#[derive(Debug, Deserialize)]
struct ApiErrorResponse {
    error: ApiErrorDetail,
}

#[derive(Debug, Deserialize)]
struct ApiErrorDetail {
    message: String,
}

/// Accumulator for content blocks during streaming.
enum ContentBlockAccum {
    Text(String),
    Thinking(String),
    ToolUse {
        id: String,
        name: String,
        input_json: String,
    },
}

/// Rough character-based token estimator used by the caching gate. The exact
/// Anthropic tokenizer is not exposed; 4 chars/token is the published rule of
/// thumb and is good enough for deciding whether to attach `cache_control`.
fn estimate_tokens(text: &str) -> u32 {
    (text.len() / 4) as u32
}

/// Build the wire-level `ApiRequest` from a `CompletionRequest`.
///
/// Handles system-prompt extraction (from `request.system` or any `Role::System`
/// message), wraps it in an `ApiSystemBlock`, and attaches `cache_control:
/// {type: "ephemeral"}` when the caller requested caching *and* the system
/// prompt meets the model's minimum cacheable token threshold.
///
/// Shared by `complete()` and `stream()` so the caching logic lives in one
/// place — adding it to only one site and missing the other is a real risk
/// called out in the plan review (finding L3).
fn build_api_request(request: &CompletionRequest, stream: bool) -> ApiRequest {
    // Extract the system prompt from either the explicit field or a Role::System message.
    let system_text = request.system.clone().or_else(|| {
        request.messages.iter().find_map(|m| {
            if m.role == Role::System {
                match &m.content {
                    MessageContent::Text(t) => Some(t.clone()),
                    _ => None,
                }
            } else {
                None
            }
        })
    });

    // Decide whether this block qualifies for cache_control.
    let system = system_text.map(|text| {
        let tokens = estimate_tokens(&text);
        let attach_cache = request.cache_system_prompt
            && request.min_cache_tokens > 0
            && tokens >= request.min_cache_tokens;
        if request.cache_system_prompt && !attach_cache {
            debug!(
                tokens,
                threshold = request.min_cache_tokens,
                "anthropic: system prompt below cache threshold; sending uncached"
            );
        }
        vec![ApiSystemBlock {
            block_type: "text",
            text,
            cache_control: if attach_cache {
                Some(ApiCacheControl { cache_type: "ephemeral" })
            } else {
                None
            },
        }]
    });

    let api_messages: Vec<ApiMessage> = request
        .messages
        .iter()
        .filter(|m| m.role != Role::System)
        .map(convert_message)
        .collect();

    let api_tools: Vec<ApiTool> = request
        .tools
        .iter()
        .map(|t| ApiTool {
            name: t.name.clone(),
            description: t.description.clone(),
            input_schema: t.input_schema.clone(),
        })
        .collect();

    ApiRequest {
        model: request.model.clone(),
        max_tokens: request.max_tokens,
        system,
        messages: api_messages,
        tools: api_tools,
        temperature: Some(request.temperature),
        stream,
    }
}

#[async_trait]
impl LlmDriver for AnthropicDriver {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, LlmError> {
        let api_request = build_api_request(&request, false);

        // Retry loop for rate limits and overloads
        let max_retries = 3;
        for attempt in 0..=max_retries {
            let url = format!("{}/v1/messages", self.base_url);
            debug!(url = %url, attempt, "Sending Anthropic API request");

            let resp = self
                .client
                .post(&url)
                .header("x-api-key", self.api_key.as_str())
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .json(&api_request)
                .send()
                .await
                .map_err(|e| LlmError::Http(e.to_string()))?;

            let status = resp.status().as_u16();

            if status == 429 || status == 529 {
                if attempt < max_retries {
                    let retry_ms = (attempt + 1) as u64 * 2000;
                    warn!(status, retry_ms, "Rate limited, retrying");
                    tokio::time::sleep(std::time::Duration::from_millis(retry_ms)).await;
                    continue;
                }
                return Err(if status == 429 {
                    LlmError::RateLimited {
                        retry_after_ms: 5000,
                    }
                } else {
                    LlmError::Overloaded {
                        retry_after_ms: 5000,
                    }
                });
            }

            if !resp.status().is_success() {
                let body = resp.text().await.unwrap_or_default();
                let message = serde_json::from_str::<ApiErrorResponse>(&body)
                    .map(|e| e.error.message)
                    .unwrap_or(body);
                return Err(LlmError::Api { status, message });
            }

            let body = resp
                .text()
                .await
                .map_err(|e| LlmError::Http(e.to_string()))?;
            let api_response: ApiResponse =
                serde_json::from_str(&body).map_err(|e| LlmError::Parse(e.to_string()))?;

            return Ok(convert_response(api_response));
        }

        Err(LlmError::Api {
            status: 0,
            message: "Max retries exceeded".to_string(),
        })
    }

    async fn stream(
        &self,
        request: CompletionRequest,
        tx: tokio::sync::mpsc::Sender<StreamEvent>,
    ) -> Result<CompletionResponse, LlmError> {
        let api_request = build_api_request(&request, true);

        // Retry loop for the initial HTTP request
        let max_retries = 3;
        for attempt in 0..=max_retries {
            let url = format!("{}/v1/messages", self.base_url);
            debug!(url = %url, attempt, "Sending Anthropic streaming request");

            let resp = self
                .client
                .post(&url)
                .header("x-api-key", self.api_key.as_str())
                .header("anthropic-version", "2023-06-01")
                .header("content-type", "application/json")
                .json(&api_request)
                .send()
                .await
                .map_err(|e| LlmError::Http(e.to_string()))?;

            let status = resp.status().as_u16();

            if status == 429 || status == 529 {
                if attempt < max_retries {
                    let retry_ms = (attempt + 1) as u64 * 2000;
                    warn!(status, retry_ms, "Rate limited (stream), retrying");
                    tokio::time::sleep(std::time::Duration::from_millis(retry_ms)).await;
                    continue;
                }
                return Err(if status == 429 {
                    LlmError::RateLimited {
                        retry_after_ms: 5000,
                    }
                } else {
                    LlmError::Overloaded {
                        retry_after_ms: 5000,
                    }
                });
            }

            if !resp.status().is_success() {
                let body = resp.text().await.unwrap_or_default();
                let message = serde_json::from_str::<ApiErrorResponse>(&body)
                    .map(|e| e.error.message)
                    .unwrap_or(body);
                return Err(LlmError::Api { status, message });
            }

            // Parse the SSE stream
            let mut buffer = String::new();
            let mut blocks: Vec<ContentBlockAccum> = Vec::new();
            let mut stop_reason = StopReason::EndTurn;
            let mut usage = TokenUsage::default();

            let mut byte_stream = resp.bytes_stream();
            while let Some(chunk_result) = byte_stream.next().await {
                let chunk = chunk_result.map_err(|e| LlmError::Http(e.to_string()))?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(pos) = buffer.find("\n\n") {
                    let event_text = buffer[..pos].to_string();
                    buffer = buffer[pos + 2..].to_string();

                    let mut event_type = String::new();
                    let mut data = String::new();
                    for line in event_text.lines() {
                        if let Some(et) = line.strip_prefix("event:") {
                            event_type = et.trim_start().to_string();
                        } else if let Some(d) = line.strip_prefix("data:") {
                            data = d.trim_start().to_string();
                        }
                    }

                    if data.is_empty() {
                        continue;
                    }

                    let json: serde_json::Value = match serde_json::from_str(&data) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    match event_type.as_str() {
                        "message_start" => {
                            // Read input + cache tokens from the initial message_start event.
                            // Without the cache-field reads, every streaming call reports
                            // cache_read=0 even on hits and the merge-gate test fails silently.
                            let u = &json["message"]["usage"];
                            if let Some(it) = u["input_tokens"].as_u64() {
                                usage.input_tokens = it;
                            }
                            if let Some(c) = u["cache_creation_input_tokens"].as_u64() {
                                usage.cache_creation_input_tokens = c;
                            }
                            if let Some(r) = u["cache_read_input_tokens"].as_u64() {
                                usage.cache_read_input_tokens = r;
                            }
                        }
                        "content_block_start" => {
                            let block = &json["content_block"];
                            match block["type"].as_str().unwrap_or("") {
                                "text" => {
                                    blocks.push(ContentBlockAccum::Text(String::new()));
                                }
                                "tool_use" => {
                                    let id = block["id"].as_str().unwrap_or("").to_string();
                                    let name = block["name"].as_str().unwrap_or("").to_string();
                                    let _ = tx
                                        .send(StreamEvent::ToolUseStart {
                                            id: id.clone(),
                                            name: name.clone(),
                                        })
                                        .await;
                                    blocks.push(ContentBlockAccum::ToolUse {
                                        id,
                                        name,
                                        input_json: String::new(),
                                    });
                                }
                                "thinking" => {
                                    blocks.push(ContentBlockAccum::Thinking(String::new()));
                                }
                                _ => {}
                            }
                        }
                        "content_block_delta" => {
                            let block_idx = json["index"].as_u64().unwrap_or(0) as usize;
                            let delta = &json["delta"];
                            match delta["type"].as_str().unwrap_or("") {
                                "text_delta" => {
                                    if let Some(text) = delta["text"].as_str() {
                                        if let Some(ContentBlockAccum::Text(ref mut t)) =
                                            blocks.get_mut(block_idx)
                                        {
                                            t.push_str(text);
                                        }
                                        let _ = tx
                                            .send(StreamEvent::TextDelta {
                                                text: text.to_string(),
                                            })
                                            .await;
                                    }
                                }
                                "input_json_delta" => {
                                    if let Some(partial) = delta["partial_json"].as_str() {
                                        if let Some(ContentBlockAccum::ToolUse {
                                            ref mut input_json,
                                            ..
                                        }) = blocks.get_mut(block_idx)
                                        {
                                            input_json.push_str(partial);
                                        }
                                        let _ = tx
                                            .send(StreamEvent::ToolInputDelta {
                                                text: partial.to_string(),
                                            })
                                            .await;
                                    }
                                }
                                "thinking_delta" => {
                                    if let Some(thinking) = delta["thinking"].as_str() {
                                        if let Some(ContentBlockAccum::Thinking(ref mut t)) =
                                            blocks.get_mut(block_idx)
                                        {
                                            t.push_str(thinking);
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                        "content_block_stop" => {
                            let block_idx = json["index"].as_u64().unwrap_or(0) as usize;
                            if let Some(ContentBlockAccum::ToolUse {
                                id,
                                name,
                                input_json,
                            }) = blocks.get(block_idx)
                            {
                                let input: serde_json::Value = serde_json::from_str(input_json)
                                    .unwrap_or_else(|_| serde_json::json!({}));
                                let _ = tx
                                    .send(StreamEvent::ToolUseEnd {
                                        id: id.clone(),
                                        name: name.clone(),
                                        input,
                                    })
                                    .await;
                            }
                        }
                        "message_delta" => {
                            if let Some(sr) = json["delta"]["stop_reason"].as_str() {
                                stop_reason = match sr {
                                    "end_turn" => StopReason::EndTurn,
                                    "tool_use" => StopReason::ToolUse,
                                    "max_tokens" => StopReason::MaxTokens,
                                    "stop_sequence" => StopReason::StopSequence,
                                    _ => StopReason::EndTurn,
                                };
                            }
                            if let Some(ot) = json["usage"]["output_tokens"].as_u64() {
                                usage.output_tokens = ot;
                            }
                        }
                        _ => {} // message_stop, ping, etc.
                    }
                }
            }

            // Build CompletionResponse from accumulated blocks
            let mut content = Vec::new();
            let mut tool_calls = Vec::new();
            for block in blocks {
                match block {
                    ContentBlockAccum::Text(text) => {
                        content.push(ContentBlock::Text {
                            text,
                            provider_metadata: None,
                        });
                    }
                    ContentBlockAccum::Thinking(thinking) => {
                        content.push(ContentBlock::Thinking { thinking });
                    }
                    ContentBlockAccum::ToolUse {
                        id,
                        name,
                        input_json,
                    } => {
                        let input: serde_json::Value = serde_json::from_str(&input_json)
                            .unwrap_or_else(|_| serde_json::json!({}));
                        content.push(ContentBlock::ToolUse {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                            provider_metadata: None,
                        });
                        tool_calls.push(ToolCall { id, name, input });
                    }
                }
            }

            let _ = tx
                .send(StreamEvent::ContentComplete { stop_reason, usage })
                .await;

            return Ok(CompletionResponse {
                content,
                stop_reason,
                tool_calls,
                usage,
            });
        }

        Err(LlmError::Api {
            status: 0,
            message: "Max retries exceeded".to_string(),
        })
    }
}

/// Ensure a `serde_json::Value` is a JSON object (dictionary).
///
/// The Anthropic API requires `tool_use.input` to be a JSON object, never a
/// string, null, or other scalar.  This helper handles:
///  - `Value::Object` → returned as-is
///  - `Value::String` → attempt to parse as JSON; if the result is an object, use it
///  - anything else (Null, Number, Bool, Array) → empty object `{}`
fn ensure_object(v: &serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Object(_) => v.clone(),
        serde_json::Value::String(s) => {
            // The input may have been double-serialized (stored as a JSON string).
            // Try to parse it back into a Value.
            match serde_json::from_str::<serde_json::Value>(s) {
                Ok(parsed) if parsed.is_object() => parsed,
                _ => serde_json::json!({}),
            }
        }
        _ => serde_json::json!({}),
    }
}

/// Convert an OpenFang Message to an Anthropic API message.
fn convert_message(msg: &Message) -> ApiMessage {
    let role = match msg.role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::System => "user", // Should be filtered out, but handle gracefully
    };

    let content = match &msg.content {
        MessageContent::Text(text) => ApiContent::Text(text.clone()),
        MessageContent::Blocks(blocks) => {
            let api_blocks: Vec<ApiContentBlock> = blocks
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text { text, .. } => {
                        Some(ApiContentBlock::Text { text: text.clone() })
                    }
                    ContentBlock::Image { media_type, data } => Some(ApiContentBlock::Image {
                        source: ApiImageSource {
                            source_type: "base64".to_string(),
                            media_type: media_type.clone(),
                            data: data.clone(),
                        },
                    }),
                    ContentBlock::ToolUse {
                        id, name, input, ..
                    } => Some(ApiContentBlock::ToolUse {
                        id: id.clone(),
                        name: name.clone(),
                        input: ensure_object(input),
                    }),
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                        ..
                    } => Some(ApiContentBlock::ToolResult {
                        tool_use_id: tool_use_id.clone(),
                        content: content.clone(),
                        is_error: *is_error,
                    }),
                    ContentBlock::Thinking { .. } => None,
                    ContentBlock::Unknown => None,
                })
                .collect();
            ApiContent::Blocks(api_blocks)
        }
    };

    ApiMessage {
        role: role.to_string(),
        content,
    }
}

/// Convert an Anthropic API response to our CompletionResponse.
fn convert_response(api: ApiResponse) -> CompletionResponse {
    let mut content = Vec::new();
    let mut tool_calls = Vec::new();

    for block in api.content {
        match block {
            ResponseContentBlock::Text { text } => {
                content.push(ContentBlock::Text {
                    text,
                    provider_metadata: None,
                });
            }
            ResponseContentBlock::ToolUse { id, name, input } => {
                content.push(ContentBlock::ToolUse {
                    id: id.clone(),
                    name: name.clone(),
                    input: input.clone(),
                    provider_metadata: None,
                });
                tool_calls.push(ToolCall { id, name, input });
            }
            ResponseContentBlock::Thinking { thinking } => {
                content.push(ContentBlock::Thinking { thinking });
            }
        }
    }

    let stop_reason = match api.stop_reason.as_str() {
        "end_turn" => StopReason::EndTurn,
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        "stop_sequence" => StopReason::StopSequence,
        _ => StopReason::EndTurn,
    };

    CompletionResponse {
        content,
        stop_reason,
        tool_calls,
        usage: TokenUsage {
            input_tokens: api.usage.input_tokens,
            output_tokens: api.usage.output_tokens,
            cache_creation_input_tokens: api.usage.cache_creation_input_tokens,
            cache_read_input_tokens: api.usage.cache_read_input_tokens,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_message_text() {
        let msg = Message::user("Hello");
        let api_msg = convert_message(&msg);
        assert_eq!(api_msg.role, "user");
    }

    #[test]
    fn test_convert_response() {
        let api_response = ApiResponse {
            content: vec![
                ResponseContentBlock::Text {
                    text: "I'll help you.".to_string(),
                },
                ResponseContentBlock::ToolUse {
                    id: "tool_1".to_string(),
                    name: "web_search".to_string(),
                    input: serde_json::json!({"query": "rust lang"}),
                },
            ],
            stop_reason: "tool_use".to_string(),
            usage: ApiUsage {
                input_tokens: 100,
                output_tokens: 50,
                ..Default::default()
            },
        };

        let response = convert_response(api_response);
        assert_eq!(response.stop_reason, StopReason::ToolUse);
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(response.tool_calls[0].name, "web_search");
        assert_eq!(response.usage.total(), 150);
    }

    #[test]
    fn test_ensure_object_from_object() {
        let obj = serde_json::json!({"key": "value"});
        assert_eq!(ensure_object(&obj), obj);
    }

    #[test]
    fn test_ensure_object_from_string() {
        // Simulates double-serialized input (stored as JSON string)
        let stringified = serde_json::Value::String(r#"{"query": "rust"}"#.to_string());
        let result = ensure_object(&stringified);
        assert_eq!(result, serde_json::json!({"query": "rust"}));
    }

    #[test]
    fn test_ensure_object_from_null() {
        let null = serde_json::Value::Null;
        assert_eq!(ensure_object(&null), serde_json::json!({}));
    }

    #[test]
    fn test_ensure_object_from_non_object_string() {
        // A string that parses to a non-object JSON value
        let s = serde_json::Value::String("42".to_string());
        assert_eq!(ensure_object(&s), serde_json::json!({}));
    }

    #[test]
    fn test_ensure_object_from_invalid_json_string() {
        let s = serde_json::Value::String("not json at all".to_string());
        assert_eq!(ensure_object(&s), serde_json::json!({}));
    }

    #[test]
    fn test_convert_message_normalizes_tool_input() {
        // Simulate a ToolUse block with a stringified JSON input (legacy session data)
        let msg = Message {
            role: Role::Assistant,
            content: MessageContent::Blocks(vec![ContentBlock::ToolUse {
                id: "tu-1".to_string(),
                name: "web_search".to_string(),
                input: serde_json::Value::String(r#"{"query": "test"}"#.to_string()),
                provider_metadata: None,
            }]),
        };
        let api_msg = convert_message(&msg);
        if let ApiContent::Blocks(blocks) = api_msg.content {
            match &blocks[0] {
                ApiContentBlock::ToolUse { input, .. } => {
                    assert!(input.is_object(), "input should be an object, got: {input}");
                    assert_eq!(input["query"], "test");
                }
                _ => panic!("Expected ToolUse block"),
            }
        } else {
            panic!("Expected Blocks content");
        }
    }

    // ── Prompt caching tests ─────────────────────────────────────────────

    /// Build a minimal CompletionRequest for caching tests.
    fn cache_test_request(
        system: &str,
        cache_system_prompt: bool,
        min_cache_tokens: u32,
    ) -> CompletionRequest {
        CompletionRequest {
            model: "claude-sonnet-4-6".to_string(),
            messages: vec![Message::user("hi")],
            tools: vec![],
            max_tokens: 100,
            temperature: 0.7,
            system: Some(system.to_string()),
            thinking: None,
            cache_system_prompt,
            min_cache_tokens,
        }
    }

    #[test]
    fn caching_disabled_emits_plain_block() {
        // Long system prompt but cache flag off → no cache_control.
        let system = "x".repeat(8192); // ~2048 tokens
        let req = cache_test_request(&system, false, 1024);
        let api = build_api_request(&req, false);
        let blocks = api.system.expect("system present");
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].cache_control.is_none());
    }

    #[test]
    fn caching_enabled_above_threshold_emits_cache_control() {
        // ~2048 tokens system prompt + caching on → cache_control attached.
        let system = "x".repeat(8192);
        let req = cache_test_request(&system, true, 1024);
        let api = build_api_request(&req, false);
        let blocks = api.system.expect("system present");
        assert_eq!(blocks.len(), 1);
        let cc = blocks[0].cache_control.as_ref().expect("cache_control present");
        assert_eq!(cc.cache_type, "ephemeral");
    }

    #[test]
    fn caching_enabled_below_threshold_emits_plain_block() {
        // Short system prompt (well below 1024-token threshold) → no cache_control.
        let req = cache_test_request("short system", true, 1024);
        let api = build_api_request(&req, false);
        let blocks = api.system.expect("system present");
        assert_eq!(blocks.len(), 1);
        assert!(
            blocks[0].cache_control.is_none(),
            "sub-threshold prompt must not emit cache_control (Anthropic ignores + charges 1.25x)"
        );
    }

    #[test]
    fn caching_with_zero_threshold_disabled() {
        // min_cache_tokens=0 is the "caching not supported" signal from non-Anthropic
        // models. Even with cache_system_prompt=true, no cache_control should emit.
        let system = "x".repeat(8192);
        let req = cache_test_request(&system, true, 0);
        let api = build_api_request(&req, false);
        let blocks = api.system.expect("system present");
        assert!(blocks[0].cache_control.is_none());
    }

    #[test]
    fn api_usage_cache_fields_default_when_absent() {
        // Legacy API response without cache fields should deserialize as 0.
        let body = r#"{"input_tokens": 100, "output_tokens": 50}"#;
        let usage: ApiUsage = serde_json::from_str(body).unwrap();
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.cache_creation_input_tokens, 0);
        assert_eq!(usage.cache_read_input_tokens, 0);
    }

    #[test]
    fn api_usage_cache_fields_parsed_when_present() {
        let body = r#"{
            "input_tokens": 10,
            "output_tokens": 50,
            "cache_creation_input_tokens": 1800,
            "cache_read_input_tokens": 0
        }"#;
        let usage: ApiUsage = serde_json::from_str(body).unwrap();
        assert_eq!(usage.cache_creation_input_tokens, 1800);
        assert_eq!(usage.cache_read_input_tokens, 0);
    }

}
