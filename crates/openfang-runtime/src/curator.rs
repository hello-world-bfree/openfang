//! Post-run learning distillation ("Repo Brain").
//!
//! After a successful agent run, [`distill_learnings`] asks the model to extract
//! 0–3 durable, tagged lessons from the run and persists them as `learning`-scoped
//! memories. Future runs recall them, so an agent accumulates knowledge across
//! sessions instead of relearning it each time.
//!
//! Gated behind [`MemoryConfig::curator_enabled`] (opt-in) and a per-run cost cap,
//! so it only fires when wanted and never on an already-expensive run. Reuses the
//! agent's own model and the existing [`crate::structured`] tool-call path.

use crate::llm_driver::LlmDriver;
use crate::structured::{complete_structured, StructuredOptions};
use openfang_memory::MemorySubstrate;
use openfang_types::agent::AgentId;
use openfang_types::memory::{Memory, MemoryFilter, MemorySource};
use openfang_types::message::{ContentBlock, Message, MessageContent, Role};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, warn};

/// Memory scope under which distilled learnings are stored and recalled.
pub const LEARNING_SCOPE: &str = "learning";

/// Minimum stored confidence a learning needs to be injected into a future prompt.
///
/// The curator stores `confidence = 0.1 + (model_confidence) * 0.9` (via the
/// `importance` metadata path in the semantic store), so `0.7` admits learnings
/// the model rated at roughly two-thirds confidence or higher. This is the primary
/// guard against memory pollution — weak lessons never reach the prompt.
pub const READBACK_MIN_CONFIDENCE: f32 = 0.7;

/// How many learnings to inject at most. Kept small to bound prompt growth.
const READBACK_LIMIT: usize = 3;

/// Tag vocabulary for a learning. Kept deliberately small.
const VALID_TAGS: [&str; 4] = ["convention", "gotcha", "fragile", "decision-rationale"];

#[derive(Debug, Deserialize)]
struct CuratorOutput {
    learnings: Vec<Learning>,
}

#[derive(Debug, Deserialize)]
struct Learning {
    /// The durable lesson, one or two sentences.
    content: String,
    /// One of [`VALID_TAGS`].
    tag: String,
    /// Model's confidence the lesson is durable and correct, 0.0–1.0.
    confidence: f64,
}

fn output_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": {
            "learnings": {
                "type": "array",
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "content": { "type": "string" },
                        "tag": { "type": "string", "enum": VALID_TAGS },
                        "confidence": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
                    },
                    "required": ["content", "tag", "confidence"]
                }
            }
        },
        "required": ["learnings"]
    })
}

const SYSTEM_PROMPT: &str = "You distill DURABLE lessons from an agent's completed run. \
Output 0 to 3 lessons that will help on FUTURE, DIFFERENT tasks for the same agent. \
A lesson must be a stable convention, a non-obvious gotcha, a fragile area to be careful with, \
or the rationale behind a decision. \
Do NOT emit task-specific facts (e.g. 'fixed a typo on line 42'), generic platitudes \
(e.g. 'write clean code'), or anything already obvious. \
If nothing durable was learned, return an empty list. Prefer fewer, higher-quality lessons.";

/// Distill durable learnings from a completed run and persist them.
///
/// `transcript` is the run's message history. `existing` are the agent's current
/// learnings, passed so the model can avoid restating them. Best-effort: errors
/// are logged, not propagated — distillation must never fail a successful run.
pub async fn distill_learnings(
    memory: &MemorySubstrate,
    agent_id: AgentId,
    model: &str,
    driver: Arc<dyn LlmDriver>,
    transcript: &[Message],
) {
    let existing = recall_existing(memory, agent_id).await;
    let user_prompt = build_prompt(transcript, &existing);

    let opts = StructuredOptions {
        temperature: 0.2,
        max_tokens: 1024,
        max_retries: 2,
    };

    let output: CuratorOutput = match complete_structured(
        driver,
        model,
        Some(SYSTEM_PROMPT.to_string()),
        &user_prompt,
        output_schema(),
        &opts,
    )
    .await
    {
        Ok(o) => o,
        Err(e) => {
            warn!(agent = %agent_id, error = %e, "curator: distillation failed");
            return;
        }
    };

    let mut stored = 0usize;
    for learning in output.learnings {
        let content = learning.content.trim();
        if content.is_empty() {
            continue;
        }
        let tag = if VALID_TAGS.contains(&learning.tag.as_str()) {
            learning.tag.as_str()
        } else {
            "gotcha"
        };
        if is_duplicate(content, &existing) {
            debug!(agent = %agent_id, "curator: skipping duplicate learning");
            continue;
        }

        let mut metadata = HashMap::new();
        metadata.insert(
            "tag".to_string(),
            serde_json::Value::String(tag.to_string()),
        );
        metadata.insert(
            "importance".to_string(),
            serde_json::Value::from((learning.confidence.clamp(0.0, 1.0) * 10.0).round() as u64),
        );

        if memory
            .remember(
                agent_id,
                content,
                MemorySource::Inference,
                LEARNING_SCOPE,
                metadata,
            )
            .await
            .is_ok()
        {
            stored += 1;
        }
    }
    if stored > 0 {
        debug!(agent = %agent_id, count = stored, "curator: stored learnings");
    }
}

/// Recall durable learnings to inject into an agent's prompt for the current turn.
///
/// Returns up to [`READBACK_LIMIT`] learnings for `agent_id`, filtered to those
/// stored at or above [`READBACK_MIN_CONFIDENCE`] and ranked by the substrate's
/// recall order (most recently accessed / most used first). Best-effort: a recall
/// error yields an empty list — read-back must never fail a run.
///
/// The query is intentionally empty: learnings are standing conventions, not
/// facts matched to the current message, so we want the top confident ones
/// regardless of lexical overlap with the user's turn. (A non-empty query would
/// trigger a `LIKE` filter on the text-search substrate and hide nearly all of
/// them.)
///
/// Self-gating: if the curator write path is disabled, no learnings exist, so this
/// returns empty without needing a separate feature flag.
pub async fn recall_learnings(memory: &MemorySubstrate, agent_id: AgentId) -> Vec<String> {
    let filter = MemoryFilter {
        agent_id: Some(agent_id),
        scope: Some(LEARNING_SCOPE.to_string()),
        min_confidence: Some(READBACK_MIN_CONFIDENCE),
        ..Default::default()
    };
    match memory.recall("", READBACK_LIMIT, Some(filter)).await {
        Ok(frags) => frags.into_iter().map(|f| f.content).collect(),
        Err(e) => {
            warn!(agent = %agent_id, error = %e, "curator: read-back recall failed");
            Vec::new()
        }
    }
}

/// Render recalled learnings as a system-prompt section, skipping any whose
/// content already appears among `already_shown` (the general recalled memories),
/// so a learning surfaced by similarity search is not injected twice.
///
/// Returns `None` when nothing new remains to show.
pub fn build_learnings_section(learnings: &[String], already_shown: &[String]) -> Option<String> {
    let shown_norm: Vec<String> = already_shown.iter().map(|s| normalize(s)).collect();
    let mut out = String::from("## Learned Conventions\n");
    out.push_str(
        "Durable lessons distilled from this agent's past runs. Treat them as established \
         conventions for this codebase unless the current task contradicts them.\n",
    );
    let mut any = false;
    for learning in learnings {
        let content = learning.trim();
        if content.is_empty() {
            continue;
        }
        let norm = normalize(content);
        if shown_norm.contains(&norm) {
            continue;
        }
        out.push_str("- ");
        out.push_str(content);
        out.push('\n');
        any = true;
    }
    any.then_some(out)
}

async fn recall_existing(memory: &MemorySubstrate, agent_id: AgentId) -> Vec<String> {
    let filter = MemoryFilter {
        agent_id: Some(agent_id),
        scope: Some(LEARNING_SCOPE.to_string()),
        ..Default::default()
    };
    match memory.recall("", 20, Some(filter)).await {
        Ok(frags) => frags.into_iter().map(|f| f.content).collect(),
        Err(e) => {
            warn!(agent = %agent_id, error = %e, "curator: recall of existing learnings failed");
            Vec::new()
        }
    }
}

fn is_duplicate(content: &str, existing: &[String]) -> bool {
    let norm = normalize(content);
    existing.iter().any(|e| {
        let en = normalize(e);
        en == norm || en.contains(&norm) || norm.contains(&en)
    })
}

fn normalize(s: &str) -> String {
    s.to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Render the transcript (truncated) plus existing learnings into a prompt.
fn build_prompt(transcript: &[Message], existing: &[String]) -> String {
    const MAX_CHARS: usize = 8000;
    let mut body = String::new();
    for msg in transcript {
        let role = match msg.role {
            Role::User => "User",
            Role::Assistant => "Assistant",
            Role::System => continue,
        };
        let text = message_text(msg);
        if text.trim().is_empty() {
            continue;
        }
        body.push_str(role);
        body.push_str(": ");
        body.push_str(&text);
        body.push('\n');
    }
    let body = crate::str_utils::safe_truncate_str(&body, MAX_CHARS);

    let existing_block = if existing.is_empty() {
        "(none)".to_string()
    } else {
        existing
            .iter()
            .map(|e| format!("- {e}"))
            .collect::<Vec<_>>()
            .join("\n")
    };

    format!(
        "Existing learnings for this agent (do not restate these):\n{existing_block}\n\n\
         Run transcript:\n{body}\n\n\
         Extract 0-3 NEW durable learnings."
    )
}

fn message_text(msg: &Message) -> String {
    match &msg.content {
        MessageContent::Text(t) => t.clone(),
        MessageContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text, .. } => Some(text.clone()),
                ContentBlock::ToolUse { name, .. } => Some(format!("[called tool: {name}]")),
                ContentBlock::ToolResult {
                    tool_name,
                    is_error,
                    ..
                } => Some(format!(
                    "[tool {tool_name} {}]",
                    if *is_error { "errored" } else { "ok" }
                )),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" "),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_matches_normalized_and_substring() {
        let existing = vec!["Always run cargo clippy before commit".to_string()];
        assert!(is_duplicate(
            "always run CARGO clippy before commit",
            &existing
        ));
        assert!(is_duplicate("cargo clippy before commit", &existing));
        assert!(!is_duplicate("the auth module uses JWT", &existing));
    }

    #[test]
    fn schema_lists_valid_tags() {
        let schema = output_schema();
        let tags = &schema["properties"]["learnings"]["items"]["properties"]["tag"]["enum"];
        assert_eq!(tags.as_array().unwrap().len(), VALID_TAGS.len());
    }

    #[tokio::test]
    async fn distill_persists_learnings_and_dedups() {
        use crate::llm_driver::{CompletionRequest, CompletionResponse, LlmDriver, LlmError};
        use async_trait::async_trait;
        use openfang_types::tool::ToolCall;

        struct FakeDriver;

        #[async_trait]
        impl LlmDriver for FakeDriver {
            async fn complete(
                &self,
                _req: CompletionRequest,
            ) -> Result<CompletionResponse, LlmError> {
                Ok(CompletionResponse {
                    content: vec![],
                    stop_reason: openfang_types::message::StopReason::ToolUse,
                    tool_calls: vec![ToolCall {
                        id: "1".to_string(),
                        name: "emit_structured_output".to_string(),
                        input: serde_json::json!({
                            "learnings": [
                                {"content": "The auth module uses JWT, not sessions",
                                 "tag": "convention", "confidence": 0.9},
                                {"content": "migrations must run before tests",
                                 "tag": "gotcha", "confidence": 0.8}
                            ]
                        }),
                    }],
                    usage: openfang_types::message::TokenUsage::default(),
                })
            }
        }

        let substrate = MemorySubstrate::open_in_memory(0.1).unwrap();
        let agent_id = AgentId::new();
        let transcript = vec![
            Message {
                role: Role::User,
                content: MessageContent::Text("how does auth work?".into()),
            },
            Message {
                role: Role::Assistant,
                content: MessageContent::Text("it uses JWT".into()),
            },
        ];

        distill_learnings(
            &substrate,
            agent_id,
            "test-model",
            Arc::new(FakeDriver),
            &transcript,
        )
        .await;

        let stored = recall_existing(&substrate, agent_id).await;
        assert_eq!(stored.len(), 2, "both learnings should persist");
        assert!(stored.iter().any(|s| s.contains("JWT")));

        // A second run returning the same learnings must not duplicate.
        distill_learnings(
            &substrate,
            agent_id,
            "test-model",
            Arc::new(FakeDriver),
            &transcript,
        )
        .await;
        let after = recall_existing(&substrate, agent_id).await;
        assert_eq!(after.len(), 2, "duplicates must be skipped on re-run");
    }

    #[test]
    fn learnings_section_dedups_against_already_shown() {
        let learnings = vec![
            "The auth module uses JWT, not sessions".to_string(),
            "Migrations must run before tests".to_string(),
        ];
        // First learning was already surfaced by general recall.
        let shown = vec!["the AUTH module uses JWT, not sessions".to_string()];
        let section = build_learnings_section(&learnings, &shown).unwrap();
        assert!(!section.contains("auth module uses JWT"));
        assert!(section.contains("Migrations must run before tests"));
    }

    #[test]
    fn learnings_section_none_when_all_shown_or_empty() {
        assert!(build_learnings_section(&[], &[]).is_none());
        let learnings = vec!["lesson one".to_string()];
        let shown = vec!["LESSON ONE".to_string()];
        assert!(build_learnings_section(&learnings, &shown).is_none());
    }

    #[tokio::test]
    async fn recall_learnings_filters_by_confidence_and_scope() {
        let substrate = MemorySubstrate::open_in_memory(0.1).unwrap();
        let agent_id = AgentId::new();

        // High-confidence learning (importance 9 -> stored confidence 0.91): admitted.
        let mut hi = HashMap::new();
        hi.insert("importance".to_string(), serde_json::Value::from(9u64));
        substrate
            .remember(
                agent_id,
                "Always run clippy before commit",
                MemorySource::Inference,
                LEARNING_SCOPE,
                hi,
            )
            .await
            .unwrap();

        // Low-confidence learning (importance 3 -> stored confidence 0.37): filtered out.
        let mut lo = HashMap::new();
        lo.insert("importance".to_string(), serde_json::Value::from(3u64));
        substrate
            .remember(
                agent_id,
                "Maybe the cache helps sometimes",
                MemorySource::Inference,
                LEARNING_SCOPE,
                lo,
            )
            .await
            .unwrap();

        // A non-learning memory in another scope must never be returned here.
        substrate
            .remember(
                agent_id,
                "User prefers dark mode",
                MemorySource::UserProvided,
                "episodic",
                HashMap::new(),
            )
            .await
            .unwrap();

        let recalled = recall_learnings(&substrate, agent_id).await;
        assert_eq!(
            recalled.len(),
            1,
            "only the high-confidence learning admits"
        );
        assert!(recalled[0].contains("clippy"));
    }

    #[test]
    fn build_prompt_skips_system_and_empty() {
        let msgs = vec![
            Message {
                role: Role::System,
                content: MessageContent::Text("sys".into()),
            },
            Message {
                role: Role::User,
                content: MessageContent::Text("how do I build?".into()),
            },
            Message {
                role: Role::Assistant,
                content: MessageContent::Text("run make".into()),
            },
        ];
        let p = build_prompt(&msgs, &[]);
        assert!(!p.contains("sys"));
        assert!(p.contains("User: how do I build?"));
        assert!(p.contains("Assistant: run make"));
        assert!(p.contains("(none)"));
    }
}
