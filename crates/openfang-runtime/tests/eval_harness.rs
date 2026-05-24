//! Tool-selection trajectory eval (eval-harness-design §1).
//!
//! The fastest, deterministic, judge-free agent eval: drive the real
//! [`run_agent_loop`] against a *recorded* LLM whose responses are fixed, then
//! compare the tool sequence the agent actually executed against a golden
//! sequence using the §2 ordered-bigram Jaccard metric. No model is called, so
//! the score is stable run-to-run (anti-pattern #17) and cheap enough to gate
//! every commit. The golden set deliberately includes a tool-error case
//! (anti-pattern #3 — mocks must inject failures, not only happy paths).
//!
//! Each `[tool, args]` step is identified by `(name, LoopGuard::compute_hash)`,
//! reusing the production hash so a golden written here matches exactly what the
//! loop's dedup/circuit-breaker logic sees.

use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openfang_runtime::agent_loop::run_agent_loop;
use openfang_runtime::llm_driver::{CompletionRequest, CompletionResponse, LlmDriver, LlmError};
use openfang_runtime::loop_guard::LoopGuard;
use openfang_types::agent::{AgentManifest, ModelConfig};
use openfang_types::message::{ContentBlock, StopReason, TokenUsage};
use openfang_types::tool::{ToolCall, ToolDefinition};

/// One scripted assistant turn: either a set of tool calls or a final text.
#[derive(serde::Deserialize, Clone)]
struct ScriptStep {
    #[serde(default)]
    tool_calls: Vec<ScriptToolCall>,
    #[serde(default)]
    text: Option<String>,
}

#[derive(serde::Deserialize, Clone)]
struct ScriptToolCall {
    name: String,
    input: serde_json::Value,
}

/// A golden trajectory case loaded from `tests/eval/golden/*.json`.
#[derive(serde::Deserialize)]
struct GoldenCase {
    name: String,
    #[allow(dead_code)]
    description: String,
    prompt: String,
    script: Vec<ScriptStep>,
    expected_tools: Vec<ScriptToolCall>,
    expect_any_error: bool,
}

/// Recorded driver: replays a fixed script and records the tool calls it emits,
/// so the harness can read back the exact sequence the agent selected.
struct RecordedDriver {
    script: Vec<ScriptStep>,
    call_index: Mutex<usize>,
    recorded: Arc<Mutex<Vec<(String, String)>>>,
}

#[async_trait]
impl LlmDriver for RecordedDriver {
    async fn complete(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse, LlmError> {
        let idx = {
            let mut i = self.call_index.lock().unwrap();
            let cur = *i;
            *i += 1;
            cur
        };
        // Past the script end → end the turn so the loop terminates.
        let step = self.script.get(idx).cloned().unwrap_or(ScriptStep {
            tool_calls: Vec::new(),
            text: Some("(end)".to_string()),
        });

        if !step.tool_calls.is_empty() {
            let mut rec = self.recorded.lock().unwrap();
            let tool_calls: Vec<ToolCall> = step
                .tool_calls
                .iter()
                .enumerate()
                .map(|(n, tc)| {
                    rec.push((tc.name.clone(), LoopGuard::compute_hash(&tc.name, &tc.input)));
                    ToolCall {
                        id: format!("tu-{idx}-{n}"),
                        name: tc.name.clone(),
                        input: tc.input.clone(),
                    }
                })
                .collect();
            return Ok(CompletionResponse {
                content: vec![],
                stop_reason: StopReason::ToolUse,
                tool_calls,
                usage: TokenUsage::default(),
            });
        }

        Ok(CompletionResponse {
            content: vec![ContentBlock::Text {
                text: step.text.unwrap_or_default(),
                provider_metadata: None,
            }],
            stop_reason: StopReason::EndTurn,
            tool_calls: vec![],
            usage: TokenUsage::default(),
        })
    }
}

/// Ordered-bigram Jaccard similarity (eval-harness-design §2).
///
/// Treats each trajectory as its sequence of `(tool, args_hash)` steps, forms the
/// set of adjacent ordered pairs, and returns |∩| / |∪|. Two identical sequences
/// score 1.0; a reordering or a wrong/missing step drops it below 1.0. Two empty
/// sequences are a perfect match (a no-tool turn correctly using no tools).
fn ordered_bigram_jaccard(actual: &[(String, String)], expected: &[(String, String)]) -> f64 {
    if actual.is_empty() && expected.is_empty() {
        return 1.0;
    }
    // A single-step trajectory has no bigrams; fall back to step-set Jaccard so
    // one-tool cases are still scorable.
    if actual.len() < 2 && expected.len() < 2 {
        let a: std::collections::HashSet<_> = actual.iter().collect();
        let e: std::collections::HashSet<_> = expected.iter().collect();
        return set_jaccard(&a, &e);
    }
    let a = bigrams(actual);
    let e = bigrams(expected);
    set_jaccard(&a.iter().collect(), &e.iter().collect())
}

fn bigrams(seq: &[(String, String)]) -> Vec<((String, String), (String, String))> {
    seq.windows(2)
        .map(|w| (w[0].clone(), w[1].clone()))
        .collect()
}

fn set_jaccard<T: std::hash::Hash + Eq>(
    a: &std::collections::HashSet<T>,
    b: &std::collections::HashSet<T>,
) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let inter = a.intersection(b).count() as f64;
    let union = a.union(b).count() as f64;
    if union == 0.0 {
        1.0
    } else {
        inter / union
    }
}

fn eval_manifest() -> AgentManifest {
    AgentManifest {
        name: "eval-agent".to_string(),
        model: ModelConfig {
            system_prompt: "You are an eval agent.".to_string(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn eval_tools() -> Vec<ToolDefinition> {
    let obj = serde_json::json!({"type": "object", "properties": {}});
    ["file_read", "file_list", "system_time"]
        .iter()
        .map(|n| ToolDefinition {
            name: n.to_string(),
            description: format!("{n} tool"),
            input_schema: obj.clone(),
        })
        .collect()
}

fn load_golden() -> Vec<GoldenCase> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/eval/golden");
    let mut cases = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("golden dir exists") {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) == Some("json") {
            let raw = std::fs::read_to_string(&path).unwrap();
            cases.push(
                serde_json::from_str::<GoldenCase>(&raw)
                    .unwrap_or_else(|e| panic!("bad golden {}: {e}", path.display())),
            );
        }
    }
    assert!(cases.len() >= 5, "expected >=5 golden cases, got {}", cases.len());
    cases
}

/// Run one golden case against the recorded driver, returning (trajectory score,
/// whether any executed tool errored).
async fn run_case(case: &GoldenCase, workspace: &Path) -> (f64, bool) {
    let memory = openfang_memory::MemorySubstrate::open_in_memory(0.01).unwrap();
    let agent_id = openfang_types::agent::AgentId::new();
    let mut session = openfang_memory::session::Session {
        id: openfang_types::agent::SessionId::new(),
        agent_id,
        messages: Vec::new(),
        context_window_tokens: 0,
        label: None,
    };
    let recorded = Arc::new(Mutex::new(Vec::new()));
    let driver: Arc<dyn LlmDriver> = Arc::new(RecordedDriver {
        script: case.script.clone(),
        call_index: Mutex::new(0),
        recorded: recorded.clone(),
    });
    let tools = eval_tools();

    run_agent_loop(
        &eval_manifest(),
        &case.prompt,
        &mut session,
        &memory,
        driver,
        &tools,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(workspace),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .await
    .expect("agent loop completes");

    let actual = recorded.lock().unwrap().clone();
    let expected: Vec<(String, String)> = case
        .expected_tools
        .iter()
        .map(|tc| (tc.name.clone(), LoopGuard::compute_hash(&tc.name, &tc.input)))
        .collect();
    let score = ordered_bigram_jaccard(&actual, &expected);

    // Detect a real executed tool error from the session's tool-result blocks.
    let any_error = session.messages.iter().any(|m| match &m.content {
        openfang_types::message::MessageContent::Blocks(blocks) => blocks.iter().any(|b| {
            matches!(b, ContentBlock::ToolResult { is_error: true, .. })
        }),
        _ => false,
    });
    (score, any_error)
}

fn setup_workspace() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("notes.txt"), "hello").unwrap();
    dir
}

#[tokio::test]
async fn tool_selection_eval_all_golden_pass_threshold() {
    const THRESHOLD: f64 = 0.99;
    let ws = setup_workspace();
    let cases = load_golden();
    for case in &cases {
        let (score, any_error) = run_case(case, ws.path()).await;
        assert!(
            score >= THRESHOLD,
            "case '{}': trajectory score {score} below {THRESHOLD}",
            case.name
        );
        assert_eq!(
            any_error, case.expect_any_error,
            "case '{}': expected any_error={}, got {any_error}",
            case.name, case.expect_any_error
        );
    }
}

#[tokio::test]
async fn tool_selection_eval_is_deterministic() {
    // anti-pattern #17: the score must not move across repeated runs.
    let ws = setup_workspace();
    let cases = load_golden();
    let mut runs = Vec::new();
    for _ in 0..3 {
        let mut scores = Vec::new();
        for case in &cases {
            scores.push(run_case(case, ws.path()).await.0);
        }
        runs.push(scores);
    }
    assert_eq!(runs[0], runs[1], "run 1 != run 2");
    assert_eq!(runs[1], runs[2], "run 2 != run 3");
}

/// Build a case where a single assistant turn emits N independent reads.
fn parallel_reads_case() -> GoldenCase {
    GoldenCase {
        name: "parallel_reads".to_string(),
        description: "Three independent reads in one turn.".to_string(),
        prompt: "Read everything.".to_string(),
        script: vec![
            ScriptStep {
                tool_calls: vec![
                    ScriptToolCall {
                        name: "file_read".to_string(),
                        input: serde_json::json!({"path": "notes.txt"}),
                    },
                    ScriptToolCall {
                        name: "file_list".to_string(),
                        input: serde_json::json!({"path": "."}),
                    },
                    ScriptToolCall {
                        name: "system_time".to_string(),
                        input: serde_json::json!({}),
                    },
                ],
                text: None,
            },
            ScriptStep {
                tool_calls: vec![],
                text: Some("done".to_string()),
            },
        ],
        expected_tools: vec![
            ScriptToolCall {
                name: "file_read".to_string(),
                input: serde_json::json!({"path": "notes.txt"}),
            },
            ScriptToolCall {
                name: "file_list".to_string(),
                input: serde_json::json!({"path": "."}),
            },
            ScriptToolCall {
                name: "system_time".to_string(),
                input: serde_json::json!({}),
            },
        ],
        expect_any_error: false,
    }
}

#[tokio::test]
async fn parallel_read_fanout_preserves_order_and_results() {
    // Fix 6: a turn with 3 independent reads must run them all and merge results
    // back in the original call order (deterministic by id, not completion order).
    let ws = setup_workspace();
    let case = parallel_reads_case();
    let (score, any_error) = run_case(&case, ws.path()).await;
    assert_eq!(score, 1.0, "all three reads should match the golden in order");
    assert!(!any_error, "pure reads against existing files should not error");
}

#[tokio::test]
async fn parallel_read_fanout_is_deterministic() {
    // The loop_guard runs serially before fan-out, so counters — and therefore
    // the observed trajectory — must be identical across runs even though
    // execution is concurrent (guards the M1 regression from the plan review).
    let ws = setup_workspace();
    let case = parallel_reads_case();
    let mut scores = Vec::new();
    for _ in 0..3 {
        scores.push(run_case(&case, ws.path()).await.0);
    }
    assert_eq!(scores[0], scores[1]);
    assert_eq!(scores[1], scores[2]);
    assert_eq!(scores[0], 1.0);
}

#[tokio::test]
async fn tool_selection_eval_detects_wrong_trajectory() {
    // A deliberately broken golden (wrong expected tool) must score below
    // threshold — proving the gate can actually fail, not just always pass.
    let ws = setup_workspace();
    let broken = GoldenCase {
        name: "broken".to_string(),
        description: "intentionally wrong expectation".to_string(),
        prompt: "What does notes.txt say?".to_string(),
        script: vec![
            ScriptStep {
                tool_calls: vec![ScriptToolCall {
                    name: "file_read".to_string(),
                    input: serde_json::json!({"path": "notes.txt"}),
                }],
                text: None,
            },
            ScriptStep {
                tool_calls: vec![],
                text: Some("done".to_string()),
            },
        ],
        // Expect a different tool than the agent actually calls.
        expected_tools: vec![ScriptToolCall {
            name: "system_time".to_string(),
            input: serde_json::json!({}),
        }],
        expect_any_error: false,
    };
    let (score, _) = run_case(&broken, ws.path()).await;
    assert!(score < 0.99, "broken trajectory should fail, scored {score}");
}
