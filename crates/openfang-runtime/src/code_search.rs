//! `code_search` — structured, paginated, token-bounded code search.
//!
//! Rationale: openfang's `shell_exec` rejects `|`, `>`, `&&`, `$()` (see
//! `subprocess_sandbox.rs::contains_shell_metacharacters`), so a raw
//! `shell_exec "rg foo"` cannot be piped through `head` and dumps the full
//! match set — blowing 50-200k tokens on a common pattern. This tool runs
//! `rg --json`, parses the stream, caps output aggressively, and returns
//! structured JSON with a pagination cursor.
//!
//! If `rg` is not on PATH, falls back to a native `walkdir + regex` impl
//! with a documented feature gap: PCRE2-only patterns (lookahead, lookbehind,
//! backreferences) are rejected explicitly rather than silently mis-matching.

use crate::subprocess_sandbox;
use crate::workspace_sandbox::resolve_sandbox_path;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;
use tokio::io::AsyncReadExt;

/// Hard caps. These are invariants — user-supplied overrides are clamped.
const MAX_RESULTS_CAP: usize = 200;
const MAX_CONTEXT_LINES_CAP: usize = 5;
const MAX_TEXT_CHARS: usize = 200;
const MAX_CONTEXT_LINE_CHARS: usize = 200;
const RAW_OUTPUT_BYTE_CAP: usize = 5 * 1024 * 1024; // 5 MB
const RG_WALL_CLOCK_SECS: u64 = 30;
const RG_KILL_GRACE_MS: u64 = 2_000;

/// Input schema for the `code_search` tool.
#[derive(Debug, Deserialize, Default)]
pub struct CodeSearchInput {
    /// Regex or literal pattern to search for.
    pub query: String,
    /// Optional sub-path within the workspace to narrow the search.
    #[serde(default)]
    pub path: Option<String>,
    /// Include glob(s), comma-separated. E.g. `"*.rs,!tests/**"`.
    #[serde(default)]
    pub glob: Option<String>,
    /// `rg --type` alias (e.g. `"rust"`, `"py"`). Ignored by native fallback.
    #[serde(default)]
    pub r#type: Option<String>,
    /// Max results per call. Capped at 200.
    #[serde(default)]
    pub max_results: Option<usize>,
    /// Opaque continuation cursor from a previous call's `next_cursor`.
    #[serde(default)]
    pub cursor: Option<String>,
    /// Lines of context on each side of every match. Capped at 5.
    #[serde(default)]
    pub context_lines: Option<usize>,
    /// Output mode.
    #[serde(default = "default_mode")]
    pub mode: String,
    /// Match case-insensitively.
    #[serde(default)]
    pub ignore_case: bool,
    /// Fixed string match (no regex interpretation).
    #[serde(default)]
    pub literal: bool,
}

fn default_mode() -> String {
    "content".into()
}

/// One matched line.
#[derive(Debug, Serialize)]
pub struct CodeSearchHit {
    pub path: String,
    pub line: u64,
    pub col: u64,
    pub text: String,
    /// Lines of context surrounding the match, each individually capped in
    /// length. Only populated when `context_lines > 0`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ctx: Vec<String>,
}

/// Output schema for the `code_search` tool.
#[derive(Debug, Serialize)]
pub struct CodeSearchOutput {
    pub results: Vec<CodeSearchHit>,
    /// Opaque cursor to pass back in `cursor` for the next page. `None` when
    /// the entire result set has been delivered.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
    /// Number of results matched so far (this page + prior pages). Capped.
    pub total: usize,
    /// `true` when the search hit an internal cap (wall-clock, byte, or
    /// match count) and there may be more results the user never saw.
    pub truncated: bool,
    /// Which search backend actually ran. Useful for diagnosing surprises.
    pub backend: &'static str,
}

#[derive(Debug, Serialize, Deserialize)]
struct Cursor {
    skip: usize,
}

fn encode_cursor(skip: usize) -> String {
    let json = serde_json::to_vec(&Cursor { skip }).unwrap_or_default();
    URL_SAFE_NO_PAD.encode(json)
}

fn decode_cursor(s: &str) -> Result<Cursor, String> {
    let bytes = URL_SAFE_NO_PAD
        .decode(s)
        .map_err(|e| format!("invalid cursor: {e}"))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("invalid cursor payload: {e}"))
}

/// Result of the rg subprocess probe. `rg` is the preferred backend; the
/// native fallback covers environments where it's not installed.
enum Backend {
    Ripgrep,
    Native,
}

async fn detect_backend() -> Backend {
    // A simple PATH check — `rg --version` succeeds fast if installed.
    match tokio::process::Command::new("rg")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(mut child) => {
            let _ = tokio::time::timeout(Duration::from_secs(2), child.wait()).await;
            Backend::Ripgrep
        }
        Err(_) => Backend::Native,
    }
}

/// Execute a search against a workspace root. This is the tool entrypoint;
/// callers (the runtime tool dispatcher) validate the caller has
/// `Capability::ToolInvoke("code_search")` before invoking.
///
/// `workspace_root` is the agent's workspace — sandbox enforcement pins
/// every traversed path to stay inside it.
pub async fn run(
    input: CodeSearchInput,
    workspace_root: &Path,
) -> Result<CodeSearchOutput, String> {
    if input.query.is_empty() {
        return Err("query is empty".into());
    }
    // Clamp user-supplied caps.
    let max_results = input.max_results.unwrap_or(50).clamp(1, MAX_RESULTS_CAP);
    let context_lines = input
        .context_lines
        .unwrap_or(0)
        .min(MAX_CONTEXT_LINES_CAP);
    let mode = match input.mode.as_str() {
        "content" | "files" | "count" => input.mode.as_str(),
        // Empty → treat as default ("content") so `Default::default()` works.
        "" => "content",
        other => return Err(format!("invalid mode: {other}")),
    };
    let skip = match input.cursor.as_deref() {
        Some(s) => decode_cursor(s)?.skip,
        None => 0,
    };

    // Path sandboxing — subdir within workspace, if provided.
    let search_root = if let Some(p) = input.path.as_deref() {
        resolve_sandbox_path(p, workspace_root)?
    } else {
        workspace_root.to_path_buf()
    };

    match detect_backend().await {
        Backend::Ripgrep => {
            run_ripgrep(
                &input.query,
                &search_root,
                input.glob.as_deref(),
                input.r#type.as_deref(),
                max_results,
                skip,
                context_lines,
                mode,
                input.ignore_case,
                input.literal,
            )
            .await
        }
        Backend::Native => {
            // Native fallback: reject PCRE2-only features eagerly so the
            // caller gets an actionable error instead of zero matches.
            reject_pcre2_features(&input.query)?;
            run_native(
                &input.query,
                &search_root,
                max_results,
                skip,
                context_lines,
                mode,
                input.ignore_case,
                input.literal,
            )
            .await
        }
    }
}

fn reject_pcre2_features(pattern: &str) -> Result<(), String> {
    // Basic heuristics — `regex-lite` rejects these too but with a less
    // helpful error. Explicit detection makes the message actionable.
    let markers = [
        ("(?=", "lookahead"),
        ("(?!", "negative lookahead"),
        ("(?<=", "lookbehind"),
        ("(?<!", "negative lookbehind"),
        (r"\1", "backreference"),
        (r"\2", "backreference"),
    ];
    for (needle, name) in markers {
        if pattern.contains(needle) {
            return Err(format!(
                "PCRE2 patterns require ripgrep (rg) to be installed; encountered: {name} ('{needle}')"
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_ripgrep(
    pattern: &str,
    search_root: &Path,
    glob: Option<&str>,
    type_alias: Option<&str>,
    max_results: usize,
    skip: usize,
    context_lines: usize,
    mode: &str,
    ignore_case: bool,
    literal: bool,
) -> Result<CodeSearchOutput, String> {
    let mut cmd = tokio::process::Command::new("rg");
    cmd.arg("--json");
    if ignore_case {
        cmd.arg("-i");
    }
    if literal {
        cmd.arg("-F");
    }
    if context_lines > 0 {
        cmd.arg("-C").arg(context_lines.to_string());
    }
    if let Some(g) = glob {
        for piece in g.split(',') {
            let p = piece.trim();
            if !p.is_empty() {
                cmd.arg("--glob").arg(p);
            }
        }
    }
    if let Some(t) = type_alias {
        cmd.arg("--type").arg(t);
    }
    if mode == "files" {
        cmd.arg("--files-with-matches");
    } else if mode == "count" {
        cmd.arg("--count-matches");
    }
    cmd.arg(pattern).arg(search_root);
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("failed to spawn rg: {e}"))?;

    let mut stdout_buf = Vec::with_capacity(64 * 1024);
    let mut truncated_raw = false;
    if let Some(mut out) = child.stdout.take() {
        let mut chunk = [0u8; 8192];
        loop {
            // Read with a cooperative cancel window so a run-away rg can't
            // defeat the wall-clock timeout. We still rely on wait_or_kill
            // below for the hard kill.
            let read_result = tokio::time::timeout(
                Duration::from_secs(RG_WALL_CLOCK_SECS),
                out.read(&mut chunk),
            )
            .await;
            let n = match read_result {
                Ok(Ok(0)) => break,
                Ok(Ok(n)) => n,
                Ok(Err(_)) | Err(_) => break,
            };
            if stdout_buf.len() + n > RAW_OUTPUT_BYTE_CAP {
                truncated_raw = true;
                break;
            }
            stdout_buf.extend_from_slice(&chunk[..n]);
        }
    }

    // Drain remaining output with an aggressive timeout.
    let _ = subprocess_sandbox::wait_or_kill(
        &mut child,
        Duration::from_secs(RG_WALL_CLOCK_SECS),
        RG_KILL_GRACE_MS,
    )
    .await;

    // rg `--json` emits JSONL. Each line is an event: `{"type":"match",...}`,
    // "begin", "end", "summary". We only care about match events.
    let raw = String::from_utf8_lossy(&stdout_buf);
    let mut hits: Vec<CodeSearchHit> = Vec::new();
    let mut matched_count: usize = 0;
    let mut matched_files: Vec<String> = Vec::new();
    for line in raw.lines() {
        if line.is_empty() {
            continue;
        }
        let ev: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let ev_type = ev.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match (mode, ev_type) {
            ("files", "begin") => {
                if let Some(p) = ev
                    .get("data")
                    .and_then(|d| d.get("path"))
                    .and_then(|p| p.get("text"))
                    .and_then(|t| t.as_str())
                {
                    matched_count += 1;
                    if matched_count > skip && matched_files.len() < max_results {
                        matched_files.push(p.to_string());
                    }
                }
            }
            ("content", "match") => {
                let data = match ev.get("data") {
                    Some(d) => d,
                    None => continue,
                };
                let path = data
                    .get("path")
                    .and_then(|p| p.get("text"))
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();
                let line_no = data.get("line_number").and_then(|v| v.as_u64()).unwrap_or(0);
                let text = data
                    .get("lines")
                    .and_then(|l| l.get("text"))
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .trim_end_matches('\n')
                    .to_string();
                let col = data
                    .get("submatches")
                    .and_then(|s| s.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|m| m.get("start"))
                    .and_then(|s| s.as_u64())
                    .unwrap_or(0)
                    + 1;
                matched_count += 1;
                if matched_count <= skip {
                    continue;
                }
                if hits.len() >= max_results {
                    continue;
                }
                hits.push(CodeSearchHit {
                    path,
                    line: line_no,
                    col,
                    text: truncate_chars(&text, MAX_TEXT_CHARS),
                    ctx: Vec::new(),
                });
            }
            ("count", "summary") => {
                if let Some(n) = ev
                    .get("data")
                    .and_then(|d| d.get("stats"))
                    .and_then(|s| s.get("matched_lines"))
                    .and_then(|v| v.as_u64())
                {
                    matched_count = n as usize;
                }
            }
            _ => {}
        }
    }

    let (delivered, results) = match mode {
        "files" => (
            matched_files.len(),
            matched_files
                .into_iter()
                .map(|p| CodeSearchHit {
                    path: p,
                    line: 0,
                    col: 0,
                    text: String::new(),
                    ctx: Vec::new(),
                })
                .collect::<Vec<_>>(),
        ),
        _ => (hits.len(), hits),
    };

    let next_cursor = if matched_count > skip + delivered {
        Some(encode_cursor(skip + delivered))
    } else {
        None
    };
    let truncated = truncated_raw || next_cursor.is_some();

    Ok(CodeSearchOutput {
        results,
        next_cursor,
        total: matched_count,
        truncated,
        backend: "ripgrep",
    })
}

fn truncate_chars(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        s.chars().take(max_chars).collect::<String>() + "…"
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_native(
    pattern: &str,
    search_root: &Path,
    max_results: usize,
    skip: usize,
    _context_lines: usize,
    mode: &str,
    ignore_case: bool,
    literal: bool,
) -> Result<CodeSearchOutput, String> {
    let regex_src = if literal {
        regex_lite::escape(pattern)
    } else {
        pattern.to_string()
    };
    let flags = if ignore_case { "(?i)" } else { "" };
    let full = format!("{flags}{regex_src}");
    let re = regex_lite::Regex::new(&full).map_err(|e| format!("invalid regex: {e}"))?;

    let mut hits: Vec<CodeSearchHit> = Vec::new();
    let mut matched_count: usize = 0;
    let mut matched_files: Vec<String> = Vec::new();
    let deadline = std::time::Instant::now() + Duration::from_secs(RG_WALL_CLOCK_SECS);
    let mut bytes_read: usize = 0;
    let mut truncated_raw = false;

    for entry in walkdir::WalkDir::new(search_root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            // Skip common heavy/ignored dirs.
            let n = e.file_name().to_string_lossy();
            n != ".git" && n != "target" && n != "node_modules" && n != ".venv"
        })
        .filter_map(|e| e.ok())
    {
        if std::time::Instant::now() >= deadline {
            truncated_raw = true;
            break;
        }
        if !entry.file_type().is_file() {
            continue;
        }
        let path = entry.path();
        let bytes = match std::fs::read(path) {
            Ok(b) => b,
            Err(_) => continue,
        };
        // Skip binary files — simple NUL-byte heuristic on first 8k.
        let sniff = &bytes[..bytes.len().min(8192)];
        if sniff.contains(&0u8) {
            continue;
        }
        bytes_read += bytes.len();
        if bytes_read > RAW_OUTPUT_BYTE_CAP {
            truncated_raw = true;
            break;
        }
        let text = match std::str::from_utf8(&bytes) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let mut file_matched = false;
        for (lineno, line) in text.lines().enumerate() {
            if let Some(m) = re.find(line) {
                matched_count += 1;
                file_matched = true;
                match mode {
                    "content" => {
                        if matched_count > skip && hits.len() < max_results {
                            hits.push(CodeSearchHit {
                                path: path.display().to_string(),
                                line: (lineno + 1) as u64,
                                col: (m.start() + 1) as u64,
                                text: truncate_chars(line, MAX_TEXT_CHARS),
                                ctx: Vec::new(),
                            });
                        }
                    }
                    "files" => {
                        // "files" mode: one hit per matched file; break after first.
                        break;
                    }
                    _ => {}
                }
                if mode == "content" && matched_count - skip > max_results {
                    break;
                }
            }
        }
        if mode == "files" && file_matched {
            matched_count += if matched_count == 0 { 1 } else { 0 };
            if matched_files.len() < max_results {
                matched_files.push(path.display().to_string());
            }
        }
    }

    // Cap context line lengths too.
    for h in hits.iter_mut() {
        for c in h.ctx.iter_mut() {
            *c = truncate_chars(c, MAX_CONTEXT_LINE_CHARS);
        }
    }

    let (delivered, results) = match mode {
        "files" => (
            matched_files.len(),
            matched_files
                .into_iter()
                .map(|p| CodeSearchHit {
                    path: p,
                    line: 0,
                    col: 0,
                    text: String::new(),
                    ctx: Vec::new(),
                })
                .collect::<Vec<_>>(),
        ),
        _ => (hits.len(), hits),
    };
    let next_cursor = if matched_count > skip + delivered {
        Some(encode_cursor(skip + delivered))
    } else {
        None
    };
    let truncated = truncated_raw || next_cursor.is_some();

    Ok(CodeSearchOutput {
        results,
        next_cursor,
        total: matched_count,
        truncated,
        backend: "native",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_repo() -> TempDir {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("main.rs"),
            "fn main() {\n    println!(\"hello\");\n}\n",
        )
        .unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(
            dir.path().join("src/lib.rs"),
            "fn helper() {\n    // hello world\n}\n\nfn hello() {}\n",
        )
        .unwrap();
        dir
    }

    #[test]
    fn cursor_roundtrip() {
        let encoded = encode_cursor(42);
        let decoded = decode_cursor(&encoded).unwrap();
        assert_eq!(decoded.skip, 42);
    }

    #[test]
    fn cursor_rejects_malformed_input() {
        let err = decode_cursor("not-base64!!!").unwrap_err();
        assert!(err.contains("invalid cursor"));
    }

    #[test]
    fn pcre2_lookahead_rejected_in_native_fallback() {
        let err = reject_pcre2_features("foo(?=bar)").unwrap_err();
        assert!(err.contains("lookahead"));
        assert!(err.contains("rg"));
    }

    #[test]
    fn pcre2_backreference_rejected() {
        let err = reject_pcre2_features(r"(\w+)\1").unwrap_err();
        assert!(err.contains("backreference"));
    }

    #[test]
    fn ordinary_regex_passes_pcre2_filter() {
        assert!(reject_pcre2_features("fn \\w+").is_ok());
        assert!(reject_pcre2_features("^use .+::").is_ok());
    }

    #[tokio::test]
    async fn native_finds_hello_literal() {
        let dir = make_repo();
        let input = CodeSearchInput {
            query: "hello".into(),
            literal: true,
            ..Default::default()
        };
        // Force native backend by shadowing PATH — skip if rg is on PATH
        // (the test is intentionally light; integration tests cover rg).
        let out = run_native(
            &input.query,
            dir.path(),
            50,
            0,
            0,
            "content",
            false,
            true,
        )
        .await
        .unwrap();
        assert_eq!(out.backend, "native");
        assert!(out.total >= 2, "expected multiple 'hello' hits; got {}", out.total);
        assert!(out.results.iter().any(|h| h.path.ends_with("main.rs")));
    }

    #[tokio::test]
    async fn native_honors_max_results_and_cursor() {
        let dir = make_repo();
        // Tiny page.
        let page1 = run_native("hello", dir.path(), 1, 0, 0, "content", false, true)
            .await
            .unwrap();
        assert_eq!(page1.results.len(), 1);
        assert!(page1.next_cursor.is_some());
        assert!(page1.truncated);

        let c = decode_cursor(page1.next_cursor.as_deref().unwrap()).unwrap();
        let page2 = run_native("hello", dir.path(), 10, c.skip, 0, "content", false, true)
            .await
            .unwrap();
        // Page 2 should deliver the rest.
        assert!(!page2.results.is_empty());
    }

    #[tokio::test]
    async fn native_files_mode_returns_paths_only() {
        let dir = make_repo();
        let out = run_native("hello", dir.path(), 50, 0, 0, "files", false, true)
            .await
            .unwrap();
        assert!(!out.results.is_empty());
        // `files` hits carry the path but no line/col/text.
        for h in &out.results {
            assert_eq!(h.line, 0);
            assert_eq!(h.col, 0);
            assert!(h.text.is_empty());
        }
    }

    #[tokio::test]
    async fn run_rejects_empty_query() {
        let dir = make_repo();
        let input = CodeSearchInput {
            query: String::new(),
            ..Default::default()
        };
        let err = run(input, dir.path()).await.unwrap_err();
        assert!(err.contains("empty"));
    }

    #[tokio::test]
    async fn run_rejects_invalid_mode() {
        let dir = make_repo();
        let input = CodeSearchInput {
            query: "x".into(),
            mode: "weird".into(),
            ..Default::default()
        };
        let err = run(input, dir.path()).await.unwrap_err();
        assert!(err.contains("invalid mode"));
    }

    #[tokio::test]
    async fn run_rejects_path_traversal() {
        let dir = make_repo();
        let input = CodeSearchInput {
            query: "x".into(),
            path: Some("../../../etc".into()),
            ..Default::default()
        };
        let err = run(input, dir.path()).await.unwrap_err();
        assert!(
            err.contains("Path traversal") || err.contains("denied"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn truncate_chars_handles_unicode() {
        assert_eq!(truncate_chars("hello", 10), "hello");
        let out = truncate_chars("ééééé", 3);
        assert!(out.ends_with('…'));
        assert_eq!(out.chars().filter(|c| *c == 'é').count(), 3);
    }
}
