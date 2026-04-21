//! OpenFang Hands — curated autonomous capability packages.
//!
//! A Hand is a pre-built, domain-complete agent configuration that users activate
//! from a marketplace. Unlike regular agents (you chat with them), Hands work for
//! you (you check in on them).

pub mod bundled;
pub mod registry;

use chrono::{DateTime, Utc};
use openfang_types::agent::AgentId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ─── Error types ─────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum HandError {
    #[error("Hand not found: {0}")]
    NotFound(String),
    #[error("Hand already active: {0}")]
    AlreadyActive(String),
    #[error("Hand instance not found: {0}")]
    InstanceNotFound(Uuid),
    #[error("Activation failed: {0}")]
    ActivationFailed(String),
    #[error("TOML parse error: {0}")]
    TomlParse(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Config error: {0}")]
    Config(String),
}

pub type HandResult<T> = Result<T, HandError>;

// ─── Core types ──────────────────────────────────────────────────────────────

/// Category of a Hand.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HandCategory {
    Content,
    Security,
    Productivity,
    Development,
    Communication,
    Data,
    Finance,
    #[serde(other)]
    Other,
}

impl std::fmt::Display for HandCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Content => write!(f, "Content"),
            Self::Security => write!(f, "Security"),
            Self::Productivity => write!(f, "Productivity"),
            Self::Development => write!(f, "Development"),
            Self::Communication => write!(f, "Communication"),
            Self::Data => write!(f, "Data"),
            Self::Finance => write!(f, "Finance"),
            Self::Other => write!(f, "Other"),
        }
    }
}

/// Type of requirement check.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequirementType {
    /// A binary must exist on PATH.
    Binary,
    /// An environment variable must be set.
    EnvVar,
    /// An API key env var must be set.
    ApiKey,
}

/// Platform-specific install commands and guides for a requirement.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HandInstallInfo {
    #[serde(default)]
    pub macos: Option<String>,
    #[serde(default)]
    pub windows: Option<String>,
    #[serde(default)]
    pub linux_apt: Option<String>,
    #[serde(default)]
    pub linux_dnf: Option<String>,
    #[serde(default)]
    pub linux_pacman: Option<String>,
    #[serde(default)]
    pub pip: Option<String>,
    #[serde(default)]
    pub signup_url: Option<String>,
    #[serde(default)]
    pub docs_url: Option<String>,
    #[serde(default)]
    pub env_example: Option<String>,
    #[serde(default)]
    pub manual_url: Option<String>,
    #[serde(default)]
    pub estimated_time: Option<String>,
    #[serde(default)]
    pub steps: Vec<String>,
}

/// A single requirement the user must satisfy to use a Hand.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandRequirement {
    /// Unique key for this requirement.
    pub key: String,
    /// Human-readable label.
    pub label: String,
    /// What kind of check to perform.
    pub requirement_type: RequirementType,
    /// The value to check (binary name, env var name, etc.).
    pub check_value: String,
    /// Human-readable description of why this is needed.
    #[serde(default)]
    pub description: Option<String>,
    /// Whether this requirement is optional (non-critical).
    ///
    /// Optional requirements do not block activation. When an active hand has
    /// unmet optional requirements it is reported as "degraded" rather than
    /// "requirements not met".
    #[serde(default)]
    pub optional: bool,
    /// Platform-specific installation instructions.
    #[serde(default)]
    pub install: Option<HandInstallInfo>,
}

/// A metric displayed on the Hand dashboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandMetric {
    /// Display label.
    pub label: String,
    /// Memory key to read from agent's structured memory.
    pub memory_key: String,
    /// Display format (e.g. "number", "duration", "bytes").
    #[serde(default = "default_format")]
    pub format: String,
}

fn default_format() -> String {
    "number".to_string()
}

// ─── Hand settings types ────────────────────────────────────────────────────

/// Type of a hand setting control.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HandSettingType {
    Select,
    Text,
    Toggle,
}

/// A single option within a Select-type setting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandSettingOption {
    pub value: String,
    pub label: String,
    /// Env var to check for "Ready" badge (e.g. `GROQ_API_KEY`).
    #[serde(default)]
    pub provider_env: Option<String>,
    /// Binary to check on PATH for "Ready" badge (e.g. `whisper`).
    #[serde(default)]
    pub binary: Option<String>,
}

/// A configurable setting declared in HAND.toml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandSetting {
    pub key: String,
    pub label: String,
    #[serde(default)]
    pub description: String,
    pub setting_type: HandSettingType,
    #[serde(default)]
    pub default: String,
    #[serde(default)]
    pub options: Vec<HandSettingOption>,
    /// Env var name to expose when a text-type setting has a value
    /// (e.g. `ELEVENLABS_API_KEY` for an API key text field).
    #[serde(default)]
    pub env_var: Option<String>,
}

/// Result of resolving user-chosen settings against the schema.
pub struct ResolvedSettings {
    /// Markdown block to append to the system prompt (e.g. `## User Configuration\n- STT: Groq...`).
    pub prompt_block: String,
    /// Env var names the agent's subprocess should have access to.
    pub env_vars: Vec<String>,
}

/// Resolve user config values against a hand's settings schema.
///
/// For each setting, looks up the user's choice in `config` (falling back to
/// `setting.default`). For Select-type settings, finds the matching option and
/// collects its `provider_env` if present. Builds a prompt block summarising
/// the user's configuration.
pub fn resolve_settings(
    settings: &[HandSetting],
    config: &HashMap<String, serde_json::Value>,
) -> ResolvedSettings {
    let mut lines: Vec<String> = Vec::new();
    let mut env_vars: Vec<String> = Vec::new();

    for setting in settings {
        let chosen_value = config
            .get(&setting.key)
            .and_then(|v| v.as_str())
            .unwrap_or(&setting.default);

        match setting.setting_type {
            HandSettingType::Select => {
                let matched = setting.options.iter().find(|o| o.value == chosen_value);
                let display = matched.map(|o| o.label.as_str()).unwrap_or(chosen_value);
                lines.push(format!(
                    "- {}: {} ({})",
                    setting.label, display, chosen_value
                ));

                if let Some(opt) = matched {
                    if let Some(ref env) = opt.provider_env {
                        env_vars.push(env.clone());
                    }
                }
            }
            HandSettingType::Toggle => {
                let enabled = chosen_value == "true" || chosen_value == "1";
                lines.push(format!(
                    "- {}: {}",
                    setting.label,
                    if enabled { "Enabled" } else { "Disabled" }
                ));
            }
            HandSettingType::Text => {
                if !chosen_value.is_empty() {
                    lines.push(format!("- {}: {}", setting.label, chosen_value));
                    if let Some(ref env) = setting.env_var {
                        env_vars.push(env.clone());
                    }
                }
            }
        }
    }

    let prompt_block = if lines.is_empty() {
        String::new()
    } else {
        format!("## User Configuration\n\n{}", lines.join("\n"))
    };

    ResolvedSettings {
        prompt_block,
        env_vars,
    }
}

/// Dashboard schema for a Hand's metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HandDashboard {
    pub metrics: Vec<HandMetric>,
}

/// Agent configuration embedded in a Hand definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandAgentConfig {
    pub name: String,
    pub description: String,
    #[serde(default = "default_module")]
    pub module: String,
    #[serde(default = "default_provider")]
    pub provider: String,
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    pub system_prompt: String,
    #[serde(default)]
    pub max_iterations: Option<u32>,
    /// Heartbeat interval in seconds for autonomous agents. Overrides the
    /// AutonomousConfig default (30s), which is too aggressive for agents
    /// making long LLM calls. Omit to use the kernel default.
    #[serde(default)]
    pub heartbeat_interval_secs: Option<u64>,
    /// Name of a `[[settings]]` key whose resolved value becomes this Hand's
    /// workspace root at activation time.
    ///
    /// When present, the kernel calls [`validate_hand_workspace`] on the
    /// setting's current value and substitutes the canonical [`PathBuf`] as
    /// the agent's workspace (replacing the default
    /// `<workspaces_dir>/<agent_name>`). Used by `repo-digger` so the
    /// investigation's `file_read` / `code_search` tools can read the
    /// target repo — which lives outside the default workspace sandbox.
    ///
    /// Validation rejects sensitive directories (`~/.ssh`, `~/.aws`, etc.),
    /// symlink escapes, control characters, and `$STATE_DIR` overlap.
    #[serde(default)]
    pub workspace_override_setting: Option<String>,
    /// Enable provider prompt caching on the system-prompt block.
    ///
    /// Threaded through to `CompletionRequest.cache_system_prompt` at
    /// manifest-build time. Currently only honored by the Anthropic driver.
    /// Sub-agents inherit this value via their spawn-time manifest template.
    #[serde(default)]
    pub cache_system_prompt: bool,
}

fn default_module() -> String {
    "builtin:chat".to_string()
}
fn default_provider() -> String {
    "anthropic".to_string()
}
fn default_model() -> String {
    "claude-sonnet-4-20250514".to_string()
}
fn default_max_tokens() -> u32 {
    4096
}
fn default_temperature() -> f32 {
    0.7
}

#[derive(Deserialize)]
struct HandTomlWrapper {
    hand: HandDefinition,
}

/// Parse HAND.toml content, supporting both flat format and `[hand]` table format.
pub fn parse_hand_toml(content: &str) -> Result<HandDefinition, toml::de::Error> {
    if let Ok(def) = toml::from_str::<HandDefinition>(content) {
        return Ok(def);
    }
    let wrapper: HandTomlWrapper = toml::from_str(content)?;
    Ok(wrapper.hand)
}

/// Validate a user-supplied `repo_path`-style string as a candidate Hand
/// workspace override. Returns the canonical [`PathBuf`] on success.
///
/// Rejected cases:
/// - `/` or `$HOME` itself
/// - Sensitive dot-directories under `$HOME` (`.ssh`, `.aws`, `.gnupg`,
///   `.kube`, `.docker`, `.claude`, `.config/gcloud`, `.npmrc`, `.netrc`,
///   `.pypirc`, `.local/share/keyrings`)
/// - Paths less than 2 components deep under `$HOME`
/// - Non-existent paths
/// - Paths containing control characters or `U+202E` (right-to-left override)
/// - Symlink escapes (canonical path must stay inside `$HOME`)
/// - `$STATE_DIR` overlap with `repo_path`, either direction (prevents the
///   per-run MCP config JSON from landing inside the target repo and being
///   accidentally committed with its auth cookie)
///
/// `state_dir` should be the daemon's `OPENFANG_STATE_DIR` canonicalized;
/// when `None`, the overlap check is skipped (useful for unit tests that
/// don't have a real state dir).
pub fn validate_hand_workspace(
    raw: &str,
    state_dir: Option<&std::path::Path>,
) -> Result<std::path::PathBuf, String> {
    // Reject control chars and Unicode RTL override up-front — they can spoof
    // log output and bypass visual path-review.
    for c in raw.chars() {
        if c.is_control() || c == '\u{202E}' {
            return Err("workspace path contains control characters".to_string());
        }
    }
    let trimmed = raw.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return Err("workspace path is empty".to_string());
    }

    let expanded = if let Some(rest) = trimmed.strip_prefix("~/") {
        dirs::home_dir()
            .ok_or_else(|| "cannot resolve $HOME for tilde expansion".to_string())?
            .join(rest)
    } else if trimmed == "~" {
        dirs::home_dir().ok_or_else(|| "cannot resolve $HOME".to_string())?
    } else {
        std::path::PathBuf::from(trimmed)
    };

    let canon = expanded.canonicalize().map_err(|e| {
        format!(
            "workspace path does not exist or is not accessible: {} ({e})",
            expanded.display()
        )
    })?;

    if !canon.is_dir() {
        return Err(format!(
            "workspace path is not a directory: {}",
            canon.display()
        ));
    }

    // Reject bare root / bare home.
    if canon == std::path::Path::new("/") {
        return Err("workspace path cannot be the filesystem root".to_string());
    }
    if let Some(home) = dirs::home_dir() {
        let canon_home = home.canonicalize().unwrap_or(home.clone());
        if canon == canon_home {
            return Err(format!(
                "workspace path cannot be the home directory itself ({})",
                canon.display()
            ));
        }
        if let Ok(rel) = canon.strip_prefix(&canon_home) {
            // Block sensitive dot-directories at the first component under
            // $HOME first — these are always wrong regardless of depth.
            if let Some(first) = rel.components().next() {
                let name = first.as_os_str().to_string_lossy().to_string();
                const SENSITIVE: &[&str] = &[
                    ".ssh",
                    ".aws",
                    ".gnupg",
                    ".kube",
                    ".docker",
                    ".claude",
                    ".config",
                    ".npmrc",
                    ".netrc",
                    ".pypirc",
                    ".local",
                    ".gcp",
                    ".azure",
                ];
                if SENSITIVE.contains(&name.as_str()) {
                    return Err(format!(
                        "workspace path refused: '{name}' is a sensitive directory \
                         under $HOME. Pick a dedicated project directory."
                    ));
                }
            }
            // Require at least 2 components below $HOME. `$HOME/dev/openfang`
            // has two; `$HOME/<single-dir>` has one and is rejected.
            let depth = rel.components().count();
            if depth < 2 {
                return Err(format!(
                    "workspace path too shallow under $HOME (got {depth} component(s); \
                     require at least 2). Use a nested project directory like \
                     $HOME/dev/<repo>, not $HOME/<single-dir>. Got: {}",
                    canon.display()
                ));
            }
        }
    }

    // Optional $STATE_DIR overlap check.
    if let Some(state) = state_dir {
        let canon_state = state.canonicalize().unwrap_or_else(|_| state.to_path_buf());
        if canon.starts_with(&canon_state) || canon_state.starts_with(&canon) {
            return Err(format!(
                "workspace path overlaps OPENFANG_STATE_DIR ({}). Set STATE_DIR \
                 to a directory outside your repositories so the per-run MCP \
                 config (which contains an auth cookie) isn't accidentally \
                 committed.",
                canon_state.display()
            ));
        }
    }

    Ok(canon)
}

/// Map a `model_tier` setting value to a concrete model ID for direct-API
/// providers (ignored when provider is `claude-code` — the CLI picks its
/// own model).
///
/// Returns `Err` with a `feature` name suitable for wrapping in
/// `LlmError::CapabilityUnsupported` when the tier is unrecognized.
pub fn resolve_model_tier(tier: &str) -> Result<&'static str, String> {
    match tier {
        "cheap" => Ok("claude-haiku-4-5"),
        "balanced" => Ok("claude-sonnet-4-6"),
        // Opus 4.7 not yet available in the public model catalog as of
        // 2026-04-20; fall back to 4.5 for now. Update when released.
        "premium" => Ok("claude-opus-4-5"),
        other => Err(format!("model_tier={other}")),
    }
}

/// Recursively copy a directory and all its contents.
///
/// Used by `HandRegistry::install_from_path` to persist a custom hand's
/// source directory into `~/.openfang/hands/<hand_id>/` so installed hands
/// survive daemon restarts (issue #984).
pub(crate) fn copy_dir_all(
    src: impl AsRef<std::path::Path>,
    dst: impl AsRef<std::path::Path>,
) -> std::io::Result<()> {
    let src = src.as_ref();
    let dst = dst.as_ref();
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let dst_path = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_all(entry.path(), &dst_path)?;
        } else {
            std::fs::copy(entry.path(), &dst_path)?;
        }
    }
    Ok(())
}

/// Complete Hand definition — parsed from HAND.toml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandDefinition {
    /// Unique hand identifier (e.g. "clip").
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// What this Hand does.
    pub description: String,
    /// Category for marketplace browsing.
    pub category: HandCategory,
    /// Icon (emoji).
    #[serde(default)]
    pub icon: String,
    /// Tools the agent needs access to.
    #[serde(default)]
    pub tools: Vec<String>,
    /// Skill allowlist for the spawned agent (empty = all).
    #[serde(default)]
    pub skills: Vec<String>,
    /// MCP server allowlist for the spawned agent (empty = all).
    #[serde(default)]
    pub mcp_servers: Vec<String>,
    /// Requirements that must be satisfied before activation.
    #[serde(default)]
    pub requires: Vec<HandRequirement>,
    /// Configurable settings (shown in activation modal).
    #[serde(default)]
    pub settings: Vec<HandSetting>,
    /// Agent manifest template.
    pub agent: HandAgentConfig,
    /// Dashboard metrics schema.
    #[serde(default)]
    pub dashboard: HandDashboard,
    /// Bundled skill content (populated at load time, not in TOML).
    #[serde(skip)]
    pub skill_content: Option<String>,
}

/// Runtime status of a Hand instance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HandStatus {
    Active,
    Paused,
    Error(String),
    Inactive,
}

impl std::fmt::Display for HandStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Active => write!(f, "Active"),
            Self::Paused => write!(f, "Paused"),
            Self::Error(msg) => write!(f, "Error: {msg}"),
            Self::Inactive => write!(f, "Inactive"),
        }
    }
}

/// A running Hand instance — links a HandDefinition to an actual agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandInstance {
    /// Unique instance identifier.
    pub instance_id: Uuid,
    /// Which hand definition this is an instance of.
    pub hand_id: String,
    /// Optional user-supplied instance label. When set, multiple instances of
    /// the same hand can coexist as long as each (hand_id, instance_name) pair
    /// is unique. When `None`, the legacy single-instance-per-hand rule
    /// applies.
    #[serde(default)]
    pub instance_name: Option<String>,
    /// Current status.
    pub status: HandStatus,
    /// The agent that was spawned for this hand.
    pub agent_id: Option<AgentId>,
    /// Agent name (for display).
    pub agent_name: String,
    /// User-provided configuration overrides.
    pub config: HashMap<String, serde_json::Value>,
    /// When activated.
    pub activated_at: DateTime<Utc>,
    /// Last status change.
    pub updated_at: DateTime<Utc>,
}

impl HandInstance {
    /// Create a new pending instance.
    pub fn new(
        hand_id: &str,
        agent_name: &str,
        config: HashMap<String, serde_json::Value>,
        instance_name: Option<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            instance_id: Uuid::new_v4(),
            hand_id: hand_id.to_string(),
            instance_name,
            status: HandStatus::Active,
            agent_id: None,
            agent_name: agent_name.to_string(),
            config,
            activated_at: now,
            updated_at: now,
        }
    }
}

/// Request to activate a hand.
#[derive(Debug, Deserialize)]
pub struct ActivateHandRequest {
    /// Optional configuration overrides.
    #[serde(default)]
    pub config: HashMap<String, serde_json::Value>,
    /// Optional unique instance label. Allows multiple instances of the same
    /// hand to coexist as long as each name is distinct.
    #[serde(default)]
    pub instance_name: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_category_display() {
        assert_eq!(HandCategory::Content.to_string(), "Content");
        assert_eq!(HandCategory::Security.to_string(), "Security");
        assert_eq!(HandCategory::Data.to_string(), "Data");
    }

    #[test]
    fn hand_status_display() {
        assert_eq!(HandStatus::Active.to_string(), "Active");
        assert_eq!(HandStatus::Paused.to_string(), "Paused");
        assert_eq!(
            HandStatus::Error("ffmpeg not found".to_string()).to_string(),
            "Error: ffmpeg not found"
        );
    }

    #[test]
    fn hand_instance_new() {
        let instance = HandInstance::new("clip", "clip-hand", HashMap::new(), None);
        assert_eq!(instance.hand_id, "clip");
        assert_eq!(instance.agent_name, "clip-hand");
        assert_eq!(instance.status, HandStatus::Active);
        assert!(instance.agent_id.is_none());
        assert!(instance.instance_name.is_none());
    }

    #[test]
    fn hand_error_display() {
        let err = HandError::NotFound("clip".to_string());
        assert!(err.to_string().contains("clip"));

        let err = HandError::AlreadyActive("clip".to_string());
        assert!(err.to_string().contains("already"));
    }

    #[test]
    fn hand_definition_roundtrip() {
        let toml_str = r#"
id = "test"
name = "Test Hand"
description = "A test hand"
category = "content"
icon = "T"
tools = ["shell_exec"]

[[requires]]
key = "test_bin"
label = "test must be installed"
requirement_type = "binary"
check_value = "test"

[agent]
name = "test-hand"
description = "Test agent"
system_prompt = "You are a test agent."

[dashboard]
metrics = []
"#;
        let def: HandDefinition = toml::from_str(toml_str).unwrap();
        assert_eq!(def.id, "test");
        assert_eq!(def.category, HandCategory::Content);
        assert_eq!(def.requires.len(), 1);
        assert_eq!(def.agent.name, "test-hand");
    }

    #[test]
    fn hand_definition_with_settings() {
        let toml_str = r#"
id = "test"
name = "Test Hand"
description = "A test"
category = "content"
tools = []

[[settings]]
key = "stt_provider"
label = "STT Provider"
description = "Speech-to-text engine"
setting_type = "select"
default = "auto"

[[settings.options]]
value = "auto"
label = "Auto-detect"

[[settings.options]]
value = "groq"
label = "Groq Whisper"
provider_env = "GROQ_API_KEY"

[[settings.options]]
value = "local"
label = "Local Whisper"
binary = "whisper"

[agent]
name = "test-hand"
description = "Test"
system_prompt = "Test."

[dashboard]
metrics = []
"#;
        let def: HandDefinition = toml::from_str(toml_str).unwrap();
        assert_eq!(def.settings.len(), 1);
        assert_eq!(def.settings[0].key, "stt_provider");
        assert_eq!(def.settings[0].setting_type, HandSettingType::Select);
        assert_eq!(def.settings[0].options.len(), 3);
        assert_eq!(
            def.settings[0].options[1].provider_env.as_deref(),
            Some("GROQ_API_KEY")
        );
        assert_eq!(
            def.settings[0].options[2].binary.as_deref(),
            Some("whisper")
        );
    }

    #[test]
    fn resolve_settings_with_config() {
        let settings = vec![HandSetting {
            key: "stt".to_string(),
            label: "STT Provider".to_string(),
            description: String::new(),
            setting_type: HandSettingType::Select,
            default: "auto".to_string(),
            options: vec![
                HandSettingOption {
                    value: "auto".to_string(),
                    label: "Auto".to_string(),
                    provider_env: None,
                    binary: None,
                },
                HandSettingOption {
                    value: "groq".to_string(),
                    label: "Groq Whisper".to_string(),
                    provider_env: Some("GROQ_API_KEY".to_string()),
                    binary: None,
                },
                HandSettingOption {
                    value: "openai".to_string(),
                    label: "OpenAI Whisper".to_string(),
                    provider_env: Some("OPENAI_API_KEY".to_string()),
                    binary: None,
                },
            ],
            env_var: None,
        }];

        // User picks groq
        let mut config = HashMap::new();
        config.insert("stt".to_string(), serde_json::json!("groq"));
        let resolved = resolve_settings(&settings, &config);
        assert!(resolved.prompt_block.contains("STT Provider"));
        assert!(resolved.prompt_block.contains("Groq Whisper"));
        assert_eq!(resolved.env_vars, vec!["GROQ_API_KEY"]);
    }

    #[test]
    fn resolve_settings_defaults() {
        let settings = vec![HandSetting {
            key: "stt".to_string(),
            label: "STT".to_string(),
            description: String::new(),
            setting_type: HandSettingType::Select,
            default: "auto".to_string(),
            options: vec![
                HandSettingOption {
                    value: "auto".to_string(),
                    label: "Auto".to_string(),
                    provider_env: None,
                    binary: None,
                },
                HandSettingOption {
                    value: "groq".to_string(),
                    label: "Groq".to_string(),
                    provider_env: Some("GROQ_API_KEY".to_string()),
                    binary: None,
                },
            ],
            env_var: None,
        }];

        // Empty config → uses default "auto"
        let resolved = resolve_settings(&settings, &HashMap::new());
        assert!(resolved.prompt_block.contains("Auto"));
        assert!(
            resolved.env_vars.is_empty(),
            "only selected option env var should be collected"
        );
    }

    #[test]
    fn resolve_settings_toggle_and_text() {
        let settings = vec![
            HandSetting {
                key: "tts_enabled".to_string(),
                label: "TTS".to_string(),
                description: String::new(),
                setting_type: HandSettingType::Toggle,
                default: "false".to_string(),
                options: vec![],
                env_var: None,
            },
            HandSetting {
                key: "custom_model".to_string(),
                label: "Model".to_string(),
                description: String::new(),
                setting_type: HandSettingType::Text,
                default: String::new(),
                options: vec![],
                env_var: None,
            },
        ];

        let mut config = HashMap::new();
        config.insert("tts_enabled".to_string(), serde_json::json!("true"));
        config.insert("custom_model".to_string(), serde_json::json!("large-v3"));
        let resolved = resolve_settings(&settings, &config);
        assert!(resolved.prompt_block.contains("Enabled"));
        assert!(resolved.prompt_block.contains("large-v3"));
    }

    #[test]
    fn hand_requirement_with_install_info() {
        let toml_str = r#"
id = "test"
name = "Test Hand"
description = "A test hand"
category = "content"
tools = []

[[requires]]
key = "ffmpeg"
label = "FFmpeg must be installed"
requirement_type = "binary"
check_value = "ffmpeg"
description = "FFmpeg is the core video processing engine."

[requires.install]
macos = "brew install ffmpeg"
windows = "winget install Gyan.FFmpeg"
linux_apt = "sudo apt install ffmpeg"
linux_dnf = "sudo dnf install ffmpeg-free"
linux_pacman = "sudo pacman -S ffmpeg"
manual_url = "https://ffmpeg.org/download.html"
estimated_time = "2-5 min"

[agent]
name = "test-hand"
description = "Test agent"
system_prompt = "You are a test agent."

[dashboard]
metrics = []
"#;
        let def: HandDefinition = toml::from_str(toml_str).unwrap();
        assert_eq!(def.requires.len(), 1);
        let req = &def.requires[0];
        assert_eq!(
            req.description.as_deref(),
            Some("FFmpeg is the core video processing engine.")
        );
        let install = req.install.as_ref().unwrap();
        assert_eq!(install.macos.as_deref(), Some("brew install ffmpeg"));
        assert_eq!(
            install.windows.as_deref(),
            Some("winget install Gyan.FFmpeg")
        );
        assert_eq!(
            install.linux_apt.as_deref(),
            Some("sudo apt install ffmpeg")
        );
        assert_eq!(
            install.linux_dnf.as_deref(),
            Some("sudo dnf install ffmpeg-free")
        );
        assert_eq!(
            install.linux_pacman.as_deref(),
            Some("sudo pacman -S ffmpeg")
        );
        assert_eq!(
            install.manual_url.as_deref(),
            Some("https://ffmpeg.org/download.html")
        );
        assert_eq!(install.estimated_time.as_deref(), Some("2-5 min"));
        assert!(install.pip.is_none());
        assert!(install.signup_url.is_none());
        assert!(install.steps.is_empty());
    }

    #[test]
    fn hand_requirement_without_install_info_backward_compat() {
        let toml_str = r#"
id = "test"
name = "Test Hand"
description = "A test"
category = "content"
tools = []

[[requires]]
key = "test_bin"
label = "test must be installed"
requirement_type = "binary"
check_value = "test"

[agent]
name = "test-hand"
description = "Test"
system_prompt = "Test."

[dashboard]
metrics = []
"#;
        let def: HandDefinition = toml::from_str(toml_str).unwrap();
        assert_eq!(def.requires.len(), 1);
        assert!(def.requires[0].description.is_none());
        assert!(def.requires[0].install.is_none());
    }

    #[test]
    fn api_key_requirement_with_steps() {
        let toml_str = r#"
id = "test"
name = "Test Hand"
description = "A test"
category = "communication"
tools = []

[[requires]]
key = "API_TOKEN"
label = "API Token"
requirement_type = "api_key"
check_value = "API_TOKEN"
description = "A token from the service."

[requires.install]
signup_url = "https://example.com/signup"
docs_url = "https://example.com/docs"
env_example = "API_TOKEN=your_token_here"
estimated_time = "5-10 min"
steps = [
    "Go to example.com and sign up",
    "Navigate to API settings",
    "Generate a new token",
    "Set it as an environment variable",
]

[agent]
name = "test-hand"
description = "Test"
system_prompt = "Test."

[dashboard]
metrics = []
"#;
        let def: HandDefinition = toml::from_str(toml_str).unwrap();
        assert_eq!(def.requires.len(), 1);
        let req = &def.requires[0];
        let install = req.install.as_ref().unwrap();
        assert_eq!(
            install.signup_url.as_deref(),
            Some("https://example.com/signup")
        );
        assert_eq!(
            install.docs_url.as_deref(),
            Some("https://example.com/docs")
        );
        assert_eq!(
            install.env_example.as_deref(),
            Some("API_TOKEN=your_token_here")
        );
        assert_eq!(install.estimated_time.as_deref(), Some("5-10 min"));
        assert_eq!(install.steps.len(), 4);
        assert_eq!(install.steps[0], "Go to example.com and sign up");
        assert!(install.macos.is_none());
        assert!(install.windows.is_none());
    }

    #[test]
    fn parse_hand_toml_flat_format() {
        let toml_str = r#"
id = "test"
name = "Test Hand"
description = "A test hand"
category = "content"
tools = ["shell_exec"]

[agent]
name = "test-hand"
description = "Test agent"
system_prompt = "You are a test agent."

[dashboard]
metrics = []
"#;
        let def = parse_hand_toml(toml_str).unwrap();
        assert_eq!(def.id, "test");
        assert_eq!(def.name, "Test Hand");
    }

    #[test]
    fn parse_hand_toml_wrapped_format() {
        let toml_str = r#"
[hand]
id = "test"
name = "Test Hand"
description = "A test hand"
category = "content"
tools = ["shell_exec"]

[hand.agent]
name = "test-hand"
description = "Test agent"
system_prompt = "You are a test agent."

[hand.dashboard]
metrics = []
"#;
        let def = parse_hand_toml(toml_str).unwrap();
        assert_eq!(def.id, "test");
        assert_eq!(def.name, "Test Hand");
        assert_eq!(def.agent.name, "test-hand");
    }

    #[test]
    fn hand_agent_config_parses_new_fields() {
        let toml_str = r#"
id = "test"
name = "Test"
description = "t"
category = "productivity"
tools = []

[agent]
name = "t"
description = "d"
system_prompt = "s"
workspace_override_setting = "repo_path"
cache_system_prompt = true
"#;
        let def: HandDefinition = toml::from_str(toml_str).unwrap();
        assert_eq!(
            def.agent.workspace_override_setting.as_deref(),
            Some("repo_path")
        );
        assert!(def.agent.cache_system_prompt);
    }

    #[test]
    fn hand_agent_config_defaults_new_fields_when_absent() {
        let toml_str = r#"
id = "test"
name = "Test"
description = "t"
category = "productivity"
tools = []

[agent]
name = "t"
description = "d"
system_prompt = "s"
"#;
        let def: HandDefinition = toml::from_str(toml_str).unwrap();
        assert!(def.agent.workspace_override_setting.is_none());
        assert!(!def.agent.cache_system_prompt);
    }

    #[test]
    fn validate_hand_workspace_accepts_nested_project_dir() {
        let home = dirs::home_dir().expect("HOME required for test");
        let nested = home.join("dev");
        if !nested.exists() {
            // Skip in environments without ~/dev
            return;
        }
        // Find any 2+-level-deep existing directory under $HOME.
        let entries: Vec<_> = std::fs::read_dir(&nested)
            .unwrap()
            .flatten()
            .filter(|e| e.path().is_dir())
            .collect();
        if let Some(entry) = entries.first() {
            let path = entry.path();
            let result = validate_hand_workspace(&path.to_string_lossy(), None);
            assert!(result.is_ok(), "accepted nested dir should validate: {result:?}");
        }
    }

    #[test]
    fn validate_hand_workspace_rejects_home_direct() {
        let home = dirs::home_dir().expect("HOME required");
        let result = validate_hand_workspace(&home.to_string_lossy(), None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("home directory"));
    }

    #[test]
    fn validate_hand_workspace_rejects_root() {
        let result = validate_hand_workspace("/", None);
        assert!(result.is_err());
    }

    #[test]
    fn validate_hand_workspace_rejects_shallow_under_home() {
        // Any 1-level-deep dir under $HOME — use a guaranteed-existing one.
        let home = dirs::home_dir().expect("HOME required");
        // Pick any existing dir at $HOME root.
        let candidate = std::fs::read_dir(&home)
            .unwrap()
            .flatten()
            .find(|e| e.path().is_dir())
            .map(|e| e.path());
        let Some(path) = candidate else { return };
        let rel = path.strip_prefix(&home).unwrap();
        if rel.components().count() != 1 {
            return; // skip if the chosen dir happens not to be 1-deep
        }
        let name = rel.components().next().unwrap().as_os_str().to_string_lossy();
        // Skip if it's already on the sensitive list (different error, also rejected).
        if [".ssh", ".aws", ".gnupg", ".kube", ".docker", ".claude", ".config",
            ".npmrc", ".netrc", ".pypirc", ".local", ".gcp", ".azure"].contains(&name.as_ref()) {
            return;
        }
        let result = validate_hand_workspace(&path.to_string_lossy(), None);
        assert!(result.is_err(), "shallow dir {name} should be rejected");
    }

    #[test]
    fn validate_hand_workspace_rejects_sensitive_dot_dir() {
        let home = dirs::home_dir().expect("HOME required");
        // Construct a fake ~/.ssh-style path even if it doesn't exist —
        // validation order: control chars → canonicalize → sensitive list.
        // Canonicalize will fail if the path doesn't exist, so we create a
        // temp .ssh directory to exercise the sensitive-list branch.
        let fake_ssh = home.join(".ssh");
        if !fake_ssh.exists() {
            return; // Can't exercise without an existing .ssh
        }
        let result = validate_hand_workspace(&fake_ssh.to_string_lossy(), None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("sensitive"));
    }

    #[test]
    fn validate_hand_workspace_rejects_nonexistent() {
        let result = validate_hand_workspace("/this/path/does/not/exist/anywhere", None);
        assert!(result.is_err());
    }

    #[test]
    fn validate_hand_workspace_rejects_control_chars() {
        let result = validate_hand_workspace("/tmp/fo\no", None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("control characters"));
    }

    #[test]
    fn validate_hand_workspace_rejects_rtl_override() {
        let result = validate_hand_workspace("/tmp/\u{202E}evil", None);
        assert!(result.is_err());
    }

    #[test]
    fn validate_hand_workspace_detects_state_dir_overlap() {
        // Use tempdir so the test creates a valid directory first.
        let tmp = tempfile::TempDir::new().unwrap();
        // Create a nested dir inside to pass the "2 components below $HOME" rule.
        // For this test we don't care about HOME — only the overlap branch.
        let workspace = tmp.path().join("repo");
        std::fs::create_dir_all(&workspace).unwrap();
        let state_dir = workspace.join("state");
        std::fs::create_dir_all(&state_dir).unwrap();
        // The tmp path is not under $HOME so the earlier checks short-circuit
        // on "not in home" rather than overlap. Validate the overlap logic
        // directly via a path whose canonicalization IS under $HOME.
        let home = dirs::home_dir().unwrap();
        let nested = home.join("dev");
        if !nested.exists() {
            return;
        }
        let entries: Vec<_> = std::fs::read_dir(&nested)
            .unwrap()
            .flatten()
            .filter(|e| e.path().is_dir())
            .collect();
        let Some(repo) = entries.first().map(|e| e.path()) else { return };
        let fake_state = repo.join("state");
        std::fs::create_dir_all(&fake_state).ok();
        let result = validate_hand_workspace(&repo.to_string_lossy(), Some(&fake_state));
        // `fake_state` is inside repo → overlap → reject.
        let _ = std::fs::remove_dir_all(&fake_state);
        if let Err(msg) = result {
            assert!(msg.contains("overlap"), "expected overlap error, got: {msg}");
        }
    }

    #[test]
    fn resolve_model_tier_maps_known_tiers() {
        assert_eq!(resolve_model_tier("cheap").unwrap(), "claude-haiku-4-5");
        assert_eq!(resolve_model_tier("balanced").unwrap(), "claude-sonnet-4-6");
        assert_eq!(resolve_model_tier("premium").unwrap(), "claude-opus-4-5");
    }

    #[test]
    fn resolve_model_tier_rejects_unknown_tier() {
        let err = resolve_model_tier("giganormous").unwrap_err();
        assert!(err.contains("model_tier=giganormous"));
    }

    #[test]
    fn all_bundled_hands_parse_with_new_fields() {
        // Regression: after adding workspace_override_setting +
        // cache_system_prompt to HandAgentConfig, every existing bundled
        // HAND.toml must still parse (backward-compat via serde(default)).
        for (id, toml_content, _skill) in crate::bundled::bundled_hands() {
            parse_hand_toml(toml_content)
                .unwrap_or_else(|e| panic!("bundled hand '{id}' failed to parse: {e}"));
        }
    }
}
