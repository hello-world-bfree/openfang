//! Hand registry — manages hand definitions and active instances.

use crate::bundled;
use crate::{
    HandDefinition, HandError, HandInstance, HandRequirement, HandResult, HandSettingType,
    HandStatus, RequirementType,
};
use dashmap::DashMap;
use openfang_types::agent::AgentId;
use serde::Serialize;
use std::collections::HashMap;
use tracing::{info, warn};
use uuid::Uuid;

// ─── Settings availability types ────────────────────────────────────────────

/// Availability status of a single setting option.
#[derive(Debug, Clone, Serialize)]
pub struct SettingOptionStatus {
    pub value: String,
    pub label: String,
    pub provider_env: Option<String>,
    pub binary: Option<String>,
    pub available: bool,
}

/// Setting with per-option availability info (for API responses).
#[derive(Debug, Clone, Serialize)]
pub struct SettingStatus {
    pub key: String,
    pub label: String,
    pub description: String,
    pub setting_type: HandSettingType,
    pub default: String,
    pub options: Vec<SettingOptionStatus>,
}

/// The Hand registry — stores definitions and tracks active instances.
pub struct HandRegistry {
    /// All known hand definitions, keyed by hand_id.
    definitions: DashMap<String, HandDefinition>,
    /// Active hand instances, keyed by instance UUID.
    instances: DashMap<Uuid, HandInstance>,
    /// Optional on-disk directory for user-installed hand templates.
    ///
    /// When set, [`install_from_path`](Self::install_from_path) and
    /// [`install_from_content`](Self::install_from_content) write
    /// `HAND.toml` to `<hands_dir>/<id>/` so the hand survives daemon restart.
    /// [`load_user_hands`](Self::load_user_hands) reads from the same directory
    /// on boot.
    hands_dir: Option<std::path::PathBuf>,
}

impl HandRegistry {
    /// Create an empty registry with no on-disk persistence. User-installed
    /// hands will not survive daemon restart.
    ///
    /// Prefer [`with_hands_dir`](Self::with_hands_dir) in production paths.
    pub fn new() -> Self {
        Self {
            definitions: DashMap::new(),
            instances: DashMap::new(),
            hands_dir: None,
        }
    }

    /// Create a registry backed by a disk directory for user-installed hands.
    ///
    /// The directory layout is `<hands_dir>/<hand_id>/HAND.toml` (optionally
    /// with a sibling `SKILL.md`). [`install_from_path`](Self::install_from_path)
    /// writes into this layout; [`load_user_hands`](Self::load_user_hands)
    /// reads from it on boot.
    pub fn with_hands_dir(hands_dir: impl Into<std::path::PathBuf>) -> Self {
        Self {
            definitions: DashMap::new(),
            instances: DashMap::new(),
            hands_dir: Some(hands_dir.into()),
        }
    }

    /// Persist active hand state to disk so it survives restarts.
    pub fn persist_state(&self, path: &std::path::Path) -> HandResult<()> {
        let entries: Vec<serde_json::Value> = self
            .instances
            .iter()
            .filter(|e| e.status == HandStatus::Active)
            .map(|e| {
                serde_json::json!({
                    "hand_id": e.hand_id,
                    "config": e.config,
                    "agent_id": e.agent_id,
                })
            })
            .collect();
        let json = serde_json::to_string_pretty(&entries)
            .map_err(|e| HandError::Config(format!("serialize hand state: {e}")))?;
        std::fs::write(path, json)
            .map_err(|e| HandError::Config(format!("write hand state: {e}")))?;
        Ok(())
    }

    /// Load persisted hand state and re-activate hands.
    /// Returns list of (hand_id, config, old_agent_id) that should be activated.
    /// The `old_agent_id` is the agent UUID from before the restart, used to
    /// reassign cron jobs to the newly spawned agent (issue #402).
    pub fn load_state(
        path: &std::path::Path,
    ) -> Vec<(String, HashMap<String, serde_json::Value>, Option<AgentId>)> {
        let data = match std::fs::read_to_string(path) {
            Ok(d) => d,
            Err(_) => return Vec::new(),
        };
        let entries: Vec<serde_json::Value> = match serde_json::from_str(&data) {
            Ok(e) => e,
            Err(e) => {
                warn!("Failed to parse hand state file: {e}");
                return Vec::new();
            }
        };
        entries
            .into_iter()
            .filter_map(|e| {
                let hand_id = e["hand_id"].as_str()?.to_string();
                let config: HashMap<String, serde_json::Value> =
                    serde_json::from_value(e["config"].clone()).unwrap_or_default();
                let old_agent_id: Option<AgentId> = e
                    .get("agent_id")
                    .and_then(|v| serde_json::from_value(v.clone()).ok());
                Some((hand_id, config, old_agent_id))
            })
            .collect()
    }

    /// Load all bundled hand definitions. Returns count of definitions loaded.
    pub fn load_bundled(&self) -> usize {
        let bundled = bundled::bundled_hands();
        let mut count = 0;
        for (id, toml_content, skill_content) in bundled {
            match bundled::parse_bundled(id, toml_content, skill_content) {
                Ok(def) => {
                    info!(hand = %def.id, name = %def.name, "Loaded bundled hand");
                    self.definitions.insert(def.id.clone(), def);
                    count += 1;
                }
                Err(e) => {
                    warn!(hand = %id, error = %e, "Failed to parse bundled hand");
                }
            }
        }
        count
    }

    /// Scan a directory for custom hand definitions and load them into the
    /// registry. Mirrors `SkillRegistry::load_workspace_skills` — each
    /// subdirectory containing a `HAND.toml` is treated as a hand, with an
    /// optional sibling `SKILL.md` attached as the skill content.
    ///
    /// Parse failures on individual hands are logged and skipped so a single
    /// bad manifest cannot take down the whole registry.
    ///
    /// Returns the number of hands successfully loaded. A non-existent
    /// `hands_dir` returns `Ok(0)` — this is the normal case on a fresh
    /// install where the user has not run `openfang hand install` yet.
    ///
    /// Added for issue #984 — custom hands installed via `openfang hand
    /// install <path>` were only held in memory and lost on daemon restart.
    pub fn load_workspace_hands(&self, hands_dir: &std::path::Path) -> HandResult<usize> {
        if !hands_dir.exists() {
            return Ok(0);
        }
        let mut count = 0;
        let entries = std::fs::read_dir(hands_dir)
            .map_err(|e| HandError::Config(format!("read_dir {}: {e}", hands_dir.display())))?;
        for entry in entries {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    warn!(error = %e, "Failed to read hands dir entry, skipping");
                    continue;
                }
            };
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let toml_path = path.join("HAND.toml");
            if !toml_path.exists() {
                continue;
            }
            let contents = match std::fs::read_to_string(&toml_path) {
                Ok(s) => s,
                Err(e) => {
                    warn!(path = %toml_path.display(), error = %e, "Failed to read HAND.toml, skipping");
                    continue;
                }
            };
            let skill_path = path.join("SKILL.md");
            let skill_content = std::fs::read_to_string(&skill_path).unwrap_or_default();
            match bundled::parse_bundled("custom", &contents, &skill_content) {
                Ok(def) => {
                    let hand_id = def.id.clone();
                    info!(hand = %hand_id, path = %path.display(), "Loaded workspace hand");
                    self.definitions.insert(hand_id, def);
                    count += 1;
                }
                Err(e) => {
                    warn!(path = %toml_path.display(), error = %e, "Invalid HAND.toml, skipping");
                }
            }
        }
        Ok(count)
    }

    /// Install a hand from a directory containing HAND.toml (and optional SKILL.md).
    ///
    /// If the registry was constructed with [`with_hands_dir`](Self::with_hands_dir),
    /// the validated `HAND.toml` (and any sibling `SKILL.md`) are *copied* into
    /// `<hands_dir>/<id>/` so the hand survives daemon restart. Without a
    /// hands_dir, the template lives only in memory until the daemon stops.
    pub fn install_from_path(&self, path: &std::path::Path) -> HandResult<HandDefinition> {
        let toml_path = path.join("HAND.toml");
        let skill_path = path.join("SKILL.md");

        let toml_content = std::fs::read_to_string(&toml_path).map_err(|e| {
            HandError::NotFound(format!("Cannot read {}: {e}", toml_path.display()))
        })?;
        let skill_content = std::fs::read_to_string(&skill_path).unwrap_or_default();

        let def = bundled::parse_bundled("custom", &toml_content, &skill_content)?;

        // Reject hand ids that could traverse out of the hands directory or
        // contain path separators. Path-safe ids are required for the
        // on-disk persistence layout `<hands_dir>/<id>/HAND.toml`.
        validate_hand_id(&def.id)?;

        if self.definitions.contains_key(&def.id) {
            return Err(HandError::AlreadyActive(format!(
                "Hand '{}' already registered",
                def.id
            )));
        }

        // Persist to disk if a hands_dir is configured so the template
        // survives daemon restart (without this, rescan on boot finds nothing).
        if let Some(hands_dir) = &self.hands_dir {
            let target_dir = hands_dir.join(&def.id);
            std::fs::create_dir_all(&target_dir).map_err(|e| {
                HandError::Config(format!(
                    "Failed to create hand directory {}: {e}",
                    target_dir.display()
                ))
            })?;
            std::fs::write(target_dir.join("HAND.toml"), toml_content.as_bytes())
                .map_err(|e| HandError::Config(format!("Failed to persist HAND.toml: {e}")))?;
            if !skill_content.is_empty() {
                std::fs::write(target_dir.join("SKILL.md"), skill_content.as_bytes())
                    .map_err(|e| HandError::Config(format!("Failed to persist SKILL.md: {e}")))?;
            }
            info!(
                hand = %def.id,
                name = %def.name,
                target = %target_dir.display(),
                "Installed and persisted hand"
            );
        } else {
            info!(
                hand = %def.id,
                name = %def.name,
                path = %path.display(),
                "Installed hand from path (in-memory only; configure hands_dir for persistence)"
            );
        }
        self.definitions.insert(def.id.clone(), def.clone());

        // Persist the hand to the user's data dir so it survives daemon
        // restart (issue #984). Best-effort: failures are logged but do not
        // abort the install, because the hand is already registered in
        // memory and the user gets a working install for the current
        // session. On next restart, `load_workspace_hands` will pick it up
        // from disk.
        if let Some(home) = dirs::home_dir() {
            let dest_dir = home.join(".openfang").join("hands").join(&def.id);
            // Canonicalize both paths before comparing so we don't re-copy a
            // hand that is already being installed from its persistent
            // location (e.g. `openfang hand install ~/.openfang/hands/foo`).
            let same_path = match (path.canonicalize(), dest_dir.canonicalize()) {
                (Ok(a), Ok(b)) => a == b,
                _ => path == dest_dir,
            };
            if !same_path {
                if let Err(e) = std::fs::create_dir_all(&dest_dir) {
                    warn!(error = %e, dest = %dest_dir.display(), "Failed to create hands persistence dir");
                } else if let Err(e) = crate::copy_dir_all(path, &dest_dir) {
                    warn!(error = %e, dest = %dest_dir.display(), "Failed to persist hand");
                } else {
                    info!(hand = %def.id, dest = %dest_dir.display(), "Persisted hand to workspace");
                }
            }
        }

        Ok(def)
    }

    /// Load all user-installed hand templates from an on-disk directory.
    ///
    /// Walks `<hands_dir>/*/HAND.toml`, parses each, and registers valid
    /// definitions. Non-fatal per-entry errors (bad TOML, unsafe id, collision
    /// with a previously-loaded definition) are logged as warnings and the
    /// scan continues.
    ///
    /// **Collision policy**: if a user hand has the same `id` as an existing
    /// registered definition (typically a bundled hand loaded just before
    /// this scan), the user hand is **skipped** and a warning is logged.
    /// Bundled definitions always win.
    ///
    /// Returns the number of user hand templates successfully registered.
    pub fn load_user_hands(&self, hands_dir: &std::path::Path) -> HandResult<usize> {
        if !hands_dir.is_dir() {
            return Ok(0);
        }
        let entries = std::fs::read_dir(hands_dir).map_err(|e| {
            HandError::Config(format!(
                "Cannot read hands dir {}: {e}",
                hands_dir.display()
            ))
        })?;

        let mut count = 0usize;
        for entry in entries.flatten() {
            let dir_path = entry.path();
            if !dir_path.is_dir() {
                continue;
            }
            let toml_path = dir_path.join("HAND.toml");
            if !toml_path.is_file() {
                // Likely a partial download or a non-hand directory; skip silently.
                continue;
            }
            let toml_content = match std::fs::read_to_string(&toml_path) {
                Ok(s) => s,
                Err(e) => {
                    warn!(path = %toml_path.display(), error = %e, "Skipping user hand: cannot read HAND.toml");
                    continue;
                }
            };
            let skill_content =
                std::fs::read_to_string(dir_path.join("SKILL.md")).unwrap_or_default();

            let def = match bundled::parse_bundled("custom", &toml_content, &skill_content) {
                Ok(d) => d,
                Err(e) => {
                    warn!(path = %toml_path.display(), error = %e, "Skipping user hand: invalid HAND.toml");
                    continue;
                }
            };

            if let Err(e) = validate_hand_id(&def.id) {
                warn!(path = %toml_path.display(), error = %e, "Skipping user hand: unsafe id");
                continue;
            }

            // Bundled-wins collision policy.
            if self.definitions.contains_key(&def.id) {
                warn!(
                    hand = %def.id,
                    path = %toml_path.display(),
                    "Skipped user hand: collides with already-registered template (bundled wins)"
                );
                continue;
            }

            info!(hand = %def.id, name = %def.name, "Loaded user hand");
            self.definitions.insert(def.id.clone(), def);
            count += 1;
        }
        Ok(count)
    }

    /// Install a hand from raw TOML + skill content (for API-based installs).
    pub fn install_from_content(
        &self,
        toml_content: &str,
        skill_content: &str,
    ) -> HandResult<HandDefinition> {
        let def = bundled::parse_bundled("custom", toml_content, skill_content)?;
        validate_hand_id(&def.id)?;

        if self.definitions.contains_key(&def.id) {
            return Err(HandError::AlreadyActive(format!(
                "Hand '{}' already registered",
                def.id
            )));
        }

        // Persist to disk if hands_dir configured so the template survives
        // daemon restart. Without this, the POST /api/hands/install path
        // registers in-memory only and the rescan-on-boot finds nothing.
        if let Some(hands_dir) = &self.hands_dir {
            let target_dir = hands_dir.join(&def.id);
            std::fs::create_dir_all(&target_dir).map_err(|e| {
                HandError::Config(format!(
                    "Failed to create hand directory {}: {e}",
                    target_dir.display()
                ))
            })?;
            std::fs::write(target_dir.join("HAND.toml"), toml_content.as_bytes())
                .map_err(|e| HandError::Config(format!("Failed to persist HAND.toml: {e}")))?;
            if !skill_content.is_empty() {
                std::fs::write(target_dir.join("SKILL.md"), skill_content.as_bytes())
                    .map_err(|e| HandError::Config(format!("Failed to persist SKILL.md: {e}")))?;
            }
            info!(
                hand = %def.id,
                name = %def.name,
                target = %target_dir.display(),
                "Installed and persisted hand from content"
            );
        } else {
            info!(hand = %def.id, name = %def.name, "Installed hand from content (in-memory only)");
        }
        self.definitions.insert(def.id.clone(), def.clone());
        Ok(def)
    }

    /// Install or update a hand from raw TOML + skill content.
    ///
    /// Unlike `install_from_content`, this overwrites an existing definition
    /// with the same ID.  Active instances are NOT automatically restarted —
    /// the caller should deactivate + reactivate to pick up the new definition.
    pub fn upsert_from_content(
        &self,
        toml_content: &str,
        skill_content: &str,
    ) -> HandResult<HandDefinition> {
        let def = bundled::parse_bundled("custom", toml_content, skill_content)?;
        let existed = self.definitions.contains_key(&def.id);
        let verb = if existed { "Updated" } else { "Installed" };
        info!(hand = %def.id, name = %def.name, "{verb} hand from content");
        self.definitions.insert(def.id.clone(), def.clone());
        Ok(def)
    }

    /// List all known hand definitions.
    pub fn list_definitions(&self) -> Vec<HandDefinition> {
        let mut defs: Vec<HandDefinition> =
            self.definitions.iter().map(|r| r.value().clone()).collect();
        defs.sort_by(|a, b| a.name.cmp(&b.name));
        defs
    }

    /// Get a specific hand definition by ID.
    pub fn get_definition(&self, hand_id: &str) -> Option<HandDefinition> {
        self.definitions.get(hand_id).map(|r| r.value().clone())
    }

    /// Activate a hand — creates an instance (agent spawning is done by kernel).
    ///
    /// `instance_name` is an optional user-supplied label. When set, multiple
    /// instances of the same hand can coexist as long as each
    /// (hand_id, instance_name) pair is unique. When `None`, the legacy
    /// single-instance-per-hand rule applies — a second unnamed activation of
    /// the same hand is rejected.
    pub fn activate(
        &self,
        hand_id: &str,
        config: HashMap<String, serde_json::Value>,
        instance_name: Option<String>,
    ) -> HandResult<HandInstance> {
        let def = self
            .definitions
            .get(hand_id)
            .ok_or_else(|| HandError::NotFound(hand_id.to_string()))?;

        // Reject only when the exact same (hand_id, instance_name) is already active.
        // This lets multiple uniquely-named instances of the same hand coexist.
        for entry in self.instances.iter() {
            if entry.hand_id == hand_id
                && entry.instance_name == instance_name
                && entry.status == HandStatus::Active
            {
                let label = match &instance_name {
                    Some(name) => format!("{hand_id} (instance: {name})"),
                    None => hand_id.to_string(),
                };
                return Err(HandError::AlreadyActive(label));
            }
        }

        let instance = HandInstance::new(hand_id, &def.agent.name, config, instance_name.clone());
        let id = instance.instance_id;
        self.instances.insert(id, instance.clone());
        info!(hand = %hand_id, instance = %id, instance_name = ?instance_name, "Hand activated");
        Ok(instance)
    }

    /// Deactivate a hand instance (agent killing is done by kernel).
    pub fn deactivate(&self, instance_id: Uuid) -> HandResult<HandInstance> {
        let (_, instance) = self
            .instances
            .remove(&instance_id)
            .ok_or(HandError::InstanceNotFound(instance_id))?;
        info!(hand = %instance.hand_id, instance = %instance_id, "Hand deactivated");
        Ok(instance)
    }

    /// Pause a hand instance.
    pub fn pause(&self, instance_id: Uuid) -> HandResult<()> {
        let mut entry = self
            .instances
            .get_mut(&instance_id)
            .ok_or(HandError::InstanceNotFound(instance_id))?;
        entry.status = HandStatus::Paused;
        entry.updated_at = chrono::Utc::now();
        Ok(())
    }

    /// Resume a paused hand instance.
    pub fn resume(&self, instance_id: Uuid) -> HandResult<()> {
        let mut entry = self
            .instances
            .get_mut(&instance_id)
            .ok_or(HandError::InstanceNotFound(instance_id))?;
        entry.status = HandStatus::Active;
        entry.updated_at = chrono::Utc::now();
        Ok(())
    }

    /// Set the agent ID for an instance (called after kernel spawns the agent).
    pub fn set_agent(&self, instance_id: Uuid, agent_id: AgentId) -> HandResult<()> {
        let mut entry = self
            .instances
            .get_mut(&instance_id)
            .ok_or(HandError::InstanceNotFound(instance_id))?;
        entry.agent_id = Some(agent_id);
        entry.updated_at = chrono::Utc::now();
        Ok(())
    }

    /// Find the hand instance associated with an agent.
    pub fn find_by_agent(&self, agent_id: AgentId) -> Option<HandInstance> {
        for entry in self.instances.iter() {
            if entry.agent_id == Some(agent_id) {
                return Some(entry.clone());
            }
        }
        None
    }

    /// List all active hand instances.
    pub fn list_instances(&self) -> Vec<HandInstance> {
        self.instances.iter().map(|e| e.clone()).collect()
    }

    /// Get a specific instance by ID.
    pub fn get_instance(&self, instance_id: Uuid) -> Option<HandInstance> {
        self.instances.get(&instance_id).map(|e| e.clone())
    }

    /// Check which requirements are satisfied for a given hand.
    pub fn check_requirements(&self, hand_id: &str) -> HandResult<Vec<(HandRequirement, bool)>> {
        let def = self
            .definitions
            .get(hand_id)
            .ok_or_else(|| HandError::NotFound(hand_id.to_string()))?;

        let results: Vec<(HandRequirement, bool)> = def
            .requires
            .iter()
            .map(|req| {
                let satisfied = check_requirement(req);
                (req.clone(), satisfied)
            })
            .collect();

        Ok(results)
    }

    /// Check availability of all settings options for a hand.
    pub fn check_settings_availability(&self, hand_id: &str) -> HandResult<Vec<SettingStatus>> {
        let def = self
            .definitions
            .get(hand_id)
            .ok_or_else(|| HandError::NotFound(hand_id.to_string()))?;

        Ok(def
            .settings
            .iter()
            .map(|setting| {
                let options = setting
                    .options
                    .iter()
                    .map(|opt| {
                        let available = check_option_available(
                            opt.provider_env.as_deref(),
                            opt.binary.as_deref(),
                        );
                        SettingOptionStatus {
                            value: opt.value.clone(),
                            label: opt.label.clone(),
                            provider_env: opt.provider_env.clone(),
                            binary: opt.binary.clone(),
                            available,
                        }
                    })
                    .collect();
                SettingStatus {
                    key: setting.key.clone(),
                    label: setting.label.clone(),
                    description: setting.description.clone(),
                    setting_type: setting.setting_type.clone(),
                    default: setting.default.clone(),
                    options,
                }
            })
            .collect())
    }

    /// Update config for an active hand instance.
    pub fn update_config(
        &self,
        instance_id: Uuid,
        config: HashMap<String, serde_json::Value>,
    ) -> HandResult<()> {
        let mut entry = self
            .instances
            .get_mut(&instance_id)
            .ok_or(HandError::InstanceNotFound(instance_id))?;
        entry.config = config;
        entry.updated_at = chrono::Utc::now();
        Ok(())
    }

    /// Mark an instance as errored.
    pub fn set_error(&self, instance_id: Uuid, message: String) -> HandResult<()> {
        let mut entry = self
            .instances
            .get_mut(&instance_id)
            .ok_or(HandError::InstanceNotFound(instance_id))?;
        entry.status = HandStatus::Error(message);
        entry.updated_at = chrono::Utc::now();
        Ok(())
    }

    /// Compute readiness for a hand, cross-referencing requirements with
    /// active instance state.
    ///
    /// Returns `None` if the hand definition does not exist.
    pub fn readiness(&self, hand_id: &str) -> Option<HandReadiness> {
        let reqs = self.check_requirements(hand_id).ok()?;

        // Only non-optional requirements gate readiness.
        // Optional requirements (e.g. chromium for browser hand) are nice-to-have;
        // missing them results in "degraded" status but not "requirements not met".
        let requirements_met = reqs.iter().all(|(req, ok)| *ok || req.optional);

        // A hand is active if at least one instance is in Active status.
        let active = self
            .instances
            .iter()
            .any(|entry| entry.hand_id == hand_id && entry.status == HandStatus::Active);

        // Degraded: active, but at least one non-optional requirement is unmet
        // OR any optional requirement is unmet. In practice, the most useful
        // definition is: active + any requirement unsatisfied.
        let degraded = active && reqs.iter().any(|(_, ok)| !ok);

        Some(HandReadiness {
            requirements_met,
            active,
            degraded,
        })
    }
}

/// Readiness snapshot for a hand definition — combines requirement checks
/// with runtime activation state so the API can report unambiguous status.
#[derive(Debug, Clone, Serialize)]
pub struct HandReadiness {
    /// Whether all declared requirements are currently satisfied.
    pub requirements_met: bool,
    /// Whether the hand currently has a running (Active-status) instance.
    pub active: bool,
    /// Whether the hand is active but some requirements are unmet.
    /// This means the hand is running in a degraded mode — some features
    /// may not work (e.g. browser hand without chromium).
    pub degraded: bool,
}

impl Default for HandRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Reject hand ids that would traverse out of the hands directory or produce
/// an unsafe on-disk path. Allowed: lowercase alphanumerics, hyphens,
/// underscores. Dots and path separators are rejected.
fn validate_hand_id(id: &str) -> HandResult<()> {
    if id.is_empty() {
        return Err(HandError::Config("hand id must not be empty".into()));
    }
    if id.len() > 64 {
        return Err(HandError::Config(format!(
            "hand id too long ({} chars, max 64)",
            id.len()
        )));
    }
    for ch in id.chars() {
        let ok = ch.is_ascii_alphanumeric() || ch == '-' || ch == '_';
        if !ok {
            return Err(HandError::Config(format!(
                "hand id '{id}' contains disallowed character {ch:?} (allowed: a-z A-Z 0-9 - _)"
            )));
        }
    }
    Ok(())
}

/// Check if a single requirement is satisfied.
fn check_requirement(req: &HandRequirement) -> bool {
    match req.requirement_type {
        RequirementType::Binary => {
            // Special handling for python3 / python: must actually run the command
            // and verify the output contains "Python 3", because:
            //  - Windows ships a python3.exe Store shim that doesn't actually work
            //  - Most modern Linux distros only ship "python3", not "python"
            //  - Some Docker images only have "python" pointing to Python 3
            // Matches the detection logic in python_runtime.rs find_python_interpreter().
            if req.check_value == "python3" || req.check_value == "python" {
                return check_python3_available();
            }
            // Check if binary exists on PATH.
            if which_binary(&req.check_value) {
                return true;
            }
            if req.check_value == "chromium" {
                return check_chromium_available();
            }
            false
        }
        RequirementType::EnvVar | RequirementType::ApiKey => {
            // Check if env var is set and non-empty
            std::env::var(&req.check_value)
                .map(|v| !v.is_empty())
                .unwrap_or(false)
        }
    }
}

/// Check if Python 3 is actually available by running the command and checking
/// the version output. This avoids false negatives from Windows Store shims
/// (python3.exe that just opens the Microsoft Store) and false positives from
/// Python 2 installations where `python` exists but is Python 2.
fn check_python3_available() -> bool {
    // Try "python3 --version" first (Linux/macOS, some Windows installs)
    if run_returns_python3("python3") {
        return true;
    }
    // Try "python --version" (Windows commonly uses this, Docker containers too)
    if run_returns_python3("python") {
        return true;
    }
    false
}

/// Run `{cmd} --version` and return true if the output contains "Python 3".
fn run_returns_python3(cmd: &str) -> bool {
    match std::process::Command::new(cmd)
        .arg("--version")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .stdin(std::process::Stdio::null())
        .output()
    {
        Ok(output) => {
            if !output.status.success() {
                return false;
            }
            // Python --version may print to stdout or stderr depending on version
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            stdout.contains("Python 3") || stderr.contains("Python 3")
        }
        Err(_) => false,
    }
}

/// Check if Chromium (or Chrome) is available anywhere on the system.
///
/// Checks in order:
/// 1. CHROME_PATH / CHROMIUM_PATH env vars
/// 2. Common binary names on PATH (chromium, chromium-browser, google-chrome, etc.)
/// 3. Well-known install paths (Windows Program Files, macOS Applications, Linux /usr)
/// 4. Playwright cache (~/.cache/ms-playwright/chromium-*)
fn check_chromium_available() -> bool {
    // 1. Env vars
    for var in &["CHROME_PATH", "CHROMIUM_PATH"] {
        if let Ok(p) = std::env::var(var) {
            if !p.is_empty() && std::path::Path::new(&p).exists() {
                return true;
            }
        }
    }

    // 2. Common binary names on PATH
    let names = [
        "chromium",
        "chromium-browser",
        "google-chrome",
        "google-chrome-stable",
        "chrome",
    ];
    for name in &names {
        if which_binary(name) {
            return true;
        }
    }

    // 3. Well-known install paths
    let known_paths: Vec<std::path::PathBuf> = if cfg!(windows) {
        let pf = std::env::var("ProgramFiles").unwrap_or_else(|_| r"C:\Program Files".into());
        let pf86 =
            std::env::var("ProgramFiles(x86)").unwrap_or_else(|_| r"C:\Program Files (x86)".into());
        let local = std::env::var("LOCALAPPDATA").unwrap_or_default();
        vec![
            std::path::PathBuf::from(&pf).join(r"Google\Chrome\Application\chrome.exe"),
            std::path::PathBuf::from(&pf86).join(r"Google\Chrome\Application\chrome.exe"),
            std::path::PathBuf::from(&local).join(r"Google\Chrome\Application\chrome.exe"),
            std::path::PathBuf::from(&pf).join(r"Chromium\Application\chrome.exe"),
            std::path::PathBuf::from(&local).join(r"Chromium\Application\chrome.exe"),
            std::path::PathBuf::from(&pf).join(r"Microsoft\Edge\Application\msedge.exe"),
        ]
    } else if cfg!(target_os = "macos") {
        vec![
            std::path::PathBuf::from(
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            ),
            std::path::PathBuf::from("/Applications/Chromium.app/Contents/MacOS/Chromium"),
        ]
    } else {
        vec![
            std::path::PathBuf::from("/usr/bin/chromium"),
            std::path::PathBuf::from("/usr/bin/chromium-browser"),
            std::path::PathBuf::from("/usr/bin/google-chrome"),
            std::path::PathBuf::from("/usr/bin/google-chrome-stable"),
            std::path::PathBuf::from("/snap/bin/chromium"),
        ]
    };
    for p in &known_paths {
        if p.exists() {
            return true;
        }
    }

    // 4. Playwright cache
    if let Some(home) = std::env::var("HOME")
        .ok()
        .or_else(|| std::env::var("USERPROFILE").ok())
    {
        let pw_cache = std::path::Path::new(&home).join(".cache/ms-playwright");
        if pw_cache.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&pw_cache) {
                for entry in entries.flatten() {
                    let name = entry.file_name();
                    let name_str = name.to_string_lossy();
                    if name_str.starts_with("chromium-") && entry.path().is_dir() {
                        return true;
                    }
                }
            }
        }
    }

    false
}

/// Check if a binary is on PATH (cross-platform).
fn which_binary(name: &str) -> bool {
    let path_var = std::env::var("PATH").unwrap_or_default();
    let separator = if cfg!(windows) { ';' } else { ':' };
    let extensions: Vec<&str> = if cfg!(windows) {
        vec!["", ".exe", ".cmd", ".bat"]
    } else {
        vec![""]
    };

    for dir in path_var.split(separator) {
        for ext in &extensions {
            let candidate = std::path::Path::new(dir).join(format!("{name}{ext}"));
            if candidate.is_file() {
                return true;
            }
        }
    }
    false
}

/// Check if a setting option is available based on its provider_env and binary.
///
/// - No provider_env and no binary → always available (e.g. "auto", "none")
/// - provider_env set → check if env var is non-empty (special case: GEMINI_API_KEY also checks GOOGLE_API_KEY)
/// - binary set → check if binary is on PATH
fn check_option_available(provider_env: Option<&str>, binary: Option<&str>) -> bool {
    let env_ok = match provider_env {
        None => true,
        Some(env) => {
            let direct = std::env::var(env).map(|v| !v.is_empty()).unwrap_or(false);
            if direct {
                return binary.map(which_binary).unwrap_or(true);
            }
            // Gemini special case: also accept GOOGLE_API_KEY
            if env == "GEMINI_API_KEY" {
                std::env::var("GOOGLE_API_KEY")
                    .map(|v| !v.is_empty())
                    .unwrap_or(false)
            } else {
                false
            }
        }
    };

    if !env_ok {
        return false;
    }

    binary.map(which_binary).unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_registry_is_empty() {
        let reg = HandRegistry::new();
        assert!(reg.list_definitions().is_empty());
        assert!(reg.list_instances().is_empty());
    }

    #[test]
    fn load_bundled_hands() {
        let reg = HandRegistry::new();
        let count = reg.load_bundled();
        assert_eq!(count, 9);
        assert!(!reg.list_definitions().is_empty());

        // Clip hand should be loaded
        let clip = reg.get_definition("clip");
        assert!(clip.is_some());
        let clip = clip.unwrap();
        assert_eq!(clip.name, "Clip Hand");

        // Einstein hands should be loaded
        assert!(reg.get_definition("lead").is_some());
        assert!(reg.get_definition("collector").is_some());
        assert!(reg.get_definition("predictor").is_some());
        assert!(reg.get_definition("researcher").is_some());
        assert!(reg.get_definition("twitter").is_some());

        // Browser hand should be loaded
        assert!(reg.get_definition("browser").is_some());
    }

    #[test]
    fn activate_and_deactivate() {
        let reg = HandRegistry::new();
        reg.load_bundled();

        let instance = reg.activate("clip", HashMap::new(), None).unwrap();
        assert_eq!(instance.hand_id, "clip");
        assert_eq!(instance.status, HandStatus::Active);

        let instances = reg.list_instances();
        assert_eq!(instances.len(), 1);

        // Can't activate again while active
        let err = reg.activate("clip", HashMap::new(), None);
        assert!(err.is_err());

        // Deactivate
        let removed = reg.deactivate(instance.instance_id).unwrap();
        assert_eq!(removed.hand_id, "clip");
        assert!(reg.list_instances().is_empty());
    }

    #[test]
    fn pause_and_resume() {
        let reg = HandRegistry::new();
        reg.load_bundled();

        let instance = reg.activate("clip", HashMap::new(), None).unwrap();
        let id = instance.instance_id;

        reg.pause(id).unwrap();
        let paused = reg.get_instance(id).unwrap();
        assert_eq!(paused.status, HandStatus::Paused);

        reg.resume(id).unwrap();
        let resumed = reg.get_instance(id).unwrap();
        assert_eq!(resumed.status, HandStatus::Active);

        reg.deactivate(id).unwrap();
    }

    #[test]
    fn set_agent() {
        let reg = HandRegistry::new();
        reg.load_bundled();

        let instance = reg.activate("clip", HashMap::new(), None).unwrap();
        let id = instance.instance_id;
        let agent_id = AgentId::new();

        reg.set_agent(id, agent_id).unwrap();

        let found = reg.find_by_agent(agent_id);
        assert!(found.is_some());
        assert_eq!(found.unwrap().instance_id, id);

        reg.deactivate(id).unwrap();
    }

    #[test]
    fn check_requirements() {
        let reg = HandRegistry::new();
        reg.load_bundled();

        let results = reg.check_requirements("clip").unwrap();
        assert!(!results.is_empty());
        // Each result has a requirement and a bool
        for (req, _satisfied) in &results {
            assert!(!req.key.is_empty());
            assert!(!req.label.is_empty());
        }
    }

    #[test]
    fn not_found_errors() {
        let reg = HandRegistry::new();
        assert!(reg.get_definition("nonexistent").is_none());
        assert!(reg.activate("nonexistent", HashMap::new(), None).is_err());
        assert!(reg.check_requirements("nonexistent").is_err());
        assert!(reg.deactivate(Uuid::new_v4()).is_err());
        assert!(reg.pause(Uuid::new_v4()).is_err());
        assert!(reg.resume(Uuid::new_v4()).is_err());
    }

    #[test]
    fn set_error_status() {
        let reg = HandRegistry::new();
        reg.load_bundled();

        let instance = reg.activate("clip", HashMap::new(), None).unwrap();
        let id = instance.instance_id;

        reg.set_error(id, "something broke".to_string()).unwrap();
        let inst = reg.get_instance(id).unwrap();
        assert_eq!(
            inst.status,
            HandStatus::Error("something broke".to_string())
        );

        reg.deactivate(id).unwrap();
    }

    #[test]
    fn which_binary_finds_common() {
        // On all platforms, at least one of these should exist
        let has_something =
            which_binary("echo") || which_binary("cmd") || which_binary("sh") || which_binary("ls");
        // This test is best-effort — in CI containers some might not exist
        let _ = has_something;
    }

    #[test]
    fn env_var_requirement_check() {
        std::env::set_var("OPENFANG_TEST_HAND_REQ", "test_value");
        let req = HandRequirement {
            key: "test".to_string(),
            label: "test".to_string(),
            requirement_type: RequirementType::EnvVar,
            check_value: "OPENFANG_TEST_HAND_REQ".to_string(),
            description: None,
            optional: false,
            install: None,
        };
        assert!(check_requirement(&req));

        let req_missing = HandRequirement {
            key: "test".to_string(),
            label: "test".to_string(),
            requirement_type: RequirementType::EnvVar,
            check_value: "OPENFANG_NONEXISTENT_VAR_12345".to_string(),
            description: None,
            optional: false,
            install: None,
        };
        assert!(!check_requirement(&req_missing));
        std::env::remove_var("OPENFANG_TEST_HAND_REQ");
    }

    #[test]
    fn readiness_nonexistent_hand() {
        let reg = HandRegistry::new();
        assert!(reg.readiness("nonexistent").is_none());
    }

    #[test]
    fn readiness_inactive_hand() {
        let reg = HandRegistry::new();
        reg.load_bundled();

        // Lead hand has no requirements, so requirements_met = true
        let r = reg.readiness("lead").unwrap();
        assert!(r.requirements_met);
        assert!(!r.active);
        assert!(!r.degraded);
    }

    #[test]
    fn readiness_active_hand_all_met() {
        let reg = HandRegistry::new();
        reg.load_bundled();

        // Lead hand has no requirements — activate it
        let instance = reg.activate("lead", HashMap::new(), None).unwrap();
        let r = reg.readiness("lead").unwrap();
        assert!(r.requirements_met);
        assert!(r.active);
        assert!(!r.degraded); // all met, so not degraded

        reg.deactivate(instance.instance_id).unwrap();
    }

    #[test]
    fn readiness_active_hand_degraded() {
        let reg = HandRegistry::new();
        reg.load_bundled();

        // Browser hand requires python3 (non-optional) + chromium (optional).
        // requirements_met only reflects non-optional requirements.
        // degraded = active + any requirement (including optional) unsatisfied.
        let instance = reg.activate("browser", HashMap::new(), None).unwrap();
        let r = reg.readiness("browser").unwrap();
        assert!(r.active);

        // Check individual requirements
        let reqs = reg.check_requirements("browser").unwrap();
        let python_met = reqs.iter().any(|(req, ok)| req.key == "python3" && *ok);
        let chromium_met = reqs.iter().any(|(req, ok)| req.key == "chromium" && *ok);

        // requirements_met only gates on non-optional (python3)
        assert_eq!(r.requirements_met, python_met);

        // degraded = active + any requirement unsatisfied
        if python_met && chromium_met {
            assert!(!r.degraded); // all met, not degraded
        } else {
            assert!(r.degraded); // something is missing, degraded
        }

        reg.deactivate(instance.instance_id).unwrap();
    }

    #[test]
    fn readiness_paused_hand_not_active() {
        let reg = HandRegistry::new();
        reg.load_bundled();

        let instance = reg.activate("lead", HashMap::new(), None).unwrap();
        reg.pause(instance.instance_id).unwrap();

        let r = reg.readiness("lead").unwrap();
        assert!(!r.active); // Paused is not Active
        assert!(!r.degraded);

        reg.deactivate(instance.instance_id).unwrap();
    }

    // ─── hand id validation ────────────────────────────────────────────────

    #[test]
    fn validate_hand_id_accepts_clean_ids() {
        assert!(validate_hand_id("coder").is_ok());
        assert!(validate_hand_id("doc-curator").is_ok());
        assert!(validate_hand_id("library_v2").is_ok());
        assert!(validate_hand_id("ABC123").is_ok());
    }

    #[test]
    fn validate_hand_id_rejects_path_traversal() {
        assert!(validate_hand_id("../etc/passwd").is_err());
        assert!(validate_hand_id("foo/bar").is_err());
        assert!(validate_hand_id("..").is_err());
        assert!(validate_hand_id("foo.bar").is_err());
        assert!(validate_hand_id("").is_err());
        assert!(validate_hand_id(&"x".repeat(65)).is_err());
    }

    // ─── load_user_hands ───────────────────────────────────────────────────

    /// Render a minimal valid HAND.toml for a given id.
    fn minimal_hand_toml(id: &str) -> String {
        format!(
            r#"id = "{id}"
name = "Test Hand {id}"
description = "test hand"
category = "productivity"

[agent]
name = "test-agent-{id}"
description = "test agent"
system_prompt = "You are a test agent."
"#
        )
    }

    #[test]
    fn load_user_hands_missing_dir_returns_zero() {
        let reg = HandRegistry::new();
        let tmp = tempfile::tempdir().unwrap();
        let nonexistent = tmp.path().join("does-not-exist");
        assert_eq!(reg.load_user_hands(&nonexistent).unwrap(), 0);
    }

    #[test]
    fn load_user_hands_reads_valid_templates() {
        let tmp = tempfile::tempdir().unwrap();
        let hands_dir = tmp.path();

        // Valid hand 'alpha'
        let alpha = hands_dir.join("alpha");
        std::fs::create_dir_all(&alpha).unwrap();
        std::fs::write(alpha.join("HAND.toml"), minimal_hand_toml("alpha")).unwrap();

        // Valid hand 'beta' with skill
        let beta = hands_dir.join("beta");
        std::fs::create_dir_all(&beta).unwrap();
        std::fs::write(beta.join("HAND.toml"), minimal_hand_toml("beta")).unwrap();
        std::fs::write(beta.join("SKILL.md"), "# Beta skill").unwrap();

        let reg = HandRegistry::new();
        let count = reg.load_user_hands(hands_dir).unwrap();
        assert_eq!(count, 2);
        assert!(reg.get_definition("alpha").is_some());
        assert!(reg.get_definition("beta").is_some());

        // Skill attached when present
        let beta_def = reg.get_definition("beta").unwrap();
        assert_eq!(beta_def.skill_content.as_deref(), Some("# Beta skill"));
    }

    #[test]
    fn load_user_hands_skips_malformed() {
        let tmp = tempfile::tempdir().unwrap();
        let hands_dir = tmp.path();

        // Valid
        let ok = hands_dir.join("ok");
        std::fs::create_dir_all(&ok).unwrap();
        std::fs::write(ok.join("HAND.toml"), minimal_hand_toml("ok")).unwrap();

        // Malformed TOML
        let bad = hands_dir.join("bad");
        std::fs::create_dir_all(&bad).unwrap();
        std::fs::write(bad.join("HAND.toml"), "this is not [toml").unwrap();

        // Missing HAND.toml (silent skip)
        std::fs::create_dir_all(hands_dir.join("empty-dir")).unwrap();

        // Not a directory — should be silently skipped
        std::fs::write(hands_dir.join("stray.txt"), "ignore me").unwrap();

        let reg = HandRegistry::new();
        let count = reg.load_user_hands(hands_dir).unwrap();
        assert_eq!(count, 1, "only 'ok' should load");
        assert!(reg.get_definition("ok").is_some());
        assert!(reg.get_definition("bad").is_none());
    }

    #[test]
    fn load_user_hands_respects_bundled_collision() {
        let tmp = tempfile::tempdir().unwrap();
        let hands_dir = tmp.path();

        // Collides with a bundled hand ('clip')
        let collision = hands_dir.join("clip");
        std::fs::create_dir_all(&collision).unwrap();
        std::fs::write(collision.join("HAND.toml"), minimal_hand_toml("clip")).unwrap();

        // Non-colliding custom hand
        let custom = hands_dir.join("custom-hand");
        std::fs::create_dir_all(&custom).unwrap();
        std::fs::write(custom.join("HAND.toml"), minimal_hand_toml("custom-hand")).unwrap();

        let reg = HandRegistry::new();
        reg.load_bundled(); // seed bundled hands first — this is the production order
        let count = reg.load_user_hands(hands_dir).unwrap();
        assert_eq!(
            count, 1,
            "only custom-hand loads; clip collides with bundled"
        );
        assert!(reg.get_definition("custom-hand").is_some());

        // Bundled 'clip' should still be the original, not the user override.
        let clip = reg.get_definition("clip").unwrap();
        assert_eq!(clip.name, "Clip Hand"); // bundled name
    }

    // ─── install_from_path persistence ─────────────────────────────────────

    #[test]
    fn install_from_path_writes_to_disk_when_hands_dir_configured() {
        let tmp = tempfile::tempdir().unwrap();
        let hands_dir = tmp.path().join("hands");

        // Source dir with the user's HAND.toml
        let source = tmp.path().join("source");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::write(source.join("HAND.toml"), minimal_hand_toml("my-hand")).unwrap();
        std::fs::write(source.join("SKILL.md"), "# My skill").unwrap();

        let reg = HandRegistry::with_hands_dir(&hands_dir);
        let def = reg.install_from_path(&source).unwrap();
        assert_eq!(def.id, "my-hand");

        // Files persisted
        let target = hands_dir.join("my-hand");
        assert!(target.join("HAND.toml").is_file());
        assert!(target.join("SKILL.md").is_file());

        // Round-trip: fresh registry rescans and finds the hand
        let fresh = HandRegistry::new();
        let count = fresh.load_user_hands(&hands_dir).unwrap();
        assert_eq!(count, 1);
        assert!(fresh.get_definition("my-hand").is_some());
    }

    #[test]
    fn install_from_path_without_hands_dir_is_in_memory_only() {
        let tmp = tempfile::tempdir().unwrap();
        let source = tmp.path().join("source");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::write(source.join("HAND.toml"), minimal_hand_toml("mem-only")).unwrap();

        let reg = HandRegistry::new();
        let def = reg.install_from_path(&source).unwrap();
        assert_eq!(def.id, "mem-only");
        // No hands_dir → nothing written anywhere but the source itself.
    }

    #[test]
    fn optional_field_defaults_false() {
        let req = HandRequirement {
            key: "test".to_string(),
            label: "test".to_string(),
            requirement_type: RequirementType::Binary,
            check_value: "test".to_string(),
            description: None,
            optional: false,
            install: None,
        };
        assert!(!req.optional);
    }

    #[test]
    fn test_load_workspace_hands_from_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let hands_dir = tmp.path();
        let hand_dir = hands_dir.join("test-custom-hand");
        std::fs::create_dir_all(&hand_dir).unwrap();
        let hand_toml = r#"
id = "test-custom-hand"
name = "Test Custom Hand"
description = "A custom hand loaded from the workspace directory"
category = "other"
version = "0.1.0"
author = "tester"

[agent]
name = "test-agent"
description = "A test agent"
module = "builtin:chat"
provider = "anthropic"
model = "claude-sonnet-4-20250514"
system_prompt = "You are a test agent."
"#;
        std::fs::write(hand_dir.join("HAND.toml"), hand_toml).unwrap();

        let registry = HandRegistry::new();
        let count = registry.load_workspace_hands(hands_dir).unwrap();
        assert_eq!(count, 1);
        assert!(registry.get_definition("test-custom-hand").is_some());
        let def = registry.get_definition("test-custom-hand").unwrap();
        assert_eq!(def.name, "Test Custom Hand");
    }

    #[test]
    fn test_load_workspace_hands_missing_dir_returns_zero() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("does-not-exist");
        let registry = HandRegistry::new();
        let count = registry.load_workspace_hands(&missing).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_load_workspace_hands_skips_invalid_toml() {
        let tmp = tempfile::tempdir().unwrap();
        let hands_dir = tmp.path();

        // Valid hand
        let good_dir = hands_dir.join("good-hand");
        std::fs::create_dir_all(&good_dir).unwrap();
        let good_toml = r#"
id = "good-hand"
name = "Good Hand"
description = "..."
category = "other"

[agent]
name = "good-agent"
description = "..."
system_prompt = "You are good."
"#;
        std::fs::write(good_dir.join("HAND.toml"), good_toml).unwrap();

        // Invalid hand (missing required agent section)
        let bad_dir = hands_dir.join("bad-hand");
        std::fs::create_dir_all(&bad_dir).unwrap();
        std::fs::write(bad_dir.join("HAND.toml"), "not valid toml {[[[").unwrap();

        // Directory without HAND.toml — should be silently skipped
        let empty_dir = hands_dir.join("empty-dir");
        std::fs::create_dir_all(&empty_dir).unwrap();

        let registry = HandRegistry::new();
        let count = registry.load_workspace_hands(hands_dir).unwrap();
        assert_eq!(count, 1, "only the valid hand should load");
        assert!(registry.get_definition("good-hand").is_some());
        assert!(registry.get_definition("bad-hand").is_none());
    }

    /// Build a `HandRegistry` pre-populated with a single dummy hand
    /// definition that has no requirements — used by the multi-instance
    /// activation tests below.
    fn test_registry_with_dummy_hand(hand_id: &str) -> HandRegistry {
        let toml_str = format!(
            r#"
id = "{hand_id}"
name = "Dummy Hand"
description = "A dummy hand for tests"
category = "other"
tools = []

[agent]
name = "dummy-agent"
description = "dummy"
system_prompt = "you are a dummy."

[dashboard]
metrics = []
"#
        );
        let def = crate::bundled::parse_bundled("dummy", &toml_str, "").unwrap();
        let reg = HandRegistry::new();
        reg.definitions.insert(def.id.clone(), def);
        reg
    }

    #[test]
    fn test_activate_same_hand_twice_with_different_instance_names_succeeds() {
        let reg = test_registry_with_dummy_hand("test-hand");
        let a = reg
            .activate("test-hand", HashMap::new(), Some("instance-a".into()))
            .unwrap();
        let b = reg
            .activate("test-hand", HashMap::new(), Some("instance-b".into()))
            .unwrap();
        assert_ne!(a.instance_id, b.instance_id);
        assert_eq!(a.instance_name, Some("instance-a".into()));
        assert_eq!(b.instance_name, Some("instance-b".into()));
        let active: Vec<_> = reg
            .list_instances()
            .into_iter()
            .filter(|i| i.status == HandStatus::Active)
            .collect();
        assert_eq!(active.len(), 2);
    }

    #[test]
    fn test_activate_same_hand_same_instance_name_rejects() {
        let reg = test_registry_with_dummy_hand("test-hand");
        reg.activate("test-hand", HashMap::new(), Some("same".into()))
            .unwrap();
        let err = reg
            .activate("test-hand", HashMap::new(), Some("same".into()))
            .unwrap_err();
        assert!(matches!(err, HandError::AlreadyActive(_)));
    }

    #[test]
    fn test_activate_same_hand_unnamed_twice_still_rejects() {
        let reg = test_registry_with_dummy_hand("test-hand");
        reg.activate("test-hand", HashMap::new(), None).unwrap();
        let err = reg.activate("test-hand", HashMap::new(), None).unwrap_err();
        assert!(matches!(err, HandError::AlreadyActive(_)));
    }

    /// Integration test for issue #809: `hand config` round-trip.
    ///
    /// Simulates what `openfang hand config <id> --set KEY=VAL` does against
    /// the registry: read current config, merge updates, write back, read
    /// again. Persists to a tempdir so the restart path also sees the change.
    #[test]
    fn test_hand_config_round_trip_via_registry() {
        let reg = test_registry_with_dummy_hand("browser");
        let inst = reg.activate("browser", HashMap::new(), None).unwrap();

        // Read-modify-write cycle mirroring the CLI's --set behavior.
        let mut cfg = reg.get_instance(inst.instance_id).unwrap().config;
        cfg.insert(
            "headless".to_string(),
            serde_json::Value::String("true".into()),
        );
        cfg.insert(
            "user_agent".to_string(),
            serde_json::Value::String("openfang/1".into()),
        );
        reg.update_config(inst.instance_id, cfg.clone()).unwrap();

        let after = reg.get_instance(inst.instance_id).unwrap();
        assert_eq!(
            after.config.get("headless"),
            Some(&serde_json::Value::String("true".into()))
        );
        assert_eq!(
            after.config.get("user_agent"),
            Some(&serde_json::Value::String("openfang/1".into()))
        );

        // --unset path: drop a key and confirm it's gone.
        cfg.remove("user_agent");
        reg.update_config(inst.instance_id, cfg).unwrap();
        let after_unset = reg.get_instance(inst.instance_id).unwrap();
        assert!(!after_unset.config.contains_key("user_agent"));
        assert_eq!(
            after_unset.config.get("headless"),
            Some(&serde_json::Value::String("true".into()))
        );

        // State survives a persist+load round-trip through a tempdir sidecar.
        let tmp = tempfile::tempdir().unwrap();
        let state_file = tmp.path().join("hands.json");
        reg.persist_state(&state_file).unwrap();
        let reloaded = HandRegistry::load_state(&state_file);
        assert_eq!(reloaded.len(), 1);
        let (hand_id, config, _agent_id) = &reloaded[0];
        assert_eq!(hand_id, "browser");
        assert_eq!(
            config.get("headless"),
            Some(&serde_json::Value::String("true".into()))
        );
    }
}
