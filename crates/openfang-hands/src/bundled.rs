//! Compile-time embedded Hand definitions.

use crate::{parse_hand_toml, HandDefinition, HandError};

/// Returns all bundled hand definitions as (id, HAND.toml content, SKILL.md content).
pub fn bundled_hands() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        (
            "clip",
            include_str!("../bundled/clip/HAND.toml"),
            include_str!("../bundled/clip/SKILL.md"),
        ),
        (
            "lead",
            include_str!("../bundled/lead/HAND.toml"),
            include_str!("../bundled/lead/SKILL.md"),
        ),
        (
            "collector",
            include_str!("../bundled/collector/HAND.toml"),
            include_str!("../bundled/collector/SKILL.md"),
        ),
        (
            "predictor",
            include_str!("../bundled/predictor/HAND.toml"),
            include_str!("../bundled/predictor/SKILL.md"),
        ),
        (
            "researcher",
            include_str!("../bundled/researcher/HAND.toml"),
            include_str!("../bundled/researcher/SKILL.md"),
        ),
        (
            "twitter",
            include_str!("../bundled/twitter/HAND.toml"),
            include_str!("../bundled/twitter/SKILL.md"),
        ),
        (
            "browser",
            include_str!("../bundled/browser/HAND.toml"),
            include_str!("../bundled/browser/SKILL.md"),
        ),
        (
            "trader",
            include_str!("../bundled/trader/HAND.toml"),
            include_str!("../bundled/trader/SKILL.md"),
        ),
        (
            "infisical-sync",
            include_str!("../bundled/infisical-sync/HAND.toml"),
            include_str!("../bundled/infisical-sync/SKILL.md"),
        ),
        (
            "repo-digger",
            include_str!("../bundled/repo-digger/HAND.toml"),
            include_str!("../bundled/repo-digger/SKILL.md"),
        ),
        (
            "book-distiller",
            include_str!("../bundled/book-distiller/HAND.toml"),
            include_str!("../bundled/book-distiller/SKILL.md"),
        ),
    ]
}

/// Parse a bundled HAND.toml into a HandDefinition with its skill content attached.
pub fn parse_bundled(
    _id: &str,
    toml_content: &str,
    skill_content: &str,
) -> Result<HandDefinition, HandError> {
    let mut def: HandDefinition =
        parse_hand_toml(toml_content).map_err(|e| HandError::TomlParse(e.to_string()))?;
    if !skill_content.is_empty() {
        def.skill_content = Some(skill_content.to_string());
    }
    Ok(def)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_hands_not_empty() {
        let hands = bundled_hands();
        assert!(!hands.is_empty());
        assert_eq!(hands[0].0, "clip");
    }

    #[test]
    fn bundled_hands_count() {
        let hands = bundled_hands();
        assert_eq!(hands.len(), 11);
    }

    #[test]
    fn repo_digger_registered() {
        let hands = bundled_hands();
        let ids: Vec<&str> = hands.iter().map(|(id, _, _)| *id).collect();
        assert!(ids.contains(&"repo-digger"));
    }

    #[test]
    fn repo_digger_hand_parses_and_has_required_fields() {
        let hands = bundled_hands();
        let (_, toml_content, skill) = hands
            .iter()
            .find(|(id, _, _)| *id == "repo-digger")
            .expect("repo-digger must be registered");
        let def = parse_hand_toml(toml_content).expect("HAND.toml must parse");
        assert_eq!(def.id, "repo-digger");
        // Load-bearing fields per the plan:
        assert_eq!(
            def.agent.workspace_override_setting.as_deref(),
            Some("repo_path"),
            "workspace_override_setting must point at repo_path"
        );
        assert!(
            def.agent.cache_system_prompt,
            "cache_system_prompt must be true (budget-critical under direct-API)"
        );
        assert_eq!(def.agent.provider, "claude-code");
        // Tools that MUST be declared (MCP-namespaced + code_* + file_*):
        for required in &[
            "code_search",
            "code_agent_spawn",
            "agent_status",
            "file_read",
            "file_write",
            "mcp_docs_mcp_search",
            "mcp_docs_mcp_get_doc",
            "mcp_docs_mcp_list_docs",
        ] {
            assert!(
                def.tools.iter().any(|t| t == required),
                "repo-digger tools missing required entry '{required}'"
            );
        }
        // Tools that MUST NOT be granted (footguns the plan deliberately withheld):
        for forbidden in &["shell_exec", "agent_spawn", "agent_send", "agent_kill"] {
            assert!(
                !def.tools.iter().any(|t| t == forbidden),
                "repo-digger must NOT grant '{forbidden}' — see plan §Architecture"
            );
        }
        assert!(!skill.is_empty(), "SKILL.md must ship with content");
    }

    #[test]
    fn book_distiller_registered() {
        let hands = bundled_hands();
        let ids: Vec<&str> = hands.iter().map(|(id, _, _)| *id).collect();
        assert!(ids.contains(&"book-distiller"));
    }

    #[test]
    fn parse_book_distiller_hand() {
        let (id, toml_content, skill_content) = bundled_hands()
            .into_iter()
            .find(|(id, _, _)| *id == "book-distiller")
            .expect("book-distiller hand must be in bundled_hands()");
        let def = parse_bundled(id, toml_content, skill_content).unwrap();
        assert_eq!(def.id, "book-distiller");
        assert_eq!(def.name, "Book Distiller Hand");
        assert_eq!(def.category, crate::HandCategory::Productivity);
        assert!(def.skill_content.is_some());

        // Required env vars per plan amendment H6.
        let req_keys: Vec<&str> = def.requires.iter().map(|r| r.key.as_str()).collect();
        for required in &[
            "ANTHROPIC_API_KEY",
            "R2_ACCESS_KEY_ID",
            "R2_SECRET_ACCESS_KEY",
            "R2_ENDPOINT_URL",
            "epub_extract_script",
            "r2_fetch_script",
            "distill_chapter_script",
            "r2_put_script",
        ] {
            assert!(
                req_keys.contains(required),
                "book-distiller must declare required '{required}'"
            );
        }

        // Tools that MUST be granted (single-underscore canonical for MCP).
        for required in &[
            "shell_exec",
            "file_read",
            "file_write",
            "file_list",
            "memory_store",
            "memory_recall",
            "event_publish",
            "mcp_dbx_execute_query",
            "mcp_dbx_list_connections",
        ] {
            assert!(
                def.tools.iter().any(|t| t == required),
                "book-distiller tools missing required entry '{required}'"
            );
        }

        // Settings — the user-facing primary inputs.
        let setting_keys: Vec<&str> = def.settings.iter().map(|s| s.key.as_str()).collect();
        for required in &[
            "collection_id",
            "collection_name",
            "book_queue_dir",
            "output_root",
            "library_db_conn",
            "r2_bucket",
            "extract_script_path",
            "r2_fetch_script_path",
            "r2_put_script_path",
            "r2_distilled_prefix",
            "upload_to_r2",
            "distill_script_path",
            "model_prose",
            "model_code",
            "budget_cap_usd",
            "force_reprocess",
            "preview_only",
        ] {
            assert!(
                setting_keys.contains(required),
                "book-distiller settings missing '{required}'"
            );
        }

        // Agent config — plan-mandated values (plan amendments H5, H7, H8).
        assert_eq!(def.agent.provider, "anthropic");
        assert!(
            def.agent.cache_system_prompt,
            "cache_system_prompt must be true per plan amendment H7"
        );
        assert_eq!(
            def.agent.max_iterations,
            Some(200),
            "max_iterations must be 200 per plan amendment H5"
        );
        assert_eq!(
            def.agent.heartbeat_interval_secs,
            Some(120),
            "heartbeat must be 120s for long LLM calls"
        );
        assert!((def.agent.temperature - 0.2).abs() < 0.05);

        // Dashboard — plan-mandated metrics for operability.
        let metric_keys: Vec<&str> = def
            .dashboard
            .metrics
            .iter()
            .map(|m| m.memory_key.as_str())
            .collect();
        for required in &[
            "book_distiller_current_collection",
            "book_distiller_books_done",
            "book_distiller_books_total",
            "book_distiller_chapters_done",
            "book_distiller_usd_active",
            "book_distiller_eta",
            "book_distiller_code_mutations_flagged",
            "book_distiller_truncated_chapters",
            "book_distiller_ratio_violations",
        ] {
            assert!(
                metric_keys.contains(required),
                "book-distiller dashboard missing metric key '{required}'"
            );
        }

        // No `:` in any memory key (plan amendment C5).
        for mk in &metric_keys {
            assert!(
                !mk.contains(':'),
                "memory key '{mk}' must not contain ':' — use '_' (plan amendment C5)"
            );
        }

        // Tools that MUST NOT be granted (footguns).
        for forbidden in &["agent_spawn", "agent_send", "agent_kill", "vault_set"] {
            assert!(
                !def.tools.iter().any(|t| t == forbidden),
                "book-distiller must NOT grant '{forbidden}'"
            );
        }

        assert!(!skill_content.is_empty(), "SKILL.md must ship with content");
        assert!(
            skill_content.contains("epub-extract.py"),
            "SKILL.md must document epub-extract.py install"
        );
    }

    #[test]
    fn parse_clip_hand() {
        let hands = bundled_hands();
        let (id, toml_content, skill_content) = hands[0];
        let def = parse_bundled(id, toml_content, skill_content).unwrap();
        assert_eq!(def.id, "clip");
        assert_eq!(def.name, "Clip Hand");
        assert_eq!(def.category, crate::HandCategory::Content);
        assert!(def.skill_content.is_some());
        assert!(!def.requires.is_empty());
        assert!(!def.tools.is_empty());
        assert!(!def.agent.system_prompt.is_empty());
        assert!(!def.dashboard.metrics.is_empty());
    }

    #[test]
    fn parse_lead_hand() {
        let (id, toml_content, skill_content) = bundled_hands()
            .into_iter()
            .find(|(id, _, _)| *id == "lead")
            .unwrap();
        let def = parse_bundled(id, toml_content, skill_content).unwrap();
        assert_eq!(def.id, "lead");
        assert_eq!(def.name, "Lead Hand");
        assert_eq!(def.category, crate::HandCategory::Data);
        assert!(def.skill_content.is_some());
        assert!(def.requires.is_empty());
        assert!(!def.tools.is_empty());
        assert!(!def.settings.is_empty());
        assert!(!def.dashboard.metrics.is_empty());
        assert!(def.agent.temperature < 0.5);
    }

    #[test]
    fn parse_collector_hand() {
        let (id, toml_content, skill_content) = bundled_hands()
            .into_iter()
            .find(|(id, _, _)| *id == "collector")
            .unwrap();
        let def = parse_bundled(id, toml_content, skill_content).unwrap();
        assert_eq!(def.id, "collector");
        assert_eq!(def.name, "Collector Hand");
        assert_eq!(def.category, crate::HandCategory::Data);
        assert!(def.skill_content.is_some());
        assert!(def.requires.is_empty());
        assert!(def.tools.contains(&"event_publish".to_string()));
        assert!(!def.settings.is_empty());
        assert!(!def.dashboard.metrics.is_empty());
    }

    #[test]
    fn parse_predictor_hand() {
        let (id, toml_content, skill_content) = bundled_hands()
            .into_iter()
            .find(|(id, _, _)| *id == "predictor")
            .unwrap();
        let def = parse_bundled(id, toml_content, skill_content).unwrap();
        assert_eq!(def.id, "predictor");
        assert_eq!(def.name, "Predictor Hand");
        assert_eq!(def.category, crate::HandCategory::Data);
        assert!(def.skill_content.is_some());
        assert!(def.requires.is_empty());
        assert!(!def.settings.is_empty());
        assert!(!def.dashboard.metrics.is_empty());
        assert!((def.agent.temperature - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn parse_researcher_hand() {
        let (id, toml_content, skill_content) = bundled_hands()
            .into_iter()
            .find(|(id, _, _)| *id == "researcher")
            .unwrap();
        let def = parse_bundled(id, toml_content, skill_content).unwrap();
        assert_eq!(def.id, "researcher");
        assert_eq!(def.name, "Researcher Hand");
        assert_eq!(def.category, crate::HandCategory::Productivity);
        assert!(def.skill_content.is_some());
        assert!(def.requires.is_empty());
        assert!(def.tools.contains(&"event_publish".to_string()));
        assert!(!def.settings.is_empty());
        assert!(!def.dashboard.metrics.is_empty());
        assert_eq!(def.agent.max_iterations, Some(25));
    }

    #[test]
    fn parse_twitter_hand() {
        let (id, toml_content, skill_content) = bundled_hands()
            .into_iter()
            .find(|(id, _, _)| *id == "twitter")
            .unwrap();
        let def = parse_bundled(id, toml_content, skill_content).unwrap();
        assert_eq!(def.id, "twitter");
        assert_eq!(def.name, "Twitter Hand");
        assert_eq!(def.category, crate::HandCategory::Communication);
        assert!(def.skill_content.is_some());
        assert!(!def.requires.is_empty()); // requires TWITTER_BEARER_TOKEN
        assert!(!def.settings.is_empty());
        assert!(!def.dashboard.metrics.is_empty());
        assert!((def.agent.temperature - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn parse_browser_hand() {
        let (id, toml_content, skill_content) = bundled_hands()
            .into_iter()
            .find(|(id, _, _)| *id == "browser")
            .unwrap();
        let def = parse_bundled(id, toml_content, skill_content).unwrap();
        assert_eq!(def.id, "browser");
        assert_eq!(def.name, "Browser Hand");
        assert_eq!(def.category, crate::HandCategory::Productivity);
        assert!(def.skill_content.is_some());
        assert!(!def.requires.is_empty()); // requires python3 + chromium
        assert_eq!(def.requires.len(), 2);
        assert!(def.tools.contains(&"browser_navigate".to_string()));
        assert!(def.tools.contains(&"browser_click".to_string()));
        assert!(def.tools.contains(&"browser_type".to_string()));
        assert!(def.tools.contains(&"browser_screenshot".to_string()));
        assert!(def.tools.contains(&"browser_read_page".to_string()));
        assert!(def.tools.contains(&"browser_close".to_string()));
        assert!(!def.settings.is_empty());
        assert!(!def.dashboard.metrics.is_empty());
        assert!((def.agent.temperature - 0.3).abs() < f32::EPSILON);
        assert_eq!(def.agent.max_iterations, Some(60));
    }

    #[test]
    fn parse_trader_hand() {
        let (id, toml_content, skill_content) = bundled_hands()
            .into_iter()
            .find(|(id, _, _)| *id == "trader")
            .unwrap();
        let def = parse_bundled(id, toml_content, skill_content).unwrap();
        assert_eq!(def.id, "trader");
        assert_eq!(def.name, "Trading Hand");
        assert_eq!(def.category, crate::HandCategory::Data);
        assert!(def.skill_content.is_some());
        assert!(def.requires.is_empty()); // no hard requirements
        assert!(!def.tools.is_empty());
        assert!(def.tools.contains(&"event_publish".to_string()));
        assert!(!def.settings.is_empty());
        assert!(!def.dashboard.metrics.is_empty());
        assert!((def.agent.temperature - 0.3).abs() < f32::EPSILON);
        assert_eq!(def.agent.max_iterations, Some(80));
    }

    #[test]
    fn all_bundled_hands_parse() {
        for (id, toml_content, skill_content) in bundled_hands() {
            let def = parse_bundled(id, toml_content, skill_content)
                .unwrap_or_else(|e| panic!("Failed to parse hand '{}': {}", id, e));
            assert_eq!(def.id, id);
            assert!(!def.name.is_empty());
            assert!(!def.tools.is_empty());
            assert!(!def.agent.system_prompt.is_empty());
            assert!(def.skill_content.is_some());
        }
    }

    #[test]
    fn all_einstein_hands_have_schedules() {
        let einstein_ids = [
            "lead",
            "collector",
            "predictor",
            "researcher",
            "twitter",
            "trader",
        ];
        for (id, toml_content, skill_content) in bundled_hands() {
            if einstein_ids.contains(&id) {
                let def = parse_bundled(id, toml_content, skill_content).unwrap();
                assert!(
                    def.tools.contains(&"schedule_create".to_string()),
                    "Einstein hand '{}' must have schedule_create tool",
                    id
                );
                assert!(
                    def.tools.contains(&"schedule_list".to_string()),
                    "Einstein hand '{}' must have schedule_list tool",
                    id
                );
                assert!(
                    def.tools.contains(&"schedule_delete".to_string()),
                    "Einstein hand '{}' must have schedule_delete tool",
                    id
                );
            }
        }
    }

    #[test]
    fn all_einstein_hands_have_memory() {
        let einstein_ids = [
            "lead",
            "collector",
            "predictor",
            "researcher",
            "twitter",
            "trader",
        ];
        for (id, toml_content, skill_content) in bundled_hands() {
            if einstein_ids.contains(&id) {
                let def = parse_bundled(id, toml_content, skill_content).unwrap();
                assert!(
                    def.tools.contains(&"memory_store".to_string()),
                    "Einstein hand '{}' must have memory_store tool",
                    id
                );
                assert!(
                    def.tools.contains(&"memory_recall".to_string()),
                    "Einstein hand '{}' must have memory_recall tool",
                    id
                );
            }
        }
    }

    #[test]
    fn parse_infisical_sync_hand() {
        let (id, toml_content, skill_content) = bundled_hands()
            .into_iter()
            .find(|(id, _, _)| *id == "infisical-sync")
            .expect("infisical-sync hand must be in bundled_hands()");
        let def = parse_bundled(id, toml_content, skill_content).unwrap();
        assert_eq!(def.id, "infisical-sync");
        assert_eq!(def.name, "Infisical Sync Hand");
        assert_eq!(def.category, crate::HandCategory::Security);
        assert!(def.skill_content.is_some());
        // Required env vars
        assert!(
            !def.requires.is_empty(),
            "infisical-sync must declare env var requirements"
        );
        let req_keys: Vec<&str> = def.requires.iter().map(|r| r.key.as_str()).collect();
        assert!(
            req_keys.contains(&"INFISICAL_URL"),
            "must require INFISICAL_URL"
        );
        assert!(
            req_keys.contains(&"INFISICAL_CLIENT_ID"),
            "must require INFISICAL_CLIENT_ID"
        );
        assert!(
            req_keys.contains(&"INFISICAL_CLIENT_SECRET"),
            "must require INFISICAL_CLIENT_SECRET"
        );
        // Einstein scheduling tools
        assert!(
            def.tools.contains(&"schedule_create".to_string()),
            "must have schedule_create"
        );
        assert!(
            def.tools.contains(&"schedule_list".to_string()),
            "must have schedule_list"
        );
        assert!(
            def.tools.contains(&"schedule_delete".to_string()),
            "must have schedule_delete"
        );
        // Memory tools
        assert!(
            def.tools.contains(&"memory_store".to_string()),
            "must have memory_store"
        );
        assert!(
            def.tools.contains(&"memory_recall".to_string()),
            "must have memory_recall"
        );
        // Knowledge graph tools
        assert!(
            def.tools.contains(&"knowledge_add_entity".to_string()),
            "must have knowledge_add_entity"
        );
        assert!(
            def.tools.contains(&"knowledge_add_relation".to_string()),
            "must have knowledge_add_relation"
        );
        assert!(
            def.tools.contains(&"knowledge_query".to_string()),
            "must have knowledge_query"
        );
        // Event bus
        assert!(
            def.tools.contains(&"event_publish".to_string()),
            "must have event_publish"
        );
        // Infisical-specific tools
        assert!(
            def.tools.contains(&"shell_exec".to_string()),
            "must have shell_exec"
        );
        assert!(
            def.tools.contains(&"vault_set".to_string()),
            "must have vault_set"
        );
        assert!(
            def.tools.contains(&"vault_get".to_string()),
            "must have vault_get"
        );
        assert!(
            def.tools.contains(&"vault_list".to_string()),
            "must have vault_list"
        );
        assert!(
            def.tools.contains(&"vault_delete".to_string()),
            "must have vault_delete"
        );
        // Dashboard
        assert!(
            !def.dashboard.metrics.is_empty(),
            "must have dashboard metrics"
        );
        let metric_keys: Vec<&str> = def
            .dashboard
            .metrics
            .iter()
            .map(|m| m.memory_key.as_str())
            .collect();
        assert!(
            metric_keys.contains(&"infisical_sync_secrets_count"),
            "must have secrets_count metric"
        );
        assert!(
            metric_keys.contains(&"infisical_sync_last_sync"),
            "must have last_sync metric"
        );
        // Agent config
        assert!(
            !def.agent.system_prompt.is_empty(),
            "must have system_prompt"
        );
        assert!(
            def.agent.temperature < 0.2,
            "security hand should use low temperature"
        );
    }

    #[test]
    fn all_einstein_hands_have_knowledge_graph() {
        let einstein_ids = [
            "lead",
            "collector",
            "predictor",
            "researcher",
            "twitter",
            "trader",
        ];
        for (id, toml_content, skill_content) in bundled_hands() {
            if einstein_ids.contains(&id) {
                let def = parse_bundled(id, toml_content, skill_content).unwrap();
                assert!(
                    def.tools.contains(&"knowledge_add_entity".to_string()),
                    "Einstein hand '{}' must have knowledge_add_entity tool",
                    id
                );
                assert!(
                    def.tools.contains(&"knowledge_add_relation".to_string()),
                    "Einstein hand '{}' must have knowledge_add_relation tool",
                    id
                );
                assert!(
                    def.tools.contains(&"knowledge_query".to_string()),
                    "Einstein hand '{}' must have knowledge_query tool",
                    id
                );
            }
        }
    }
}
