//! Slack Socket Mode adapter for the OpenFang channel bridge.
//!
//! Uses Slack Socket Mode WebSocket (app token) for receiving events and the
//! Web API (bot token) for sending responses. No external Slack crate.

use crate::types::{
    split_message, ChannelAdapter, ChannelContent, ChannelMessage, ChannelType, ChannelUser,
};
use async_trait::async_trait;
use dashmap::DashMap;
use futures::{SinkExt, Stream, StreamExt};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, watch, RwLock};
use tracing::{debug, error, info, warn};
use zeroize::Zeroizing;

const SLACK_API_BASE: &str = "https://slack.com/api";
const MAX_BACKOFF: Duration = Duration::from_secs(60);
const INITIAL_BACKOFF: Duration = Duration::from_secs(1);
const SLACK_MSG_LIMIT: usize = 3000;

/// Slack Socket Mode adapter.
pub struct SlackAdapter {
    /// SECURITY: Tokens are zeroized on drop to prevent memory disclosure.
    app_token: Zeroizing<String>,
    bot_token: Zeroizing<String>,
    client: reqwest::Client,
    allowed_channels: Vec<String>,
    shutdown_tx: Arc<watch::Sender<bool>>,
    shutdown_rx: watch::Receiver<bool>,
    /// Bot's own user ID (populated after auth.test).
    bot_user_id: Arc<RwLock<Option<String>>>,
    /// Threads where the bot was @-mentioned. Maps thread_ts -> last interaction time.
    active_threads: Arc<DashMap<String, Instant>>,
    /// How long to track a thread after last interaction.
    thread_ttl: Duration,
    /// Whether auto-thread-reply is enabled.
    auto_thread_reply: bool,
    /// Whether to unfurl (expand previews for) links in posted messages.
    unfurl_links: bool,
}

impl SlackAdapter {
    pub fn new(
        app_token: String,
        bot_token: String,
        allowed_channels: Vec<String>,
        auto_thread_reply: bool,
        thread_ttl_hours: u64,
        unfurl_links: bool,
    ) -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        Self {
            app_token: Zeroizing::new(app_token),
            bot_token: Zeroizing::new(bot_token),
            client: reqwest::Client::new(),
            allowed_channels,
            shutdown_tx: Arc::new(shutdown_tx),
            shutdown_rx,
            bot_user_id: Arc::new(RwLock::new(None)),
            active_threads: Arc::new(DashMap::new()),
            thread_ttl: Duration::from_secs(thread_ttl_hours * 3600),
            auto_thread_reply,
            unfurl_links,
        }
    }

    /// Validate the bot token by calling auth.test.
    async fn validate_bot_token(&self) -> Result<String, Box<dyn std::error::Error>> {
        let resp: serde_json::Value = self
            .client
            .post(format!("{SLACK_API_BASE}/auth.test"))
            .header(
                "Authorization",
                format!("Bearer {}", self.bot_token.as_str()),
            )
            .send()
            .await?
            .json()
            .await?;

        if resp["ok"].as_bool() != Some(true) {
            let err = resp["error"].as_str().unwrap_or("unknown error");
            return Err(format!("Slack auth.test failed: {err}").into());
        }

        let user_id = resp["user_id"].as_str().unwrap_or("unknown").to_string();
        Ok(user_id)
    }

    /// Send a message to a Slack channel via chat.postMessage.
    async fn api_send_message(
        &self,
        channel_id: &str,
        text: &str,
        thread_ts: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let chunks = split_message(text, SLACK_MSG_LIMIT);

        for chunk in chunks {
            let mut body = serde_json::json!({
                "channel": channel_id,
                "text": chunk,
                "unfurl_links": self.unfurl_links,
                "unfurl_media": self.unfurl_links,
            });
            if let Some(ts) = thread_ts {
                body["thread_ts"] = serde_json::json!(ts);
            }

            let resp: serde_json::Value = self
                .client
                .post(format!("{SLACK_API_BASE}/chat.postMessage"))
                .header(
                    "Authorization",
                    format!("Bearer {}", self.bot_token.as_str()),
                )
                .json(&body)
                .send()
                .await?
                .json()
                .await?;

            if resp["ok"].as_bool() != Some(true) {
                let err = resp["error"].as_str().unwrap_or("unknown");
                warn!("Slack chat.postMessage failed: {err}");
            }
        }
        Ok(())
    }
}

/// Convert GitHub/CommonMark-flavored markdown to Slack mrkdwn.
///
/// Handles the common cases that show up in agent output:
/// - `**bold**` + `__bold__` → `*bold*`
/// - `*italic*` + `_italic_` → `_italic_` (idempotent where possible)
/// - `### Heading` / `## Heading` / `# Heading` → `*Heading*` on own line
/// - `---` horizontal rule → blank line
/// - GitHub-style `| col | col |` tables → plain-text aligned rows
/// - Leaves code fences (```) and inline code (`) alone (same in both).
/// - Leaves list bullets (-, *) + link syntax alone.
///
/// Not a perfect parser — Slack mrkdwn is whitespace- and position-sensitive,
/// and a full Commonmark→mrkdwn converter would be its own crate. This
/// covers the 90% case for agent-emitted curation reports without dragging
/// in a dependency.
fn markdown_to_slack_mrkdwn(input: &str) -> String {
    use std::fmt::Write as _;

    let mut out = String::with_capacity(input.len());
    let mut in_code_fence = false;

    for raw_line in input.split('\n') {
        // Code fences are passthrough — don't transform inside a fenced block.
        let fence = raw_line.trim_start();
        if fence.starts_with("```") {
            in_code_fence = !in_code_fence;
            out.push_str(raw_line);
            out.push('\n');
            continue;
        }
        if in_code_fence {
            out.push_str(raw_line);
            out.push('\n');
            continue;
        }

        let line = raw_line;

        // Horizontal rule
        let t = line.trim();
        if t == "---" || t == "***" || t == "___" {
            out.push('\n');
            continue;
        }

        // Headings: collapse #, ##, ### into bold on their own line.
        if let Some(stripped) = t
            .strip_prefix("#### ")
            .or_else(|| t.strip_prefix("### "))
            .or_else(|| t.strip_prefix("## "))
            .or_else(|| t.strip_prefix("# "))
        {
            let _ = writeln!(out, "*{}*", stripped.trim_end_matches(['*', ':']));
            continue;
        }

        // Markdown tables: lines starting with `|` and containing `|` separators.
        // The separator row (`|---|---|`) is dropped; data rows become plain
        // tab-joined text so Slack doesn't render `|` as literal pipes.
        if t.starts_with('|') && t.contains('|') {
            let trimmed = t.trim_matches('|').trim();
            // Skip the ----|---- separator row.
            if trimmed.chars().all(|c| matches!(c, '-' | '|' | ':' | ' ')) {
                continue;
            }
            let cells: Vec<&str> = trimmed.split('|').map(|c| c.trim()).collect();
            out.push_str(&cells.join("    "));
            out.push('\n');
            continue;
        }

        out.push_str(&convert_inline(line));
        out.push('\n');
    }

    // Pop trailing newline to match input convention.
    if out.ends_with('\n') && !input.ends_with('\n') {
        out.pop();
    }
    out
}

/// Inline Markdown → Slack mrkdwn conversions. Applied per line, outside
/// code fences.
fn convert_inline(line: &str) -> String {
    // Early-out on empty / whitespace-only lines.
    if line.trim().is_empty() {
        return line.to_string();
    }

    // `**bold**` → `*bold*`. Do this before single-star rules.
    let mut s = replace_pair(line, "**", "*");
    // `__bold__` → `*bold*` (GitHub bold alternative).
    s = replace_pair(&s, "__", "*");
    // `[text](url)` → `<url|text>` — Slack link syntax.
    s = convert_links(&s);
    s
}

/// Replace paired delimiter `delim` with `replacement` (same on both ends).
///
/// Walks the string left to right toggling an "open" flag; an odd trailing
/// delimiter (no closing partner) is left as-is so asterisks in literal
/// prose aren't mangled.
fn replace_pair(input: &str, delim: &str, replacement: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    let mut open = false;
    while let Some(pos) = rest.find(delim) {
        out.push_str(&rest[..pos]);
        // Peek: is there another delim later in this line to pair with?
        let after = &rest[pos + delim.len()..];
        if !open && after.contains(delim) {
            out.push_str(replacement);
            open = true;
        } else if open {
            out.push_str(replacement);
            open = false;
        } else {
            // Orphan delimiter — keep it literal.
            out.push_str(delim);
        }
        rest = after;
    }
    out.push_str(rest);
    out
}

/// Convert GitHub `[text](url)` to Slack `<url|text>`.
fn convert_links(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(open) = rest.find('[') {
        out.push_str(&rest[..open]);
        let after_open = &rest[open + 1..];
        if let Some(mid) = after_open.find("](") {
            let text = &after_open[..mid];
            let after_mid = &after_open[mid + 2..];
            if let Some(close) = after_mid.find(')') {
                let url = &after_mid[..close];
                if url.starts_with("http://") || url.starts_with("https://") {
                    out.push('<');
                    out.push_str(url);
                    out.push('|');
                    out.push_str(text);
                    out.push('>');
                    rest = &after_mid[close + 1..];
                    continue;
                }
            }
        }
        // Not a link — keep the `[` and continue past it.
        out.push('[');
        rest = after_open;
    }
    out.push_str(rest);
    out
}

#[async_trait]
impl ChannelAdapter for SlackAdapter {
    fn name(&self) -> &str {
        "slack"
    }

    fn channel_type(&self) -> ChannelType {
        ChannelType::Slack
    }

    async fn start(
        &self,
    ) -> Result<Pin<Box<dyn Stream<Item = ChannelMessage> + Send>>, Box<dyn std::error::Error>>
    {
        // Validate bot token first
        let bot_user_id_val = self.validate_bot_token().await?;
        *self.bot_user_id.write().await = Some(bot_user_id_val.clone());
        info!("Slack bot authenticated (user_id: {bot_user_id_val})");

        let (tx, rx) = mpsc::channel::<ChannelMessage>(256);

        let app_token = self.app_token.clone();
        let bot_user_id = self.bot_user_id.clone();
        let allowed_channels = self.allowed_channels.clone();
        let client = self.client.clone();
        let mut shutdown = self.shutdown_rx.clone();
        let active_threads = self.active_threads.clone();
        let auto_thread_reply = self.auto_thread_reply;

        // Spawn periodic cleanup of expired thread entries.
        {
            let active_threads = self.active_threads.clone();
            let thread_ttl = self.thread_ttl;
            let mut cleanup_shutdown = self.shutdown_rx.clone();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_secs(300));
                loop {
                    tokio::select! {
                        _ = interval.tick() => {
                            active_threads.retain(|_, last| last.elapsed() < thread_ttl);
                        }
                        _ = cleanup_shutdown.changed() => {
                            if *cleanup_shutdown.borrow() {
                                return;
                            }
                        }
                    }
                }
            });
        }

        tokio::spawn(async move {
            let mut backoff = INITIAL_BACKOFF;

            loop {
                if *shutdown.borrow() {
                    break;
                }

                // Get a fresh WebSocket URL
                let ws_url_result = get_socket_mode_url(&client, &app_token)
                    .await
                    .map_err(|e| e.to_string());
                let ws_url = match ws_url_result {
                    Ok(url) => url,
                    Err(err_msg) => {
                        warn!("Slack: failed to get WebSocket URL: {err_msg}, retrying in {backoff:?}");
                        tokio::time::sleep(backoff).await;
                        backoff = (backoff * 2).min(MAX_BACKOFF);
                        continue;
                    }
                };

                info!("Connecting to Slack Socket Mode...");

                let ws_result = tokio_tungstenite::connect_async(&ws_url).await;
                let ws_stream = match ws_result {
                    Ok((stream, _)) => stream,
                    Err(e) => {
                        warn!("Slack WebSocket connection failed: {e}, retrying in {backoff:?}");
                        tokio::time::sleep(backoff).await;
                        backoff = (backoff * 2).min(MAX_BACKOFF);
                        continue;
                    }
                };

                backoff = INITIAL_BACKOFF;
                info!("Slack Socket Mode connected");

                let (mut ws_tx, mut ws_rx) = ws_stream.split();

                let should_reconnect = 'inner: loop {
                    let msg = tokio::select! {
                        msg = ws_rx.next() => msg,
                        _ = shutdown.changed() => {
                            if *shutdown.borrow() {
                                let _ = ws_tx.close().await;
                                return;
                            }
                            continue;
                        }
                    };

                    let msg = match msg {
                        Some(Ok(m)) => m,
                        Some(Err(e)) => {
                            warn!("Slack WebSocket error: {e}");
                            break 'inner true;
                        }
                        None => {
                            info!("Slack WebSocket closed");
                            break 'inner true;
                        }
                    };

                    let text = match msg {
                        tokio_tungstenite::tungstenite::Message::Text(t) => t,
                        tokio_tungstenite::tungstenite::Message::Close(_) => {
                            info!("Slack Socket Mode closed by server");
                            break 'inner true;
                        }
                        _ => continue,
                    };

                    let payload: serde_json::Value = match serde_json::from_str(&text) {
                        Ok(v) => v,
                        Err(e) => {
                            warn!("Slack: failed to parse message: {e}");
                            continue;
                        }
                    };

                    let envelope_type = payload["type"].as_str().unwrap_or("");

                    match envelope_type {
                        "hello" => {
                            debug!("Slack Socket Mode hello received");
                        }

                        "events_api" => {
                            // Acknowledge the envelope
                            let envelope_id = payload["envelope_id"].as_str().unwrap_or("");
                            if !envelope_id.is_empty() {
                                let ack = serde_json::json!({ "envelope_id": envelope_id });
                                if let Err(e) = ws_tx
                                    .send(tokio_tungstenite::tungstenite::Message::Text(
                                        serde_json::to_string(&ack).unwrap(),
                                    ))
                                    .await
                                {
                                    error!("Slack: failed to send ack: {e}");
                                    break 'inner true;
                                }
                            }

                            // Extract the event
                            let event = &payload["payload"]["event"];
                            if let Some(msg) = parse_slack_event(
                                event,
                                &bot_user_id,
                                &allowed_channels,
                                &active_threads,
                                auto_thread_reply,
                            )
                            .await
                            {
                                debug!(
                                    "Slack message from {}: {:?}",
                                    msg.sender.display_name, msg.content
                                );
                                if tx.send(msg).await.is_err() {
                                    return;
                                }
                            }
                        }

                        "disconnect" => {
                            let reason = payload["reason"].as_str().unwrap_or("unknown");
                            info!("Slack disconnect request: {reason}");
                            break 'inner true;
                        }

                        _ => {
                            debug!("Slack envelope type: {envelope_type}");
                        }
                    }
                };

                if !should_reconnect || *shutdown.borrow() {
                    break;
                }

                warn!("Slack: reconnecting in {backoff:?}");
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(MAX_BACKOFF);
            }

            info!("Slack Socket Mode loop stopped");
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Ok(Box::pin(stream))
    }

    async fn send(
        &self,
        user: &ChannelUser,
        content: ChannelContent,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let channel_id = &user.platform_id;
        match content {
            ChannelContent::Text(text) => {
                let converted = markdown_to_slack_mrkdwn(&text);
                self.api_send_message(channel_id, &converted, None).await?;
            }
            _ => {
                self.api_send_message(channel_id, "(Unsupported content type)", None)
                    .await?;
            }
        }
        Ok(())
    }

    async fn send_in_thread(
        &self,
        user: &ChannelUser,
        content: ChannelContent,
        thread_id: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let channel_id = &user.platform_id;
        match content {
            ChannelContent::Text(text) => {
                let converted = markdown_to_slack_mrkdwn(&text);
                self.api_send_message(channel_id, &converted, Some(thread_id))
                    .await?;
            }
            _ => {
                self.api_send_message(channel_id, "(Unsupported content type)", Some(thread_id))
                    .await?;
            }
        }
        Ok(())
    }

    async fn stop(&self) -> Result<(), Box<dyn std::error::Error>> {
        let _ = self.shutdown_tx.send(true);
        Ok(())
    }
}

/// Helper to get Socket Mode WebSocket URL.
async fn get_socket_mode_url(
    client: &reqwest::Client,
    app_token: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let resp: serde_json::Value = client
        .post(format!("{SLACK_API_BASE}/apps.connections.open"))
        .header("Authorization", format!("Bearer {app_token}"))
        .header("Content-Type", "application/x-www-form-urlencoded")
        .send()
        .await?
        .json()
        .await?;

    if resp["ok"].as_bool() != Some(true) {
        let err = resp["error"].as_str().unwrap_or("unknown error");
        return Err(format!("Slack apps.connections.open failed: {err}").into());
    }

    resp["url"]
        .as_str()
        .map(String::from)
        .ok_or_else(|| "Missing 'url' in connections.open response".into())
}

/// Parse a Slack event into a `ChannelMessage`.
async fn parse_slack_event(
    event: &serde_json::Value,
    bot_user_id: &Arc<RwLock<Option<String>>>,
    allowed_channels: &[String],
    active_threads: &Arc<DashMap<String, Instant>>,
    auto_thread_reply: bool,
) -> Option<ChannelMessage> {
    let event_type = event["type"].as_str()?;
    if event_type != "message" && event_type != "app_mention" {
        return None;
    }

    // Handle message_changed subtype: extract inner message
    let subtype = event["subtype"].as_str();
    let (msg_data, is_edit) = match subtype {
        Some("message_changed") => {
            // Edited messages have the new content in event.message
            match event.get("message") {
                Some(inner) => (inner, true),
                None => return None,
            }
        }
        Some(_) => return None, // Skip other subtypes (joins, leaves, etc.)
        None => (event, false),
    };

    // Filter out bot's own messages
    if msg_data.get("bot_id").is_some() {
        return None;
    }
    let user_id = msg_data["user"]
        .as_str()
        .or_else(|| event["user"].as_str())?;
    if let Some(ref bid) = *bot_user_id.read().await {
        if user_id == bid {
            return None;
        }
    }

    let channel = event["channel"].as_str()?;

    // Filter by allowed channels
    if !allowed_channels.is_empty() && !allowed_channels.contains(&channel.to_string()) {
        return None;
    }

    let text = msg_data["text"].as_str().unwrap_or("");
    if text.is_empty() {
        return None;
    }

    let ts = if is_edit {
        msg_data["ts"]
            .as_str()
            .unwrap_or(event["ts"].as_str().unwrap_or("0"))
    } else {
        event["ts"].as_str().unwrap_or("0")
    };

    // Parse timestamp (Slack uses epoch.microseconds format)
    let timestamp = ts
        .split('.')
        .next()
        .and_then(|s| s.parse::<i64>().ok())
        .and_then(|epoch| chrono::DateTime::from_timestamp(epoch, 0))
        .unwrap_or_else(chrono::Utc::now);

    // Parse commands (messages starting with /)
    let content = if text.starts_with('/') {
        let parts: Vec<&str> = text.splitn(2, ' ').collect();
        let cmd_name = &parts[0][1..];
        let args = if parts.len() > 1 {
            parts[1].split_whitespace().map(String::from).collect()
        } else {
            vec![]
        };
        ChannelContent::Command {
            name: cmd_name.to_string(),
            args,
        }
    } else {
        ChannelContent::Text(text.to_string())
    };

    // Extract thread_id: threaded replies have `thread_ts`, top-level messages
    // use their own `ts` so the reply will start a thread under the original.
    let thread_id = msg_data["thread_ts"]
        .as_str()
        .or_else(|| event["thread_ts"].as_str())
        .map(|s| s.to_string())
        .or_else(|| Some(ts.to_string()));

    // Check if the bot was @-mentioned (for group_policy = "mention_only")
    let mut metadata = HashMap::new();
    if event_type == "app_mention" {
        metadata.insert("was_mentioned".to_string(), serde_json::Value::Bool(true));
    }

    // Determine the real thread_ts from the event (None for top-level messages).
    let real_thread_ts = msg_data["thread_ts"]
        .as_str()
        .or_else(|| event["thread_ts"].as_str());

    let mut explicitly_mentioned = false;
    if let Some(ref bid) = *bot_user_id.read().await {
        let mention_tag = format!("<@{bid}>");
        if text.contains(&mention_tag) {
            explicitly_mentioned = true;
            metadata.insert("was_mentioned".to_string(), serde_json::json!(true));

            // Track thread for auto-reply on subsequent messages.
            if let Some(tts) = real_thread_ts {
                active_threads.insert(tts.to_string(), Instant::now());
            }
        }
    }

    // Auto-reply to follow-up messages in tracked threads.
    if !explicitly_mentioned && auto_thread_reply {
        if let Some(tts) = real_thread_ts {
            if let Some(mut entry) = active_threads.get_mut(tts) {
                // Refresh TTL and mark as mentioned so dispatch proceeds.
                *entry = Instant::now();
                metadata.insert("was_mentioned".to_string(), serde_json::json!(true));
            }
        }
    }

    Some(ChannelMessage {
        channel: ChannelType::Slack,
        platform_message_id: ts.to_string(),
        sender: ChannelUser {
            platform_id: channel.to_string(),
            display_name: user_id.to_string(), // Slack user IDs as display name
            openfang_user: None,
        },
        content,
        target_agent: None,
        timestamp,
        is_group: true,
        thread_id,
        metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_parse_slack_event_basic() {
        let bot_id = Arc::new(RwLock::new(Some("B123".to_string())));
        let event = serde_json::json!({
            "type": "message",
            "user": "U456",
            "channel": "C789",
            "text": "Hello agent!",
            "ts": "1700000000.000100"
        });

        let msg = parse_slack_event(&event, &bot_id, &[], &Arc::new(DashMap::new()), true)
            .await
            .unwrap();
        assert_eq!(msg.channel, ChannelType::Slack);
        assert_eq!(msg.sender.platform_id, "C789");
        assert!(matches!(msg.content, ChannelContent::Text(ref t) if t == "Hello agent!"));
    }

    #[tokio::test]
    async fn test_parse_slack_event_filters_bot() {
        let bot_id = Arc::new(RwLock::new(Some("B123".to_string())));
        let event = serde_json::json!({
            "type": "message",
            "user": "U456",
            "channel": "C789",
            "text": "Bot message",
            "ts": "1700000000.000100",
            "bot_id": "B999"
        });

        let msg = parse_slack_event(&event, &bot_id, &[], &Arc::new(DashMap::new()), true).await;
        assert!(msg.is_none());
    }

    #[tokio::test]
    async fn test_parse_slack_event_filters_own_user() {
        let bot_id = Arc::new(RwLock::new(Some("U456".to_string())));
        let event = serde_json::json!({
            "type": "message",
            "user": "U456",
            "channel": "C789",
            "text": "My message",
            "ts": "1700000000.000100"
        });

        let msg = parse_slack_event(&event, &bot_id, &[], &Arc::new(DashMap::new()), true).await;
        assert!(msg.is_none());
    }

    #[tokio::test]
    async fn test_parse_slack_event_channel_filter() {
        let bot_id = Arc::new(RwLock::new(None));
        let event = serde_json::json!({
            "type": "message",
            "user": "U456",
            "channel": "C789",
            "text": "Hello",
            "ts": "1700000000.000100"
        });

        // Not in allowed channels
        let msg = parse_slack_event(
            &event,
            &bot_id,
            &["C111".to_string(), "C222".to_string()],
            &Arc::new(DashMap::new()),
            true,
        )
        .await;
        assert!(msg.is_none());

        // In allowed channels
        let msg = parse_slack_event(
            &event,
            &bot_id,
            &["C789".to_string()],
            &Arc::new(DashMap::new()),
            true,
        )
        .await;
        assert!(msg.is_some());
    }

    #[tokio::test]
    async fn test_parse_slack_event_skips_other_subtypes() {
        let bot_id = Arc::new(RwLock::new(None));
        // Non-message_changed subtypes should still be filtered
        let event = serde_json::json!({
            "type": "message",
            "subtype": "channel_join",
            "user": "U456",
            "channel": "C789",
            "text": "joined",
            "ts": "1700000000.000100"
        });

        let msg = parse_slack_event(&event, &bot_id, &[], &Arc::new(DashMap::new()), true).await;
        assert!(msg.is_none());
    }

    #[tokio::test]
    async fn test_parse_slack_command() {
        let bot_id = Arc::new(RwLock::new(None));
        let event = serde_json::json!({
            "type": "message",
            "user": "U456",
            "channel": "C789",
            "text": "/agent hello-world",
            "ts": "1700000000.000100"
        });

        let msg = parse_slack_event(&event, &bot_id, &[], &Arc::new(DashMap::new()), true)
            .await
            .unwrap();
        match &msg.content {
            ChannelContent::Command { name, args } => {
                assert_eq!(name, "agent");
                assert_eq!(args, &["hello-world"]);
            }
            other => panic!("Expected Command, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_parse_slack_event_message_changed() {
        let bot_id = Arc::new(RwLock::new(Some("B123".to_string())));
        let event = serde_json::json!({
            "type": "message",
            "subtype": "message_changed",
            "channel": "C789",
            "message": {
                "user": "U456",
                "text": "Edited message text",
                "ts": "1700000000.000100"
            },
            "ts": "1700000001.000200"
        });

        let msg = parse_slack_event(&event, &bot_id, &[], &Arc::new(DashMap::new()), true)
            .await
            .unwrap();
        assert_eq!(msg.channel, ChannelType::Slack);
        assert_eq!(msg.sender.platform_id, "C789");
        assert!(matches!(msg.content, ChannelContent::Text(ref t) if t == "Edited message text"));
    }

    #[test]
    fn test_slack_adapter_creation() {
        let adapter = SlackAdapter::new(
            "xapp-test".to_string(),
            "xoxb-test".to_string(),
            vec!["C123".to_string()],
            true,
            24,
            true,
        );
        assert_eq!(adapter.name(), "slack");
        assert_eq!(adapter.channel_type(), ChannelType::Slack);
    }

    #[test]
    fn test_slack_adapter_unfurl_links_enabled() {
        let adapter = SlackAdapter::new(
            "xapp-test".to_string(),
            "xoxb-test".to_string(),
            vec![],
            true,
            24,
            true,
        );
        assert!(adapter.unfurl_links);
    }

    #[test]
    fn test_slack_adapter_unfurl_links_disabled() {
        let adapter = SlackAdapter::new(
            "xapp-test".to_string(),
            "xoxb-test".to_string(),
            vec![],
            true,
            24,
            false,
        );
        assert!(!adapter.unfurl_links);
    }

    // ── markdown → slack mrkdwn converter ─────────────────────────────────

    #[test]
    fn mrkdwn_bold_double_star() {
        assert_eq!(markdown_to_slack_mrkdwn("**bold**"), "*bold*");
        assert_eq!(
            markdown_to_slack_mrkdwn("a **bold** b"),
            "a *bold* b"
        );
    }

    #[test]
    fn mrkdwn_bold_double_underscore() {
        assert_eq!(markdown_to_slack_mrkdwn("__bold__"), "*bold*");
    }

    #[test]
    fn mrkdwn_heading_becomes_bold() {
        assert_eq!(markdown_to_slack_mrkdwn("# Title"), "*Title*");
        assert_eq!(markdown_to_slack_mrkdwn("## Phase 1"), "*Phase 1*");
        assert_eq!(markdown_to_slack_mrkdwn("### Sub:"), "*Sub*");
    }

    #[test]
    fn mrkdwn_horizontal_rule_is_blank() {
        assert_eq!(markdown_to_slack_mrkdwn("---"), "");
    }

    #[test]
    fn mrkdwn_table_strips_pipes() {
        let input = "| Title | Category | Source |\n| --- | --- | --- |\n| A | b | c |";
        let out = markdown_to_slack_mrkdwn(input);
        assert!(out.contains("Title    Category    Source"));
        assert!(out.contains("A    b    c"));
        assert!(!out.contains("|---|"));
    }

    #[test]
    fn mrkdwn_link_becomes_slack_syntax() {
        let out = markdown_to_slack_mrkdwn("See [docs](https://example.com/x) now.");
        assert_eq!(out, "See <https://example.com/x|docs> now.");
    }

    #[test]
    fn mrkdwn_relative_link_untouched() {
        // Non-http links are left literal (Slack won't link them anyway).
        let out = markdown_to_slack_mrkdwn("See [spec](./spec.md) now.");
        assert_eq!(out, "See [spec](./spec.md) now.");
    }

    #[test]
    fn mrkdwn_code_fence_passthrough() {
        let input = "start\n```\n**not bold**\n# not header\n```\nend **x**";
        let out = markdown_to_slack_mrkdwn(input);
        // Inside fence: untouched. Outside: converted.
        assert!(out.contains("**not bold**"));
        assert!(out.contains("# not header"));
        assert!(out.contains("end *x*"));
    }

    #[test]
    fn mrkdwn_orphan_delimiter_kept_literal() {
        // Odd number of `**` — leave the trailing one alone.
        assert_eq!(markdown_to_slack_mrkdwn("a **b"), "a **b");
    }

    #[test]
    fn mrkdwn_curation_report_shape() {
        let input = "### Phase 1: Quality Audit\n- **50 low-quality docs**: valid pattern.\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\nSee [arXiv](https://arxiv.org/abs/2604.14725).";
        let out = markdown_to_slack_mrkdwn(input);
        assert!(out.contains("*Phase 1: Quality Audit*"));
        assert!(out.contains("*50 low-quality docs*"));
        assert!(out.contains("A    B"));
        assert!(out.contains("1    2"));
        assert!(out.contains("<https://arxiv.org/abs/2604.14725|arXiv>"));
        assert!(!out.contains("|---"));
    }
}
