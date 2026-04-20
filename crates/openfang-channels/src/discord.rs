//! Discord Gateway adapter for the OpenFang channel bridge.
//!
//! Uses Discord Gateway WebSocket (v10) for receiving messages and the REST API
//! for sending responses. No external Discord crate — just `tokio-tungstenite` + `reqwest`.

use crate::bridge::channel_command_specs;
use crate::types::{
    split_message, ChannelAdapter, ChannelContent, ChannelMessage, ChannelType, ChannelUser,
};
use async_trait::async_trait;
use chrono::Utc;
use dashmap::DashMap;
use futures::{SinkExt, Stream, StreamExt};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, watch, Mutex, RwLock};
use tokio::task::JoinHandle;
use tracing::{debug, error, info, warn};
use zeroize::Zeroizing;

const DISCORD_API_BASE: &str = "https://discord.com/api/v10";
const MAX_BACKOFF: Duration = Duration::from_secs(60);
const INITIAL_BACKOFF: Duration = Duration::from_secs(1);
const DISCORD_MSG_LIMIT: usize = 2000;
/// Maximum pending interactions before rejecting new ones.
const MAX_PENDING_INTERACTIONS: usize = 500;
/// TTL for pending interaction contexts (Discord tokens expire at 15 min).
const INTERACTION_TTL: Duration = Duration::from_secs(840);

// ---------------------------------------------------------------------------
// Security helpers
// ---------------------------------------------------------------------------

/// Build an `Authorization: Bot {token}` header marked as sensitive so HTTP
/// tracing / logging middleware will not leak the bot token.
fn bot_auth_header(token: &str) -> reqwest::header::HeaderValue {
    let mut val = reqwest::header::HeaderValue::from_str(&format!("Bot {token}"))
        .expect("bot token produced an invalid header value");
    val.set_sensitive(true);
    val
}

/// Validate a Discord snowflake ID (17-20 ASCII digit string).
fn is_valid_snowflake(s: &str) -> bool {
    let len = s.len();
    (17..=20).contains(&len) && s.bytes().all(|b| b.is_ascii_digit())
}

// ---------------------------------------------------------------------------
// Interaction types
// ---------------------------------------------------------------------------

/// Tracks a pending Discord interaction awaiting a deferred response.
///
/// Written by the ACK worker after successfully acknowledging the interaction.
/// Read/removed by `send()` to route the response through the webhook endpoint.
struct InteractionContext {
    interaction_id: String,
    /// SENSITIVE — redacted in Debug to prevent log leakage.
    interaction_token: String,
    application_id: String,
    /// Channel where the interaction originated (for fallback to regular message).
    channel_id: String,
    created_at: Instant,
}

impl std::fmt::Debug for InteractionContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InteractionContext")
            .field("interaction_id", &self.interaction_id)
            .field("interaction_token", &"[REDACTED]")
            .field("application_id", &self.application_id)
            .field("channel_id", &self.channel_id)
            .finish()
    }
}

/// Payload sent from the gateway loop to the ACK worker task.
struct InteractionPayload {
    interaction_id: String,
    interaction_token: String,
    application_id: String,
    channel_id: String,
    /// Pre-built message forwarded to the bridge after a successful ACK.
    message: ChannelMessage,
}

/// Discord Gateway opcodes.
mod opcode {
    pub const DISPATCH: u64 = 0;
    pub const HEARTBEAT: u64 = 1;
    pub const IDENTIFY: u64 = 2;
    pub const RESUME: u64 = 6;
    pub const RECONNECT: u64 = 7;
    pub const INVALID_SESSION: u64 = 9;
    pub const HELLO: u64 = 10;
    pub const HEARTBEAT_ACK: u64 = 11;
}

/// Build a Discord gateway heartbeat (opcode 1) payload.
///
/// Per the Discord gateway spec, the payload `d` field is the last received
/// dispatch sequence number, or `null` if no dispatch has been received yet.
/// See: <https://discord.com/developers/docs/topics/gateway#sending-heartbeats>
fn build_heartbeat_payload(last_sequence: Option<u64>) -> serde_json::Value {
    serde_json::json!({
        "op": opcode::HEARTBEAT,
        "d": last_sequence,
    })
}

/// Discord Gateway adapter using WebSocket.
pub struct DiscordAdapter {
    /// SECURITY: Bot token is zeroized on drop to prevent memory disclosure.
    token: Zeroizing<String>,
    client: reqwest::Client,
    allowed_guilds: Vec<String>,
    allowed_users: Vec<String>,
    ignore_bots: bool,
    intents: u64,
    shutdown_tx: Arc<watch::Sender<bool>>,
    shutdown_rx: watch::Receiver<bool>,
    /// Bot's own user ID (populated after READY event).
    bot_user_id: Arc<RwLock<Option<String>>>,
    /// Application ID from READY event (NOT always the same as bot_user_id).
    application_id: Arc<RwLock<Option<String>>>,
    /// Session ID for resume (populated after READY event).
    session_id: Arc<RwLock<Option<String>>>,
    /// Resume gateway URL.
    resume_gateway_url: Arc<RwLock<Option<String>>>,
    /// Pending interaction contexts keyed by interaction_id.
    /// Written by ack_worker, read/removed by send().
    pending_interactions: Arc<DashMap<String, InteractionContext>>,
}

impl DiscordAdapter {
    pub fn new(
        token: String,
        allowed_guilds: Vec<String>,
        allowed_users: Vec<String>,
        ignore_bots: bool,
        intents: u64,
    ) -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        Self {
            token: Zeroizing::new(token),
            client: reqwest::Client::new(),
            allowed_guilds,
            allowed_users,
            ignore_bots,
            intents,
            shutdown_tx: Arc::new(shutdown_tx),
            shutdown_rx,
            bot_user_id: Arc::new(RwLock::new(None)),
            application_id: Arc::new(RwLock::new(None)),
            session_id: Arc::new(RwLock::new(None)),
            resume_gateway_url: Arc::new(RwLock::new(None)),
            pending_interactions: Arc::new(DashMap::new()),
        }
    }

    /// Get the WebSocket gateway URL from the Discord API.
    async fn get_gateway_url(&self) -> Result<String, Box<dyn std::error::Error>> {
        let url = format!("{DISCORD_API_BASE}/gateway/bot");
        let resp: serde_json::Value = self
            .client
            .get(&url)
            .header("Authorization", bot_auth_header(self.token.as_str()))
            .send()
            .await?
            .json()
            .await?;

        let ws_url = resp["url"]
            .as_str()
            .ok_or("Missing 'url' in gateway response")?;

        Ok(format!("{ws_url}/?v=10&encoding=json"))
    }

    /// Send a message to a Discord channel via REST API.
    async fn api_send_message(
        &self,
        channel_id: &str,
        text: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let url = format!("{DISCORD_API_BASE}/channels/{channel_id}/messages");
        let chunks = split_message(text, DISCORD_MSG_LIMIT);

        for chunk in chunks {
            let body = serde_json::json!({ "content": chunk });
            let resp = self
                .client
                .post(&url)
                .header("Authorization", bot_auth_header(self.token.as_str()))
                .json(&body)
                .send()
                .await?;

            if !resp.status().is_success() {
                let body_text = resp.text().await.unwrap_or_default();
                warn!("Discord sendMessage failed: {body_text}");
            }
        }
        Ok(())
    }

    /// Send typing indicator to a Discord channel.
    async fn api_send_typing(&self, channel_id: &str) -> Result<(), Box<dyn std::error::Error>> {
        let url = format!("{DISCORD_API_BASE}/channels/{channel_id}/typing");
        let _ = self
            .client
            .post(&url)
            .header("Authorization", bot_auth_header(self.token.as_str()))
            .send()
            .await?;
        Ok(())
    }

    /// Edit the original deferred interaction response.
    ///
    /// Replaces the "Bot is thinking..." message with the actual response.
    /// `PATCH /webhooks/{app_id}/{token}/messages/@original`
    async fn edit_interaction_original(
        &self,
        app_id: &str,
        interaction_token: &str,
        text: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let url = format!(
            "{DISCORD_API_BASE}/webhooks/{app_id}/{interaction_token}/messages/@original"
        );
        let body = serde_json::json!({ "content": text });
        let resp = self.client.patch(&url).json(&body).send().await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text: String = resp
                .text()
                .await
                .unwrap_or_default()
                .chars()
                .take(256)
                .collect();
            return Err(
                format!("Discord edit interaction failed ({status}): {body_text}").into(),
            );
        }
        Ok(())
    }

    /// Send a follow-up message to an interaction (for multi-chunk responses).
    ///
    /// `POST /webhooks/{app_id}/{token}`
    async fn send_interaction_followup(
        &self,
        app_id: &str,
        interaction_token: &str,
        text: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let url = format!("{DISCORD_API_BASE}/webhooks/{app_id}/{interaction_token}");
        let body = serde_json::json!({ "content": text });
        let resp = self.client.post(&url).json(&body).send().await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body_text: String = resp
                .text()
                .await
                .unwrap_or_default()
                .chars()
                .take(256)
                .collect();
            warn!("Discord interaction follow-up failed ({status}): {body_text}");
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Interaction ACK (standalone — called from spawned tasks)
// ---------------------------------------------------------------------------

/// Send a deferred acknowledgment for an interaction.
///
/// `POST /interactions/{id}/{token}/callback` with `{"type": 5}`
/// Must be called within 3 seconds of receiving the interaction.
async fn ack_interaction(
    client: &reqwest::Client,
    interaction_id: &str,
    interaction_token: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let url = format!(
        "{DISCORD_API_BASE}/interactions/{interaction_id}/{interaction_token}/callback"
    );
    let body = serde_json::json!({ "type": 5 });
    let resp = client.post(&url).json(&body).send().await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body_text: String = resp
            .text()
            .await
            .unwrap_or_default()
            .chars()
            .take(256)
            .collect();
        return Err(format!("Discord interaction ACK failed ({status}): {body_text}").into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Interaction option parsing
// ---------------------------------------------------------------------------

/// Extract a Discord option's `value` field as a String regardless of JSON type.
fn option_value_to_string(opt: &serde_json::Value) -> Option<String> {
    opt["value"]
        .as_str()
        .map(str::to_owned)
        .or_else(|| opt["value"].as_i64().map(|n| n.to_string()))
        .or_else(|| opt["value"].as_f64().map(|n| n.to_string()))
        .or_else(|| opt["value"].as_bool().map(|b| b.to_string()))
}

/// Flatten a nested Discord interaction options tree into `(command_name, args)`.
///
/// For subcommands (type 1), the subcommand name becomes the first arg element
/// so that `handle_command("schedule", ["add", "agent1", "* * * * *", "hi"])`
/// matches the existing bridge interface.
fn flatten_interaction_options(
    command_name: &str,
    options: &serde_json::Value,
) -> (String, Vec<String>) {
    let Some(opts) = options.as_array() else {
        return (command_name.to_string(), vec![]);
    };

    for opt in opts {
        let t = opt["type"].as_u64().unwrap_or(0);
        // Subcommand group (type 2) — recurse one level deeper
        if t == 2 {
            let sub_name = opt["name"].as_str().unwrap_or(command_name);
            return flatten_interaction_options(sub_name, &opt["options"]);
        }
        // Subcommand (type 1) — its name becomes the first arg
        if t == 1 {
            let sub_name = opt["name"].as_str().unwrap_or("");
            let mut args = vec![sub_name.to_string()];
            if let Some(sub_opts) = opt["options"].as_array() {
                args.extend(sub_opts.iter().filter_map(option_value_to_string));
            }
            return (command_name.to_string(), args);
        }
    }

    // No subcommand — collect top-level option values
    let args = opts.iter().filter_map(option_value_to_string).collect();
    (command_name.to_string(), args)
}

// ---------------------------------------------------------------------------
// Slash command registration
// ---------------------------------------------------------------------------

/// Build Discord Application Command definitions from OpenFang's command specs.
fn build_command_definitions() -> Vec<serde_json::Value> {
    channel_command_specs()
        .iter()
        .map(|spec| {
            let options = build_options_for_command(spec.name, spec.help);
            let mut cmd = serde_json::json!({
                "name": spec.name,
                "type": 1, // CHAT_INPUT
                "description": truncate_desc(spec.desc),
            });
            if !options.is_empty() {
                cmd["options"] = serde_json::json!(options);
            }
            cmd
        })
        .collect()
}

/// Truncate description to Discord's 100-character limit.
fn truncate_desc(s: &str) -> String {
    if s.len() <= 100 {
        s.to_string()
    } else {
        format!("{}...", &s[..97])
    }
}

/// Derive Discord command options from the help string pattern.
///
/// Matches patterns like:
///   `/agent <name>`  → required STRING option
///   `/model [name]`  → optional STRING option
///   `/think [on|off]` → optional STRING with choices
///   `/schedule add <agent> <cron> <message> | /schedule del <id>` → subcommands
fn build_options_for_command(name: &str, help: &str) -> Vec<serde_json::Value> {
    match name {
        // --- Subcommand commands ---
        "workflow" => vec![serde_json::json!({
            "name": "run",
            "type": 1, // SUB_COMMAND
            "description": "Run a workflow",
            "options": [
                {"name": "name", "type": 3, "description": "Workflow name", "required": true},
                {"name": "input", "type": 3, "description": "Input text", "required": false}
            ]
        })],
        "trigger" => vec![
            serde_json::json!({
                "name": "add",
                "type": 1,
                "description": "Add a new trigger",
                "options": [
                    {"name": "agent", "type": 3, "description": "Agent name", "required": true},
                    {"name": "pattern", "type": 3, "description": "Match pattern", "required": true},
                    {"name": "prompt", "type": 3, "description": "Prompt text", "required": true}
                ]
            }),
            serde_json::json!({
                "name": "del",
                "type": 1,
                "description": "Delete a trigger",
                "options": [
                    {"name": "id", "type": 3, "description": "Trigger ID", "required": true}
                ]
            }),
        ],
        "schedule" => vec![
            serde_json::json!({
                "name": "add",
                "type": 1,
                "description": "Add a new schedule",
                "options": [
                    {"name": "agent", "type": 3, "description": "Agent name", "required": true},
                    {"name": "cron", "type": 3, "description": "Cron expression (5 fields)", "required": true},
                    {"name": "message", "type": 3, "description": "Message to send", "required": true}
                ]
            }),
            serde_json::json!({
                "name": "del",
                "type": 1,
                "description": "Delete a schedule",
                "options": [
                    {"name": "id", "type": 3, "description": "Schedule ID", "required": true}
                ]
            }),
            serde_json::json!({
                "name": "run",
                "type": 1,
                "description": "Run a schedule now",
                "options": [
                    {"name": "id", "type": 3, "description": "Schedule ID", "required": true}
                ]
            }),
        ],

        // --- Choice-arg commands ---
        "think" => vec![serde_json::json!({
            "name": "toggle",
            "type": 3, // STRING
            "description": "Enable or disable",
            "required": false,
            "choices": [
                {"name": "on", "value": "on"},
                {"name": "off", "value": "off"}
            ]
        })],

        // --- Single required arg ---
        "agent" => vec![serde_json::json!({
            "name": "name", "type": 3, "description": "Agent name", "required": true
        })],
        "approve" => vec![serde_json::json!({
            "name": "id", "type": 3, "description": "Approval request ID", "required": true
        })],
        "reject" => vec![serde_json::json!({
            "name": "id", "type": 3, "description": "Approval request ID", "required": true
        })],

        // --- Single optional arg ---
        "model" => vec![serde_json::json!({
            "name": "name", "type": 3, "description": "Model name", "required": false
        })],

        // --- No-arg commands (everything else) ---
        _ => {
            // Check help string for a trailing argument hint we may have missed
            if help.contains('<') || help.contains('[') {
                // Fallback: generic optional string arg
                vec![serde_json::json!({
                    "name": "args", "type": 3, "description": "Arguments", "required": false
                })]
            } else {
                vec![]
            }
        }
    }
}

/// Register all slash commands with Discord (bulk overwrite).
///
/// Standalone function because it runs inside `tokio::spawn(async move {...})`.
async fn register_commands_impl(
    client: &reqwest::Client,
    token: &Zeroizing<String>,
    app_id: &str,
) -> Result<usize, Box<dyn std::error::Error>> {
    let commands = build_command_definitions();
    let count = commands.len();
    let url = format!("{DISCORD_API_BASE}/applications/{app_id}/commands");

    let resp = client
        .put(&url)
        .header("Authorization", bot_auth_header(token.as_str()))
        .json(&commands)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body: String = resp
            .text()
            .await
            .unwrap_or_default()
            .chars()
            .take(512)
            .collect();
        return Err(format!("Discord command registration failed ({status}): {body}").into());
    }
    Ok(count)
}

#[async_trait]
impl ChannelAdapter for DiscordAdapter {
    fn name(&self) -> &str {
        "discord"
    }

    fn channel_type(&self) -> ChannelType {
        ChannelType::Discord
    }

    async fn start(
        &self,
    ) -> Result<Pin<Box<dyn Stream<Item = ChannelMessage> + Send>>, Box<dyn std::error::Error>>
    {
        let gateway_url = self.get_gateway_url().await?;
        info!("Discord gateway URL obtained");

        let (tx, rx) = mpsc::channel::<ChannelMessage>(256);

        let token = self.token.clone();
        let intents = self.intents;
        let allowed_guilds = self.allowed_guilds.clone();
        let allowed_users = self.allowed_users.clone();
        let ignore_bots = self.ignore_bots;
        let bot_user_id = self.bot_user_id.clone();
        let application_id_store = self.application_id.clone();
        let session_id_store = self.session_id.clone();
        let resume_url_store = self.resume_gateway_url.clone();
        let mut shutdown = self.shutdown_rx.clone();

        // --- ACK worker: decouples interaction ACK from gateway loop ---
        let (ack_tx, mut ack_rx) = mpsc::channel::<InteractionPayload>(64);
        let ack_client = self.client.clone();
        let gateway_client = self.client.clone(); // separate clone for the gateway spawn
        let ack_pending = self.pending_interactions.clone();
        // The ack_worker needs its own tx to forward messages to the bridge
        // after successful ACK. We create a second clone of the bridge tx.
        let (bridge_tx, bridge_rx) = mpsc::channel::<ChannelMessage>(256);

        tokio::spawn({
            let mut shutdown_ack = self.shutdown_rx.clone();
            async move {
                loop {
                    tokio::select! {
                        payload = ack_rx.recv() => {
                            let Some(payload) = payload else { break };

                            // Enforce hard cap on pending interactions
                            if ack_pending.len() >= MAX_PENDING_INTERACTIONS {
                                warn!("Discord: pending interactions at capacity ({MAX_PENDING_INTERACTIONS}), dropping interaction {}", payload.interaction_id);
                                continue;
                            }

                            // Send deferred ACK (must complete before bridge processes the command)
                            if let Err(e) = ack_interaction(
                                &ack_client,
                                &payload.interaction_id,
                                &payload.interaction_token,
                            ).await {
                                warn!("Discord: interaction ACK failed for {}: {e}", payload.interaction_id);
                                continue; // Don't forward to bridge — user sees "interaction failed"
                            }

                            // Store context so send() can route through webhook
                            ack_pending.insert(
                                payload.interaction_id.clone(),
                                InteractionContext {
                                    interaction_id: payload.interaction_id,
                                    interaction_token: payload.interaction_token,
                                    application_id: payload.application_id,
                                    channel_id: payload.channel_id,
                                    created_at: Instant::now(),
                                },
                            );

                            // Forward to bridge for command processing
                            if bridge_tx.send(payload.message).await.is_err() {
                                break;
                            }
                        }
                        _ = shutdown_ack.changed() => {
                            if *shutdown_ack.borrow() { break; }
                        }
                    }
                }
            }
        });

        // --- TTL cleanup: evict expired interaction contexts ---
        {
            let pending_ttl = self.pending_interactions.clone();
            let mut shutdown_ttl = self.shutdown_rx.clone();
            tokio::spawn(async move {
                loop {
                    tokio::select! {
                        _ = tokio::time::sleep(Duration::from_secs(60)) => {
                            let before = pending_ttl.len();
                            pending_ttl.retain(|_, ctx| ctx.created_at.elapsed() < INTERACTION_TTL);
                            let evicted = before - pending_ttl.len();
                            if evicted > 0 {
                                debug!("Discord: evicted {evicted} expired interaction context(s)");
                            }
                        }
                        _ = shutdown_ttl.changed() => {
                            if *shutdown_ttl.borrow() { break; }
                        }
                    }
                }
            });
        }

        tokio::spawn(async move {
            let mut backoff = INITIAL_BACKOFF;
            let mut connect_url = gateway_url;
            // Sequence persists across reconnections for RESUME
            let sequence: Arc<RwLock<Option<u64>>> = Arc::new(RwLock::new(None));

            loop {
                if *shutdown.borrow() {
                    break;
                }

                info!("Connecting to Discord gateway...");

                let ws_result = tokio_tungstenite::connect_async(&connect_url).await;
                let ws_stream = match ws_result {
                    Ok((stream, _)) => stream,
                    Err(e) => {
                        warn!("Discord gateway connection failed: {e}, retrying in {backoff:?}");
                        tokio::time::sleep(backoff).await;
                        backoff = (backoff * 2).min(MAX_BACKOFF);
                        continue;
                    }
                };

                backoff = INITIAL_BACKOFF;
                info!("Discord gateway connected");

                let (ws_tx_raw, mut ws_rx) = ws_stream.split();
                // Wrap the sink so the periodic heartbeat task and the inner
                // loop can both write to it.
                let ws_tx = Arc::new(Mutex::new(ws_tx_raw));
                let mut heartbeat_handle: Option<JoinHandle<()>> = None;
                // Tracks whether the most recent heartbeat we sent has been
                // ACKed (opcode 11). Initialized to `true` so the first
                // heartbeat is always allowed to fire.
                let heartbeat_acked = Arc::new(AtomicBool::new(true));

                // Inner message loop — returns true if we should reconnect
                let should_reconnect = 'inner: loop {
                    let msg = tokio::select! {
                        msg = ws_rx.next() => msg,
                        _ = shutdown.changed() => {
                            if *shutdown.borrow() {
                                info!("Discord shutdown requested");
                                if let Some(h) = heartbeat_handle.take() {
                                    h.abort();
                                }
                                let _ = ws_tx.lock().await.close().await;
                                return;
                            }
                            continue;
                        }
                    };

                    let msg = match msg {
                        Some(Ok(m)) => m,
                        Some(Err(e)) => {
                            warn!("Discord WebSocket error: {e}");
                            break 'inner true;
                        }
                        None => {
                            info!("Discord WebSocket closed");
                            break 'inner true;
                        }
                    };

                    let text = match msg {
                        tokio_tungstenite::tungstenite::Message::Text(t) => t,
                        tokio_tungstenite::tungstenite::Message::Close(_) => {
                            info!("Discord gateway closed by server");
                            break 'inner true;
                        }
                        _ => continue,
                    };

                    let payload: serde_json::Value = match serde_json::from_str(&text) {
                        Ok(v) => v,
                        Err(e) => {
                            warn!("Discord: failed to parse gateway message: {e}");
                            continue;
                        }
                    };

                    let op = payload["op"].as_u64().unwrap_or(999);

                    // Update sequence number from any payload that carries one
                    // (typically dispatch events, opcode 0).
                    if let Some(s) = payload["s"].as_u64() {
                        *sequence.write().await = Some(s);
                    }

                    match op {
                        opcode::HELLO => {
                            let interval =
                                payload["d"]["heartbeat_interval"].as_u64().unwrap_or(45000);
                            debug!("Discord HELLO: heartbeat_interval={interval}ms");

                            // Spawn the periodic heartbeat task BEFORE we send
                            // IDENTIFY/RESUME, per the Discord gateway flow.
                            // Abort any stale handle from a previous attempt
                            // first (defensive — should normally be None here).
                            if let Some(h) = heartbeat_handle.take() {
                                h.abort();
                            }
                            heartbeat_acked.store(true, Ordering::Relaxed);
                            let hb_sink = ws_tx.clone();
                            let hb_seq = sequence.clone();
                            let hb_acked = heartbeat_acked.clone();
                            let mut hb_shutdown = shutdown.clone();
                            heartbeat_handle = Some(tokio::spawn(async move {
                                let mut ticker =
                                    tokio::time::interval(Duration::from_millis(interval));
                                // Skip the immediate first tick — we want to
                                // wait one full interval before the first beat.
                                ticker.tick().await;
                                loop {
                                    tokio::select! {
                                        _ = ticker.tick() => {}
                                        _ = hb_shutdown.changed() => {
                                            if *hb_shutdown.borrow() {
                                                return;
                                            }
                                            continue;
                                        }
                                    }

                                    // If the previous heartbeat was never
                                    // ACKed, the connection is zombied — close
                                    // the sink so the read loop sees EOF and
                                    // triggers a reconnect (Discord spec).
                                    if !hb_acked.swap(false, Ordering::Relaxed) {
                                        warn!(
                                            "Discord: previous heartbeat not ACKed, \
                                             forcing reconnect"
                                        );
                                        let _ = hb_sink.lock().await.close().await;
                                        return;
                                    }

                                    let seq = *hb_seq.read().await;
                                    let payload = build_heartbeat_payload(seq);
                                    let text = match serde_json::to_string(&payload) {
                                        Ok(s) => s,
                                        Err(e) => {
                                            error!("Discord: failed to serialize heartbeat: {e}");
                                            return;
                                        }
                                    };
                                    let send_res = hb_sink
                                        .lock()
                                        .await
                                        .send(tokio_tungstenite::tungstenite::Message::Text(text))
                                        .await;
                                    if let Err(e) = send_res {
                                        warn!("Discord: failed to send heartbeat: {e}");
                                        return;
                                    }
                                    debug!("Discord heartbeat sent (seq={:?})", seq);
                                }
                            }));

                            // Try RESUME if we have a session, otherwise IDENTIFY
                            let has_session = session_id_store.read().await.is_some();
                            let has_seq = sequence.read().await.is_some();

                            let gateway_msg = if has_session && has_seq {
                                let sid = session_id_store.read().await.clone().unwrap();
                                let seq = *sequence.read().await;
                                info!("Discord: sending RESUME (session={sid})");
                                serde_json::json!({
                                    "op": opcode::RESUME,
                                    "d": {
                                        "token": token.as_str(),
                                        "session_id": sid,
                                        "seq": seq
                                    }
                                })
                            } else {
                                info!("Discord: sending IDENTIFY");
                                serde_json::json!({
                                    "op": opcode::IDENTIFY,
                                    "d": {
                                        "token": token.as_str(),
                                        "intents": intents,
                                        "properties": {
                                            "os": "linux",
                                            "browser": "openfang",
                                            "device": "openfang"
                                        }
                                    }
                                })
                            };

                            if let Err(e) = ws_tx
                                .lock()
                                .await
                                .send(tokio_tungstenite::tungstenite::Message::Text(
                                    serde_json::to_string(&gateway_msg).unwrap(),
                                ))
                                .await
                            {
                                error!("Discord: failed to send IDENTIFY/RESUME: {e}");
                                break 'inner true;
                            }
                        }

                        opcode::DISPATCH => {
                            let event_name = payload["t"].as_str().unwrap_or("");
                            let d = &payload["d"];

                            match event_name {
                                "READY" => {
                                    let user_id =
                                        d["user"]["id"].as_str().unwrap_or("").to_string();
                                    let username =
                                        d["user"]["username"].as_str().unwrap_or("unknown");
                                    let app_id = d["application"]["id"]
                                        .as_str()
                                        .unwrap_or("")
                                        .to_string();
                                    let sid = d["session_id"].as_str().unwrap_or("").to_string();
                                    let resume_url =
                                        d["resume_gateway_url"].as_str().unwrap_or("").to_string();

                                    *bot_user_id.write().await = Some(user_id.clone());
                                    *application_id_store.write().await = Some(app_id.clone());
                                    *session_id_store.write().await = Some(sid);
                                    if !resume_url.is_empty() {
                                        *resume_url_store.write().await = Some(resume_url);
                                    }

                                    info!("Discord bot ready: {username} ({user_id}), app_id={app_id}");

                                    // Register slash commands (non-blocking, non-fatal)
                                    if !app_id.is_empty() {
                                        let reg_client = gateway_client.clone();
                                        let reg_token = token.clone();
                                        tokio::spawn(async move {
                                            match register_commands_impl(
                                                &reg_client,
                                                &reg_token,
                                                &app_id,
                                            )
                                            .await
                                            {
                                                Ok(n) => info!(
                                                    "Discord: registered {n} slash commands"
                                                ),
                                                Err(e) => warn!(
                                                    "Discord: command registration failed (non-fatal): {e}"
                                                ),
                                            }
                                        });
                                    }
                                }

                                "MESSAGE_CREATE" | "MESSAGE_UPDATE" => {
                                    if let Some(msg) = parse_discord_message(
                                        d,
                                        &bot_user_id,
                                        &allowed_guilds,
                                        &allowed_users,
                                        ignore_bots,
                                    )
                                    .await
                                    {
                                        debug!(
                                            "Discord {event_name} from {}: {:?}",
                                            msg.sender.display_name, msg.content
                                        );
                                        if tx.send(msg).await.is_err() {
                                            return;
                                        }
                                    }
                                }

                                "INTERACTION_CREATE" => {
                                    // Only handle application commands (type 2) for now
                                    let interaction_type = d["type"].as_u64().unwrap_or(0);
                                    if interaction_type != 2 {
                                        debug!("Discord: ignoring interaction type {interaction_type}");
                                        continue;
                                    }

                                    let interaction_id =
                                        d["id"].as_str().unwrap_or("").to_string();
                                    let interaction_token =
                                        d["token"].as_str().unwrap_or("").to_string();
                                    let app_id = d["application_id"]
                                        .as_str()
                                        .unwrap_or("")
                                        .to_string();
                                    let channel_id =
                                        d["channel_id"].as_str().unwrap_or("").to_string();

                                    // Validate snowflake IDs before using in URLs
                                    if !is_valid_snowflake(&interaction_id) {
                                        warn!("Discord: invalid interaction_id, skipping");
                                        continue;
                                    }
                                    if interaction_token.is_empty() {
                                        warn!("Discord: empty interaction token, skipping");
                                        continue;
                                    }

                                    // Extract user info — guild: member.user, DM: user
                                    let user_data = d["member"]["user"]
                                        .as_object()
                                        .or_else(|| d["user"].as_object());
                                    let (author_id, username) = match user_data {
                                        Some(u) => (
                                            u.get("id")
                                                .and_then(|v| v.as_str())
                                                .unwrap_or("")
                                                .to_string(),
                                            u.get("username")
                                                .and_then(|v| v.as_str())
                                                .unwrap_or("Unknown")
                                                .to_string(),
                                        ),
                                        None => {
                                            warn!("Discord: no user data in interaction");
                                            continue;
                                        }
                                    };

                                    // Filter by allowed users (if configured)
                                    if !allowed_users.is_empty()
                                        && !allowed_users.iter().any(|u| u == &author_id)
                                    {
                                        debug!("Discord: ignoring interaction from unlisted user {author_id}");
                                        continue;
                                    }

                                    // Filter by allowed guilds (if configured)
                                    if !allowed_guilds.is_empty() {
                                        if let Some(guild_id) = d["guild_id"].as_str() {
                                            if !allowed_guilds.iter().any(|g| g == guild_id) {
                                                debug!("Discord: ignoring interaction from unlisted guild {guild_id}");
                                                continue;
                                            }
                                        }
                                    }

                                    // Extract command name and options
                                    let data = &d["data"];
                                    let cmd_name =
                                        data["name"].as_str().unwrap_or("").to_string();
                                    if cmd_name.is_empty() {
                                        continue;
                                    }

                                    let (name, args) = flatten_interaction_options(
                                        &cmd_name,
                                        &data["options"],
                                    );

                                    let is_group = d["guild_id"].as_str().is_some();

                                    let mut metadata = HashMap::new();
                                    metadata.insert(
                                        "is_interaction".to_string(),
                                        serde_json::json!(true),
                                    );
                                    metadata.insert(
                                        "discord_channel_id".to_string(),
                                        serde_json::json!(channel_id),
                                    );

                                    let message = ChannelMessage {
                                        channel: ChannelType::Discord,
                                        // platform_id = interaction_id so send() routes correctly
                                        platform_message_id: interaction_id.clone(),
                                        sender: ChannelUser {
                                            platform_id: interaction_id.clone(),
                                            display_name: username,
                                            openfang_user: None,
                                        },
                                        content: ChannelContent::Command { name, args },
                                        target_agent: None,
                                        timestamp: Utc::now(),
                                        is_group,
                                        thread_id: None,
                                        metadata,
                                    };

                                    let payload = InteractionPayload {
                                        interaction_id: interaction_id.clone(),
                                        interaction_token,
                                        application_id: app_id,
                                        channel_id,
                                        message,
                                    };

                                    // Send to ack_worker (non-blocking)
                                    if ack_tx.try_send(payload).is_err() {
                                        warn!("Discord: ack_worker channel full, dropping interaction {interaction_id}");
                                    }
                                }

                                "RESUMED" => {
                                    info!("Discord session resumed successfully");
                                }

                                _ => {
                                    debug!("Discord event: {event_name}");
                                }
                            }
                        }

                        opcode::HEARTBEAT => {
                            // Server requests immediate heartbeat
                            let seq = *sequence.read().await;
                            let hb = build_heartbeat_payload(seq);
                            let _ = ws_tx
                                .lock()
                                .await
                                .send(tokio_tungstenite::tungstenite::Message::Text(
                                    serde_json::to_string(&hb).unwrap(),
                                ))
                                .await;
                            // The server-requested heartbeat counts as a fresh
                            // beat — reset the ACK gate so the periodic task
                            // doesn't see a stale "unacked" flag.
                            heartbeat_acked.store(false, Ordering::Relaxed);
                        }

                        opcode::HEARTBEAT_ACK => {
                            debug!("Discord heartbeat ACK received");
                            heartbeat_acked.store(true, Ordering::Relaxed);
                        }

                        opcode::RECONNECT => {
                            info!("Discord: server requested reconnect");
                            break 'inner true;
                        }

                        opcode::INVALID_SESSION => {
                            let resumable = payload["d"].as_bool().unwrap_or(false);
                            if resumable {
                                info!("Discord: invalid session (resumable)");
                            } else {
                                info!("Discord: invalid session (not resumable), clearing session");
                                *session_id_store.write().await = None;
                                *sequence.write().await = None;
                            }
                            break 'inner true;
                        }

                        _ => {
                            debug!("Discord: unknown opcode {op}");
                        }
                    }
                };

                // Tear down the heartbeat task before we either exit or
                // reconnect, so it doesn't outlive its WebSocket sink.
                if let Some(h) = heartbeat_handle.take() {
                    h.abort();
                }

                if !should_reconnect || *shutdown.borrow() {
                    break;
                }

                // Try resume URL if available
                if let Some(ref url) = *resume_url_store.read().await {
                    connect_url = format!("{url}/?v=10&encoding=json");
                }

                warn!("Discord: reconnecting in {backoff:?}");
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(MAX_BACKOFF);
            }

            info!("Discord gateway loop stopped");
        });

        // Merge streams: regular messages (rx) + interaction commands (bridge_rx)
        let stream_regular = tokio_stream::wrappers::ReceiverStream::new(rx);
        let stream_interactions = tokio_stream::wrappers::ReceiverStream::new(bridge_rx);
        let merged = futures::stream::select(stream_regular, stream_interactions);
        Ok(Box::pin(merged))
    }

    async fn send(
        &self,
        user: &ChannelUser,
        content: ChannelContent,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let key = &user.platform_id;

        // Check if this is a pending interaction response (platform_id == interaction_id)
        if let Some((_, ctx)) = self.pending_interactions.remove(key) {
            let text = match content {
                ChannelContent::Text(t) => t,
                _ => "(Unsupported content type)".to_string(),
            };
            let chunks = split_message(&text, DISCORD_MSG_LIMIT);

            // First chunk: edit the deferred "thinking..." message
            let fallback_channel = ctx.channel_id.clone();
            let edit_ok = self
                .edit_interaction_original(
                    &ctx.application_id,
                    &ctx.interaction_token,
                    chunks[0],
                )
                .await
                .is_ok();

            if !edit_ok {
                // Fallback: send as regular channel message (token expired, etc.)
                warn!(
                    "Discord: interaction edit failed, falling back to channel"
                );
                return self.api_send_message(&fallback_channel, &text).await;
            }

            // Remaining chunks: follow-up messages
            for chunk in &chunks[1..] {
                if self
                    .send_interaction_followup(
                        &ctx.application_id,
                        &ctx.interaction_token,
                        chunk,
                    )
                    .await
                    .is_err()
                {
                    warn!("Discord: interaction follow-up failed");
                    break;
                }
            }
            return Ok(());
        }

        // Regular message path (platform_id is the channel_id)
        match content {
            ChannelContent::Text(text) => {
                self.api_send_message(key, &text).await?;
            }
            _ => {
                self.api_send_message(key, "(Unsupported content type)")
                    .await?;
            }
        }
        Ok(())
    }

    async fn send_typing(&self, user: &ChannelUser) -> Result<(), Box<dyn std::error::Error>> {
        // Interaction deferred ACK already shows "Bot is thinking..."
        if self.pending_interactions.contains_key(&user.platform_id) {
            return Ok(());
        }
        self.api_send_typing(&user.platform_id).await
    }

    async fn stop(&self) -> Result<(), Box<dyn std::error::Error>> {
        let _ = self.shutdown_tx.send(true);
        Ok(())
    }
}

/// Parse a Discord MESSAGE_CREATE or MESSAGE_UPDATE payload into a `ChannelMessage`.
async fn parse_discord_message(
    d: &serde_json::Value,
    bot_user_id: &Arc<RwLock<Option<String>>>,
    allowed_guilds: &[String],
    allowed_users: &[String],
    ignore_bots: bool,
) -> Option<ChannelMessage> {
    let author = d.get("author")?;
    let author_id = author["id"].as_str()?;

    // Filter out bot's own messages
    if let Some(ref bid) = *bot_user_id.read().await {
        if author_id == bid {
            return None;
        }
    }

    // Filter out other bots (configurable via ignore_bots)
    if ignore_bots && author["bot"].as_bool() == Some(true) {
        return None;
    }

    // Filter by allowed users
    if !allowed_users.is_empty() && !allowed_users.iter().any(|u| u == author_id) {
        debug!("Discord: ignoring message from unlisted user {author_id}");
        return None;
    }

    // Filter by allowed guilds
    if !allowed_guilds.is_empty() {
        if let Some(guild_id) = d["guild_id"].as_str() {
            if !allowed_guilds.iter().any(|g| g == guild_id) {
                return None;
            }
        }
    }

    let content_text = d["content"].as_str().unwrap_or("");
    if content_text.is_empty() {
        return None;
    }

    let channel_id = d["channel_id"].as_str()?;
    let message_id = d["id"].as_str().unwrap_or("0");
    let username = author["username"].as_str().unwrap_or("Unknown");
    let discriminator = author["discriminator"].as_str().unwrap_or("0000");
    let display_name = if discriminator == "0" {
        username.to_string()
    } else {
        format!("{username}#{discriminator}")
    };

    let timestamp = d["timestamp"]
        .as_str()
        .and_then(|ts| chrono::DateTime::parse_from_rfc3339(ts).ok())
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .unwrap_or_else(chrono::Utc::now);

    // Parse commands (messages starting with /)
    let content = if content_text.starts_with('/') {
        let parts: Vec<&str> = content_text.splitn(2, ' ').collect();
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
        ChannelContent::Text(content_text.to_string())
    };

    // Determine if this is a group message (guild_id present = server channel)
    let is_group = d["guild_id"].as_str().is_some();

    // Check if bot was @mentioned (for MentionOnly policy enforcement)
    let was_mentioned = if let Some(ref bid) = *bot_user_id.read().await {
        // Check Discord mentions array
        let mentioned_in_array = d["mentions"]
            .as_array()
            .map(|arr| arr.iter().any(|m| m["id"].as_str() == Some(bid.as_str())))
            .unwrap_or(false);
        // Also check content for <@bot_id> or <@!bot_id> patterns
        let mentioned_in_content = content_text.contains(&format!("<@{bid}>"))
            || content_text.contains(&format!("<@!{bid}>"));
        mentioned_in_array || mentioned_in_content
    } else {
        false
    };

    let mut metadata = HashMap::new();
    if was_mentioned {
        metadata.insert("was_mentioned".to_string(), serde_json::json!(true));
    }

    Some(ChannelMessage {
        channel: ChannelType::Discord,
        platform_message_id: message_id.to_string(),
        sender: ChannelUser {
            platform_id: channel_id.to_string(),
            display_name,
            openfang_user: None,
        },
        content,
        target_agent: None,
        timestamp,
        is_group,
        thread_id: None,
        metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_parse_discord_message_basic() {
        let bot_id = Arc::new(RwLock::new(Some("bot123".to_string())));
        let d = serde_json::json!({
            "id": "msg1",
            "channel_id": "ch1",
            "content": "Hello agent!",
            "author": {
                "id": "user456",
                "username": "alice",
                "discriminator": "0",
                "bot": false
            },
            "timestamp": "2024-01-01T00:00:00+00:00"
        });

        let msg = parse_discord_message(&d, &bot_id, &[], &[], true)
            .await
            .unwrap();
        assert_eq!(msg.channel, ChannelType::Discord);
        assert_eq!(msg.sender.display_name, "alice");
        assert_eq!(msg.sender.platform_id, "ch1");
        assert!(matches!(msg.content, ChannelContent::Text(ref t) if t == "Hello agent!"));
    }

    #[tokio::test]
    async fn test_parse_discord_message_filters_bot() {
        let bot_id = Arc::new(RwLock::new(Some("bot123".to_string())));
        let d = serde_json::json!({
            "id": "msg1",
            "channel_id": "ch1",
            "content": "My own message",
            "author": {
                "id": "bot123",
                "username": "openfang",
                "discriminator": "0"
            },
            "timestamp": "2024-01-01T00:00:00+00:00"
        });

        let msg = parse_discord_message(&d, &bot_id, &[], &[], true).await;
        assert!(msg.is_none());
    }

    #[tokio::test]
    async fn test_parse_discord_message_filters_other_bots() {
        let bot_id = Arc::new(RwLock::new(Some("bot123".to_string())));
        let d = serde_json::json!({
            "id": "msg1",
            "channel_id": "ch1",
            "content": "Bot message",
            "author": {
                "id": "other_bot",
                "username": "somebot",
                "discriminator": "0",
                "bot": true
            },
            "timestamp": "2024-01-01T00:00:00+00:00"
        });

        let msg = parse_discord_message(&d, &bot_id, &[], &[], true).await;
        assert!(msg.is_none());
    }

    #[tokio::test]
    async fn test_parse_discord_ignore_bots_false_allows_other_bots() {
        let bot_id = Arc::new(RwLock::new(Some("bot123".to_string())));
        let d = serde_json::json!({
            "id": "msg1",
            "channel_id": "ch1",
            "content": "Bot message",
            "author": {
                "id": "other_bot",
                "username": "somebot",
                "discriminator": "0",
                "bot": true
            },
            "timestamp": "2024-01-01T00:00:00+00:00"
        });

        // With ignore_bots=false, other bots' messages should be allowed
        let msg = parse_discord_message(&d, &bot_id, &[], &[], false).await;
        assert!(msg.is_some());
        let msg = msg.unwrap();
        assert_eq!(msg.sender.display_name, "somebot");
        assert!(matches!(msg.content, ChannelContent::Text(ref t) if t == "Bot message"));
    }

    #[tokio::test]
    async fn test_parse_discord_ignore_bots_false_still_filters_self() {
        let bot_id = Arc::new(RwLock::new(Some("bot123".to_string())));
        let d = serde_json::json!({
            "id": "msg1",
            "channel_id": "ch1",
            "content": "My own message",
            "author": {
                "id": "bot123",
                "username": "openfang",
                "discriminator": "0",
                "bot": true
            },
            "timestamp": "2024-01-01T00:00:00+00:00"
        });

        // Even with ignore_bots=false, the bot's own messages must still be filtered
        let msg = parse_discord_message(&d, &bot_id, &[], &[], false).await;
        assert!(msg.is_none());
    }

    #[tokio::test]
    async fn test_parse_discord_message_guild_filter() {
        let bot_id = Arc::new(RwLock::new(Some("bot123".to_string())));
        let d = serde_json::json!({
            "id": "msg1",
            "channel_id": "ch1",
            "guild_id": "999",
            "content": "Hello",
            "author": {
                "id": "user1",
                "username": "bob",
                "discriminator": "0"
            },
            "timestamp": "2024-01-01T00:00:00+00:00"
        });

        // Not in allowed guilds
        let msg =
            parse_discord_message(&d, &bot_id, &["111".into(), "222".into()], &[], true).await;
        assert!(msg.is_none());

        // In allowed guilds
        let msg = parse_discord_message(&d, &bot_id, &["999".into()], &[], true).await;
        assert!(msg.is_some());
    }

    #[tokio::test]
    async fn test_parse_discord_command() {
        let bot_id = Arc::new(RwLock::new(None));
        let d = serde_json::json!({
            "id": "msg1",
            "channel_id": "ch1",
            "content": "/agent hello-world",
            "author": {
                "id": "user1",
                "username": "alice",
                "discriminator": "0"
            },
            "timestamp": "2024-01-01T00:00:00+00:00"
        });

        let msg = parse_discord_message(&d, &bot_id, &[], &[], true)
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
    async fn test_parse_discord_empty_content() {
        let bot_id = Arc::new(RwLock::new(None));
        let d = serde_json::json!({
            "id": "msg1",
            "channel_id": "ch1",
            "content": "",
            "author": {
                "id": "user1",
                "username": "alice",
                "discriminator": "0"
            },
            "timestamp": "2024-01-01T00:00:00+00:00"
        });

        let msg = parse_discord_message(&d, &bot_id, &[], &[], true).await;
        assert!(msg.is_none());
    }

    #[tokio::test]
    async fn test_parse_discord_discriminator() {
        let bot_id = Arc::new(RwLock::new(None));
        let d = serde_json::json!({
            "id": "msg1",
            "channel_id": "ch1",
            "content": "Hi",
            "author": {
                "id": "user1",
                "username": "alice",
                "discriminator": "1234"
            },
            "timestamp": "2024-01-01T00:00:00+00:00"
        });

        let msg = parse_discord_message(&d, &bot_id, &[], &[], true)
            .await
            .unwrap();
        assert_eq!(msg.sender.display_name, "alice#1234");
    }

    #[tokio::test]
    async fn test_parse_discord_message_update() {
        let bot_id = Arc::new(RwLock::new(Some("bot123".to_string())));
        let d = serde_json::json!({
            "id": "msg1",
            "channel_id": "ch1",
            "content": "Edited message content",
            "author": {
                "id": "user456",
                "username": "alice",
                "discriminator": "0",
                "bot": false
            },
            "timestamp": "2024-01-01T00:00:00+00:00",
            "edited_timestamp": "2024-01-01T00:01:00+00:00"
        });

        // MESSAGE_UPDATE uses the same parse function as MESSAGE_CREATE
        let msg = parse_discord_message(&d, &bot_id, &[], &[], true)
            .await
            .unwrap();
        assert_eq!(msg.channel, ChannelType::Discord);
        assert!(
            matches!(msg.content, ChannelContent::Text(ref t) if t == "Edited message content")
        );
    }

    #[tokio::test]
    async fn test_parse_discord_allowed_users_filter() {
        let bot_id = Arc::new(RwLock::new(Some("bot123".to_string())));
        let d = serde_json::json!({
            "id": "msg1",
            "channel_id": "ch1",
            "content": "Hello",
            "author": {
                "id": "user999",
                "username": "bob",
                "discriminator": "0"
            },
            "timestamp": "2024-01-01T00:00:00+00:00"
        });

        // Not in allowed users
        let msg = parse_discord_message(
            &d,
            &bot_id,
            &[],
            &["user111".into(), "user222".into()],
            true,
        )
        .await;
        assert!(msg.is_none());

        // In allowed users
        let msg = parse_discord_message(&d, &bot_id, &[], &["user999".into()], true).await;
        assert!(msg.is_some());

        // Empty allowed_users = allow all
        let msg = parse_discord_message(&d, &bot_id, &[], &[], true).await;
        assert!(msg.is_some());
    }

    #[tokio::test]
    async fn test_parse_discord_mention_detection() {
        let bot_id = Arc::new(RwLock::new(Some("bot123".to_string())));

        // Message with bot mentioned in mentions array
        let d = serde_json::json!({
            "id": "msg1",
            "channel_id": "ch1",
            "guild_id": "guild1",
            "content": "Hey <@bot123> help me",
            "mentions": [{"id": "bot123", "username": "openfang"}],
            "author": {
                "id": "user1",
                "username": "alice",
                "discriminator": "0"
            },
            "timestamp": "2024-01-01T00:00:00+00:00"
        });

        let msg = parse_discord_message(&d, &bot_id, &[], &[], true)
            .await
            .unwrap();
        assert!(msg.is_group);
        assert_eq!(
            msg.metadata.get("was_mentioned").and_then(|v| v.as_bool()),
            Some(true)
        );

        // Message without mention in group
        let d2 = serde_json::json!({
            "id": "msg2",
            "channel_id": "ch1",
            "guild_id": "guild1",
            "content": "Just chatting",
            "author": {
                "id": "user1",
                "username": "alice",
                "discriminator": "0"
            },
            "timestamp": "2024-01-01T00:00:00+00:00"
        });

        let msg2 = parse_discord_message(&d2, &bot_id, &[], &[], true)
            .await
            .unwrap();
        assert!(msg2.is_group);
        assert!(!msg2.metadata.contains_key("was_mentioned"));
    }

    #[tokio::test]
    async fn test_parse_discord_dm_not_group() {
        let bot_id = Arc::new(RwLock::new(None));
        let d = serde_json::json!({
            "id": "msg1",
            "channel_id": "dm-ch1",
            "content": "Hello",
            "author": {
                "id": "user1",
                "username": "alice",
                "discriminator": "0"
            },
            "timestamp": "2024-01-01T00:00:00+00:00"
        });

        let msg = parse_discord_message(&d, &bot_id, &[], &[], true)
            .await
            .unwrap();
        assert!(!msg.is_group);
    }

    #[test]
    fn test_build_heartbeat_payload_with_sequence() {
        let payload = build_heartbeat_payload(Some(42));
        assert_eq!(payload["op"], 1);
        assert_eq!(payload["d"], 42);
        // Round-trip through serde_json::to_string and re-parse to assert
        // valid JSON matching {"op":1,"d":42} regardless of key ordering.
        let s = serde_json::to_string(&payload).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed, serde_json::json!({"op": 1, "d": 42}));
    }

    #[test]
    fn test_build_heartbeat_payload_without_sequence() {
        let payload = build_heartbeat_payload(None);
        assert_eq!(payload["op"], 1);
        assert!(payload["d"].is_null());
        let s = serde_json::to_string(&payload).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(
            parsed,
            serde_json::json!({"op": 1, "d": serde_json::Value::Null})
        );
    }

    #[test]
    fn test_discord_adapter_creation() {
        let adapter = DiscordAdapter::new(
            "test-token".to_string(),
            vec!["123".to_string(), "456".to_string()],
            vec![],
            true,
            37376,
        );
        assert_eq!(adapter.name(), "discord");
        assert_eq!(adapter.channel_type(), ChannelType::Discord);
    }

    // --- Slash command / interaction tests ---

    #[test]
    fn test_is_valid_snowflake() {
        // Valid snowflakes (17-20 digits)
        assert!(is_valid_snowflake("12345678901234567")); // 17 digits
        assert!(is_valid_snowflake("123456789012345678")); // 18
        assert!(is_valid_snowflake("1234567890123456789")); // 19
        assert!(is_valid_snowflake("12345678901234567890")); // 20

        // Invalid
        assert!(!is_valid_snowflake("")); // empty
        assert!(!is_valid_snowflake("123456789012345")); // too short (15)
        assert!(!is_valid_snowflake("123456789012345678901")); // too long (21)
        assert!(!is_valid_snowflake("1234567890123456a")); // non-digit
        assert!(!is_valid_snowflake("abc12345678901234")); // letters
    }

    #[test]
    fn test_interaction_context_debug_redacts_token() {
        let ctx = InteractionContext {
            interaction_id: "12345678901234567".to_string(),
            interaction_token: "super-secret-token-value".to_string(),
            application_id: "99988877766655544".to_string(),
            channel_id: "11122233344455566".to_string(),
            created_at: Instant::now(),
        };
        let debug_output = format!("{ctx:?}");
        assert!(debug_output.contains("[REDACTED]"));
        assert!(!debug_output.contains("super-secret-token-value"));
        assert!(debug_output.contains("12345678901234567"));
    }

    #[test]
    fn test_option_value_to_string_variants() {
        // String value
        let opt = serde_json::json!({"value": "hello"});
        assert_eq!(option_value_to_string(&opt), Some("hello".to_string()));

        // Integer value
        let opt = serde_json::json!({"value": 42});
        assert_eq!(option_value_to_string(&opt), Some("42".to_string()));

        // Float value (use a non-PI-approximating literal so clippy::approx_constant
        // doesn't misread test data as a math constant).
        let opt = serde_json::json!({"value": 2.5});
        assert_eq!(option_value_to_string(&opt), Some("2.5".to_string()));

        // Boolean value
        let opt = serde_json::json!({"value": true});
        assert_eq!(option_value_to_string(&opt), Some("true".to_string()));

        // Null/missing value
        let opt = serde_json::json!({"name": "foo"});
        assert_eq!(option_value_to_string(&opt), None);
    }

    #[test]
    fn test_flatten_interaction_options_simple() {
        // No-arg command like /agents
        let options = serde_json::json!([]);
        let (name, args) = flatten_interaction_options("agents", &options);
        assert_eq!(name, "agents");
        assert!(args.is_empty());

        // Null options
        let (name, args) = flatten_interaction_options("help", &serde_json::Value::Null);
        assert_eq!(name, "help");
        assert!(args.is_empty());
    }

    #[test]
    fn test_flatten_interaction_options_with_args() {
        // Single required arg like /agent <name>
        let options = serde_json::json!([
            {"name": "name", "type": 3, "value": "assistant"}
        ]);
        let (name, args) = flatten_interaction_options("agent", &options);
        assert_eq!(name, "agent");
        assert_eq!(args, vec!["assistant"]);
    }

    #[test]
    fn test_flatten_interaction_options_subcommand() {
        // /schedule add agent1 "* * * * *" "hello"
        let options = serde_json::json!([{
            "name": "add",
            "type": 1,
            "options": [
                {"name": "agent", "type": 3, "value": "agent1"},
                {"name": "cron", "type": 3, "value": "* * * * *"},
                {"name": "message", "type": 3, "value": "hello"}
            ]
        }]);
        let (name, args) = flatten_interaction_options("schedule", &options);
        assert_eq!(name, "schedule");
        assert_eq!(args, vec!["add", "agent1", "* * * * *", "hello"]);
    }

    #[test]
    fn test_flatten_interaction_options_integer_value() {
        let options = serde_json::json!([
            {"name": "count", "type": 4, "value": 5}
        ]);
        let (name, args) = flatten_interaction_options("test", &options);
        assert_eq!(name, "test");
        assert_eq!(args, vec!["5"]);
    }

    #[test]
    fn test_build_command_definitions_count() {
        let defs = build_command_definitions();
        let spec_count = channel_command_specs().len();
        assert_eq!(defs.len(), spec_count);

        // Every definition must have a name, type, and description
        for def in &defs {
            assert!(def["name"].as_str().is_some(), "command missing name");
            assert_eq!(def["type"].as_u64(), Some(1), "command type must be CHAT_INPUT (1)");
            assert!(def["description"].as_str().is_some(), "command missing description");
            // Description must be <= 100 chars
            let desc = def["description"].as_str().unwrap();
            assert!(desc.len() <= 100, "description too long for {}: {} chars", def["name"], desc.len());
        }
    }

    #[test]
    fn test_build_command_definitions_subcommands() {
        let defs = build_command_definitions();

        // Find the "schedule" command — it should have subcommand options
        let schedule = defs.iter().find(|d| d["name"] == "schedule").unwrap();
        let opts = schedule["options"].as_array().unwrap();
        assert!(opts.len() >= 2, "schedule should have subcommand options");
        // Each option should be type 1 (SUB_COMMAND)
        for opt in opts {
            assert_eq!(opt["type"].as_u64(), Some(1));
        }
    }

    #[test]
    fn test_bot_auth_header_is_sensitive() {
        let header = bot_auth_header("test-token-123");
        assert!(header.is_sensitive());
        assert_eq!(header.to_str().unwrap(), "Bot test-token-123");
    }
}
