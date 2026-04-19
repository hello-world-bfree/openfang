# Cron Job `delivery` Schema

Every cron job has a `delivery` field that controls where the job's output goes
after it fires. Prior to this revision, the field existed but was undocumented
and had no CLI surface, leading to the 15-day silent-failure incident that
motivated this change.

## Schema

`delivery` is a `#[serde(tag = "kind")]` enum with four variants:

```jsonc
// Fire-and-forget. Job runs; output is discarded. Failures are silent.
{ "kind": "none" }

// Deliver to any configured channel adapter (Discord, Slack, Telegram,
// WhatsApp, email, Teams, Matrix, Signal, ...). `channel` is the adapter id;
// `to` is the recipient within that channel (user id, chat id, #room, etc.).
{ "kind": "channel", "channel": "discord", "to": "1234567890" }

// Deliver to whichever channel the agent most recently interacted on.
// Useful for personal assistants; risky for cron (see "Caveats" below).
{ "kind": "last_channel" }

// Deliver via HTTPS webhook POST. URL must start with https://
// (plain http:// is rejected for safety).
{ "kind": "webhook", "url": "https://example.com/hook" }
```

There is **no** separate `discord`, `slack`, `email`, or `log` kind — the
`channel` variant delegates to `ChannelAdapter`, which already implements every
supported medium. Adapters are configured in `config.toml` under
`[channels.discord]`, `[channels.slack]`, etc.; whatever you have enabled there
is selectable by name here.

## CLI

```bash
# Configure delivery on an existing job
openfang cron set-delivery <id> channel --channel discord --to <channel-id>
openfang cron set-delivery <id> webhook --url https://hooks.example.com/cron
openfang cron set-delivery <id> last-channel
openfang cron set-delivery <id> none

# Trigger a job immediately to verify delivery without waiting for the schedule
openfang cron trigger <id>
```

The flat `--kind=... --channel=... --to=...` shape was rejected in favor of a
subcommand-per-kind so that invalid combinations (e.g. `webhook --channel=...`)
are unrepresentable at parse time.

## API

```
PATCH /api/cron/jobs/{id}/delivery
Content-Type: application/json

{"kind":"channel","channel":"discord","to":"#ops"}
```

Body is the tagged `CronDelivery` JSON. Invalid variants, empty `to`/`channel`
strings, or non-https webhook URLs return `400 Bad Request`.

## Default

**New jobs default to `{"kind":"none"}`**. This is *not* a bug — it's a deliberate
choice because `last_channel` could deliver to a stale recipient (see below).
`openfang cron create` emits a `Warning: no delivery configured — failures will
be silent` message to stderr on creation to nudge users toward explicit
configuration.

## Caveats

- **`last_channel` on cron is risky**. The "last" channel reflects the agent's
  most recent interaction, which may have been days or weeks before the cron
  fires. Channel ids can be recycled, renamed, or reassigned to different
  owners. Prefer `channel` with an explicit recipient for cron output.
- **Webhook URLs go out from your machine**. There is no SSRF mitigation in v1
  beyond the `https://` scheme check — targeting `https://169.254.169.254/` or
  an internal service will succeed. Configure your firewall accordingly. Full
  connect-time RFC1918 / link-local / IMDS blocking is tracked for v1.1.
- **`cron_jobs.json` is authoritative at runtime**. Manual edits to the JSON
  file while the daemon is running will be clobbered on the next persist.
  Use `openfang cron set-delivery` (or the PATCH endpoint) instead.

## Verifying Delivery Works

After configuring delivery, don't wait for the schedule to fire:

```bash
openfang cron set-delivery <id> channel --channel discord --to <id>
openfang cron trigger <id>
# ... check the Discord channel; should see output within seconds
openfang cron list   # confirm DELIVERY column shows discord:<id>
```

## Deferred to v1.1

- `on: "error" | "always"` filter (deliver only on failure, vs every fire).
- Webhook HMAC signing / `secret_env` auth.
- Full SSRF mitigation (block RFC1918/link-local/IMDS at HTTP connect time via
  custom `reqwest` resolver).
- Per-delivery retry policy on transient webhook/channel failures.
