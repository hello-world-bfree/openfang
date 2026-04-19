# Cron Job Reentrancy / In-Flight Lock

## Summary

Cron scheduler has no documented policy for what happens when a scheduled fire arrives while the previous run is still executing. No `max_in_flight` field in job schema, no advisory-lock behavior observable from CLI/config.

## Impact

Scenario (example — could happen under current config):
- `library-curation-weekly` with `timeout_secs: 1200` and schedule `0 2 * * 0` (weekly Sunday 02:00).
- A catastrophic run exceeds 1200s and doesn't get killed (observed: kernel's timeout enforcement may not always terminate a stuck MCP tool call).
- Next week's fire arrives while prior still in-flight.
- Two concurrent library-curation agents: both hit DB (race on UPDATE books SET), both POST to Open Library (double rate-limit consumption), both write report to memory (overwrite each other), both consume API tokens.

Same pattern could hit `doc-curation-weekly` + any future cron that invokes MCP tool calls with their own timeouts that don't respect the cron wrapper's budget.

## Current State

- `cron_jobs.json` schema contains `enabled`, `schedule`, `action`, `delivery` — no `max_in_flight` or `overlap_policy` field.
- `openfang cron create --help` has no overlap-related flag.
- Scheduler implementation unknown (binary closed); boot log says `Cron scheduler active with N job(s)` but no concurrency semantics surfaced.

## Proposed Fix

Add to cron job schema:

```json
{
  "action": { ... },
  "overlap_policy": "skip" | "queue" | "kill-and-run" | "allow",
  "max_in_flight": 1
}
```

Default: `overlap_policy = "skip"`, `max_in_flight = 1`. Behavior:

- **skip**: next fire is logged, recorded in delivery channel, does NOT execute. Agent_turn action skipped.
- **queue**: next fire held until current completes. Bounded queue depth = 3; further fires drop to "skip" + warn.
- **kill-and-run**: terminate current run, start new. Only for idempotent jobs — dangerous default.
- **allow**: current behavior. Explicit opt-in to concurrent runs.

Implementation: per-agent pidfile or in-memory advisory lock in `openfang_kernel::cron`. Lock taken before `agent_turn` dispatch, released on completion or timeout.

## Workaround

None at config level. User option: shorten `timeout_secs` aggressively so jobs cannot overlap by design (requires decomposing work). Example: `timeout_secs: 1200` on a weekly schedule means overlap impossible by the 7-day margin, but the policy is not explicit and a bug could change that.

## Priority

Medium. Overlap window for weekly crons is practically zero (20-min job, 7-day schedule). Becomes high-priority when chaining cron-driven workflows or scheduling hourly/per-minute jobs. Documenting expected behavior (even if "no guarantee") is higher priority than implementing the policy.
