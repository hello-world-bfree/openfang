# Workflow Inter-Step Approval Gates

## Summary

Workflow JSON schema supports `error_mode: "fail" | "skip"` per step but has no mechanism to pause between steps for human approval. This forces workflows to be either fully automatic (security risk when upstream steps consume external content) or fully manual (no orchestration value).

## Impact

Direct security risk in this user's `code-review-loop` workflow:
- Step 1 `coder` has `web_fetch` + `gh_mcp` capabilities → can fetch adversarial content from issues, READMEs, or arbitrary URLs.
- Step 2 `debugger` has `shell_exec` → runs build/test commands.
- Step 3 `code-reviewer` has `shell_exec` (via coder references).
- Without an approval gate, a prompt injection in fetched content could flow coder → debugger shell → arbitrary code execution on the user's machine.

Same shape in `simulation-analysis`: researcher (web_fetch) → analyst → coder (shell). Injection from an arXiv-ish URL could reach shell via 2 hops.

Plan mitigation recommended `require_approval: true` between any `web_fetch`/`gh_mcp`-consuming step and any `shell_exec` step. Not implementable in current JSON schema.

## Current Schema

```json
{
  "name": "step-name",
  "agent": { "name": "agent-name" },
  "prompt_template": "...",
  "mode": "sequential",
  "timeout_secs": 300,
  "error_mode": "fail" | "skip",
  "output_var": "..."
}
```

No `approval_required`, `human_review`, or `pause_for` field.

## Proposed Fix

Add `approval_required: bool` (default false) to the step schema:

```json
{
  "name": "triage",
  "agent": { "name": "debugger" },
  "prompt_template": "...",
  "mode": "sequential",
  "timeout_secs": 400,
  "error_mode": "fail",
  "output_var": "triage",
  "approval_required": true,
  "approval_prompt": "Debugger will shell_exec based on coder's implementation. Review before proceeding?"
}
```

When kernel reaches a step with `approval_required: true`:
1. Pause workflow execution, persist intermediate state.
2. Surface approval request via `openfang approvals list` (already-present CLI verb).
3. `openfang approvals approve <workflow-run-id>` resumes with the step.
4. `openfang approvals reject <workflow-run-id>` halts with status "rejected-by-user".

`openfang approvals` CLI already exists per `openfang --help`. Wiring it into workflow scheduler should be a small integration.

## Alternative: Capability-based gating

Instead of per-step flag, kernel could auto-gate: any workflow step whose input `output_var` came from a step with `gh_mcp` or `web_fetch` tool usage → auto-require approval before a shell_exec-capable step runs.

More conservative, no workflow author mistakes. Less flexibility for fully-trusted pipelines.

## Priority

High for any workflow chaining external-content agents with shell-exec agents. Reported by debate as CRIT before severity-demoted to HIGH based on audit-trail detection surface — audit visibility doesn't prevent the attack, only records it.
