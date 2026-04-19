# Anthropic Prompt Caching Support

## Summary

OpenFang does not expose Anthropic's `cache_control: {type: "ephemeral"}` on system prompt blocks. Long autonomous loops (default `max_iterations = 200`) re-send the full system prompt every turn, billing base input rate on tokens that should hit the 0.1× cache-read rate.

## Impact

- `doc-curator` system prompt: ~3,200 tokens. 200 iterations × 3,200 = 640,000 redundant tokens per cron run.
- `library-curator` system prompt: ~2,800 tokens. 200 × 2,800 = 560,000 redundant per run.
- At Sonnet 4.6 input rate ($3/M): ~$1.80 wasted per curation run, per cron, per week.
- Within-session cache (5-min TTL): first turn 1.25× write cost, turns 2–200 0.1× read → ~85% savings on system-prompt tokens.
- Cross-run cache not relevant (weekly schedule > 5-min TTL).

## Current State

- Binary strings probe: no `cache_control`, `anthropic-cache`, or `ephemeral` tokens found.
- No `cache_control` field accepted in agent TOML `[model]` or `[[fallback_models]]` blocks.
- Anthropic SDK call site in `openfang_runtime::drivers::anthropic` (inferred) does not pass cache directives.

## Proposed Fix

1. Add `cache_system_prompt: bool` (default true for system prompts ≥ 1024 tokens — Sonnet minimum cacheable block).
2. When sending to Anthropic API, wrap system prompt block with `cache_control: {type: "ephemeral"}`.
3. Expose in agent TOML as:
   ```toml
   [model]
   provider = "anthropic"
   model = "claude-sonnet-4-6"
   cache_system_prompt = true  # default true for blocks ≥1024 tokens
   ```
4. For multi-block system prompts (rare but possible), cache only the stable prefix.

## Reference

- https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
- Minimum cacheable block: 1,024 tokens (Sonnet), 2,048 (Haiku).
- TTL: 5 min (default) or 1 hour (beta, `cache_control: {type: "ephemeral", ttl: "1h"}`).
- Works for both native Anthropic SDK and Bedrock/Vertex routes.

## Priority

High for cron-driven agents and workflows that loop. Low for interactive single-turn agents.

## Verification Post-Fix

Track `cache_creation_input_tokens` and `cache_read_input_tokens` in the `usage` block of each Anthropic API response. Expected after implementation:
- Turn 1: `cache_creation_input_tokens ≈ system_prompt_tokens`
- Turns 2+: `cache_read_input_tokens ≈ system_prompt_tokens`
- Sum of regular `input_tokens` drops by system-prompt-size × (iterations - 1).
