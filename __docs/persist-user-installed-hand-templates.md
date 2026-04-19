# Persist User-Installed Hand Templates Across Daemon Restarts

## Summary

`openfang hand install <path>` registers a hand template into the running kernel's in-memory registry. After `openfang stop && openfang start`, the template is gone — only bundled hands are re-seeded. Any active instance of a user hand dies with no recovery path other than re-running `install` then `activate`.

## Reproduction

```
# Install user hand
openfang hand install ~/.openfang/hands/coder/
# → Installed hand: Coder Hand (coder)

openfang hand activate coder
# → Hand 'coder' activated (instance: <uuid>)

openfang hand list | grep coder
# → coder         Coder Hand           productivity ...

# Restart daemon
openfang stop && openfang start

openfang hand list | grep coder
# → (no output — template gone)

openfang hand activate coder
# → Failed to activate hand 'coder': Agent not found: Hand not found: coder
```

Bundled hands (`researcher`, `browser`, `clip`, `collector`, `infisical-sync`, `lead`, `predictor`, `trader`, `twitter`) survive restart — registry rebuilt from binary-embedded manifests. User-installed hands (authored at `~/.openfang/hands/<id>/HAND.toml`) are not re-scanned on boot.

## Current State (observed 2026-04-19)

- Boot log `openfang_kernel::kernel: Loaded 9 bundled hand(s)` — fixed count; no line scanning `~/.openfang/hands/`.
- No `hands_dir` field in `config.toml`.
- `hand_state.json` persists *active instances* (which hand is running with which agent), but not *template registry*.
- `researcher-hand` previously-active survives because its template is bundled; user hands die because template is not re-registered before `hand_state.json` replay.

## Impact

- Any user relying on a custom hand must re-run `openfang hand install <path>` after every restart. Silent workflow breakage — restart can happen via OS reboot, `openfang stop`, crash recovery, or update-triggered reload.
- Crons that depend on a user-hand's agent will fail with "Agent not found" until the user manually re-installs.
- Users writing agent templates for personal use hit this on day one; diverges from "install once, forget" UX of every other agent framework.

## Proposed Fix

On kernel boot, after bundled hand registry load but before `hand_state.json` replay:

1. Scan `~/.openfang/hands/*/HAND.toml` (configurable via `[paths] hands_dir`).
2. Parse each as already done in `hand install`.
3. Merge into registry. Skip if ID collides with bundled (or warn + override based on policy).
4. Only then replay `hand_state.json` to restore active instances.

Equivalent to the `~/.openfang/skills/` scan that runs on boot (log line: `Loaded N skills from /Users/poirot/.openfang/skills`).

## Workaround (until fixed)

Shell alias or wrapper script:

```bash
# ~/.zshrc or ~/.bashrc
openfang-start() {
  openfang start
  sleep 2
  for hand_dir in ~/.openfang/hands/*/; do
    [ -f "$hand_dir/HAND.toml" ] || continue
    hand_id="$(basename "$hand_dir")"
    openfang hand install "$hand_dir" 2>/dev/null
    openfang hand activate "$hand_id" 2>/dev/null
  done
}
```

## Related

- `hand_state.json` replay logic in `openfang_kernel::kernel::restore_hands` (boot log: `Restoring N persisted hand(s)`).
- Skill registry scan at `openfang_skills::registry::load_user_skills` — reference for hand equivalent.
- `config.toml` currently has no `[paths]` section; adding one for `hands_dir`, `agents_dir` overrides would help users with non-standard layouts.

## Priority

High for any user with custom hands. Low for users only using bundled hands.
