#!/usr/bin/env bash
# check-openfang-specialist-facts.sh
# Detects drift between facts baked into ~/.claude/agents/openfang-specialist.md
# and the live repo. Single-line summary suitable for a statusline or pre-commit hook.
#
# Update baselines at each 90-day re-audit. Bumping a baseline = acknowledging the
# agent body was refreshed to match.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Baselines captured 2026-04-19
EXPECTED_PLATFORM="${EXPECTED_PLATFORM:-Darwin}"
EXPECTED_CRATE_COUNT="${EXPECTED_CRATE_COUNT:-13}"
EXPECTED_ROUTE_COUNT="${EXPECTED_ROUTE_COUNT:-171}"
EXPECTED_PEER_REGISTRY_TYPE="${EXPECTED_PEER_REGISTRY_TYPE:-OnceLock}"
EXPECTED_RESPONSE_FIELD="${EXPECTED_RESPONSE_FIELD:-pub response:}"

drifts=()

platform="$(uname -s)"
[ "$platform" = "$EXPECTED_PLATFORM" ] || drifts+=("platform:$platform")

crate_count="$(ls -d crates/*/ 2>/dev/null | wc -l | tr -d ' ')"
[ "$crate_count" = "$EXPECTED_CRATE_COUNT" ] || drifts+=("crates:$crate_count")

route_count="$(grep -c '\.route(' crates/openfang-api/src/server.rs 2>/dev/null || echo 0)"
[ "$route_count" = "$EXPECTED_ROUTE_COUNT" ] || drifts+=("routes:$route_count")

grep -q "$EXPECTED_PEER_REGISTRY_TYPE.*PeerRegistry" crates/openfang-kernel/src/kernel.rs 2>/dev/null \
    || drifts+=("peer_registry_type:drift")

grep -q "$EXPECTED_RESPONSE_FIELD" crates/openfang-runtime/src/agent_loop.rs 2>/dev/null \
    || drifts+=("AgentLoopResult.response:drift")

if [ ${#drifts[@]} -eq 0 ]; then
    echo "openfang-specialist: fresh (routes=$route_count, crates=$crate_count, platform=$platform)"
    exit 0
else
    echo "openfang-specialist: DRIFT [${drifts[*]}] — re-audit agent body"
    exit 1
fi
