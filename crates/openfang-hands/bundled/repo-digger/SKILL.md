# Repo Digger

**Multi-agent codebase investigator.** Takes a local repo + intent (explain | debug | plan) + question, emits a structured markdown artifact with file:line citations.

## Quick start

```bash
# Daemon with no API keys — relies on `claude` CLI being on PATH:
target/release/openfang start

# Fire a streaming investigation:
curl -N -X POST http://127.0.0.1:4200/api/repo-digger/run \
  -H 'Content-Type: application/json' \
  -d '{
    "repo_path": "/Users/me/dev/myproj",
    "intent": "plan",
    "question": "add Cohere LLM provider"
  }'
```

SSE events stream back:
```
{"phase":"cache_probe","run_id":"..."}
{"phase":"navigator","run_id":"..."}
{"phase":"searcher","run_id":"..."}
{"phase":"researcher","run_id":"..."}
{"phase":"synthesize","run_id":"..."}
{"phase":"citations","run_id":"..."}
{"event":"investigation_complete","artifact_path":"/Users/me/dev/myproj/.openfang/plans/plan-add-cohere-llm-provider-2026-04-20.md"}
```

Add `.openfang/` to your repo's `.gitignore`.

## Intents

- **explain** — architecture tour + key flow + extension points + glossary
- **debug** — reproduction + ranked hypotheses + root-cause evidence + fix sketch + verification
- **plan** — Claude-Code-compatible plan file: reuse-first pre-pass, files to create/modify, step-by-step, verification, tradeoffs

## Architecture

```
COORDINATOR (repo-digger, claude-code provider)
  │  workspace = repo_path (validated by validate_hand_workspace)
  │  --mcp-config → openfang-mcp-bridge subprocess → UDS → daemon
  │  --strict-mcp-config --disallowedTools 'Bash,Write,Read,…'
  │
  ├─ code_agent_spawn(role="navigator", seed)  → {agent_id}
  │    Navigator tools: file_list, file_read, code_search, memory_store, knowledge_add_entity
  │    Writes: inv:<run_id>:navigator {tree, languages, entrypoints, build_cmds, hotspots}
  │
  ├─ code_agent_spawn(role="searcher", seed)   → {agent_id}
  │    Searcher tools: code_search, file_read, memory_store, knowledge_add_relation
  │    Writes: inv:<run_id>:searcher:<qid> [hits]
  │
  ├─ code_agent_spawn(role="researcher", seed) → {agent_id}  (debug|plan only)
  │    Researcher tools: mcp_docs_mcp_*, web_search, web_fetch, memory_recall, memory_store
  │    Writes: inv:<run_id>:researcher
  │
  ├─ coordinator synthesizes artifact (no separate Reasoner — saves a hop)
  │
  └─ code_agent_spawn(role="checker")  — batched file_read citation verifier
```

Concurrency under `claude-code`: 1 sub-agent subprocess at a time (enforced by `Arc<AtomicU32>` in `code_agent_spawn`). Under direct-API: 3.

## Citations

Every factual claim ends with `[path:LN]` or `[path:LN-LN]`. Extension-agnostic — matches `Dockerfile:10`, `scripts/foo.sh:5-10`, `Cargo.lock:44`.

**Read denylist** (Citation Checker refuses to read):
- `.env`, `.envrc`, `.npmrc`, `.pypirc`, `.netrc`
- `.git/config`, `.git/credentials`
- `*.pem`, `*.key`, `*.p12`, `*.jks`
- `secrets.*`, `credentials.*`
- `.openfang/**` — self-investigation laundering

**Write denylist** (enforced by `workspace_sandbox::resolve_sandbox_path_for_write`):
- `.git/**` — blocks `.git/hooks/post-commit` RCE
- `.github/workflows/**`, `.gitlab-ci.yml`, `.circleci/**`
- `Makefile`, `Dockerfile`, `rakefile`, `BUILD`, `WORKSPACE`

Failed citations (≥2 repair passes): moved to "Unverified claims" section; artifact still ships.

## Budget

- `claude-code`: iteration-equivalent cap. `budget_cap=2.00` → ≤60 coord+sub iterations, ≤20min wall. Billing lives on your Claude subscription; openfang cannot meter it. UI warns.
- Direct-API: projected = Σ(iters × max_tokens × model_catalog price). Over cap → abort with visible "budget cap reached" section.

## docs-mcp integration

Optional but recommended for plan / debug intents. Ship disabled-by-default; user installs and pins a package version (see `crates/openfang-extensions/integrations/docs_mcp.toml`). Tools surface as `mcp_docs_mcp_search`, `mcp_docs_mcp_get_doc`, `mcp_docs_mcp_list_docs`.

When unhealthy or disabled: Researcher falls back to `web_search` + `web_fetch` and notes "external docs unavailable" in Methodology limitations.

## Degradation matrix (selected)

| Failure | Fallback |
|---|---|
| `claude` CLI not on PATH | Fallback provider chain; error if none configured |
| MCP bridge subprocess crashes | Kernel restarts once; second failure aborts |
| `rg` missing | Native walkdir+regex; PCRE2 patterns rejected with actionable error |
| code_search wall-clock timeout (30s) | Abort, retry narrower path |
| docs_mcp unhealthy | web_search fallback |
| Sub-agent hits max_iterations | Coordinator requeues once with narrower scope |
| Citation Checker fails 2x | Emit "Unverified" section |
| repo_path rejects validation | Activation fails with specific reason |
| $STATE_DIR overlaps repo_path | Activation fails (prevents cookie leak into git) |
| Repo > 50k files | Navigator returns "too_large"; asks user to narrow |

Every degradation lands in the artifact's "Methodology limitations" section — coordinator never silently fails.

## Security notes

- Claude Code built-in tools (`Bash`/`Read`/`Write`/`WebFetch`/`WebSearch`/`Glob`/`Grep`/`Task`) are disabled via `--disallowedTools` — without this flag they bypass the workspace sandbox.
- Per-agent MCP cookie + `agent_id` in every UDS RPC — one sub-agent can't impersonate another or the coordinator.
- `spawn_agent_checked` enforces capability subset — a child manifest can never widen the coordinator's tools.
- MCP config JSON written `O_CREAT|O_EXCL` with `0600` before any write is visible — TOCTOU-safe.
- $STATE_DIR overlap rejection prevents cookies from landing in a git-tracked dir.
- Sensitive dot-directories under $HOME (`.ssh`, `.aws`, `.claude`, …) refused as `repo_path`.

## Artifacts

Default output dir: `<repo_path>/.openfang/plans/<intent>-<slug>-YYYY-MM-DD.md`. Override via `plan_output_dir` setting.

Cache: `~/.openfang/cache/repo-digger/<key>.artifact.md` — TTL 30 days or until `git_rev` changes.
