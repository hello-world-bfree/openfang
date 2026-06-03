//! Startup reaper for orphaned browser child processes.
//!
//! The native CDP browser ([`openfang_runtime::browser`]) launches one
//! Chromium process per agent and kills it on session drop. But a hard kill
//! or crash of the daemon strands those children: they reparent to init
//! (`ppid 1`) and survive, each holding a recycling pool of outbound sockets.
//! Enough orphans across crash-restart cycles can exhaust the host's
//! ephemeral-port range and break unrelated services.
//!
//! This reaper runs once at daemon startup, before any new browser is
//! launched, and kills orphaned Chromium children left by prior runs. It
//! matches on the exact launch signature openfang uses
//! (`--remote-debugging-port=0` together with a headless flag) so it never
//! touches a user's own interactive browser.
//!
//! Linux-only: it reads `/proc`. A no-op on other platforms — the browser
//! session `Drop` still covers the graceful path everywhere.

#[cfg(target_os = "linux")]
use tracing::{info, warn};

/// Reap orphaned Chromium children left by a dead daemon.
///
/// Returns the number of processes killed. Best-effort: scan/parse/kill
/// failures are logged and skipped, never propagated.
#[cfg(target_os = "linux")]
pub fn reap_orphan_browsers() -> usize {
    let mut killed = 0usize;
    let entries = match std::fs::read_dir("/proc") {
        Ok(e) => e,
        Err(e) => {
            warn!(error = %e, "Browser orphan reap: cannot read /proc; skipping");
            return 0;
        }
    };

    for entry in entries.flatten() {
        let fname = entry.file_name();
        let name = match fname.to_str() {
            Some(n) => n,
            None => continue,
        };
        // Only numeric entries are PIDs.
        let pid: i32 = match name.parse() {
            Ok(p) => p,
            Err(_) => continue,
        };

        let proc_dir = entry.path();
        if !is_orphaned_openfang_chromium(&proc_dir) {
            continue;
        }

        // SAFETY: kill(2) with SIGTERM; pid validated as a live /proc entry
        // matching our exact launch signature.
        let rc = unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
        if rc == 0 {
            killed += 1;
            info!(pid, "Reaped orphaned Chromium child from a prior daemon run");
        } else {
            let err = std::io::Error::last_os_error();
            // ESRCH (already gone) is fine; anything else is worth noting.
            if err.raw_os_error() != Some(libc::ESRCH) {
                warn!(pid, error = %err, "Browser orphan reap: SIGTERM failed");
            }
        }
    }

    killed
}

/// True if `/proc/<pid>` is an orphaned (`ppid == 1`) Chromium launched with
/// openfang's CDP signature.
#[cfg(target_os = "linux")]
fn is_orphaned_openfang_chromium(proc_dir: &std::path::Path) -> bool {
    // cmdline is NUL-separated argv.
    let cmdline = match std::fs::read(proc_dir.join("cmdline")) {
        Ok(b) => b,
        Err(_) => return false,
    };
    if cmdline.is_empty() {
        return false;
    }
    let args: Vec<&[u8]> = cmdline.split(|&b| b == 0).collect();

    // Signature: ephemeral debug port + headless, exactly as launch() builds it.
    // Both must be present so we never kill a user's interactive Chrome.
    let has_debug_port = args
        .iter()
        .any(|a| a == b"--remote-debugging-port=0");
    let has_headless = args
        .iter()
        .any(|a| a.starts_with(b"--headless"));
    if !(has_debug_port && has_headless) {
        return false;
    }

    // Orphaned == reparented to init. Field 4 of /proc/<pid>/stat is ppid, but
    // the comm field (2) can contain spaces/parens, so split on the last ')'.
    let stat = match std::fs::read_to_string(proc_dir.join("stat")) {
        Ok(s) => s,
        Err(_) => return false,
    };
    let rest = match stat.rsplit_once(')') {
        Some((_, r)) => r,
        None => return false,
    };
    // After ")": " <state> <ppid> ..." → ppid is the 2nd whitespace field.
    let ppid = rest.split_whitespace().nth(1).and_then(|s| s.parse::<i32>().ok());
    ppid == Some(1)
}

/// No-op on non-Linux: the browser session `Drop` covers the graceful path,
/// and there is no `/proc` to scan for orphans.
#[cfg(not(target_os = "linux"))]
pub fn reap_orphan_browsers() -> usize {
    0
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::is_orphaned_openfang_chromium;
    use std::path::Path;

    /// Build a fake `/proc/<pid>` dir with the given NUL-joined argv and a
    /// `stat` line whose ppid is `ppid`.
    fn fake_proc(dir: &Path, argv: &[&str], ppid: i32) {
        std::fs::create_dir_all(dir).unwrap();
        let cmdline: Vec<u8> = argv
            .iter()
            .flat_map(|a| a.bytes().chain(std::iter::once(0u8)))
            .collect();
        std::fs::write(dir.join("cmdline"), cmdline).unwrap();
        // comm deliberately contains a ')' to prove the rsplit_once parse is
        // robust; ppid is the 2nd field after the final ')'.
        let stat = format!("1234 (chrome (foo)) S {ppid} 1234 1234 0 -1 0");
        std::fs::write(dir.join("stat"), stat).unwrap();
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("ofreaper-{name}"));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn orphaned_openfang_chromium_is_reaped() {
        // The exact signature launch() emits, reparented to init.
        let d = tmp("orphan");
        fake_proc(
            &d,
            &["chrome", "--headless=new", "--remote-debugging-port=0"],
            1,
        );
        assert!(is_orphaned_openfang_chromium(&d));
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn live_daemon_child_is_left_alone() {
        // Same signature but still parented to a live daemon (ppid != 1) —
        // killing it would murder an in-use session on a running daemon.
        let d = tmp("live");
        fake_proc(
            &d,
            &["chrome", "--headless=new", "--remote-debugging-port=0"],
            4242,
        );
        assert!(!is_orphaned_openfang_chromium(&d));
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn user_interactive_chrome_is_never_touched() {
        // A real browser the user is using: orphaned-looking (ppid 1, e.g.
        // launched from a terminal that exited) but NOT headless and no debug
        // port. Must not match, or we'd kill the user's browser.
        let d = tmp("user");
        fake_proc(&d, &["chrome", "https://example.com"], 1);
        assert!(!is_orphaned_openfang_chromium(&d));
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn headless_without_debug_port_is_not_ours() {
        // Some other tool's headless Chrome with a fixed debug port — not our
        // ephemeral-port signature, so we leave it.
        let d = tmp("other-headless");
        fake_proc(
            &d,
            &["chrome", "--headless", "--remote-debugging-port=9222"],
            1,
        );
        assert!(!is_orphaned_openfang_chromium(&d));
        std::fs::remove_dir_all(&d).unwrap();
    }
}

// ---------------------------------------------------------------------------
// Orphaned MCP-grandchild reaper (ssh tunnels + lightpanda)
// ---------------------------------------------------------------------------
//
// openfang launches external MCP servers as child processes; those servers in
// turn spawn their own children — the `dbx` server opens `ssh -N -L …:5432`
// tunnels, the lightpanda server runs `lightpanda serve` CDP instances. When a
// daemon is hard-killed before group-teardown runs, those grandchildren
// reparent to init/launchd (`ppid 1`) and survive, each holding sockets that
// accumulate toward ephemeral-port exhaustion.
//
// Unlike the Chromium reaper above this is `ps`-based, so it works on both
// Linux and macOS (the platform where the orphans were observed). It matches
// each process's full command line against an exact signature plus `ppid == 1`,
// so it never touches an interactive ssh session or a live daemon's children.

/// Reap orphaned MCP grandchildren (ssh `-N -L` tunnels, `lightpanda serve`)
/// left by a dead daemon. Returns the number signalled. Best-effort: any
/// failure to scan/parse/signal is logged and skipped, never propagated.
#[cfg(unix)]
pub fn reap_orphan_mcp_children() -> usize {
    let output = match std::process::Command::new("ps")
        .args(["-axo", "pid=,ppid=,command="])
        .output()
    {
        Ok(o) if o.status.success() => o,
        Ok(o) => {
            tracing::warn!(status = ?o.status, "MCP orphan reap: ps exited non-zero; skipping");
            return 0;
        }
        Err(e) => {
            tracing::warn!(error = %e, "MCP orphan reap: cannot run ps; skipping");
            return 0;
        }
    };

    let listing = String::from_utf8_lossy(&output.stdout);
    let mut killed = 0usize;
    for line in listing.lines() {
        let Some((pid, ppid, cmd)) = parse_ps_line(line) else {
            continue;
        };
        if ppid != 1 || !is_orphan_mcp_grandchild(cmd) {
            continue;
        }

        // SAFETY: kill(2) SIGTERM to a pid parsed from ps whose command matches
        // our exact orphan signature and whose ppid is 1.
        let rc = unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) };
        if rc == 0 {
            killed += 1;
            tracing::info!(pid, cmd, "Reaped orphaned MCP child from a prior daemon run");
        } else {
            let err = std::io::Error::last_os_error();
            if err.raw_os_error() != Some(libc::ESRCH) {
                tracing::warn!(pid, error = %err, "MCP orphan reap: SIGTERM failed");
            }
        }
    }

    killed
}

/// No-op on non-Unix: no `ps`/`kill` and no orphan-to-init reparenting.
#[cfg(not(unix))]
pub fn reap_orphan_mcp_children() -> usize {
    0
}

/// Parse one `ps -axo pid=,ppid=,command=` line into `(pid, ppid, command)`.
/// `command` is everything after the first two whitespace-delimited fields, so
/// spaces inside the command line are preserved.
#[cfg_attr(not(unix), allow(dead_code))]
fn parse_ps_line(line: &str) -> Option<(i32, i32, &str)> {
    let line = line.trim_start();
    let after_pid = line.split_once(char::is_whitespace)?;
    let pid: i32 = after_pid.0.parse().ok()?;
    let rest = after_pid.1.trim_start();
    let after_ppid = rest.split_once(char::is_whitespace)?;
    let ppid: i32 = after_ppid.0.parse().ok()?;
    let cmd = after_ppid.1.trim();
    if cmd.is_empty() {
        return None;
    }
    Some((pid, ppid, cmd))
}

/// True if `cmd` is an MCP-grandchild signature we own: a Postgres ssh tunnel
/// (`ssh … -N … -L …:5432 …`) or a lightpanda CDP server (`lightpanda … serve`).
/// Matches are deliberately narrow so unrelated ssh sessions or browsers are
/// never reaped.
#[cfg_attr(not(unix), allow(dead_code))]
fn is_orphan_mcp_grandchild(cmd: &str) -> bool {
    let is_ssh_tunnel = cmd.contains("ssh ")
        && cmd.contains(" -N")
        && cmd.contains(" -L")
        && cmd.contains(":5432");
    let is_lightpanda = cmd.contains("lightpanda") && cmd.contains("serve");
    is_ssh_tunnel || is_lightpanda
}

#[cfg(test)]
mod mcp_reaper_tests {
    use super::{is_orphan_mcp_grandchild, parse_ps_line};

    #[test]
    fn parses_pid_ppid_and_spaced_command() {
        let line = "  421     1 ssh -p22 -N -L 62835:library:5432 root@library";
        let (pid, ppid, cmd) = parse_ps_line(line).expect("parse");
        assert_eq!(pid, 421);
        assert_eq!(ppid, 1);
        assert_eq!(cmd, "ssh -p22 -N -L 62835:library:5432 root@library");
    }

    #[test]
    fn matches_postgres_ssh_tunnel() {
        assert!(is_orphan_mcp_grandchild(
            "ssh -p22 -o StrictHostKeyChecking=no -4 -N -L 62835:library:5432 root@library"
        ));
    }

    #[test]
    fn matches_lightpanda_serve() {
        assert!(is_orphan_mcp_grandchild(
            "/Users/x/.cache/lightpanda-node/lightpanda serve --host 127.0.0.1 --port 0"
        ));
    }

    #[test]
    fn ignores_interactive_ssh() {
        // A normal login shell over ssh: no -N, no -L tunnel, no :5432.
        assert!(!is_orphan_mcp_grandchild("ssh root@library"));
    }

    #[test]
    fn ignores_non_postgres_tunnel() {
        // An ssh tunnel to some other service must not be reaped.
        assert!(!is_orphan_mcp_grandchild("ssh -N -L 8080:internal:80 host"));
    }

    #[test]
    fn ignores_unrelated_lightpanda_invocation() {
        // lightpanda doing something other than `serve` (e.g. a one-shot fetch)
        // is not a long-lived CDP server.
        assert!(!is_orphan_mcp_grandchild(
            "lightpanda fetch https://example.com"
        ));
    }
}
