//! Workspace filesystem sandboxing.
//!
//! Confines agent file operations to their workspace directory.
//! Prevents path traversal, symlink escapes, and access outside the sandbox.

use std::path::{Path, PathBuf};

/// Repo-root-relative paths that are never allowed as WRITE targets, even
/// though the sandbox permits them for read.
///
/// Rationale: when an investigation Hand's workspace is a user's code
/// repository, a prompt-injected README could cause the LLM to write a
/// `.git/hooks/post-commit` that achieves persistent arbitrary code
/// execution on the user's next `git commit`. Similarly, CI config
/// (`.github/workflows/*.yml`, `.gitlab-ci.yml`, `.circleci/**`) and
/// build drivers (`Makefile`, `Dockerfile`, `rakefile`, `BUILD`,
/// `WORKSPACE`) grant code execution at build or CI time. Reads are still
/// permitted — they're needed to explain / debug / plan — but writes are not.
///
/// The check uses repo-root-relative path components, so sub-directories
/// (e.g. `submodule/.git/hooks/x`) are also caught.
const WRITE_DENY_PREFIXES: &[&str] = &[
    ".git/",
    ".github/workflows/",
    ".circleci/",
    ".drone/",
    ".buildkite/",
];

const WRITE_DENY_FILES: &[&str] = &[
    ".gitlab-ci.yml",
    "Makefile",
    "makefile",
    "GNUmakefile",
    "Dockerfile",
    "dockerfile",
    "rakefile",
    "Rakefile",
    "BUILD",
    "BUILD.bazel",
    "WORKSPACE",
    "WORKSPACE.bazel",
];

/// Returns `Some(reason)` if `path_in_workspace` (a workspace-relative
/// path, already sandbox-verified) matches the write denylist.
///
/// `path_in_workspace` is the path AFTER sandbox resolution, expressed
/// relative to the workspace root (with forward-slash separators on all
/// platforms).
fn matches_write_denylist(path_in_workspace: &str) -> Option<&'static str> {
    let norm = path_in_workspace.replace('\\', "/");
    for prefix in WRITE_DENY_PREFIXES {
        if norm.starts_with(prefix) || norm.contains(&format!("/{prefix}")) {
            return Some(prefix);
        }
    }
    // Exact-filename match on the basename (case-sensitive on POSIX, case-
    // insensitive checks via dedicated entries above for lowercase/capitalized
    // variants).
    if let Some(basename) = std::path::Path::new(&norm).file_name().and_then(|s| s.to_str()) {
        for deny in WRITE_DENY_FILES {
            if basename == *deny {
                return Some(deny);
            }
        }
    }
    None
}

/// Resolve a user-supplied path for a WRITE operation.
///
/// Performs all checks of [`resolve_sandbox_path`], then rejects writes that
/// target CI/CD config, `.git/` internals, or build driver files (see
/// [`WRITE_DENY_PREFIXES`] / [`WRITE_DENY_FILES`]).
pub fn resolve_sandbox_path_for_write(
    user_path: &str,
    workspace_root: &Path,
) -> Result<PathBuf, String> {
    let canon = resolve_sandbox_path(user_path, workspace_root)?;
    // Compute workspace-relative form for denylist check.
    let canon_root = workspace_root
        .canonicalize()
        .map_err(|e| format!("Failed to resolve workspace root: {e}"))?;
    let rel = canon
        .strip_prefix(&canon_root)
        .map_err(|_| "path resolves outside workspace".to_string())?
        .to_string_lossy()
        .to_string();
    if let Some(matched) = matches_write_denylist(&rel) {
        return Err(format!(
            "Write denied: '{user_path}' matches write denylist '{matched}'. \
             Writes to CI config, .git/, and build drivers are forbidden to prevent \
             persistent code execution via prompt-injected content.",
        ));
    }
    Ok(canon)
}

/// Resolve a user-supplied path within a workspace sandbox.
///
/// - Rejects `..` components outright.
/// - Relative paths are joined with `workspace_root`.
/// - Absolute paths are checked against the workspace root after canonicalization.
/// - For new files: canonicalizes the parent directory and appends the filename.
/// - The final canonical path must start with the canonical workspace root.
pub fn resolve_sandbox_path(user_path: &str, workspace_root: &Path) -> Result<PathBuf, String> {
    let path = Path::new(user_path);

    // Reject any `..` components
    for component in path.components() {
        if matches!(component, std::path::Component::ParentDir) {
            return Err("Path traversal denied: '..' components are forbidden".to_string());
        }
    }

    // Build the candidate path
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace_root.join(path)
    };

    // Canonicalize the workspace root
    let canon_root = workspace_root
        .canonicalize()
        .map_err(|e| format!("Failed to resolve workspace root: {e}"))?;

    // Canonicalize the candidate (or its parent for new files)
    let canon_candidate = if candidate.exists() {
        candidate
            .canonicalize()
            .map_err(|e| format!("Failed to resolve path: {e}"))?
    } else {
        // For new files: canonicalize the parent and append the filename
        let parent = candidate
            .parent()
            .ok_or_else(|| "Invalid path: no parent directory".to_string())?;
        let filename = candidate
            .file_name()
            .ok_or_else(|| "Invalid path: no filename".to_string())?;
        let canon_parent = parent
            .canonicalize()
            .map_err(|e| format!("Failed to resolve parent directory: {e}"))?;
        // Defense-in-depth: fail early if the parent itself escapes the
        // workspace via a symlink (even though the join+prefix check below
        // would still catch it, failing here produces a clearer error).
        if !canon_parent.starts_with(&canon_root) {
            return Err(format!(
                "Access denied: parent directory of '{user_path}' resolves outside workspace",
            ));
        }
        canon_parent.join(filename)
    };

    // Verify the canonical path is inside the workspace
    if !canon_candidate.starts_with(&canon_root) {
        return Err(format!(
            "Access denied: path '{}' resolves outside workspace. \
             If you have an MCP filesystem server configured, use the \
             mcp_filesystem_* tools (e.g. mcp_filesystem_read_file, \
             mcp_filesystem_list_directory) to access files outside \
             the workspace.",
            user_path
        ));
    }

    Ok(canon_candidate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_relative_path_inside_workspace() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();
        std::fs::write(data_dir.join("test.txt"), "hello").unwrap();

        let result = resolve_sandbox_path("data/test.txt", dir.path());
        assert!(result.is_ok());
        let resolved = result.unwrap();
        assert!(resolved.starts_with(dir.path().canonicalize().unwrap()));
    }

    #[test]
    fn test_absolute_path_inside_workspace() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("file.txt"), "ok").unwrap();
        let abs_path = dir.path().join("file.txt");

        let result = resolve_sandbox_path(abs_path.to_str().unwrap(), dir.path());
        assert!(result.is_ok());
    }

    #[test]
    fn test_absolute_path_outside_workspace_blocked() {
        let dir = TempDir::new().unwrap();
        let outside = std::env::temp_dir().join("outside_test.txt");
        std::fs::write(&outside, "nope").unwrap();

        let result = resolve_sandbox_path(outside.to_str().unwrap(), dir.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Access denied"));

        let _ = std::fs::remove_file(&outside);
    }

    #[test]
    fn test_dotdot_component_blocked() {
        let dir = TempDir::new().unwrap();
        let result = resolve_sandbox_path("../../../etc/passwd", dir.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Path traversal denied"));
    }

    #[test]
    fn test_nonexistent_file_with_valid_parent() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        let result = resolve_sandbox_path("data/new_file.txt", dir.path());
        assert!(result.is_ok());
        let resolved = result.unwrap();
        assert!(resolved.starts_with(dir.path().canonicalize().unwrap()));
        assert!(resolved.ends_with("new_file.txt"));
    }

    #[cfg(unix)]
    #[test]
    fn test_symlink_escape_blocked() {
        let dir = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        std::fs::write(outside.path().join("secret.txt"), "secret").unwrap();

        // Create a symlink inside the workspace pointing outside
        let link_path = dir.path().join("escape");
        std::os::unix::fs::symlink(outside.path(), &link_path).unwrap();

        let result = resolve_sandbox_path("escape/secret.txt", dir.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Access denied"));
    }

    #[cfg(unix)]
    #[test]
    fn test_nonexistent_file_through_symlinked_parent_blocked() {
        // Regression: the else-branch (non-existent file) must reject a
        // parent directory that is itself a symlink escaping the workspace.
        let dir = TempDir::new().unwrap();
        let outside = TempDir::new().unwrap();
        // Don't create the file — just the parent symlink.
        let link_path = dir.path().join("escape");
        std::os::unix::fs::symlink(outside.path(), &link_path).unwrap();

        let result = resolve_sandbox_path("escape/brand_new_file.txt", dir.path());
        assert!(result.is_err(), "parent symlink escape must be blocked for writes too");
        let err = result.unwrap_err();
        assert!(
            err.contains("Access denied") || err.contains("outside workspace"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_write_denylist_blocks_git_hooks() {
        let dir = TempDir::new().unwrap();
        // Simulate an existing .git directory so the parent exists for a new file.
        std::fs::create_dir_all(dir.path().join(".git/hooks")).unwrap();

        let result = resolve_sandbox_path_for_write(".git/hooks/post-commit", dir.path());
        assert!(result.is_err(), "writing .git/hooks/* must be denied");
        assert!(result.unwrap_err().contains("Write denied"));
    }

    #[test]
    fn test_write_denylist_blocks_github_workflows() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join(".github/workflows")).unwrap();

        let result = resolve_sandbox_path_for_write(".github/workflows/deploy.yml", dir.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Write denied"));
    }

    #[test]
    fn test_write_denylist_blocks_makefile_and_dockerfile() {
        let dir = TempDir::new().unwrap();
        // The parent (workspace root itself) must exist.
        for name in ["Makefile", "Dockerfile", "makefile", "rakefile"] {
            let result = resolve_sandbox_path_for_write(name, dir.path());
            assert!(
                result.is_err(),
                "writing top-level {name} must be denied; got {result:?}"
            );
        }
    }

    #[test]
    fn test_write_denylist_blocks_gitlab_circleci_configs() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join(".circleci")).unwrap();
        let r1 = resolve_sandbox_path_for_write(".gitlab-ci.yml", dir.path());
        assert!(r1.is_err(), ".gitlab-ci.yml write must be denied");
        let r2 = resolve_sandbox_path_for_write(".circleci/config.yml", dir.path());
        assert!(r2.is_err(), ".circleci/config.yml write must be denied");
    }

    #[test]
    fn test_write_denylist_permits_ordinary_files() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::create_dir_all(dir.path().join(".openfang/plans")).unwrap();

        // Artifact output path — must NOT be in the denylist.
        let result = resolve_sandbox_path_for_write(".openfang/plans/plan-foo-2026-04-20.md", dir.path());
        assert!(result.is_ok(), ".openfang/plans writes should be permitted: {result:?}");

        let result = resolve_sandbox_path_for_write("src/main.rs", dir.path());
        assert!(result.is_ok(), "src/*.rs writes should be permitted");
    }

    #[test]
    fn test_write_denylist_still_returns_canonical_path() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        let result = resolve_sandbox_path_for_write("src/a.rs", dir.path()).unwrap();
        assert!(result.ends_with("a.rs"));
    }

    #[test]
    fn test_read_path_still_permits_git_config() {
        // READ must still be allowed for .git/config so the Citation Checker
        // can verify citations that reference it (the read-denylist is enforced
        // separately, in the Citation Checker logic — NOT in the path sandbox).
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join(".git/config"), "[core]\n").unwrap();
        let result = resolve_sandbox_path(".git/config", dir.path());
        assert!(result.is_ok(), "READ of .git/config must be permitted at the path level");
    }
}
