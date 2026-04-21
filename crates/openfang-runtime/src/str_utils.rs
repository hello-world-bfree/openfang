//! UTF-8-safe string utilities.

/// Truncate a string to at most `max_bytes` bytes without splitting a multi-byte
/// character.  Returns the full string when it already fits.
///
/// This avoids panics that occur when using `&s[..max_bytes]` on strings containing
/// multi-byte characters (e.g. Chinese, emoji, accented Latin).
#[inline]
pub fn safe_truncate_str(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    // Walk backwards to the nearest char boundary
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Escape control characters and Unicode line-separators so an LLM-authored
/// tool argument can't inject fake log lines when emitted through structured
/// logging. Replaces `\n`, `\r`, `\t`, DEL, and non-printable bytes with
/// `<LF>`, `<CR>`, etc. Pass the result to `tracing::info!(.. = %sanitized, ..)`.
///
/// OWASP A09 defense-in-depth — the production tracing subscriber should ALSO
/// use JSON format (`tracing_subscriber::fmt::format::Json`), which naturally
/// escapes newlines. This function is the second line.
#[inline]
pub fn sanitize_for_log(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\n' => out.push_str("<LF>"),
            '\r' => out.push_str("<CR>"),
            '\t' => out.push_str("<TAB>"),
            '\x7f' => out.push_str("<DEL>"),
            // Unicode RTL override — reserved for intentional display spoofing.
            '\u{202E}' => out.push_str("<RTL>"),
            c if c.is_control() => {
                // Other C0 controls (\x00-\x1F except the ones above).
                out.push_str(&format!("<U+{:04X}>", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_within_limit() {
        let s = "hello";
        assert_eq!(safe_truncate_str(s, 10), "hello");
    }

    #[test]
    fn ascii_exact_limit() {
        let s = "hello";
        assert_eq!(safe_truncate_str(s, 5), "hello");
    }

    #[test]
    fn ascii_truncated() {
        let s = "hello world";
        assert_eq!(safe_truncate_str(s, 5), "hello");
    }

    #[test]
    fn multibyte_chinese() {
        // Each Chinese character is 3 bytes in UTF-8
        let s = "\u{4f60}\u{597d}\u{4e16}\u{754c}"; // "hello world" in Chinese, 12 bytes
                                                    // Truncating at 7 bytes should not split the 3rd char (bytes 6..9)
        let t = safe_truncate_str(s, 7);
        assert_eq!(t, "\u{4f60}\u{597d}"); // 6 bytes, 2 chars
        assert!(t.len() <= 7);
    }

    #[test]
    fn multibyte_emoji() {
        let s = "\u{1f600}\u{1f601}\u{1f602}"; // 3 emoji, 4 bytes each = 12 bytes
        let t = safe_truncate_str(s, 5);
        assert_eq!(t, "\u{1f600}"); // 4 bytes, 1 emoji
    }

    #[test]
    fn zero_limit() {
        let s = "hello";
        assert_eq!(safe_truncate_str(s, 0), "");
    }

    #[test]
    fn empty_string() {
        assert_eq!(safe_truncate_str("", 10), "");
    }

    #[test]
    fn sanitize_for_log_escapes_newline() {
        // The attack: LLM-crafted tool arg injects fake log lines after a newline.
        let input = "hello\n[CRITICAL] authenticated as root\n";
        let out = sanitize_for_log(input);
        assert!(!out.contains('\n'));
        assert!(out.contains("<LF>"));
    }

    #[test]
    fn sanitize_for_log_escapes_carriage_return() {
        let out = sanitize_for_log("line1\rline2");
        assert!(!out.contains('\r'));
        assert!(out.contains("<CR>"));
    }

    #[test]
    fn sanitize_for_log_escapes_tab() {
        let out = sanitize_for_log("a\tb");
        assert!(out.contains("<TAB>"));
    }

    #[test]
    fn sanitize_for_log_escapes_rtl_override() {
        let out = sanitize_for_log("user\u{202E}evil.exe");
        assert!(out.contains("<RTL>"));
        assert!(!out.contains('\u{202E}'));
    }

    #[test]
    fn sanitize_for_log_escapes_other_control_chars() {
        let out = sanitize_for_log("a\x00b\x07c");
        assert!(out.contains("<U+0000>"));
        assert!(out.contains("<U+0007>"));
    }

    #[test]
    fn sanitize_for_log_passes_through_ordinary_text() {
        let input = "regular tool argument: /path/to/file.rs line 42";
        assert_eq!(sanitize_for_log(input), input);
    }

    #[test]
    fn sanitize_for_log_passes_through_multibyte_unicode() {
        let input = "héllo \u{4f60}\u{597d} 😀";
        assert_eq!(sanitize_for_log(input), input);
    }
}
