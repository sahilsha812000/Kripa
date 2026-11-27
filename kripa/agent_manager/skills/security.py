"""Conservative checks for local, non-executable skill packages.

Secret-detection limitation
----------------------------
The ``secret_findings`` function applies a fixed set of regular-expression
patterns that target common credential formats (OpenAI-style ``sk-`` keys,
Google ``AIza`` keys, Groq ``gsk_`` keys, OpenRouter ``sk-or-v1-`` keys,
PEM private-key headers, and ``api_key``/``password``/``bearer`` label-value
pairs).

**These patterns are intentionally conservative, not comprehensive.**  They
are designed to catch accidental inclusion of well-known credential shapes,
not to serve as a general data-loss-prevention (DLP) system.  Arbitrary
sensitive text that does not match any pattern will pass through without
detection.  Callers that need broader protection must apply additional
application-level controls; they must not rely on this function alone as a
guarantee that no secret is ever stored or transmitted.
"""

import re


MAX_SKILL_FILE_BYTES = 256 * 1024
EXPECTED_FILES = frozenset({"skill.json", "SKILL.md"})

_SECRET_PATTERNS = (
    re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\b\s*[:=]\s*['\"]?[^\s'\"]{6,}"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|sk-or-v1-[A-Fa-f0-9]{20,}|gsk_[A-Za-z0-9]{16,}|AIza[A-Za-z0-9_-]{30,})\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def secret_findings(text):
    findings = []
    if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
        findings.append("Possible credential material detected in skill text.")
    return tuple(findings)


def validate_package_directory(path):
    """Validate package shape and read only the two fixed, non-executable files."""
    import os
    import stat

    if not os.path.isdir(path) or os.path.islink(path):
        raise ValueError("Skill package must be a real local directory, not a symlink.")
    try:
        entries = list(os.scandir(path))
    except OSError:
        raise ValueError("Skill package directory could not be read.")
    names = set()
    for entry in entries:
        if entry.name not in EXPECTED_FILES:
            raise ValueError("Skill package contains unsupported files; only skill.json and SKILL.md are allowed.")
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise ValueError("Skill package files must be regular files without symlinks.")
        mode = entry.stat(follow_symlinks=False).st_mode
        if mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
            raise ValueError("Executable skill files are not allowed.")
        if entry.stat(follow_symlinks=False).st_size > MAX_SKILL_FILE_BYTES:
            raise ValueError("Skill package file exceeds the size limit.")
        names.add(entry.name)
    if names != EXPECTED_FILES:
        raise ValueError("Skill package must contain both skill.json and SKILL.md.")
    root = os.path.realpath(path)
    contents = {}
    for name in sorted(EXPECTED_FILES):
        candidate = os.path.join(path, name)
        if os.path.commonpath((root, os.path.realpath(candidate))) != root:
            raise ValueError("Skill package contains an unsafe file path.")
        try:
            with open(candidate, "rb") as handle:
                raw = handle.read(MAX_SKILL_FILE_BYTES + 1)
        except OSError:
            raise ValueError("Skill package file could not be read.")
        if len(raw) > MAX_SKILL_FILE_BYTES:
            raise ValueError("Skill package file exceeds the size limit.")
        try:
            contents[name] = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("Skill package files must be UTF-8 text.")
    return contents
