# Security Policy

## Supported Versions

Only the latest release on `main` receives security fixes.

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for security vulnerabilities.

Report them privately by emailing the maintainer or using [GitHub's private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability).

Include:
- A description of the vulnerability
- Steps to reproduce
- Potential impact

You will receive a response within 72 hours. If confirmed, a fix will be released as soon as possible.

## Scope

This is a portfolio/demo project. The main surfaces to consider are:

- **Flask app** — runs on localhost by default, not intended for public deployment without authentication
- **LLM prompt injection** — the nudge prompt includes user session data; do not deploy with untrusted external input without sanitisation
- **Dependencies** — report any known CVEs in the dependency tree
