# Extraction verifier

## Source documentation

For new or changed owned source code, document actual behavior and non-obvious
rationale in docstrings and inline comments, including scientific/data contracts;
verify claims against the implementation, and preserve native documentation
conventions. Avoid boilerplate and unrelated, upstream, or generated
documentation rewrites.

## Agent Communication

- Discover or contact Claude Code only when the user explicitly requests Claude
  communication in the current task. Shared files, environments and running
  processes do not authorize contact.
- Automatic coordination was rejected because shared use does not establish
  overlapping edits and unsolicited messages interrupt other sessions.
- For authorized Claude communication, use the configured `claude-code` MCP
  server with only `ListAgents` and `SendMessage`. List first, then send only the
  requested message to the exact listed recipient.
- Re-read shared files before editing and preserve other agents' changes.

## Node runtime for development

- Use a patched host Node for Pyright. The `pyright[nodejs]` extra was rejected
  because its wheel currently installs an older Node and can undo runtime
  security updates during environment synchronization.
