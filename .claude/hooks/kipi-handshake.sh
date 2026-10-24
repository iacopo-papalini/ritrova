#!/usr/bin/env bash
set -euo pipefail

# Hooks run with arbitrary CWD; anchor to project root via Claude Code's env var.
: "${CLAUDE_PROJECT_DIR:?CLAUDE_PROJECT_DIR not set — this script must run inside a Claude Code hook}"
STATE_DIR="${CLAUDE_PROJECT_DIR}/.claude/state"
DIRTY_FLAG="${STATE_DIR}/kipi-dirty"
mkdir -p "${STATE_DIR}"

case "${1:-}" in
  mark)
    touch "${DIRTY_FLAG}"
    ;;
  check)
    if [ -f "${DIRTY_FLAG}" ]; then
      rm -f "${DIRTY_FLAG}"
      # A Stop hook at exit 0 discards stderr and only debug-logs stdout — a bare
      # `echo` reaches neither the user nor Claude. The user-facing channel is the
      # `systemMessage` JSON field, so emit that. It renders emoji but strips ANSI
      # colour, hence the emoji-and-unicode styling instead of escape codes.
      printf '{"systemMessage": "%s", "suppressOutput": true}\n' \
        "✨🧠 KIPI · knowledge base may be stale 🧠✨\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n✏️  Files changed this session\n🔄  /kipi update  → refresh existing chapters\n🔍  /kipi scan <path>  → cover new areas\n📊  /kipi status  → coverage report"
    fi
    ;;
  *)
    echo "Usage: $0 {mark|check}" >&2
    exit 1
    ;;
esac
