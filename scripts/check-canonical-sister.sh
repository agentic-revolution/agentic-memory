#!/usr/bin/env bash
set -euo pipefail

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

find_fixed() {
  local pattern="$1"
  shift
  if command -v rg >/dev/null 2>&1; then
    rg -nF "$pattern" "$@"
  else
    grep -R -n -F -- "$pattern" "$@"
  fi
}

assert_file() {
  [ -f "$1" ] || fail "Missing required file: $1"
}

assert_contains() {
  local pattern="$1"
  shift
  find_fixed "$pattern" "$@" >/dev/null || fail "Missing required pattern: ${pattern}"
}

assert_image_spacing() {
  local min_gap=10
  local prev=0
  local line
  while IFS= read -r line; do
    local n="${line%%:*}"
    if [ "$prev" -ne 0 ] && [ $((n - prev)) -lt "$min_gap" ]; then
      fail "README image blocks too close together (lines ${prev} and ${n})"
    fi
    prev="$n"
  done < <(grep -n '<img src="assets/' README.md || true)
}

assert_file "docs/internal/CANONICAL_SISTER_KIT.md"
assert_file "templates/sister-bootstrap/README.template.md"
assert_file "scripts/install.sh"
assert_file "scripts/check-install-commands.sh"

assert_contains '<img src="assets/github-hero-pane.svg"' README.md
assert_contains '<img src="assets/github-terminal-pane.svg"' README.md
assert_contains '## Install' README.md
assert_contains '## Quickstart' README.md
assert_contains '## How It Works' README.md
assert_image_spacing

assert_contains 'MCP client summary:' scripts/install.sh
assert_contains 'For Codex, Cursor, Windsurf, VS Code, Cline, or any MCP client, add:' scripts/install.sh
assert_contains 'Quick terminal check:' scripts/install.sh
assert_contains 'echo "  args: [\"serve\"]"' scripts/install.sh

echo "Canonical sister guardrails passed (memory)."
