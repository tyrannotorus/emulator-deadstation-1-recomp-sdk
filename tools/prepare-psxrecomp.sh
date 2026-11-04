#!/usr/bin/env bash
# Bring the psxrecomp submodule to its audited state: verify the pin, keep its agent instructions from
# auto-loading, and apply the local patches (patches/psxrecomp). Idempotent.
# The submodule is never updated from the network here; its nested submodules are never fetched.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
sub="$root/psxrecomp"
pin="a6deb9ca06cbbc90e2caa1475ac465cbc99ea28c"

head="$(git -C "$sub" rev-parse HEAD)"
[ "$head" = "$pin" ] || { echo "psxrecomp is at $head, expected the audited pin $pin" >&2; exit 1; }

# Upstream CLAUDE.md is an agent's instructions from another project: renamed, never followed.
if [ -e "$sub/CLAUDE.md" ]; then
  mv "$sub/CLAUDE.md" "$sub/UPSTREAM-AGENT-NOTES.untrusted.md"
fi

for p in "$root"/patches/psxrecomp/*.patch; do
  if git -C "$sub" apply --reverse --check "$p" 2>/dev/null; then
    echo "already applied: $(basename "$p")"
  else
    git -C "$sub" apply "$p"
    echo "applied: $(basename "$p")"
  fi
done
