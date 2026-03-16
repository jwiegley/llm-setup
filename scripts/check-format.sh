#!/usr/bin/env bash
# Check that Emacs Lisp files are properly formatted with elisp-autofmt.
# Usage: check-format.sh [file ...]
set -euo pipefail

EXIT=0
for file in "$@"; do
  tmp=$(mktemp "${TMPDIR:-/tmp}/elfmt-XXXXXX.el")
  cp "$file" "$tmp"
  emacs --batch -L . -l elisp-autofmt \
    --eval "(progn
              (find-file \"$tmp\")
              (elisp-autofmt-buffer)
              (save-buffer))" 2>/dev/null
  if ! diff -q "$file" "$tmp" >/dev/null 2>&1; then
    echo "Format check failed: $file"
    diff -u "$file" "$tmp" | head -40 || true
    EXIT=1
  fi
  rm -f "$tmp"
done
exit $EXIT
