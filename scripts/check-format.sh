#!/usr/bin/env bash
# Check that Emacs Lisp files are properly formatted with elisp-autofmt.
# Usage: check-format.sh [file ...]
set -uo pipefail

EXIT=0
TMPS=()
cleanup() { rm -f "${TMPS[@]}"; }
trap cleanup EXIT

for file in "$@"; do
  tmp=$(mktemp "${TMPDIR:-/tmp}/elfmt-XXXXXX.el")
  TMPS+=("$tmp")
  cp "$file" "$tmp"
  emacs --batch -L . -l elisp-autofmt \
    --eval "(progn
              (find-file (car command-line-args-left))
              (elisp-autofmt-buffer)
              (save-buffer))" \
    "$tmp" 2>/dev/null
  if ! diff -q "$file" "$tmp" >/dev/null 2>&1; then
    echo "Format check failed: $file"
    diff -u "$file" "$tmp" | head -40 || true
    EXIT=1
  fi
done
exit "$EXIT"
