#!/usr/bin/env bash
# Format Emacs Lisp files in place with elisp-autofmt.
# Usage: format.sh [file ...]
set -euo pipefail

files=("$@")
if [ ${#files[@]} -eq 0 ]; then
  files=(*.el)
fi

for file in "${files[@]}"; do
  emacs --batch -L . -l elisp-autofmt \
    --eval "(progn
              (find-file \"$file\")
              (elisp-autofmt-buffer)
              (save-buffer))" 2>/dev/null
  echo "Formatted: $file"
done
