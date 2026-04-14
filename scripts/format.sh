#!/usr/bin/env bash
# Format Emacs Lisp files in place with elisp-autofmt.
# Usage: format.sh [file ...]
set -euo pipefail

files=("$@")
if [ ${#files[@]} -eq 0 ]; then
  shopt -s nullglob
  files=(*.el)
  shopt -u nullglob
fi

for file in "${files[@]}"; do
  emacs --batch -L . -l elisp-autofmt \
    --eval "(progn
              (find-file (car command-line-args-left))
              (elisp-autofmt-buffer)
              (save-buffer))" \
    "$file" 2>/dev/null
  echo "Formatted: $file"
done
