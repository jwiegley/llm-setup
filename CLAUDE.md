# CLAUDE.md — llm-setup.el

## Package Overview

`llm-setup.el` is a single-file LLM model management system for Emacs. It
maintains deployed model facts in `llm-setup-models-list`, provider and
exceptional Nix-only static-model facts in `llm-setup-nix-provider-defs`, and
exact default and Claude selections in explicit Custom variables. It owns three
outputs:

```
llm-setup-models-list (Elisp structs)
    │
    ├─► /Users/johnw/Models/llama-swap.yaml (hera, clio)
    │     └─ Model-switching proxy on port 8080
    │
    ├─► config/ai/model-registry.json (Nix source)
    │     └─ Schema-v2 nonsecret model facts and four exact selections
    │
    └─► gptel backends (Emacs)
          └─ In-editor LLM interaction
```

The package does not generate or deploy configuration for downstream model
gateways or manage their services.

**Infrastructure topology:**

- **hera** (primary) — runs most GGUF/MLX models via llama-swap
- **clio** (secondary) — runs a subset of models via llama-swap

## Development Commands

No traditional build system (no Makefile, Eask, or Cask). Nix provides the
development environment and repository checks.

**ERT:**

```bash
nix develop -c emacs --batch -L . --eval '(setq load-prefer-newer t)' \
  -l llm-setup-test.el -f ert-run-tests-batch-and-exit
```

**Byte-compile:**

```bash
emacs -batch -L . -f batch-byte-compile llm-setup.el
```

**Validate configuration** (checks installed GGUF models plus enum, hostname,
and path fields):

```elisp
(llm-setup-check-instances)
```

**Full deployment** (validate → rebuild hera/clio llama-swap YAML → stop each
llama-swap process for service-manager restart → publish Nix registry → update
gptel):

```elisp
(llm-setup-reset)
```

**Interactive development** — after modifying `llm-setup.el`:

```elisp
(unload-feature 'llm-setup t)
(load-file "llm-setup.el")
```

The deployed path (`~/.emacs.d/lisp/llm-setup`) is the same physical directory
as the source (via Nix home-manager symlinks), so changes take effect immediately
after `eval-buffer` or reload.

## Architecture

### Data Model

Two `cl-defstruct` types form the registry:

- **`llm-setup-model`** — Family-level metadata, sampling parameters, and a list
  of instances.
- **`llm-setup-instance`** — Deployment-level provider, engine, hostnames, model
  paths, llama.cpp cache settings, arguments, and concurrency limits.

The deployed-model registry lives in `llm-setup-models-list`. Downstream
generation iterates it via `llm-setup-instances-list`, which flattens it into
`(model . instance)` cons pairs. The Nix projection combines those instances
with `llm-setup-nix-provider-defs` and the explicit default and Claude selection
variables. It emits the ordered `default`, `claudeDefault`, `claudeHaiku`, and
`claudeSubagent` selections in schema version 2.

The Nix projection describes direct providers only; downstream gateways discover
llama-swap models independently.

### Naming System

| Accessor | Returns | Used For |
|---|---|---|
| `llm-setup-model-name` | Family symbol | Internal registry key |
| `llm-setup-instance-name` | Public instance symbol, or nil | llama-swap, Nix, and GPTel model key |
| `llm-setup-short-model-name` | Directory name without organization/GGUF suffixes | Installed-model matching |

### llama-swap Generation

`llm-setup-generate-llama-swap-yaml` generates per-host YAML. It filters
instances by hostname and local provider eligibility, emits engine-specific CLI
commands with `${PORT}` placeholders, and appends groups and preload hooks for
the models actually emitted.

`llm-setup-build-llama-swap-yaml` writes
`/Users/johnw/Models/llama-swap.yaml` locally for hera or through TRAMP for
clio, then stops the corresponding llama-swap process so its service manager
can restart it.

### `llm-setup-reset` Orchestration (5 steps)

1. `llm-setup-check-instances` — validate registry; abort on any warning
2. `llm-setup-build-llama-swap-yaml` — write hera YAML and stop llama-swap locally
3. `llm-setup-build-llama-swap-yaml "clio"` — write clio YAML via TRAMP and stop it via SSH
4. `llm-setup-build-nix-model-registry` — atomically publish the nonsecret JSON registry
5. Set `gptel-model` and `gptel-backend` via `gptel-backends-llama-swap`

## Adding a New Model

1. Download: `M-x llm-setup-download` (or `llm-setup-checkout` for git-lfs)
2. Optionally inspect: `M-x llm-setup-show` to view GGUF metadata
3. Add a `make-llm-setup-model` + `make-llm-setup-instance` entry
4. Run `M-x llm-setup-reset` to validate, deploy, and publish

## Critical Constraints

### External Dependencies Not Defined Here

- `gptel-backends-llama-swap` — direct llama-swap gptel integration called in
  reset step 5
- `yaml-mode`, `json-mode` — used for display buffers but never `require`d

### TRAMP Patterns

Remote model operations use `/ssh:hostname:` paths constructed by
`llm-setup-remote-path`. Clio uses the same `/Users/johnw/Models` pathname as
hera. Remote `executable-find` works by temporarily setting `default-directory`
to the remote host.

### Allowed Enum Values

All valid provider, engine, kind, characteristic, and capability values are
defined in the corresponding `llm-setup-all-*` constants and validated by
`llm-setup-check-instances`.
