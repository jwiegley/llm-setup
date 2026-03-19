# CLAUDE.md — llm-setup.el

## Package Overview

`llm-setup.el` is a single-file LLM model management system for Emacs. It maintains a **model registry** (`llm-setup-models-list`) as the single source of truth and generates deployment configurations for a multi-host infrastructure:

```
llm-setup-models-list (Elisp structs)
    │
    ├─► llama-swap.yaml (per-host: hera, clio)
    │     └─ Model-switching proxy on port 8080
    │
    ├─► litellm/config.yaml (global: vulcan)
    │     └─ Unified OpenAI-compatible proxy aggregating all local + cloud providers
    │
    └─► gptel backends (Emacs)
          └─ In-editor LLM interaction via LiteLLM
```

**Infrastructure topology:**
- **hera** (primary) — runs most GGUF/MLX models via llama-swap
- **clio** (secondary) — runs a subset of models via llama-swap
- **vulcan** (remote) — runs LiteLLM as a systemd service, config deployed via multi-hop TRAMP (`/ssh:vulcan|sudo:root@vulcan:/etc/litellm/config.yaml`)

## Development Commands

No build system (no Makefile, Eask, or Cask). No test suite.

**Byte-compile:**
```bash
emacs -batch -L . -f batch-byte-compile llm-setup.el
```

**Validate configuration** (checks installed models match registry, validates all fields):
```elisp
(llm-setup-check-instances)
```

**Full deployment** (validate → rebuild all YAMLs → restart services → update gptel):
```elisp
(llm-setup-reset)
```

**Interactive development** — after modifying `llm-setup.el`:
```elisp
(unload-feature 'llm-setup t)
(load-file "llm-setup.el")
```

The deployed path (`~/.emacs.d/lisp/llm-setup`) is the same physical directory as the source (via Nix home-manager symlinks), so changes take effect immediately after `eval-buffer` or reload.

## Architecture

### Data Model

Two `cl-defstruct` types form the registry:

- **`llm-setup-model`** — Family-level: name, description, characteristics (`high`/`medium`/`low`/`remote`/`local`/`thinking`/`instruct`/`coding`/`rewrite`), capabilities (`media`/`tool`/`json`/`url`), kind (`text-generation`/`embedding`/`reranker`), sampling parameters, and a list of instances.
- **`llm-setup-instance`** — Deployment-level: provider, engine, hostnames, model-path, file-path, draft-model, cache settings, fallbacks. Each instance belongs to exactly one `llm-setup-model`.

The registry lives in `llm-setup-models-list` (a large `defcustom`). All downstream generation iterates this via `llm-setup-instances-list`, which flattens it into `(model . instance)` cons pairs.

### Naming System

Each model has multiple names used in different contexts (documented in comments at line ~380):

| Accessor | Returns | Used For |
|---|---|---|
| `llm-setup-model-name` | Symbol like `Qwen3.5-27B` | Internal registry key |
| `llm-setup-instance-name` | Symbol like `mlx-community/Qwen3.5-27B-4bit`, or nil → falls back to model name | llama-swap model key, LiteLLM model name |
| `llm-setup-instance-model-name` | Override for provider-facing name (e.g. Claude vibe-proxy) | Rarely used; currently bound but unused in LiteLLM generation |
| `llm-setup-get-full-litellm-name` | `"host/name"` or `"provider/name"` | LiteLLM entries, fallback resolution |
| `llm-setup-short-model-name` | Strips org prefix and GGUF suffix from directory name | Matching installed models to registry |

### YAML Generation Pipeline

**llama-swap** (`llm-setup-generate-llama-swap-yaml`): Generates per-host. Iterates all instances, filters by hostname membership, emits engine-specific CLI commands with `${PORT}` placeholder. The prolog/epilog (`llm-setup-llama-swap-prolog`/`llm-setup-llama-swap-epilog`) wrap the generated model entries.

**LiteLLM** (`llm-setup-generate-litellm-yaml`): Generates globally. All instances are included — local instances get one entry per hostname, remote instances get one entry per provider. Credentials come from `llm-setup-litellm-credentials`, API keys from `llm-setup-litellm-environment-function` (calls `lookup-password`). Router fallbacks are dynamically generated from instance `fallbacks` fields.

### `llm-setup-reset` Orchestration (5 steps)

1. `llm-setup-check-instances` — validate registry; abort on any warning
2. `llm-setup-build-llama-swap-yaml` — write YAML for hera, kill llama-swap locally
3. `llm-setup-build-llama-swap-yaml "clio"` — write YAML for clio via TRAMP, kill via SSH
4. `llm-setup-build-litellm-yaml` — write config to vulcan via TRAMP, restart systemd service
5. Set `gptel-model` and `gptel-backend` via `gptel-backends-make-litellm` (defined externally)

## Adding a New Model

1. Download: `M-x llm-setup-download` (or `llm-setup-checkout` for git-lfs)
2. Optionally inspect: `M-x llm-setup-show` to view GGUF metadata
3. Add a `make-llm-setup-model` + `make-llm-setup-instance` entry to `llm-setup-models-list`
4. **If the model belongs to a llama-swap group** (always_on, large_models, embeddings, rerankings, stt), update `llm-setup-llama-swap-epilog` — it contains hardcoded model names
5. Run `M-x llm-setup-reset` to validate and deploy
6. Use `llm-setup-generate-instance-declarations` to scaffold declarations from `~/Models`

## Critical Constraints

### Manual Synchronization Required
`llm-setup-llama-swap-epilog` contains hardcoded model names in its `groups:` section. When models are added/removed from `llm-setup-models-list`, the epilog must be updated separately or llama-swap grouping will be wrong.

### External Dependencies Not Defined Here
- `lookup-password` — used to fetch API keys from auth-source; defined elsewhere in the Emacs config
- `gptel-backends-make-litellm` — called in `llm-setup-reset` step 5; defined in gptel configuration
- `yaml-mode`, `json-mode` — used for display buffers but never `require`'d

### TRAMP Patterns
Remote operations use `/ssh:hostname:` prefix for file paths (constructed by `llm-setup-remote-path`). LiteLLM config uses multi-hop: `/ssh:vulcan|sudo:root@vulcan:`. Remote `executable-find` works by temporarily setting `default-directory` to the remote host.

### Allowed Enum Values
All valid values for provider, engine, kind, characteristics, and capabilities are defined as `defconst` lists (`llm-setup-all-model-providers`, `llm-setup-all-model-engines`, etc.) near line 351. `llm-setup-check-instances` validates against these.

### Provider-to-LiteLLM Mapping
In `llm-setup-insert-instance-litellm`, provider symbols map to LiteLLM provider strings: `local` → `"openai"`, `positron` → `"openai"`, `positron_anthropic` → `"anthropic"`, etc. Each provider also maps to a credential name.
