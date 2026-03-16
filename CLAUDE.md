# CLAUDE.md — llm-setup (hf.el)

## Package Overview

`hf.el` is a single-file LLM model management system for Emacs. It maintains a **model registry** (`hf-models-list`) as the single source of truth and generates deployment configurations for a multi-host infrastructure:

```
hf-models-list (Elisp structs)
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
emacs -batch -L . -f batch-byte-compile hf.el
```

**Validate configuration** (checks installed models match registry, validates all fields):
```elisp
(hf-check-instances)
```

**Full deployment** (validate → rebuild all YAMLs → restart services → update gptel):
```elisp
(hf-reset)
```

**Interactive development** — after modifying `hf.el`:
```elisp
(unload-feature 'hf t)
(load-file "hf.el")
```

The deployed path (`~/.emacs.d/lisp/llm-setup`) is the same physical directory as the source (via Nix home-manager symlinks), so changes take effect immediately after `eval-buffer` or reload.

## Architecture

### Data Model

Two `cl-defstruct` types form the registry:

- **`hf-model`** — Family-level: name, description, characteristics (`high`/`medium`/`low`/`remote`/`local`/`thinking`/`instruct`/`coding`/`rewrite`), capabilities (`media`/`tool`/`json`/`url`), kind (`text-generation`/`embedding`/`reranker`), sampling parameters, and a list of instances.
- **`hf-instance`** — Deployment-level: provider, engine, hostnames, model-path, file-path, draft-model, cache settings, fallbacks. Each instance belongs to exactly one `hf-model`.

The registry lives in `hf-models-list` (a large `defcustom`). All downstream generation iterates this via `hf-instances-list`, which flattens it into `(model . instance)` cons pairs.

### Naming System

Each model has multiple names used in different contexts (documented in comments at line ~380):

| Accessor | Returns | Used For |
|---|---|---|
| `hf-model-name` | Symbol like `Qwen3.5-27B` | Internal registry key |
| `hf-instance-name` | Symbol like `mlx-community/Qwen3.5-27B-4bit`, or nil → falls back to model name | llama-swap model key, LiteLLM model name |
| `hf-instance-model-name` | Override for provider-facing name (e.g. Claude vibe-proxy) | Rarely used; currently bound but unused in LiteLLM generation |
| `hf-get-full-litellm-name` | `"host/name"` or `"provider/name"` | LiteLLM entries, fallback resolution |
| `hf-short-model-name` | Strips org prefix and GGUF suffix from directory name | Matching installed models to registry |

### YAML Generation Pipeline

**llama-swap** (`hf-generate-llama-swap-yaml`): Generates per-host. Iterates all instances, filters by hostname membership, emits engine-specific CLI commands with `${PORT}` placeholder. The prolog/epilog (`hf-llama-swap-prolog`/`hf-llama-swap-epilog`) wrap the generated model entries.

**LiteLLM** (`hf-generate-litellm-yaml`): Generates globally. All instances are included — local instances get one entry per hostname, remote instances get one entry per provider. Credentials come from `hf-litellm-credentials`, API keys from `hf-litellm-environment-function` (calls `lookup-password`). Router fallbacks are dynamically generated from instance `fallbacks` fields.

### `hf-reset` Orchestration (5 steps)

1. `hf-check-instances` — validate registry; abort on any warning
2. `hf-build-llama-swap-yaml` — write YAML for hera, kill llama-swap locally
3. `hf-build-llama-swap-yaml "clio"` — write YAML for clio via TRAMP, kill via SSH
4. `hf-build-litellm-yaml` — write config to vulcan via TRAMP, restart systemd service
5. Set `gptel-model` and `gptel-backend` via `gptel-backends-make-litellm` (defined externally)

## Adding a New Model

1. Download: `M-x hf-download` (or `hf-checkout` for git-lfs)
2. Optionally inspect: `M-x hf-show` to view GGUF metadata
3. Add a `make-hf-model` + `make-hf-instance` entry to `hf-models-list`
4. **If the model belongs to a llama-swap group** (always_on, large_models, embeddings, rerankings, stt), update `hf-llama-swap-epilog` — it contains hardcoded model names
5. Run `M-x hf-reset` to validate and deploy
6. Use `hf-generate-instance-declarations` to scaffold declarations from `~/Models`

## Critical Constraints

### Manual Synchronization Required
`hf-llama-swap-epilog` contains hardcoded model names in its `groups:` section. When models are added/removed from `hf-models-list`, the epilog must be updated separately or llama-swap grouping will be wrong.

### External Dependencies Not Defined Here
- `lookup-password` — used to fetch API keys from auth-source; defined elsewhere in the Emacs config
- `gptel-backends-make-litellm` — called in `hf-reset` step 5; defined in gptel configuration
- `yaml-mode`, `json-mode` — used for display buffers but never `require`'d

### TRAMP Patterns
Remote operations use `/ssh:hostname:` prefix for file paths (constructed by `hf-remote-path`). LiteLLM config uses multi-hop: `/ssh:vulcan|sudo:root@vulcan:`. Remote `executable-find` works by temporarily setting `default-directory` to the remote host.

### Allowed Enum Values
All valid values for provider, engine, kind, characteristics, and capabilities are defined as `defconst` lists (`hf-all-model-providers`, `hf-all-model-engines`, etc.) near line 351. `hf-check-instances` validates against these.

### Provider-to-LiteLLM Mapping
In `hf-insert-instance-litellm`, provider symbols map to LiteLLM provider strings: `local` → `"openai"`, `positron` → `"openai"`, `positron_anthropic` → `"anthropic"`, etc. Each provider also maps to a credential name.
