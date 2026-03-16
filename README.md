# hf.el — LLM model management for Emacs

I've been running a growing collection of local LLMs across multiple
machines, and the bookkeeping got out of hand pretty fast. Which models
are on which host? What engine does each one need? How do I keep
llama-swap, LiteLLM, and GPTel all in sync when I add or remove a model?

`hf.el` is my answer to that. It's a single Emacs Lisp file that
maintains a model registry -- a list of `hf-model` and `hf-instance`
structs -- and generates all the downstream configuration from it:
llama-swap YAML for each host, LiteLLM config for the central proxy, and
GPTel backend definitions for in-editor use.

## How it works

The registry in `hf-models-list` is the single source of truth. Each
model has family-level metadata (name, characteristics, capabilities,
sampling parameters) and one or more deployment instances (provider,
engine, hostnames, file paths). Everything else is derived.

The infrastructure looks like this:

```
hf-models-list (Elisp structs)
    │
    ├─► llama-swap.yaml (per-host: hera, clio)
    │     └─ Model-switching proxy on port 8080
    │
    ├─► litellm/config.yaml (global: vulcan)
    │     └─ Unified OpenAI-compatible proxy
    │
    └─► gptel backends (Emacs)
          └─ In-editor LLM interaction via LiteLLM
```

Running `M-x hf-reset` validates the registry, rebuilds all YAML configs,
restarts the remote services, and updates GPTel -- five steps, fully
automated.

## Getting started

The easiest way to get a development shell with all the tooling is
through Nix:

```bash
nix develop
```

This gives you Emacs (with package-lint, elisp-autofmt, and relint),
lefthook for pre-commit hooks, and everything needed to run the checks.

For day-to-day use, `hf.el` loads into your Emacs session like any other
package. The typical workflow is:

1. Download a model with `M-x hf-download`
2. Add a `make-hf-model` / `make-hf-instance` entry to `hf-models-list`
3. Run `M-x hf-reset` to validate and deploy

## Development

After modifying `hf.el`, reload it in your running Emacs:

```elisp
(unload-feature 'hf t)
(load-file "hf.el")
```

### Checks

All checks run via `nix flake check`, which covers:

- **Byte-compilation** with warnings treated as errors
- **package-lint** for package header and dependency conventions
- **checkdoc** for docstring style
- **relint** for regexp correctness
- **Format check** via elisp-autofmt

Pre-commit hooks (via lefthook) run the same checks in parallel on
staged files.

### Formatting

To format `hf.el` in place:

```bash
scripts/format.sh hf.el
```

To check formatting without modifying:

```bash
scripts/check-format.sh hf.el
```

## License

BSD 3-Clause. See [LICENSE.md](LICENSE.md).
