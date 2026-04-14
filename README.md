# llm-setup.el — LLM model management for Emacs

I've been running a growing collection of local LLMs across multiple
machines, and the bookkeeping got out of hand pretty fast. Which models
are on which host? What engine does each one need? How do I keep
llama-swap, LiteLLM, and GPTel all in sync when I add or remove a model?

`llm-setup.el` is my answer to that. It's a single Emacs Lisp file that
maintains a model registry -- a list of `llm-setup-model` and `llm-setup-instance`
structs -- and generates all the downstream configuration from it:
llama-swap YAML for each host, LiteLLM config for the central proxy, and
GPTel backend definitions for in-editor use.

## How it works

The registry in `llm-setup-models-list` is the single source of truth. Each
model has family-level metadata (name, characteristics, capabilities,
sampling parameters) and one or more deployment instances (provider,
engine, hostnames, file paths). Everything else is derived.

The infrastructure looks like this:

```
llm-setup-models-list (Elisp structs)
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

Running `M-x llm-setup-reset` validates the registry, rebuilds all YAML configs,
restarts the remote services, and updates GPTel -- five steps, fully
automated.

## Getting started

The easiest way to get a development shell with all the tooling is
through Nix:

```bash
nix develop
```

This gives you Emacs (with `package-lint` and `relint`), `lefthook` for
`pre-commit` hooks, and everything needed to run the checks.

For day-to-day use, `llm-setup.el` loads into your Emacs session like any other
package. The typical workflow is:

1. Download a model with `M-x llm-setup-download`
2. Add a `make-llm-setup-model` / `make-llm-setup-instance` entry to `llm-setup-models-list`
3. Run `M-x llm-setup-reset` to validate and deploy

## Development

After modifying `llm-setup.el`, reload it in your running Emacs:

```elisp
(unload-feature 'llm-setup t)
(load-file "llm-setup.el")
```

### Checks

All checks run via `nix flake check`, which covers:

- **Byte-compilation** with warnings treated as errors
- **package-lint** for package header and dependency conventions
- **checkdoc** for docstring style
- **relint** for regexp correctness
- **Format check** via `indent-region`

Pre-commit hooks (via lefthook) run the same checks in parallel on
staged files.

### Formatting

To format `llm-setup.el` in place:

```bash
scripts/format.sh llm-setup.el
```

To check formatting without modifying:

```bash
scripts/check-format.sh llm-setup.el
```

## License

BSD 3-Clause. See [LICENSE.md](LICENSE.md).
