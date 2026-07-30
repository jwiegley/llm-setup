# llm-setup.el — LLM model management for Emacs

I've been running a growing collection of local LLMs across multiple
machines, and the bookkeeping got out of hand pretty fast. Which models
are on which host? What engine does each one need? How do I keep
llama-swap, the Nix model registry, and GPTel in sync when I add or remove
a model?

`llm-setup.el` is my answer to that. It's a single Emacs Lisp file that
maintains a model registry -- a list of `llm-setup-model` and
`llm-setup-instance` structs -- plus explicit model selections. From these
facts it generates llama-swap YAML for hera and clio, publishes a nonsecret
model registry for Nix, and updates GPTel for in-editor use.

## How it works

The registry in `llm-setup-models-list` is the source of deployed model facts.
Each model has family-level metadata (name, characteristics, capabilities,
sampling parameters) and one or more deployment instances (provider, engine,
hostnames, file paths). `llm-setup-nix-provider-defs` supplies provider facts
and exceptional Nix-only static routes such as NVIDIA. The explicit default and
Claude variables are the source of the exact model selections. Everything else
is derived.

The infrastructure managed by this package looks like this:

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

Running `M-x llm-setup-reset` validates the registry, rebuilds and restarts
llama-swap on hera and clio, publishes the nonsecret Nix registry, and updates
GPTel in five steps. It does not generate or deploy configuration for downstream
model gateways.

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
3. Run `M-x llm-setup-reset` to validate, deploy, and publish the Nix registry

## Development

After modifying `llm-setup.el`, reload it in your running Emacs:

```elisp
(unload-feature 'llm-setup t)
(load-file "llm-setup.el")
```

### Checks

All static checks run via `nix flake check`, which covers:

- **Byte-compilation** with warnings treated as errors
- **package-lint** for package header and dependency conventions
- **checkdoc** for docstring style
- **relint** for regexp correctness

Pre-commit hooks (via lefthook) run the same checks in parallel on staged
files, plus ERT regression tests for generated configuration and a format
check via `format-all`.

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
