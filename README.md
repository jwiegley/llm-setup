# llm-setup.el — LLM model management for Emacs

I've been running a growing collection of local LLMs across multiple
machines, and the bookkeeping got out of hand pretty fast. Which models
are on which host? What engine does each one need? How do I keep
llama-swap and GPTel in sync when I add or remove a model?

`llm-setup.el` is my answer to that. It's a single Emacs Lisp file that
maintains a model registry -- a list of `llm-setup-model` and
`llm-setup-instance` structs -- plus a selected default instance. From these
facts it generates llama-swap YAML for hera and clio and updates GPTel for
in-editor use.

## How it works

The registry in `llm-setup-models-list` is the source of deployed model facts.
Each model has family-level metadata (name, characteristics, capabilities,
sampling parameters) and one or more deployment instances (provider, engine,
hostnames, file paths). `llm-setup-default-instance-name` selects the GPTel and
Aider default.

The infrastructure managed by this package looks like this:

```
llm-setup-models-list (Elisp structs)
    │
    ├─► /Users/johnw/Models/llama-swap.yaml (hera, clio)
    │     └─ Model-switching proxy on port 8080
    │
    └─► gptel backends (Emacs)
          └─ In-editor LLM interaction
```

The package is oriented to run from clio: clio paths and processes are local,
while hera is managed through TRAMP and SSH. Model host assignments remain
registry data and do not depend on the execution host.

Running `M-x llm-setup-reset` validates the registry, rebuilds and restarts
llama-swap locally on clio and remotely on hera, and updates GPTel in four
steps. It does not generate or deploy configuration for downstream model
gateways.

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

1. Add a `make-llm-setup-model` / `make-llm-setup-instance` entry to `llm-setup-models-list`
2. Run `M-x llm-setup-reset` to validate and deploy

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
