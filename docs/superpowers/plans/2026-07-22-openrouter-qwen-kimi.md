# OpenRouter Qwen and Kimi Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register Qwen3.7 Max and Kimi K3 as deterministic OpenRouter models and restore the model registry's uniqueness invariant.

**Architecture:** Extend the existing `llm-setup-models-list` source of truth with two OpenRouter-only families. Merge the pending remote GLM instance into its existing local family so all generated GPTel and Nix registry routes continue to derive from one unique family entry.

**Tech Stack:** Emacs Lisp, ERT, GPTel backend generation, Nix model-registry generation, Nix quality checks.

## Global Constraints

- Use `qwen/qwen3.7-max`; OpenRouter currently has no Qwen 3.8 model.
- Use the stable `moonshotai/kimi-k3` identifier, not a moving alias.
- Do not resolve, print, or otherwise inspect credentials.
- Preserve the existing local GLM 5.2 launch configuration.

---

### Task 1: Add verified OpenRouter registry entries

**Files:**
- Modify: `llm-setup-test.el`
- Modify: `llm-setup.el`

**Interfaces:**
- Consumes: `llm-setup-models-list`, `llm-setup-get-instance-gptel-backend`, and `llm-setup-render-nix-model-registry`.
- Produces: deterministic GPTel and Nix registry routes for `qwen/qwen3.7-max`, `moonshotai/kimi-k3`, and the consolidated `z-ai/glm-5.2` instance.

- [ ] **Step 1: Write the failing registry-route test**

Add a table-driven ERT test that locates `Qwen3.7-Max` and `Kimi-K3`, checks
their exact OpenRouter instance names, and checks their generated GPTel and
Nix registry identifiers. Add a focused assertion that `GLM-5.2` is a unique family
with both its local and OpenRouter instances.

- [ ] **Step 2: Run the focused tests to verify they fail**

Run:

```bash
emacs --batch -Q -L . \
  --eval '(setq load-prefer-newer t)' \
  -l llm-setup-test.el \
  --eval '(ert-run-tests-batch-and-exit "llm-setup-test-openrouter")'
```

Expected: failure because the Qwen and Kimi families are absent and GLM is
still duplicated.

- [ ] **Step 3: Implement the minimal registry changes**

Add sorted `Kimi-K3` and `Qwen3.7-Max` families, each with a single
`openrouter` instance using its exact remote identifier. Move the pending
`z-ai/glm-5.2` instance into the existing `GLM-5.2` family with its remote
context override.

- [ ] **Step 4: Run focused and complete verification**

Run the focused ERT selector, the complete ERT suite, format checks for both
Elisp files, and `nix flake check`.

- [ ] **Step 5: Commit**

Stage `llm-setup.el` and `llm-setup-test.el`, review the staged diff, and
commit the complete approved registry update. Then update, commit, and push
the parent repository's submodule pointer.
