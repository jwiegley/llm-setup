# OpenRouter Qwen and Kimi Model Design

## Goal

Expose the strongest currently available Qwen 3.7 model and Kimi K3 through
the existing OpenRouter-backed model registry, while preserving the local
GLM 5.2 deployment.

## Chosen model identifiers

- `qwen/qwen3.7-max`: OpenRouter's flagship Qwen 3.7 model. No Qwen 3.8
  model is present in the current OpenRouter catalog.
- `moonshotai/kimi-k3`: the stable Kimi K3 identifier. The moving
  `~moonshotai/kimi-latest` alias is deliberately not used.

## Registry design

Add `Qwen3.7-Max` and `Kimi-K3` as model families with one instance each.
Each instance uses the exact remote model identifier and the `openrouter`
provider. Family context lengths mirror OpenRouter's advertised limits, and
both families are marked as reasoning-capable.

The pending GLM OpenRouter entry currently duplicates the existing local
`GLM-5.2` family. Consolidate it into the existing family as a second instance,
with a 1,048,576-token instance-level context override. Keep the family's
200,000-token local context and sampling behavior unchanged.

## Verification

Add a table-driven ERT test that proves the two new families have only the
requested OpenRouter instances and that GPTel and Nix registry projection
preserve their exact identifiers. Extend coverage for the consolidated GLM route, then
run the complete ERT suite, formatting checks, and `nix flake check`.
