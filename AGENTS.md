# Concurrent GPU development

- Render mathematical expressions in LaTeX. Prefer shorter sentences and fewer hyphens.
- Read `docs/concurrent-development.md` before running experiments or publishing results.
- Use one task branch per worktree with a `mono/`, `octacos/`, or `kilia/` prefix. After the infrastructure PR merges, start tasks from current `origin/main`.
- Each machine contributes raw outputs in its own GPU directory and edits its own `results/inputs/<gpu>.json`.
- Keep experiment outputs immutable. Use `results/record_run.py` to record source revision, environment, validation, and commands. Command success alone does not establish a scientific conclusion.
- Only the mono integration checkout commits generated `results/cuda_performance.xlsx` and `results/README.md`, after the referenced raw results merge. Other branches may edit the generator.
- Preserve the large workload report convention in `docs/concurrent-development.md`: primary GPU summaries and GPU analysis compare large workloads; small workloads are RTX 3090 diagnostics only. Keep the report layout tests passing.
- Compare GPUs at the same source revision, shapes, inputs, and timing settings. Record compiler/driver differences instead of treating them as GPU-only effects.
- Serialize GPU work with the shared `GPU_LOCK`. Timing scripts already lock; wrap checks and `profile_ncu_all.sh` as documented.
- Use explicit input paths. Null means pending. Never substitute another GPU's value or an unvalidated prediction.
- Preserve existing worktrees, unpushed commits, and historical profiles. Do not force-push shared branches.
- The private Loom suite lives in `applications/loom_private/` with its own report in `results/loom_private/`. Keep it separate from the legacy workloads; `applications/loom_private/tools/legacy_freeze.py verify` must pass.
