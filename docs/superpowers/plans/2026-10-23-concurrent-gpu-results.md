# Concurrent GPU Results Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task. The user approved execution in this conversation.

**Goal:** Support independent GPU development and reproducible result contributions, with mono publishing the combined reports.

**Architecture:** Keep existing GPU output directories and the report generator. Replace implicit latest-file selection with one explicit JSON input manifest per GPU. Record commands and environment alongside each new experiment. Publish reports from mono after raw results merge.

**Tech Stack:** Git, Python standard library, existing openpyxl generator, NVIDIA CLI tools.

**Spec:** User-approved design in this conversation, summarized above and in docs/concurrent-development.md.

## Global Constraints

- Preserve existing primary checkouts, unpushed branches, worktrees, measurements, report layout, and CUDA code.
- Use task branches with machine prefixes. Only the integration workflow commits generated reports.
- Record measurement and validation independently; never report new GPU measurements from setup checks.
- Keep RTX 5090 measurements pending and prediction disabled until there is a justified model for that architecture.
- Never select a different experiment merely because a newer filename exists.
- Git 2.34 on octacos requires diff3; newer Git on mono and kilia uses zdiff3.

## Review Focus

- Missing selected files must fail visibly rather than silently become pending.
- Paths must stay inside the selected GPU's directory, with no wildcards or traversal.
- Failed validation must prevent measurement and leave a failure record.
- Concurrent experiments on one GPU must use the existing shared GPU lock and distinct output names.
- Newly added RTX 5090 rows must be pending, with no inherited 4090 fallback data or predictions.

### Task 1: Explicit GPU inputs and pending RTX 5090 support

**Files:** results/build_spreadsheet.py, results/inputs/*.json, tests/test_results.py.

**Interfaces:** GPU.file(key) returns one repository-relative input path or None. load_gpus() reads the three manifests.

- [x] Write tests for pinned selection, missing input errors, path containment, and pending 5090 rows/predictions; observe failures.
- [x] Replace inline GPU configurations and glob selection with validated manifests. Preserve all existing selected inputs.
- [x] Run unit tests and regenerate reports on mono. Compare all old measurements and formulas to the committed workbook.

### Task 2: Experiment provenance

**Files:** results/record_run.py, tests/test_record_run.py.

**Interfaces:** CLI --check SHELL_COMMAND --device DEVICE -- COMMAND...; write metadata and logs under <gpu>/runs/<unique-run-id>/.

- [x] Write tests for successful validation/measurement, validation failure, measurement failure, clean source requirement, and unique run IDs; observe failures.
- [x] Record source commit, GPU UUID, driver/tool versions, commands, UTC times, statuses, and output paths. Set one device and a unique TIMESTAMP for existing scripts. Keep logs without streaming large measurement output into chat.
- [x] Run the complete Python test suite and command-line help checks.

### Task 3: Workflow and rollout

**Files:** docs/concurrent-development.md, AGENTS.md, README.md, .gitignore, .gitattributes.

- [x] Document task branches/worktrees, exact inputs, source revision comparisons, per-GPU locking, mono report publication, and each machine's next experiment.
- [x] Mark the workbook binary and ignore local worktree/cache directories.
- [x] Run checks and a fresh code review; fix material findings.
- [x] Commit and push a reviewable setup branch, create and attach a draft PR, and make task worktrees available on all three machines. Keep main unchanged pending review.
