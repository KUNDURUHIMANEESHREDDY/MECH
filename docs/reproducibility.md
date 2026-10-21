# Reproducibility Guide

A core tenet of this platform is one-click reproducibility of landmark mechanistic interpretability papers.

## Workflow
1. Select the target paper in the UI.
2. The `ModelManager` securely fetches weights from Hugging Face.
3. The specific pipeline executes against the dataset.
4. A `ReproducibilityReport` is generated.
5. The `ProjectValidator` ensures all artifacts (model hash, figures, notebook) are present.
6. Export the `.interp-project` archive: `mech projects export <project_id>` (or `POST /api/exports/<project_id>`). The archive holds that project's sessions and a `metadata.json` that records what was included, what was omitted and why. An export estimated above `MECH_EXPORT_CONFIRM_BYTES` (default 100 MiB) needs confirmation first -- `--yes` in the CLI, `{"confirm": true}` over HTTP -- and a project that no stored session names is refused rather than exported empty.

Other researchers can import this archive to independently verify the findings.
