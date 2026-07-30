# Reproducibility Guide

A core tenet of this platform is one-click reproducibility of landmark mechanistic interpretability papers.

## Workflow
1. Select the target paper in the UI.
2. The `ModelManager` securely fetches weights from Hugging Face.
3. The specific pipeline executes against the dataset.
4. A `ReproducibilityReport` is generated.
5. The `ProjectValidator` ensures all artifacts (model hash, figures, notebook) are present.
6. Export the `.interp-project` archive.

Other researchers can import this archive to independently verify the findings.
