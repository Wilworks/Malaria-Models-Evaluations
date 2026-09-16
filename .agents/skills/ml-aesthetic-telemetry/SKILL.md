---
name: ml-aesthetic-telemetry
description: >-
  Enforces state-of-the-art terminal aesthetics, rich console telemetry, glowing status badges,
  animated progress bars, and formatted diagnostic tables for any machine learning model training,
  evaluation, inference, or benchmarking pipeline. Automatically apply whenever working on ML pipelines.
---

# Machine Learning Aesthetic Telemetry & Terminal UI Skill

This skill defines the mandatory standard for all terminal outputs, CLI tools, model training loops, and evaluation benchmarks built for Wilfred.

## Core Mandate
Whenever writing or executing scripts for machine learning, deep learning, zero-shot evaluation, dataset auditing, or pipeline benchmarking:
* **NEVER output plain, unstyled print dumps.**
* **ALWAYS implement interactive, vibrant terminal telemetry** using `rich` (with resilient ANSI 24-bit TrueColor fallback).
* The user must experience an engaging, transparent, and aesthetically pleasing console from the moment the script boots until the final metric seal.

---

## Architecture & Standard Components

### 1. Hero Splash Banner
Every pipeline must open with a double-edge bordered panel displaying:
* Project Title & Research Domain
* Principal Investigator / Author Affiliation
* Dataset / Clinical Cohort Overview
* Execution Mode Tag (`[bold bright_magenta]Master Telemetry Console[/]`)

### 2. Status Badge Pills
Use high-contrast colored pill badges for state transitions:
* `[bold black on bright_green]  ✔ PASS  [/]` for successful verifications.
* `[bold black on bright_cyan]  ⚡ READY [/]` for initialization and hardware probes.
* `[bold black on bright_blue]  ℹ INFO  [/]` for environment details.
* `[bold black on bright_yellow]  ⚠ WARN  [/]` for missing assets or warnings.
* `[bold black on bright_red]  ✖ FAIL  [/]` for fatal errors.
* `[bold black on bright_magenta]  ★ RUN   [/]` for active inference/training loops.
* `[bold black on bright_green]  🔒 HASH [/]` for cryptographic determinism seals.

### 3. Hardware & Asset Telemetry Cards
Display system resources and model checkpoints in styled tables:
* Host OS, CPU cores, Python version.
* GPU accelerator name, VRAM, CUDA runtime, and cuDNN status.
* Model Zoo table: Name, Architecture, Training Domain, Weights Format, File Size.

### 4. Dynamic Progress Bars
Never let a long-running batch or inference step hang without visual feedback:
* Use `rich.progress.Progress` or high-resolution Unicode block bars (`━`, `█`).
* Include: Task label, percentage, processed/total count, throughput (`img/s` or `it/s`), and dynamic countdown timer (`ETA: MM:SS`).

### 5. Publication-Grade Diagnostic Results Matrix
Format evaluation outcomes in double-edge styled `rich.table.Table`:
* Modality, Model Identifier, Sample Size $N$.
* Sensitivity [95% Wilson Score CI], Specificity [95% CI].
* F1-Score, Accuracy, Precision.
* Highlight champion models and statistical advantages in `bold bright_green`.

### 6. Key Empirical Discoveries & Determinism Seal
End every benchmark run with:
* An insights card summarizing key statistical findings.
* A cryptographic determinism seal reporting total elapsed time, master SHA-256 hash, and output paths.

---

## Ready-to-Use Template

```python
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console(force_terminal=True)

# 1. Hero Banner
console.print(Panel("🔬 [bold bright_cyan]MODEL BENCHMARK PIPELINE[/]", box=box.DOUBLE_EDGE, border_style="bold bright_cyan"))

# 2. Results Table
table = Table(box=box.DOUBLE_EDGE, border_style="bold bright_green", title="[bold white on dark_green] CLINICAL DIAGNOSTIC MATRIX [/]")
table.add_column("Model", style="bold white")
table.add_column("Sensitivity", style="bold bright_green")
table.add_column("Specificity", style="bold bright_yellow")
table.add_row("Champion_Model", "87.43% [86.2-88.6]", "58.82% [42.2-73.6]")
console.print(table)
```
