# LaTeX Impact Check Report

Compiled by: Agent 4.5 (continuation of Agent 4 patch pass)
Date: 2026-05-28
Branch: `refactor/ml-taxonomy-eval-plan`
Scope: Full chapter-by-chapter audit of `HK252-DATN-142/` report tree.

---

## Overview

Agent 4 patched nine files and created seven table assets. Agent 4.5
reconstructed the current disk state, ran a complete static impact audit
across all chapters and front/back matter, confirmed consistency, and
produced this report.

No additional source patches were required. All stale/overclaim strings
targeted by the impact audit have been resolved. The `0.984` value remaining
in `chapters/back/sweeps.tex` is in a legacy appendix table whose caption
and section header explicitly label it as parameter context and not
sensor-real accuracy evidence.

---

## Per-Chapter Impact Check

### Front Matter

| File | Last modified | Stale strings checked | Stale strings found | Action |
|------|--------------|----------------------|---------------------|--------|
| `chapters/front/abstract.tex` | 2026-05-28 02:26 | synthesized thermal, 47.8, 0.984, all objectives met, validated on real, raw radiometric, source-held-out, no domain-shift | None in body | Patched by Agent 4 — current |
| `chapters/front/abbreviations.tex` | 2026-05-21 01:31 | (no claims to check) | N/A | Not touched; no stale claims |
| `chapters/front/acknowledge.tex` | 2026-05-21 01:31 | (no claims to check) | N/A | Not touched; no stale claims |
| `chapters/front/declaration.tex` | 2026-05-21 01:31 | (no claims to check) | N/A | Not touched; no stale claims |

**Abstract finding:** Current abstract states RF200 balanced accuracy 0.7463 and macro-F1 0.7362 on the duplicate-cluster-aware split; captured runtime fusion and raw radiometric thermal validation are correctly labelled as future work. No overclaim.

---

### Chapter 1 — Introduction

| File | Last modified | Status |
|------|--------------|--------|
| `chapters/main/ch1-introduction.tex` | 2026-05-21 01:31 | Not patched by Agent 4 |

**Stale string scan result:** NONE found for target patterns
(`synthesized thermal`, `synthetic thermal`, `47.8`, `0.984`, `all objectives met`,
`no domain-shift`, `20 FPS`, `validated on real`, `runtime fusion.*valid`,
`raw radiometric`, `source-held-out`).

**Additional checks:** No empty `\cite{}`, no bare `\begin{tabular}` (only `tabularx` used), no banned hardware names in body lines.

**Verdict:** No impact; no patch required.

---

### Chapter 2 — Background

| File | Last modified | Status |
|------|--------------|--------|
| `chapters/main/ch2-background.tex` | 2026-05-21 01:31 | Not patched by Agent 4 |

**Stale string scan result:** NONE found for target patterns.

**Additional checks:** No empty `\cite{}`, no banned hardware names, no bare `\begin{tabular}`.
Unicode bytes (`§` and `→`) found only in comment lines referencing old guide files — not in any PDF-visible body text.

**Verdict:** No impact; no patch required.

---

### Chapter 3 — Requirements, Scope, and Methodology

| File | Last modified | Status |
|------|--------------|--------|
| `chapters/main/ch3-requirements.tex` | 2026-05-28 02:30 | Patched by Agent 4 |

**What was patched:**
- Q1 caveat updated: now states 584 frame-strict paired-input, generated offline fusion, raw radiometric thermal data and alignment metadata are future validation gates.
- NFR table updated: NFR-P1 now reads "Caveated" with "full current mode-matrix acceptance pending"; NFR-P3 reads "Pending" with "RF100/RF200 feature-plus-predict latency not measured on target hardware"; NFR-A1/A2 read "Met offline" with RF200 duplicate-cluster-aware figures.
- Non-claims table NC4 updated to reflect sensor proxy audit with user-confirmed labels absent.

**Post-patch stale string scan:** NONE found in body text.

**Verdict:** Patched and consistent.

---

### Chapter 4 — System Design

| File | Last modified | Status |
|------|--------------|--------|
| `chapters/main/ch4-system-design.tex` | 2026-05-28 02:30 | Patched by Agent 4 |

**What was patched:**
- Stage timing budget section: wording updated so existing timing telemetry is not reused as proof of RF100/RF200 model migration latency or as generated fusion quality validation.
- Clarification added that timing exports establish optimization pressure points, not current model-latency or captured runtime fusion validation gates.

**Post-patch stale string scan:** NONE found. The word "synthesis" on line 50 is a design framing sentence ("algorithmic synthesis"), not a claim about synthesized thermal data.

**Verdict:** Patched and consistent.

---

### Chapter 5 — Implementation

| File | Last modified | Status |
|------|--------------|--------|
| `chapters/main/ch5-implementation.tex` | 2026-05-28 02:30 | Patched by Agent 4 |

**What was patched:**
- ML section: removed ablation table referencing old cross-validation scores; added rationale for RF200/RF100/MLP/feature-v2 decisions consistent with duplicate-cluster-aware evidence.
- Fusion section: wording updated so global `fusion_alpha` is "an implementation trade-off rather than a final quality optimum"; paired-input generated fusion distinction made explicit; captured runtime fusion caveat added.
- No implementation details reference synthesized thermal, 47.8 ms as field proof, or stale cross-val macro-F1.

**Post-patch stale string scan:** NONE found for target patterns. Only legitimate reference found (line 476): explicitly states "that evidence is not captured runtime fusion validation."

**Verdict:** Patched and consistent.

---

### Chapter 6 — Evaluation and Testing

| File | Last modified | Status |
|------|--------------|--------|
| `chapters/main/ch6-evaluation.tex` | 2026-05-28 02:30 | Patched by Agent 4 (major rewrite of ML and fusion sections) |

**What was patched:**
- Evaluation methodology section: three-tier evidence framework introduced (offline algorithm/ML evidence, paired-input sensor evidence, target-hardware telemetry).
- ML section: replaced with duplicate-cluster-aware benchmark (11,981 train / 2,113 test, RF200 baseline, RF100 lightweight candidate), sensor-domain proxy audit (caveated, no user-confirmed labels), and model-selection gate table.
- Fusion section: replaced with strict paired fusion evidence (584 frame-strict rows), foreground-mask overlay vs alpha-blend comparison; explicit caveat that thermal stream is display/heatmap-like, not raw radiometric; captured runtime fusion explicitly not claimed.
- Latency section: telemetry retained as implementation context only; not mixed with new classifier or fusion claims.
- New table inputs: `ml_offline_model_comparison_table.tex`, `ml_model_decision_table.tex`, `ml_domain_shift_summary_table.tex`, `ml_class_decision_table.tex`, `fusion_result_summary_table.tex`, `per_bucket_evidence_table.tex`.

**Post-patch stale string scan:**
- `raw radiometric`: found in lines 26, 398 — both are correct caveats ("raw radiometric thermal arrays remain unavailable", "display/heatmap-like rather than raw radiometric thermal").
- `source-held-out`: found in line 125 — correct caveat ("split is not source-held-out").
- No overclaim variants found.

**Verdict:** Patched and consistent. New tables and figures all verified to exist on disk.

---

### Chapter 7 — Conclusion

| File | Last modified | Status |
|------|--------------|--------|
| `chapters/main/ch7-conclusion.tex` | 2026-05-28 02:25 | Patched by Agent 4 |

**What was patched:**
- Objectives table: O1 now "Caveated" (runtime fusion pending); O3 now "Met offline" with RF200 figures; O4 "Met" with evidence separation framework.
- RQ answers: Q1 answer now explicitly states generated offline fusion with foreground-mask overlay improvement figures, and explicitly does not claim captured runtime fusion, raw radiometric thermal, or task-level detection accuracy.
- Q3 answer: "Null result (transparent)" — no ground-truth labels; E1 remains an indicator only.
- Contributions: all four contributions stated with correct evidence tier.
- Limitations: domain shift addressed as proxy-behavior audit (590 raw sensor frames, 584 strict paired NIR rows), user-confirmed labels absent.
- Future work: captured runtime fusion, user-confirmed sensor labels, target-hardware model latency all listed as future validation gates.
- Closing statement: explicitly names outstanding validation gates.

**Post-patch stale string scan:**
- `captured runtime fusion` in lines 105, 471 — correct caveats (not claiming this).
- `raw radiometric thermal` in lines 105, 211, 411, 471 — correct non-claim or future work citations.
- `synthetic thermal` in line 126 — in context: "algorithmic response to synthetic thermal spikes has been confirmed through offline evaluation" — this is a legitimate claim about offline anomaly evaluation using surrogate thermal data, which is consistent with Ch6 Section 6.5.

**Verdict:** Patched and consistent.

---

### Back Matter

#### Appendix B — Unit Test Suite

| File | Last modified | Status |
|------|--------------|--------|
| `chapters/back/test-suite.tex` | 2026-05-21 01:31 | Not patched |

**Scan result:** NONE for target stale patterns. Describes the 31 pytest modules — factual, no stale claims.

**Verdict:** No impact; no patch required.

---

#### Appendix C — Parameter Sweep Results

| File | Last modified | Status |
|------|--------------|--------|
| `chapters/back/sweeps.tex` | 2026-05-28 02:28 | Patched by Agent 4 |

**What was patched:**
- ML threshold sweep section: renamed to "ML Confidence Gate Sweeps (Legacy)"; added caveat that these rows are "parameter context, not sensor-real accuracy evidence."
- Fusion alpha sweep section: labelled explicitly as "legacy proxy sweep … retained as historical parameter context only."
- Thermal parameter sweeps: surrogate data caveat added at bottom.

**Post-patch stale string scan:**
- `0.984` appears in line 38 of the sweep table body. This is the legacy night-class F1 for the τ₁=0.62 threshold row. The table caption (line 25–27) reads: "Legacy primary (τ₁) and secondary (τ₂) threshold sweeps. These rows are parameter context, not sensor-real accuracy evidence." Section header (lines 15–18) also explicitly states: "These legacy threshold sweeps are retained as parameter context … They are not the current main classifier performance claim." The `0.984` figure is thus correctly contextualized as an appendix-only legacy value and is not overclaimed.

**Verdict:** Patched and consistent. `0.984` in appendix table is acceptable with current caveat wording.

---

#### Appendix D — Design Documentation

| File | Last modified | Status |
|------|--------------|--------|
| `chapters/back/design-docs.tex` | 2026-05-28 02:30 | Patched by Agent 4 |

**What was patched:**
- Legacy sidecar/sweep rows: marked as "parameter context only."
- Current RF200 macro-F1 rows added with caveat "Duplicate-cluster-aware offline split; not sensor-real accuracy."

**Post-patch stale string scan:** NONE for target patterns.

**Verdict:** Patched and consistent.

---

#### Appendix — Code Excerpts

| File | Last modified | Status |
|------|--------------|--------|
| `chapters/back/code-excerpts.tex` | 2026-05-21 01:31 | Not patched |

**Scan result:** NONE for target stale patterns.

**Verdict:** No impact; no patch required.

---

### Bibliography

| File | Last modified | Status |
|------|--------------|--------|
| `refs/example.bib` | 2026-05-21 01:31 | Not patched |

**Citation key audit:** All keys used in patched chapters (`ref11`, `ref13`, `ref14`, `ref15`, `ref19`, `ref22`, `ref28`, `ref33`, `ref36`, `ref37`, `ref61`, `ref63`, `ref65`, `ref66`, `ref67`, `ref69`, `ref70`, `ref73`, `ref74`, `ref77`) verified present in `refs/example.bib`.
Empty `\cite{}` check: NONE found across all chapters.

**Verdict:** No impact; no patch required.

---

### Thesis Entry Point

| File | Last modified | Status |
|------|--------------|--------|
| `thesis.tex` | 2026-05-28 02:29 | Patched by Agent 4 (ASCII comment cleanup) |

**Scan result:** NONE for target stale patterns in body text.

**Verdict:** Patched (minor) and consistent.

---

## Table Assets Verified

All seven table files inserted via `\input{...}` in Ch6 and Ch7 exist on disk:

| Table file | Path exists? |
|-----------|-------------|
| `ml_offline_model_comparison_table.tex` | OK |
| `ml_model_decision_table.tex` | OK |
| `ml_domain_shift_summary_table.tex` | OK |
| `ml_class_decision_table.tex` | OK |
| `fusion_result_summary_table.tex` | OK |
| `per_bucket_evidence_table.tex` | OK |
| `limitations_table.tex` | OK |

---

## Figure Assets Verified

All 22 `\includegraphics{...}` paths in Ch5–Ch7 verified to exist on disk. Zero missing figure paths.

---

## Label/Reference Sanity

- Duplicate label check across Ch5–Ch7 and all table snippets: NONE found.
- All `\ref{}` and `\Cref{}` targets in Ch5–Ch7 resolve to labels defined within the report tree.

---

## Unicode / Encoding Check

- Direct non-ASCII bytes (`—`, `–`, `…`, `µ`, `→`, `≤`, `≥`, `≈`, `§`) checked across all patched chapter files and table snippets: NONE found in PDF-visible body text.
- Unicode bytes found only in comment lines (`% ...`) in Ch2 — these do not appear in compiled output.

---

## Stale/Overclaim Summary

| Claim type | Found in body? | Action |
|-----------|---------------|--------|
| "synthesized thermal" / "synthetic thermal" as main accuracy claim | No | Agent 4 removed |
| Macro-F1 0.984 as current production metric | No | Moved to legacy appendix with caveat |
| `47.8 ms` as RPi4 field proof | No | Agent 4 removed |
| "all objectives met" unconditionally | No | Agent 4 changed to tiered status |
| "no domain-shift evaluation" | No | Agent 4 replaced with proxy audit caveat |
| Captured runtime fusion claimed | No | Consistently not claimed throughout |
| Raw radiometric thermal claimed | No | Consistently labelled as not available |
| User-confirmed sensor accuracy claimed | No | Consistently labelled as future gate |
| RPi4 model latency claimed | No | Consistently labelled as pending/not measured |

---

## Final Impact Verdict — No-Impact Conclusion

**All stale/overclaim issues identified in the A4 audit have been resolved.**
No additional patches are required for claim accuracy, evidence gates, or
chapter consistency. The report is internally consistent across all chapters.

**No-impact chapters:** Ch1, Ch2, abbreviations, acknowledge, declaration,
code-excerpts appendix, and bibliography — all scanned, no stale claims found,
no patches applied.

**Patched chapters:** abstract, Ch3, Ch4, Ch5, Ch6, Ch7, sweeps appendix,
design-docs appendix, thesis.tex — all stale claims resolved.

Remaining open items are documentation-only (build toolchain not available in
current shell) and future validation work items documented transparently in
Ch7 and the Non-claims table in Ch3.
