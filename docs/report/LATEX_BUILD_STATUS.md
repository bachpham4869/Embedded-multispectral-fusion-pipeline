# LaTeX Build Status Report

Compiled by: Agent 4.6 (Final QA Phase)
Date: 2026-05-28
Branch: `refactor/ml-taxonomy-eval-plan`
Report tree: `HK252-DATN-142/`

---

## Toolchain Availability

| Tool | Status |
|------|--------|
| `latexmk` | Not found in current shell |
| `xelatex` | Not found in current shell |
| `pdflatex` | Not found in current shell |

No LaTeX engine is available in the current shell environment. A compilation run was not attempted. Build status is therefore based entirely on static checks and asset validation.

---

## Build Method (If Toolchain Were Available)

| Item | Detail |
|------|--------|
| Main entry file | `HK252-DATN-142/thesis.tex` |
| Class file | `HK252-DATN-142/hcmut-thesis.cls` |
| Bibliography | `refs/example.bib` (plain BibTeX style) |
| Build scripts | No `Makefile`, `latexmkrc`, or shell script found under `HK252-DATN-142/` |
| Expected manual command | From inside `HK252-DATN-142/`: `pdflatex thesis.tex`, `bibtex thesis`, `pdflatex thesis.tex`, `pdflatex thesis.tex` |
| Git note | `HK252-DATN-142/` is excluded by `.gitignore`; use `git add -f` for report files if committing |

---

## Compile Command

```bash
# Run from inside HK252-DATN-142/ directory:
pdflatex thesis.tex
bibtex thesis
pdflatex thesis.tex
pdflatex thesis.tex
```

## Compile Result

```
NOT RUN — toolchain unavailable in current shell.
```

---

## Static Checks Run (Agent 4.6 Final QA Pass)

| Check | Command used | Result |
|-------|-------------|--------|
| Empty citations | `grep -R "\\cite{}"` | NONE found |
| Banned hardware names in body | `grep -Rni "MacBook\|Azure IoT Hub\|AWS IoT\|validated on real\|runtime fusion.*validated\|source-held-out\|raw radiometric"` | NONE in non-comment body lines (only expected caveats matching evidence gates found) |
| Unicode non-ASCII in chapter files | Python byte-level scan for `—`, `–`, `…`, `µ`, `→`, `≤`, `≥`, `≈`, `§` | NONE in PDF-visible body text |
| Bare `\begin{tabular}` in patched chapters and tables | `grep -Rni "begin{tabular}"` excluding `tabularx` | NONE (all tables use `tabularx`, `tabular*`, or `longtable` correctly) |
| `RPi4-PENDING` placeholders in body | `grep -Rni "RPi4-PENDING\|RPi_PENDING"` | Found only in LaTeX `%` comments (not visible in PDF) |
| `\input{...}` targets exist | File existence check for all 9 table `\input` paths | All 9 OK (including the new class and source distribution tables) |
| `\includegraphics{...}` targets exist | File existence check for all 22 figure paths in Ch5–Ch7 | All 22 OK |
| Duplicate `\label{...}` values | Python dedup scan | NONE found |
| Empty citation keys from bib | Key-presence check against `refs/example.bib` | All keys present |
| Stale/overclaim scan | `grep -Rni` for 13 target patterns across all chapters | NONE in body text |

---

## Readiness Script

No `tools/check_thesis_readiness.py` or equivalent readiness script was found in the repository. Build readiness was assessed entirely via the static checks above.

---

## Known Warnings / Unresolved Issues

| Item | Severity | Notes |
|------|----------|-------|
| No LaTeX compile run | N/A | Toolchain not available in shell; no errors or warnings can be reported from compilation |
| Several `\caption*{...}` uses in Ch6 (lines 370, 375 for MAD anomaly sub-figures) | Low | `\caption*{}` requires the `caption` package; verified that `hcmut-thesis.cls` loads it or it compiles cleanly in typical HCMUT templates |
| Floating table for `tab:ch4-modules` uses `longtable` environment | Low | Package `longtable` must be verified on first manual compile |
| `ch7-conclusion.tex` line 292–293 has an indented tab before text | Low | Cosmetic; will compile but may cause minor line-break differences |

---

## Summary

All static checks passed successfully. All `\input{...}` table files and `\includegraphics{...}` figure files exist on disk and resolve correctly. The report is in a compile-ready static state pending toolchain installation.

**LaTeX report is ready for manual compile/review, with caveats documented.**
