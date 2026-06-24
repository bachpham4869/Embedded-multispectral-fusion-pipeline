# A4 LaTeX Build Audit

Status: A4 build audit only. No report source was patched.

## Build Method Found

| Item | Finding |
| --- | --- |
| Main file | `HK252-DATN-142/thesis.tex` |
| Build scripts | No `Makefile`, `latexmkrc`, or shell build script found under `HK252-DATN-142/`. |
| Bibliography flow | `plain` BibTeX style with `refs/example.bib`. |
| Expected manual command if toolchain exists | `pdflatex thesis.tex`, `bibtex thesis`, `pdflatex thesis.tex`, `pdflatex thesis.tex` from `HK252-DATN-142/`. |

## Toolchain Availability

| Tool | Current availability |
| --- | --- |
| `latexmk` | not found in current shell |
| `xelatex` | not found in current shell |
| `pdflatex` | not found in current shell |

## Current Build Status

- A current compile was not run because no LaTeX engine was available in the shell.
- Static audit only was performed for A4.
- No missing asset or broken reference claim is made from compilation.

## Static Checks For Later Patch

- Check empty citations: `rg -n -F '\\cite{}' HK252-DATN-142/chapters HK252-DATN-142/thesis.tex`
- Check brand-specific workstation wording before final patch: `rg -n -F -e 'Mac' -e 'MacBook' HK252-DATN-142/chapters`
- Check pending gates: `rg -n -F -e 'RPi4-PENDING' -e 'RPi_PENDING' HK252-DATN-142/chapters`
- Check broad tabular usage after patch: `rg -n -F '\\begin{tabular}' HK252-DATN-142/chapters`
