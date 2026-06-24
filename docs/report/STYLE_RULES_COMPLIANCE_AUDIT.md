# A4 Style Rules Compliance Audit

Status: A4 audit/preparation. This file defines the wording and asset rules to use before any later LaTeX patching. Current report source and new A1/A2 evidence override legacy guides.

## Sources Read

| Source | source_role | How A4 uses it |
| --- | --- | --- |
| `HK252-DATN-142/thesis.tex` | current_report_source | Determines the real entrypoint, loaded class, loaded packages, chapter order, bibliography style, and appendix flow. |
| `HK252-DATN-142/chapters/main/*.tex` | current_report_source | Determines existing section names, labels, current figures/tables, and stale text to patch later. |
| `HK252-DATN-142/refs/example.bib` | current_report_source | Only valid citation keys for future report text. Empty citations are forbidden. |
| `HK252-DATN-142/hcmut-thesis.cls` | current_report_source | Real thesis class loaded by `thesis.tex`. |
| `docs/ml/*`, `docs/fusion/*`, `docs/tables/*`, `docs/paired/*` | authoritative | New A1/A2 evidence and tables for this A4 integration pass. |
| `HK252-DATN-142/README.txt` | legacy_reference | Template provenance only. No report claim should depend on it. |
| `HK252-DATN-142/THESIS_EVIDENCE_INDEX.md` | legacy_reference | Advisory convention and old asset idea source only; it cannot force inclusion or override A1/A2 evidence. |
| `HK252-DATN-142/THESIS_STYLE_RULES.md` | legacy_reference | Advisory style/convention source. Its generic writing rules are useful, but stale evidence entries are not authoritative. |
| `HK252-DATN-142/THESIS_STYLE_RULES_TEMPLATE.md` | template | Template reference only. |
| `HK252-DATN-142/THESIS_WRITING_GUIDE.md` | legacy_reference | Redirect-only guide. It points to split legacy files and is not authoritative. |
| `HK252-DATN-142/THESIS_FIGURE_MANIFEST.md` | legacy_reference | Advisory figure idea list. Old required assets are not mandatory for this A4 plan. |
| `docs/PIPELINE_EVIDENCE_REGISTER.md` | legacy_reference | Useful for terminology and earlier non-claims, but new A1/A2 evidence supersedes stale timing or fusion claims. |
| `Thesis_report/ASSET_MANIFEST.md` | legacy_reference | Advisory cross-check for legacy copied assets only. |
| `HK252-DATN-142/ASSET_MANIFEST.md` | missing | Not present in the current report folder. |

## Rules Affecting ML Wording

- Write report-facing ML text in English academic style and avoid first person.
- Do not mention implementation file names or tool commands in report body. Use logical names such as "environment classifier", "feature extraction stage", and "offline evaluation platform".
- Use the new A1 evidence wording: "offline optical RGB-proxy baseline", "duplicate-cluster-aware split", and "not source-held-out".
- RF200 may be described as the current offline accuracy baseline. RF100 may be described as a lightweight candidate pending target-hardware latency.
- MLP rejection must be framed around weaker balanced accuracy and macro-F1, not around latency alone.
- Raw and paired sensor runs are domain-shift/proxy behavior evidence only. They are not user-confirmed sensor accuracy.
- The agent-labeled low-score subset is a limitation/internal diagnostic. It must not appear in the main result table as final model performance.

## Rules Affecting Fusion And Image Wording

- Separate original algorithms and citations from pipeline adaptation and trade-offs when future Ch2/Ch5 text discusses algorithms.
- Use "paired generated fusion" or "generated offline fusion" for current fusion metrics.
- Do not write that captured runtime fusion is validated. `captured_runtime_fusion_available=false`.
- Thermal evidence is display/heatmap-like, not raw radiometric thermal.
- Pairing quality evidence supports synchronization only; it does not prove spatial alignment or fusion quality.
- Failure mining is diagnostic and belongs in limitations or appendix unless the main text needs a short caveat.

## Rules Affecting Tables, Figures, And Captions

- Every future table/figure must be introduced in prose before the float.
- Captions must include the evidence tier or caveat when a metric is proxy, generated, or preliminary.
- Multi-column tables should use `tabularx` with `\small` or `\footnotesize`.
- Avoid wide raw CSV dumps in the thesis body. Summarize in compact tables and keep full artifacts in docs/artifacts.
- Legacy assets are advisory. If a legacy figure is stale or superseded by current A1/A2 evidence, mark it `rejected_legacy_asset`.

## Rules Affecting Citations

- Do not emit empty `\cite{}`.
- Only use keys present in `HK252-DATN-142/refs/example.bib`; currently available keys include `ref11`, `ref12`, `ref13`, `ref14`, `ref15`, `ref19`, `ref22`, `ref33`, `ref61`, `ref63`, `ref64`, `ref65`, `ref66`, `ref67`, `ref69`, `ref70`, `ref73`, `ref74`, `ref75`, `ref76`, `ref77`, and `huang2025cascaded`.
- Internal artifacts do not need BibTeX citations. They need evidence-category wording and traceable artifact paths in A4 docs.
- `thesis.tex` uses `plain` bibliography style, so numeric reference order may be alphabetical by author rather than first citation order.

## Rules Affecting Limitations And Future Work

- Preserve `RPi4-PENDING` gates. Do not replace pending placeholders with proxy values.
- Use "development workstation", "x86-64 development workstation", or "offline evaluation platform"; avoid "Mac" or "MacBook" in future report body.
- Service-specific cloud products should not appear in main body text unless explicitly scoped; generic "cloud-based IoT platform" wording is safer.
- Missing user-confirmed labels, missing RPi4 latency, missing captured runtime fusion, missing raw radiometric thermal arrays, missing homography metadata, and missing rain/dawn evidence should be explicit limitations.

## Claims Or Assets Excluded From Main Report

| Item | Decision | Reason |
| --- | --- | --- |
| `docs/tables/ml/agent_labeled_sensor_eval.md` as main metric | omit from main result | Agent/manual labels are not user-confirmed; metrics are a diagnostic for domain-shift risk only. |
| Legacy fusion benefit by class figure | rejected_legacy_asset | Superseded by strict paired generated fusion evidence and positive-first caveats. |
| Legacy alpha sweep as final fusion optimum | rejected_legacy_asset | Stale/proxy and not aligned with captured-runtime validation. |
| Raw radiometric thermal claim | omit | Current thermal modality is display/heatmap-like. |
| Captured runtime fusion validation claim | omit | Runtime captured fusion output is not measured. |
| Rain temporal median performance claim | limitation only | Current paired evidence says not measured for rain/wet sequence evidence. |
| Dawn/dusk blend performance claim | limitation only | Current paired evidence says not measured for dawn/dusk metadata or confident visual evidence. |
