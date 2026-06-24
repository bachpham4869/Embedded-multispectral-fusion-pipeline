#!/usr/bin/env python3
"""
tools/check_thesis_readiness.py
================================
Thesis pre-submission readiness checker. Runs three checks:

  1. Artifact completeness  — verifies every file path referenced in
     THESIS_EVIDENCE_INDEX.md actually exists on disk.
  2. RPi4-PENDING scan      — lists all % RPi4-PENDING comments in .tex files
     so authors know exactly what to update when hardware data is available.
  3. UNVERIFIED parameter count — summarises hand-tuned parameters for Appendix A.

Usage:
    python tools/check_thesis_readiness.py

Output is printed to stdout. No files are modified.
"""

import os, re, subprocess, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(BASE, 'Thesis_report', 'THESIS_EVIDENCE_INDEX.md')
CHAPTERS_DIR = os.path.join(BASE, 'Thesis_report', 'chapters')
PIPELINE_REG = os.path.join(BASE, 'docs', 'PIPELINE_EVIDENCE_REGISTER.md')

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

# ─────────────────────────────────────────────────────────────────────────────
# Check 1: Artifact completeness
# ─────────────────────────────────────────────────────────────────────────────

def check_artifacts():
    print("=" * 65)
    print("CHECK 1 — Artifact Completeness (THESIS_EVIDENCE_INDEX.md)")
    print("=" * 65)

    with open(INDEX_PATH) as f:
        text = f.read()

    raw = re.findall(r'`([^`]+)`', text)
    paths = set()
    for p in raw:
        p = p.strip()
        if p.startswith(('docs/', 'models/', 'tests/', 'Thesis_report/figures/', 'data/')):
            if any(p.endswith(ext) for ext in ('.csv', '.md', '.json', '.png', '.py', '.joblib')):
                paths.add(p)

    ok, missing = [], []
    for p in sorted(paths):
        full = os.path.join(BASE, p)
        if os.path.exists(full):
            ok.append(p)
        else:
            missing.append(p)

    print(f"  Checked : {len(paths)} unique paths")
    print(f"  {PASS} Exists: {len(ok)}")
    print(f"  {FAIL} Missing: {len(missing)}")
    if missing:
        print()
        print("  Missing artifacts — action needed:")
        for p in missing:
            if 'session_index' in p or 'iqa_host_delta' in p:
                tag = "[RPi4-PENDING]"
            elif 'kalman_qr' in p or '3dnr_alpha' in p:
                tag = "[THESIS_SKIP — needs thermal .npy sequences]"
            else:
                tag = "[DATA_PENDING]"
            print(f"    {FAIL}  {p}  {tag}")
    return len(missing)


# ─────────────────────────────────────────────────────────────────────────────
# Check 2: RPi4-PENDING scan
# ─────────────────────────────────────────────────────────────────────────────

def check_rpi4_pending():
    print()
    print("=" * 65)
    print("CHECK 2 — RPi4-PENDING Comments in .tex Files")
    print("=" * 65)

    result = subprocess.run(
        ['grep', '-rn', 'RPi4-PENDING', CHAPTERS_DIR],
        capture_output=True, text=True
    )

    lines = [l for l in result.stdout.splitlines() if l.strip()]
    print(f"  Total % RPi4-PENDING comments: {len(lines)}")

    by_file = {}
    for line in lines:
        parts = line.split(':', 2)
        if len(parts) >= 3:
            fpath = os.path.basename(parts[0])
            linenum = parts[1]
            content = parts[2].strip().lstrip('%').strip()
            by_file.setdefault(fpath, []).append((linenum, content))

    for fpath, items in sorted(by_file.items()):
        print(f"\n  📄 {fpath} ({len(items)} pending item(s)):")
        for linenum, content in items:
            preview = content[:90] + ('…' if len(content) > 90 else '')
            print(f"     L{linenum}: {preview}")

    return len(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Check 3: UNVERIFIED parameters
# ─────────────────────────────────────────────────────────────────────────────

def check_unverified():
    print()
    print("=" * 65)
    print("CHECK 3 — UNVERIFIED Parameters (for Appendix A honesty)")
    print("=" * 65)

    if not os.path.exists(PIPELINE_REG):
        print(f"  {WARN} Pipeline evidence register not found — skipping")
        return 0

    with open(PIPELINE_REG) as f:
        text = f.read()

    count = text.count('UNVERIFIED')
    print(f"  Total 'UNVERIFIED' mentions in evidence register: {count}")
    print()

    # Core list (hardcoded from §D — kept stable for thesis)
    KNOWN_UNVERIFIED = [
        ("ML posterior EMA α (general)",         "0.55",          "D.2"),
        ("ML posterior EMA α glare (up)",         "0.85",          "D.2"),
        ("ML posterior EMA α glare (down)",       "0.45",          "D.2"),
        ("3D-NR EMA α (ThermalTemporalFilter)",   "0.65",          "D.2"),
        ("Glare IIR prev_weight",                 "0.42",          "D.4"),
        ("Kalman process noise Q",                "0.5",           "D.7"),
        ("Kalman measurement noise R",            "4.0",           "D.7"),
        ("Kalman initial covariance P₀",          "100.0",         "D.7"),
        ("MAD temporal window",                   "3 frames",      "D.9"),
        ("JerkGate diff_threshold",               "8.5",           "D.9"),
        ("ENV FSM hysteresis N_confirm",          "3 frames",      "D.10"),
        ("Fusion alpha base (NIR weight)",        "0.55",          "D.3"),
        ("Display L-cap",                         "220",           "D.4"),
        ("OneEuro min_cutoff (shake reducer)",    "1.15",          "D.9"),
    ]

    print(f"  {'Parameter':<42} {'Value':<14} {'§Evidence'}")
    print(f"  {'-'*42} {'-'*14} {'-'*10}")
    for name, val, sec in KNOWN_UNVERIFIED:
        print(f"  {name:<42} {val:<14} §{sec}")

    print(f"\n  Total UNVERIFIED parameters listed: {len(KNOWN_UNVERIFIED)}")
    print(f"  → These MUST be written as \"hand-tuned, no sweep conducted\" in report")
    return len(KNOWN_UNVERIFIED)


# ─────────────────────────────────────────────────────────────────────────────
# Check 4: "Mac" or specific product names in visible LaTeX text
# ─────────────────────────────────────────────────────────────────────────────

def check_naming():
    print()
    print("=" * 65)
    print("CHECK 4 — Naming Convention (Mac / Azure / product names)")
    print("=" * 65)

    issues = []
    for root, _, files in os.walk(CHAPTERS_DIR):
        for fname in files:
            if not fname.endswith('.tex'):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath) as f:
                lines = f.readlines()
            for i, line in enumerate(lines, 1):
                # strip LaTeX comments
                visible = re.sub(r'%.*', '', line)
                if re.search(r'\bMac\b(?! OS|book)', visible):
                    issues.append((fname, i, "bare 'Mac' → use 'development workstation'", visible.strip()))
                if re.search(r'Azure IoT Hub|AWS IoT|Google Cloud IoT', visible, re.I):
                    issues.append((fname, i, "specific IoT product name → use 'cloud-based IoT platform'", visible.strip()))

    if issues:
        print(f"  {WARN} Found {len(issues)} naming issue(s):")
        for fname, linenum, msg, snippet in issues:
            print(f"    {fname}:L{linenum} — {msg}")
            print(f"      ↳ {snippet[:80]}")
    else:
        print(f"  {PASS} No naming convention violations found")

    return len(issues)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    missing   = check_artifacts()
    pending   = check_rpi4_pending()
    unverif   = check_unverified()
    naming    = check_naming()

    print()
    print("=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  {FAIL if missing  else PASS} Artifact gaps:          {missing} missing files")
    print(f"  {WARN}             RPi4-PENDING items:      {pending} comments in .tex")
    print(f"  {WARN}             UNVERIFIED parameters:   {unverif} must be disclosed")
    print(f"  {FAIL if naming   else PASS} Naming violations:      {naming}")
    print()

    if missing == 0 and naming == 0:
        print("  Ready for chapter writing — all available artifacts present.")
    else:
        print("  Fix naming violations and generate missing artifacts before submitting.")

    sys.exit(0 if (missing == 0 and naming == 0) else 1)
