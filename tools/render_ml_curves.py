"""render_ml_curves.py — Generate ML classifier figures for the thesis.

Produces six figures and a per-class summary CSV from existing artifacts:
  1. confusion_matrix.png         — at τ=0.62 on held-out test set
  2. roc_per_night_class.png      — OVR ROC for all 9 classes (night classes highlighted)
  3. pr_per_night_class.png       — OVR Precision-Recall for all 9 classes
  4. f1_tau_abstention.png        — macro-night-F1 vs τ (from threshold_sweep.csv) +
                                     dual-axis abstention rate
  5. learning_curve.png           — macro-night-F1 vs training fraction (5 points)
  6. feature_importance.png       — MDI importances for 12 optical features
  7. per_class_summary.csv        — per-class F1, precision, recall, ECE at τ=0.62

Usage:
    python tools/render_ml_curves.py
    python tools/render_ml_curves.py \\
        --bundle   models/production/env_classifier.joblib \\
        --sidecar  models/production/env_classifier.json \\
        --test-jsonl  data/training/from_logs_test.jsonl \\
        --train-jsonl data/training/merged_logs_ml.jsonl \\
        --sweep-csv   docs/tables/ml/threshold_sweep.csv \\
        --out-dir  docs/thesis_eval/ml_classifier/figures/

Data provenance:
    • Bundle:  models/production/env_classifier.joblib
    • Sidecar: models/production/env_classifier.json
    • Test:    data/training/from_logs_test.jsonl  (2113 rows)
    • Train:   data/training/merged_logs_ml.jsonl  (for learning curve)
    • Sweep:   docs/tables/ml/threshold_sweep.csv  (pre-computed)

Caption note appended to every figure title:
    "IQA proxy — optical test set; domain shift to NIR deployment not validated."
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from smartbinocular.feature_schema import (
    ENV_CLASS_TO_INT,
    ENV_CLASSES,
    ENV_INT_TO_CLASS,
    FEATURE_SET_OPTICAL_ONLY,
    FeatureRecord,
)

CAPTION = "Optical test set; domain shift to NIR deployment not validated."
TAU_PROD = 0.62

NIGHT_CLASSES = ["night_clear", "normal_night", "nir_night"]
ALL_CLASSES = ENV_CLASSES  # 9 classes

# ── Matplotlib import (lazy, with graceful degradation) ──────────────────────

def _import_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("ERROR: matplotlib not installed. Install with: pip install matplotlib", file=sys.stderr)
        sys.exit(1)


# ── Bundle / data loading (mirrors threshold_sweep.py exactly) ───────────────

def _load_apply_scalers():
    path = ROOT / "models" / "train_classifier.py"
    spec = importlib.util.spec_from_file_location("sb_train_classifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.apply_scalers


def _load_dataset(path: Path):
    """Return (X float32 [N,12], y_true int32 [N], nir_groups str [N])."""
    xs, ys, groups = [], [], []
    skipped = 0
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = FeatureRecord.from_dict(json.loads(line))
            label_str = rec.effective_label()
            if label_str is None or not rec.is_compatible_with(FEATURE_SET_OPTICAL_ONLY):
                skipped += 1
                continue
            label_int = ENV_CLASS_TO_INT.get(label_str)
            if label_int is None:
                skipped += 1
                continue
            xs.append(rec.to_feature_array(FEATURE_SET_OPTICAL_ONLY))
            ys.append(label_int)
            groups.append(str(rec.nir_channel))
    print(f"  Loaded {len(xs)} records ({skipped} skipped) from {path.name}")
    if not xs:
        raise ValueError(f"No compatible labeled rows in {path}")
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.int32), np.array(groups, dtype=object)


def _get_proba(rf, scalers, normalize_by, apply_scalers_fn, X, nir_groups):
    X_scaled = apply_scalers_fn(X, nir_groups, scalers, normalize_by)
    return rf.predict_proba(X_scaled)  # [N, n_classes]


# ── Figure 1: Confusion matrix ───────────────────────────────────────────────

def plot_confusion_matrix(
    y_true, y_pred_accepted, accepted_mask, out_dir: Path, plt
) -> None:
    from sklearn.metrics import confusion_matrix  # type: ignore

    present_ints = sorted(set(y_true[accepted_mask]) | set(y_pred_accepted))
    labels = [ENV_INT_TO_CLASS.get(i, str(i)) for i in present_ints]
    cm = confusion_matrix(y_true[accepted_mask], y_pred_accepted, labels=present_ints)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    n_acc = int(accepted_mask.sum())
    n_total = len(y_true)
    ax.set_title(
        f"Confusion matrix at τ={TAU_PROD} (n={n_acc}/{n_total} accepted)\n"
        f"({CAPTION})",
        fontsize=9,
    )
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm_norm[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if val > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = out_dir / "confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 2 & 3: ROC and PR curves ─────────────────────────────────────────

def _ovr_roc_pr(y_true, proba, classes, class_int):
    """Return (fpr, tpr, auc_roc), (precision, recall, auc_pr) for one class OVR."""
    from sklearn.metrics import auc, precision_recall_curve, roc_curve  # type: ignore

    y_bin = (y_true == class_int).astype(int)
    col = list(classes).index(class_int)
    scores = proba[:, col]
    fpr, tpr, _ = roc_curve(y_bin, scores)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_bin, scores)
    pr_auc = auc(rec, prec)
    return (fpr, tpr, roc_auc), (prec, rec, pr_auc)


def plot_roc_pr(y_true, proba, classes: np.ndarray, out_dir: Path, plt) -> None:
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))

    for cls_name in ALL_CLASSES:
        cls_int = ENV_CLASS_TO_INT.get(cls_name)
        if cls_int not in classes:
            continue
        is_night = cls_name in NIGHT_CLASSES
        lw = 2.5 if is_night else 1.0
        ls = "-" if is_night else "--"
        alpha = 1.0 if is_night else 0.5

        (fpr, tpr, roc_auc), (prec, rec, pr_auc) = _ovr_roc_pr(y_true, proba, classes, cls_int)
        label_roc = f"{cls_name} (AUC={roc_auc:.3f})"
        label_pr = f"{cls_name} (AP={pr_auc:.3f})"
        ax_roc.plot(fpr, tpr, lw=lw, ls=ls, alpha=alpha, label=label_roc)
        ax_pr.plot(rec, prec, lw=lw, ls=ls, alpha=alpha, label=label_pr)

    ax_roc.plot([0, 1], [0, 1], "k:", lw=1)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title(f"ROC curves (OVR per class)\n({CAPTION})", fontsize=9)
    ax_roc.legend(fontsize=7, loc="lower right")
    fig_roc.tight_layout()
    out_roc = out_dir / "roc_per_night_class.png"
    fig_roc.savefig(out_roc, dpi=150)
    plt.close(fig_roc)
    print(f"  Saved: {out_roc}")

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"Precision-recall curves (OVR per class)\n({CAPTION})", fontsize=9)
    ax_pr.legend(fontsize=7, loc="lower left")
    fig_pr.tight_layout()
    out_pr = out_dir / "pr_per_night_class.png"
    fig_pr.savefig(out_pr, dpi=150)
    plt.close(fig_pr)
    print(f"  Saved: {out_pr}")


# ── Figure 4: F1 vs τ vs abstention (from threshold_sweep.csv) ──────────────

def plot_f1_tau_abstention(sweep_csv: Path, out_dir: Path, plt) -> None:
    rows = list(csv.DictReader(sweep_csv.open()))
    taus = [float(r["threshold"]) for r in rows]
    f1_nc = [float(r["f1_night_clear"]) for r in rows]
    f1_nn = [float(r["f1_normal_night"]) for r in rows]
    f1_nir = [float(r["f1_nir_night"]) for r in rows]
    macro = [float(r["macro_f1_night"]) for r in rows]
    abstain = [float(r["abstention_rate"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(taus, macro, "k-o", lw=2, ms=5, label="Macro F1 (night)")
    ax1.plot(taus, f1_nc, "b-^", lw=1.5, ms=4, alpha=0.8, label="F1 night_clear")
    ax1.plot(taus, f1_nn, "g-s", lw=1.5, ms=4, alpha=0.8, label="F1 normal_night")
    ax1.plot(taus, f1_nir, "r-D", lw=1.5, ms=4, alpha=0.8, label="F1 nir_night")
    ax1.axvline(TAU_PROD, color="grey", ls="--", lw=1.2, label=f"τ={TAU_PROD} (production)")
    ax1.set_xlabel("Confidence threshold τ")
    ax1.set_ylabel("F1 score")
    ax1.set_ylim(0.88, 1.01)
    ax1.legend(fontsize=8, loc="lower left")

    ax2.plot(taus, abstain, "m-x", lw=1.5, ms=6, alpha=0.7, label="Abstention rate")
    ax2.set_ylabel("Abstention rate", color="m")
    ax2.tick_params(axis="y", labelcolor="m")
    ax2.set_ylim(0, 0.35)
    ax2.legend(fontsize=8, loc="lower right")

    ax1.set_title(
        f"Night-class F1 vs confidence threshold (abstention dual-axis)\n({CAPTION})",
        fontsize=9,
    )
    fig.tight_layout()
    out = out_dir / "f1_tau_abstention.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 5: Learning curve ─────────────────────────────────────────────────

def plot_learning_curve(
    rf_orig, scalers, normalize_by, apply_scalers_fn,
    X_train, y_train, nir_train,
    X_test, y_test, nir_test,
    classes, out_dir: Path, plt,
) -> None:
    """Train RF on fractions of training data; report macro-night-F1 on fixed test set."""
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
    from sklearn.metrics import f1_score  # type: ignore

    NIGHT_INTS = [ENV_CLASS_TO_INT[c] for c in NIGHT_CLASSES if c in ENV_CLASS_TO_INT]
    fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
    f1_means, f1_stds = [], []
    n_train = len(X_train)

    X_test_scaled = apply_scalers_fn(X_test, nir_test, scalers, normalize_by)

    for frac in fractions:
        n = max(len(NIGHT_INTS) * 3, int(frac * n_train))
        # stratified subsample
        rng = np.random.default_rng(42)
        indices = []
        for cls in np.unique(y_train):
            cls_idx = np.where(y_train == cls)[0]
            k = max(1, int(round(frac * len(cls_idx))))
            chosen = rng.choice(cls_idx, size=min(k, len(cls_idx)), replace=False)
            indices.extend(chosen.tolist())
        idx = np.array(indices)
        X_sub = X_train[idx]
        y_sub = y_train[idx]
        grp_sub = nir_train[idx]
        X_sub_scaled = apply_scalers_fn(X_sub, grp_sub, scalers, normalize_by)

        # rf_orig may be CalibratedClassifierCV; unwrap to get the base RF params
        base_rf = getattr(rf_orig, "estimator", rf_orig)
        n_est = getattr(base_rf, "n_estimators", 100)
        max_d = getattr(base_rf, "max_depth", None)

        run_f1s = []
        for seed in (0, 1, 2):
            clf = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=max_d,
                random_state=seed,
                n_jobs=-1,
            )
            clf.fit(X_sub_scaled, y_sub)
            y_pred = np.array(clf.classes_)[clf.predict_proba(X_test_scaled).argmax(axis=1)]
            night_f1s = []
            for ci in NIGHT_INTS:
                if ci not in clf.classes_:
                    continue
                tp = int(np.sum((y_pred == ci) & (y_test == ci)))
                fp = int(np.sum((y_pred == ci) & (y_test != ci)))
                fn = int(np.sum((y_pred != ci) & (y_test == ci)))
                if tp == 0:
                    night_f1s.append(0.0)
                    continue
                p = tp / (tp + fp)
                r = tp / (tp + fn)
                night_f1s.append(2 * p * r / (p + r))
            run_f1s.append(float(np.mean(night_f1s)) if night_f1s else 0.0)

        f1_means.append(float(np.mean(run_f1s)))
        f1_stds.append(float(np.std(run_f1s)))
        print(f"    fraction={frac:.2f}  n={n}  macro_f1={f1_means[-1]:.4f}±{f1_stds[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ns = [int(f * n_train) for f in fractions]
    ax.errorbar(ns, f1_means, yerr=f1_stds, fmt="-o", capsize=4, lw=2, ms=5)
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Macro-night F1 (test set)")
    ax.set_ylim(0.8, 1.01)
    ax.set_title(
        f"Learning curve — macro-night F1 vs training fraction\n"
        f"(3-seed runs; pre-fitted scalers; {CAPTION})",
        fontsize=9,
    )
    fig.tight_layout()
    out = out_dir / "learning_curve.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 6: Feature importance ─────────────────────────────────────────────

def plot_feature_importance(sidecar: dict, out_dir: Path, plt) -> None:
    fi = sidecar.get("feature_importances", {})
    if not fi:
        print("  WARNING: no feature_importances in sidecar, skipping figure")
        return
    names = list(fi.keys())
    vals = [fi[n] for n in names]
    order = np.argsort(vals)[::-1]
    names_s = [names[i] for i in order]
    vals_s = [vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2c7bb6" if n in ["nir_mean_brightness", "nir_sharpness", "nir_saturation_mean", "nir_dark_fraction", "nir_blue_mean_ema"]
              else "#d7191c" for n in names_s]
    ax.barh(range(len(names_s)), vals_s[::-1], color=colors[::-1])
    ax.set_yticks(range(len(names_s)))
    ax.set_yticklabels(names_s[::-1], fontsize=9)
    ax.set_xlabel("Mean decrease in impurity (MDI)")
    ax.set_title(
        "RF feature importance — 12 optical features (MDI)\n"
        "(nir_saturation_mean is highest; hour/prev_env_class are zero)",
        fontsize=9,
    )
    fig.tight_layout()
    out = out_dir / "feature_importance.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Table: per-class summary ─────────────────────────────────────────────────

def write_per_class_summary(
    y_true, y_pred_all, accepted_mask, classes, sidecar: dict, out_path: Path
) -> None:
    from sklearn.metrics import precision_recall_fscore_support  # type: ignore

    y_t = y_true[accepted_mask]
    y_p = y_pred_all[accepted_mask]
    ece_all = sidecar.get("ece_by_class", {})

    rows = []
    for cls_name in ALL_CLASSES:
        cls_int = ENV_CLASS_TO_INT.get(cls_name)
        if cls_int is None or cls_int not in classes:
            continue
        mask_true = y_t == cls_int
        n_true = int(mask_true.sum())
        if n_true == 0:
            continue
        tp = int(np.sum((y_p == cls_int) & (y_t == cls_int)))
        fp = int(np.sum((y_p == cls_int) & (y_t != cls_int)))
        fn = int(np.sum((y_p != cls_int) & (y_t == cls_int)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        ece = ece_all.get(cls_name, float("nan"))
        rows.append({
            "env_class": cls_name,
            "n_true": n_true,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "ece": round(ece, 5) if not (isinstance(ece, float) and np.isnan(ece)) else "nan",
            "is_night_class": cls_name in NIGHT_CLASSES,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["env_class", "n_true", "precision", "recall", "f1", "ece", "is_night_class"]
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path, default=ROOT / "models/production/env_classifier.joblib")
    p.add_argument("--sidecar", type=Path, default=ROOT / "models/production/env_classifier.json")
    p.add_argument("--test-jsonl", type=Path, default=ROOT / "data/training/from_logs_test.jsonl")
    p.add_argument("--train-jsonl", type=Path, default=ROOT / "data/training/merged_logs_ml.jsonl")
    p.add_argument("--sweep-csv", type=Path, default=ROOT / "docs/tables/ml/threshold_sweep.csv")
    p.add_argument("--out-dir", type=Path, default=ROOT / "docs/thesis_eval/ml_classifier/figures")
    p.add_argument("--skip-learning-curve", action="store_true",
                   help="Skip the learning curve (slow, ~30s)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    plt = _import_mpl()

    # ── Validate inputs ──────────────────────────────────────────────────────
    missing = [p for p in (args.bundle, args.sidecar, args.test_jsonl, args.sweep_csv) if not p.exists()]
    if missing:
        print("ERROR: missing required inputs:")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = args.out_dir.parent / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # ── Load sidecar ────────────────────────────────────────────────────────
    sidecar = json.loads(args.sidecar.read_text())
    print(f"Sidecar loaded: {args.sidecar.name}")

    # ── Load bundle ─────────────────────────────────────────────────────────
    apply_scalers_fn = _load_apply_scalers()
    bundle = joblib.load(args.bundle)
    rf = bundle["rf"]
    scalers = bundle["scalers"]
    normalize_by = str(bundle.get("normalize_by", "nir_channel"))
    classes = np.array(rf.classes_)
    print(f"Bundle loaded: {args.bundle.name}  classes={classes}")

    # ── Load test set ────────────────────────────────────────────────────────
    print("Loading test set...")
    X_test, y_test, nir_test = _load_dataset(args.test_jsonl)

    # ── Get predictions ──────────────────────────────────────────────────────
    print("Running predict_proba on test set...")
    proba = _get_proba(rf, scalers, normalize_by, apply_scalers_fn, X_test, nir_test)
    conf = proba.max(axis=1)
    pred_idx = proba.argmax(axis=1)
    y_pred_all = classes[pred_idx]
    accepted_mask = conf >= TAU_PROD
    y_pred_accepted = y_pred_all[accepted_mask]
    print(f"  Accepted at τ={TAU_PROD}: {accepted_mask.sum()}/{len(y_test)}")

    # ── Figure 1: Confusion matrix ───────────────────────────────────────────
    print("Plotting confusion matrix...")
    plot_confusion_matrix(y_test, y_pred_accepted, accepted_mask, args.out_dir, plt)

    # ── Figures 2 & 3: ROC / PR ──────────────────────────────────────────────
    print("Plotting ROC and PR curves...")
    plot_roc_pr(y_test, proba, classes, args.out_dir, plt)

    # ── Figure 4: F1 vs τ ────────────────────────────────────────────────────
    print("Plotting F1 vs τ vs abstention...")
    plot_f1_tau_abstention(args.sweep_csv, args.out_dir, plt)

    # ── Figure 5: Learning curve ─────────────────────────────────────────────
    if not args.skip_learning_curve and args.train_jsonl.exists():
        print("Computing learning curve (may take ~30s)...")
        X_train, y_train, nir_train = _load_dataset(args.train_jsonl)
        plot_learning_curve(
            rf, scalers, normalize_by, apply_scalers_fn,
            X_train, y_train, nir_train,
            X_test, y_test, nir_test,
            classes, args.out_dir, plt,
        )
    else:
        reason = "--skip-learning-curve" if args.skip_learning_curve else f"{args.train_jsonl} not found"
        print(f"  Skipping learning curve ({reason})")

    # ── Figure 6: Feature importance ─────────────────────────────────────────
    print("Plotting feature importance...")
    plot_feature_importance(sidecar, args.out_dir, plt)

    # ── Table: per-class summary ─────────────────────────────────────────────
    print("Writing per-class summary CSV...")
    write_per_class_summary(
        y_test, y_pred_all, accepted_mask, classes, sidecar,
        tables_dir / "per_class_summary.csv",
    )

    print(f"\nDone. All outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
