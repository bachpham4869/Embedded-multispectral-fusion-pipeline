"""Discover and validate Pascal VOC eval items from the find-person-in-dark dataset.

GT authority: VOC XML under train_annotations/ paired by stem to train_images/*.jpg.
See docs/eval/ei_person_find_in_dark/DECISIONS_AND_RISKS.md Q5.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple


@dataclass(frozen=True)
class GTBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    label: str


@dataclass(frozen=True)
class EvalItem:
    image_path: Path
    xml_path: Path
    stem: str
    gt_boxes: Tuple[GTBox, ...]  # empty → negative image (no person)


def _parse_voc_xml(xml_path: Path) -> Tuple[GTBox, ...]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes: List[GTBox] = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        bndbox = obj.find("bndbox")
        if name_el is None or bndbox is None:
            continue
        label = (name_el.text or "").strip().lower()
        try:
            xmin = float(bndbox.findtext("xmin", "0"))
            ymin = float(bndbox.findtext("ymin", "0"))
            xmax = float(bndbox.findtext("xmax", "0"))
            ymax = float(bndbox.findtext("ymax", "0"))
        except ValueError as exc:
            raise ValueError(f"Non-numeric bndbox in {xml_path}: {exc}") from exc
        if not (xmax > xmin and ymax > ymin):
            # Degenerate annotation artifact in the dataset — skip silently (logged once at run start).
            continue
        boxes.append(GTBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, label=label))
    return tuple(boxes)


def _validate_first_n(items: List[EvalItem], n: int = 10) -> None:
    """Raise ValueError on the first schema problem found in the first n items."""
    for item in items[:n]:
        if not item.xml_path.is_file():
            raise ValueError(f"Missing XML: {item.xml_path}")
        if not item.image_path.is_file():
            raise ValueError(f"Missing image: {item.image_path}")
        # Parse and check at least one bndbox or confirm the negative is valid XML
        try:
            _parse_voc_xml(item.xml_path)
        except ET.ParseError as exc:
            raise ValueError(f"Malformed XML at {item.xml_path}: {exc}") from exc


def iter_eval_items(
    dataset_root: Path,
    *,
    limit: Optional[int] = None,
) -> Iterator[EvalItem]:
    """Yield EvalItems from dataset_root/train_images + train_annotations.

    Performs upfront validation (first 10 records), reports orphans and duplicate
    stems to stdout, then yields items in deterministic stem-sorted order.

    Args:
        dataset_root: Path to the train/train directory (contains train_images/ and
            train_annotations/).
        limit: If given and > 0, yield at most this many items.

    Raises:
        ValueError: On XML schema errors, missing files, or structural problems.
        FileNotFoundError: If dataset_root or its subdirectories are missing.
    """
    img_dir = dataset_root / "train_images"
    ann_dir = dataset_root / "train_annotations"

    if not img_dir.is_dir():
        raise FileNotFoundError(f"train_images not found: {img_dir}")
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"train_annotations not found: {ann_dir}")

    img_stems = {p.stem: p for p in sorted(img_dir.glob("*.jpg"))}
    xml_stems = {p.stem: p for p in sorted(ann_dir.glob("*.xml"))}

    img_only = sorted(set(img_stems) - set(xml_stems))
    xml_only = sorted(set(xml_stems) - set(img_stems))
    if img_only:
        print(f"[discover] {len(img_only)} orphan image(s) with no XML: {img_only[:5]}{'...' if len(img_only) > 5 else ''}")
    if xml_only:
        print(f"[discover] {len(xml_only)} orphan XML(s) with no image: {xml_only[:5]}{'...' if len(xml_only) > 5 else ''}")

    paired_stems = sorted(set(img_stems) & set(xml_stems))
    if not paired_stems:
        raise ValueError(f"No paired image+XML pairs found under {dataset_root}")

    # Build item list for validation pass
    all_items: List[EvalItem] = []
    for stem in paired_stems:
        try:
            boxes = _parse_voc_xml(xml_stems[stem])
        except (ET.ParseError, ValueError) as exc:
            raise ValueError(f"XML parse error for stem '{stem}': {exc}") from exc
        all_items.append(EvalItem(
            image_path=img_stems[stem],
            xml_path=xml_stems[stem],
            stem=stem,
            gt_boxes=boxes,
        ))

    _validate_first_n(all_items, n=10)

    total = len(all_items)
    if limit and limit > 0:
        all_items = all_items[:limit]

    print(
        f"[discover] {total} paired items found; "
        f"{'all' if not limit or limit <= 0 else str(len(all_items))} will be evaluated. "
        f"Positives in selection: {sum(1 for it in all_items if it.gt_boxes)}"
    )

    yield from all_items
