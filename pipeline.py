"""
Chocolathon end-to-end pipeline.

One image in, chocolates out.

    python pipeline.py --image path/to/box.jpg --out outputs/run1

Optional overrides:
    --corners corners.json     # 4 corners you specify manually (see --dump-template)
    --box-size 16              # force a layout, skip auto-search
    --device cpu|cuda

Outputs (in --out):
    result.json          full structured result
    summary.txt          human-readable chocolate list
    tray.png             rectified top-down tray
    predictions.png      slot montage with predictions
    box_overlay.png      original with detected tray outline

Uses only:
    data/experiments/synthetic_slot_embeddings_v1/deployment/*
Classical OpenCV does localization; EfficientNet-B0 + prototypes do classification.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

LAYOUTS = {6: (3, 2), 16: (4, 4), 30: (6, 5), 50: (10, 5)}
CORNER_NAMES = ("top_left", "top_right", "bottom_right", "bottom_left")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# =============================================================================
# Deployment package (classifier)
# =============================================================================
class Deployment:
    def __init__(self, root: Path, device: str = "cpu", threads: int = 4):
        self.root = root.resolve()
        self.device = torch.device(device)
        torch.set_num_threads(threads)

        dep = self.root / "data/experiments/synthetic_slot_embeddings_v1/deployment"
        self.config = json.loads((dep / "deployment_config.json").read_text())
        if self.config["backbone"] != "efficientnet_b0":
            raise ValueError("Unsupported backbone.")

        self.threshold = float(self.config["provisional_margin_threshold"])
        self.empty_id = self.config["empty_class_id"]

        with (dep / self.config["class_catalog_file"]).open(newline="") as f:
            self.catalog = {r["flavor_id"]: r for r in csv.DictReader(f)}

        with np.load(dep / self.config["prototype_file"], allow_pickle=False) as data:
            ids = data["class_ids"].astype(str)
            prototypes = data["prototypes"].copy()

        if prototypes.shape != (int(self.config["classes"]), 1280):
            raise ValueError("Bad prototype shape.")
        if not np.allclose(np.linalg.norm(prototypes, axis=1), 1.0, atol=1e-5):
            raise ValueError("Prototypes must be L2-normalized.")
        if set(ids.tolist()) != set(self.catalog):
            raise ValueError("Prototype/catalog mismatch.")

        self.ids = ids.tolist()
        self.empty_idx = self.ids.index(self.empty_id)
        self.prototypes = torch.as_tensor(prototypes, dtype=torch.float32, device=self.device)

        backbone = efficientnet_b0(weights=None)
        backbone.classifier = nn.Identity()
        backbone.load_state_dict(
            torch.load(dep / self.config["local_backbone_file"],
                       map_location="cpu", weights_only=True),
            strict=True,
        )
        backbone.to(self.device).eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.backbone = backbone
        self.transform = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()

    @torch.inference_mode()
    def classify(self, crops: list[np.ndarray], batch_size: int = 32) -> list[dict]:
        out: list[dict] = []
        for start in range(0, len(crops), batch_size):
            batch = torch.stack(
                [self.transform(Image.fromarray(c)) for c in crops[start:start + batch_size]]
            ).to(self.device)
            scores = (F.normalize(self.backbone(batch), dim=1) @ self.prototypes.T).cpu().numpy()
            if not np.isfinite(scores).all():
                raise ValueError("Non-finite scores.")
            for row in scores:
                rank = np.argsort(-row, kind="stable")
                top1, top2 = int(rank[0]), int(rank[1])
                fid = self.ids[top1]
                occ_max = max(float(row[i]) for i in range(len(self.ids)) if i != self.empty_idx)
                out.append({
                    "flavor_id": fid,
                    "flavor_name": self.catalog[fid]["flavor_name"],
                    "label_type": "empty" if fid == self.empty_id else "occupied",
                    "similarity": float(row[top1]),
                    "margin": float(row[top1] - row[top2]),
                    "needs_review": bool(row[top1] - row[top2] < self.threshold),
                    "occupancy_score_gap": occ_max - float(row[self.empty_idx]),
                })
        return out


# =============================================================================
# Localization (classical CV)
# =============================================================================
def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Cannot decode image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    total, diff = pts.sum(1), np.diff(pts, axis=1).ravel()
    ordered = pts[[np.argmin(total), np.argmin(diff), np.argmax(total), np.argmax(diff)]]
    if not np.isfinite(ordered).all() or len(np.unique(ordered, axis=0)) != 4:
        raise ValueError("Degenerate corners.")
    if not cv2.isContourConvex(ordered):
        raise ValueError("Corners not convex.")
    return ordered


def detect_tray(rgb: np.ndarray) -> np.ndarray:
    """Return 4 tray corners [TL, TR, BR, BL]. Raises on failure."""
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    candidates: list[np.ndarray] = []

    # --- Strategy 1: Canny + close ---
    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    candidates.extend(_quad_from_edges(edges, w * h))

    # --- Strategy 2: Adaptive threshold, invert, close ---
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 51, 10)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)
    candidates.extend(_quad_from_edges(th, w * h))

    # --- Strategy 3: Otsu on saturation (trays are often darker or desaturated) ---
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    _, s_bin = cv2.threshold(hsv[..., 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    s_bin = cv2.morphologyEx(s_bin, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)
    candidates.extend(_quad_from_edges(s_bin, w * h))

    if not candidates:
        raise RuntimeError(
            "Auto tray detection failed. Provide --corners corners.json "
            "(run --dump-template for the format)."
        )

    # Pick the candidate whose area is largest and whose shape is closest to a rectangle.
    def score(q: np.ndarray) -> float:
        area = cv2.contourArea(q)
        if area < 0.05 * w * h:
            return -1.0
        # prefer near-rectangular quads
        rect = cv2.minAreaRect(q.astype(np.float32))
        rect_area = rect[1][0] * rect[1][1] + 1e-6
        return area * min(1.0, rect_area / max(area, 1.0))

    best = max(candidates, key=score)
    if score(best) <= 0:
        raise RuntimeError("No plausible tray found; provide --corners manually.")
    return order_corners(best)


def _quad_from_edges(binary: np.ndarray, img_area: int) -> list[np.ndarray]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    quads: list[np.ndarray] = []
    for c in contours:
        if cv2.contourArea(c) < 0.05 * img_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quads.append(approx.reshape(4, 2).astype(np.float32))
    return quads


def warp_tray(rgb: np.ndarray, corners: np.ndarray, rows: int, cols: int, cell: int) -> np.ndarray:
    w, h = cols * cell, rows * cell
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
    return cv2.warpPerspective(rgb, M, (w, h))


def slice_slots(tray: np.ndarray, rows: int, cols: int, cell: int = 224) -> list[np.ndarray]:
    expected_h, expected_w = rows * cell, cols * cell
    if tray.shape[:2] != (expected_h, expected_w):
        tray = cv2.resize(tray, (expected_w, expected_h), interpolation=cv2.INTER_AREA)
    return [
        tray[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell]
        for r in range(rows) for c in range(cols)
    ]


# =============================================================================
# Layout selection
# =============================================================================
def score_layout(engine: Deployment, crops: list[dict]) -> float:
    """Score a layout by mean confidence of its non-empty slots.

    Rationale: when a layout misaligns, slots straddle boundaries and the
    classifier becomes uncertain. Wrong layouts therefore score lower.
    """
    occupied = [r for r in crops if r["label_type"] == "occupied"]
    if not occupied:
        return float("-inf")
    mean_sim = float(np.mean([r["similarity"] for r in occupied]))
    mean_margin = float(np.mean([r["margin"] for r in occupied]))
    # Penalize layouts whose slots are barely ahead of the runner-up.
    return mean_sim + 0.5 * mean_margin


def choose_layout(
    engine: Deployment,
    rgb: np.ndarray,
    corners: np.ndarray,
    forced_size: int | None = None,
) -> tuple[int, list[dict], dict]:
    """Return (box_size, slot records, per-size scores)."""
    if forced_size is not None:
        if forced_size not in LAYOUTS:
            raise ValueError(f"Unsupported box size: {forced_size}")
        rows, cols = LAYOUTS[forced_size]
        tray = warp_tray(rgb, corners, rows, cols, 224)
        crops = slice_slots(tray, rows, cols)
        return forced_size, engine.classify(crops), {str(forced_size): None}

    scores: dict[str, float] = {}
    best_size = None
    best_records: list[dict] = []
    best_score = float("-inf")

    for size, (rows, cols) in LAYOUTS.items():
        tray = warp_tray(rgb, corners, rows, cols, 224)
        crops = slice_slots(tray, rows, cols)
        records = engine.classify(crops)
        s = score_layout(engine, records)
        scores[str(size)] = s
        if s > best_score:
            best_score, best_size, best_records = s, size, records

    assert best_size is not None
    return best_size, best_records, scores


# =============================================================================
# Top-level run
# =============================================================================
def run(
    engine: Deployment,
    image_path: Path,
    out_dir: Path,
    corners_json: Path | None,
    forced_size: int | None,
) -> dict:
    started = time.perf_counter()
    rgb = read_rgb(image_path)
    h, w = rgb.shape[:2]

    # 1. Localize
    if corners_json is not None:
        ann = json.loads(corners_json.read_text())
        if (int(ann["image_width"]), int(ann["image_height"])) != (w, h):
            raise ValueError(
                f"Annotation resolution {ann['image_width']}x{ann['image_height']} "
                f"does not match image {w}x{h}."
            )
        corners = np.array([ann["corners"][k] for k in CORNER_NAMES], np.float32)
        geometry_mode = "manual_corners"
    else:
        corners = detect_tray(rgb)
        geometry_mode = "auto_classical_cv"

    corners = order_corners(corners)
    if np.any(corners < 0) or np.any(corners[:, 0] >= w) or np.any(corners[:, 1] >= h):
        raise ValueError("Corners fall outside image; check orientation.")

    # 2. Choose layout and classify
    size, records, size_scores = choose_layout(engine, rgb, corners, forced_size)
    rows, cols = LAYOUTS[size]
    tray = warp_tray(rgb, corners, rows, cols, 224)

    for i, rec in enumerate(records):
        rec["row"], rec["column"] = i // cols + 1, i % cols + 1

    counts = Counter(r["flavor_id"] for r in records if r["label_type"] == "occupied")

    # 3. Human-readable summary
    summary_lines = [f"Image: {image_path}", f"Box size: {size} ({rows}x{cols})",
                     f"Occupied: {sum(counts.values())}", f"Empty: {sum(r['label_type']=='empty' for r in records)}",
                     f"Review slots: {sum(r['needs_review'] for r in records)}", "", "Chocolates found:"]
    for fid, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        summary_lines.append(f"  {engine.catalog[fid]['flavor_name']:>25}  x{n}")
    summary_lines.append("")
    summary_lines.append("Per-slot:")
    for r in records:
        flag = " REVIEW" if r["needs_review"] else ""
        summary_lines.append(
            f"  r{r['row']:02d} c{r['column']:02d}  {r['flavor_name']:>25}  m={r['margin']:.3f}{flag}"
        )
    summary = "\n".join(summary_lines)

    result = {
        "image": str(image_path.resolve()),
        "geometry_mode": geometry_mode,
        "box_size": size,
        "layout_rows": rows,
        "layout_columns": cols,
        "corners": corners.tolist(),
        "layout_scores": size_scores,
        "occupied": sum(counts.values()),
        "empty": sum(r["label_type"] == "empty" for r in records),
        "review_slots": sum(r["needs_review"] for r in records),
        "flavor_counts": dict(counts),
        "flavor_names": {fid: engine.catalog[fid]["flavor_name"] for fid in counts},
        "slots": records,
        "warnings": [
            "Raw predictions, including review slots.",
            "Real filled-box accuracy is not established.",
            "Auto geometry uses classical CV; verify box_overlay.png on first run.",
        ],
        "seconds": time.perf_counter() - started,
    }

    # 4. Write outputs
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    (out_dir / "summary.txt").write_text(summary)
    Image.fromarray(tray).save(out_dir / "tray.png")

    overlay = rgb.copy()
    cv2.polylines(overlay, [np.round(corners).astype(np.int32)], True,
                  (0, 255, 0), max(2, round(w / 400)))
    Image.fromarray(overlay).save(out_dir / "box_overlay.png")

    montage = Image.new("RGB", (cols * 224, rows * 264), "white")
    draw = ImageDraw.Draw(montage)
    for crop, slot in zip(slice_slots(tray, rows, cols), records):
        x, y = (slot["column"] - 1) * 224, (slot["row"] - 1) * 264
        montage.paste(Image.fromarray(crop), (x, y + 40))
        label = f"{slot['flavor_name']} | m={slot['margin']:.3f}{' REVIEW' if slot['needs_review'] else ''}"
        draw.text((x + 3, y + 3), label, fill="red" if slot["needs_review"] else "black")
    montage.save(out_dir / "predictions.png")

    return result, summary


# =============================================================================
# CLI
# =============================================================================
def dump_template(image_path: Path, out: Path) -> None:
    rgb = read_rgb(image_path)
    h, w = rgb.shape[:2]
    template = {
        "image_width": w,
        "image_height": h,
        "corners": {name: [0, 0] for name in CORNER_NAMES},
        "_note": (
            "Fill in pixel coordinates of the tray's four corners, in the "
            "same orientation as the decoded JPEG. TL=top-left, TR=top-right, "
            "BR=bottom-right, BL=bottom-left."
        ),
    }
    out.write_text(json.dumps(template, indent=2))
    print(f"Wrote template to {out} for image {w}x{h}. Fill in corner pixels, then "
          f"rerun with --corners {out}.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--corners", type=Path, default=None)
    ap.add_argument("--box-size", type=int, default=None, choices=list(LAYOUTS),
                    help="Force a layout instead of auto-search.")
    ap.add_argument("--dump-template", type=Path, default=None,
                    help="Write a corners template for this image and exit.")
    args = ap.parse_args()

    if args.threads < 1:
        ap.error("--threads must be >= 1")

    if args.dump_template is not None:
        dump_template(args.image, args.dump_template)
        return 0

    engine = Deployment(args.root, args.device, args.threads)
    try:
        result, summary = run(engine, args.image, args.out, args.corners, args.box_size)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"\nHint: run  python {Path(__file__).name} --image {args.image} "
              f"--out {args.out} --dump-template corners.json", file=sys.stderr)
        return 2

    printable = {k: v for k, v in result.items() if k != "slots"}
    print(json.dumps(printable, indent=2))
    print("\n" + summary)
    print(f"\nWrote outputs to: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())