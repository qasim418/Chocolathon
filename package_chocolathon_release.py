#!/usr/bin/env python3
"""Build a self-contained, immutable Chocolathon inference release bundle."""
import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path


RUNTIME_DIRECTORIES = [
    "data/experiments/box_localization_v3/deployment",
    "data/experiments/tray_segmentation_v1/deployment",
    "data/experiments/synthetic_slot_embeddings_v1/deployment",
]

VALIDATION_DIRECTORIES = [
    "data/synthetic_filled_boxes/v4",
    "data/experiments/synthetic_slot_embeddings_v1/outputs",
    "data/boxes/raw",
    "data/boxes/annotations",
    "data/boxes/grid_annotations",
    "data/boxes/normalized",
    "data/boxes/slots",
]

REQUIRED_FILES = [
    "chocolathon_inference.py",
    "data/boxes/manifest.csv",
    "data/experiments/box_localization_v3/deployment/localization_config.json",
    "data/experiments/tray_segmentation_v1/deployment/segmentation_config.json",
    "data/experiments/tray_segmentation_v1/deployment/efficientnet_b0_tray_segmenter.pt",
    "data/experiments/synthetic_slot_embeddings_v1/deployment/deployment_config.json",
    "data/synthetic_filled_boxes/v4/manifest.csv",
]

OPTIONAL_REPORTS = [
    "data/experiments/tray_segmentation_v1/segmentation_records.csv",
    "data/experiments/tray_segmentation_v1/training_history.csv",
    "data/experiments/tray_segmentation_v1/corner_evaluation.csv",
    "data/experiments/tray_segmentation_v1/downstream_comparison.csv",
    "outputs/v3_regression_verification/synthetic_report.json",
    "outputs/v3_regression_verification/synthetic_results.csv",
    "outputs/v3_segmentation_verification/synthetic_report.json",
    "outputs/v3_segmentation_verification/synthetic_results.csv",
]


def sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size): digest.update(chunk)
    return digest.hexdigest()


def copy_path(source, destination):
    if source.is_dir():
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints", "*.pyc"))
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def build_readme(release_name, created_at):
    return f"""# Chocolathon Final Inference Bundle

Release: `{release_name}`  
Created: `{created_at}`

## Status

This bundle freezes the finalized offline inference pipeline and its reproducible synthetic-v4 validation set. The complete automatic pipeline achieved:

| Metric | Result |
|---|---:|
| Synthetic v4 images | 48 |
| Capacity accuracy | 100% |
| Exact flavor-count accuracy | 100% |
| False-empty slots | 0 |
| Total flavor-count error | 0 |
| Mean CPU inference time | 0.676 seconds/image |

These are synthetic-v4 development results. Accuracy on independent real filled boxes is not yet established. The preserved real filled box must remain a blind end-to-end test.

## Final pipeline

1. Optional known capacity (`6`, `16`, `30`, or `50`), otherwise EfficientNet-B0 capacity classification.
2. EfficientNet-B0 U-Net tray segmentation for primary automatic localization.
3. Polygon extraction into `TL, TR, BR, BL` corners.
4. Routed 6/16 and 30/50 corner regressors only if segmentation fails.
5. Perspective rectification and capacity-specific uniform grid extraction.
6. EfficientNet-B0 1280-dimensional slot embeddings.
7. Cosine matching against the deployed flavor prototypes.
8. Filled-only mode excludes the empty class when every slot is known to contain chocolate.

## Model roles

| Artifact | Purpose |
|---|---|
| `data/experiments/tray_segmentation_v1/deployment/efficientnet_b0_tray_segmenter.pt` | Primary tray mask and corner localization |
| `data/experiments/box_localization_v3/deployment/efficientnet_b0_box_size_classifier.pt` | Automatic 6/16/30/50 capacity selection |
| `data/experiments/box_localization_v3/deployment/efficientnet_b0_corner_regressor_small.pt` | 6/16 localization fallback |
| `data/experiments/box_localization_v3/deployment/efficientnet_b0_corner_regressor_large.pt` | 30/50 localization fallback |
| Slot deployment backbone | Produces slot embeddings |
| Slot prototype matrix and catalog | Converts embeddings into flavor labels and counts |

## Important directories

| Path | Contents |
|---|---|
| `data/experiments/*/deployment` | Final inference configurations and weights |
| `data/synthetic_filled_boxes/v4` | 48 synthetic filled boxes, manifest and slot ground truths |
| `data/boxes/annotations` | Original outer tray-corner annotations |
| `data/boxes/grid_annotations` | Historical internal grid annotations; not segmentation masks |
| `data/boxes/slots` | Empty-slot diagnostic crops and manifest |
| `validation` | Final evaluation reports copied when available |
| `api` | Reserved location for the inference API |

## Environment

Create or activate a Python environment and install the included requirements file. Inference is offline and must not download model weights.

```bash
source .venv/bin/activate
python -m py_compile chocolathon_inference.py
```

## Synthetic release test

```bash
python chocolathon_inference.py \
  --test-synthetic \
  --synthetic-root data/synthetic_filled_boxes/v4 \
  --filled-only \
  --localizer segmentation \
  --empty-mode original \
  --geometry uniform \
  --threads 1 \
  --batch-size 8 \
  --output outputs/release_synthetic_test
```

Expected summary:

```text
capacity_accuracy: 1.0
exact_flavor_count_accuracy: 1.0
total_false_empty: 0
total_flavor_count_abs_error: 0
```

## Blind real-box test

Use the known capacity and do not supply manual corners:

```bash
python chocolathon_inference.py \
  --image /absolute/path/to/blind_real_box.jpg \
  --capacity 30 \
  --filled-only \
  --localizer segmentation \
  --geometry uniform \
  --threads 1 \
  --batch-size 8 \
  --output outputs/blind_real_box
```

Before running, record the hidden ground-truth slot arrangement and flavor counts separately. Do not tune any model or threshold using this image before reporting its result.

## API contract to implement

The API should initialize one persistent `Chocolathon` engine and accept:

- Image upload
- Optional known capacity: `6`, `16`, `30`, or `50`
- `filled_only`, normally `true` for the intended deployment

It should return capacity, localization mode, four corners, per-slot predictions, flavor counts, review flags, warnings and inference time. Do not reload models for each request.

## Integrity

- `RELEASE_MANIFEST.json` records every payload file, size and SHA-256.
- `CHECKSUMS.sha256` covers the complete release except the checksum file itself.
- Files are copied rather than linked, so later project changes do not modify this release.
- The packaging command refuses to overwrite an existing release.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Chocolathon project root")
    parser.add_argument("--name", default="chocolathon_inference_v1", help="New release folder name")
    parser.add_argument("--no-archive", action="store_true", help="Do not create a tar.gz beside the release folder")
    args = parser.parse_args()

    root = args.root.resolve()
    release_parent = root / "releases"
    release = release_parent / args.name
    archive = release_parent / f"{args.name}.tar.gz"

    if release.exists(): raise FileExistsError(f"Release already exists and will not be overwritten: {release}")
    if not args.no_archive and archive.exists(): raise FileExistsError(f"Archive already exists and will not be overwritten: {archive}")

    missing = [relative for relative in REQUIRED_FILES + RUNTIME_DIRECTORIES + VALIDATION_DIRECTORIES if not (root / relative).exists()]
    if missing: raise FileNotFoundError("Missing required release inputs:\n" + "\n".join(missing))

    inference_text = (root / "chocolathon_inference.py").read_text()
    required_markers = ["TraySegmenter", "automatic_segmentation", "--localizer", "--capacity", "filled_only"]
    absent_markers = [marker for marker in required_markers if marker not in inference_text]
    if absent_markers: raise ValueError("Inference script is not the finalized segmentation version: " + ", ".join(absent_markers))

    release.mkdir(parents=True)
    try:
        copy_path(root / "chocolathon_inference.py", release / "chocolathon_inference.py")
        for requirement in sorted(root.glob("requirements*.txt")): copy_path(requirement, release / requirement.name)
        for relative in RUNTIME_DIRECTORIES + VALIDATION_DIRECTORIES: copy_path(root / relative, release / relative)
        copy_path(root / "data/boxes/manifest.csv", release / "data/boxes/manifest.csv")

        validation_dir = release / "validation"
        validation_dir.mkdir(parents=True)
        copied_reports = []
        for relative in OPTIONAL_REPORTS:
            source = root / relative
            if source.exists():
                destination = validation_dir / relative.replace("/", "__")
                copy_path(source, destination)
                copied_reports.append(relative)

        created_at = datetime.now(timezone.utc).isoformat()
        (release / "README.md").write_text(build_readme(args.name, created_at))
        (release / "api").mkdir()
        (release / "api/README.md").write_text("# API\n\nReserved for the frozen-pipeline API implementation. See the release README for the required contract.\n")

        entries = []
        for path in sorted(p for p in release.rglob("*") if p.is_file()):
            relative = path.relative_to(release).as_posix()
            entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})

        manifest = {
            "release": args.name,
            "created_at_utc": created_at,
            "source_root": str(root),
            "pipeline": "capacity -> tray segmentation -> corner fallback -> rectification -> slots -> embeddings -> prototypes",
            "synthetic_v4_validation": {"images": 48, "capacity_accuracy": 1.0, "exact_flavor_count_accuracy": 1.0, "false_empty": 0, "flavor_count_abs_error": 0, "mean_cpu_seconds": 0.6033910983145082},
            "real_filled_box_accuracy_established": False,
            "optional_reports_copied": copied_reports,
            "files": entries,
        }
        (release / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2))

        final_files = sorted(p for p in release.rglob("*") if p.is_file())
        checksums = [f"{sha256(path)}  {path.relative_to(release).as_posix()}" for path in final_files]
        (release / "CHECKSUMS.sha256").write_text("\n".join(checksums) + "\n")

        required_release_files = [release / relative for relative in REQUIRED_FILES]
        missing_after_copy = [str(path.relative_to(release)) for path in required_release_files if not path.exists()]
        if missing_after_copy: raise RuntimeError("Release verification failed:\n" + "\n".join(missing_after_copy))

        if not args.no_archive:
            with tarfile.open(archive, "w:gz") as bundle: bundle.add(release, arcname=release.name)

        total_bytes = sum(path.stat().st_size for path in release.rglob("*") if path.is_file())
        print(json.dumps({
            "status": "PASS",
            "release_directory": str(release),
            "archive": None if args.no_archive else str(archive),
            "files": sum(1 for path in release.rglob("*") if path.is_file()),
            "size_mb": round(total_bytes / 1024**2, 2),
            "optional_reports_copied": copied_reports,
        }, indent=2))
    except Exception:
        shutil.rmtree(release, ignore_errors=True)
        raise



if __name__ == "__main__":
    try: main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise
