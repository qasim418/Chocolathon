# Chocolathon Final Inference Bundle

Release: `chocolathon_inference_v2`  
Created: `2026-09-19T02:23:38.837887+00:00`

## Status

This bundle freezes the finalized offline filled-box pipeline for capacities 6, 10, 16, 30 and 50.

| Validation set | Images | Capacity accuracy | Exact flavor counts | Count error |
|---|---:|---:|---:|---:|
| Original sizes, synthetic v4 | 48 | 100% | 100% | 0 |
| 10-slot, synthetic v5 | 9 | 100% | 100% | 0 |
| Combined | 57 | 100% | 100% | 0 |

All 19 original empty boxes passed capacity detection. One real filled 10-slot photograph passed automatic capacity selection, segmentation and orientation-aware cropping. Independent real filled-box accuracy is not established.

## Final pipeline

1. Optional known capacity (`6`, `10`, `16`, `30`, or `50`), otherwise EfficientNet-B0 capacity classification.
2. EfficientNet-B0 U-Net tray segmentation for primary automatic localization.
3. Polygon extraction into `TL, TR, BR, BL` corners.
4. Routed 6/10/16 and 30/50 corner regressors only if segmentation fails.
5. Perspective rectification and capacity-specific uniform grid extraction.
6. EfficientNet-B0 1280-dimensional slot embeddings.
7. Cosine matching against the deployed flavor prototypes.
8. Filled-only mode excludes the empty class when every slot is known to contain chocolate.

## Model roles

| Artifact | Purpose |
|---|---|
| `data/experiments/tray_segmentation_v1/deployment/efficientnet_b0_tray_segmenter.pt` | Primary tray mask and corner localization |
| `data/experiments/box_localization_v4/deployment/efficientnet_b0_box_size_classifier.pt` | Automatic 6/10/16/30/50 capacity selection |
| `data/experiments/box_localization_v4/deployment/efficientnet_b0_corner_regressor_small.pt` | 6/10/16 localization fallback |
| `data/experiments/box_localization_v4/deployment/efficientnet_b0_corner_regressor_large.pt` | 30/50 localization fallback |
| Slot deployment backbone | Produces slot embeddings |
| Slot prototype matrix and catalog | Converts embeddings into flavor labels and counts |

## Important directories

| Path | Contents |
|---|---|
| `data/experiments/*/deployment` | Final inference configurations and weights |
| `data/synthetic_filled_boxes/v4` | 48 synthetic boxes for capacities 6/16/30/50 |
| `data/synthetic_filled_boxes/v5_10_slot` | 9 synthetic 10-slot boxes |
| `data/boxes/annotations` | Original outer tray-corner annotations |
| `data/boxes/grid_annotations` | Historical internal grid annotations; not segmentation masks |
| `data/boxes/slots` | Empty-slot diagnostic crops and manifest |
| `validation` | Final evaluation reports copied when available |
| `Application/Deployment` | FastAPI service and browser application |

## Environment

Create or activate a Python environment and install the included requirements file. Inference is offline and must not download model weights.

```bash
source .venv/bin/activate
python -m py_compile Application/Runtime/chocolathon_inference.py Application/Deployment/chocolathon_api.py
```

## Synthetic release test

```bash
python Application/Runtime/chocolathon_inference.py   --test-synthetic   --synthetic-root data/synthetic_filled_boxes/v4   --filled-only   --localizer segmentation   --empty-mode original   --geometry uniform   --threads 1   --batch-size 8   --output outputs/release_synthetic_test
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
python Application/Runtime/chocolathon_inference.py   --image /absolute/path/to/blind_real_box.jpg   --capacity 30   --filled-only   --localizer segmentation   --geometry uniform   --threads 1   --batch-size 8   --output outputs/blind_real_box
```

Before running, record the hidden ground-truth slot arrangement and flavor counts separately. Do not tune any model or threshold using this image before reporting its result.

## API

The included API initializes one persistent `Chocolathon` engine and accepts:

- Image upload
- Optional known capacity: `6`, `10`, `16`, `30`, or `50`
- `filled_only`, normally `true` for the intended deployment

It should return capacity, localization mode, four corners, per-slot predictions, flavor counts, review flags, warnings and inference time. Do not reload models for each request.

## Integrity

- `RELEASE_MANIFEST.json` records every payload file, size and SHA-256.
- `CHECKSUMS.sha256` covers the complete release except the checksum file itself.
- Files are copied rather than linked, so later project changes do not modify this release.
- The packaging command refuses to overwrite an existing release.
