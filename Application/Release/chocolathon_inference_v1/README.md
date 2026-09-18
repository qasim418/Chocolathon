# Chocolathon Final Inference Bundle

Release: `chocolathon_inference_v1`  
Created: `2026-09-18T20:30:49.545859+00:00`

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
| `data/experiments/box_localization_v2/deployment/efficientnet_b0_box_size_classifier.pt` | Automatic 6/16/30/50 capacity selection |
| `data/experiments/box_localization_v2/deployment/efficientnet_b0_corner_regressor_small.pt` | 6/16 localization fallback |
| `data/experiments/box_localization_v2/deployment/efficientnet_b0_corner_regressor_large.pt` | 30/50 localization fallback |
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
python chocolathon_inference.py   --test-synthetic   --synthetic-root data/synthetic_filled_boxes/v4   --filled-only   --localizer segmentation   --empty-mode original   --geometry uniform   --threads 1   --batch-size 8   --output outputs/release_synthetic_test
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
python chocolathon_inference.py   --image /absolute/path/to/blind_real_box.jpg   --capacity 30   --filled-only   --localizer segmentation   --geometry uniform   --threads 1   --batch-size 8   --output outputs/blind_real_box
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
