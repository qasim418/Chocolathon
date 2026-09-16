## 1. Project objective

Chocolathon is a computer-vision prototype for analyzing chocolate boxes. The intended pipeline will:

1. Locate and normalize an empty or filled chocolate box.
2. identify individual box slots.
3. determine whether each slot is empty or occupied.
4. crop chocolates from occupied slots.
5. classify each detected chocolate by flavor.
6. present box-level predictions and confidence values.

The current repository contains two completed dataset-preparation pipelines:

- Empty box and slot-background preparation.
- Individual chocolate flavor dataset preparation.

Model training and evaluation have not started yet.

## 2. Repository portability

The project may run under different parent home directories on Beocat and Beoshock. Notebooks dynamically locate the repository root instead of relying on a fixed `/homes/<username>/...` path.

All paths stored in the frozen dataset are relative to the repository or frozen snapshot whenever possible.

## 3. Empty-box processing

Notebook:

- `01_box_detection.ipynb`

Completed work:

- 16 empty box photographs processed.
- Four tray corners manually annotated for every photograph.
- Perspective correction applied to normalize tray orientation.
- Slot-grid centers manually reviewed.
- Empty slots extracted as 224 × 224 images.
- Box capacities represented: 6, 16, 30, and 50.
- Total empty-slot crops: 418.

Empty-slot distribution:

| Box capacity | Source boxes | Empty-slot crops |
|---:|---:|---:|
| 6 | 3 | 18 |
| 16 | 5 | 80 |
| 30 | 4 | 120 |
| 50 | 4 | 200 |
| **Total** | **16** | **418** |

Important box data:

- `data/boxes/manifest.csv`
- `data/boxes/annotations/`
- `data/boxes/grid_annotations/`
- `data/boxes/slots/empty_slots_manifest.csv`
- `data/boxes/slots/empty/`

Generated normalized and oriented box images may be ignored by Git when they are reproducible from the committed annotations and notebook.

## 4. Chocolate tray processing

Notebook:

- `02_chocolate_dataset_creation.ipynb`

### 4.1 Source photographs

The initial collection contained 39 chocolate tray photographs.

Dataset decisions:

- `IMG_1962.jpg` was deleted because it duplicated `IMG_1961.jpg`.
- 38 unique source photographs remained.
- `IMG_1993.jpg`, Chocolate Dinosaur, was retained as an artifact but excluded from the classification dataset because it is not an official menu item.
- 37 tray photographs were therefore usable for chocolate extraction.

### 4.2 Tray cropping and perspective correction

Each source photograph was manually annotated using four tray corners in this order:

1. top-left
2. top-right
3. bottom-right
4. bottom-left

Perspective correction produced one cropped tray image per source photograph.

Artifacts:

- `data/chocolates/raw/flavor_trays/`
- `data/chocolates/tray_annotations/`
- `data/chocolates/cropped_trays/`

Counts:

- Source tray images: 38
- Tray-corner annotations: 38
- Cropped trays: 38

### 4.3 Flavor mapping

Every tray photograph was manually mapped to an official catalog flavor.

Artifacts:

- `data/chocolates/labels/flavor_catalog.csv`
- `data/chocolates/labels/tray_flavor_mapping.csv`

Catalog status:

- Official catalog classes: 36
- Classes represented by available photographs: 33
- Missing classes: Blackberry Sangria, Mint, and Peach Cobbler
- Excluded non-menu product: Chocolate Dinosaur

Flavor IDs include the product family to avoid ambiguity. For example, Truffle Amaretto and CocoaShot Amaretto are treated as separate classes.

### 4.4 Chocolate-center annotations

Automatic Hough-circle detection was tested but rejected because it produced false positives, missed pieces, and inaccurate overlapping detections.

Final annotation modes:

- `regular_grid`: four corner centers plus row and column counts.
- `irregular_manual`: every usable chocolate center clicked manually.
- `exclude`: source intentionally omitted from sample extraction.

Final reviewed annotations:

| Annotation mode | Trays | Candidate centers |
|---|---:|---:|
| Regular grid | 28 | 669 |
| Irregular manual | 9 | 230 |
| **Total** | **37** | **899** |

All 37 annotations passed structural and image-boundary validation.

Artifacts:

- `data/chocolates/center_annotations/`
- `data/chocolates/labels/tray_layout_review.csv`

## 5. Individual chocolate crops

Each reviewed center was converted into a 224 × 224 candidate image. Crop size was estimated separately for each tray using center spacing. Boundary crops used padding where necessary.

Initial extraction:

- Candidate crops: 899
- Source trays: 37
- Represented classes: 33
- Crops containing some padding: 110

Working artifacts:

- `data/chocolates/individual_candidates/`
- `data/chocolates/labels/individual_candidates_manifest.csv`

The working candidate directory is ignored by Git because the crops are preserved in the frozen dataset snapshot.

## 6. Candidate quality review

Every candidate received automatic quality measurements:

- padding fraction
- Laplacian blur score
- bright-pixel fraction
- dark-pixel fraction
- contrast

Automatic checks initially flagged:

| Reason | Candidates |
|---|---:|
| Large padding | 44 |
| Low contrast | 1 |
| Severe glare or overexposure | 1 |
| **Total flagged** | **46** |

All 46 flagged samples were manually reviewed.

A balanced audit then inspected two accepted samples from each of the 33 represented classes. One additional Creamsicle crop, `IMG_1989_c001`, was rejected because it was dominated by camera glare.

Final decisions:

| Decision | Samples |
|---|---:|
| Accepted | 889 |
| Rejected | 10 |
| Unresolved | 0 |
| **Total** | **899** |

Accepted samples per class range from 12 to 56.

## 7. Frozen reusable dataset

The complete reusable snapshot is located at:

- `data/chocolates/frozen/chocolate_dataset_v1/`

Snapshot contents:

```text
chocolate_dataset_v1/
├── source_trays/
├── cropped_trays/
├── annotations/
│   ├── trays/
│   └── centers/
├── images/
│   ├── accepted/
│   │   └── <flavor_id>/
│   └── rejected/
│       └── <flavor_id>/
└── metadata/
    ├── accepted_dataset_manifest.csv
    ├── rejected_candidates_manifest.csv
    ├── all_candidates_manifest.csv
    ├── file_checksums.csv
    ├── snapshot_info.json
    └── supporting catalog and review files