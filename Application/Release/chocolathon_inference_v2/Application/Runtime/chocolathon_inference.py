"""Offline Chocolathon inference, reconstructed from notebooks 01, 05, 06 and 07.

Automatic geometry reproduces notebook 07's uniform slot extraction; it is not
equivalent to notebook 01's manually annotated slot centers. Use --self-test to
measure saved-crop classification independently of automatic geometry. The
optional --empty-mode capacity builds four training-only empty references from
the saved cache; it does not retrain the backbone or overwrite saved artifacts.
No model downloads, threshold tuning or implicit annotation lookup occurs.

The deployment pipeline uses a box-size classifier, an EfficientNet-B0 tray
segmenter, routed corner regressors as automatic fallback, and an EfficientNet-B0
slot feature extractor. Slot labels are obtained by cosine matching those
features to the deployed prototype matrix and catalog. ``--capacity`` bypasses
box-size prediction when capacity is known. ``--filled-only`` excludes the empty
prototype when every tray slot is known to contain chocolate.
"""
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image, ImageDraw
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

LAYOUTS = {6: (3, 2), 10: (5, 2), 16: (4, 4), 30: (6, 5), 50: (10, 5)}
AUTO_CAPACITIES = (6, 10, 16, 30, 50)
CORNER_NAMES = ('top_left', 'top_right', 'bottom_right', 'bottom_left')
MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)


class BoxSizeClassifier(nn.Module):
    def __init__(self, classes=len(AUTO_CAPACITIES)):
        super().__init__()
        self.features = efficientnet_b0(weights=None).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(1280, 128), nn.SiLU(), nn.Dropout(.2), nn.Linear(128, classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


class SpatialCornerRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = efficientnet_b0(weights=None).features
        self.regressor = nn.Sequential(nn.Conv2d(1280, 64, 1), nn.BatchNorm2d(64), nn.SiLU(), nn.AdaptiveAvgPool2d((7, 7)),
                                       nn.Flatten(), nn.Linear(3136, 256), nn.SiLU(), nn.Dropout(.25), nn.Linear(256, 8), nn.Sigmoid())

    def forward(self, x):
        return self.regressor(self.features(x))


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels, output_channels):
        super().__init__(nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
                         nn.BatchNorm2d(output_channels), nn.SiLU(inplace=True),
                         nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
                         nn.BatchNorm2d(output_channels), nn.SiLU(inplace=True))


class UpBlock(nn.Module):
    def __init__(self, input_channels, skip_channels, output_channels):
        super().__init__()
        self.block = ConvBlock(input_channels + skip_channels, output_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class TraySegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = efficientnet_b0(weights=None).features
        self.up16, self.up8 = UpBlock(1280, 112, 256), UpBlock(256, 40, 128)
        self.up4, self.up2 = UpBlock(128, 24, 64), UpBlock(64, 16, 32)
        self.head = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16),
                                  nn.SiLU(inplace=True), nn.Conv2d(16, 1, 1))

    def forward(self, x):
        output_size, skips = x.shape[-2:], {}
        for index, layer in enumerate(self.encoder):
            x = layer(x)
            if index in {1, 2, 3, 5}: skips[index] = x
        x = self.up16(x, skips[5])
        x = self.up8(x, skips[3])
        x = self.up4(x, skips[2])
        x = self.up2(x, skips[1])
        return self.head(F.interpolate(x, size=output_size, mode='bilinear', align_corners=False))


def read_rgb(path):
    # Same OpenCV decode/orientation handling used by notebook 07.
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError('Cannot decode image: ' + str(path))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def localization_tensor(rgb):
    image = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(((image.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0)


def segmentation_tensor(rgb, input_size):
    image = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(((image.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0)


def order_polygon(points):
    points = np.asarray(points, np.float32).reshape(4, 2)
    center = points.mean(0)
    points = points[np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))]
    return np.roll(points, -int(np.argmin(points.sum(1))), axis=0)


def mask_to_quad(binary_mask):
    mask = (np.asarray(binary_mask) > 0).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, statistics, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1: raise ValueError('Segmentation produced no foreground tray.')
    label = 1 + int(np.argmax(statistics[1:, cv2.CC_STAT_AREA]))
    component = (labels == label).astype(np.uint8) * 255
    area_ratio = float((component > 0).mean())
    if not .02 < area_ratio < .95: raise ValueError(f'Invalid segmented tray area ratio: {area_ratio:.4f}.')
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    perimeter = cv2.arcLength(hull, True)
    for ratio in np.linspace(.003, .08, 78):
        polygon = cv2.approxPolyDP(hull, ratio * perimeter, True)
        if len(polygon) == 4: return order_polygon(polygon[:, 0]), area_ratio, 'polygon'
    rectangle = cv2.boxPoints(cv2.minAreaRect(hull))
    return order_polygon(rectangle), area_ratio, 'rectangle_fallback'


def ordered_corners(points):
    points = order_polygon(points)
    if not np.isfinite(points).all() or len(np.unique(points, axis=0)) != 4 or not cv2.isContourConvex(points):
        raise ValueError('Invalid or ambiguous corner geometry; provide explicit annotation.')
    if abs(cv2.contourArea(points)) < 16:
        raise ValueError('Predicted tray is too small to rectify safely.')
    return points


def oriented_layout(corners, capacity):
    rows, cols = LAYOUTS[capacity]
    if rows == cols: return rows, cols
    corners = np.asarray(corners, np.float32)
    observed_width = (np.linalg.norm(corners[1]-corners[0]) + np.linalg.norm(corners[2]-corners[3]))/2
    observed_height = (np.linalg.norm(corners[3]-corners[0]) + np.linalg.norm(corners[2]-corners[1]))/2
    return (cols, rows) if (observed_width > observed_height) != (cols > rows) else (rows, cols)


def warp(rgb, corners, capacity, cell_size, layout=None):
    rows, cols = layout or LAYOUTS[capacity]
    width, height = cols * cell_size, rows * cell_size
    destination = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
    return cv2.warpPerspective(rgb, matrix, (width, height))


def centered_crop(rgb, x, y):
    x, y = round(x), round(y)
    padded = cv2.copyMakeBorder(rgb, 112, 112, 112, 112, cv2.BORDER_REFLECT_101)
    crop = padded[y:y + 224, x:x + 224]
    if crop.shape != (224, 224, 3):
        raise ValueError('Slot center outside the normalized tray.')
    return crop


def grid_crops(tray, grid, capacity):
    rows, cols = LAYOUTS[capacity]
    if (int(grid['rows']), int(grid['columns'])) != (rows, cols):
        raise ValueError('Grid annotation and box capacity disagree.')
    tl, tr, br, bl = np.array([grid['anchors'][k] for k in CORNER_NAMES], np.float32)
    crops = []
    for r in range(rows):
        for c in range(cols):
            u, v = c / max(cols - 1, 1), r / max(rows - 1, 1)
            xy = (tl + u * (tr - tl)) * (1 - v) + (bl + u * (br - bl)) * v
            if not np.isfinite(xy).all() or not (0 <= xy[0] < tray.shape[1] and 0 <= xy[1] < tray.shape[0]):
                raise ValueError('Invalid grid anchor coordinates.')
            crops.append(centered_crop(tray, *xy))
    return crops


def load_embedding_cache(outputs):
    with (outputs / 'embedding_index.csv').open(newline='') as f:
        index = sorted(csv.DictReader(f), key=lambda r: int(r['embedding_row']))
    with np.load(outputs / 'efficientnet_b0_slot_embeddings.npz', allow_pickle=False) as cache:
        # Legacy sample_ids is an object array; do not unpickle it. Validate the
        # numeric rows against the independently saved training prototypes.
        embeddings = cache['embeddings'].copy()
    if embeddings.shape != (len(index), 1280) or not np.isfinite(embeddings).all():
        raise ValueError('Invalid cached embedding matrix.')
    if any(int(r['embedding_row']) != i for i, r in enumerate(index)) or len({r['sample_id'] for r in index}) != len(index):
        raise ValueError('Invalid embedding index.')
    with np.load(outputs / 'efficientnet_b0_slot_class_prototypes.npz', allow_pickle=False) as saved:
        for class_id, reference in zip(saved['class_ids'].astype(str), saved['prototypes']):
            selected = [i for i, r in enumerate(index) if r['split'] == 'train' and r['flavor_id'] == class_id]
            if not selected:
                raise ValueError('Missing training class in embedding index.')
            vector = embeddings[selected].mean(0)
            vector /= np.linalg.norm(vector)
            if not np.allclose(vector, reference, atol=1e-6, rtol=1e-5):
                raise ValueError('Cache/index cannot reproduce saved training prototypes.')
    return embeddings, index


def read_slot_truth(path):
    """Read one generated slot CSV and return capacity, slot count, flavor counts."""
    with Path(path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Empty slot ground-truth CSV: ' + str(path))
    capacity = int(rows[0].get('box_size', rows[0].get('capacity')))
    occupied = [r for r in rows if str(r.get('label_type', 'occupied')).lower() == 'occupied']
    return capacity, len(rows), Counter(str(r['flavor_id']) for r in occupied)


class Chocolathon:
    def __init__(self, root, device='cpu', threads=4, batch_size=32, empty_mode='original', geometry='uniform', localizer='segmentation'):
        self.root, self.device, self.batch_size = Path(root).resolve(), torch.device(device), batch_size
        torch.set_num_threads(threads)

        slot_deployment = self.root / 'data/experiments/synthetic_slot_embeddings_v1/deployment'
        localization = self.root / 'data/experiments/box_localization_v4/deployment'
        localization_config_path = localization / 'localization_config.json'
        segmentation = self.root / 'data/experiments/tray_segmentation_v1/deployment'
        segmentation_config_path = segmentation / 'segmentation_config.json'

        self.config = json.loads((slot_deployment / 'deployment_config.json').read_text())
        self.localization_config = json.loads(localization_config_path.read_text())
        self.segmentation_config = json.loads(segmentation_config_path.read_text())
        self.localizer = localizer

        if self.config['backbone'] != 'efficientnet_b0' or self.config['weights_origin'] != 'IMAGENET1K_V1':
            raise ValueError('Unsupported slot deployment configuration.')
        if localizer not in {'segmentation', 'regression'}:
            raise ValueError('Localizer must be segmentation or regression.')
        if self.segmentation_config.get('architecture') != 'efficientnet_b0_unet':
            raise ValueError('Unsupported tray-segmentation architecture.')
        if self.segmentation_config.get('corner_order') != list(CORNER_NAMES):
            raise ValueError('Segmentation corner order does not match inference.')

        expected_capacities = list(AUTO_CAPACITIES)
        if self.localization_config.get('capacity_class_order') != expected_capacities:
            raise ValueError('Localization capacity order does not match LAYOUTS.')

        routing = self.localization_config.get('corner_routing', {})
        if set(map(int, routing)) != set(expected_capacities):
            raise ValueError('Localization corner routing must define 6, 10, 16, 30 and 50.')

        self.threshold = float(self.config['provisional_margin_threshold'])
        self.empty_id = self.config['empty_class_id']

        with (slot_deployment / self.config['class_catalog_file']).open(newline='') as f:
            catalog = list(csv.DictReader(f))
        self.catalog = {row['flavor_id']: row for row in catalog}

        with np.load(slot_deployment / self.config['prototype_file'], allow_pickle=False) as data:
            ids, prototypes = data['class_ids'].astype(str), data['prototypes'].copy()

        self.ids = ids.tolist()
        if len(set(self.ids)) != len(ids) or set(self.ids) != set(self.catalog) or len(catalog) != len(ids):
            raise ValueError('Catalog/prototype class mapping mismatch.')
        if prototypes.shape != (int(self.config['classes']), 1280) or not np.isfinite(prototypes).all():
            raise ValueError('Invalid prototype matrix.')
        if not np.allclose(np.linalg.norm(prototypes, axis=1), 1, atol=1e-5):
            raise ValueError('Prototypes must be L2 normalized.')
        if self.empty_id not in self.ids:
            raise ValueError('Empty class is missing.')

        self.prototypes = torch.as_tensor(prototypes, dtype=torch.float32, device=self.device)
        self.empty_mode, self.empty_references = empty_mode, None

        if empty_mode == 'capacity':
            outputs = slot_deployment.parent / 'outputs'
            embeddings, index = load_embedding_cache(outputs)
            references = []

            for capacity in AUTO_CAPACITIES:
                selected = [
                    int(row['embedding_row']) for row in index
                    if row['split'] == 'train'
                    and row['flavor_id'] == self.empty_id
                    and int(row['box_size']) == capacity
                ]
                if not selected:
                    raise ValueError(f'Missing training empty crops for capacity {capacity}.')

                vector = embeddings[selected].mean(0)
                norm = np.linalg.norm(vector)
                if norm <= 0:
                    raise ValueError(f'Invalid empty reference for capacity {capacity}.')
                references.append(vector / norm)

            self.empty_references = torch.as_tensor(np.stack(references), dtype=torch.float32, device=self.device)

        elif empty_mode != 'original':
            raise ValueError('Unknown empty-class mode.')

        self.geometry, self.centers = geometry, {}

        if geometry == 'training-centers':
            _, index = load_embedding_cache(slot_deployment.parent / 'outputs')
            sources = {row['empty_source_file'] for row in index if row['split'] == 'train'}

            with (self.root / 'data/boxes/slots/empty_slots_manifest.csv').open(newline='') as f:
                slots = list(csv.DictReader(f))

            for capacity in AUTO_CAPACITIES:
                rows, columns = LAYOUTS[capacity]
                centers = []

                for row in range(1, rows + 1):
                    for column in range(1, columns + 1):
                        selected = [
                            slot for slot in slots
                            if slot['source_file'] in sources
                            and int(slot['box_size']) == capacity
                            and int(slot['row']) == row
                            and int(slot['column']) == column
                        ]

                        if not selected or len({slot['source_file'] for slot in selected}) != len(selected):
                            raise ValueError(f'Missing or duplicated centers for capacity={capacity}, row={row}, column={column}.')

                        xy = np.array([[float(slot['center_x']), float(slot['center_y'])] for slot in selected]).mean(0)
                        if not np.isfinite(xy).all() or not (0 <= xy[0] < columns * 256 and 0 <= xy[1] < rows * 256):
                            raise ValueError('Invalid training slot-center coordinates.')

                        centers.append(xy)

                self.centers[capacity] = centers

        elif geometry != 'uniform':
            raise ValueError('Unknown automatic geometry mode.')

        size_path = localization / self.localization_config['size_classifier']
        self.size_model = self.load(BoxSizeClassifier(), size_path)

        unique_corner_files = sorted(set(routing.values()))
        self.corner_models = {
            filename: self.load(SpatialCornerRegressor(), localization / filename)
            for filename in unique_corner_files
        }
        self.corner_routing = {int(capacity): filename for capacity, filename in routing.items()}

        segmentation_path = segmentation / self.segmentation_config['model_file']
        segmentation_checkpoint = torch.load(segmentation_path, map_location='cpu', weights_only=True)
        if not isinstance(segmentation_checkpoint, dict) or 'model_state_dict' not in segmentation_checkpoint:
            raise ValueError('Invalid tray-segmentation checkpoint.')
        self.segmenter = TraySegmenter()
        self.segmenter.load_state_dict(segmentation_checkpoint['model_state_dict'], strict=True)
        self.segmenter = self.segmenter.to(self.device).eval()
        self.segmentation_input_size = int(self.segmentation_config['input_size'])
        self.segmentation_threshold = float(self.segmentation_config['mask_threshold'])

        feature_path = slot_deployment / self.config['local_backbone_file']
        feature_model = efficientnet_b0(weights=None)
        feature_model.classifier = nn.Identity()
        self.features = self.load(feature_model, feature_path)
        self.slot_transform = EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()

        self.artifacts = {
            'localization_config': str(localization_config_path.resolve()),
            'segmentation_config': str(segmentation_config_path.resolve()),
            'tray_segmenter': str(segmentation_path.resolve()),
            'box_size_classifier': str(size_path.resolve()),
            'corner_regressor_6_16': str((localization / routing['6']).resolve()),
            'corner_regressor_30_50': str((localization / routing['30']).resolve()),
            'slot_feature_extractor': str(feature_path.resolve()),
            'slot_prototypes': str((slot_deployment / self.config['prototype_file']).resolve()),
            'class_catalog': str((slot_deployment / self.config['class_catalog_file']).resolve()),
            'slot_deployment_config': str((slot_deployment / 'deployment_config.json').resolve()),
        }
    @torch.inference_mode()
    def segment_corners(self, rgb):
        height, width = rgb.shape[:2]
        tensor = segmentation_tensor(rgb, self.segmentation_input_size).to(self.device)
        probability = self.segmenter(tensor).sigmoid()[0, 0].cpu().numpy()
        small_corners, area_ratio, extraction = mask_to_quad(probability >= self.segmentation_threshold)
        corners = small_corners * np.array([width / self.segmentation_input_size, height / self.segmentation_input_size], np.float32)
        return corners, {
            'threshold': self.segmentation_threshold,
            'area_ratio': area_ratio,
            'extraction': extraction,
            'probability_min': float(probability.min()),
            'probability_max': float(probability.max()),
        }

    @torch.inference_mode()
    def smoke_test(self):
        x = torch.zeros(1, 3, 224, 224, device=self.device)
        segmentation_x = torch.zeros(1, 3, self.segmentation_input_size, self.segmentation_input_size, device=self.device)
        corner_outputs = {filename: list(model(x).shape) for filename, model in self.corner_models.items()}

        if list(self.size_model(x).shape) != [1, len(AUTO_CAPACITIES)]:
            raise ValueError('Invalid box-size model output.')
        if any(shape != [1, 8] for shape in corner_outputs.values()):
            raise ValueError('Invalid corner-model output.')
        if list(self.segmenter(segmentation_x).shape) != [1, 1, self.segmentation_input_size, self.segmentation_input_size]:
            raise ValueError('Invalid tray-segmentation output.')
        if list(self.features(x).shape) != [1, 1280]:
            raise ValueError('Invalid feature-extractor output.')

        return {
            'box_size_output': [1, len(AUTO_CAPACITIES)],
            'segmentation_output': [1, 1, self.segmentation_input_size, self.segmentation_input_size],
            'corner_outputs': corner_outputs,
            'feature_output': [1, 1280],
            'prototype_shape': list(self.prototypes.shape),
            'corner_routing': self.corner_routing,
            'model_roles': {
                'box_size_classifier': 'predict 6/10/16/30/50 slot capacity',
                'tray_segmenter': 'primary automatic tray localization for all capacities',
                'corner_regressor_6_16': 'fallback tray corners for 6-slot, 10-slot and 16-slot boxes',
                'corner_regressor_30_50': 'fallback tray corners for 30-slot and 50-slot boxes',
                'slot_feature_extractor': 'produce normalized 1280-D slot embeddings',
                'slot_class_prototypes': 'cosine-match 34 classes including empty',
            },
            'artifact_validation': 'PASS',
        }

    def load(self, model, path):
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
        return model.to(self.device).eval()

    @torch.inference_mode()
    def classify(self, crops, filled_only=False):
        records = []
        empty_index = self.ids.index(self.empty_id)

        for start in range(0, len(crops), self.batch_size):
            batch = torch.stack([self.slot_transform(Image.fromarray(crop)) for crop in crops[start:start+self.batch_size]]).to(self.device)
            embeddings = F.normalize(self.features(batch), dim=1)
            similarities = embeddings@self.prototypes.T

            if self.empty_references is not None:
                similarities[:, empty_index] = (embeddings@self.empty_references.T).max(dim=1).values

            scores = similarities.cpu().numpy()
            if not np.isfinite(scores).all(): raise ValueError("Non-finite model scores.")

            for original_score in scores:
                score = original_score.copy()
                empty_similarity = float(score[empty_index])
                occupied_score = max(float(score[i]) for i, class_id in enumerate(self.ids) if class_id != self.empty_id)

                if filled_only: score[empty_index] = -np.inf

                ranking = np.argsort(-score, kind="stable")
                first, second = ranking[:2]
                flavor_id = self.ids[first]
                margin = float(score[first]-score[second])

                records.append({
                    "flavor_id": flavor_id,
                    "flavor_name": self.catalog[flavor_id]["flavor_name"],
                    "label_type": "empty" if flavor_id == self.empty_id else "occupied",
                    "similarity": float(score[first]),
                    "margin": margin,
                    "empty_similarity": empty_similarity,
                    "occupancy_score_gap": occupied_score-empty_similarity,
                    "needs_review": margin < self.threshold,
                })

        return records

    @torch.inference_mode()
    def predict(self, image, annotation=None, grid=None, filled_only=False, capacity_override=None):
        started = time.perf_counter()
        rgb = read_rgb(image)
        height, width = rgb.shape[:2]
        tensor = localization_tensor(rgb).to(self.device)

        if capacity_override is not None:
            capacity_override = int(capacity_override)
            if capacity_override not in LAYOUTS: raise ValueError('Capacity override must be 6, 10, 16, 30 or 50.')
            capacity_labels = tuple(LAYOUTS)
            probabilities = np.array([float(value == capacity_override) for value in capacity_labels], np.float32)
            predicted_capacity = capacity = capacity_override
            capacity_source = 'override'
        else:
            capacity_labels = AUTO_CAPACITIES
            probabilities = self.size_model(tensor).softmax(1)[0].cpu().numpy()
            predicted_capacity = AUTO_CAPACITIES[int(probabilities.argmax())]
            capacity = predicted_capacity
            capacity_source = 'classifier'

        if annotation is None:
            localization_diagnostics, fallback_reason = {}, None
            if self.localizer == 'segmentation':
                try:
                    corners, localization_diagnostics = self.segment_corners(rgb)
                    corner_source = self.segmentation_config['model_file']
                except Exception as error:
                    fallback_reason = str(error)
                    corner_filename = self.corner_routing[capacity]
                    corners = self.corner_models[corner_filename](tensor)[0].cpu().numpy().reshape(4, 2)
                    corners *= np.array([width, height], np.float32)
                    corner_source = corner_filename
            else:
                corner_filename = self.corner_routing[capacity]
                corners = self.corner_models[corner_filename](tensor)[0].cpu().numpy().reshape(4, 2)
                corners *= np.array([width, height], np.float32)
                corner_source = corner_filename
        else:
            if (int(annotation['image_width']), int(annotation['image_height'])) != (width, height):
                raise ValueError('Annotation resolution differs from the decoded image; use its original oriented image.')
            capacity = int(annotation['box_size'])
            if capacity_override is not None and capacity != capacity_override:
                raise ValueError('Capacity override and annotation box_size disagree.')
            capacity_source = 'override' if capacity_override is not None else 'annotation'
            corners = np.array([annotation['corners'][name] for name in CORNER_NAMES], np.float32)
            corner_source = 'annotation'
            localization_diagnostics, fallback_reason = {}, None

        if capacity not in LAYOUTS:
            raise ValueError('Unsupported box capacity.')
        if grid is not None and annotation is None:
            raise ValueError('Per-image grid centers require the corresponding corner annotation.')

        corners = ordered_corners(corners)
        if np.any(corners < 0) or np.any(corners[:, 0] >= width) or np.any(corners[:, 1] >= height):
            raise ValueError('Corner coordinates exceed image bounds; check annotation resolution/orientation.')

        rows, columns = LAYOUTS[capacity]
        if grid is None and not self.centers: rows, columns = oriented_layout(corners, capacity)
        tray = warp(rgb, corners, capacity, 256 if grid is not None or self.centers else 224, layout=(rows, columns))

        if grid is not None:
            crops = grid_crops(tray, grid, capacity)
        elif self.centers:
            crops = [centered_crop(tray, *xy) for xy in self.centers[capacity]]
        else:
            crops = [
                tray[row * 224:(row + 1) * 224, column * 224:(column + 1) * 224]
                for row in range(rows)
                for column in range(columns)
            ]

        records = self.classify(crops, filled_only=filled_only)
        for index, record in enumerate(records):
            record.update(row=index // columns + 1, column=index % columns + 1)

        counts = Counter(record['flavor_id'] for record in records if record['label_type'] == 'occupied')
        warnings = (
            [
                'Counts are raw predictions, including review slots.',
                'Real filled-box accuracy is not established.',
                'Automatic segmentation geometry passed synthetic v4 testing but requires validation on new real boxes.',
            ]
            if grid is None else
            [
                'Annotation-assisted diagnostic; not an automatic accuracy test.',
                'Real filled-box accuracy is not established.',
            ]
        )

        if self.empty_mode != 'original':
            warnings.append('Experimental four-reference empty classifier; the review threshold needs revalidation on real boxes.')
        if fallback_reason:
            warnings.append('Tray segmentation failed; routed corner regression was used as fallback: ' + fallback_reason)
        if filled_only:
            warnings.append('Filled-only mode excludes the empty class and forces every slot to receive a chocolate flavor.')
            warnings.append('Review-margin calibration has not been independently validated for filled-only mode.')

        result = {
            'image': str(Path(image).resolve()),
            'geometry_mode': 'annotation_assisted' if annotation else ('automatic_segmentation' if not fallback_reason and self.localizer == 'segmentation' else 'automatic_regression'),
            'empty_mode': self.empty_mode,
            'filled_only': filled_only,
            'capacity': capacity,
            'predicted_capacity': predicted_capacity,
            'capacity_source': capacity_source,
            'layout_rows': rows,
            'layout_columns': columns,
            'size_probabilities': dict(zip(map(str, capacity_labels), map(float, probabilities))),
            'corner_model': corner_source,
            'localization_diagnostics': localization_diagnostics,
            'corners': corners.tolist(),
            'occupied': sum(counts.values()),
            'empty': sum(record['label_type'] == 'empty' for record in records),
            'review_slots': sum(record['needs_review'] for record in records),
            'flavor_counts': dict(counts),
            'slots': records,
            'non_review_occupied': sum(record['label_type'] == 'occupied' and not record['needs_review'] for record in records),
            'non_review_empty': sum(record['label_type'] == 'empty' and not record['needs_review'] for record in records),
            'warnings': warnings,
            'seconds': time.perf_counter() - started,
        }

        assert result['occupied'] + result['empty'] == capacity == len(records)
        if filled_only:
            assert result['occupied'] == capacity and result['empty'] == 0
        return result, rgb, tray, crops

    def self_test(self):
        """Development diagnostics only: these empty crops contributed to prototypes."""
        directory = self.root / 'data/boxes'
        with (directory / 'slots/empty_slots_manifest.csv').open(newline='') as f:
            manifest = list(csv.DictReader(f))
        by_size = {}
        for capacity in AUTO_CAPACITIES:
            rows = [r for r in manifest if int(r['box_size']) == capacity]
            paths = [directory / 'slots/empty' / str(capacity) / r['slot_file'] for r in rows]
            # PIL matches the original embedding notebook's JPEG loader.
            crops = []
            for path in paths:
                with Image.open(path) as image:
                    crops.append(np.asarray(image.convert('RGB')))
            predictions = self.classify(crops)
            by_size[str(capacity)] = dict(total=len(rows), false_occupied=sum(r['label_type'] == 'occupied' for r in predictions))
        with (directory / 'manifest.csv').open(newline='') as f:
            boxes = list(csv.DictReader(f))
        automatic = []
        for box in boxes:
            result, _, _, _ = self.predict(directory / 'raw' / box['file'])
            automatic.append(dict(image=box['file'], expected_capacity=int(box['box_size']), predicted_capacity=result['capacity'],
                                  false_occupied=result['occupied'], review_slots=result['review_slots']))
        total = sum(r['total'] for r in by_size.values())
        false_occupied = sum(r['false_occupied'] for r in by_size.values())
        return dict(test_scope='Development empty-box diagnostics, NOT independent or filled-box accuracy', empty_mode=self.empty_mode, geometry=self.geometry,
                    saved_crop_summary=dict(total=total, false_occupied=false_occupied, empty_accuracy=1 - false_occupied / total),
                    automatic_summary=dict(images=len(boxes), capacity_correct=sum(r['expected_capacity'] == r['predicted_capacity'] for r in automatic),
                                           false_occupied=sum(r['false_occupied'] for r in automatic)), saved_crops=by_size, automatic=automatic)

    def audit_cache(self):
        """Compare original/deployed/current scoring on the same cached samples."""
        outputs = self.root / 'data/experiments/synthetic_slot_embeddings_v1/outputs'
        embeddings, index = load_embedding_cache(outputs)
        truth = np.array([r['flavor_id'] for r in index])
        validation = np.array([r['split'] == 'validation' for r in index])
        empty = truth == self.empty_id
        report = {'scope': 'Cached development data; candidate selected after inspecting validation results; new real-box evaluation remains required.'}
        with np.load(outputs / 'efficientnet_b0_slot_class_prototypes.npz', allow_pickle=False) as original:
            candidates = [('original_evaluation', original['class_ids'].astype(str).copy(), embeddings @ original['prototypes'].T)]
        if self.empty_references is not None:
            train_ids, train_scores = candidates[0][1], candidates[0][2].copy()
            train_scores[:, list(train_ids).index(self.empty_id)] = (embeddings @ self.empty_references.cpu().numpy().T).max(1)
            candidates.append(('training_only_four_empty_references', train_ids, train_scores))
        scores = embeddings @ self.prototypes.cpu().numpy().T
        candidates.append(('deployed', np.array(self.ids), scores.copy()))
        if self.empty_references is not None:
            scores[:, self.ids.index(self.empty_id)] = (embeddings @ self.empty_references.cpu().numpy().T).max(1)
            candidates.append(('four_empty_references', np.array(self.ids), scores))
        for name, ids, scores in candidates:
            predictions = ids[scores.argmax(1)]
            report[name] = dict(validation_total=int(validation.sum()), validation_errors=int((predictions[validation] != truth[validation]).sum()),
                                all_empty_total=int(empty.sum()), all_empty_errors=int((predictions[empty] != truth[empty]).sum()),
                                occupied_total=int((~empty).sum()), occupied_errors=int((predictions[~empty] != truth[~empty]).sum()))
        return report

    def test_synthetic(self, synthetic_root, filled_only=False):
        """Run automatic inference on generated filled boxes and compare GT CSVs."""
        synthetic_root = Path(synthetic_root)
        candidates = [synthetic_root / 'manifest.csv', synthetic_root / 'metadata' / 'manifest.csv']
        manifest_path = next((p for p in candidates if p.exists()), None)
        if manifest_path is None:
            found = sorted(synthetic_root.rglob('manifest*.csv'))
            manifest_path = found[0] if found else None
        if manifest_path is None:
            raise FileNotFoundError('No synthetic manifest.csv under ' + str(synthetic_root))
        with manifest_path.open(newline='') as f:
            manifest = list(csv.DictReader(f))
        if not manifest:
            raise ValueError('Synthetic manifest is empty: ' + str(manifest_path))
        rows = []
        for item in manifest:
            image_ref = item.get('image') or item.get('image_path') or item.get('file')
            gt_ref = item.get('gt_path') or item.get('slots_csv') or item.get('ground_truth')
            if not image_ref or not gt_ref:
                raise ValueError('Synthetic manifest needs image and gt_path columns.')
            image, gt = Path(image_ref), Path(gt_ref)
            if not image.is_absolute():
                image = synthetic_root / image
            if not gt.is_absolute():
                gt = synthetic_root / gt
            expected_capacity, expected_slots, expected_counts = read_slot_truth(gt)
            result, _, _, _ = self.predict(image, filled_only=filled_only)
            predicted_counts = Counter(result['flavor_counts'])
            keys = set(expected_counts) | set(predicted_counts)
            count_error = sum(abs(expected_counts[k] - predicted_counts[k]) for k in keys)
            rows.append({
                'image': str(image), 'ground_truth': str(gt),
                'filled_only': filled_only,
                'expected_capacity': expected_capacity, 'predicted_capacity': result['predicted_capacity'],
                'capacity_correct': result['predicted_capacity'] == expected_capacity,
                'expected_slots': expected_slots, 'predicted_slots': len(result['slots']),
                'expected_occupied': expected_slots, 'predicted_occupied': result['occupied'],
                'false_empty': expected_slots - result['occupied'],
                'flavor_count_abs_error': count_error,
                'exact_flavor_counts': expected_counts == predicted_counts,
                'review_slots': result['review_slots'], 'seconds': result['seconds'],
                'flavor_counts': dict(result['flavor_counts']),
            })
        n = len(rows)
        return {
            'scope': 'Generated filled-box deployment test; all ground-truth slots are occupied.',
            'filled_only': filled_only,
            'manifest': str(manifest_path.resolve()), 'images': n,
            'capacity_accuracy': sum(r['capacity_correct'] for r in rows) / n,
            'exact_flavor_count_accuracy': sum(r['exact_flavor_counts'] for r in rows) / n,
            'total_false_empty': sum(r['false_empty'] for r in rows),
            'total_flavor_count_abs_error': sum(r['flavor_count_abs_error'] for r in rows),
            'mean_seconds': sum(r['seconds'] for r in rows) / n,
            'results': rows,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/chocolathon'))
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--corners-json', type=Path, help='Explicit diagnostic annotation; coordinates must match decoded image.')
    parser.add_argument('--grid-json', type=Path, help='Matching notebook 01 grid annotation; requires --corners-json.')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--audit-cache', action='store_true', help='Reproduce original, deployed and selected scoring on the saved embeddings.')
    parser.add_argument('--test-synthetic', action='store_true', help='Evaluate generated filled boxes against their slot CSV ground truth.')
    parser.add_argument('--synthetic-root', type=Path, default=Path('data/synthetic_filled_boxes/v4'))
    parser.add_argument('--empty-mode', choices=['original', 'capacity'], default='original', help='capacity: experimental four-reference empty class built from training cache; original: unchanged deployment prototypes.')
    parser.add_argument('--geometry', choices=['uniform', 'training-centers'], default='uniform', help='training-centers: automatic average grid centers from training images only, on notebook 01 canvases.')
    parser.add_argument('--localizer', choices=['segmentation', 'regression'], default='segmentation', help='segmentation: primary tray-mask localizer with regression fallback; regression: legacy routed corner models.')
    parser.add_argument('--capacity', type=int, choices=list(LAYOUTS), help='Known box capacity; bypasses automatic size classification for single-image inference.')
    parser.add_argument('--filled-only', action='store_true', help='Exclude the empty class and assign every slot a chocolate flavor.')
    args = parser.parse_args()
    if args.threads < 1 or args.batch_size < 1 or not (args.image or args.self_test or args.audit_cache or args.test_synthetic):
        parser.error('Provide --image, --self-test, --audit-cache or --test-synthetic, with positive threads and batch size.')
    if args.grid_json and not args.corners_json:
        parser.error('--grid-json requires --corners-json.')
    if args.filled_only and args.empty_mode != 'original':
        parser.error('--filled-only requires --empty-mode original because empty references are not used.')
    if args.capacity and not args.image:
        parser.error('--capacity is only valid with --image.')
    engine = Chocolathon(args.root, args.device, args.threads, args.batch_size, args.empty_mode, args.geometry, args.localizer)
    args.output.mkdir(parents=True, exist_ok=True)
    smoke = engine.smoke_test()
    smoke['artifacts'] = engine.artifacts
    (args.output / 'smoke_test.json').write_text(json.dumps(smoke, indent=2))
    if args.audit_cache:
        audit = engine.audit_cache()
        (args.output / 'embedding_audit.json').write_text(json.dumps(audit, indent=2))
        print(json.dumps(audit, indent=2))
    if args.test_synthetic:
        report = engine.test_synthetic(args.synthetic_root, filled_only=args.filled_only)
        (args.output / 'synthetic_report.json').write_text(json.dumps(report, indent=2))
        with (args.output / 'synthetic_results.csv').open('w', newline='') as f:
            fields = [k for k, v in report['results'][0].items() if not isinstance(v, dict)]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in report['results']:
                writer.writerow({k: row[k] for k in fields})
        print(json.dumps({k: v for k, v in report.items() if k != 'results'}, indent=2))
    if args.self_test:
        result = engine.self_test()
        (args.output / 'diagnostics.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    if args.image:
        annotation = json.loads(args.corners_json.read_text()) if args.corners_json else None
        grid = json.loads(args.grid_json.read_text()) if args.grid_json else None
        result, rgb, tray, crops = engine.predict(args.image, annotation, grid, filled_only=args.filled_only, capacity_override=args.capacity)
        (args.output / 'result.json').write_text(json.dumps(result, indent=2))
        Image.fromarray(tray).save(args.output / 'tray.png')
        overlay = rgb.copy()
        cv2.polylines(overlay, [np.round(result['corners']).astype(np.int32)], True, (0, 255, 0), max(2, round(rgb.shape[1] / 400)))
        Image.fromarray(overlay).save(args.output / 'box.png')
        rows, cols = LAYOUTS[result['capacity']]
        montage = Image.new('RGB', (cols * 224, rows * 264), 'white')
        draw = ImageDraw.Draw(montage)
        for crop, slot in zip(crops, result['slots']):
            Image.fromarray(crop).save(args.output / ('r%02d_c%02d.png' % (slot['row'], slot['column'])))
            x, y = (slot['column'] - 1) * 224, (slot['row'] - 1) * 264
            montage.paste(Image.fromarray(crop), (x, y + 40))
            label = '%s | margin %.3f%s' % (slot['flavor_name'], slot['margin'], ' REVIEW' if slot['needs_review'] else '')
            draw.text((x + 3, y + 3), label, fill='red' if slot['needs_review'] else 'black')
        montage.save(args.output / 'predictions.png')
        print(json.dumps({k: v for k, v in result.items() if k != 'slots'}, indent=2))


if __name__ == '__main__':
    main()
