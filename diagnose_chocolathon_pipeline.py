"""Stage-by-stage Chocolathon diagnosis on synthetic filled boxes."""
import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from chocolathon_inference import Chocolathon, CORNER_NAMES


def resolve_path(value, base):
    path = Path(value)
    if path.is_absolute() and path.exists(): return path
    for candidate in (base / path, base.parent / path, Path.cwd() / path):
        if candidate.exists(): return candidate.resolve()
    raise FileNotFoundError(value)


def parse_json(value):
    if isinstance(value, (dict, list)): return value
    return json.loads(value)


def parse_corners(row):
    value = row.get('corners') or row.get('corners_cam') or row.get('box_corners')
    if not value: raise ValueError('Manifest has no corners/corners_cam column.')
    value = parse_json(value)
    if isinstance(value, dict): return {k: value[k] for k in CORNER_NAMES}
    points = np.asarray(value, np.float32).reshape(4, 2)
    return {k: points[i].tolist() for i, k in enumerate(CORNER_NAMES)}


def load_truth(path):
    with path.open(newline='') as f: rows = list(csv.DictReader(f))
    if not rows: raise ValueError(f'Empty GT file: {path}')
    truth = {}
    for i, row in enumerate(rows):
        r, c = int(row.get('row', 0)), int(row.get('column', 0))
        if r < 1 or c < 1: raise ValueError(f'GT row/column must be 1-based: {path}')
        truth[(r, c)] = row
    return rows, truth


def score(result, truth):
    expected = Counter(r['flavor_id'] for r in truth.values() if r.get('label_type', 'occupied').lower() == 'occupied')
    predicted = Counter(result['flavor_counts'])
    pairs = {(s['row'], s['column']): s for s in result['slots']}
    comparable = set(pairs) == set(truth)
    occupied_total = sum(r.get('label_type', 'occupied').lower() == 'occupied' for r in truth.values())
    occupied_correct = flavor_correct = 0
    if comparable:
        for key, target in truth.items():
            pred, label = pairs[key], target.get('label_type', 'occupied').lower()
            occupied_correct += pred['label_type'] == label
            flavor_correct += label == 'occupied' and pred['flavor_id'] == target['flavor_id']
    keys = set(expected) | set(predicted)
    return {
        'slots_comparable': comparable,
        'expected_occupied': occupied_total,
        'predicted_occupied': result['occupied'],
        'false_empty': max(0, occupied_total - result['occupied']),
        'occupancy_correct': occupied_correct if comparable else 0,
        'flavor_correct': flavor_correct if comparable else 0,
        'count_abs_error': sum(abs(expected[k] - predicted[k]) for k in keys),
        'exact_flavor_counts': expected == predicted,
    }


def summarize(rows, prefix):
    n = len(rows); slots = sum(r['expected_slots'] for r in rows); occupied = sum(r[f'{prefix}_expected_occupied'] for r in rows)
    comparable = [r for r in rows if r[f'{prefix}_slots_comparable']]
    return {
        'images': n,
        'comparable_images': len(comparable),
        'slot_occupancy_accuracy': sum(r[f'{prefix}_occupancy_correct'] for r in rows) / slots,
        'occupied_flavor_accuracy': sum(r[f'{prefix}_flavor_correct'] for r in rows) / max(occupied, 1),
        'exact_flavor_count_accuracy': sum(r[f'{prefix}_exact_flavor_counts'] for r in rows) / n,
        'false_empty': sum(r[f'{prefix}_false_empty'] for r in rows),
        'flavor_count_abs_error': sum(r[f'{prefix}_count_abs_error'] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--synthetic-root', type=Path, default=Path('data/synthetic_filled_boxes/v4'))
    parser.add_argument('--output', type=Path, default=Path('outputs/stage_diagnosis'))
    parser.add_argument('--geometry', choices=['uniform', 'training-centers'], default='uniform')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    root, synthetic = args.root.resolve(), args.synthetic_root.resolve()
    manifest = synthetic / 'manifest.csv'
    if not manifest.exists(): raise FileNotFoundError(manifest)
    with manifest.open(newline='') as f: items = list(csv.DictReader(f))
    if not items: raise ValueError('Synthetic manifest is empty.')
    engine = Chocolathon(root, args.device, args.threads, args.batch_size, 'original', args.geometry)
    rows = []
    for index, item in enumerate(items, 1):
        image = resolve_path(item.get('image') or item.get('image_path') or item.get('file'), synthetic)
        gt = resolve_path(item.get('gt_path') or item.get('slots_csv') or item.get('ground_truth'), synthetic)
        gt_rows, truth = load_truth(gt)
        capacity = int(item.get('box_size') or gt_rows[0].get('box_size') or gt_rows[0].get('capacity'))
        rgb = cv2.imread(str(image))
        if rgb is None: raise ValueError(f'Cannot decode {image}')
        height, width = rgb.shape[:2]
        corners = parse_corners(item)
        annotation = {'image_width': width, 'image_height': height, 'box_size': capacity, 'corners': corners}
        automatic, _, _, _ = engine.predict(image)
        assisted, _, _, _ = engine.predict(image, annotation=annotation)
        auto_score, assisted_score = score(automatic, truth), score(assisted, truth)
        gt_points = np.asarray([corners[k] for k in CORNER_NAMES], np.float32)
        pred_points = np.asarray(automatic['corners'], np.float32)
        corner_error = float(np.linalg.norm(pred_points - gt_points, axis=1).mean())
        row = {
            'image': image.name, 'expected_capacity': capacity, 'predicted_capacity': automatic['predicted_capacity'],
            'capacity_correct': automatic['predicted_capacity'] == capacity, 'expected_slots': len(gt_rows),
            'corner_error_pixels': corner_error, 'corner_error_normalized': corner_error / np.hypot(width, height),
        }
        row.update({f'auto_{k}': v for k, v in auto_score.items()})
        row.update({f'assisted_{k}': v for k, v in assisted_score.items()})
        rows.append(row)
        print(f'[{index:02d}/{len(items)}] {image.name}: capacity={capacity}/{automatic["predicted_capacity"]} '
              f'corner={corner_error:.1f}px auto_empty={auto_score["false_empty"]} assisted_empty={assisted_score["false_empty"]} '
              f'auto_flavor={auto_score["flavor_correct"]}/{auto_score["expected_occupied"]} '
              f'assisted_flavor={assisted_score["flavor_correct"]}/{assisted_score["expected_occupied"]}')
    report = {
        'images': len(rows),
        'capacity_accuracy': sum(r['capacity_correct'] for r in rows) / len(rows),
        'mean_corner_error_pixels': float(np.mean([r['corner_error_pixels'] for r in rows])),
        'mean_corner_error_normalized': float(np.mean([r['corner_error_normalized'] for r in rows])),
        'automatic': summarize(rows, 'auto'),
        'ground_truth_geometry': summarize(rows, 'assisted'),
    }
    report['diagnosis'] = {
        'false_empty_removed_by_gt_geometry': report['automatic']['false_empty'] - report['ground_truth_geometry']['false_empty'],
        'flavor_errors_removed_by_gt_geometry': report['automatic']['flavor_count_abs_error'] - report['ground_truth_geometry']['flavor_count_abs_error'],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'stage_report.json').write_text(json.dumps(report, indent=2))
    with (args.output / 'stage_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    by_capacity = defaultdict(list)
    for row in rows: by_capacity[row['expected_capacity']].append(row)
    print('\nSUMMARY\n' + json.dumps(report, indent=2))
    print('\nBY CAPACITY')
    for capacity, group in sorted(by_capacity.items()):
        print(capacity, 'images=', len(group), 'capacity_correct=', sum(r['capacity_correct'] for r in group),
              'auto_false_empty=', sum(r['auto_false_empty'] for r in group),
              'assisted_false_empty=', sum(r['assisted_false_empty'] for r in group))


if __name__ == '__main__': main()
