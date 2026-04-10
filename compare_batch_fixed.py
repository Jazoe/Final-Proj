import os
import glob
import csv
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from matplotlib.patches import Rectangle
from skimage.feature import peak_local_max


def load_ground_truth(xml_path):
    if not os.path.exists(xml_path):
        return []
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall('object'):
        bb = obj.find('bndbox')
        boxes.append((
            int(bb.find('xmin').text),
            int(bb.find('ymin').text),
            int(bb.find('xmax').text),
            int(bb.find('ymax').text),
        ))
    return boxes


def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def evaluate(detections_rowcol, ground_truth_xyxy, iou_threshold=0.3):
    det_xyxy = [(x, y, x+w, y+h) for (y, x, h, w) in detections_rowcol]

    matched_gt = set()
    tp = 0

    for det in det_xyxy:
        best_iou, best_idx = 0.0, -1
        for i, gt in enumerate(ground_truth_xyxy):
            if i in matched_gt:
                continue
            iou = iou_xyxy(det, gt)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_idx)

    fp = len(det_xyxy) - tp
    fn = len(ground_truth_xyxy) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }


def draw_detections(image, detections, ground_truth=None, save_path='output.png', title=None):
    img = image.copy().astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0

    H, W = img.shape[:2]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)

    for (y, x, h_t, w_t) in detections:
        x0 = max(0, min(int(round(x)), W-1))
        y0 = max(0, min(int(round(y)), H-1))
        ax.add_patch(Rectangle((x0, y0), min(w_t, W-x0), min(h_t, H-y0),
                               linewidth=2, edgecolor='red', facecolor='none'))

    if ground_truth:
        for (xmin, ymin, xmax, ymax) in ground_truth:
            ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                   linewidth=2, edgecolor='lime',
                                   facecolor='none', linestyle='--'))

    if title:
        ax.set_title(title)

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05, dpi=180)
    plt.close()


def apply_circular_mask(template):
    h, w = template.shape
    Y, X = np.ogrid[:h, :w]
    mask = (X - w/2)**2 + (Y - h/2)**2 <= min(w/2, h/2)**2
    out = template.astype(np.float32, copy=True)
    out[~mask] = 0.0
    return out


def rotate_template(template, angle):
    h, w = template.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += new_w / 2 - cx
    M[1, 2] += new_h / 2 - cy
    return cv.warpAffine(template, M, (new_w, new_h))


def improved_detect(img, template):
    H, W = img.shape
    scales = np.linspace(0.25, 1.25, 20)
    angles = np.arange(0, 360, 30)

    confirmed = []

    def center_inside_confirmed(row, col, h_t, w_t):
        cy = row + h_t // 2
        cx = col + w_t // 2
        for (ky, kx, kh, kw, _) in confirmed:
            if ky <= cy < ky + kh and kx <= cx < kx + kw:
                return True
        return False

    img_float = img.astype(np.float32)

    for scale in sorted(scales, reverse=True):
        new_h = max(1, int(template.shape[0] * scale))
        new_w = max(1, int(template.shape[1] * scale))
        tmpl_scaled = apply_circular_mask(cv.resize(template, (new_w, new_h)))

        for angle in angles:
            tmpl = rotate_template(tmpl_scaled, angle)
            h_t, w_t = tmpl.shape
            if h_t > H or w_t > W:
                continue

            corr = cv.matchTemplate(img_float, tmpl, cv.TM_CCOEFF_NORMED)
            inverted = -corr
            min_dist = max(10, min(h_t, w_t) // 3)
            peaks = peak_local_max(inverted, min_distance=min_dist, threshold_abs=0.65)

            for (row, col) in sorted(peaks, key=lambda p: inverted[p[0], p[1]], reverse=True):
                if not center_inside_confirmed(row, col, h_t, w_t):
                    confirmed.append((row, col, h_t, w_t, inverted[row, col]))

    return [(d[0], d[1], d[2], d[3]) for d in confirmed]


def baseline_detect(img, template):
    H, W = img.shape
    h, w = template.shape

    if h > H or w > W:
        scale = min(H / h, W / w) * 0.9
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        template = cv.resize(template, (new_w, new_h), interpolation=cv.INTER_AREA)
        h, w = template.shape

    template_resized = np.zeros(img.shape, dtype=np.float32)
    template_resized[:h, :w] = template

    img_zm = img.astype(np.float32) - np.mean(img)
    img_fourier = np.fft.fft2(img_zm)

    template_fourier = np.fft.fft2(template_resized)
    cc_img_fourier = img_fourier * np.conj(template_fourier)
    cc_img = np.fft.ifft2(cc_img_fourier)
    corr = np.real(cc_img)

    inverted_corr = -corr
    peaks = peak_local_max(inverted_corr, min_distance=10, threshold_rel=0.5)

    detections = []
    for (y, x) in peaks:
        detections.append((int(y), int(x), int(h), int(w)))
    return detections


def main():
    image_folder = 'coins-1'
    template_path = 'cointemp.jpg'
    out_dir = 'comparison_outputs'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'improved'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'baseline'), exist_ok=True)

    csv_path = os.path.join(out_dir, 'comparison_results.csv')
    image_paths = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))
    template = cv.imread(template_path, 0)

    if template is None:
        raise FileNotFoundError(template_path)
    if not image_paths:
        raise FileNotFoundError(f'No images found in {image_folder}')

    fieldnames = [
        'filename',
        'status',
        'gt_count',
        'improved_count',
        'baseline_count',
        'improved_precision',
        'improved_recall',
        'improved_f1',
        'baseline_precision',
        'baseline_recall',
        'baseline_f1',
        'error_message',
    ]

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        img = cv.imread(image_path, 0)

        if img is None:
            row = {
                'filename': filename,
                'status': 'failed',
                'gt_count': '',
                'improved_count': '',
                'baseline_count': '',
                'improved_precision': '',
                'improved_recall': '',
                'improved_f1': '',
                'baseline_precision': '',
                'baseline_recall': '',
                'baseline_f1': '',
                'error_message': 'Could not read image',
            }
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)
            print(f'Skipped {filename}: could not read image')
            continue

        xml_path = os.path.splitext(image_path)[0] + '.xml'
        gt = load_ground_truth(xml_path)

        try:
            improved_det = improved_detect(img, template)
            baseline_det = baseline_detect(img, template)

            improved_metrics = evaluate(improved_det, gt, iou_threshold=0.3) if gt else {}
            baseline_metrics = evaluate(baseline_det, gt, iou_threshold=0.3) if gt else {}

            row = {
                'filename': filename,
                'status': 'ok',
                'gt_count': len(gt),
                'improved_count': len(improved_det),
                'baseline_count': len(baseline_det),
                'improved_precision': improved_metrics.get('precision', ''),
                'improved_recall': improved_metrics.get('recall', ''),
                'improved_f1': improved_metrics.get('f1', ''),
                'baseline_precision': baseline_metrics.get('precision', ''),
                'baseline_recall': baseline_metrics.get('recall', ''),
                'baseline_f1': baseline_metrics.get('f1', ''),
                'error_message': '',
            }

            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)

            print(f'Processed {filename}')

        except Exception as e:
            row = {
                'filename': filename,
                'status': 'failed',
                'gt_count': len(gt),
                'improved_count': '',
                'baseline_count': '',
                'improved_precision': '',
                'improved_recall': '',
                'improved_f1': '',
                'baseline_precision': '',
                'baseline_recall': '',
                'baseline_f1': '',
                'error_message': str(e),
            }

            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)

            print(f'Failed {filename}: {e}')
            continue

    print(f'\nSaved comparison table to: {csv_path}')


if __name__ == '__main__':
    main()
