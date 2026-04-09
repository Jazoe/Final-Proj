import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from matplotlib.patches import Rectangle
from skimage.feature import peak_local_max


# Helpers

def load_ground_truth(xml_path):
    """Parse Pascal VOC XML, return list of (xmin, ymin, xmax, ymax)."""
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall('object'):
        bb = obj.find('bndbox')
        boxes.append((
            int(bb.find('xmin').text), int(bb.find('ymin').text),
            int(bb.find('xmax').text), int(bb.find('ymax').text),
        ))
    return boxes


def apply_circular_mask(template):
    """Zero out pixels outside the largest inscribed circle of the template."""
    h, w = template.shape
    Y, X = np.ogrid[:h, :w]
    mask = (X - w/2)**2 + (Y - h/2)**2 <= min(w/2, h/2)**2
    out = template.astype(np.float32, copy=True)
    out[~mask] = 0.0
    return out


def rotate_template(template, angle):
    """Rotate template by angle degrees, expanding canvas so content is not cropped."""
    h, w = template.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += new_w / 2 - cx
    M[1, 2] += new_h / 2 - cy
    return cv.warpAffine(template, M, (new_w, new_h))


def fft_correlate(img_f, template, img_shape):
    """FFT cross-correlation, normalized by template energy for cross-scale comparison."""
    H, W = img_shape
    h_t, w_t = template.shape
    tmpl_padded = np.zeros((H, W), dtype=np.float32)
    tmpl_padded[:h_t, :w_t] = template.astype(np.float32)
    tmpl_f = np.fft.fft2(tmpl_padded)
    corr = np.real(np.fft.ifft2(img_f * np.conj(tmpl_f)))
    energy = np.sum(template.astype(np.float64) ** 2)
    if energy > 0:
        corr /= energy
    return corr


def nms(detections, iou_threshold=0.3):
    """IoU-based non-maximum suppression. detections: (row, col, h, w, score)."""
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d[4], reverse=True)
    kept = []
    while detections:
        best = detections.pop(0)
        kept.append(best)
        y1, x1, h1, w1, _ = best
        remaining = []
        for det in detections:
            y2, x2, h2, w2, _ = det
            inter = max(0, min(y1+h1, y2+h2) - max(y1, y2)) * \
                    max(0, min(x1+w1, x2+w2) - max(x1, x2))
            union = h1*w1 + h2*w2 - inter
            if inter / union < iou_threshold if union > 0 else True:
                remaining.append(det)
        detections = remaining
    return kept


def iou_xyxy(a, b):
    """IoU between two (xmin, ymin, xmax, ymax) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0


def evaluate(detections_rowcol, ground_truth_xyxy, iou_threshold=0.5):
    """
    Compare detections (row, col, h, w) against ground truth (xmin, ymin, xmax, ymax).
    Returns dict with TP, FP, FN, precision, recall, F1, and per-GT IoU.
    """
    # Convert detections to (xmin, ymin, xmax, ymax)
    det_xyxy = [(x, y, x+w, y+h) for (y, x, h, w) in detections_rowcol]

    matched_gt = set()
    tp = 0
    per_det_iou = []

    for det in det_xyxy:
        best_iou, best_idx = 0.0, -1
        for i, gt in enumerate(ground_truth_xyxy):
            if i in matched_gt:
                continue
            iou = iou_xyxy(det, gt)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        per_det_iou.append(best_iou)
        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_idx)

    fp = len(det_xyxy) - tp
    fn = len(ground_truth_xyxy) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2*prec*rec / (prec+rec) if (prec+rec) else 0.0

    return dict(TP=tp, FP=fp, FN=fn,
                precision=prec, recall=rec, f1=f1,
                per_det_iou=per_det_iou)


def draw_detections(image, detections, ground_truth=None,
                    figsize=(12, 12), linewidth=2, save_path="output.png"):
    """
    Draw detections (red) and optionally ground truth boxes (green).
    detections : list of (row, col, h, w)
    ground_truth: list of (xmin, ymin, xmax, ymax)
    """
    img = image.copy().astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    H, W = img.shape[:2]

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)

    for (y, x, h_t, w_t) in detections:
        x0 = max(0, min(int(round(x)), W-1))
        y0 = max(0, min(int(round(y)), H-1))
        ax.add_patch(Rectangle((x0, y0), min(w_t, W-x0), min(h_t, H-y0),
                                linewidth=linewidth, edgecolor='red', facecolor='none'))

    if ground_truth:
        for (xmin, ymin, xmax, ymax) in ground_truth:
            ax.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                                    linewidth=linewidth, edgecolor='lime', facecolor='none',
                                    linestyle='--'))

    # Legend
    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], color='red',  lw=2, label='Detected'),
              Line2D([0], [0], color='lime', lw=2, linestyle='--', label='Ground truth')]
    ax.legend(handles=legend, loc='upper right', fontsize=10)

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.show()


# Load
img_t = cv.imread('cointemp.jpg', 0)
img   = cv.imread('coins-1/20210324_151121.jpg', 0)
gt    = load_ground_truth('coins-1/20210324_151121.xml')

H, W = img.shape

img_zm = img.astype(np.float32) - np.mean(img)
img_f  = np.fft.fft2(img_zm)

# Search parameters
scales        = np.linspace(0.25, 1, 11)
angles        = np.arange(0, 360, 30)
THRESHOLD_REL = 0.5
IOU_THRESHOLD = 0.3

# Multi-scale, multi-rotation correlation
# Largest scale first — big coins are confirmed before smaller scales run,
# so any peak whose center falls inside an already-confirmed box is skipped.

confirmed = []  # (row, col, h, w, score)

def center_inside_confirmed(row, col, h_t, w_t):
    """Check if the CENTER of this detection falls inside any confirmed box."""
    cy = row + h_t // 2
    cx = col + w_t // 2
    for (ky, kx, kh, kw, _) in confirmed:
        if ky <= cy < ky + kh and kx <= cx < kx + kw:
            return True
    return False

for scale in sorted(scales, reverse=True):
    new_h = max(1, int(img_t.shape[0] * scale))
    new_w = max(1, int(img_t.shape[1] * scale))
    tmpl_scaled = apply_circular_mask(cv.resize(img_t, (new_w, new_h)))

    for angle in angles:
        tmpl = rotate_template(tmpl_scaled, angle)
        h_t, w_t = tmpl.shape
        if h_t > H or w_t > W:
            continue

        corr = fft_correlate(img_f, tmpl, (H, W))
        inverted = -corr
        min_dist = max(10, min(h_t, w_t) // 3)
        peaks = peak_local_max(inverted, min_distance=min_dist,
                               threshold_rel=THRESHOLD_REL)

        # Best-scoring peak at this scale/angle confirmed first
        for (row, col) in sorted(peaks, key=lambda p: inverted[p[0], p[1]], reverse=True):
            if not center_inside_confirmed(row, col, h_t, w_t):
                confirmed.append((row, col, h_t, w_t, inverted[row, col]))

final_detections = confirmed

# Evaluate against ground truth
det_boxes = [(d[0], d[1], d[2], d[3]) for d in final_detections]
metrics   = evaluate(det_boxes, gt, iou_threshold=0.5)

print(f"\nDetected: {len(final_detections)}   Ground truth: {len(gt)}")
print(f"TP={metrics['TP']}  FP={metrics['FP']}  FN={metrics['FN']}")
print(f"Precision={metrics['precision']:.2f}  Recall={metrics['recall']:.2f}  F1={metrics['f1']:.2f}")
print(f"\nPer-detection best IoU: {[round(v,2) for v in metrics['per_det_iou']]}")

# Draw
draw_detections(img, det_boxes, ground_truth=gt)
