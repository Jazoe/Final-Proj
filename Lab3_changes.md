# Detect.py — Changes from Original Lab3.py

## Original (Lab3.py)

Single-scale FFT cross-correlation at a fixed template size, no evaluation.

```python
img_zm = img.astype(np.float32) - np.mean(img)
img_fourier = np.fft.fft2(img_zm)

img_t_resized = np.zeros(img.shape, dtype=np.float32)
img_t_resized[:h, :w] = img_t

corr = np.real(np.fft.ifft2(img_fourier * np.conj(np.fft.fft2(img_t_resized))))
peaks = peak_local_max(-corr, min_distance=10, threshold_rel=0.5)
draw_template_boxes(img, peaks, template_shape=img_t.shape)
```

---

## Current (Detect.py)

### New functions

| Function | Purpose |
|---|---|
| `apply_circular_mask(template)` | Zeros pixels outside the inscribed circle so background corners don't contribute to correlation |
| `rotate_template(template, angle)` | Rotates with an expanded canvas so rotated content is never cropped |
| `nms(detections, iou_threshold)` | IoU-based NMS — defined but not used in the main pipeline; replaced by greedy center suppression |
| `iou_xyxy(a, b)` | IoU between two `(xmin, ymin, xmax, ymax)` boxes |
| `load_ground_truth(xml_path)` | Parses Pascal VOC XML → list of `(xmin, ymin, xmax, ymax)` |
| `evaluate(detections, gt, iou_threshold)` | Greedy TP/FP/FN matching; returns precision, recall, F1, per-detection IoU |
| `draw_detections(image, detections, ground_truth)` | Replaces `draw_template_boxes`; draws red detected boxes and dashed green GT boxes with a legend |

### Pipeline changes

| | Original | Current |
|---|---|---|
| Template scale | Fixed 1× | `np.linspace(0.25, 1.25, 20)` — 20 levels, largest first |
| Rotation | None | Every 30° (12 angles) |
| Template preprocessing | None | Circular mask applied after each resize |
| Correlation method | FFT cross-correlation | `cv.matchTemplate` with `TM_CCOEFF_NORMED` (OpenCV NCC) |
| Peak threshold | `threshold_rel=0.5` | `threshold_abs=0.65` — absolute NCC score on `[-1, 1]` scale |
| Peak finding | Single fixed map | Per `(scale, angle)` with `min_distance = max(10, min(h,w)//3)` |
| Post-processing | None | Greedy center-inside-box suppression, largest scale first |
| Ground truth | None | Loaded from Pascal VOC XML, overlaid on output |
| Metrics output | `print(peaks)` | TP, FP, FN, Precision, Recall, F1, per-detection IoU |

### Correlation method — cv.matchTemplate TM_CCOEFF_NORMED

Replaces FFT cross-correlation. `TM_CCOEFF_NORMED` computes:

```
NCC(r,c) = Σ[(I_patch - mean_I) · (T - mean_T)] / (‖I_patch_zm‖ · ‖T_zm‖)
```

Output is in `[-1, 1]`. Values near 1 indicate a strong match regardless of local brightness or contrast. This is the same quantity as NCC but computed by OpenCV's optimised sliding-window implementation rather than via FFT, which is simpler and gives identical results for the template sizes used here.

### Duplicate suppression — greedy center-inside-box

Scales processed largest → smallest. Within each `(scale, angle)`, peaks are sorted by score descending. A candidate is kept only if its center point `(row + h//2, col + w//2)` does not fall inside any already-confirmed box.

**Why center instead of IoU:** boxes at different scales have very different areas, making IoU unstable — a small box inside a large one scores near-zero IoU even though they represent the same coin. Center-point containment is scale-invariant and directly answers "is this the same coin?"

**Why largest first:** a coarse-scale detection claims the region before finer-scale duplicates can re-detect it, preventing many small boxes from accumulating inside one large coin.

### Tunable parameters

| Parameter | Value | Effect |
|---|---|---|
| `scales` | `linspace(0.25, 1.25, 20)` | Range and density of scale search |
| `angles` | `arange(0, 360, 30)` | Rotation search step |
| `threshold_abs` | `0.65` | Minimum NCC score to consider as a candidate; raise to reduce false positives |
| `IOU_THRESHOLD` | `0.3` | Used by `evaluate()` for TP/FP matching (not suppression) |
