import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.ndimage import binary_dilation
from ultralytics import YOLO

# ==============================
# Parameters
# ==============================
NUM_SAMPLES = 10
N_CLUSTERS = 2
YOLO_WEIGHTS = "yolov8n.pt"

# ==============================
# Folders
# ==============================
EXTRACTED_RGB = "data/rgb"
EXTRACTED_DEPTH = "data/depth"
INTRINSICS_FILE = "data/intrinsics.txt"
OVERLAY_OUT = "results"
os.makedirs(OVERLAY_OUT, exist_ok=True)

# ==============================
# Camera intrinsics
# ==============================
with open(INTRINSICS_FILE, "r") as f:
    fx, fy, cx, cy = map(float, f.readline().strip().split(","))


# ==============================
# Depth → XYZ
# ==============================
def depth_to_xyz(depth_img, fx, fy, cx, cy):
    h, w = depth_img.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth_img.astype(np.float32) / 1000.0
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    return np.stack((X, Y, Z), axis=-1)


# ==============================
# KMeans segmentation
# ==============================
def clean_and_cluster(roi_rgb, roi_xyz, n_clusters=N_CLUSTERS):
    h, w = roi_rgb.shape[:2]
    rgb_flat = roi_rgb.reshape(-1, 3).astype(np.float32)
    xyz_flat = roi_xyz.reshape(-1, 3).astype(np.float32)
    valid = np.isfinite(xyz_flat).all(axis=1) & (xyz_flat[:, 2] > 0)
    if valid.sum() < n_clusters:
        return np.zeros((h, w), dtype=np.uint8)
    features = np.concatenate([rgb_flat[valid], xyz_flat[valid]], axis=1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    labels = np.full(rgb_flat.shape[0], -1, dtype=np.int32)
    labels[valid] = kmeans.labels_
    labels = labels.reshape(h, w)
    cluster_means = [
        xyz_flat[labels.reshape(-1) == k][:, 2].mean() if np.any(labels == k) else 1e9
        for k in range(n_clusters)
    ]
    fg_cluster = np.argmin(cluster_means)
    mask = (labels == fg_cluster).astype(np.uint8)
    mask = binary_dilation(mask, iterations=2)
    return mask


# ==============================
# Overlay function
# ==============================
def overlay_function(image, mask, alpha=0.5):
    color_map = np.array([[0, 0, 0], [255, 255, 50]])
    colored = color_map[mask]
    return (image * alpha + colored * (1 - alpha)).astype(np.uint8)


# ==============================
# Load YOLO model
# ==============================
model = YOLO(YOLO_WEIGHTS)

# ==============================
# Run full pipeline
# ==============================
for i in range(NUM_SAMPLES):
    rgb_path = os.path.join(EXTRACTED_RGB, f"frame_{i:04d}.png")
    depth_path = os.path.join(EXTRACTED_DEPTH, f"frame_{i:04d}.png")
    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        print(f"Skipping frame {i}: file missing")
        continue

    rgb = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    xyz = depth_to_xyz(depth, fx, fy, cx, cy)

    results = model(rgb_path)  # YOLO detection

    segmentation_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(rgb.shape[1], x2), min(rgb.shape[0], y2)
        roi_rgb = rgb[y1:y2, x1:x2]
        roi_xyz = xyz[y1:y2, x1:x2]
        if roi_rgb.size == 0:
            continue
        mask = clean_and_cluster(roi_rgb, roi_xyz)
        segmentation_mask[y1:y2, x1:x2] = mask

    # Save outputs
    cv2.imwrite(os.path.join(OVERLAY_OUT, f"mask_{i:04d}.png"), segmentation_mask * 255)
    overlay = overlay_function(rgb, segmentation_mask)
    cv2.imwrite(os.path.join(OVERLAY_OUT, f"segmented_{i:04d}.png"), overlay)

print("Segmentation completed successfully!")
