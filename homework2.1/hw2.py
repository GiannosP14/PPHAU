import os
import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge
from sklearn.cluster import KMeans
from scipy.ndimage import binary_dilation
from ultralytics import YOLO

# ==============================
# Parameters
# ==============================
BAG_FILE = "PPHAU DATA/group_d_seq_1_2025-10-30-16-13-14.bag"
NUM_SAMPLES = 10
RGB_TOPIC = "/camera/135122071615/color/image_raw/compressed"
DEPTH_TOPIC = "/camera/135122071615/aligned_depth_to_color/image_raw"
CAMERA_INFO_TOPIC = "/camera/135122071615/color/camera_info"
YOLO_WEIGHTS = "yolov8n.pt"
N_CLUSTERS = 2

# ==============================
# Folders
# ==============================
EXTRACTED_RGB = "extracted_samples/rgb"
EXTRACTED_DEPTH = "extracted_samples/depth"
OVERLAY_OUT = "segmentation_overlays"
os.makedirs(EXTRACTED_RGB, exist_ok=True)
os.makedirs(EXTRACTED_DEPTH, exist_ok=True)
os.makedirs(OVERLAY_OUT, exist_ok=True)

bridge = CvBridge()

# ==============================
# 0. Camera intrinsics
# ==============================
def get_intrinsics_from_bag(bag_file, camera_info_topic):
    bag = rosbag.Bag(bag_file)
    for topic, msg, t in bag.read_messages(topics=[camera_info_topic]):
        fx = msg.K[0]
        fy = msg.K[4]
        cx = msg.K[2]
        cy = msg.K[5]
        bag.close()
        return fx, fy, cx, cy
    bag.close()
    raise RuntimeError("No CameraInfo found!")

fx, fy, cx, cy = get_intrinsics_from_bag(BAG_FILE, CAMERA_INFO_TOPIC)

# ==============================
# 1. Extract frames
# ==============================
bag = rosbag.Bag(BAG_FILE)
rgb_count = 0
depth_count = 0
for topic, msg, t in bag.read_messages(topics=[RGB_TOPIC, DEPTH_TOPIC]):
    if rgb_count >= NUM_SAMPLES and depth_count >= NUM_SAMPLES:
        break
    if topic == RGB_TOPIC and rgb_count < NUM_SAMPLES:
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv2.imwrite(f"{EXTRACTED_RGB}/frame_{rgb_count:04d}.png", cv_image)
        rgb_count += 1
    elif topic == DEPTH_TOPIC and depth_count < NUM_SAMPLES:
        cv_depth = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        cv2.imwrite(f"{EXTRACTED_DEPTH}/frame_{depth_count:04d}.png", cv_depth)
        depth_count += 1
bag.close()
print(f"Extracted {rgb_count} RGB frames and {depth_count} Depth frames")

# ==============================
# 2. Depth → XYZ
# ==============================
def depth_to_xyz(depth_img, fx, fy, cx, cy):
    h, w = depth_img.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth_img.astype(np.float32) / 1000.0
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    return np.stack((X, Y, Z), axis=-1)

# ==============================
# 3. KMeans segmentation
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
    cluster_means = [xyz_flat[labels.reshape(-1) == k][:, 2].mean() if np.any(labels == k) else 1e9
                     for k in range(n_clusters)]
    fg_cluster = np.argmin(cluster_means)
    mask = (labels == fg_cluster).astype(np.uint8)
    mask = binary_dilation(mask, iterations=2)
    return mask

# ==============================
# 4. Overlay
# ==============================
def overlay_function(image, mask, alpha=0.5):
    color_map = np.array([[0,0,0],[255,255,50]])
    colored = color_map[mask]
    return (image*alpha + colored*(1-alpha)).astype(np.uint8)

# ==============================
# 5. Load YOLO model
# ==============================
model = YOLO(YOLO_WEIGHTS)

# ==============================
# 6. Run full pipeline
# ==============================
for i in range(NUM_SAMPLES):
    rgb_path = f"{EXTRACTED_RGB}/frame_{i:04d}.png"
    depth_path = f"{EXTRACTED_DEPTH}/frame_{i:04d}.png"
    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
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

    cv2.imwrite(f"{OVERLAY_OUT}/mask_{i:04d}.png", segmentation_mask*255)
    overlay = overlay_function(rgb, segmentation_mask)
    cv2.imwrite(f"{OVERLAY_OUT}/segmented_{i:04d}.png", overlay)

print("Segmentation completed successfully!")
