import os
import cv2
import numpy as np
import random
import shutil
import rosbag
from cv_bridge import CvBridge
from sklearn.cluster import KMeans
from scipy.ndimage import binary_dilation
from ultralytics import YOLO
import rospy
from sensor_msgs.msg import CameraInfo

# ============================================================
#               0. Extract Camera Intrinsics
# ============================================================

def get_intrinsics_from_bag(bag_file, camera_info_topic):
    print("Reading intrinsics from:", camera_info_topic)
    bag = rosbag.Bag(bag_file)
    for topic, msg, t in bag.read_messages(topics=[camera_info_topic]):
        fx = msg.K[0]
        fy = msg.K[4]
        cx = msg.K[2]
        cy = msg.K[5]
        bag.close()
        print("Found intrinsics:")
        print("fx =", fx, " fy =", fy, " cx =", cx, " cy =", cy)
        return fx, fy, cx, cy

    bag.close()
    raise RuntimeError("Could not find CameraInfo in bag file!")

# ============================================================
# Parameters
# ============================================================
bag_file = "PPHAU DATA/group_d_seq_1_2025-10-30-16-13-14.bag"
num_samples = 10
train_ratio = 0.8
yolo_weights = "yolov8n.pt"
n_clusters = 2

# ROS topics
rgb_topic = "/camera/135122071615/color/image_raw"
depth_topic = "/camera/135122071615/aligned_depth_to_color/image_raw"
camera_info_topic = "/camera/135122071615/color/camera_info"

# Extract intrinsics dynamically
fx, fy, cx, cy = get_intrinsics_from_bag(bag_file, camera_info_topic)

# ============================================================
# Folder creation
# ============================================================
extracted_rgb = "extracted_samples/rgb"
extracted_depth = "extracted_samples/depth"
dataset_images = "dataset/images"
dataset_labels = "dataset/labels"
overlay_out = "segmentation_overlays"

os.makedirs(extracted_rgb, exist_ok=True)
os.makedirs(extracted_depth, exist_ok=True)
os.makedirs(dataset_images + "/train", exist_ok=True)
os.makedirs(dataset_images + "/val", exist_ok=True)
os.makedirs(dataset_labels + "/train", exist_ok=True)
os.makedirs(dataset_labels + "/val", exist_ok=True)
os.makedirs(overlay_out, exist_ok=True)

bridge = CvBridge()

# ============================================================
# 1. Extract frames from bag file
# ============================================================
bag = rosbag.Bag(bag_file)
count = 0
for topic, msg, t in bag.read_messages(topics=[rgb_topic, depth_topic]):
    if count >= num_samples:
        break

    if topic == rgb_topic:
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imwrite(f"{extracted_rgb}/frame_{count:04d}.png", cv_image)

    elif topic == depth_topic:
        cv_depth = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        cv2.imwrite(f"{extracted_depth}/frame_{count:04d}.png", cv_depth)
        count += 1

bag.close()
print(f"Extracted {count} samples")

# ============================================================
# 2. Split into train/val
# ============================================================
images = [f for f in os.listdir(extracted_rgb) if f.endswith(".png")]
random.shuffle(images)
split_idx = int(len(images) * train_ratio)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

for img in train_imgs:
    shutil.copy(os.path.join(extracted_rgb, img), os.path.join(dataset_images, "train", img))
for img in val_imgs:
    shutil.copy(os.path.join(extracted_rgb, img), os.path.join(dataset_images, "val", img))

print(f"Dataset split -> Train: {len(train_imgs)}, Val: {len(val_imgs)}")

# ============================================================
# 3. Create dummy YOLO labels
# ============================================================
def create_dummy_labels(img_folder, label_folder):
    for img_file in os.listdir(img_folder):
        if not img_file.endswith(".png"):
            continue
        with open(os.path.join(label_folder, img_file.replace(".png", ".txt")), "w") as f:
            f.write("0 0.5 0.5 1.0 1.0\n")

create_dummy_labels(dataset_images + "/train", dataset_labels + "/train")
create_dummy_labels(dataset_images + "/val", dataset_labels + "/val")

print("Dummy YOLO labels created.")

# ============================================================
# 4. Write data.yaml
# ============================================================
train_abs = os.path.abspath(dataset_images + "/train")
val_abs   = os.path.abspath(dataset_images + "/val")

yaml_text = f"""train: {train_abs}
val: {val_abs}

nc: 2
names: ['object1', 'object2']
"""

with open("dataset/data.yaml", "w") as f:
    f.write(yaml_text)

print("Wrote corrected dataset/data.yaml:")
print(yaml_text)

# ============================================================
# 5. Train YOLO
# ============================================================
model = YOLO(yolo_weights)
model.train(data=os.path.abspath("dataset/data.yaml"), epochs=10, imgsz=640)

# ============================================================
# 6. Depth → XYZ conversion
# ============================================================
def depth_to_xyz(depth_img, fx, fy, cx, cy):
    h, w = depth_img.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth_img.astype(np.float32) / 1000.0
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    return np.stack((X, Y, Z), axis=-1)

# ============================================================
# 7. Segmentation color overlay
# ============================================================
def overlay_function(image, mask, alpha=0.5):
    color_map = np.array([
        [0, 0, 0],
        [255, 255, 50],
        [255, 50, 50],
        [50, 255, 50],
    ])
    output = image.copy()
    colored = color_map[mask]
    output = (image * alpha + colored * (1 - alpha)).astype(np.uint8)
    return output

# ============================================================
# 8. YOLO + KMeans segmentation 
# ============================================================

def clean_and_cluster(roi_rgb, roi_xyz, n_clusters):
    """
    Performs K-means on combined RGB + XYZ features.
    Handles invalid depth values & returns clean mask.
    """
    h, w = roi_rgb.shape[:2]

    # Flatten
    rgb_flat = roi_rgb.reshape(-1, 3).astype(np.float32)
    xyz_flat = roi_xyz.reshape(-1, 3).astype(np.float32)

    # Remove invalid 3D points
    valid = np.isfinite(xyz_flat).all(axis=1) & (xyz_flat[:, 2] > 0)

    if valid.sum() < n_clusters:
        # Not enough valid pixels → return empty mask
        return np.zeros((h, w), dtype=np.int32)

    features = np.concatenate([rgb_flat[valid], xyz_flat[valid]], axis=1)

    # Run KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
    labels = np.full(rgb_flat.shape[0], -1, dtype=np.int32)
    labels[valid] = kmeans.labels_

    # Reshape back
    labels = labels.reshape(h, w)

    # Make cluster with highest depth (closest object) = foreground
    cluster_means = []
    for k in range(n_clusters):
        depth_vals = xyz_flat[labels.reshape(-1) == k][:, 2]
        cluster_means.append(depth_vals.mean() if len(depth_vals) > 0 else 1e9)

    # Foreground = smallest Z (closest to camera)
    fg_cluster = np.argmin(cluster_means)

    # Produce binary mask: 1 = object, 0 = background
    object_mask = (labels == fg_cluster).astype(np.uint8)

    # Morphological cleanup
    object_mask = binary_dilation(object_mask, iterations=2)

    return object_mask


for i in range(num_samples):
    rgb_path = f"{extracted_rgb}/frame_{i:04d}.png"
    depth_path = f"{extracted_depth}/frame_{i:04d}.png"

    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        continue

    rgb = cv2.imread(rgb_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    xyz = depth_to_xyz(depth, fx, fy, cx, cy)

    results = model(rgb_path)

    # Full-frame segmentation mask
    segmentation_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Ensure valid ROI
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(rgb.shape[1], x2)
        y2 = min(rgb.shape[0], y2)

        roi_rgb = rgb[y1:y2, x1:x2]
        roi_xyz = xyz[y1:y2, x1:x2]

        if roi_rgb.size == 0:
            continue

        # Run improved K-means segmentation
        object_mask = clean_and_cluster(roi_rgb, roi_xyz, n_clusters)

        # Insert into global mask
        segmentation_mask[y1:y2, x1:x2] = object_mask * 255

    # Save binary mask
    cv2.imwrite(f"{overlay_out}/mask_{i:04d}.png", segmentation_mask)

    # Save overlay
    overlay = overlay_function(rgb, segmentation_mask // 255)
    cv2.imwrite(f"{overlay_out}/segmented_{i:04d}.png", overlay)

print("Segmentation completed successfully.")
