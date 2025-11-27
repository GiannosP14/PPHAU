import os
import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge
import shutil

# ==============================
# Parameters
# ==============================
BAG_FILE = "/workspace/data/group_d_seq_1_2025-10-30-16-13-14.bag"
NUM_SAMPLES = 10
RGB_TOPIC = "/camera/135122071615/color/image_raw/compressed"
DEPTH_TOPIC = "/camera/135122071615/aligned_depth_to_color/image_raw"
CAMERA_INFO_TOPIC = "/camera/135122071615/color/camera_info"

# ==============================
# Folders
# ==============================
EXTRACTED_RGB = "/workspace/data/extracted_samples/rgb"
EXTRACTED_DEPTH = "/workspace/data/extracted_samples/depth"
os.makedirs(EXTRACTED_RGB, exist_ok=True)
os.makedirs(EXTRACTED_DEPTH, exist_ok=True)

bridge = CvBridge()


# ==============================
# Extract camera intrinsics
# ==============================
def get_camera_intrinsics(bag_file, camera_info_topic):
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


fx, fy, cx, cy = get_camera_intrinsics(BAG_FILE, CAMERA_INFO_TOPIC)

# Save intrinsics to file
with open("/workspace/data/extracted_samples/intrinsics.txt", "w") as f:
    f.write(f"{fx},{fy},{cx},{cy}\n")

# ==============================
# Extract frames
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
print(f"Saved camera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

# ==============================
# Copy extracted samples to Homework2.1/data
# ==============================

DEST_FOLDER = "/workspace/scripts/data"
os.makedirs(DEST_FOLDER, exist_ok=True)
shutil.copytree("/workspace/data/extracted_samples", DEST_FOLDER, dirs_exist_ok=True)

print(f"Copied extracted_samples to {DEST_FOLDER}")
