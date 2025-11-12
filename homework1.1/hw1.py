import cv2
import numpy as np
import os

# Parameters
focal_length = 970.0  
baseline = 50.0      
data_path = "HW1-1-data"
data_path_save = "HW1-1-output"

# The four image sets to test
pairs = ["1262", "1755", "1131", "0000"]

# Stereo parameters
num_disparities = 512 
block_size = 7

for pair_id in pairs:
    print(f"\nProcessing pair: {pair_id}")

    # Load infrared and color images
    ir1_path = os.path.join(data_path, f"infra1_{pair_id}.jpg")
    ir2_path = os.path.join(data_path, f"infra2_{pair_id}.jpg")
    color_path = os.path.join(data_path, f"color{pair_id}.jpg")  

    ir1 = cv2.imread(ir1_path, cv2.IMREAD_GRAYSCALE)
    ir2 = cv2.imread(ir2_path, cv2.IMREAD_GRAYSCALE)
    color = cv2.imread(color_path)

    if ir1 is None or ir2 is None or color is None:
        print("Could not load images for pair {pair_id}")
        continue

    # Create stereo block matcher
    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)

    # Compute disparity map
    disparity = stereo.compute(ir1, ir2).astype(np.float32) / 16.0
    disparity[disparity <= 0] = np.nan  # Handle invalid disparities by setting them to NaN

    # Compute depth map: Z = f * B / disparity
    depth = (focal_length * baseline) / disparity

    # Normalize for visualization
    disp_vis = cv2.normalize(np.nan_to_num(disparity, nan=0.0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_vis = cv2.normalize(np.nan_to_num(depth, nan=0.0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Save disparity and depth images
    cv2.imwrite(os.path.join(data_path_save, f"disparity_{pair_id}.jpg"), disp_vis)
    cv2.imwrite(os.path.join(data_path_save, f"depth_{pair_id}.jpg"), depth_vis)

    # Display images
    cv2.imshow(f"Infrared 1 - {pair_id}", ir1)
    cv2.imshow(f"Infrared 2 - {pair_id}", ir2)
    cv2.imshow(f"Disparity Map - {pair_id}", disp_vis)
    cv2.imshow(f"Depth Map - {pair_id}", depth_vis)

    print("Press any key to continue to the next pair...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print(f"\nDone! Disparity and depth maps saved in HW1-1-data folder.")
print(f"\nThe checkerboard provides high contrast features (edges, corners, patterns) making it easier for the algorithm to find correspondences between the stereo images, leading to more accurate disparity and depth estimation.")
print(f"\nIn contrast, a plane without texture (the PC case) looks almost identical across neighboor pixels. As a result, the stereo mathing cannot find that good correspondences, leading to poor disparity and depth estimation.")
print(f"\nLaser pattern adds artificial detail and contrast, which help textureless regions to match better. Result is higher quality disparity and depth maps.")
print(f"\nImages without laser pattern depend entirely on natural textures, which may be insufficient in some areas, leading to lower quality disparity and depth maps.")