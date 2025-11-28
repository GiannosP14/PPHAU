import open3d as o3d
import os


# Method to save screenshopt, cause docker doesnt allow gui interaction, not necessary if run in Ubuntu with gui
def save_screenshot(pcd, filename, width=1024, height=768):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    vis.add_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(filename)
    vis.destroy_window()


# Task 1
pcd = o3d.io.read_point_cloud(
    "assets/models/oats/texturedMesh_alligned_vertex_color.ply"
)
print(f"original: num points is {len(pcd.points)}")

pcd_down = pcd.voxel_down_sample(voxel_size=0.01)
print(f"downsampling to {len(pcd_down.points)}")

os.makedirs("outputs", exist_ok=True)
o3d.io.write_point_cloud("outputs/textured_oats_voxel_10mm.ply", pcd_down)
print(f"saved downsampled point cloud to outputs/textured_oats_voxel_10mm.ply")

# for ubuntu uncomment below and comment the save_screenshot line
# o3d.visualization.draw_geometries([pcd_down])
save_screenshot(pcd_down, "outputs/textured_oats_voxel_10mm.png")

# Task 2
import numpy as np
from load_ros_bag import (
    load_sequence,
    color_topic,
    depth_topic,
    color_camera_info_topic,
)


def task2_from_bag(
    bag_path=os.path.join("/workspace/assets", "icp_tracking_oats.bag"),
    max_frames=50,
    outdir="outputs",
):
    os.makedirs(outdir, exist_ok=True)
    data, _ = load_sequence(bag_path)
    merged = o3d.geometry.PointCloud()

    for frame in data[:max_frames]:
        try:
            color = frame[color_topic]["msg"]
            depth = frame[depth_topic]["msg"]
            cam_info = frame[color_camera_info_topic]["msg"]
        except KeyError:
            continue

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth),
            depth_scale=1000.0,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False,
        )

        K = np.array(cam_info.K).reshape(3, 3)
        h, w = color.shape[:2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        merged += pcd.voxel_down_sample(0.005)

    merged_down = merged.voxel_down_sample(0.005)
    out_path = os.path.join(outdir, "realsense_merged.ply")
    o3d.io.write_point_cloud(out_path, merged_down)
    print("Saved merged point cloud to", out_path)

    # for ubuntu uncomment below and comment the save_screenshot line
    #    o3d.visualization.draw_geometries([merged_down])
    save_screenshot(merged_down, "outputs/realsense_merged.png")


if __name__ == "__main__":
    task2_from_bag()
