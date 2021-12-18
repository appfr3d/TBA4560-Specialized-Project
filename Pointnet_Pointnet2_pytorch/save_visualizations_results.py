import os
import numpy as np
import open3d as o3d
import laspy

root = 'log/'
result_dirs = ['log/part_seg/gt',
               'log/part_seg/pointnet2_part_seg_msg_tr3d/results',
               'log/inst_seg/gt',
               'log/inst_seg/pointnet2_inst_seg_msg_tr3d/results',
               'log/inst_seg/pointnet2_inst_seg_msg_tr3d_3/results',
               'log/inst_seg/pointnet2_inst_seg_msg_tr3d_swap/results',
               'log/inst_seg/pointnet2_inst_seg_msg_tr3d_scale/results',
               'log/inst_seg/pointnet2_inst_seg_msg_tr3d_swap_scale/results']


for dir in result_dirs:
    fns = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

    for fn in fns:
        # Read file
        file_name = os.path.join(dir, fn)
        point_cloud = laspy.read(file_name)

        # Get points and color
        points = np.vstack(
            (point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
        colors = np.vstack(
            (point_cloud.red, point_cloud.green, point_cloud.blue)).transpose()

        # Create o3d point cloud with color
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors/65535)

        # Visualize
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # Save
        file_name = fn.split('.')[0]
        vis.capture_screen_image(os.path.join(
            dir, 'results', 'images', file_name + '.png'))
        vis.destroy_window()
