import os
import numpy as np
import open3d as o3d
import laspy
import time

root = 'log/'
result_dirs = ['log/part_seg/gt',
               'log/part_seg/pointnet2_part_seg_msg_tr3d/results',
               'log/inst_seg/gt',
               'log/inst_seg/pointnet2_inst_seg_msg_tr3d/results',
               'log/inst_seg/pointnet2_inst_seg_msg_tr3d_3/results',
               'log/inst_seg/pointnet2_inst_seg_msg_tr3d_scale/results',
               'log/inst_seg/pointnet2_inst_seg_msg_tr3d_swap_scale/results',
               'log/inst_seg/pointnet2_inst_seg_msg_tr3d_swap/results',
               'log/inst_seg/pointnet2_inst_seg_msg_tr3d_swap_scale_new/results']


# Create window
vis = o3d.visualization.Visualizer()
vis.create_window(width=1000, height=1000)



for dir in result_dirs:
    print('Creating images for', dir, '...', end=' ')
    if not os.path.exists(os.path.join(dir, 'images')):
        os.makedirs(os.path.join(dir, 'images'))

    fns = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    
    for fn in fns:
        # Read file
        file_name = os.path.join(dir, fn) # 'Cross Element.las') # 
        point_cloud = laspy.read(file_name)

        # Get points and color
        points = np.vstack(
            (point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
        colors = np.vstack(
            (point_cloud.red, point_cloud.green, point_cloud.blue)).transpose()

        # Create o3d point cloud with color
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors/255)

        vis.add_geometry(pcd)
        vis.get_render_option().load_from_json("visualization_viewpoint.json") # set point size
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # Set viewpoint
        ctr = vis.get_view_control()
        ctr.change_field_of_view(60)
        ctr.set_lookat([ -0.059999999999999998, 0.010000000000000009, -0.0050000000000000044])
        ctr.set_up([ 0.020758910712584521, 0.86761340229561579, 0.49680584918361509 ])
        ctr.set_zoom(0.69999999999999996)
        ctr.set_front([ -0.29112078289923043, -0.47013475316588499, 0.83319985815516784 ])
        

        # vis.run()
        # time.sleep(5)

        # vis.get_render_option().save_to_json("visualization_viewpoint.json")

        # break 
        # Save
        file_name = fn.split('.')[0]
        vis.capture_screen_image(os.path.join(
            dir, 'images', file_name + '.png'))
        
        # Reset
        vis.remove_geometry(pcd)
    # break
    print('done!')
        

vis.destroy_window()
