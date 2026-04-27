"""
Basic usage example - demonstrates the basic usage of BCPD
"""
import os
import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import open3d as o3d
# from pybcpd import bcpd, bcpd_advanced, TransformationType
from pybcpd import register


# absPath = os.path.dirname(os.path.abspath(__file__))

def custom_draw_geometry(vis, geometry_list, map_file=None, recording=False, param_file='camera_view.json', save_fov=False):
    vis.clear_geometries()
    # R = o3d.geometry.get_rotation_matrix_from_xyz([-np.pi/2,0,0])
    for pcd in geometry_list:
        # pcd.rotate(R,[0,0,0])
        vis.add_geometry(pcd)
    param = o3d.io.read_pinhole_camera_parameters(param_file)
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
    ctr = vis.get_view_control()
    # ctr.set_zoom(0.4)
    # ctr.set_up((0, -1, 0))
    # ctr.set_front((1, 0, 0))
    ctr.convert_from_pinhole_camera_parameters(param)
    # vis.register_animation_callback(rotate_view)
    vis.run()
    # time.sleep(5)
    if save_fov:
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters('camera_view.json', param)
    if recording:
        vis.capture_screen_image(map_file, True)
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        # if test == 1:
        o3d.io.write_pinhole_camera_parameters(param_file, param)

def main():
    # Create sample point clouds
    np.random.seed(42)  # Set random seed for reproducible results
    # absPath = os.path.dirname(os.path.abspath(__file__))
    X = sio.loadmat('./src.mat')['X']  # Target point cloud: 100 points in 3D space
    Y = sio.loadmat('./tgt.mat')['Y']  # Source point cloud: 80 points in 3D space

    # Run the BCPD algorithm
    result = register(
        X, Y, random_seed=1,
        omega=0.0, lambda_param=20.0, gamma=10.0, beta=2.0,
        max_iter=500, nystrom_J=300, nystrom_K=70, normalization_type='e'
    )

    # print(f"Estimated number of inliers (N_hat): {result['N_hat']}")
    showFlag = 0
    if showFlag == 0:
        # Read trajectory data
        try:
            with open('.optpath.bin', 'rb') as f:
                import struct

                # Read header information
                N = struct.unpack('i', f.read(4))[0]
                D = struct.unpack('i', f.read(4))[0]
                M = struct.unpack('i', f.read(4))[0]
                lp = struct.unpack('i', f.read(4))[0]

                # Read trajectory data
                trajectory = np.fromfile(f, dtype=np.float64, count=lp * D * M)
                trajectory = trajectory.reshape(lp, M, D)

            # Create animation
            vis = o3d.visualization.Visualizer()
            vis.create_window(width=960 * 2, height=640 * 2, left=5, top=5)
            vis.get_render_option().background_color = np.array([0, 0, 0])
            # vis.get_render_option().background_color = np.array([1, 1, 1])
            vis.get_render_option().show_coordinate_frame = False
            vis.get_render_option().point_size = 1.0
            vis.get_render_option().line_width = 3.0
            vis_list = []

            pcd_src = o3d.geometry.PointCloud()
            pcd_src.points = o3d.utility.Vector3dVector(X)
            pcd_src.paint_uniform_color([0, 1.0, 0])
            # vis_list += [pcd_src]

            pcd_tgt = o3d.geometry.PointCloud()
            pcd_tgt.points = o3d.utility.Vector3dVector(Y)
            pcd_tgt.paint_uniform_color([1.0, 0, 0])
            # vis_list += [pcd_tgt]

            for i in lp:
                vis_list = []
                vis_list += [pcd_src]
                vis_list += [pcd_tgt]
                pcd_pred = o3d.geometry.PointCloud()
                pcd_pred.points = o3d.utility.Vector3dVector(trajectory[i, :, :])
                pcd_pred.paint_uniform_color([0.5, 0.5, 0.5])
                vis_list += [pcd_pred]
                # pcd_pred = o3d.geometry.PointCloud()
                # flow = result["displacement_field"]
                # Y_trans = Y + flow
                # pcd_pred.points = o3d.utility.Vector3dVector(Y_trans)
                # pcd_pred.paint_uniform_color([0,0,1.0])
                # vis_list += [pcd_pred]
                custom_draw_geometry(vis, vis_list, map_file=None, recording=False, param_file='camera_view.json', save_fov=False)
        except FileNotFoundError:
            print("Trajectory file not found, unable to create animation")
        except Exception as e:
            print(f"Error while creating animation: {e}")

    elif showFlag == 1:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=960 * 2, height=640 * 2, left=5, top=5)
        vis.get_render_option().background_color = np.array([0, 0, 0])
        # vis.get_render_option().background_color = np.array([1, 1, 1])
        vis.get_render_option().show_coordinate_frame = False
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().line_width = 3.0
        vis_list = []

        pcd_src = o3d.geometry.PointCloud()
        # flow = result["displacement_field"]
        # Y_trans = Y + flow
        pcd_src.points = o3d.utility.Vector3dVector(X)
        pcd_src.paint_uniform_color([0, 1.0, 0])
        vis_list += [pcd_src]

        pcd_tgt = o3d.geometry.PointCloud()
        pcd_tgt.points = o3d.utility.Vector3dVector(Y)
        pcd_tgt.paint_uniform_color([1.0, 0, 0])
        vis_list += [pcd_tgt]

        pcd_pred = o3d.geometry.PointCloud()

        pcd_pred.points = o3d.utility.Vector3dVector(result['transformed_points'])
        pcd_pred.paint_uniform_color([0, 0, 1.0])
        vis_list += [pcd_pred]
        custom_draw_geometry(vis, vis_list, map_file=None, recording=False, param_file='camera_view.json', save_fov=False)
    else:
        # Visualize results
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot points
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='blue', label='Target (X)')
        ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2],
                   c='red', label='Source (Y)')
        ax.scatter(result['Y_transformed'][:, 0], result['Y_transformed'][:, 1],
                   result['Y_transformed'][:, 2], c='green', label='Transformed (Y)')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.title('BCPD Registration Result')
        plt.show()

        # Plot convergence behavior
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(result['profile'][:, 0])
        plt.title('Computation Time per Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Time (s)')

        plt.subplot(1, 2, 2)
        plt.plot(result['profile'][:, 2])
        plt.title('Variance (sigma) Change')
        plt.xlabel('Iteration')
        plt.ylabel('Sigma')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()