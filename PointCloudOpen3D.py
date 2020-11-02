import numpy as np
import cv2 as cv
import open3d as o3d
from matplotlib import image, pyplot
from imageio import imread

def lerp(x, x1, x2, y1, y2):
    x_span = x2 - x1
    y_span = y2 - y1
    scaled = (x - x1) / x_span
    return y1 + (scaled * y_span)

def pcl_from_open3d(color_image, depth_image, camera_matrix):
    color_o3d = o3d.Image(color_image.astype(np.uint8))
    depth_o3d = o3d.Image(depth_image.astype(np.uint8))
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.intrinsic_matrix = camera_matrix
    rgbd = o3d.create_rgbd_image_from_color_and_depth(color_o3d, depth_o3d, convert_rgb_to_intensity=False)
    pcl = o3d.create_point_cloud_from_rgbd_image(image=rgbd, intrinsic=intrinsics)
    pcl.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcl


def pcl_from_depth_image(depth_image, camera_matrix):
    far = 3.0
    depth_image = depth_image[:, :, 0] / 255.0
    depth_image = lerp(depth_image, 0.0, 1.0, 0.0, far)
    depth_o3d = o3d.Image(depth_image.astype(np.uint8))
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.intrinsic_matrix = camera_matrix
    pcl = o3d.create_point_cloud_from_depth_image(depth=depth_o3d, intrinsic=intrinsics)
    pcl.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcl


def main():
    f_depth_x = 1.0580353759137747e+03
    f_depth_y = 1.0604471766882732e+03
    c_depth_x = 9.5177505245974123e+02
    c_depth_y = 5.4839653660364445e+02
    depth_path1 = 'Depth/2028.png'
    img1_depth = np.array(imread(depth_path1))
    camera_matrix = np.array(
        [[1 / f_depth_x, 0., -c_depth_x / f_depth_x],
         [0., 1 / f_depth_y, -c_depth_y / f_depth_y],
         [0., 0., 1.]])
    camera_matrix_opencv = np.array(
        [[f_depth_x, 0., c_depth_x],
         [0., f_depth_y, c_depth_y],
         [0., 0., 1.]])
    # pcl1 = pcl_from_open3d(img1_RGB, img1_depth, camera_matrix)
    # pcl2 = pcl_from_open3d(img2_RGB, img2_depth, camera_matrix)
    pcl1 = pcl_from_depth_image(img1_depth, camera_matrix)
    o3d.write_point_cloud("pcl1.ply", pcl1)

if __name__ == '__main__':
    main()
