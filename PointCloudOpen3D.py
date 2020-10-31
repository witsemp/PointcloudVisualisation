import numpy as np
import cv2 as cv
import open3d as o3d


def pcl_from_open3d(color_image, depth_image, camera_matrix):
    color_o3d = o3d.Image(color_image.astype(np.uint8))
    depth_o3d = o3d.Image(depth_image.astype(np.uint8))
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.intrinsic_matrix = camera_matrix
    rgbd = o3d.create_rgbd_image_from_color_and_depth(color_o3d, depth_o3d, convert_rgb_to_intensity=False)
    pcl = o3d.create_point_cloud_from_rgbd_image(image=rgbd, intrinsic=intrinsics)
    pcl.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcl


def main():
    f_depth_x = 1.0580353759137747e+03
    f_depth_y = 1.0604471766882732e+03
    c_depth_x = 9.5177505245974123e+02
    c_depth_y = 5.4839653660364445e+02
    img1_RGB = cv.imread('RGB/1161.png')
    img1_RGB = cv.cvtColor(img1_RGB, cv.COLOR_BGR2RGB)
    img2_RGB = cv.imread('RGB/1162.png')
    img2_RGB = cv.cvtColor(img1_RGB, cv.COLOR_BGR2RGB)
    img1_depth = cv.imread('Depth/1161.png')
    img2_depth = cv.imread('Depth/1162.png')
    camera_matrix = np.array(
        [1 / f_depth_x, 0., -c_depth_x / f_depth_x,
         0., 1 / f_depth_y, -c_depth_y / f_depth_y,
         0., 0., 1.]).reshape(3, 3)
    camera_matrix_opencv = np.array(
        [f_depth_x, 0., c_depth_x,
         0., f_depth_y, c_depth_y,
         0., 0., 1.]).reshape(3, 3)
    pcl_from_open3d(img1_RGB, img1_depth, camera_matrix)
    pcl1 = pcl_from_open3d(img1_RGB, img1_depth, camera_matrix)
    pcl2 = pcl_from_open3d(img2_RGB, img2_depth, camera_matrix)
    o3d.draw_geometries([pcl1, pcl2])


if __name__ == '__main__':
    main()
