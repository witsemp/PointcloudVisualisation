import numpy as np
import cv2 as cv
import open3d as o3d


def pcl_from_images(rgb_image, depth_image, camera_matrix):
    pcl = []
    shape = np.shape(rgb_image)
    for row in range(0, shape[0]):
        for col in range(0, shape[1]):
            world_coords = depth_image[row, col] * (np.matmul(camera_matrix, np.array([row, col, 1]).reshape(3, 1)))
            pcl.append(world_coords)
    pcl = np.array(pcl).reshape(len(pcl), 3)
    # print(pcl)
    return pcl
def pcl_from_open3d(color_image, depth_image, camera_matrix):
    color = o3d.Image(color_image.astype(np.uint8))
    depth = o3d.Image(depth_image.astype(np.uint8))
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.intrinsic_matrix = camera_matrix
    rgbd = o3d.create_rgbd_image_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
    pcd = o3d.create_point_cloud_from_rgbd_image(image=rgbd, intrinsic=intrinsics)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.draw_geometries([pcd])





def main():
    f_depth_x = 1.0580353759137747e+03
    f_depth_y = 1.0604471766882732e+03
    c_depth_x = 9.5177505245974123e+02
    c_depth_y = 5.4839653660364445e+02
    img1_greyscale = cv.imread('RGB/1161.png', 0)
    img2_greyscale = cv.imread('RGB/1162.png', 0)
    img1_RGB = cv.imread('RGB/1161.png')
    img1_RGB = cv.cvtColor(img1_RGB, cv.COLOR_BGR2RGB)
    img2_RGB = cv.imread('RGB/1162.png')
    img2_RGB = cv.cvtColor(img1_RGB, cv.COLOR_BGR2RGB)
    img1_depth = cv.imread('Depth/1161.png', 0)
    img2_depth = cv.imread('Depth/1162.png', 0)
    camera_matrix = np.array(
        [1 / f_depth_x, 0., -c_depth_x / f_depth_x,
         0., 1 / f_depth_y, -c_depth_y / f_depth_y,
         0., 0., 1.]).reshape(3, 3)
    camera_matrix_opencv = np.array(
        [f_depth_x, 0., c_depth_x,
         0., f_depth_y, c_depth_y,
         0., 0., 1.]).reshape(3, 3)
    # pcl1 = pcl_from_images(img1_greyscale, img1_depth, camera_matrix)
    # print(pcl1)
    # pcl2 = pcl_from_images(img2_greyscale, img2_depth, camera_matrix)
    # point_cloud = o3d.PointCloud()
    # point_cloud.points = o3d.Vector3dVector(pcl1)
    # o3d.draw_geometries([point_cloud])
    pcl_from_open3d(img1_RGB, img1_depth, camera_matrix)


if __name__ == '__main__':
    main()
