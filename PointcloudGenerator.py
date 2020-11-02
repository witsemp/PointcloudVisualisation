import numpy as np
import cv2 as cv
from imageio import imread
import open3d as o3d

def lerp(x, x1, x2, y1, y2):
    # Figure out how 'wide' each range is
    x_span = x2 - x1
    y_span = y2 - y1

    # Convert the left range into a 0-1 range (float)
    scaled = (x - x1) / x_span

    # Convert the 0-1 range into a value in the right range.
    return y1 + (scaled * y_span)

def pcl_from_images(depth_image, camera_matrix):
    fov = np.deg2rad(33.39849)
    near = 1.0
    far = 3.0
    pcl = []
    depth_image = depth_image[:, :, 0] / 255.0
    depth_image = lerp(depth_image, 0.0, 1.0, 0.0, far)
    rows, cols = np.shape(depth_image)
    for row in range(rows):
        for col in range(cols):
            world_coords = depth_image[row, col] * np.dot(camera_matrix, np.array([row, col, 1]).reshape(3, 1))
            pcl.append(world_coords)
    pcl = np.array(pcl).reshape(len(pcl), 3)
    print(pcl)
    return pcl

def main():

    f_depth_x = 1.0580353759137747e+03
    f_depth_y = 1.0604471766882732e+03
    c_depth_x = 9.5177505245974123e+02
    c_depth_y = 5.4839653660364445e+02
    depth_path1 = 'Depth/871.png'
    img1_depth = np.array(imread(depth_path1))
    camera_matrix = np.array(
        [1 / f_depth_x, 0., -c_depth_x / f_depth_x,
         0., 1 / f_depth_y, -c_depth_y / f_depth_y,
         0., 0., 1.]).reshape(3, 3)
    camera_matrix_opencv = np.array(
        [f_depth_x, 0., c_depth_x,
         0., f_depth_y, c_depth_y,
         0., 0., 1.]).reshape(3, 3)
    pcl1 = pcl_from_images(img1_depth, camera_matrix)
    point_cloud = o3d.PointCloud()
    point_cloud.points = o3d.Vector3dVector(pcl1)
    o3d.write_point_cloud('pcl1.ply', point_cloud)



if __name__ == '__main__':
    main()
