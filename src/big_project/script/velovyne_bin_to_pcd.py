import numpy as np
import open3d as o3d
import time
import struct

def read_velodyne_bin(path):
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)



def main():
    filename = "../../../data/KITTI/000000.bin"
    db_np = read_velodyne_bin(filename)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(db_np)
    o3d.io.write_point_cloud("../../../data/KITTI/000000.pcd", cloud, write_ascii=True)

if __name__ == '__main__':
    main()