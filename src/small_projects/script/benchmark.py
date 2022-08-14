# 对数据集中的点云，批量执行构建树和查找，包括kdtree和octree，并评测其运行时间

import random
import math
import numpy as np
import time
import os
import struct

import octree
import kdtree
from result_set import KNNResultSet, RadiusNNResultSet

def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

def main():
    # configuration
    leaf_size = 32
    min_extent = 0.0001
    k = 8
    radius = 1

    filename = '/home/xz/Desktop/3d_point_cloud_process/data/KITTI/000000.bin'
    db_np = read_velodyne_bin(filename)

    # octree
    print("octree --------------")
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    begin_t = time.time()
    # 建 Octree
    root = octree.octree_construction(db_np, leaf_size, min_extent)
    construction_time_sum += time.time() - begin_t
    # 建立查询点
    query = db_np[0,:]
    begin_t = time.time()
    # KNN搜索 Octree
    result_set = KNNResultSet(capacity=k)
    octree.octree_knn_search(root, db_np, result_set, query)
    knn_time_sum += time.time() - begin_t
    begin_t = time.time()
    # RNN搜索 Octree
    result_set = RadiusNNResultSet(radius=radius)
    octree.octree_radius_search_fast(root, db_np, result_set, query)
    radius_time_sum += time.time() - begin_t
    begin_t = time.time()
    # 暴力搜索 Octree
    diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    nn_idx = np.argsort(diff)
    nn_dist = diff[nn_idx]
    brute_time_sum += time.time() - begin_t
    print("Octree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum*1000,
                                                                     knn_time_sum*1000,
                                                                     radius_time_sum*1000,
                                                                     brute_time_sum*1000))
    # kdtree
    print("kdtree --------------")
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    # 建 kdtree
    begin_t = time.time()
    root = kdtree.kdtree_construction(db_np, leaf_size)
    construction_time_sum += time.time() - begin_t
    # KNN search
    begin_t = time.time()
    result_set = KNNResultSet(capacity=k)
    kdtree.kdtree_knn_search(root, db_np, result_set, query)
    knn_time_sum += time.time() - begin_t
    # RNN search
    begin_t = time.time()
    result_set = RadiusNNResultSet(radius=radius)
    kdtree.kdtree_radius_search(root, db_np, result_set, query)
    radius_time_sum += time.time() - begin_t
    # brute search
    begin_t = time.time()
    diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    nn_idx = np.argsort(diff)
    nn_dist = diff[nn_idx]
    brute_time_sum += time.time() - begin_t

    print("Kdtree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum * 1000,
                                                                     knn_time_sum * 1000,
                                                                     radius_time_sum * 1000,
                                                                     brute_time_sum * 1000))
    print("adaptive kdtree --------------")
    adaptive_construction_time_sum = 0
    adaptive_knn_time_sum = 0
    adaptive_radius_time_sum = 0
    # 建 adaptive kdtree
    begin_t = time.time()
    root = kdtree.kdtree_construction_adaptive(db_np, leaf_size)
    adaptive_construction_time_sum += time.time() - begin_t
    # KNN search adaptive kdtree
    begin_t = time.time()
    result_set = KNNResultSet(capacity=k)
    kdtree.kdtree_knn_search(root, db_np, result_set, query)
    adaptive_knn_time_sum += time.time() - begin_t
    # RNN search adaptive kdtree
    begin_t = time.time()
    result_set = RadiusNNResultSet(radius=radius)
    kdtree.kdtree_radius_search(root, db_np, result_set, query)
    adaptive_radius_time_sum += time.time() - begin_t
    print("Adaptive kdtree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (adaptive_construction_time_sum * 1000,
                                                                     adaptive_knn_time_sum * 1000,
                                                                     adaptive_radius_time_sum * 1000,
                                                                     brute_time_sum * 1000))


if __name__ == '__main__':
    main()
