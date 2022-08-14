# kdtree的具体实现，包括构建和查找

import random
import math
import numpy as np
import time

from result_set import KNNResultSet, RadiusNNResultSet

# Node 即是节点又是树
class Node:
    def __init__(self, axis, value, left, right, point_indices):
        self.axis = axis # 超平面所在的维度
        self.value = value # 超平面所在维度的分割值
        self.left = left
        self.right = right
        self.point_indices = point_indices # ndarray

    def is_leaf(self):
        if self.value is None:
            return True
        else:
            return False

    def __str__(self):
        output = ''
        output += 'axis %d, ' % self.axis
        if self.value is None:
            output += 'split value: leaf, '
        else:
            output += 'split value: %.2f, ' % self.value
        output += 'point_indices: '
        output += str(self.point_indices.tolist())
        return output

# 对节点内的点进行排序，目的是为了找到这些点的中位数，一遍更好的分割出平衡的树
def sort_key_by_vale(key, value):
    assert key.shape == value.shape # assert 可以在条件不满足程序运行的情况下直接返回错误，而不必等待程序运行后出现崩溃的情况
    assert len(key.shape) == 1
    sorted_idx = np.argsort(value)
    key_sorted = key[sorted_idx]
    value_sorted = value[sorted_idx]
    return key_sorted, value_sorted

# 轮流方法选择 axis
def axis_round_robin(axis, dim):
    if axis == dim-1:
        return 0
    else:
        return axis + 1

# 自适应方法选择 axis
def axis_adaptive(db, point_indices):
    std_deviation = np.std(db[point_indices, :], 0) # 求该节点内的所有点在三个轴上的标准差
    axis = np.argmax(std_deviation) # 选择标准差较大的那个坐标轴维度
    return axis


# 通过递归的方式构建树，自适应的版本
def kdtree_recursive_build_adaptive(root, db, point_indices, axis, leaf_size):
    if root is None:
        root = Node(axis, None, None, None, point_indices)

    # determine whether to split into left and right
    if len(point_indices) > leaf_size:
        # --- get the split position ---
        point_indices_sorted, _ = sort_key_by_vale(point_indices, db[point_indices, axis])  # M
        middle_left_idx = math.ceil(point_indices_sorted.shape[0] / 2) - 1 # ceil 向下取整，-1 是因为 ndarray 的索引值是从 0 开始的
        middle_left_point_idx = point_indices_sorted[middle_left_idx] # 较小的中位点在 db 中的 idx
        middle_left_point_idx = db[middle_left_point_idx, axis] # 较小的中位点在 db 中的 value
        middle_right_idx = middle_left_idx + 1
        middle_right_point_idx = point_indices_sorted[middle_right_idx] # 较大的中位点在 db 中的 idx
        middle_right_point_idx = db[middle_right_point_idx, axis] # 较大的中位点在 db 中的 value
        root.value = (middle_right_point_idx + middle_left_point_idx) * 0.5

        root.left = kdtree_recursive_build(root.left,
                                           db,
                                           point_indices_sorted[0: middle_right_idx], # ndarry 的切片范围:[begin, end)
                                           axis_adaptive(db, point_indices), # 使用自适应的方法选择坐标轴
                                           leaf_size)
        root.right = kdtree_recursive_build(root.right,
                                           db,
                                           point_indices_sorted[middle_right_idx:],  # ndarry 的切片范围:[begin, end)
                                           axis_adaptive(db, point_indices), # 使用自适应的方法选择坐标轴
                                           leaf_size)

    return root

# 通过递归的方式构建树，轮流坐标轴版本
def kdtree_recursive_build(root, db, point_indices, axis, leaf_size):
    if root is None:
        root = Node(axis, None, None, None, point_indices)

    # determine whether to split into left and right
    if len(point_indices) > leaf_size:
        # --- get the split position ---
        point_indices_sorted, _ = sort_key_by_vale(point_indices, db[point_indices, axis])  # M
        middle_left_idx = math.ceil(point_indices_sorted.shape[0] / 2) - 1 # ceil 向下取整，-1 是因为 ndarray 的索引值是从 0 开始的
        middle_left_point_idx = point_indices_sorted[middle_left_idx] # 较小的中位点在 db 中的 idx
        middle_left_point_idx = db[middle_left_point_idx, axis] # 较小的中位点在 db 中的 value
        middle_right_idx = middle_left_idx + 1
        middle_right_point_idx = point_indices_sorted[middle_right_idx] # 较大的中位点在 db 中的 idx
        middle_right_point_idx = db[middle_right_point_idx, axis] # 较大的中位点在 db 中的 value
        root.value = (middle_right_point_idx + middle_left_point_idx) * 0.5

        root.left = kdtree_recursive_build(root.left,
                                           db,
                                           point_indices_sorted[0: middle_right_idx], # ndarry 的切片范围:[begin, end)
                                           axis_round_robin(axis, db.shape[1]), # 轮流改变坐标轴方法，使 axis 自增1，增加到维度极限则降为0
                                           leaf_size)
        root.right = kdtree_recursive_build(root.right,
                                           db,
                                           point_indices_sorted[middle_right_idx:],  # ndarry 的切片范围:[begin, end)
                                           axis_round_robin(axis, db.shape[1]),  # 轮流改变坐标轴方法，使 axis 自增1，增加到维度极限则降为0
                                           leaf_size)

    return root

# 翻转一个kd树
def traverse_kdtree(root: Node, depth, max_depth):
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    if root.is_leaf():
        print(root)
    else:
        traverse_kdtree(root.left, depth, max_depth)
        traverse_kdtree(root.right, depth, max_depth)

    depth[0] -= 1

# 功能：构建kd树（利用kdtree_recursive_build功能函数实现的对外接口）
def kdtree_construction(db_np, leaf_size):
    N, dim = db_np.shape[0], db_np.shape[1]

    # build kd_tree recursively
    root = None
    root = kdtree_recursive_build(root,
                                  db_np,
                                  np.arange(N),
                                  axis=0,
                                  leaf_size=leaf_size)
    return root

# 功能：构建kd树，自适应版本（利用kdtree_recursive_build功能函数实现的对外接口）
def kdtree_construction_adaptive(db_np, leaf_size):
    N, dim = db_np.shape[0], db_np.shape[1]

    # build kd_tree recursively
    root = None
    root = kdtree_recursive_build_adaptive(root,
                                  db_np,
                                  np.arange(N),
                                  axis=0,
                                  leaf_size=leaf_size)
    return root


# 功能：通过kd树实现knn搜索，即找出最近的k个近邻
def kdtree_knn_search(root: Node, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray): # 冒号后是实参建议类型
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :] # leaf_points (N * 3) ndarray
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        # np.expand_dims(query, 0)，把 query 的尺寸从3变成(1*3), 方便与 leaf_points 直接做减法运算
        # np.linalg.norm 的功能是求范数， diff 是一个(n*1) 的ndarray
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i]) # diff[i] 表示的是该点距离查询点的距离，root.point_indeces[i] 表示的是该点在原始数据中的索引值
            # 在 add_point 方法中，也会更新 worst_dist
        return False

    if query[root.axis] <= root.value: # 如果查询点在指定维度下的位置在超平面的位置左侧（小于）
        kdtree_knn_search(root.left, db, result_set, query) # 优先搜索查询点坐在的超平面区域
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist(): # math.fabs() 用来计算绝对值，即查询点和超平面的距离，如果worst_dist大于这个距离，则还需要检查超平面另一半的区域
            kdtree_knn_search(root.right, db, result_set, query) # 还需要检查超平面另一半的区域
    else: # 如果查询点在指定维度下的位置在超平面的位置右侧（大于）
        kdtree_knn_search(root.right, db, result_set, query)  # 优先搜索查询点坐在的超平面区域
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist(): # math.fabs() 用来计算绝对值，即查询点和超平面的距离，如果worst_dist大于这个距离，则还需要检查超平面另一半的区域
            kdtree_knn_search(root.left, db, result_set, query) # 还需要检查超平面另一半的区域

    return False

# 功能：通过kd树实现radius搜索，即找出距离radius以内的近邻
def kdtree_radius_search(root: Node, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf():
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        return False

    if query[root.axis] <= root.value: # 如果查询点在指定维度下的位置在超平面的位置左侧（小于）
        kdtree_radius_search(root.left, db, result_set, query) # 优先搜索查询点坐在的超平面区域
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist(): # math.fabs() 用来计算绝对值，即查询点和超平面的距离，如果worst_dist大于这个距离，则还需要检查超平面另一半的区域
            kdtree_radius_search(root.right, db, result_set, query) # 还需要检查超平面另一半的区域
    else: # 如果查询点在指定维度下的位置在超平面的位置右侧（大于）
        kdtree_radius_search(root.right, db, result_set, query)  # 优先搜索查询点坐在的超平面区域
        if math.fabs(query[root.axis] - root.value) < result_set.worstDist(): # math.fabs() 用来计算绝对值，即查询点和超平面的距离，如果worst_dist大于这个距离，则还需要检查超平面另一半的区域
            kdtree_radius_search(root.left, db, result_set, query) # 还需要检查超平面另一半的区域

    return False



def main():
    # configuration
    db_size = 64
    dim = 3
    leaf_size = 4
    k = 8

    db_np = np.random.rand(db_size, dim)

    root = kdtree_construction(db_np, leaf_size=leaf_size)

    #depth = [0]
    #max_depth = [0]
    #traverse_kdtree(root, depth, max_depth)
    #print("tree max depth: %d" % max_depth[0])

    query = np.asarray([0, 0, 0])
    result_set = KNNResultSet(capacity=k)
    kdtree_knn_search(root, db_np, result_set, query)
    #
    print(result_set)
    #
    diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    nn_idx = np.argsort(diff)
    nn_dist = diff[nn_idx]
    print(nn_idx[0:k])
    print(nn_dist[0:k])
    #
    #
    print("Radius search:")
    query = np.asarray([0, 0, 0])
    result_set = RadiusNNResultSet(radius = 0.5)
    kdtree_radius_search(root, db_np, result_set, query)
    print(result_set)


if __name__ == '__main__':
    main()