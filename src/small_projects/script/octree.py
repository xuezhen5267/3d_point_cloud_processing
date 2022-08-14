# octree的具体实现，包括构建和查找

import random
import math
import numpy as np
import time

from result_set import KNNResultSet, RadiusNNResultSet
contain = 0
# 节点，构成OCtree的基本元素
class Octant:
    def __init__(self, children, center, extent, point_indices, is_leaf):
        self.children = children
        self.center = center
        self.extent = extent # 从中心点到一个面的距离，边长的一半
        self.point_indices = point_indices
        self.is_leaf = is_leaf

    def __str__(self):
        output = ''
        output += 'center: [%.2f, %.2f, %.2f], ' % (self.center[0], self.center[1], self.center[2])
        output += 'extent: %.2f, ' % self.extent
        output += 'is_leaf: %d, ' % self.is_leaf
        output += 'children: ' + str([x is not None for x in self.children]) + ", "
        output += 'point_indices: ' + str(self.point_indices)
        return output

# 功能：翻转octree
# 输入：
#     root: 构建好的octree
#     depth: 当前深度
#     max_depth：最大深度
def traverse_octree(root: Octant, depth, max_depth):
    depth[0] += 1
    if max_depth[0] < depth[0]:
        max_depth[0] = depth[0]

    if root is None:
        pass
    elif root.is_leaf:
        print(root)
    else:
        for child in root.children:
            traverse_octree(child, depth, max_depth)
    depth[0] -= 1

# 功能：通过递归的方式构建octree
# 输入：
#     root：根节点
#     db：原始数据
#     center: 中心
#     extent: 当前分割区间
#     point_indices: 点的key
#     leaf_size: scale
#     min_extent: 最小分割区间
def octree_recursive_build(root, db, center, extent, point_indices, leaf_size, min_extent):
    if len(point_indices) == 0:
        return None

    if root is None:
        root = Octant([None for i in range(8)], center, extent, point_indices, is_leaf=True)

    # determine whether to split this octant
    if len(point_indices) <= leaf_size or extent <= min_extent: # 设置两个条件来判断是否继续分割下去，一个条件是点数是否小于leaf_size, 一个条件是边长的一半，extent 是否足够小，小于 min_extent
        root.is_leaf = True
    else:
        root.is_leaf = False # 先将 is_leaf 置 False
        children_point_indices = []
        for i in range(8):
            children_point_indices.append([]) # 建立一个空的嵌套的 list，用来存储每个子节点内存储的点的 indices。 children_point_indices = [[],[],[],[],[],[],[],[]]
        for point_idx in point_indices:
            point_db = db[point_idx] # point_db 是 1*3 的向量
            # 根据点的3D位置(与该节点的中心位置)，生成一个morton_code(0~7)
            morton_code = 0
            if point_db[0] > center [0]:
                morton_code = morton_code | 1 # 按位与 0 | 1 = 1;
            if point_db[1] > center[1]:
                morton_code = morton_code | 2  # 按位与 0 | 2 = 2; 1 | 2 = 3
            if point_db[2] > center[2]:
                morton_code = morton_code | 4  # 按位与 0 | 4 = 4; 1 | 4 = 5; 2 | 4 = 6; 3 | 4 = 7
            children_point_indices[morton_code].append(point_idx) # 把对应的 idx 存入 0~7 对应的列表中

        factor= [-0.5, 0.5]
        for i in range(8):
            # 计算第 i 个子节点的中心位置坐标
            child_center_x = center[0] + factor[(i & 1) > 0] * extent
            child_center_y = center[1] + factor[(i & 2) > 0] * extent
            child_center_z = center[2] + factor[(i & 4) > 0] * extent
            child_extent = 0.5 * extent # 计算子节点的 extent，即边长的一半
            child_center = np.array([child_center_x, child_center_y, child_center_z])
            # 递归，分别将 0~7 列表中的index 存入到标号分别为 0~7 的子节点中
            root.children[i] = octree_recursive_build(root.children[i],
                                                      db,
                                                      child_center,
                                                      child_extent,
                                                      children_point_indices[i],
                                                      leaf_size,
                                                      min_extent)
    return root

# 功能：判断当前query区间是否在octant内
# 输入：
#     query: 索引信息
#     radius：索引半径
#     octant：octree
# 输出：
#     判断结果，即True/False
def inside(query: np.ndarray, radius: float, octant:Octant):
    """
    Determines if the query ball is inside the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    query_offset = query - octant.center # 以 Octant 中心为原点， 查询点的相对坐标
    query_offset_abs = np.fabs(query_offset) # 查询点距离 Octant 中心店的距离
    possible_space = query_offset_abs + radius # 判断一个 Octant 是否包括整个邻域球的标准: 邻域球半径 + 查询点到 Octant 中心的距离 < Octant 的边长的一半
    return np.all(possible_space < octant.extent) # np.all() 用来对每个维度单独进行判断

# 功能：判断当前query区间是否和octant有重叠部分
# 输入：
#     query: 索引信息
#     radius：索引半径
#     octant：octree
# 输出：
#     判断结果，即True/False
def overlaps(query: np.ndarray, radius: float, octant:Octant):
    """
    Determines if the query ball overlaps with the octant
    :param query:
    :param radius:
    :param octant:
    :return:
    """
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)

    # completely outside, since query is outside the relevant area
    max_dist = radius + octant.extent
    if np.any(query_offset_abs > max_dist): # 在三个维度上，球与立方体的距离都很远，与立方体所在平面都没有交点
        return False

    # if pass the above check, consider the case that the ball is contacting the face of the octant
    # 这个判断语句一定要放在第一条判断语句之后，即该判断语句的前提为：邻域球在三个维度上都距离立方体比较近，
    if np.sum((query_offset_abs < octant.extent).astype(np.int32)) >= 2: # 对8个面进行判断，球与平面的交点是否在正方形的内部
        return True

    # conside the case that the ball is contacting the edge or corner of the octant
    # since the case of the ball center (query) inside octant has been considered,
    # we only consider the ball center (query) outside octant
    x_diff = max(query_offset_abs[0] - octant.extent, 0)
    y_diff = max(query_offset_abs[1] - octant.extent, 0)
    z_diff = max(query_offset_abs[2] - octant.extent, 0)

    return x_diff * x_diff + y_diff * y_diff + z_diff * z_diff < radius * radius


# 功能：判断当前query是否包含octant
def contains(query: np.ndarray, radius: float, octant:Octant):
    query_offset = query - octant.center
    query_offset_abs = np.fabs(query_offset)
    query_offset_to_farthest_corner = query_offset_abs + octant.extent
    return np.linalg.norm(query_offset_to_farthest_corner) < radius

# 功能：在octree中查找信息
# 输入：
#    root: octree
#    db：原始数据
#    result_set: 索引结果
#    query：索引信息
def octree_radius_search_fast(root: Octant, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if contains(query, result_set.worstDist(), root):
        global contain
        contain = contain + 1
        # compare the contents of the octant
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # don't need to check any child
        return False

    if root.is_leaf and len(root.point_indices) > 0: # 如果这个节点是 leaf 节点，且内含点，则计算所有店与查询点之间的距离并插入到结果集中
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # check whether we can stop search now
        return inside(query, result_set.worstDist(), root)

    # 不需要优先搜索查询点坐在的 octant，因为 RNN 的搜索邻域不会变化，因此优先搜索查询点所在的 Octant 并没有意义，因为所有与邻域相交的Octant都必须要搜索。
    for child in root.children: # enumerate 可以同时给出偏移量和元素，详见《Python学习手册》 P410
        if child is None: # 如果该子节点是查询点坐在的空间（已经被优先查找过了），或者该子节点为空
            continue # 则跳过该子节点
        if not overlaps(query, result_set.worstDist(), child): # 如果查询点的最坏距离与该子节点的空间没有交集
            continue # 也跳过该子节点
        if octree_radius_search_fast(child, db, result_set, query):
            return True

    return inside(query, result_set.worstDist(), root)


# 功能：在octree中查找radius范围内的近邻
# 输入：
#     root: octree
#     db: 原始数据
#     result_set: 搜索结果
#     query: 搜索信息
def octree_radius_search(root: Octant, db: np.ndarray, result_set: RadiusNNResultSet, query: np.ndarray):
    if root is None:
        return False

    if root.is_leaf and len(root.point_indices) > 0: # 如果这个节点是 leaf 节点，且内含点，则计算所有店与查询点之间的距离并插入到结果集中
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # check whether we can stop search now
        return inside(query, result_set.worstDist(), root)

    # 不需要优先搜索查询点坐在的 octant，因为 RNN 的搜索邻域不会变化，因此优先搜索查询点所在的 Octant 并没有意义，因为所有与邻域相交的Octant都必须要搜索。
    for child in root.children: # enumerate 可以同时给出偏移量和元素，详见《Python学习手册》 P410
        if child is None: # 如果该子节点是查询点坐在的空间（已经被优先查找过了），或者该子节点为空
            continue # 则跳过该子节点
        if not overlaps(query, result_set.worstDist(), child): # 如果查询点的最坏距离与该子节点的空间没有交集
            continue # 也跳过该子节点
        if octree_radius_search(child, db, result_set, query):
            return True

    # final check of if we can stop search
    return inside(query, result_set.worstDist(), root)

# 功能：在octree中查找最近的k个近邻
# 输入：
#     root: octree
#     db: 原始数据
#     result_set: 搜索结果
#     query: 搜索信息
def octree_knn_search(root: Octant, db: np.ndarray, result_set: KNNResultSet, query: np.ndarray):
    if root is None:
        return False
    # 如果当前节点为末端节点，即 is_leaf = True
    if root.is_leaf and len(root.point_indices) > 0:
        # compare the contents of a leaf
        leaf_points = db[root.point_indices, :]
        diff = np.linalg.norm(np.expand_dims(query, 0) - leaf_points, axis=1)
        for i in range(diff.shape[0]):
            result_set.add_point(diff[i], root.point_indices[i])
        # check whether we can stop search now
        return inside(query, result_set.worstDist(), root)

    # 则根据查询点的坐标和节点坐标，计算所在子节点的编号
    morton_code = 0
    if query[0] > root.center[0]:
        morton_code = morton_code | 1
    if query[1] > root.center[1]:
        morton_code = morton_code | 2
    if query[2] > root.center[2]:
        morton_code = morton_code | 4
    # 优先查找该查询点所在的子节点
    if octree_knn_search(root.children[morton_code], db, result_set, query): # 如果最坏距离的球被这个子节点包围
        return True # return 表示提前终止， true 表示如果子节点可以包围查询点的邻域球，则父节点也可以包围查询点的邻域球

    # 然后再再其他子7个节点查找该查询点的邻域
    for c, child in enumerate(root.children): # enumerate 可以同时给出偏移量和元素，详见《Python学习手册》 P410
        if c == morton_code or child is None: # 如果该子节点是查询点坐在的空间（已经被优先查找过了），或者该子节点为空
            continue # 则跳过该子节点
        if not overlaps(query, result_set.worstDist(), child): # 如果查询点的最坏距离与该子节点的空间没有交集
            continue # 也跳过该子节点
        if octree_knn_search(child, db, result_set, query):
            return True

    # final check of if we can stop search
    return inside(query, result_set.worstDist(), root)

# 功能：构建octree，即通过调用octree_recursive_build函数实现对外接口
# 输入：
#    dp_np: 原始数据
#    leaf_size：scale
#    min_extent：最小划分区间
def octree_construction(db_np, leaf_size, min_extent):
    N, dim = db_np.shape[0], db_np.shape[1]
    db_np_min = np.amin(db_np, axis=0)
    db_np_max = np.amax(db_np, axis=0)
    db_extent = np.max(db_np_max - db_np_min) * 0.5
    db_center = np.mean(db_np, axis=0)

    root = None
    root = octree_recursive_build(root, db_np, db_center, db_extent, list(range(N)),
                                  leaf_size, min_extent)

    return root

def main():
    # configuration
    db_size = 64000
    dim = 3
    leaf_size = 4
    min_extent = 0.0001
    k = 8

    db_np = np.random.rand(db_size, dim)

    root = octree_construction(db_np, leaf_size, min_extent)

    # depth = [0]
    # max_depth = [0]
    # traverse_octree(root, depth, max_depth)
    # print("tree max depth: %d" % max_depth[0])

    query = np.asarray([0, 0, 0])
    result_set = KNNResultSet(capacity=k)
    octree_knn_search(root, db_np, result_set, query)
    print(result_set)
    #
    diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    nn_idx = np.argsort(diff)
    nn_dist = diff[nn_idx]
    print(nn_idx[0:k])
    print(nn_dist[0:k])

    begin_t = time.time()
    print("Radius search normal:")
    for i in range(100):
        query = np.random.rand(3)
        result_set = RadiusNNResultSet(radius=0.5)
        octree_radius_search(root, db_np, result_set, query)
    # print(result_set)
    print("Search takes %.3fms\n" % ((time.time() - begin_t) * 1000))

    begin_t = time.time()
    print("Radius search fast:")
    for i in range(100):
        query = np.random.rand(3)
        result_set = RadiusNNResultSet(radius = 0.5)
        octree_radius_search_fast(root, db_np, result_set, query)
    # print(result_set)
    print("Search takes %.3fms\n" % ((time.time() - begin_t)*1000))

if __name__ == '__main__':
    main()