#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import open3d as o3d

np.random.seed(0)
n_samples = 500

# generate and save noisy_circles point cloud
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
noisy_circles_point_cloud = o3d.geometry.PointCloud()
noisy_circles_point_cloud.points = o3d.utility.Vector3dVector(np.append(noisy_circles[0], np.zeros((n_samples, 1)), axis=1))
o3d.io.write_point_cloud("../../../data/cluster_data/noisy_circles.pcd", noisy_circles_point_cloud, write_ascii=True)

noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
noisy_moons_point_cloud = o3d.geometry.PointCloud()
noisy_moons_point_cloud.points = o3d.utility.Vector3dVector(np.append(noisy_moons[0], np.zeros((n_samples, 1)), axis=1))
o3d.io.write_point_cloud("../../../data/cluster_data/noisy_moons.pcd", noisy_moons_point_cloud, write_ascii=True)

blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
blobs_point_cloud = o3d.geometry.PointCloud()
blobs_point_cloud.points = o3d.utility.Vector3dVector(np.append(blobs[0], np.zeros((n_samples, 1)), axis=1))
o3d.io.write_point_cloud("../../../data/cluster_data/blobs.pcd", blobs_point_cloud, write_ascii=True)

no_structure = np.random.rand(n_samples, 2), None
no_structure_point_cloud = o3d.geometry.PointCloud()
no_structure_point_cloud.points = o3d.utility.Vector3dVector(np.append(no_structure[0], np.zeros((n_samples, 1)), axis=1))
o3d.io.write_point_cloud("../../../data/cluster_data/no_structure.pcd", no_structure_point_cloud, write_ascii=True)

random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso_point_cloud = o3d.geometry.PointCloud()
aniso_point_cloud.points = o3d.utility.Vector3dVector(np.append(X_aniso, np.zeros((n_samples, 1)), axis=1))
o3d.io.write_point_cloud("../../../data/cluster_data/aniso.pcd", aniso_point_cloud, write_ascii=True)

varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)
varied_point_cloud = o3d.geometry.PointCloud()
varied_point_cloud.points = o3d.utility.Vector3dVector(np.append(varied[0], np.zeros((n_samples, 1)), axis=1))
o3d.io.write_point_cloud("../../../data/cluster_data/varied.pcd", varied_point_cloud, write_ascii=True)

