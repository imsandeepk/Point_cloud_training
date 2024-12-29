import os 
import numpy as np
import open3d as o3d

mesh_path = '/Users/sandeep/DOCS/Point Cloud Training/data/point_cloud_meshes/point_cloud_5.pcd'
seg_path = '/Users/sandeep/DOCS/Point Cloud Training/data/point_cloud_labels/point_cloud_5.txt'

mesh = o3d.io.read_point_cloud(mesh_path)
seg = np.loadtxt(seg_path)

colors = [[1,0,0] if x == 1 else [0,1,0] for x in seg]
mesh.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([mesh])
