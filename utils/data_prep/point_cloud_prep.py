import os
import open3d as o3d
import numpy as np


def create_point_cloud(mesh_path, seg_path,mesh_file,seg_file,index):
    mesh = os.path.join(mesh_path, mesh_file)
    seg = os.path.join(seg_path, seg_file)
    mesh = o3d.io.read_triangle_mesh(mesh)
    seg = np.loadtxt(seg)
    colored_set = set()
    for i,v in enumerate(seg):
        if v == 0:
            colored_set.add(mesh.triangles[i][0])
            colored_set.add(mesh.triangles[i][1])
            colored_set.add(mesh.triangles[i][2])

    verts = np.asarray(mesh.vertices)
    labels = np.zeros(len(verts))

    for i in colored_set:
        labels[i] = 1
    colors = np.zeros((len(verts), 3))
    colors[labels == 1] = [1,0,0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(f'/Users/sandeep/DOCS/Point Cloud Training/data/point_cloud_meshes/point_cloud_{index}.pcd', pcd)
    np.savetxt(f'/Users/sandeep/DOCS/Point Cloud Training/data/point_cloud_labels/point_cloud_{index}.txt', labels)

path = os.path.join(os.path.dirname(os.path.dirname(__file__) ), 'data')


seg_path = os.path.join(path, 'label')
mesh_path = os.path.join(path, 'mesh')

list_of_seg = os.listdir(seg_path)
list_of_mesh = os.listdir(mesh_path)

mesh_files = sorted([name for name in list_of_mesh ]) 
seg_files = sorted([name for name in list_of_seg ])

for seg_file , mesh_file in zip(seg_files, mesh_files):
    print(f'Processing {seg_file} and {mesh_file}')
    create_point_cloud(mesh_path, seg_path, mesh_file,seg_file ,seg_file.split('.')[0])





