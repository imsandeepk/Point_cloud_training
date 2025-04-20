import os


path = os.path.join(os.path.dirname(os.path.dirname(__file__) ), 'data')


seg_path = os.path.join(path, 'label')
mesh_path = os.path.join(path, 'mesh')

list_of_seg = os.listdir(seg_path)
list_of_mesh = os.listdir(mesh_path)

mesh_files = sorted([name for name in list_of_mesh ]) 
seg_files = sorted([name for name in list_of_seg ])

