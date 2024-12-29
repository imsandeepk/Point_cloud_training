from train_test_split import TrainTestSplit

# Create an instance of TrainTestSplit
splitter = TrainTestSplit(
    point_cloud_dir="/Users/sandeep/DOCS/Point Cloud Training/data/point_cloud_meshes",
    label_dir="/Users/sandeep/DOCS/Point Cloud Training/data/point_cloud_labels",
    train_size=100,
    test_size=20,
    train_dir="/Users/sandeep/DOCS/Point Cloud Training/data/train",
    test_dir="/Users/sandeep/DOCS/Point Cloud Training/data/test"
)

splitter.split()