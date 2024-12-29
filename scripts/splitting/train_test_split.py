import os
import shutil
import random

class TrainTestSplit:
    def __init__(self, point_cloud_dir, label_dir, train_size, test_size, train_dir, test_dir):
        """
        Args:
            point_cloud_dir (str): Directory containing point cloud files (.pcd).
            label_dir (str): Directory containing label files (e.g., .txt for labels).
            train_size (int): Number of files to include in the training set.
            test_size (int): Number of files to include in the test set.
            train_dir (str): Path to the directory where train data will be stored.
            test_dir (str): Path to the directory where test data will be stored.
        """
        self.point_cloud_dir = point_cloud_dir
        self.label_dir = label_dir
        self.train_size = train_size
        self.test_size = test_size
        self.train_dir = train_dir
        self.test_dir = test_dir

        # Make sure train and test directories exist
        os.makedirs(os.path.join(train_dir, "point_clouds"), exist_ok=True)
        os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)
        os.makedirs(os.path.join(test_dir, "point_clouds"), exist_ok=True)
        os.makedirs(os.path.join(test_dir, "labels"), exist_ok=True)

    def split(self):
        # Get the list of point cloud and label files
        point_cloud_files = sorted(os.listdir(self.point_cloud_dir))
        label_files = sorted(os.listdir(self.label_dir))

        # Ensure the number of point cloud files matches the number of label files
        assert len(point_cloud_files) == len(label_files), \
            "Mismatch between number of point cloud and label files."

        # Shuffle the file indices
        indices = list(range(len(point_cloud_files)))
        random.shuffle(indices)

        # Select files for training and testing
        train_indices = indices[:self.train_size]
        test_indices = indices[self.train_size:self.train_size + self.test_size]

        # Helper function to move files
        def move_files(indices, dest_dir):
            for idx in indices:
                pc_file = point_cloud_files[idx]
                label_file = label_files[idx]

                shutil.copy(
                    os.path.join(self.point_cloud_dir, pc_file),
                    os.path.join(dest_dir, "point_clouds", pc_file)
                )
                shutil.copy(
                    os.path.join(self.label_dir, label_file),
                    os.path.join(dest_dir, "labels", label_file)
                )

        # Move train files
        move_files(train_indices, self.train_dir)
        print(f"Moved {len(train_indices)} files to training set at {self.train_dir}.")

        # Move test files
        move_files(test_indices, self.test_dir)
        print(f"Moved {len(test_indices)} files to test set at {self.test_dir}.")


