import os

import numpy as np

from utils.utils import get_nth_file

dataset_folder = os.path.join(os.path.expanduser("~"), "datasets", "Humanoid-X")

keypoints = os.path.join(dataset_folder, "humanoid_keypoint")

for name in os.listdir(keypoints):

    sub_folder = os.path.join(keypoints, name)

    keypoint1 = np.load(get_nth_file(sub_folder))

    print(keypoint1.shape)

    keypoint2 = np.load(get_nth_file(sub_folder, n=1))

    print(keypoint2.shape)

    keypoint3 = np.load(get_nth_file(sub_folder, n=2))

    print(keypoint3.shape)

    break
