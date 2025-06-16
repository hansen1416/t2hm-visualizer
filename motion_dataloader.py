import os.path
from glob import glob
from itertools import islice


import torch
import numpy as np


def get_nth_npy_file(folder, n):
    with os.scandir(folder) as entries:
        npy_files = (entry.path for entry in entries if entry.name.endswith(".npy"))
        return next(islice(npy_files, n, None), None)


class MotionDataLoader:

    def __init__(self):
        self.root_path = os.path.join(os.path.expanduser("~"), "Downloads", "motion-x")

        self.categories = [
            "animation",
            # "fitness",
            "haa500",
            "humman",
            "idea400",
            "kungfu",
            "music",
            "perform",
        ]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get(self, idx=0, category="animation"):

        motion_folder_path = os.path.join(
            self.root_path, "motion", "motion_generation", "smplx322", category
        )

        # motion_files = glob(os.path.join(motion_folder_path, "*.npy"))
        # motion_file = motion_files[idx]
        motion_file = get_nth_npy_file(motion_folder_path, idx)

        video_name = os.path.splitext(os.path.basename(motion_file))[0]

        video_file = os.path.join(
            self.root_path, "video", category, f"{video_name}.mp4"
        )
        # check if video file exists
        assert os.path.exists(video_file), f"Video file {video_file} does not exist."

        motion = np.load(motion_file)
        motion = torch.tensor(motion).float().to(self.device)

        root_orient = motion[:, :3]  # controls the global root orientation
        pose_body = motion[:, 3 : 3 + 63]  # controls the body
        pose_hand = motion[:, 66 : 66 + 90]  # controls the finger articulation
        pose_jaw = motion[:, 66 + 90 : 66 + 93]  # controls the yaw pose
        face_expr = motion[:, 159 : 159 + 50]  # controls the face expression
        face_shape = motion[:, 209 : 209 + 100]  # controls the face shape
        trans = motion[:, 309 : 309 + 3]  # controls the global body position
        betas = motion[:, 312:]  # controls the body shape. Body shape is static

        leye_pose = torch.zeros(root_orient.shape[0], 3, device=self.device)
        reye_pose = torch.zeros(root_orient.shape[0], 3, device=self.device)

        left_hand_pose = pose_hand[:, :45]  # left hand articulation
        right_hand_pose = pose_hand[:, 45:]  # right hand articulation

        # swap the x,z axes for `root_orient`
        root_orient[:, [0, 1]] = root_orient[:, [1, 0]]

        motion_params = {
            "betas": betas,
            "body_pose": pose_body,
            "global_orient": root_orient,
            "transl": trans,
            "batch_size": motion.shape[0],
            "jaw_pose": pose_jaw,
            "leye_pose": leye_pose,
            "reye_pose": reye_pose,
            "left_hand_pose": left_hand_pose,
            "right_hand_pose": right_hand_pose,
            "expression": face_expr,
        }

        return motion_params, video_file
