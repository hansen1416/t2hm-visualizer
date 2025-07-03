import os.path
from glob import glob

import torch
import numpy as np
from pycocotools.coco import COCO

from utils.utils import get_nth_file


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
        motion_file = get_nth_file(motion_folder_path, idx)

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

        # -90° rotation around Y-axis (in axis-angle format)
        # fix_rotation = torch.tensor([-math.pi, 0, 0], device=root_orient.device)

        # Apply the fix to each frame
        # root_orient = root_orient + fix_rotation  # broadcasted addition
        # swap the x,z axes for `root_orient`
        # 1,0,2； 0,2,1； 1,2,0； 2,0,1； 2,1,0
        # root_orient[:, [1, 0, 2]] = root_orient[:, [0, 1, 2]]
        # root_orient[:, 1] = -root_orient[:, 1]

        motion_params = {
            "betas": betas,  # torch.Size([n, 10])
            "body_pose": pose_body,  # torch.Size([45, 63])
            "global_orient": root_orient,  # torch.Size([45, 3])
            "transl": trans,  # torch.Size([45, 3])
            "batch_size": motion.shape[0],
            "jaw_pose": pose_jaw,  # torch.Size([45, 3])
            "leye_pose": leye_pose,  # torch.Size([45, 3])
            "reye_pose": reye_pose,  # torch.Size([45, 3])
            "left_hand_pose": left_hand_pose,  # torch.Size([45, 45])
            "right_hand_pose": right_hand_pose,  # torch.Size([45, 45])
            "expression": face_expr,  # torch.Size([45, 50])
        }

        return motion_params, video_file

    def get_local_json(self, idx=0, category="animation"):
        motion_folder_path = os.path.join(
            self.root_path, "motion", "mesh_recovery", "local_motion", category
        )

        motion_file = get_nth_file(motion_folder_path, idx, ext=".json")

        video_name = os.path.splitext(os.path.basename(motion_file))[0]

        video_file = os.path.join(
            self.root_path, "video", category, f"{video_name}.mp4"
        )
        # check if video file exists
        assert os.path.exists(video_file), f"Video file {video_file} does not exist."

        db = COCO(motion_file)

        n_frames = len(db.anns.keys())

        betas = torch.zeros(n_frames, 10, device=self.device)
        body_pose = torch.zeros(n_frames, 63, device=self.device)
        global_orient = torch.zeros(n_frames, 3, device=self.device)
        trans = torch.zeros(n_frames, 3, device=self.device)
        jaw_pose = torch.zeros(n_frames, 3, device=self.device)
        leye_pose = torch.zeros(n_frames, 3, device=self.device)
        reye_pose = torch.zeros(n_frames, 3, device=self.device)
        left_hand_pose = torch.zeros(n_frames, 45, device=self.device)
        right_hand_pose = torch.zeros(n_frames, 45, device=self.device)
        expression = torch.zeros(n_frames, 10, device=self.device)

        for i, aid in enumerate(db.anns.keys()):
            ann = db.anns[aid]

            betas[i] = torch.tensor(ann["smplx_params"]["shape"], device=self.device)
            body_pose[i] = torch.tensor(
                ann["smplx_params"]["body_pose"], device=self.device
            )
            global_orient[i] = torch.tensor(
                ann["smplx_params"]["root_pose"], device=self.device
            )
            trans[i] = torch.tensor(ann["smplx_params"]["trans"], device=self.device)
            jaw_pose[i] = torch.tensor(
                ann["smplx_params"]["jaw_pose"], device=self.device
            )

            left_hand_pose[i] = torch.tensor(
                ann["smplx_params"]["lhand_pose"], device=self.device
            )
            right_hand_pose[i] = torch.tensor(
                ann["smplx_params"]["rhand_pose"], device=self.device
            )

            expression[i] = torch.tensor(
                ann["smplx_params"]["expr"], device=self.device
            )

        motion_params = {
            "betas": betas,  # torch.Size([n, 10])
            "body_pose": body_pose,  # torch.Size([45, 63])
            "global_orient": global_orient,  # torch.Size([45, 3])
            "transl": trans,  # torch.Size([45, 3])
            "batch_size": n_frames,
            "jaw_pose": jaw_pose,  # torch.Size([45, 3])
            "leye_pose": leye_pose,  # torch.Size([45, 3])
            "reye_pose": reye_pose,  # torch.Size([45, 3])
            "left_hand_pose": left_hand_pose,  # torch.Size([45, 45])
            "right_hand_pose": right_hand_pose,  # torch.Size([45, 45])
            "expression": expression,  # torch.Size([45, 50])
        }

        return motion_params, video_file

    def get_global_json(self, idx=0, category="animation"):
        motion_folder_path = os.path.join(
            self.root_path, "motion", "mesh_recovery", "global_motion", category
        )

        motion_file = get_nth_file(motion_folder_path, idx, ext=".json")

        video_name = os.path.splitext(os.path.basename(motion_file))[0]

        video_file = os.path.join(
            self.root_path, "video", category, f"{video_name}.mp4"
        )
        # check if video file exists
        assert os.path.exists(video_file), f"Video file {video_file} does not exist."

        text_file = os.path.join(
            self.root_path, "text", "semantic_label", category, f"{video_name}.txt"
        )

        assert os.path.exists(text_file), f"Text file {text_file} does not exist."

        with open(text_file, "r") as f:
            label = f.read()

        db = COCO(motion_file)

        n_frames = len(db.anns.keys())

        betas = torch.zeros(n_frames, 10, device=self.device)
        body_pose = torch.zeros(n_frames, 63, device=self.device)
        global_orient = torch.zeros(n_frames, 3, device=self.device)
        trans = torch.zeros(n_frames, 3, device=self.device)
        jaw_pose = torch.zeros(n_frames, 3, device=self.device)
        leye_pose = torch.zeros(n_frames, 3, device=self.device)
        reye_pose = torch.zeros(n_frames, 3, device=self.device)
        left_hand_pose = torch.zeros(n_frames, 45, device=self.device)
        right_hand_pose = torch.zeros(n_frames, 45, device=self.device)
        expression = torch.zeros(n_frames, 10, device=self.device)

        for i, aid in enumerate(db.anns.keys()):
            ann = db.anns[aid]

            betas[i] = torch.tensor(ann["smplx_params"]["betas"], device=self.device)
            body_pose[i] = torch.tensor(
                ann["smplx_params"]["pose_body"], device=self.device
            )
            # global_orient[i] = torch.tensor(
            #     ann["smplx_params"]["root_orient"], device=self.device
            # )
            # trans[i] = torch.tensor(ann["smplx_params"]["trans"], device=self.device)
            jaw_pose[i] = torch.tensor(
                ann["smplx_params"]["pose_jaw"], device=self.device
            )

            # left_hand_pose[i] = torch.tensor(
            #     ann["smplx_params"]["lhand_pose"], device=self.device
            # )
            # right_hand_pose[i] = torch.tensor(
            #     ann["smplx_params"]["rhand_pose"], device=self.device
            # )

            expression[i] = torch.tensor(
                ann["smplx_params"]["face_expr"][:10], device=self.device
            )

        # print(global_orient.shape)

        # # flip y,z of global_orient
        # global_orient[:, 1] *= -1
        # global_orient[:, 2] *= -1

        motion_params = {
            "betas": betas,  # torch.Size([n, 10])
            "body_pose": body_pose,  # torch.Size([45, 63])
            "global_orient": global_orient,  # torch.Size([45, 3])
            "transl": trans,  # torch.Size([45, 3])
            "batch_size": n_frames,
            "jaw_pose": jaw_pose,  # torch.Size([45, 3])
            "leye_pose": leye_pose,  # torch.Size([45, 3])
            "reye_pose": reye_pose,  # torch.Size([45, 3])
            "left_hand_pose": left_hand_pose,  # torch.Size([45, 45])
            "right_hand_pose": right_hand_pose,  # torch.Size([45, 45])
            "expression": expression,  # torch.Size([45, 50])
        }

        return motion_params, video_file, label


if __name__ == "__main__":
    loader = MotionDataLoader()
    # motion_params, video_file = loader.get(0, "animation")
    # motion_params, video_file = loader.get_local_json(0, "animation")
    motion_params, video_file = loader.get_global_json(0, "animation")
    print(motion_params)
    print(video_file)
    print("Motion data loaded successfully.")
