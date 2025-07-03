import os
from itertools import islice

import cv2
import torch
import numpy as np
import open3d as o3d
from smplx import SMPLX


def get_checkerboard_plane(plane_width=20, num_boxes=15, center=True, groun_level=0):

    pw = plane_width / num_boxes
    # white = [0.8, 0.8, 0.8]
    # black = [0.2, 0.2, 0.2]
    white = [230.0 / 255.0, 244.0 / 255.0, 244.0 / 255.0]
    black = [int(150 / 1.3) / 255.0, int(217 / 1.3) / 255.0, int(217 / 1.3) / 255.0]

    meshes = []
    for i in range(num_boxes):
        for j in range(num_boxes):
            c = i * pw, j * pw
            # ground = trimesh.primitives.Box(
            #     center=[0, 0, -0.0001],
            #     extents=[pw, pw, 0.0002]
            # )
            ground = o3d.geometry.TriangleMesh.create_box(
                width=pw, height=0.0002, depth=pw
            )

            if center:
                c = c[0] + (pw / 2) - (plane_width / 2), c[1] + (pw / 2) - (
                    plane_width / 2
                )
            # trans = trimesh.transformations.scale_and_translate(scale=1, translate=[c[0], c[1], 0])
            ground.translate([c[0], groun_level, c[1]])
            # ground.apply_transform(trimesh.transformations.rotation_matrix(np.rad2deg(-120), direction=[1,0,0]))
            ground.paint_uniform_color(black if ((i + j) % 2) == 0 else white)
            meshes.append(ground)

    return meshes


def gvhmr_result_loader(joints_glob_path, verts_glob_path):

    # get the folder name from the path
    results_folder = os.path.dirname(joints_glob_path)

    video_name = os.path.basename(os.path.normpath(results_folder))

    joints_glob: torch.FloatTensor = torch.load(joints_glob_path)
    verts_glob: torch.FloatTensor = torch.load(verts_glob_path)

    joints_glob = joints_glob.cpu().numpy()
    verts_glob = verts_glob.cpu().numpy()

    # data: dict = torch.load(hmr_result)
    # print(data.keys())
    # dict_keys(['smpl_params_global', 'smpl_params_incam', 'K_fullimg', 'net_outputs'])

    # print(data["smpl_params_global"].keys())
    # dict_keys(['body_pose', 'betas', 'global_orient', 'transl'])

    # for k, v in data["smpl_params_global"].items():
    # print(f"{k}: {v.shape}")
    # body_pose: torch.Size([336, 63])
    # betas: torch.Size([336, 10])
    # global_orient: torch.Size([336, 3])
    # transl: torch.Size([336, 3])

    # print(data["smpl_params_incam"].keys())
    # dict_keys(['body_pose', 'betas', 'global_orient', 'transl'])

    # for k, v in data["smpl_params_incam"].items():
    #     print(f"{k}: {v.shape}")
    # body_pose: torch.Size([336, 63])
    # betas: torch.Size([336, 10])
    # global_orient: torch.Size([336, 3])
    # transl: torch.Size([336, 3])

    # print(data["K_fullimg"].shape)
    # torch.Size([336, 3, 3])

    # print(data["net_outputs"].keys())
    # dict_keys(['model_output', 'decode_dict', 'pred_smpl_params_incam', 'pred_smpl_params_global', 'static_conf_logits'])
    # this the full output of the network, including both 'smpl_params_global' and 'smpl_params_incam'
    # for more information, refer to hmr4d/model/gvhmr/gvhmr_pl_demo.py

    video_path = os.path.join(
        os.path.expanduser("~"),
        "Downloads",
        "videos",
        f"{video_name}.mp4",
    )

    # check if video file exists
    if not os.path.exists(video_path):
        print(f"Video file does not exist: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    return total_frame_count, 1 / fps, verts_glob, joints_glob


def motionx_loader(file_path):

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    motion = np.load(file_path)
    motion = torch.tensor(motion).float()

    root_orient = motion[:, :3].to(device)
    pose_body = motion[:, 3 : 3 + 63].to(device)
    pose_hand = motion[:, 66 : 66 + 90].to(device)
    pose_jaw = motion[:, 66 + 90 : 66 + 93].to(device)
    face_expr = motion[:, 159 : 159 + 50].to(device)
    face_shape = motion[:, 209 : 209 + 100].to(device)
    trans = motion[:, 309 : 309 + 3].to(device)
    betas = motion[:, 312:].to(device)

    left_hand_pose = pose_hand[:, :45]
    right_hand_pose = pose_hand[:, 45:]

    # output = smplx.forward(
    #     betas=betas,
    #     transl=trans,
    #     global_orient=root_orient,
    #     body_pose=pose_body,
    #     jaw_pose=pose_jaw,
    #     left_hand_pose=left_hand_pose,
    #     right_hand_pose=right_hand_pose,
    #     expression=face_expr,
    # )

    # print(output)

    return root_orient, pose_body, trans, betas


def get_nth_file(folder, n=0, ext=".npy"):
    with os.scandir(folder) as entries:
        npy_files = (entry.path for entry in entries if entry.name.endswith(ext))
        return next(islice(npy_files, n, None), None)
