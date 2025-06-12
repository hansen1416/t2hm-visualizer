import os

import cv2
import torch
import open3d as o3d


def get_checkerboard_plane(plane_width=20, num_boxes=15, center=True):

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
            ground.translate([c[0], 0, c[1]])
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
