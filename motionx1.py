import os
import json

import numpy as np
import torch
import smplx

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model_path = os.path.join("data", "body_models", "smplx", "SMPLX_MALE.npz")
model_path = os.path.join("data", "body_models")

model = smplx.create(
    model_path,
    model_type="smplx",
    gender="neutral",
    use_pca=False,
    num_expression_coeffs=50,
).to(device)

# /home/hlz/Downloads/motionx/motion/motion_generation/smplx322/animation/animation
data_folder = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "motionx",
    "motion",
    "motion_generation",
    "smplx322",
    "animation",
    "animation",
)

# iterate through all files in the folder
for file in os.listdir(data_folder):
    # get the name without the extension
    filename = os.path.splitext(file)[0]

    motion = np.load(os.path.join(data_folder, file))
    motion = torch.tensor(motion).float().to(device)

    root_orient = motion[:, :3]  # controls the global root orientation
    pose_body = motion[:, 3 : 3 + 63]  # controls the body
    pose_hand = motion[:, 66 : 66 + 90]  # controls the finger articulation
    pose_jaw = motion[:, 66 + 90 : 66 + 93]  # controls the yaw pose
    face_expr = motion[:, 159 : 159 + 50]  # controls the face expression
    face_shape = motion[:, 209 : 209 + 100]  # controls the face shape
    trans = motion[:, 309 : 309 + 3]  # controls the global body position
    betas = motion[:, 312:]  # controls the body shape. Body shape is static

    # print(root_orient.shape, pose_body.shape, trans.shape, betas.shape, face_expr.shape)
    # torch.Size([62, 3]) torch.Size([62, 63]) torch.Size([62, 3]) torch.Size([62, 10])

    leye_pose = torch.zeros(root_orient.shape[0], 3, device=device)
    reye_pose = torch.zeros(root_orient.shape[0], 3, device=device)

    left_hand_pose = pose_hand[:, :45]  # left hand articulation
    right_hand_pose = pose_hand[:, 45:]  # right hand articulation

    # print(root_orient.shape)
    # root_orient[:, 1], root_orient[:, 2] = root_orient[:, 2], root_orient[:, 1]

    # Get mesh vertices
    output = model(
        betas=betas,
        body_pose=pose_body,
        global_orient=root_orient,
        transl=trans,
        batch_size=motion.shape[0],
        jaw_pose=pose_jaw,
        leye_pose=leye_pose,
        reye_pose=reye_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        expression=face_expr,
        return_verts=True,
        # num_betas=60,
    )

    vertices = output.vertices  # [10, 10475, 3]

    print(vertices.shape)

    break
