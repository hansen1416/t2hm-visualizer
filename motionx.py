import os
import json

import numpy as np
import torch
from smplx import SMPLX

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
smplx = (
    SMPLX(
        os.path.join("data", "body_models", "smplx", "SMPLX_MALE.npz"),
        use_pca=False,
        flat_hand_mean=True,
    )
    .eval()
    .to(device)
)
faces = smplx.faces

# /home/hlz/Downloads/motionx/motion/motion_generation/smplx322/animation/animation
folder = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "motionx",
    "motion",
    "motion_generation",
    "smplx322",
    "animation",
    "animation",
)
# /home/hlz/Downloads/motionx/text/semantic_label
labels_folder = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "motionx",
    "text",
    "semantic_label",
    "animation",
    "animation",
)
# /home/hlz/Downloads/motionx/text/wholebody_pose_description
desc_folder = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "motionx",
    "text",
    "wholebody_pose_description",
    "animation",
    "animation",
)

filename = "Ways_to_Catch_360_clip1.npy"

# iterate through all files in the folder
for file in os.listdir(folder):
    # get the name without the extension
    filename = os.path.splitext(file)[0]

    motion = np.load(os.path.join(folder, file))
    motion = torch.tensor(motion).float()

    print(motion.shape)

    motion_parms = {
        "root_orient": motion[:, :3],  # controls the global root orientation
        "pose_body": motion[:, 3 : 3 + 63],  # controls the body
        "pose_hand": motion[:, 66 : 66 + 90],  # controls the finger articulation
        "pose_jaw": motion[:, 66 + 90 : 66 + 93],  # controls the yaw pose
        "face_expr": motion[:, 159 : 159 + 50],  # controls the face expression
        "face_shape": motion[:, 209 : 209 + 100],  # controls the face shape
        "trans": motion[:, 309 : 309 + 3],  # controls the global body position
        "betas": motion[:, 312:],  # controls the body shape. Body shape is static
    }

    for k, v in motion_parms.items():
        print(f"{k}: {v.shape}")

    with open(os.path.join(desc_folder, filename + ".json"), "r") as f:
        desc = json.load(f)

        print(f"desc length:", len(desc.keys()))

    with open(os.path.join(labels_folder, filename + ".txt"), "r") as f:
        label = f.read()

        print(label)

    left_hand_pose = motion_parms["pose_hand"][:, :45]
    right_hand_pose = motion_parms["pose_hand"][:, 45:]

    betas = motion_parms["betas"][0, :].unsqueeze(0).to(device)
    trans = motion_parms["trans"][0, :].unsqueeze(0).to(device)
    root_orient = motion_parms["root_orient"][0, :].unsqueeze(0).to(device)
    pose_body = motion_parms["pose_body"][0, :].unsqueeze(0).to(device)
    pose_jaw = motion_parms["pose_jaw"][0, :].unsqueeze(0).to(device)
    left_hand_pose = left_hand_pose[0, :].unsqueeze(0).to(device)
    right_hand_pose = right_hand_pose[0, :].unsqueeze(0).to(device)
    face_expr = motion_parms["face_expr"][0, :10].unsqueeze(0).to(device)

    print(f"betas shape: {betas.shape}")
    print(f"trans shape: {trans.shape}")
    print(f"root_orient shape: {root_orient.shape}")
    print(f"pose_body shape: {pose_body.shape}")
    print(f"pose_jaw shape: {pose_jaw.shape}")
    print(f"left_hand_pose shape: {left_hand_pose.shape}")
    print(f"right_hand_pose shape: {right_hand_pose.shape}")
    print(f"face_expr shape: {face_expr.shape}")

    output = smplx.forward(
        betas=betas,
        transl=trans,
        global_orient=root_orient,
        body_pose=pose_body,
        jaw_pose=pose_jaw,
        leye_pose=torch.zeros([1, 3]).to(device),
        reye_pose=torch.zeros([1, 3]).to(device),
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        expression=face_expr,
    )

    vertices = output.vertices

    print(f"vertices shape: {vertices.shape}")

    # print(basename)
    break

# # read motion and save as smplx representation
# # motion = np.load("motion_data/smplx_322/000001.npy")
# motion = np.load(os.path.join(folder, filename))
# motion = torch.tensor(motion).float()
# motion_parms = {
#     "root_orient": motion[:, :3],  # controls the global root orientation
#     "pose_body": motion[:, 3 : 3 + 63],  # controls the body
#     "pose_hand": motion[:, 66 : 66 + 90],  # controls the finger articulation
#     "pose_jaw": motion[:, 66 + 90 : 66 + 93],  # controls the yaw pose
#     "face_expr": motion[:, 159 : 159 + 50],  # controls the face expression
#     "face_shape": motion[:, 209 : 209 + 100],  # controls the face shape
#     "trans": motion[:, 309 : 309 + 3],  # controls the global body position
#     "betas": motion[:, 312:],  # controls the body shape. Body shape is static
# }


# # read text labels
# semantic_text = np.loadtxt("semantic_labels/000001.npy")  # semantic labels
