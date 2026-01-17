import sys
import os
from pathlib import Path
from easydict import EasyDict

import torch
import numpy as np

module_dir = Path("/home/hlz/repos/ASE/ase").as_posix()
sys.path.insert(0, module_dir)  # put it first so it wins resolution

from utils.motion_lib_smpl import MotionLibSMPL
from poselib.poselib.skeleton.skeleton3d import SkeletonTree

device = "cuda:0"

motion_file = Path(
    "/home/hlz/repos/ASE/ase/data/motions/0-ACCAD_Female1Running_c3d_C4-Runtowalk1_poses.pkl"
)

motion_file = Path(
    "/home/hlz/datasets/amass-pkls/0-ACCAD_Male1General_c3d_GeneralA11-MilitaryCrawlForward_poses.pkl"
)

_key_body_ids = torch.tensor(
    range(24),
    device="cuda:0",
)

motion_lib_cfg = EasyDict(
    {
        "motion_file": motion_file,
        "device": torch.device("cpu"),
        "fix_height": 1,
        "min_length": -1,
        "max_length": -1,
        "im_eval": True,
        "multi_thread": False,
        "smpl_type": "smpl",
        "randomrize_heading": True,
        "device": device,
        "min_length": -1,
        "step_dt": 1 / 60,
        "key_body_ids": _key_body_ids,
    }
)

motion_lib = MotionLibSMPL(motion_lib_cfg=motion_lib_cfg)

asset_root = Path("/home/hlz/repos/ASE/ase/data/assets").as_posix()

asset_file_names = [
    "mjcf/smpl/a0f02530_smpl.xml",
    "mjcf/smpl/b156eebd_smpl.xml",
    "mjcf/smpl/584b793a_smpl.xml",
    "mjcf/smpl/aef91182_smpl.xml",
    "mjcf/smpl/0c3f729e_smpl.xml",
    "mjcf/smpl/1f0234a6_smpl.xml",
    "mjcf/smpl/aaab922b_smpl.xml",
    "mjcf/smpl/97d89d08_smpl.xml",
    "mjcf/smpl/f16065b0_smpl.xml",
    "mjcf/smpl/5dcdf59a_smpl.xml",
    "mjcf/smpl/7e43c211_smpl.xml",
    "mjcf/smpl/f00a7a9f_smpl.xml",
    "mjcf/smpl/25bef108_smpl.xml",
    "mjcf/smpl/85c00aec_smpl.xml",
    "mjcf/smpl/5636a12a_smpl.xml",
    "mjcf/smpl/33a17abb_smpl.xml",
    "mjcf/smpl/124f1e57_smpl.xml",
    "mjcf/smpl/e585e9fd_smpl.xml",
    "mjcf/smpl/520ed34f_smpl.xml",
    "mjcf/smpl/5dbdeb54_smpl.xml",
    "mjcf/smpl/65f23504_smpl.xml",
    "mjcf/smpl/948528ba_smpl.xml",
    "mjcf/smpl/b3428686_smpl.xml",
    "mjcf/smpl/0e091a72_smpl.xml",
    "mjcf/smpl/f0f7976f_smpl.xml",
    "mjcf/smpl/20cf78b3_smpl.xml",
    "mjcf/smpl/aca66100_smpl.xml",
    "mjcf/smpl/af4dbe08_smpl.xml",
    "mjcf/smpl/f56ef3f7_smpl.xml",
    "mjcf/smpl/fba2c39a_smpl.xml",
    "mjcf/smpl/75e01b05_smpl.xml",
    "mjcf/smpl/31f56211_smpl.xml",
    "mjcf/smpl/b80aed9f_smpl.xml",
    "mjcf/smpl/706ed6c9_smpl.xml",
    "mjcf/smpl/39f19cab_smpl.xml",
    "mjcf/smpl/fcc491cd_smpl.xml",
    "mjcf/smpl/bc793600_smpl.xml",
    "mjcf/smpl/74fc526e_smpl.xml",
    "mjcf/smpl/09016021_smpl.xml",
    "mjcf/smpl/6895f004_smpl.xml",
    "mjcf/smpl/602dbc36_smpl.xml",
    "mjcf/smpl/22e9a6f8_smpl.xml",
    "mjcf/smpl/0f637664_smpl.xml",
    "mjcf/smpl/85011266_smpl.xml",
    "mjcf/smpl/ef483c8a_smpl.xml",
    "mjcf/smpl/9f630f94_smpl.xml",
    "mjcf/smpl/dc46f761_smpl.xml",
    "mjcf/smpl/c51c47d2_smpl.xml",
    "mjcf/smpl/3577c351_smpl.xml",
    "mjcf/smpl/18ce6b2c_smpl.xml",
    "mjcf/smpl/ef892c76_smpl.xml",
    "mjcf/smpl/349bdc0e_smpl.xml",
    "mjcf/smpl/15c4d2e9_smpl.xml",
    "mjcf/smpl/6803e1fa_smpl.xml",
    "mjcf/smpl/e9f8d7a4_smpl.xml",
    "mjcf/smpl/8a24b3b7_smpl.xml",
    "mjcf/smpl/00c972db_smpl.xml",
    "mjcf/smpl/6046abb1_smpl.xml",
    "mjcf/smpl/76144ae7_smpl.xml",
    "mjcf/smpl/e698f1e8_smpl.xml",
    "mjcf/smpl/0a1ece18_smpl.xml",
    "mjcf/smpl/638a4fb7_smpl.xml",
    "mjcf/smpl/b944e212_smpl.xml",
    "mjcf/smpl/2a31c8ac_smpl.xml",
]

asset_file_names = [os.path.join(asset_root, p) for p in asset_file_names]

sk_tree = SkeletonTree.from_mjcf(asset_file_names[0])

template_betas = []

for af in asset_file_names:

    beta_rel_dir = os.path.dirname(af)  # e.g. "mjcf/smpl"

    smpl_stem = os.path.splitext(os.path.basename(af))[0]  # "a0f02530_smpl"
    beta_prefix = smpl_stem.rsplit("_", 1)[0]  # "a0f02530"
    beta_filename = beta_prefix + "_betas.pt"  # "a0f02530_betas.pt"
    beta_path = os.path.join(beta_rel_dir, beta_filename)

    betas = torch.load(beta_path, weights_only=True)
    # here the betas should be torch.Size([1, 10])
    if len(betas.shape) > 1:
        betas = betas[0]

    betas = torch.as_tensor(betas, dtype=torch.float32, device=device)
    template_betas.append(betas)

# [num_env, 10]
_template_betas = torch.stack(template_betas, dim=0)

motion_lib.load_motions(
    skeleton_trees=[sk_tree], gender_betas=_template_betas.cpu(), random_sample=False
)

print(motion_lib.motion_ids)

# [num_envs]
motion_ids = motion_lib.sample_motions(_template_betas.shape[0])
# [num_envs]
motion_times = torch.zeros(_template_betas.shape[0], dtype=torch.float, device=device)

motion_res = motion_lib.get_motion_state(motion_ids, motion_times)

# print(motion_res["key_pos"].shape)
# print(motion_res["motion_bodies"].shape)
# print(_template_betas.shape)
print(motion_res.keys())


# template_betas = _template_betas.detach().to("cpu", torch.float32)  # [64, 10]
# motion_bodies = (
#     motion_res["motion_bodies"].detach().to("cpu", torch.float32)
# )  # [64, 10]
# key_pos = motion_res["key_pos"].detach().to("cpu", torch.float32)
