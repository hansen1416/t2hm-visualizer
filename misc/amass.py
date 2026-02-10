import os
from glob import glob

from tqdm import tqdm
import numpy as np
import torch
from smplx import SMPLX

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# /home/hlz/datasets/AMASS/datasets
folder = os.path.join(
    os.path.expanduser("~"),
    "datasets",
    "AMASS",
    "datasets",
)

all_npzs = glob(f"{folder}/**/*.npz", recursive=True)
print(f"Found {len(all_npzs)} .npz files.")

# keys_shape = {
#     "trans": (1688, 3),
#     "gender": (),
#     "mocap_framerate": (),
#     "betas": (16,),
#     "dmpls": (1688, 8),
#     "poses": (1688, 156),
# }


cate_keys = {}

invalid_count = 0
invalid_files = []

for data_path in tqdm(all_npzs):
    data = dict(np.load(open(data_path, "rb"), allow_pickle=True))

    # check if all these keys: trans, poses, betas, gender exist in data
    if not all(key in data.keys() for key in ["trans", "poses", "betas", "gender"]):
        print(f"Keys missing in {data_path}, {data.keys()}")
        invalid_files.append(data_path)
        invalid_count += 1
        continue


# save invalid files to a txt
with open("invalid_amass_files.txt", "w") as f:
    for item in invalid_files:
        f.write("%s\n" % item)

print(f"Found {invalid_count} invalid files.")
